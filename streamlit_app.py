import json
import os
import re
import subprocess
from collections.abc import Iterable
from typing import BinaryIO

import numpy as np
import pydicom
import streamlit as st
from dotenv import load_dotenv
from mlx_vlm import generate, load
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image, ImageDraw
from pydicom.pixels import apply_rescale

load_dotenv()

MODEL_ID = "mlx-community/medgemma-1.5-4b-it-bf16"

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

DEFAULT_INSTRUCTION_IMAGE = "You are an expert radiologist."
DEFAULT_INSTRUCTION_TEXT = "You are a helpful medical assistant."
DEFAULT_INSTRUCTION_COMPARE = (
    "You are an expert radiologist comparing two medical images, such as "
    "longitudinal studies of the same patient. Describe the key differences and "
    "changes between the first and second image."
)

LOCALIZATION_INSTRUCTION = (
    "You are an expert radiologist localizing anatomy on a medical image. "
    "Return only a JSON list inside a ```json code block. Each item must be "
    '{"box_2d": [y0, x0, y1, x1], "label": "<structure>"}, where (y0, x0) is the '
    "top-left corner and (y1, x1) the bottom-right corner, each normalized to the "
    'range [0, 1000]. "left" and "right" refer to the patient\'s anatomical sides.'
)

DEFAULT_INSTRUCTION_CT = (
    "You are an expert radiologist analyzing a contiguous block of CT slices from "
    "a single volume. Review the windowed slices in order and describe the salient "
    "findings."
)

# MedGemma 1.5 is trained to read CT as a 3-window false-color image: each RGB
# channel is a distinct Hounsfield-unit window (wide / soft-tissue / brain). These
# ranges are part of the model's trained input format, so they are fixed.
CT_WINDOWS: list[tuple[int, int]] = [(-1024, 1024), (-135, 215), (0, 80)]


@st.cache_resource
def load_model():
    model, processor = load(MODEL_ID)
    config = load_config(MODEL_ID)
    return model, processor, config


def parse_response(response: str, is_thinking: bool) -> tuple[str | None, str]:
    if is_thinking and "<unused95>" in response:
        thought, answer = response.split("<unused95>", 1)
        thought = thought.removeprefix("<unused94>thought\n")
        return thought, answer
    return None, response


def build_messages(
    prompt: str,
    system_instruction: str,
    images: list[Image.Image] | None = None,
    image_labels: list[str] | None = None,
) -> list:
    user_content: list[dict] = [{"type": "text", "text": prompt}]
    for i, _ in enumerate(images or []):
        # Anchor each image with a label (e.g. "First image:") when provided, so a
        # prompt that says "first/second image" binds to a specific image.
        if image_labels and i < len(image_labels):
            user_content.append({"type": "text", "text": image_labels[i]})
        user_content.append({"type": "image"})
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_content},
    ]


def get_generation_params(
    has_image: bool,
    is_thinking: bool,
    system_instruction: str,
    is_localizing: bool = False,
    is_comparing: bool = False,
    is_ct: bool = False,
) -> tuple[str, int]:
    if is_localizing:
        return LOCALIZATION_INSTRUCTION, 1300 if is_thinking else 1000
    if is_thinking:
        # Thinking needs room for the trace AND the answer. A multi-slice CT read
        # or a two-image comparison needs more than a single-image answer.
        budget = 2500 if is_ct else 1600 if is_comparing else 1300
        return (
            f"SYSTEM INSTRUCTION: think silently if needed. {system_instruction}",
            budget,
        )
    if is_ct:
        # Reading a multi-slice CT volume needs a large budget for slice-by-slice
        # reasoning (cf. the reference notebook's 2000); the editable CT persona is
        # kept as-is.
        return system_instruction, 2000
    if is_comparing:
        # Comparing two images needs more room than a single-image answer; the
        # editable system instruction (a comparison persona) is kept as-is.
        return system_instruction, 600
    max_new_tokens = 300 if has_image else 500
    return system_instruction, max_new_tokens


def pad_to_square(image: Image.Image) -> Image.Image:
    """Pad an image to an RGB square (top-left aligned) so localization
    coordinates, normalized over a square frame, map back without an offset."""
    image = image.convert("RGB")
    width, height = image.size
    if width == height:
        return image
    size = max(width, height)
    padded = Image.new("RGB", (size, size))
    padded.paste(image, (0, 0))
    return padded


def parse_boxes(response: str) -> list[dict]:
    """Extract bounding boxes from a model response.

    Accepts an optional ```json fence (or a bare JSON list) and returns a list of
    {"box_2d": [y0, x0, y1, x1], "label": str}. Malformed items are dropped; an
    unparseable response yields [].
    """
    fence = re.search(r"```(?:json)?\s*(.*?)```", response, re.DOTALL)
    if fence:
        payload = fence.group(1)
    else:
        start, end = response.find("["), response.rfind("]")
        payload = response[start : end + 1] if start != -1 and end > start else response
    try:
        data = json.loads(payload)
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    boxes: list[dict] = []
    for item in data:
        box = item.get("box_2d") if isinstance(item, dict) else None
        # Require 4 numeric coords (bools excluded): scale_box divides on these,
        # and it runs outside the inference try/except, so bad values would crash.
        if (
            isinstance(box, list)
            and len(box) == 4
            and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in box
            )
        ):
            boxes.append({"box_2d": box, "label": item.get("label", "")})
    return boxes


def scale_box(box_2d: list, size: int) -> tuple[int, int, int, int]:
    """Convert a [y0, x0, y1, x1] box normalized to [0, 1000] into pixel
    (x0, y0, x1, y1) corners for a square image of side ``size``. Corners are
    ordered (x0 <= x1, y0 <= y1) so a model box with swapped corners still draws
    instead of raising in ImageDraw.rectangle."""
    y0, x0, y1, x1 = box_2d
    px0, px1 = sorted((round(x0 / 1000 * size), round(x1 / 1000 * size)))
    py0, py1 = sorted((round(y0 / 1000 * size), round(y1 / 1000 * size)))
    return px0, py0, px1, py1


def draw_boxes(image: Image.Image, boxes: list[dict]) -> Image.Image:
    """Draw labeled boxes onto a copy of ``image`` (assumed square)."""
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    for box in boxes:
        x0, y0, x1, y1 = scale_box(box["box_2d"], annotated.width)
        draw.rectangle((x0, y0, x1, y1), outline="red", width=3)
        if box.get("label"):
            draw.text((x0 + 4, y0 + 4), box["label"], fill="red")
    return annotated


def normalize_hu(hu_slice: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Clip a Hounsfield-unit slice to [lo, hi] and rescale to 0-255 floats."""
    clipped = np.clip(hu_slice.astype(np.float32), lo, hi)
    return (clipped - lo) / (hi - lo) * 255.0


def window_ct_slice(
    hu_slice: np.ndarray, windows: list[tuple[int, int]] = CT_WINDOWS
) -> Image.Image:
    """Pack three Hounsfield-unit windows into the R/G/B channels of one image.

    MedGemma 1.5 is trained to read CT where each channel is a distinct window
    (wide / soft-tissue / brain), so a single false-color slice carries three
    diagnostic views at once. Mirrors the reference notebook's norm()/window().
    """
    channels = np.stack([normalize_hu(hu_slice, lo, hi) for lo, hi in windows], axis=-1)
    return Image.fromarray(np.round(channels).astype(np.uint8), mode="RGB")


def subsample_indices(n: int, max_slices: int) -> list[int]:
    """Uniformly pick up to ``max_slices`` indices across an ``n``-slice volume.

    Returns every index when ``n <= max_slices``; otherwise spreads the picks
    evenly including both endpoints (index 0 and n-1). A cap of 1 (or less) yields
    the middle slice; an empty volume yields [].
    """
    if n <= 0:
        return []
    if max_slices >= n:
        return list(range(n))
    if max_slices <= 1:
        return [n // 2]
    return [round(i / (max_slices - 1) * (n - 1)) for i in range(max_slices)]


def load_ct_volume(
    dicom_files: Iterable[BinaryIO], max_slices: int
) -> list[np.ndarray]:
    """Read uploaded per-slice DICOM files into ordered, subsampled HU arrays.

    Each file is one CT slice from a single series. Slices are sorted by
    InstanceNumber, uniformly subsampled to at most ``max_slices`` (see
    ``subsample_indices``), and converted to Hounsfield units via the DICOM rescale
    slope/intercept. Raises ``ValueError`` with a user-facing message for inputs the
    CT path cannot handle: multiple series, compressed pixel data, or
    multi-frame/color (non-2D) images.
    """
    datasets = sorted(
        (pydicom.dcmread(f) for f in dicom_files),
        key=lambda d: int(d.InstanceNumber),
    )
    if len({getattr(d, "SeriesInstanceUID", None) for d in datasets}) > 1:
        raise ValueError(
            "Multiple DICOM series detected; please upload one series at a time."
        )
    slices: list[np.ndarray] = []
    for i in subsample_indices(len(datasets), max_slices):
        dataset = datasets[i]
        try:
            hu = apply_rescale(dataset.pixel_array, dataset)
        except Exception as exc:
            transfer_syntax = getattr(dataset.file_meta, "TransferSyntaxUID", None)
            if transfer_syntax is not None and transfer_syntax.is_compressed:
                raise ValueError(
                    "This DICOM uses a compressed transfer syntax that isn't "
                    "supported. Export the series as uncompressed (or install a "
                    "decoder such as pylibjpeg)."
                ) from exc
            raise
        # window_ct_slice (called outside the caller's try/except) assumes a 2D
        # grayscale array; multi-frame (3D) or color (H,W,3) slices would crash it.
        if hu.ndim != 2:
            raise ValueError(
                "Unsupported DICOM: expected single-frame grayscale CT slices "
                "(got a multi-frame or color image)."
            )
        slices.append(hu)
    return slices


def _detect_total_ram_gib() -> float:
    """Best-effort installed-RAM detection in binary GiB; conservative on failure."""
    names = getattr(os, "sysconf_names", {})
    if "SC_PHYS_PAGES" in names and "SC_PAGE_SIZE" in names:
        try:
            return os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1024**3
        except (ValueError, OSError):
            pass
    try:
        out = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        return int(out.stdout.strip()) / 1024**3
    except (ValueError, OSError, subprocess.SubprocessError):
        return 16.0


def ram_aware_slice_cap(total_ram_gib: float | None = None) -> tuple[int, int]:
    """Return ``(default, max)`` CT slice counts scaled to installed memory.

    Each windowed slice costs ~0.5 GB of peak GPU memory over a ~13 GB base
    (measured: medgemma-1.5-4b-bf16 on a 32 GiB M2 Max — 16 slices peaked
    ~20.7 GB, 32 slices OOMed). A fixed headroom is reserved for the OS, and the
    max is clamped to a practical ceiling. On a 32 GiB machine this yields (8, 16).
    """
    if total_ram_gib is None:
        total_ram_gib = _detect_total_ram_gib()
    base_gib, per_slice_gib, headroom_gib, hard_max = 13.0, 0.5, 11.0, 64
    budget = total_ram_gib - base_gib - headroom_gib
    max_slices = max(2, min(hard_max, int(budget / per_slice_gib)))
    default = max(2, max_slices // 2)
    return default, max_slices


def show_response(response: str) -> None:
    st.markdown("### Response")
    st.markdown(response)


def load_and_preview_image(uploaded_file, caption: str) -> Image.Image | None:
    """Open an uploaded file as an image and preview it under ``caption``.

    Returns the PIL image, or None if no file was provided or it failed to load
    (in which case an error is shown).
    """
    if uploaded_file is None:
        return None
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption=caption, width="stretch")
        return image
    except Exception:
        st.error("Failed to load image. Please upload a valid image file.")
        return None


st.set_page_config(page_title="MedGemma Pipeline")


def run_model(model, processor, config, messages, images, max_new_tokens):
    """Format the prompt, generate, and return the raw response text.

    Runs inside a spinner; on failure it shows an error and returns ``None`` so
    callers can bail out.
    """
    image_for_model = images or None
    num_images = len(images)
    with st.spinner("Generating response..."):
        try:
            formatted_prompt = apply_chat_template(
                processor, config, messages, num_images=num_images
            )
            output = generate(
                model,
                processor,
                formatted_prompt,  # ty: ignore[invalid-argument-type]
                image_for_model,  # ty: ignore[invalid-argument-type]
                max_tokens=max_new_tokens,
                temperature=0.0,
                verbose=False,
            )
            return output.text
        except Exception as e:
            st.error(f"Inference failed: {e}")
            return None


def render_thought(raw_response: str, is_thinking: bool) -> str:
    """Split off any thinking trace into an expander; return the answer text."""
    thought, response = parse_response(raw_response, is_thinking)
    if thought is not None:
        with st.expander("Thinking trace"):
            st.markdown(thought)
    return response


def tab_settings(
    key_prefix: str, default_instruction: str, auto_switch: bool = False
) -> tuple[str, bool]:
    """Render a per-tab System instruction + Thinking toggle.

    Returns ``(instruction, is_thinking)``. Each tab keeps independent widget state
    via ``key_prefix``. When ``auto_switch`` is set, the instruction tracks
    ``default_instruction`` until the user edits it (used by the Chest X-ray tab,
    whose default depends on the image count); otherwise the default is set once.
    """
    instr_key = f"{key_prefix}_instruction"
    touched_key = f"{key_prefix}_instruction_touched"
    if auto_switch:
        if not st.session_state.get(touched_key):
            st.session_state[instr_key] = default_instruction
    else:
        st.session_state.setdefault(instr_key, default_instruction)

    def _mark_touched():
        st.session_state[touched_key] = True

    instruction = st.text_area(
        "System instruction",
        key=instr_key,
        height=100,
        on_change=_mark_touched,
    )
    is_thinking = st.toggle("Thinking", key=f"{key_prefix}_thinking")
    return instruction, is_thinking


def render_ask_tab(model, processor, config):
    st.caption("Ask a medical question. No image required.")
    prompt = st.text_input(
        "Enter your question",
        placeholder="e.g. What causes a pleural effusion?",
        key="ask_prompt",
    ).strip()
    instruction, is_thinking = tab_settings("ask", DEFAULT_INSTRUCTION_TEXT)

    if not st.button("Run", type="primary", disabled=not prompt, key="ask_run"):
        return

    full_instruction, max_new_tokens = get_generation_params(
        has_image=False, is_thinking=is_thinking, system_instruction=instruction
    )
    messages = build_messages(prompt, full_instruction)
    raw = run_model(model, processor, config, messages, [], max_new_tokens)
    if raw is None:
        return
    show_response(render_thought(raw, is_thinking))


def render_cxr_tab(model, processor, config):
    prompt = st.text_input(
        "Enter your question",
        placeholder="e.g. Describe this chest X-ray",
        key="cxr_prompt",
    ).strip()

    image1 = load_and_preview_image(
        st.file_uploader("Upload a chest X-ray", type=IMAGE_TYPES, key="cxr_image1"),
        caption="Uploaded image",
    )
    # A second slot appears only once the first image exists, so the model can
    # compare two studies (e.g. longitudinal CXR) in a single prompt.
    image2 = None
    if image1 is not None:
        image2 = load_and_preview_image(
            st.file_uploader(
                "Upload a second image to compare (optional)",
                type=IMAGE_TYPES,
                key="cxr_image2",
            ),
            caption="Second image",
        )

    images = [img for img in (image1, image2) if img is not None]
    has_image = len(images) >= 1
    is_comparing = len(images) == 2

    default_instruction = (
        DEFAULT_INSTRUCTION_COMPARE if is_comparing else DEFAULT_INSTRUCTION_IMAGE
    )
    instruction, is_thinking = tab_settings(
        "cxr", default_instruction, auto_switch=True
    )

    is_localizing = st.toggle(
        "Locate anatomy (bounding boxes)",
        disabled=len(images) != 1,
        help="Outline anatomy with bounding boxes. Requires a single image.",
        key="cxr_localize",
    )
    if is_localizing and len(images) == 1:
        st.caption(
            "ℹ️ Localization uses a built-in prompt; the System instruction above "
            "is ignored in this mode."
        )
    elif is_comparing:
        st.caption("ℹ️ Comparison mode: both images are sent to the model together.")

    if not st.button("Run", type="primary", disabled=not prompt, key="cxr_run"):
        return

    # Localization is single-image only; with two images it is unavailable.
    localize = is_localizing and len(images) == 1
    localize_size: tuple[int, int] | None = None
    full_instruction, max_new_tokens = get_generation_params(
        has_image,
        is_thinking,
        instruction,
        is_localizing=localize,
        is_comparing=is_comparing,
    )

    if localize:
        # Pad to a square so the model's [0, 1000] coordinates map back without an
        # offset, then crop the annotated result to the original size.
        localize_size = images[0].size
        model_images = [pad_to_square(images[0])]
    else:
        model_images = images

    # Label the two studies so the comparison persona's "first/second image" wording
    # binds to a specific image regardless of attention ordering.
    image_labels = ["First image:", "Second image:"] if is_comparing else None
    messages = build_messages(
        prompt, full_instruction, model_images, image_labels=image_labels
    )
    raw = run_model(model, processor, config, messages, model_images, max_new_tokens)
    if raw is None:
        return
    response = render_thought(raw, is_thinking)

    if localize:
        boxes = parse_boxes(response)
        # localize_size is always set when localize is True; the explicit check also
        # narrows it from Optional for the type checker.
        if boxes and localize_size is not None:
            width, height = localize_size
            annotated = draw_boxes(model_images[0], boxes).crop((0, 0, width, height))
            st.image(annotated, caption="Localized anatomy", width="stretch")
            st.markdown("### Detected structures")
            st.markdown(
                "\n".join(
                    f"- **{box['label'] or 'unlabeled'}**: {box['box_2d']}"
                    for box in boxes
                )
            )
        else:
            st.warning("No bounding boxes were returned.")
            show_response(response)
    else:
        show_response(response)


def render_ct_tab(model, processor, config):
    st.caption(
        "Upload a CT series as individual DICOM slice files. Each slice is windowed "
        "into a false-color image (the representation MedGemma 1.5 is trained on)."
    )
    prompt = st.text_input(
        "Enter your question",
        placeholder="e.g. Are there hypodense liver lesions?",
        key="ct_prompt",
    ).strip()
    dicom_files = st.file_uploader(
        "Upload CT DICOM slices",
        accept_multiple_files=True,
        key="ct_files",
    )

    default_slices, max_slices = ram_aware_slice_cap()
    if max_slices > 2:
        n_slices = st.slider(
            "Slices to analyze",
            min_value=2,
            max_value=max_slices,
            value=default_slices,
            help="Slices are sampled uniformly across the volume. The cap scales to "
            "your machine's memory.",
            key="ct_slices",
        )
    else:
        n_slices = 2
        st.caption("Limited memory detected: analyzing 2 slices.")

    instruction, is_thinking = tab_settings("ct", DEFAULT_INSTRUCTION_CT)

    if not st.button(
        "Run", type="primary", disabled=not (prompt and dicom_files), key="ct_run"
    ):
        return

    try:
        hu_slices = load_ct_volume(dicom_files, n_slices)
    except Exception as e:
        st.error(f"Failed to read DICOM series: {e}")
        return

    slice_images = [window_ct_slice(hu) for hu in hu_slices]
    st.image(
        slice_images[0],
        caption=f"Sample windowed slice (1 of {len(slice_images)})",
        width="stretch",
    )
    labels = [f"SLICE {i}" for i in range(1, len(slice_images) + 1)]
    full_instruction, max_new_tokens = get_generation_params(
        has_image=True,
        is_thinking=is_thinking,
        system_instruction=instruction,
        is_ct=True,
    )
    messages = build_messages(
        prompt, full_instruction, slice_images, image_labels=labels
    )
    raw = run_model(model, processor, config, messages, slice_images, max_new_tokens)
    if raw is None:
        return
    show_response(render_thought(raw, is_thinking))


def main():
    st.title("MedGemma Pipeline")
    with st.spinner("Loading model..."):
        model, processor, config = load_model()
    tab_ask, tab_cxr, tab_ct = st.tabs(["Ask", "Chest X-ray", "Computed Tomography"])
    with tab_ask:
        render_ask_tab(model, processor, config)
    with tab_cxr:
        render_cxr_tab(model, processor, config)
    with tab_ct:
        render_ct_tab(model, processor, config)


if __name__ == "__main__":
    main()
