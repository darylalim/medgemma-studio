import contextlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from typing import BinaryIO

import numpy as np
import openslide
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

# Greedy decoding (temperature 0) can fall into degenerate repetition loops on
# longer generations (e.g. a multi-slice CT read). A repetition penalty over a wide
# context breaks them while staying deterministic; verified not to corrupt the
# localization JSON output.
REPETITION_PENALTY = 1.3
REPETITION_CONTEXT_SIZE = 256

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

WSI_TYPES = ["svs", "ndpi", "tif", "tiff"]
WSI_PATCH_SIZE = 896  # MedGemma's native image size
WSI_MAGNIFICATIONS = [5, 10, 20, 40]
WSI_DEFAULT_MAG = 10
WSI_THUMBNAIL_SIZE = 2048  # longest side of the tissue-mask thumbnail
WSI_SATURATION_THRESHOLD = 20  # a pixel is tissue when max(RGB) - min(RGB) exceeds this
WSI_MIN_TISSUE_FRACTION = 0.25  # min tissue fraction for a patch to qualify

DEFAULT_INSTRUCTION_WSI = (
    "You are an expert pathologist reviewing patches sampled from a whole-slide "
    "image. Describe the salient histologic findings across the patches."
)


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
    is_wsi: bool = False,
) -> tuple[str, int]:
    if is_localizing:
        return LOCALIZATION_INSTRUCTION, 1300 if is_thinking else 1000
    if is_thinking:
        # Thinking needs room for the trace AND the answer. A multi-slice CT read, a
        # multi-patch whole-slide read, or a two-image comparison needs more than a
        # single-image answer.
        budget = 2500 if (is_ct or is_wsi) else 1600 if is_comparing else 1300
        return (
            f"SYSTEM INSTRUCTION: think silently if needed. {system_instruction}",
            budget,
        )
    if is_ct or is_wsi:
        # A multi-slice CT volume or multi-patch whole-slide read needs a large
        # budget for tile-by-tile reasoning (cf. the reference notebooks' 2000); the
        # editable persona is kept as-is.
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
    # Rewind each upload before reading: Streamlit keeps the UploadedFile (a
    # BytesIO) in session_state across reruns and dcmread reads from the current
    # position, so a second Run on the same files would otherwise fail mid-stream.
    datasets = []
    for f in dicom_files:
        f.seek(0)
        datasets.append(pydicom.dcmread(f))
    datasets.sort(key=lambda d: int(d.InstanceNumber))
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


@st.cache_data(show_spinner=False)
def _cached_total_ram_gib() -> float:
    """Installed RAM is fixed for the process, so memoize it: the CT/WSI sliders
    call ``ram_aware_slice_cap`` on every rerun, and detection can shell out to
    ``sysctl``. Kept separate from ``_detect_total_ram_gib`` so that helper stays
    uncached and directly unit-testable."""
    return _detect_total_ram_gib()


def ram_aware_slice_cap(total_ram_gib: float | None = None) -> tuple[int, int]:
    """Return ``(default, max)`` CT slice counts scaled to installed memory.

    Each windowed slice costs ~0.5 GB of peak GPU memory over a ~13 GB base
    (measured: medgemma-1.5-4b-bf16 on a 32 GiB M2 Max — 16 slices peaked
    ~20.7 GB, 32 slices OOMed). A fixed headroom is reserved for the OS, and the
    max is clamped to a practical ceiling. On a 32 GiB machine this yields (8, 16).
    """
    if total_ram_gib is None:
        total_ram_gib = _cached_total_ram_gib()
    base_gib, per_slice_gib, headroom_gib, hard_max = 13.0, 0.5, 11.0, 64
    budget = total_ram_gib - base_gib - headroom_gib
    max_slices = max(2, min(hard_max, int(budget / per_slice_gib)))
    default = max(2, max_slices // 2)
    return default, max_slices


def mag_from_mpp(mpp_x: float) -> float:
    """Approximate objective power from microns-per-pixel (0.25 um/px ~ 40x)."""
    return 10.0 / mpp_x


def effective_magnification(objective_power: float, downsample: float) -> float:
    """Effective magnification of a pyramid level: base power / its downsample."""
    return objective_power / downsample


def pick_level(
    level_downsamples: Sequence[float], objective_power: float, target_mag: float
) -> int:
    """Index of the pyramid level whose effective magnification
    (``objective_power / downsample``) is closest to ``target_mag``."""
    mags = [objective_power / d for d in level_downsamples]
    return min(range(len(mags)), key=lambda i: abs(mags[i] - target_mag))


def patch_grid(level_w: int, level_h: int, patch_size: int) -> list[tuple[int, int]]:
    """Top-left (x, y) coords of a non-overlapping patch grid over a level, in that
    level's pixel frame. Partial edge tiles are dropped; row-major order so an even
    subsample spreads spatially across the slide."""
    return [
        (x, y)
        for y in range(0, level_h - patch_size + 1, patch_size)
        for x in range(0, level_w - patch_size + 1, patch_size)
    ]


def tissue_mask(
    thumbnail_rgb: np.ndarray, sat_threshold: int = WSI_SATURATION_THRESHOLD
) -> np.ndarray:
    """Boolean (H, W) tissue mask from an RGB thumbnail: a pixel is tissue when its
    saturation proxy ``max(R,G,B) - min(R,G,B)`` exceeds ``sat_threshold`` — i.e. it
    is not white/grey glass. Pure numpy."""
    rgb = thumbnail_rgb.astype(np.int16)
    saturation = rgb.max(axis=-1) - rgb.min(axis=-1)
    return saturation > sat_threshold


def tissue_patches(
    grid: list[tuple[int, int]],
    mask: np.ndarray,
    level_size: tuple[int, int],
    patch_size: int,
    min_fraction: float = WSI_MIN_TISSUE_FRACTION,
) -> list[tuple[int, int]]:
    """Keep grid coords whose footprint, projected onto the thumbnail-scale ``mask``,
    is at least ``min_fraction`` tissue. ``level_size`` is the (w, h) of the level the
    grid was computed on; ``mask`` is shaped (h_mask, w_mask)."""
    level_w, level_h = level_size
    mask_h, mask_w = mask.shape
    sx, sy = mask_w / level_w, mask_h / level_h
    kept: list[tuple[int, int]] = []
    for x, y in grid:
        mx0, my0 = int(x * sx), int(y * sy)
        mx1 = max(mx0 + 1, int((x + patch_size) * sx))
        my1 = max(my0 + 1, int((y + patch_size) * sy))
        window = mask[my0:my1, mx0:mx1]
        if window.size and float(window.mean()) >= min_fraction:
            kept.append((x, y))
    return kept


def mark_patches(
    thumbnail: Image.Image,
    coords: list[tuple[int, int]],
    level_size: tuple[int, int],
    patch_size: int,
) -> Image.Image:
    """Outline each kept patch's footprint on a copy of ``thumbnail``. ``coords`` are
    level-pixel top-lefts, scaled to the thumbnail's size so the user sees which
    regions were sampled."""
    annotated = thumbnail.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    level_w, level_h = level_size
    sx, sy = annotated.width / level_w, annotated.height / level_h
    for x, y in coords:
        draw.rectangle(
            (
                round(x * sx),
                round(y * sy),
                round((x + patch_size) * sx),
                round((y + patch_size) * sy),
            ),
            outline="red",
            width=2,
        )
    return annotated


def _slide_objective_power(slide) -> float:
    """Base objective power: the slide's objective-power property (when positive),
    else derived from its microns-per-pixel, else a 40x default. A non-positive value
    is treated as a miss — some scanners emit 0 for a missing objective power, which
    would otherwise collapse pick_level to level 0 and disclose a bogus 0x."""
    props = slide.properties
    raw = props.get(openslide.PROPERTY_NAME_OBJECTIVE_POWER)
    if raw is not None:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (TypeError, ValueError):
            pass
    mpp = props.get(openslide.PROPERTY_NAME_MPP_X)
    if mpp is not None:
        try:
            val = mag_from_mpp(float(mpp))
            if val > 0:
                return val
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return 40.0


def _read_patch(slide, x: int, y: int, level: int, downsample: float, size: int):
    """Read one patch as RGB. ``read_region`` takes a LEVEL-0 location but a
    target-level size, so the level-pixel coords are scaled by ``downsample``. RGBA
    out-of-bounds pixels are composited onto white (a bare convert would blacken
    them)."""
    location = (round(x * downsample), round(y * downsample))
    region = slide.read_region(location, level, (size, size))
    background = Image.new("RGBA", region.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, region).convert("RGB")


def load_wsi_patches(
    uploaded_file,
    target_mag: float,
    max_patches: int,
    patch_size: int = WSI_PATCH_SIZE,
    min_fraction: float = WSI_MIN_TISSUE_FRACTION,
) -> tuple[list[Image.Image], Image.Image, float]:
    """Read tissue patches from an uploaded whole-slide image.

    Spills the upload to a temp file (OpenSlide opens by path), picks the pyramid
    level nearest ``target_mag``, tiles it into ``patch_size`` patches over tissue,
    deterministically caps to ``max_patches`` (see ``subsample_indices``), and reads
    each patch as RGB. Returns ``(patches, overlay, actual_mag)`` where ``overlay`` is
    the thumbnail with the sampled patches outlined and ``actual_mag`` is the chosen
    level's true magnification. Raises ``ValueError`` with a user-facing message for an
    unreadable slide, a slide too small for one patch, or one with no detectable
    tissue.
    """
    suffix = os.path.splitext(getattr(uploaded_file, "name", "") or "")[1] or ".svs"
    # delete=False (not a `with`): OpenSlide opens by path, so the file must outlive
    # this scope; it is unlinked in the finally below.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)  # noqa: SIM115
    path = tmp.name
    try:
        # Bind ``path`` and enter the cleanup scope before writing, so a failed write
        # (e.g. disk-full on a multi-GB slide) still unlinks the spilled temp file.
        try:
            tmp.write(uploaded_file.getvalue())
        finally:
            tmp.close()
        try:
            slide = openslide.OpenSlide(path)
        except (openslide.OpenSlideError, OSError) as exc:
            raise ValueError(
                "Could not read this file as a whole-slide image. Supported "
                "formats: .svs, .ndpi, .tif/.tiff."
            ) from exc
        try:
            objective_power = _slide_objective_power(slide)
            level = pick_level(slide.level_downsamples, objective_power, target_mag)
            downsample = slide.level_downsamples[level]
            level_w, level_h = slide.level_dimensions[level]
            grid = patch_grid(level_w, level_h, patch_size)
            if not grid:
                raise ValueError(
                    "Slide is too small for 896px patches at this magnification; "
                    "try a higher magnification."
                )
            thumbnail = slide.get_thumbnail(
                (WSI_THUMBNAIL_SIZE, WSI_THUMBNAIL_SIZE)
            ).convert("RGB")
            mask = tissue_mask(np.asarray(thumbnail))
            tissue = tissue_patches(
                grid, mask, (level_w, level_h), patch_size, min_fraction
            )
            if not tissue:
                raise ValueError(
                    "No tissue detected on this slide. Try a lower magnification or "
                    "a different slide."
                )
            kept = [tissue[i] for i in subsample_indices(len(tissue), max_patches)]
            patches = [
                _read_patch(slide, x, y, level, downsample, patch_size) for x, y in kept
            ]
            overlay = mark_patches(thumbnail, kept, (level_w, level_h), patch_size)
            actual_mag = effective_magnification(objective_power, downsample)
            return patches, overlay, actual_mag
        finally:
            slide.close()
    finally:
        os.unlink(path)


def show_response(response: str) -> None:
    st.divider()
    st.markdown("### Response")
    st.markdown(response)


def load_uploaded_image(uploaded_file) -> Image.Image | None:
    """Open an uploaded file as an image (loading is decoupled from previewing so
    the CXR tab can lay two studies out side by side in comparison mode).

    Returns the PIL image, or None if no file was provided or it failed to load (in
    which case an error is shown). ``image.load()`` forces the decode so invalid data
    fails here rather than later at ``st.image`` time.
    """
    if uploaded_file is None:
        return None
    try:
        image = Image.open(uploaded_file)
        image.load()
        return image
    except Exception:
        st.error(
            "Failed to load image. Please upload a valid image file.",
            icon=":material/error:",
        )
        return None


st.set_page_config(
    page_title="MedGemma Studio",
    page_icon=":material/clinical_notes:",
    layout="centered",
)


def run_model(
    model, processor, config, messages, images, max_new_tokens, show_spinner=True
):
    """Format the prompt, generate, and return the raw response text.

    Wrapped in a spinner unless ``show_spinner`` is False — the CT and WSI tabs
    drive their own ``st.status`` across preprocessing + generation and pass
    ``show_spinner=False`` so a spinner isn't nested inside that status. On failure
    it shows an error and returns ``None`` so callers can bail out.
    """
    image_for_model = images or None
    num_images = len(images)
    spinner = (
        st.spinner("Generating response...")
        if show_spinner
        else contextlib.nullcontext()
    )
    with spinner:
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
                repetition_penalty=REPETITION_PENALTY,
                repetition_context_size=REPETITION_CONTEXT_SIZE,
                verbose=False,
            )
            return output.text
        except Exception as e:
            st.error(f"Inference failed: {e}", icon=":material/error:")
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

    # Tuck the persona + thinking toggle (advanced, rarely-edited) into a collapsed
    # expander so the primary flow — prompt → upload → Run — leads each tab. The
    # widgets are still created every run, so the auto_switch persona tracking and
    # all key-based lookups are unaffected.
    with st.expander("Model settings", expanded=False):
        instruction = st.text_area(
            "System instruction",
            key=instr_key,
            height=100,
            on_change=_mark_touched,
        )
        is_thinking = st.toggle("Thinking", key=f"{key_prefix}_thinking")
    return instruction, is_thinking


def _file_sig(uploaded) -> tuple:
    """Identity of an uploaded file (name + size) for staleness checks; () when no
    file. A persisted result stores the signature of the inputs that produced it,
    so the render block drops it once the upload (or any tracked input) changes."""
    if uploaded is None:
        return ()
    return (getattr(uploaded, "name", ""), getattr(uploaded, "size", 0))


STALE_RESULT_HINT = "Inputs changed since this result — click Run to refresh."


def fresh_result_or_hint(key: str, live_sig) -> dict | None:
    """Single source of truth for the staleness gate shared by every tab.

    Returns the persisted result stored under ``key`` when its recorded ``sig`` still
    matches the live inputs. When a result exists but is stale, show a hint and return
    None — an abrupt vanish (the old behavior) reads as a bug in a clinical tool.
    Returns None with no hint when there is no stored result at all.
    """
    result = st.session_state.get(key)
    if result is None:
        return None
    if result["sig"] != live_sig:
        st.info(STALE_RESULT_HINT, icon=":material/refresh:")
        return None
    return result


def render_ask_tab(model, processor, config):
    st.caption("Ask a medical question. No image required.")
    prompt = st.text_input(
        "Enter your question",
        placeholder="e.g. What causes a pleural effusion?",
        key="ask_prompt",
    ).strip()
    instruction, is_thinking = tab_settings("ask", DEFAULT_INSTRUCTION_TEXT)

    if st.button(
        "Run", type="primary", disabled=not prompt, width="stretch", key="ask_run"
    ):
        full_instruction, max_new_tokens = get_generation_params(
            has_image=False, is_thinking=is_thinking, system_instruction=instruction
        )
        messages = build_messages(prompt, full_instruction)
        raw = run_model(model, processor, config, messages, [], max_new_tokens)
        # Persist the run so it survives later reruns: editing any widget reruns
        # the script and the Run button returns False, which would otherwise wipe
        # the answer. Inference stays inside the button block; the render below
        # reads from session_state on every rerun. The stored ``sig`` (the prompt)
        # lets the render drop the result once the question changes.
        st.session_state["ask_result"] = (
            None
            if raw is None
            else {"raw": raw, "is_thinking": is_thinking, "sig": prompt}
        )

    result = fresh_result_or_hint("ask_result", prompt)
    if result is not None:
        show_response(render_thought(result["raw"], result["is_thinking"]))


def render_cxr_tab(model, processor, config):
    st.caption(
        "Analyze a chest X-ray. Add a second image to compare two studies, or turn "
        "on 'Locate anatomy' to outline structures with bounding boxes."
    )
    prompt = st.text_input(
        "Enter your question",
        placeholder="e.g. Describe this chest X-ray",
        key="cxr_prompt",
    ).strip()

    upload1 = st.file_uploader(
        "Upload a chest X-ray", type=IMAGE_TYPES, key="cxr_image1"
    )
    image1 = load_uploaded_image(upload1)
    # A second slot appears only once the first image exists, so the model can
    # compare two studies (e.g. longitudinal CXR) in a single prompt.
    upload2 = None
    image2 = None
    if image1 is not None:
        upload2 = st.file_uploader(
            "Upload a second image to compare (optional)",
            type=IMAGE_TYPES,
            key="cxr_image2",
        )
        image2 = load_uploaded_image(upload2)

    # Preview side by side in comparison mode — seeing both studies at once is the
    # whole point of a longitudinal read; otherwise a single full-width preview.
    if image1 is not None and image2 is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.image(image1, caption="First image", width="stretch")
        with col2:
            st.image(image2, caption="Second image", width="stretch")
    elif image1 is not None:
        st.image(image1, caption="Uploaded image", width="stretch")

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
            ":material/info: Localization uses a built-in prompt; the System "
            "instruction above is ignored in this mode."
        )
    elif is_comparing:
        st.caption(
            ":material/info: Comparison mode: both images are sent to the model "
            "together."
        )

    # Signature of the inputs this result depends on: a stale result is dropped
    # (not rendered) once the prompt, either upload, or the localize mode changes.
    cxr_sig = (prompt, is_localizing, _file_sig(upload1), _file_sig(upload2))

    if st.button(
        "Run", type="primary", disabled=not prompt, width="stretch", key="cxr_run"
    ):
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
            # Pad to a square so the model's [0, 1000] coordinates map back without
            # an offset, then crop the annotated result to the original size.
            localize_size = images[0].size
            model_images = [pad_to_square(images[0])]
        else:
            model_images = images

        # Label the two studies so the comparison persona's "first/second image"
        # wording binds to a specific image regardless of attention ordering.
        image_labels = ["First image:", "Second image:"] if is_comparing else None
        messages = build_messages(
            prompt, full_instruction, model_images, image_labels=image_labels
        )
        raw = run_model(
            model, processor, config, messages, model_images, max_new_tokens
        )
        # Persist the finished run (see render_ask_tab). For localization, parse the
        # boxes and draw the annotation once here — strip any thinking trace with
        # parse_response so the expander is rendered only in the block below.
        if raw is None:
            st.session_state["cxr_result"] = None
        elif localize:
            _, answer = parse_response(raw, is_thinking)
            boxes = parse_boxes(answer)
            annotated = None
            if boxes and localize_size is not None:
                width, height = localize_size
                annotated = draw_boxes(model_images[0], boxes).crop(
                    (0, 0, width, height)
                )
            st.session_state["cxr_result"] = {
                "mode": "localize",
                "raw": raw,
                "is_thinking": is_thinking,
                "annotated": annotated,
                "boxes": boxes,
                "sig": cxr_sig,
            }
        else:
            st.session_state["cxr_result"] = {
                "mode": "text",
                "raw": raw,
                "is_thinking": is_thinking,
                "sig": cxr_sig,
            }

    result = fresh_result_or_hint("cxr_result", cxr_sig)
    if result is None:
        return
    response = render_thought(result["raw"], result["is_thinking"])
    if result["mode"] == "localize":
        if result["annotated"] is not None:
            st.image(result["annotated"], caption="Localized anatomy", width="stretch")
            st.divider()
            st.markdown("### Detected structures")
            # The boxes are drawn on the image above, so list the labels plainly
            # rather than the raw normalized coordinates (cryptic to a clinician).
            st.markdown(
                "\n".join(
                    f"- **{box['label'] or 'unlabeled'}**" for box in result["boxes"]
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

    # Drop a persisted result once the prompt, uploaded slices, or slice count change.
    ct_sig = (prompt, tuple(_file_sig(f) for f in dicom_files or []), n_slices)

    if st.button(
        "Run",
        type="primary",
        disabled=not (prompt and dicom_files),
        width="stretch",
        key="ct_run",
    ):
        # Clear any prior run, then do all the heavy work (DICOM read, windowing,
        # inference) strictly inside the button block; persist the result so it
        # survives later reruns (see render_ask_tab). An st.status narrates the
        # phases so the otherwise-silent DICOM read + windowing (which run before
        # generation) show continuous progress instead of a frozen-looking gap.
        st.session_state["ct_result"] = None
        with st.status("Analyzing CT series…", expanded=False) as status:
            status.update(label="Reading DICOM series…")
            try:
                hu_slices = load_ct_volume(dicom_files, n_slices)
            except Exception as e:
                st.error(f"Failed to read DICOM series: {e}", icon=":material/error:")
                hu_slices = None
            if hu_slices is None:
                status.update(
                    label="Could not read DICOM series", state="error", expanded=True
                )
            else:
                status.update(label="Windowing slices…")
                slice_images = [window_ct_slice(hu) for hu in hu_slices]
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
                status.update(label=f"Running MedGemma on {len(slice_images)} slices…")
                raw = run_model(
                    model,
                    processor,
                    config,
                    messages,
                    slice_images,
                    max_new_tokens,
                    show_spinner=False,
                )
                if raw is None:
                    status.update(
                        label="Inference failed", state="error", expanded=True
                    )
                else:
                    st.session_state["ct_result"] = {
                        "preview": slice_images[0],
                        "count": len(slice_images),
                        "raw": raw,
                        "is_thinking": is_thinking,
                        "sig": ct_sig,
                    }
                    status.update(label="Analysis complete", state="complete")

    result = fresh_result_or_hint("ct_result", ct_sig)
    if result is None:
        return
    st.image(
        result["preview"],
        caption=f"Sample windowed slice (1 of {result['count']})",
        width="stretch",
    )
    show_response(render_thought(result["raw"], result["is_thinking"]))


def render_wsi_tab(model, processor, config):
    st.caption(
        "Upload a whole-slide image (.svs/.ndpi/.tiff). Tissue patches are sampled at "
        "a chosen magnification and read as the 896px tiles MedGemma 1.5 is trained on."
    )
    prompt = st.text_input(
        "Enter your question",
        placeholder="e.g. Describe the histologic findings",
        key="wsi_prompt",
    ).strip()
    slide_file = st.file_uploader("Upload a slide", type=WSI_TYPES, key="wsi_files")
    # segmented_control (not select_slider): these are four discrete objective-power
    # modes, like a microscope turret, and one-tap selection beats landing a slider
    # handle on a tick. ``or WSI_DEFAULT_MAG`` guards the (single-select) None case.
    target_mag = (
        st.segmented_control(
            "Magnification",
            options=WSI_MAGNIFICATIONS,
            default=WSI_DEFAULT_MAG,
            format_func=lambda m: f"{m}×",
            help="Higher magnification shows finer detail over less area. Clamped to "
            "the slide's available pyramid levels.",
            key="wsi_mag",
        )
        or WSI_DEFAULT_MAG
    )

    default_patches, max_patches = ram_aware_slice_cap()
    if max_patches > 2:
        n_patches = st.slider(
            "Patches to analyze",
            min_value=2,
            max_value=max_patches,
            value=default_patches,
            help="Tissue patches are sampled uniformly across the slide. The cap "
            "scales to your machine's memory.",
            key="wsi_patches",
        )
    else:
        n_patches = 2
        st.caption("Limited memory detected: analyzing 2 patches.")

    instruction, is_thinking = tab_settings("wsi", DEFAULT_INSTRUCTION_WSI)

    # Drop a persisted result once the prompt, slide, magnification, or patch count
    # change.
    wsi_sig = (prompt, _file_sig(slide_file), target_mag, n_patches)

    if st.button(
        "Run",
        type="primary",
        disabled=not (prompt and slide_file),
        width="stretch",
        key="wsi_run",
    ):
        # Clear any prior run, then do all the heavy work (OpenSlide read, tiling,
        # inference) strictly inside the button block; persist the result so it
        # survives later reruns (see render_ask_tab). An st.status narrates the
        # phases so the otherwise-silent slide read + tissue sampling (which can
        # take many seconds on a multi-GB slide) show progress before generation.
        st.session_state["wsi_result"] = None
        with st.status("Analyzing slide…", expanded=False) as status:
            status.update(label="Reading slide and sampling tissue…")
            try:
                patches, overlay, actual_mag = load_wsi_patches(
                    slide_file, target_mag, n_patches
                )
            except Exception as e:
                st.error(f"Failed to read slide: {e}", icon=":material/error:")
                patches = None
            if patches is None:
                status.update(
                    label="Could not read slide", state="error", expanded=True
                )
            else:
                labels = [f"PATCH {i}" for i in range(1, len(patches) + 1)]
                full_instruction, max_new_tokens = get_generation_params(
                    has_image=True,
                    is_thinking=is_thinking,
                    system_instruction=instruction,
                    is_wsi=True,
                )
                messages = build_messages(
                    prompt, full_instruction, patches, image_labels=labels
                )
                status.update(label=f"Running MedGemma on {len(patches)} patches…")
                raw = run_model(
                    model,
                    processor,
                    config,
                    messages,
                    patches,
                    max_new_tokens,
                    show_spinner=False,
                )
                if raw is None:
                    status.update(
                        label="Inference failed", state="error", expanded=True
                    )
                else:
                    st.session_state["wsi_result"] = {
                        "overlay": overlay,
                        "actual_mag": actual_mag,
                        "count": len(patches),
                        "preview": patches[0],
                        "raw": raw,
                        "is_thinking": is_thinking,
                        "sig": wsi_sig,
                    }
                    status.update(label="Analysis complete", state="complete")

    result = fresh_result_or_hint("wsi_result", wsi_sig)
    if result is None:
        return
    st.image(result["overlay"], caption="Tissue overview", width="stretch")
    st.caption(f"{result['count']} patches sampled at ~{result['actual_mag']:.1f}x.")
    st.image(
        result["preview"],
        caption=f"Sample patch (1 of {result['count']})",
        width="stretch",
    )
    show_response(render_thought(result["raw"], result["is_thinking"]))


def main():
    st.title("MedGemma Studio")
    with st.spinner("Loading model..."):
        model, processor, config = load_model()
    tab_ask, tab_cxr, tab_ct, tab_wsi = st.tabs(
        [
            ":material/forum: Ask",
            ":material/radiology: Chest X-ray",
            ":material/readiness_score: Computed Tomography",
            ":material/biotech: Pathology (WSI)",
        ]
    )
    with tab_ask:
        render_ask_tab(model, processor, config)
    with tab_cxr:
        render_cxr_tab(model, processor, config)
    with tab_ct:
        render_ct_tab(model, processor, config)
    with tab_wsi:
        render_wsi_tab(model, processor, config)


if __name__ == "__main__":
    main()
