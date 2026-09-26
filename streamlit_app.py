import json
import os
import re
import subprocess
import tempfile
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import openslide
import pydicom
import streamlit as st
from dotenv import load_dotenv
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config
from PIL import Image, ImageDraw
from pydicom.pixels import apply_rescale

load_dotenv()

MODEL_ID = "mlx-community/medgemma-1.5-4b-it-8bit"
MODEL_CARD_URL = f"https://huggingface.co/{MODEL_ID}"
HAI_DEF_TERMS_URL = (
    "https://developers.google.com/health-ai-developer-foundations/terms"
)

# Browser-tab favicon, vendored as a local PNG rather than named as
# ":material/clinical_notes:". Streamlit resolves a :material/...: page_icon to an SVG
# on fonts.gstatic.com and refetches it on every page load -- the only outbound request
# an otherwise fully on-device app would make. A local file is served from Streamlit's
# own /media/ endpoint instead. Resolved against __file__, not the CWD, so `streamlit
# run` from any directory finds it. Source glyph: Material Symbols Rounded
# `clinical_notes` (Apache-2.0), recolored to the theme's primary #88c0d0 -- the stock
# glyph is black and all but invisible against a dark browser tab strip.
FAVICON_PATH = Path(__file__).resolve().parent / "assets" / "favicon.png"

IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

# .streamlit/config.toml raises server.maxUploadSize to 2000 MB so real whole-slide
# images get through (Streamlit's 200 MB default rejects them at the HTTP layer). That
# ceiling is global, so every uploader that is NOT the slide one narrows itself back to
# the stock 200 MB. It matters most on the CT uploader: that one is
# accept_multiple_files=True, carries no type= filter by design, and reads every slice
# fully into memory, so it has the widest blast radius of the four.
NON_WSI_MAX_UPLOAD_MB = 200

# Greedy decoding (temperature 0) can fall into degenerate repetition loops on
# longer generations (e.g. a multi-slice CT read). A repetition penalty over a wide
# context breaks them while staying deterministic. Re-verified on the 8-bit weights:
# localization still emits well-formed JSON that parse_boxes accepts. It does bias
# the model toward shorter lists, though — two localize prompts each returned one
# fewer box with the penalty than without (temperature 0, so the penalty is the only
# variable), since every extra box repeats the same structural tokens. That is fewer
# structures returned, not corrupted output.
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
CT_THUMBNAIL_SIZE = (256, 256)  # bounding box for the "view all slices" gallery

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

# Persistent st.warning shown above the tabs on every view. App users never see the
# README, so this mirrors its research-only / not-a-medical-device Disclaimer in-app.
DISCLAIMER_TEXT = (
    "**Research and educational use only — not a medical device.** "
    "Outputs are AI-generated, may be inaccurate, and are not medical advice. "
    "Always consult a qualified healthcare professional."
)


# The cache's own spinner is the one that renders on a cold start -- exactly the miss
# this message exists for -- so own the text here instead of wrapping the call site.
# The default show_spinner=True builds its message from the function name, leaking
# "Running load_model()." into the UI alongside any outer st.spinner. show_time gives
# the ~6 GB first-run download visible evidence that it is progressing; a warm rerun
# shows nothing, since a cache hit lands inside the spinner's 0.5s delay.
@st.cache_resource(
    show_spinner="Loading MedGemma (first run downloads ~6 GB)…", show_time=True
)
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
        # apply_rescale hands back float64 (apply_modality_lut casts before applying
        # the slope), doubling what cached_ct_volume holds -- ~42 MB per entry at the
        # 20-slice default -- for precision nothing downstream uses: normalize_hu's
        # first move is .astype(np.float32). Guarded rather than unconditional,
        # because with a Modality LUT Sequence (0028,3000), or with no rescale tags at
        # all, apply_rescale returns the uint8/uint16 array unchanged, where a blind
        # cast would *double* the footprint instead of halving it.
        if hu.dtype == np.float64:
            hu = hu.astype(np.float32)
        slices.append(hu)
    return slices


def _detect_total_ram_gib() -> float:
    """Best-effort installed-RAM detection in binary GiB; conservative on failure."""
    names = getattr(os, "sysconf_names", {})
    if "SC_PHYS_PAGES" in names and "SC_PAGE_SIZE" in names:
        try:
            pages, page_size = os.sysconf("SC_PHYS_PAGES"), os.sysconf("SC_PAGE_SIZE")
            # POSIX allows -1 for an indeterminate limit, which Python returns rather
            # than raising. Without this check a negative total would skip the sysctl
            # fallback silently and floor ram_aware_slice_cap at 2 slices on any
            # machine, with nothing in the UI to explain it.
            if pages > 0 and page_size > 0:
                return pages * page_size / 1024**3
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


def ct_thumbnails(slices: Sequence[Image.Image]) -> list[Image.Image]:
    """Downscaled copies of the windowed slices, for the "view all" gallery.

    These are persisted in ``st.session_state``, so size is the whole point: at
    ``ram_aware_slice_cap``'s 64-slice hard cap the full-resolution slices would be
    well over 100 MB per session, while a 256px thumbnail is ~0.2 MB. ``thumbnail``
    mutates in place and preserves aspect ratio, so copy before scaling -- the
    originals are what get sent to the model.
    """
    thumbs = []
    for image in slices:
        thumb = image.copy()
        thumb.thumbnail(CT_THUMBNAIL_SIZE)
        thumbs.append(thumb)
    return thumbs


def ram_aware_slice_cap(total_ram_gib: float | None = None) -> tuple[int, int]:
    """Return ``(default, max)`` CT slice counts scaled to installed memory.

    Measured for the 8-bit weights on a 32 GiB M2 Max: ~6.9 GB base + ~0.55 GB
    per windowed slice (16 slices peaked ~15.8 GB). Both terms are rounded up
    below — 9 GB and 0.6 GB/slice — so the cap sits ~2 GB clear of that fit
    rather than on it: the curve is not perfectly linear (MLX reuses cached
    buffers) and other Macs will differ.

    Both terms moved relative to the bf16 tuning these replace (13 GB +
    0.5 GB/slice; 16 slices peaked ~20.7 GB, 32 slices OOMed), and in opposite
    directions: quantization shrank the weights but not the per-slice
    activations, so the base fell while the per-slice cost rose. The two
    formulas therefore cross at 44 GiB — below it machines gain capacity
    (32 GiB: (8, 16) -> (10, 20)), above it they lose a little (48 GiB:
    (24, 48) -> (23, 46)). That is deliberate: 0.5 GB/slice understated the
    measured cost, so the old cap was over-optimistic exactly where slice
    counts are highest.

    A fixed headroom is reserved for the OS, and the max is clamped to a
    practical ceiling. On a 32 GiB machine this yields (10, 20). The 2-slice
    floor covers everything below 21.8 GB (= base + headroom + one slice), so
    a 16 GB Mac and a 20 GB Mac both sit on it; 22 GB is the first tier off.
    """
    if total_ram_gib is None:
        total_ram_gib = _cached_total_ram_gib()
    base_gib, per_slice_gib, headroom_gib, hard_max = 9.0, 0.6, 11.0, 64
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


# Streamlit's default UploadedFile hasher is name + tell() + getvalue(): it walks the
# entire payload on every Run, and it mixes in the stream position -- which both
# loaders leave at EOF -- so under the default the CT key would differ every Run and
# never hit. ``file_id`` is stable for the life of an upload and is what
# ``UploadedFile.__hash__`` itself uses -- and, via ``_file_sig``, the same identity
# the staleness gate keys on, so the cache and the freshness check agree on what
# "the same file" means. Both wrappers below are ``scope="session"``: a ``file_id``
# is minted per upload per session, so an entry outliving its session can never be
# hit again -- it is only ~21 MB (CT) / ~48 MB (WSI) of decoded imaging waiting on
# max_entries eviction, in an app whose slice budget is already hand-tuned against
# installed RAM.
_UPLOAD_HASH: dict[str | type[Any], Callable[[Any], Any]] = {
    "streamlit.runtime.uploaded_file_manager.UploadedFile": lambda f: f.file_id
}


@st.cache_data(
    max_entries=3, show_spinner=False, hash_funcs=_UPLOAD_HASH, scope="session"
)
def cached_ct_volume(
    dicom_files: Iterable[BinaryIO], max_slices: int
) -> list[np.ndarray]:
    """Cached wrapper over ``load_ct_volume`` (see the note on ``_UPLOAD_HASH``).

    Iterating on the prompt or the persona against a fixed series is the tab's
    central workflow, and without this every Run re-reads every DICOM. The wrapper is
    deliberately *separate* from the pure loader: decorating ``load_ct_volume``
    itself would turn ``TestLoadCtVolume``'s rewind guard into a cache hit and stop
    it testing the rewind. ``max_entries=3`` leaves room to A/B a couple of slice
    counts and still hit; the cache trades memory for that, so it is bounded on
    purpose -- at the RAM-gated default of 20 slices an entry is ~21 MB of float32 HU
    arrays (it was double that before ``load_ct_volume`` downcast them), and
    ``ram_aware_slice_cap`` only lets the count grow on machines with the memory to
    absorb it.
    """
    return load_ct_volume(dicom_files, max_slices)


@st.cache_data(
    max_entries=3, show_spinner=False, hash_funcs=_UPLOAD_HASH, scope="session"
)
def cached_wsi_patches(
    uploaded_file, target_mag: float, max_patches: int
) -> tuple[list[Image.Image], Image.Image, float]:
    """Cached wrapper over ``load_wsi_patches`` -- the same bargain as
    ``cached_ct_volume``, and worth more here: a miss re-spills a multi-GB slide to a
    temp file, re-opens it in OpenSlide, then re-thumbnails, re-masks and re-reads
    every 896px patch. Kept separate from the pure loader for the same reason (see
    ``TestLoadWsiPatches``'s rewind guard). Costs more memory than the CT cache --
    an 896px RGB patch is ~2.4 MB, so an entry at the 20-patch default is ~48 MB --
    which is the bargain: RAM for not re-reading a multi-GB slide on every Run."""
    return load_wsi_patches(uploaded_file, target_mag, max_patches)


def show_response(response: str) -> None:
    st.markdown("### Response")
    st.markdown(response)


def hit_token_cap(finish: dict) -> bool:
    """True when ``run_model``'s ``finish`` out-param reports a token-cap stop.

    A named helper rather than the literal at five store sites: it is the single
    place a second surfaced finish reason would land, and it keeps a typo in the
    comparison from silently disabling the warning on one tab only.
    """
    return finish.get("reason") == "length"


def warn_if_truncated(result: dict) -> None:
    """Flag a persisted result whose generation hit the token cap.

    A run cut off at ``max_new_tokens`` renders byte-identically to one that
    finished -- the answer simply stops, mid-sentence or mid-word -- so without this
    the reader has no way to tell a complete report from a clipped one. Truncation is
    routine here rather than exotic: a plain "Describe this chest X-ray" does not fit
    the 300-token budget, and the CT/WSI reads regularly reach 2000.

    Every tab calls it first in its output column, so it sits directly above the
    text it qualifies: the imagery that used to separate the two on CT/WSI now lives
    in the inputs column (see ``workspace_columns``). It is not folded into
    ``show_response`` because the CXR localization path draws boxes and a label
    legend and never calls ``show_response`` -- precisely where a JSON list clipped
    at the cap silently costs structures -- so on CXR it runs before the mode branch.
    """
    if result.get("truncated"):
        st.warning(
            "Output stopped at the token limit and may be incomplete.",
            icon=":material/content_cut:",
        )


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


# Wide, because every tab is a two-column workspace (see workspace_columns): the study
# and the report sit side by side. Under "centered" the ~730px column forced them into
# one stack, where a full-width radiograph or slide overview pushed the report below
# the fold -- the reader could never see an image and the findings about it together.
st.set_page_config(
    page_title="MedGemma Studio",
    page_icon=str(FAVICON_PATH),
    layout="wide",
)


def run_model(
    model,
    processor,
    config,
    messages,
    images,
    max_new_tokens,
    penalize_repetition=True,
    finish: dict | None = None,
):
    """Stream generation live and return the full accumulated response text.

    Tokens are streamed via ``mlx_vlm.stream_generate`` through ``st.write_stream``
    — on a slow local model this turns a blank wait into visibly arriving text. The
    caller stores the returned string and immediately ``st.rerun()``s, so the streamed
    output is replaced by the clean persisted render (``render_thought`` / the
    localization annotation) rather than duplicated beneath it. (An earlier attempt to
    clear an ``st.empty()`` placeholder in a ``finally`` was unreliable inside
    ``st.tabs`` + ``st.fragment`` — the emptied slot still rendered — so the run is
    discarded by a rerun instead.) ``generate`` is itself just ``text += chunk.text``
    over ``stream_generate``, so the returned string is identical; determinism
    (``temperature=0`` + the repetition penalties) is preserved. Returns ``None`` on
    failure after showing an error, so callers bail out.

    ``finish`` is an optional out-parameter: pass a dict and it comes back carrying
    ``{"reason": <finish_reason of the last chunk>}``. It is an out-param rather than
    part of the return value because all four call sites gate on ``if raw is None``,
    and widening the ``str | None`` contract to a tuple would churn every one of them
    plus the tests that mock this path. ``stream_generate`` reports ``finish_reason``
    only on its final chunk (``"length"`` when the token budget ran out, ``"stop"`` on
    EOS), so recording every chunk and letting the last write win is exactly right.
    """
    image_for_model = images or None
    num_images = len(images)
    try:
        formatted_prompt = apply_chat_template(
            processor, config, messages, num_images=num_images
        )
    except Exception as e:
        st.error(f"Inference failed: {e}", icon=":material/error:")
        return None

    # The localization path opts out. The penalty cannot tell a degenerate loop from
    # legitimate schema repetition, and every extra box repeats the same structural
    # tokens ('{"box_2d": [', '"label"'), so it truncates the JSON list. Measured
    # over 8 localize prompts on the 8-bit weights at temperature 0: 14 boxes with the
    # penalty vs 19 without, one prompt returning nothing parseable with it and boxes
    # without, and no duplicate labels or token-cap runs either way -- i.e. the looping
    # it guards against did not appear on this path. Long CT/WSI reads keep it.
    penalty_kwargs = (
        {
            "repetition_penalty": REPETITION_PENALTY,
            "repetition_context_size": REPETITION_CONTEXT_SIZE,
        }
        if penalize_repetition
        else {}
    )

    def _record_finish(chunk) -> None:
        # getattr, not chunk.finish_reason: the tests hand run_model MagicMock chunks
        # carrying only .text, and a hard attribute access there would be an
        # AttributeError on a path the real backend always populates.
        if finish is not None:
            finish["reason"] = getattr(chunk, "finish_reason", None)

    def _deltas():
        # Prefill -- the vision-tower encode of up to 20 slices/patches, plus prompt
        # processing -- all happens before the first token, and st.write_stream
        # creates no element until the first non-empty chunk. Without a spinner here
        # the output column is blank through the longest silence of the run (on CT/WSI
        # the st.status has already resolved to complete by this point, so there is
        # nothing live on screen at all). st.spinner is a transient element, so it
        # leaves no gap above the stream once tokens arrive, and its built-in 0.5s
        # delay keeps it invisible on the fast text-only path.
        stream = iter(
            stream_generate(
                model,
                processor,
                formatted_prompt,  # ty: ignore[invalid-argument-type]
                image_for_model,
                max_tokens=max_new_tokens,
                temperature=0.0,
                **penalty_kwargs,
            )
        )
        with st.spinner("Analyzing…", show_time=True):
            first = next(stream, None)
        if first is None:
            return
        # stream_generate yields GenerationResult chunks; .text is the incremental
        # delta, not the running text, so write_stream concatenates them.
        _record_finish(first)
        yield first.text
        for chunk in stream:
            _record_finish(chunk)
            yield chunk.text

    try:
        # write_stream renders the tokens live and returns the concatenated string.
        # ``cursor`` marks that a slow local model is still emitting during the gaps
        # between bursts; it is drawn only in the live container and never enters the
        # accumulated return value, so parse_response and the stored raw text are
        # bit-identical with or without it.
        # The streamed output (incl. thinking sentinels / raw localization JSON) is
        # replaced by the caller's clean persisted render on the st.rerun() it issues
        # after storing this return value.
        return st.write_stream(_deltas(), cursor="▌")
    except Exception as e:
        st.error(f"Inference failed: {e}", icon=":material/error:")
        return None


def render_thought(raw_response: str, is_thinking: bool) -> str:
    """Split off any thinking trace into an expander; return the answer text."""
    thought, response = parse_response(raw_response, is_thinking)
    if thought is not None:
        # type="compact" is documented as "ideal for displaying AI reasoning,
        # thoughts, or collapsible metadata without visual clutter" -- a literal
        # description of this element. Kept default-bordered on the settings expander
        # in tab_settings, which holds real widgets rather than read-only prose.
        #
        # No icon= here, or on any other expander in this app, however apt the glyph:
        # AppTest classifies an expandable block by whether it carries an icon
        # (element_tree.py: `if block.expandable.icon: Status(...) else Expander(...)`),
        # so an icon moves the element out of ``at.expander`` and into ``at.status``,
        # where ``.state`` then raises "Unknown Status state" on any glyph that isn't
        # one of st.status's own three. That breaks both the "Thinking trace" lookups
        # and the CT/WSI ``at.status[0].state`` error assertions.
        with st.expander("Thinking trace", type="compact"):
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
    # No icon= (see render_thought): an icon reclassifies this as an st.status under
    # AppTest and breaks the harness's status/expander discrimination.
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
    """Identity of an uploaded file for staleness checks; () when no file.

    Keys on ``file_id`` -- the same identity ``_UPLOAD_HASH`` hands the caches, and
    what ``UploadedFile.__hash__`` itself uses. Name + size collides on exactly the
    input this app sees most: per-slice DICOMs off one scanner share a name pattern
    (``IM_0001``) and, at a fixed protocol, a byte size, so two different studies
    could compare equal and ``fresh_result_or_hint`` would serve the previous
    patient's result as fresh, with no "Inputs changed" hint. Falls back to name +
    size for anything without a ``file_id`` (a plain BytesIO in a unit test).

    A persisted result stores the signature of the inputs that produced it, so the
    render block drops it once the upload (or any tracked input) changes.
    """
    if uploaded is None:
        return ()
    file_id = getattr(uploaded, "file_id", None)
    if file_id is not None:
        return (file_id,)
    return (getattr(uploaded, "name", ""), getattr(uploaded, "size", 0))


STALE_RESULT_HINT = "Inputs changed since this result — click Run to refresh."
EMPTY_OUTPUT_HINT = "The response appears here after you click **Run**."

# Inputs : output, evenly. The inputs side carries more than its name suggests -- the
# controls plus every image the model saw -- and at 2:3 it wrapped file chips and
# captions on a 1280px screen while the output column sat half empty beside a short
# CXR report. Even halves also hold the report near the ~70-character measure prose
# reads best at on a laptop; 3/5 of the page ran it to ~90.
WORKSPACE_SPEC = [1, 1]


def workspace_columns():
    """Split a tab into ``(inputs, output)`` columns -- one grammar for all four.

    Inputs: the question, uploads, mode controls, Model settings and Run, then the
    imagery the model sees (the CXR preview, the windowed CT slices, the WSI overview
    and a sample patch). Output: what the model says -- preprocessing status, the live
    stream, the stale hint, and the persisted report. Keeping Run in the inputs column
    and every image under it means Run stays above the fold once a study is attached,
    while the report starts at the top of its own column beside the study.

    Below 640px Streamlit stacks the columns, inputs above output. That is NOT the
    old single-column order: the study preview (and, on a CT/WSI re-run, the previous
    result's imagery, faded as stale) now sits between Run and the live stream, so on
    a phone-width window a tap on Run changes nothing in view until you scroll. The
    server can't see the viewport to reorder for narrow screens, and reordering for
    everyone would put the image back above Run on the desktop this app runs on.
    """
    return st.columns(WORKSPACE_SPEC, gap="large")


def fresh_result_or_hint(key: str, live_sig, empty_hint: bool = False) -> dict | None:
    """Single source of truth for the staleness gate shared by every tab.

    Returns the persisted result stored under ``key`` when its recorded ``sig`` still
    matches the live inputs. When a result exists but is stale, show a hint and return
    None — an abrupt vanish (the old behavior) reads as a bug in a clinical tool.
    Returns None when there is no stored result at all, captioning the empty output
    column when ``empty_hint`` is set -- in the wide workspace a blank right half
    reads as broken. Callers pass ``not clicked`` (the raw button value, not the
    ``run_requested`` gate), so a click never ends beside an invitation to click:
    a run that just failed shows its error alone, and a click with no question
    shows only "Enter a question first." under the button.
    """
    result = st.session_state.get(key)
    if result is None:
        if empty_hint:
            st.caption(EMPTY_OUTPUT_HINT)
        return None
    if result["sig"] != live_sig:
        st.info(STALE_RESULT_HINT, icon=":material/refresh:")
        return None
    return result


def run_requested(clicked: bool, prompt: str) -> bool:
    """True when Run was clicked *and* a question has been entered.

    The Run buttons are deliberately not gated on ``prompt`` via ``disabled=``. An
    st.text_input commits its value to session state on blur or Enter only, so while
    the typed text is still uncommitted the button renders disabled -- and a disabled
    button dispatches no click event. The user's first click is spent blurring the
    input (which commits the text and enables the button) and the app appears to do
    nothing; only the second click runs. That is guaranteed whenever typing the
    prompt is the last action before clicking Run, i.e. on every tab's first use.

    Keeping the button live lets the text change and the click arrive in the same
    rerun. Gates on *uploads* stay on ``disabled=`` -- a file_uploader commits
    eagerly, so those never hit this trap. Callers must wrap the button call rather
    than returning early on an empty prompt, so the staleness render below still runs
    and a persisted result does not blink out.
    """
    if not clicked:
        return False
    if not prompt:
        st.warning("Enter a question first.", icon=":material/edit_note:")
        return False
    return True


@st.fragment
def render_ask_tab(model, processor, config):
    inputs, output = workspace_columns()
    with inputs:
        st.caption("Ask a medical question. No image required.")
        prompt = st.text_input(
            "Enter your question",
            placeholder="e.g. What causes a pleural effusion?",
            key="ask_prompt",
        ).strip()
        instruction, is_thinking = tab_settings("ask", DEFAULT_INSTRUCTION_TEXT)
        clicked = st.button("Run", type="primary", width="stretch", key="ask_run")
        go = run_requested(clicked, prompt)

    # Drop a persisted answer once the question or the system instruction changes:
    # the persona is fed to the model as the system message, so it is as much a
    # run-defining input as the prompt itself.
    ask_sig = (prompt, instruction)

    with output:
        if go:
            full_instruction, max_new_tokens = get_generation_params(
                has_image=False, is_thinking=is_thinking, system_instruction=instruction
            )
            messages = build_messages(prompt, full_instruction)
            finish: dict = {}
            raw = run_model(
                model, processor, config, messages, [], max_new_tokens, finish=finish
            )
            # Persist the run so it survives later reruns: editing any widget reruns
            # the script and the Run button returns False, which would otherwise wipe
            # the answer. The render below reads from session_state on every rerun;
            # the stored ``sig`` (prompt + persona) lets it drop the result once
            # either changes. On success, st.rerun() so the just-streamed raw text is
            # discarded and only the clean persisted render shows (no duplicate). On
            # failure the error stays put.
            if raw is None:
                st.session_state["ask_result"] = None
            else:
                st.session_state["ask_result"] = {
                    "raw": raw,
                    "is_thinking": is_thinking,
                    # Deliberately not part of ``sig``: truncation is an outcome of
                    # the run, not an input to it, so including it would strand the
                    # very result it describes on the next rerun.
                    "truncated": hit_token_cap(finish),
                    "sig": ask_sig,
                }
                st.rerun()

        result = fresh_result_or_hint("ask_result", ask_sig, empty_hint=not clicked)
        if result is not None:
            warn_if_truncated(result)
            show_response(render_thought(result["raw"], result["is_thinking"]))


@st.fragment
def render_cxr_tab(model, processor, config):
    inputs, output = workspace_columns()
    with inputs:
        st.caption(
            "Analyze a chest X-ray. Add a second image to compare two studies, or "
            "turn on 'Locate anatomy' to outline structures with bounding boxes."
        )
        prompt = st.text_input(
            "Enter your question",
            placeholder="e.g. Describe this chest X-ray",
            key="cxr_prompt",
        ).strip()

        # max_upload_size pins these back to Streamlit's stock 200 MB: the global
        # ceiling in .streamlit/config.toml is raised to 2000 for whole-slide images,
        # and a radiograph has no business anywhere near that.
        upload1 = st.file_uploader(
            "Upload a chest X-ray",
            type=IMAGE_TYPES,
            max_upload_size=NON_WSI_MAX_UPLOAD_MB,
            key="cxr_image1",
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
                max_upload_size=NON_WSI_MAX_UPLOAD_MB,
                key="cxr_image2",
            )
            image2 = load_uploaded_image(upload2)

        images = [img for img in (image1, image2) if img is not None]
        has_image = len(images) >= 1
        is_comparing = len(images) == 2

        is_localizing = st.toggle(
            "Locate anatomy (bounding boxes)",
            disabled=len(images) != 1,
            help="Outline anatomy with bounding boxes. Requires a single image.",
            key="cxr_localize",
        )
        if is_localizing and len(images) == 1:
            st.caption(
                ":material/info: Localization uses a built-in prompt; the System "
                "instruction below is ignored in this mode."
            )
        elif is_comparing:
            st.caption(
                ":material/info: Comparison mode: both images are sent to the model "
                "together."
            )

        # Rendered last, immediately above Run, the way the other three tabs do it:
        # the collapsed persona box is advanced and rarely edited, so it should not
        # sit in the middle of the tab's primary controls. (auto_switch still tracks
        # the comparison persona -- the widget is created on every run either way.)
        default_instruction = (
            DEFAULT_INSTRUCTION_COMPARE if is_comparing else DEFAULT_INSTRUCTION_IMAGE
        )
        instruction, is_thinking = tab_settings(
            "cxr", default_instruction, auto_switch=True
        )
        clicked = st.button("Run", type="primary", width="stretch", key="cxr_run")
        go = run_requested(clicked, prompt)

        # The preview renders under Run, not between the uploader and the controls
        # as it did in the single-column layout: a portrait radiograph is taller than
        # the viewport's spare height, so above Run it pushed the button off-screen
        # the moment an image was attached. Side by side in comparison mode --
        # seeing both studies at once is the whole point of a longitudinal read.
        if image1 is not None and image2 is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.image(image1, caption="First image", width="stretch")
            with col2:
                st.image(image2, caption="Second image", width="stretch")
        elif image1 is not None:
            st.image(image1, caption="Uploaded image", width="stretch")

    # Signature of the inputs this result depends on: a stale result is dropped
    # (not rendered) once the prompt, either upload, the localize mode, or the system
    # instruction changes.
    cxr_sig = (
        prompt,
        is_localizing,
        _file_sig(upload1),
        _file_sig(upload2),
        # Localization runs a built-in prompt and ignores the editable persona (the
        # caption above says so), so editing it must not strand that result.
        None if (is_localizing and len(images) == 1) else instruction,
    )

    with output:
        if go:
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
            finish: dict = {}
            raw = run_model(
                model,
                processor,
                config,
                messages,
                model_images,
                max_new_tokens,
                penalize_repetition=not localize,
                finish=finish,
            )
            # Persist the finished run (see render_ask_tab). For localization, parse
            # the boxes and draw the annotation once here — strip any thinking trace
            # with parse_response so the expander is rendered only in the block below.
            # On success st.rerun() so the streamed raw text is replaced by the clean
            # render.
            if raw is None:
                st.session_state["cxr_result"] = None
            else:
                if localize:
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
                        "truncated": hit_token_cap(finish),
                        "sig": cxr_sig,
                    }
                else:
                    st.session_state["cxr_result"] = {
                        "mode": "text",
                        "raw": raw,
                        "is_thinking": is_thinking,
                        "truncated": hit_token_cap(finish),
                        "sig": cxr_sig,
                    }
                st.rerun()

        result = fresh_result_or_hint("cxr_result", cxr_sig, empty_hint=not clicked)
        if result is None:
            return
        # Ahead of the mode branch on purpose: the localization path below renders boxes
        # and a legend but never reaches show_response, and a box list clipped at the
        # token cap is exactly where an unflagged truncation costs the most.
        warn_if_truncated(result)
        response = render_thought(result["raw"], result["is_thinking"])
        if result["mode"] == "localize":
            if result["annotated"] is not None:
                st.image(
                    result["annotated"], caption="Localized anatomy", width="stretch"
                )
                st.markdown("### Detected structures")
                # The boxes are drawn on the image above, so list the labels rather
                # than the raw normalized coordinates (cryptic to a clinician). Badges
                # wrap into a compact legend instead of a bullet per box, keeping the
                # annotated image and its labels on screen together. A ']' inside a
                # model-emitted label would terminate the directive early, so strip it.
                labels = (
                    (box["label"] or "unlabeled").replace("]", " ")
                    for box in result["boxes"]
                )
                st.markdown(" ".join(f":blue-badge[{label}]" for label in labels))
            else:
                st.warning(
                    "No bounding boxes were returned.", icon=":material/search_off:"
                )
                show_response(response)
        else:
            show_response(response)


@st.fragment
def render_ct_tab(model, processor, config):
    inputs, output = workspace_columns()
    with inputs:
        st.caption(
            "Upload a CT series as individual DICOM slice files. Each slice is "
            "windowed into a false-color image (the representation MedGemma 1.5 is "
            "trained on)."
        )
        prompt = st.text_input(
            "Enter your question",
            placeholder="e.g. Are there hypodense liver lesions?",
            key="ct_prompt",
        ).strip()
        dicom_files = st.file_uploader(
            "Upload CT DICOM slices",
            accept_multiple_files=True,
            # No type= filter on purpose: per-slice DICOMs off a PACS or a study CD are
            # routinely extensionless. Say so, since this is the only uploader of the
            # four whose dropzone lists no accepted types.
            help="Files may be extensionless (e.g. IM_0001) — a .dcm extension is not "
            "required.",
            # The narrowing that matters most of the three: this uploader takes many
            # files at once, filters none of them, and reads every slice fully into
            # memory, so it should not inherit the slide tab's 2000 MB ceiling.
            max_upload_size=NON_WSI_MAX_UPLOAD_MB,
            key="ct_files",
        )

        default_slices, max_slices = ram_aware_slice_cap()
        if max_slices > 2:
            n_slices = st.slider(
                "Slices to analyze",
                min_value=2,
                max_value=max_slices,
                value=default_slices,
                help="Slices are sampled uniformly across the volume. The cap scales "
                "to your machine's memory.",
                key="ct_slices",
            )
        else:
            n_slices = 2
            st.caption("Limited memory detected: analyzing 2 slices.")

        instruction, is_thinking = tab_settings("ct", DEFAULT_INSTRUCTION_CT)
        clicked = st.button(
            "Run",
            type="primary",
            disabled=not dicom_files,
            width="stretch",
            key="ct_run",
        )
        go = run_requested(clicked, prompt)

    # Drop a persisted result once the prompt, uploaded slices, slice count, or
    # system instruction change.
    ct_sig = (
        prompt,
        tuple(_file_sig(f) for f in dicom_files or []),
        n_slices,
        instruction,
    )

    with output:
        if go:
            # Clear any prior run, then do the heavy work inside the button block;
            # persist the result so it survives later reruns (see render_ask_tab). An
            # st.status narrates the otherwise-silent preprocessing (DICOM read +
            # windowing); generation then streams in this column below the status,
            # so its live tokens aren't buried in a collapsed box.
            st.session_state["ct_result"] = None
            slice_images = None
            with st.status("Preparing CT series…", expanded=False) as status:
                status.update(label="Reading DICOM series…")
                try:
                    hu_slices = cached_ct_volume(dicom_files, n_slices)
                except Exception as e:
                    st.error(
                        f"Failed to read DICOM series: {e}", icon=":material/error:"
                    )
                    hu_slices = None
                if hu_slices is None:
                    status.update(
                        label="Could not read DICOM series",
                        state="error",
                        expanded=True,
                    )
                else:
                    status.update(label="Windowing slices…")
                    slice_images = [window_ct_slice(hu) for hu in hu_slices]
                    status.update(
                        label=f"Prepared {len(slice_images)} slices", state="complete"
                    )
            if slice_images is not None:
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
                finish: dict = {}
                raw = run_model(
                    model,
                    processor,
                    config,
                    messages,
                    slice_images,
                    max_new_tokens,
                    finish=finish,
                )
                if raw is not None:
                    st.session_state["ct_result"] = {
                        "preview": slice_images[0],
                        "thumbs": ct_thumbnails(slice_images),
                        "labels": labels,
                        "count": len(slice_images),
                        "raw": raw,
                        "is_thinking": is_thinking,
                        "truncated": hit_token_cap(finish),
                        "sig": ct_sig,
                    }
                    # Discard the streamed run so only the clean render (preview +
                    # response) shows; the status was live feedback during this run.
                    st.rerun()

        result = fresh_result_or_hint("ct_result", ct_sig, empty_hint=not clicked)
    if result is None:
        return
    with inputs:
        st.image(
            result["preview"],
            caption=f"Sample windowed slice (1 of {result['count']})",
            width="stretch",
        )
        # The model numbers its findings by slice, so give the reader a way to
        # actually look at the slice a finding names -- the single preview above left
        # "slice 7" unauditable. Collapsed by default: this is for checking a claim,
        # not browsing. (WSI needs no equivalent; its overlay already outlines every
        # sampled patch.)
        thumbs = result.get("thumbs")
        if thumbs:
            # An expander body is computed and shipped to the frontend even while
            # collapsed (Streamlit documents this), and st.image re-encodes every
            # thumbnail to PNG -- work paid on every fragment rerun of a panel most
            # sessions never open, and it scales with ram_aware_slice_cap's 64-slice
            # ceiling. on_change="rerun" + .open defers it to the click. The key is what
            # makes .open readable at all, and it doubles as the handle a test needs:
            # under AppTest the expander starts closed, so without it the body would be
            # unreachable and the gallery untestable.
            gallery = st.expander(
                f"View all {result['count']} windowed slices",
                type="compact",
                on_change="rerun",
                key="ct_gallery",
            )
            if gallery.open:
                with gallery:
                    st.image(thumbs, caption=result["labels"], width=180)
    with output:
        # The slice imagery lives in the inputs column, so the report column opens
        # with this warning and it sits directly above the text it qualifies.
        warn_if_truncated(result)
        show_response(render_thought(result["raw"], result["is_thinking"]))


@st.fragment
def render_wsi_tab(model, processor, config):
    inputs, output = workspace_columns()
    with inputs:
        st.caption(
            "Upload a whole-slide image (.svs/.ndpi/.tiff). Tissue patches are "
            "sampled at a chosen magnification and read as the 896px tiles MedGemma "
            "1.5 is trained on."
        )
        prompt = st.text_input(
            "Enter your question",
            placeholder="e.g. Describe the histologic findings",
            key="wsi_prompt",
        ).strip()
        slide_file = st.file_uploader("Upload a slide", type=WSI_TYPES, key="wsi_files")
        # segmented_control (not select_slider): these are four discrete
        # objective-power modes, like a microscope turret, and one-tap selection beats
        # landing a slider handle on a tick. ``required=True`` because a single-select
        # segmented_control otherwise returns None when the user taps the
        # already-selected chip -- which left no chip highlighted while an
        # ``or WSI_DEFAULT_MAG`` fallback quietly analyzed at 10x, i.e. the visible
        # control and the magnification sent to the model disagreed. It also narrows
        # the return type from ``int | None`` to ``int``.
        target_mag = st.segmented_control(
            "Magnification",
            options=WSI_MAGNIFICATIONS,
            default=WSI_DEFAULT_MAG,
            required=True,
            format_func=lambda m: f"{m}×",
            help="Higher magnification shows finer detail over less area. Clamped to "
            "the slide's available pyramid levels.",
            key="wsi_mag",
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
        clicked = st.button(
            "Run",
            type="primary",
            disabled=not slide_file,
            width="stretch",
            key="wsi_run",
        )
        go = run_requested(clicked, prompt)

    # Drop a persisted result once the prompt, slide, magnification, patch count, or
    # system instruction change.
    wsi_sig = (prompt, _file_sig(slide_file), target_mag, n_patches, instruction)

    with output:
        if go:
            # Clear any prior run, then do the heavy work inside the button block;
            # persist the result so it survives later reruns (see render_ask_tab). An
            # st.status narrates the otherwise-silent slide read + tissue sampling
            # (which can take many seconds on a multi-GB slide); generation then
            # streams in this column below the status, so its live tokens aren't
            # buried in a box.
            st.session_state["wsi_result"] = None
            overlay = None
            actual_mag = None
            with st.status("Preparing slide…", expanded=False) as status:
                status.update(label="Reading slide and sampling tissue…")
                try:
                    patches, overlay, actual_mag = cached_wsi_patches(
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
                    status.update(
                        label=f"Prepared {len(patches)} patches", state="complete"
                    )
            if patches is not None:
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
                finish: dict = {}
                raw = run_model(
                    model,
                    processor,
                    config,
                    messages,
                    patches,
                    max_new_tokens,
                    finish=finish,
                )
                if raw is not None:
                    st.session_state["wsi_result"] = {
                        "overlay": overlay,
                        "actual_mag": actual_mag,
                        "count": len(patches),
                        "preview": patches[0],
                        "raw": raw,
                        "is_thinking": is_thinking,
                        "truncated": hit_token_cap(finish),
                        "sig": wsi_sig,
                    }
                    # Discard the streamed run so only the clean render (overlay +
                    # sample patch + response) shows; the status was live feedback
                    # this run.
                    st.rerun()

        result = fresh_result_or_hint("wsi_result", wsi_sig, empty_hint=not clicked)
    if result is None:
        return
    with inputs:
        st.image(result["overlay"], caption="Tissue overview", width="stretch")
        st.caption(
            f"{result['count']} patches sampled at ~{result['actual_mag']:.1f}x."
        )
        st.image(
            result["preview"],
            caption=f"Sample patch (1 of {result['count']})",
            width="stretch",
        )
    with output:
        # First in the report column, as in render_ct_tab: the overview and sample
        # patch live in the inputs column, so nothing separates this from the text.
        warn_if_truncated(result)
        show_response(render_thought(result["raw"], result["is_thinking"]))


def render_sidebar() -> None:
    """App-level panel: what the app is, which model answers, what this Mac can take.

    Global facts only, by design. The per-tab controls stay in each tab's inputs
    column: ``st.tabs`` doesn't tell the server which tab is showing, so a sidebar
    can't follow the active tab without ``on_change="rerun"`` plus hand-rolled
    persistence for every hidden tab's widgets (Streamlit drops the state of a
    widget that isn't rendered on a run).

    The memory block reports what ``ram_aware_slice_cap`` decided, which before this
    was surfaced only in a slider tooltip -- or, on a small Mac, as a bare "Limited
    memory detected" caption with nothing saying what the limit was measured against.
    """
    # Local file, like page_icon: served from /media/, so no off-host request. When
    # the sidebar is collapsed it stays in the header as the app's only mark.
    st.logo(str(FAVICON_PATH), size="large")
    with st.sidebar:
        st.title("MedGemma Studio")
        st.caption(
            "Ask medical questions and analyze chest X-rays, CT series, and "
            "whole-slide pathology images with Google MedGemma."
        )

        st.subheader("Model")
        # A plain link, not a code span: linkUnderline = false in the theme, so the
        # link color is the only cue, and inline-code styling overrode it.
        st.markdown(f"[{MODEL_ID.split('/')[-1]}]({MODEL_CARD_URL})")
        st.caption(
            "Runs on this Mac through MLX: no image, scan, or slide leaves the "
            "machine. Decoding is greedy (temperature 0), so the same inputs give the "
            "same answer — rephrase the question rather than re-running it."
        )

        st.subheader("This Mac")
        total_gib = _cached_total_ram_gib()
        default_count, max_count = ram_aware_slice_cap(total_gib)
        with st.container(horizontal=True):
            st.metric("Memory", f"{total_gib:.0f} GiB")
            st.metric(
                "Per-run cap",
                max_count,
                help="The most CT slices or WSI patches a single run can analyze.",
            )
        if max_count > 2:
            st.caption(
                f"CT and pathology runs analyze up to {max_count} slices or patches "
                f"(default {default_count}), sized to fit in memory beside the model."
            )
        else:
            st.caption(
                "Limited memory: CT and pathology runs analyze 2 slices or patches, "
                "the floor below which multi-image inference would not fit."
            )

        st.caption(
            "MedGemma is governed by Google's "
            f"[Health AI Developer Foundations Terms of Use]({HAI_DEF_TERMS_URL})."
        )


def main():
    render_sidebar()
    # Above the tabs on every view, and in the main area rather than the sidebar: the
    # sidebar collapses (and starts collapsed on a phone), and this notice must not.
    st.warning(DISCLAIMER_TEXT, icon=":material/warning:")
    model, processor, config = load_model()
    tab_ask, tab_cxr, tab_ct, tab_wsi = st.tabs(
        [
            ":material/forum: Ask",
            ":material/radiology: Chest X-ray",
            ":material/readiness_score: Computed tomography",
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
