import ast
import dataclasses
import fnmatch
import inspect
import io
import json
import os
import re
import shutil
import subprocess
import tomllib
import typing
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from streamlit_app import (
    CT_THUMBNAIL_SIZE,
    LOCALIZATION_INSTRUCTION,
    _detect_total_ram_gib,
    _file_sig,
    _read_patch,
    _slide_objective_power,
    build_messages,
    ct_thumbnails,
    draw_boxes,
    effective_magnification,
    get_generation_params,
    load_ct_volume,
    load_wsi_patches,
    mag_from_mpp,
    mark_patches,
    normalize_hu,
    pad_to_square,
    parse_boxes,
    parse_response,
    patch_grid,
    pick_level,
    ram_aware_slice_cap,
    scale_box,
    subsample_indices,
    tissue_mask,
    tissue_patches,
    window_ct_slice,
)
from tests.dicom_helpers import dicom_bytes as _dicom_bytes

THINKING_INSTRUCTION = "SYSTEM INSTRUCTION: think silently if needed. Be helpful."

APP_SOURCE = Path(__file__).resolve().parent.parent / "streamlit_app.py"


def _calls_to(dotted: str) -> list[ast.Call]:
    """Every ``st.<name>(...)`` call in the app, as AST nodes.

    Structural rather than textual: a `source.count("...")` guard is satisfied by a
    comment that merely mentions the string, and says nothing about *which* call
    carries a kwarg. Parsing gives per-call-site assertions instead.
    """
    obj, attr = dotted.split(".", 1)
    tree = ast.parse(APP_SOURCE.read_text())
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == attr
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == obj
    ]


def _file_uploader_kwargs() -> dict[str, dict[str, str]]:
    """``{widget key: {kwarg name: source text of its value}}`` for every uploader."""
    out = {}
    for call in _calls_to("st.file_uploader"):
        kwargs = {k.arg: ast.unparse(k.value) for k in call.keywords if k.arg}
        key = kwargs.pop("key", None)
        if key is not None:
            out[ast.literal_eval(key)] = kwargs
    return out


@pytest.fixture
def sample_image():
    return Image.new("RGB", (10, 10))


class TestParseResponse:
    def test_plain_response(self):
        thought, answer = parse_response("Normal answer text", is_thinking=False)
        assert thought is None
        assert answer == "Normal answer text"

    def test_thinking_with_markers(self):
        raw = "<unused94>thought\nSome reasoning here<unused95>Final answer"
        thought, answer = parse_response(raw, is_thinking=True)
        assert thought == "Some reasoning here"
        assert answer == "Final answer"

    def test_thinking_enabled_no_markers(self):
        thought, answer = parse_response("Just a plain reply", is_thinking=True)
        assert thought is None
        assert answer == "Just a plain reply"

    def test_thinking_missing_prefix(self):
        raw = "Some reasoning<unused95>Final answer"
        thought, answer = parse_response(raw, is_thinking=True)
        assert thought == "Some reasoning"
        assert answer == "Final answer"


class TestBuildMessages:
    def test_text_only(self):
        msgs = build_messages("What is a fracture?", "You are a doctor.", images=None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a doctor."
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == [{"type": "text", "text": "What is a fracture?"}]

    def test_with_image(self, sample_image):
        msgs = build_messages(
            "Describe this", "You are a radiologist.", images=[sample_image]
        )
        assert len(msgs) == 2
        assert msgs[0]["content"] == "You are a radiologist."
        user_content = msgs[1]["content"]
        assert len(user_content) == 2
        assert user_content[0] == {"type": "text", "text": "Describe this"}
        assert user_content[1] == {"type": "image"}

    def test_with_two_images(self, sample_image):
        # One image placeholder per image, appended after the text, for comparison.
        msgs = build_messages(
            "Compare these",
            "You are a radiologist.",
            images=[sample_image, sample_image],
        )
        user_content = msgs[1]["content"]
        assert len(user_content) == 3
        assert user_content[0] == {"type": "text", "text": "Compare these"}
        assert user_content[1] == {"type": "image"}
        assert user_content[2] == {"type": "image"}

    def test_empty_image_list(self):
        # An empty list behaves like no image (no placeholder appended).
        msgs = build_messages("Hello", "You are a doctor.", images=[])
        assert msgs[1]["content"] == [{"type": "text", "text": "Hello"}]

    def test_with_image_labels(self, sample_image):
        # Labels are interleaved as text parts before each image (comparison mode).
        msgs = build_messages(
            "Compare these",
            "You are a radiologist.",
            images=[sample_image, sample_image],
            image_labels=["First image:", "Second image:"],
        )
        assert msgs[1]["content"] == [
            {"type": "text", "text": "Compare these"},
            {"type": "text", "text": "First image:"},
            {"type": "image"},
            {"type": "text", "text": "Second image:"},
            {"type": "image"},
        ]


class TestGetGenerationParams:
    @pytest.mark.parametrize(
        "has_image, is_thinking, expected_instruction, expected_tokens",
        [
            (True, True, THINKING_INSTRUCTION, 1300),
            (True, False, "Be helpful.", 300),
            (False, False, "Be helpful.", 500),
            (False, True, THINKING_INSTRUCTION, 1300),
        ],
        ids=["image+thinking", "image", "text", "text+thinking"],
    )
    def test_params(
        self, has_image, is_thinking, expected_instruction, expected_tokens
    ):
        instruction, tokens = get_generation_params(
            has_image=has_image,
            is_thinking=is_thinking,
            system_instruction="Be helpful.",
        )
        assert instruction == expected_instruction
        assert tokens == expected_tokens

    @pytest.mark.parametrize(
        "is_thinking, expected_tokens",
        [(False, 1000), (True, 1300)],
        ids=["localize", "localize+thinking"],
    )
    def test_localization_overrides_instruction(self, is_thinking, expected_tokens):
        instruction, tokens = get_generation_params(
            has_image=True,
            is_thinking=is_thinking,
            system_instruction="Be helpful.",
            is_localizing=True,
        )
        # Localization ignores the user's persona and uses the dedicated prompt.
        assert instruction == LOCALIZATION_INSTRUCTION
        assert tokens == expected_tokens

    @pytest.mark.parametrize(
        "is_thinking, expected_instruction, expected_tokens",
        [
            (False, "Be helpful.", 600),
            (True, THINKING_INSTRUCTION, 1600),
        ],
        ids=["compare", "compare+thinking"],
    )
    def test_comparison_params(
        self, is_thinking, expected_instruction, expected_tokens
    ):
        instruction, tokens = get_generation_params(
            has_image=True,
            is_thinking=is_thinking,
            system_instruction="Be helpful.",
            is_comparing=True,
        )
        # Comparison keeps the editable instruction but allocates a larger budget;
        # thinking still takes precedence over the comparison branch.
        assert instruction == expected_instruction
        assert tokens == expected_tokens

    @pytest.mark.parametrize(
        "is_thinking, expected_tokens",
        [(False, 1000), (True, 1300)],
        ids=["localize+compare", "localize+compare+thinking"],
    )
    def test_localization_takes_precedence_over_comparison(
        self, is_thinking, expected_tokens
    ):
        # Branch order is localizing > thinking > comparing: with both flags set,
        # localization wins and the comparison persona/budget is never reached. (The
        # UI call site is mutually exclusive, so this pins the helper's contract.)
        instruction, tokens = get_generation_params(
            has_image=True,
            is_thinking=is_thinking,
            system_instruction="Be helpful.",
            is_localizing=True,
            is_comparing=True,
        )
        assert instruction == LOCALIZATION_INSTRUCTION
        assert tokens == expected_tokens

    @pytest.mark.parametrize(
        "is_thinking, expected_instruction, expected_tokens",
        [
            (False, "Be helpful.", 2000),
            (True, THINKING_INSTRUCTION, 2500),
        ],
        ids=["ct", "ct+thinking"],
    )
    def test_ct_params(self, is_thinking, expected_instruction, expected_tokens):
        # CT keeps the editable persona but allocates a large multi-slice budget;
        # thinking takes precedence and bumps the budget further.
        instruction, tokens = get_generation_params(
            has_image=True,
            is_thinking=is_thinking,
            system_instruction="Be helpful.",
            is_ct=True,
        )
        assert instruction == expected_instruction
        assert tokens == expected_tokens

    @pytest.mark.parametrize(
        "is_thinking, expected_instruction, expected_tokens",
        [
            (False, "Be helpful.", 2000),
            (True, THINKING_INSTRUCTION, 2500),
        ],
        ids=["wsi", "wsi+thinking"],
    )
    def test_wsi_params(self, is_thinking, expected_instruction, expected_tokens):
        # WSI shares CT's multi-image budget (2000 / 2500 with thinking); the editable
        # pathology persona is kept as-is.
        instruction, tokens = get_generation_params(
            has_image=True,
            is_thinking=is_thinking,
            system_instruction="Be helpful.",
            is_wsi=True,
        )
        assert instruction == expected_instruction
        assert tokens == expected_tokens


class TestPadToSquare:
    def test_already_square_unchanged(self):
        img = Image.new("RGB", (32, 32))
        assert pad_to_square(img).size == (32, 32)

    def test_landscape_padded_to_square(self):
        assert pad_to_square(Image.new("RGB", (40, 20))).size == (40, 40)

    def test_portrait_padded_to_square(self):
        assert pad_to_square(Image.new("RGB", (20, 40))).size == (40, 40)

    def test_original_pinned_to_top_left(self):
        # White content goes in the top-left; the new region stays black padding.
        img = Image.new("RGB", (10, 6), color="white")
        padded = pad_to_square(img)
        assert padded.size == (10, 10)
        assert padded.getpixel((0, 0)) == (255, 255, 255)  # original region
        assert padded.getpixel((0, 9)) == (0, 0, 0)  # padded region (y=9 > 6)


class TestParseBoxes:
    def test_json_fence(self):
        resp = '```json\n[{"box_2d": [10, 20, 30, 40], "label": "right clavicle"}]\n```'
        assert parse_boxes(resp) == [
            {"box_2d": [10, 20, 30, 40], "label": "right clavicle"}
        ]

    def test_bare_list_no_fence(self):
        resp = 'Here you go: [{"box_2d": [1, 2, 3, 4], "label": "x"}] done.'
        assert parse_boxes(resp) == [{"box_2d": [1, 2, 3, 4], "label": "x"}]

    def test_multiple_boxes(self):
        resp = (
            '```\n[{"box_2d":[0,0,1,1],"label":"a"},'
            '{"box_2d":[2,2,3,3],"label":"b"}]\n```'
        )
        assert len(parse_boxes(resp)) == 2

    def test_unparseable_returns_empty(self):
        assert parse_boxes("no boxes here at all") == []

    def test_non_list_returns_empty(self):
        assert parse_boxes('```json\n{"box_2d": [1, 2, 3, 4]}\n```') == []

    def test_drops_wrong_length_box(self):
        resp = '[{"box_2d":[1,2,3],"label":"bad"},{"box_2d":[1,2,3,4],"label":"good"}]'
        assert parse_boxes(resp) == [{"box_2d": [1, 2, 3, 4], "label": "good"}]

    def test_missing_label_defaults_empty(self):
        assert parse_boxes('[{"box_2d":[1,2,3,4]}]') == [
            {"box_2d": [1, 2, 3, 4], "label": ""}
        ]

    @pytest.mark.parametrize(
        "bad_box",
        [
            '[{"box_2d": ["100", "200", "300", "400"], "label": "str coords"}]',
            '[{"box_2d": [100, null, 300, 400], "label": "null coord"}]',
            '[{"box_2d": [true, false, true, false], "label": "bool coords"}]',
        ],
        ids=["strings", "null", "bools"],
    )
    def test_drops_non_numeric_coords(self, bad_box):
        # Non-numeric coords would crash scale_box() (which runs outside the
        # inference try/except), so they must be dropped at parse time.
        assert parse_boxes(bad_box) == []

    def test_accepts_float_coords(self):
        assert parse_boxes('[{"box_2d": [1.5, 2.0, 3.5, 4.0], "label": "f"}]') == [
            {"box_2d": [1.5, 2.0, 3.5, 4.0], "label": "f"}
        ]

    def test_multiple_arrays_no_fence_returns_empty(self):
        # No fence + two arrays: rfind("]") spans both -> malformed JSON -> safely [].
        resp = 'First: [] then [{"box_2d": [1, 2, 3, 4], "label": "x"}]'
        assert parse_boxes(resp) == []


class TestScaleBox:
    def test_full_frame(self):
        assert scale_box([0, 0, 1000, 1000], 896) == (0, 0, 896, 896)

    def test_y_x_ordering(self):
        # box_2d is [y0, x0, y1, x1]; output is (x0, y0, x1, y1) pixels.
        assert scale_box([0, 500, 1000, 1000], 1000) == (500, 0, 1000, 1000)

    def test_rounds_every_corner(self):
        # Distinct fractional values per corner so a swap or per-axis rounding bug
        # is caught. Over a 900px square: 334->300.6->301, 666->599.4->599,
        # 333->299.7->300, 667->600.3->600. box_2d=[y0,x0,y1,x1] -> (x0,y0,x1,y1).
        assert scale_box([333, 334, 667, 666], 900) == (301, 300, 599, 600)

    def test_rounds_down_and_up(self):
        # Over a 10px square: 140 -> 1.4 -> 1 (down), 160 -> 1.6 -> 2 (up).
        assert scale_box([140, 160, 140, 160], 10) == (2, 1, 2, 1)

    def test_orders_inverted_box(self):
        # A model box with swapped corners is reordered so ImageDraw.rectangle
        # receives x0 <= x1, y0 <= y1 instead of raising.
        assert scale_box([800, 800, 200, 200], 1000) == (200, 200, 800, 800)


class TestDrawBoxes:
    def test_returns_rgb_same_size(self):
        out = draw_boxes(
            Image.new("RGB", (100, 100)),
            [{"box_2d": [100, 100, 500, 500], "label": "lung"}],
        )
        assert out.size == (100, 100)
        assert out.mode == "RGB"

    def test_empty_boxes_is_noop_copy(self):
        out = draw_boxes(Image.new("RGB", (50, 50)), [])
        assert out.size == (50, 50)

    def test_accepts_unlabeled_box(self):
        # A box with an empty label must not raise (skips the text draw).
        out = draw_boxes(
            Image.new("RGB", (60, 60)), [{"box_2d": [0, 0, 100, 100], "label": ""}]
        )
        assert out.size == (60, 60)

    def test_draws_box_at_scaled_location(self):
        # box_2d [100, 100, 900, 900] over a 100px square -> pixel rect (10,10,90,90).
        # The outline should be red; the unfilled interior should stay black.
        out = draw_boxes(
            Image.new("RGB", (100, 100)),
            [{"box_2d": [100, 100, 900, 900], "label": "box"}],
        )
        assert out.getpixel((10, 50)) == (255, 0, 0)  # on the box's left edge
        assert out.getpixel((50, 50)) == (0, 0, 0)  # interior is not filled

    def test_draws_inverted_box_without_error(self):
        # An inverted box (corners swapped) must not raise in ImageDraw.rectangle.
        out = draw_boxes(
            Image.new("RGB", (100, 100)),
            [{"box_2d": [900, 900, 100, 100], "label": "flipped"}],
        )
        assert out.size == (100, 100)

    def test_pad_draw_crop_round_trip_portrait(self):
        # Mirror main()'s pipeline for a portrait image: pad -> draw -> crop back.
        original = Image.new("RGB", (10, 20))  # portrait, padded on the right
        padded = pad_to_square(original)
        assert padded.size == (20, 20)
        # Box over the left half, inside the original 10px-wide region.
        annotated = draw_boxes(padded, [{"box_2d": [0, 0, 1000, 500], "label": "left"}])
        cropped = annotated.crop((0, 0, original.width, original.height))
        assert cropped.size == (10, 20)
        assert cropped.getpixel((0, 10)) == (255, 0, 0)  # left edge survives the crop


class TestNormalizeHu:
    def test_clamps_below_window_to_zero(self):
        assert normalize_hu(np.array([-2000.0]), -1024, 1024)[0] == 0.0

    def test_clamps_above_window_to_255(self):
        assert normalize_hu(np.array([5000.0]), -1024, 1024)[0] == 255.0

    def test_midpoint_is_half_scale(self):
        # HU 0 sits halfway through the wide window -> 127.5.
        assert normalize_hu(np.array([0.0]), -1024, 1024)[0] == pytest.approx(127.5)

    def test_linear_within_window(self):
        # 20 is a quarter of the [0, 80] brain window.
        assert normalize_hu(np.array([20.0]), 0, 80)[0] == pytest.approx(0.25 * 255)

    def test_preserves_shape(self):
        assert normalize_hu(np.zeros((3, 4)), -1024, 1024).shape == (3, 4)


class TestWindowCtSlice:
    def test_returns_rgb_image_same_hw(self):
        img = window_ct_slice(np.zeros((8, 6)))
        assert isinstance(img, Image.Image)
        assert img.mode == "RGB"
        assert img.size == (6, 8)  # PIL size is (width, height)

    def test_channel_values_for_constant_slice(self):
        # HU 0 everywhere, per CT_WINDOWS:
        #   R wide  (-1024, 1024): (0+1024)/2048*255 = 127.5 -> 128
        #   G soft  (-135, 215):   (0+135)/350*255   = 98.36 -> 98
        #   B brain (0, 80):       (0-0)/80*255       = 0
        img = window_ct_slice(np.zeros((2, 2)))
        assert img.getpixel((0, 0)) == (128, 98, 0)

    def test_respects_custom_windows(self):
        # Three identical windows -> a gray image; HU 50 of [0,100] -> 127.5 -> 128.
        img = window_ct_slice(
            np.full((2, 2), 50.0), windows=[(0, 100), (0, 100), (0, 100)]
        )
        assert img.getpixel((0, 0)) == (128, 128, 128)


class TestCtThumbnails:
    def test_clamps_to_the_bounding_box_and_keeps_aspect(self):
        (thumb,) = ct_thumbnails([Image.new("RGB", (1024, 512))])
        assert thumb.size == (256, 128)
        assert max(thumb.size) <= max(CT_THUMBNAIL_SIZE)

    def test_leaves_the_originals_untouched(self):
        # thumbnail() scales in place, and the originals are what reach the model.
        original = Image.new("RGB", (600, 600))
        ct_thumbnails([original])
        assert original.size == (600, 600)

    def test_preserves_count_and_order(self):
        slices = [Image.new("RGB", (400, 100 * (i + 1))) for i in range(3)]
        assert [t.size for t in ct_thumbnails(slices)] == [
            (256, 64),
            (256, 128),
            (256, 192),
        ]

    def test_does_not_upscale_a_small_slice(self):
        (thumb,) = ct_thumbnails([Image.new("RGB", (100, 80))])
        assert thumb.size == (100, 80)


class TestSubsampleIndices:
    def test_returns_all_when_fewer_than_cap(self):
        assert subsample_indices(3, 10) == [0, 1, 2]

    def test_returns_all_when_equal_to_cap(self):
        assert subsample_indices(5, 5) == [0, 1, 2, 3, 4]

    def test_empty_volume(self):
        assert subsample_indices(0, 8) == []

    def test_even_spread_includes_endpoints(self):
        assert subsample_indices(10, 4) == [0, 3, 6, 9]

    def test_cap_of_one_picks_middle(self):
        assert subsample_indices(10, 1) == [5]

    def test_never_exceeds_cap_and_spans_volume(self):
        idx = subsample_indices(100, 16)
        assert len(idx) == 16
        assert idx[0] == 0
        assert idx[-1] == 99

    def test_indices_sorted_and_in_range(self):
        idx = subsample_indices(57, 13)
        assert idx == sorted(idx)
        assert all(0 <= i < 57 for i in idx)


class TestFileSig:
    """Guard the identity the staleness gate compares uploads by.

    Reverting ``_file_sig`` to (name, size) leaves every AppTest staleness test green,
    because those all vary the filename -- so without this class the fix it encodes is
    a comment, not an invariant.
    """

    def test_distinguishes_uploads_sharing_a_name_and_size(self):
        # The failure this exists to prevent: per-slice DICOMs off one scanner share
        # a name pattern and, at a fixed protocol, a byte size. Under (name, size)
        # two different studies compare equal, so fresh_result_or_hint serves the
        # previous patient's report as fresh with no "Inputs changed" hint.
        a = SimpleNamespace(file_id="upload-a", name="IM_0001", size=524288)
        b = SimpleNamespace(file_id="upload-b", name="IM_0001", size=524288)
        assert _file_sig(a) != _file_sig(b)

    def test_is_stable_for_one_upload_across_reruns(self):
        # The other half: the signature must NOT change for the same upload, or every
        # rerun would strand its own result behind a staleness hint.
        upload = SimpleNamespace(file_id="upload-a", name="IM_0001", size=524288)
        assert _file_sig(upload) == _file_sig(upload)

    def test_falls_back_to_name_and_size_without_a_file_id(self):
        # A plain BytesIO (unit tests, and anything not routed through Streamlit's
        # uploader) has no file_id; the fallback keeps it comparable rather than
        # collapsing every such object to one signature.
        x = SimpleNamespace(name="a.dcm", size=10)
        y = SimpleNamespace(name="b.dcm", size=10)
        assert _file_sig(x) == ("a.dcm", 10)
        assert _file_sig(x) != _file_sig(y)

    def test_no_file_is_the_empty_signature(self):
        assert _file_sig(None) == ()


class TestLoadCtVolume:
    def test_sorts_by_instance_number_and_converts_to_hu(self):
        # Files out of order; the fill value encodes order so we can verify sorting.
        files = [_dicom_bytes(3, 300), _dicom_bytes(1, 100), _dicom_bytes(2, 200)]
        vol = load_ct_volume(files, max_slices=10)
        assert len(vol) == 3
        # Sorted 1,2,3 -> fills 100,200,300; HU = fill + intercept(-1024).
        assert [v[0, 0] for v in vol] == [100 - 1024, 200 - 1024, 300 - 1024]

    def test_subsamples_to_cap(self):
        files = [_dicom_bytes(i, 100 + i) for i in range(1, 21)]  # 20 slices
        assert len(load_ct_volume(files, max_slices=5)) == 5

    def test_downcasts_float64_hu_arrays_to_float32(self):
        # apply_rescale returns float64; nothing downstream wants that precision
        # (normalize_hu opens with .astype(np.float32)), and it doubles what
        # cached_ct_volume holds. Values must survive the cast unchanged.
        vol = load_ct_volume([_dicom_bytes(1, 100)], max_slices=1)
        assert vol[0].dtype == np.float32
        assert vol[0][0, 0] == 100 - 1024

    def test_leaves_an_integer_hu_array_alone(self, monkeypatch):
        # The `if hu.dtype == np.float64` guard is load-bearing, not defensive noise.
        # apply_rescale returns uint8/uint16 when a Modality LUT Sequence (0028,3000)
        # is present, and the array untouched when there are no rescale tags at all.
        # An unconditional .astype(np.float32) -- which reads as obviously equivalent
        # -- would *quadruple* a uint8 slice inside the very cache the downcast exists
        # to shrink. Nothing else notices: dicom_helpers only builds rescale-tagged
        # slices, so this is the only test that fails when the guard is dropped.
        import streamlit_app

        monkeypatch.setattr(
            streamlit_app,
            "apply_rescale",
            lambda arr, ds: np.asarray(arr, dtype=np.uint8),
        )
        vol = load_ct_volume([_dicom_bytes(1, 100)], max_slices=1)
        assert vol[0].dtype == np.uint8, (
            "the float64 guard was dropped; an integer Modality-LUT slice is now "
            "upcast, growing the cache instead of shrinking it"
        )

    def test_applies_rescale_slope_and_intercept(self):
        files = [_dicom_bytes(1, 10, slope=2.0, intercept=-1000.0)]
        assert load_ct_volume(files, max_slices=4)[0][0, 0] == 10 * 2.0 - 1000.0

    def test_rejects_multiple_series(self):
        # Mixing series would interleave anatomically unrelated slices into one
        # bogus volume, so it is rejected rather than silently merged.
        files = [
            _dicom_bytes(1, 100, series_uid="1.2.3"),
            _dicom_bytes(2, 200, series_uid="1.2.4"),
        ]
        with pytest.raises(ValueError, match="Multiple DICOM series"):
            load_ct_volume(files, max_slices=10)

    def test_rejects_multi_frame_slice(self):
        # A multi-frame DICOM yields a 3D pixel array that window_ct_slice (run
        # outside the caller's try/except) cannot handle, so it is rejected here.
        with pytest.raises(ValueError, match="single-frame"):
            load_ct_volume([_dicom_bytes(1, 100, frames=3)], max_slices=4)

    def test_rereads_same_streams_after_position_advances(self):
        # Streamlit keeps the uploaded BytesIO objects in session_state, so a
        # second Run re-reads the SAME streams. dcmread advances the file position,
        # so without an internal rewind the second pass would raise
        # InvalidDicomError. Reuse one list across two calls to lock in the rewind.
        files = [_dicom_bytes(2, 200), _dicom_bytes(1, 100)]
        first = load_ct_volume(files, max_slices=10)
        second = load_ct_volume(files, max_slices=10)
        assert len(first) == len(second) == 2
        assert [v[0, 0] for v in second] == [v[0, 0] for v in first]


class TestRamAwareSliceCap:
    def test_32gib_yields_10_and_20(self):
        assert ram_aware_slice_cap(total_ram_gib=32) == (10, 20)

    def test_24gib_unlocks_a_small_slider(self):
        # The 8-bit retune moved this tier off the 2-slice floor; pin it, since
        # a 24 GB Mac is the smallest machine that now gets a real slider.
        assert ram_aware_slice_cap(total_ram_gib=24) == (3, 6)

    def test_floor_extends_to_just_under_21_8_gib(self):
        # max > 2 needs budget >= 1.8 GB, i.e. base + headroom + one slice.
        assert ram_aware_slice_cap(total_ram_gib=21) == (2, 2)
        assert ram_aware_slice_cap(total_ram_gib=22) == (2, 3)

    def test_scales_up_with_more_ram(self):
        # 40 GiB, not 64: at 64 the result is the hard_max clamp, so the
        # assertion would guard clamping (already covered below) instead of the
        # scaling this test is named for. int((40 - 9 - 11) / 0.6) == 33.
        default, maximum = ram_aware_slice_cap(total_ram_gib=40)
        assert maximum == 33
        assert default <= maximum

    def test_clamped_to_hard_max(self):
        _, maximum = ram_aware_slice_cap(total_ram_gib=512)
        assert maximum == 64

    def test_floors_on_low_ram(self):
        # Below base + headroom -> the 2-slice floor, never zero or negative.
        assert ram_aware_slice_cap(total_ram_gib=16) == (2, 2)

    def test_default_never_exceeds_max(self):
        for ram in (16, 24, 32, 48, 64, 128):
            default, maximum = ram_aware_slice_cap(total_ram_gib=ram)
            assert 2 <= default <= maximum


class TestDetectTotalRamGib:
    def test_sysconf_branch(self, monkeypatch):
        # 32 GiB via SC_PHYS_PAGES * SC_PAGE_SIZE.
        monkeypatch.setattr(
            os, "sysconf_names", {"SC_PHYS_PAGES": 0, "SC_PAGE_SIZE": 1}
        )
        pages = 32 * 1024**3 // 4096
        monkeypatch.setattr(
            os, "sysconf", lambda n: pages if n == "SC_PHYS_PAGES" else 4096
        )
        assert _detect_total_ram_gib() == 32.0

    def test_sysconf_indeterminate_falls_back_to_sysctl(self, monkeypatch):
        # POSIX -1 (indeterminate) is returned, not raised, so the try/except alone
        # would let it through as a negative RAM total.
        monkeypatch.setattr(
            os, "sysconf_names", {"SC_PHYS_PAGES": 0, "SC_PAGE_SIZE": 1}
        )
        monkeypatch.setattr(os, "sysconf", lambda n: -1)
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout=f"{64 * 1024**3}\n"),
        )
        assert _detect_total_ram_gib() == 64.0

    def test_sysctl_fallback_when_sysconf_unavailable(self, monkeypatch):
        # No sysconf keys -> parse `sysctl -n hw.memsize`.
        monkeypatch.setattr(os, "sysconf_names", {})
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **k: SimpleNamespace(stdout=f"{32 * 1024**3}\n"),
        )
        assert _detect_total_ram_gib() == 32.0

    def test_conservative_default_when_both_fail(self, monkeypatch):
        monkeypatch.setattr(os, "sysconf_names", {})

        def _raise(*a, **k):
            raise OSError("no sysctl")

        monkeypatch.setattr(subprocess, "run", _raise)
        assert _detect_total_ram_gib() == 16.0


class TestMagFromMpp:
    @pytest.mark.parametrize("mpp, mag", [(0.25, 40.0), (0.5, 20.0), (1.0, 10.0)])
    def test_objective_power_from_microns(self, mpp, mag):
        assert mag_from_mpp(mpp) == pytest.approx(mag)


class TestEffectiveMagnification:
    def test_base_level_is_objective_power(self):
        assert effective_magnification(40.0, 1.0) == 40.0

    def test_downsampled_level_scales_down(self):
        assert effective_magnification(40.0, 4.0) == 10.0


class TestPickLevel:
    def test_picks_closest_magnification(self):
        # downsamples [1, 4, 16] @ 40x objective -> effective mags [40, 10, 2.5].
        assert pick_level([1, 4, 16], 40, 10) == 1
        assert pick_level([1, 4, 16], 40, 40) == 0
        assert pick_level([1, 4, 16], 40, 5) == 2  # 2.5 is closer to 5 than 10 is

    def test_single_level_always_zero(self):
        assert pick_level([1.0], 40, 10) == 0


class TestPatchGrid:
    def test_non_overlapping_row_major(self):
        assert patch_grid(2000, 2000, 896) == [(0, 0), (896, 0), (0, 896), (896, 896)]

    def test_drops_partial_edge_tiles(self):
        assert all(
            x + 896 <= 2000 and y + 896 <= 2000 for x, y in patch_grid(2000, 2000, 896)
        )

    def test_too_small_is_empty(self):
        assert patch_grid(500, 500, 896) == []

    def test_exact_fit_single_tile(self):
        assert patch_grid(896, 896, 896) == [(0, 0)]


class TestTissueMask:
    def test_white_glass_is_not_tissue(self):
        assert not tissue_mask(np.full((4, 4, 3), 255, dtype=np.uint8)).any()

    def test_grey_is_not_tissue(self):
        # Zero saturation (R == G == B) reads as background regardless of brightness.
        assert not tissue_mask(np.full((4, 4, 3), 128, dtype=np.uint8)).any()

    def test_saturated_stain_is_tissue(self):
        purple = np.zeros((4, 4, 3), dtype=np.uint8)
        purple[..., 0], purple[..., 2] = 150, 140  # high R/B, low G -> saturated
        assert tissue_mask(purple).all()

    def test_preserves_2d_shape(self):
        assert tissue_mask(np.zeros((6, 5, 3), dtype=np.uint8)).shape == (6, 5)


class TestTissuePatches:
    def test_keeps_only_tissue_side(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[:, :5] = True  # left half of the slide is tissue
        grid = patch_grid(1000, 1000, 500)  # (0,0),(500,0),(0,500),(500,500)
        kept = tissue_patches(grid, mask, (1000, 1000), 500, min_fraction=0.25)
        assert kept == [(0, 0), (0, 500)]  # only the x < 500 column survives

    def test_min_fraction_threshold(self):
        mask = np.zeros((10, 10), dtype=bool)
        mask[:, :5] = True  # a single full-width patch is exactly half tissue
        grid = [(0, 0)]
        assert tissue_patches(grid, mask, (1000, 1000), 1000, 0.25) == [(0, 0)]
        assert tissue_patches(grid, mask, (1000, 1000), 1000, 0.75) == []

    def test_empty_grid(self):
        assert tissue_patches([], np.ones((4, 4), dtype=bool), (1000, 1000), 500) == []


class TestMarkPatches:
    def test_returns_rgb_same_size(self):
        out = mark_patches(Image.new("RGB", (100, 100)), [(0, 0)], (1000, 1000), 500)
        assert out.size == (100, 100)
        assert out.mode == "RGB"

    def test_draws_red_outline_at_scaled_location(self):
        # patch (0,0) size 500 over a 1000px level -> rect (0,0,50,50) on the thumbnail.
        out = mark_patches(Image.new("RGB", (100, 100)), [(0, 0)], (1000, 1000), 500)
        assert out.getpixel((0, 25)) == (255, 0, 0)  # on the left edge
        assert out.getpixel((25, 25)) == (0, 0, 0)  # interior is not filled

    def test_empty_coords_is_noop_copy(self):
        out = mark_patches(Image.new("RGB", (40, 40)), [], (1000, 1000), 500)
        assert out.size == (40, 40)


class _FakeSlide:
    """Minimal OpenSlide stand-in: just the surface load_wsi_patches touches."""

    def __init__(self, *, level_dimensions, level_downsamples, properties, thumbnail):
        self.level_dimensions = level_dimensions
        self.level_downsamples = level_downsamples
        self.dimensions = level_dimensions[0]
        self.properties = properties
        self._thumbnail = thumbnail
        self.closed = False
        self.read_calls: list = []

    def get_thumbnail(self, size):
        return self._thumbnail

    def read_region(self, location, level, size):
        self.read_calls.append((location, level, size))
        return Image.new("RGBA", size, (150, 40, 140, 255))

    def close(self):
        self.closed = True


def _tissue_thumbnail(size=(800, 800)):
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    arr[..., 0], arr[..., 2] = 150, 140  # saturated purple -> all tissue
    return Image.fromarray(arr, "RGB")


def _make_slide(**overrides):
    kwargs = {
        "level_dimensions": [(3000, 3000)],
        "level_downsamples": [1.0],
        "properties": {"openslide.objective-power": "40"},
        "thumbnail": _tissue_thumbnail(),
    }
    kwargs.update(overrides)
    return _FakeSlide(**kwargs)


class TestLoadWsiPatches:
    def test_returns_capped_rgb_patches(self, monkeypatch):
        slide = _make_slide()  # 3000x3000 -> a 3x3 grid of nine tissue patches
        monkeypatch.setattr("openslide.OpenSlide", lambda path: slide)
        patches, overlay, actual_mag = load_wsi_patches(
            io.BytesIO(b"x"), 40, max_patches=4
        )
        assert len(patches) == 4  # capped from nine
        assert all(p.mode == "RGB" and p.size == (896, 896) for p in patches)
        assert isinstance(overlay, Image.Image)
        assert actual_mag == 40.0

    def test_rereads_the_same_stream_after_position_advances(self, monkeypatch):
        # The WSI mirror of TestLoadCtVolume's rewind guard. Streamlit keeps the
        # uploaded BytesIO in session_state, so a second Run spills the SAME stream --
        # already at EOF from the first. ``getvalue()`` is position-independent and so
        # survives that; a chunked ``copyfileobj``/``.read()`` refactor (tempting,
        # since a slide can be multi-GB) would spill an EMPTY file the second time and
        # fail on a slide that worked a moment earlier. Only reading back what landed
        # on disk catches it -- a fake that ignores ``path`` passes either way.
        spilled = []

        def _open(path):
            spilled.append(Path(path).read_bytes())
            return _make_slide()

        monkeypatch.setattr("openslide.OpenSlide", _open)
        upload = io.BytesIO(b"slide-bytes")
        load_wsi_patches(upload, 40, max_patches=2)
        upload.read()  # leave the stream at EOF, where a second Run finds it
        load_wsi_patches(upload, 40, max_patches=2)
        assert spilled == [b"slide-bytes", b"slide-bytes"]

    def test_tissue_filtering_reduces_patch_count(self, monkeypatch):
        # Tissue only on the slide's left third: the 3x3 grid (nine tiles) is filtered
        # end-to-end down to the three left-column patches, even though eight were
        # requested. This is the real 6->3-style reduction seen on actual slides
        # (the all-tissue fixture above never exercises the filter shrinking the set).
        thumb = np.full((800, 800, 3), 255, dtype=np.uint8)  # white glass
        thumb[:, :250, 0], thumb[:, :250, 2] = 150, 140  # left third = saturated tissue
        slide = _make_slide(thumbnail=Image.fromarray(thumb, "RGB"))
        monkeypatch.setattr("openslide.OpenSlide", lambda path: slide)
        patches, _, _ = load_wsi_patches(io.BytesIO(b"x"), 40, max_patches=8)
        assert len(patches) == 3  # nine candidates, only the left column is tissue

    def test_reads_at_level0_coordinates(self, monkeypatch):
        # A 10x target on a 40x slide selects level 1 (downsample 4); read_region must
        # get LEVEL-0 locations (level-pixel coords * downsample) at that level.
        slide = _make_slide(
            level_dimensions=[(8000, 8000), (2000, 2000)],
            level_downsamples=[1.0, 4.0],
        )
        monkeypatch.setattr("openslide.OpenSlide", lambda path: slide)
        load_wsi_patches(io.BytesIO(b"x"), 10, max_patches=16)
        assert slide.read_calls, "expected at least one patch read"
        assert all(level == 1 for _, level, _ in slide.read_calls)
        # Level grid coords are multiples of 896; level-0 locations are 4x those.
        assert all(
            lx % (896 * 4) == 0 and ly % (896 * 4) == 0
            for (lx, ly), _, _ in slide.read_calls
        )

    def test_no_tissue_raises(self, monkeypatch):
        slide = _make_slide(thumbnail=Image.new("RGB", (800, 800), (255, 255, 255)))
        monkeypatch.setattr("openslide.OpenSlide", lambda path: slide)
        with pytest.raises(ValueError, match="No tissue"):
            load_wsi_patches(io.BytesIO(b"x"), 40, max_patches=4)

    def test_too_small_raises(self, monkeypatch):
        slide = _make_slide(level_dimensions=[(500, 500)])
        monkeypatch.setattr("openslide.OpenSlide", lambda path: slide)
        with pytest.raises(ValueError, match="too small"):
            load_wsi_patches(io.BytesIO(b"x"), 40, max_patches=4)

    def test_unreadable_slide_raises(self, monkeypatch):
        def _boom(path):
            raise OSError("not a slide")

        monkeypatch.setattr("openslide.OpenSlide", _boom)
        with pytest.raises(ValueError, match="whole-slide image"):
            load_wsi_patches(io.BytesIO(b"x"), 40, max_patches=4)

    def test_objective_power_falls_back_to_mpp(self, monkeypatch):
        # No objective-power property; 0.5 um/px -> 20x base, so level 0 reports ~20x.
        slide = _make_slide(properties={"openslide.mpp-x": "0.5"})
        monkeypatch.setattr("openslide.OpenSlide", lambda path: slide)
        _, _, actual_mag = load_wsi_patches(io.BytesIO(b"x"), 20, max_patches=2)
        assert actual_mag == pytest.approx(20.0)

    def test_closes_slide_and_removes_tempfile(self, monkeypatch):
        slide = _make_slide()
        monkeypatch.setattr("openslide.OpenSlide", lambda path: slide)
        unlinked: list = []
        real_unlink = os.unlink
        monkeypatch.setattr(
            os, "unlink", lambda p: (unlinked.append(p), real_unlink(p))
        )
        load_wsi_patches(io.BytesIO(b"x"), 40, max_patches=2)
        assert slide.closed is True
        assert unlinked and not os.path.exists(unlinked[0])

    def test_removes_tempfile_when_upload_read_fails(self, monkeypatch):
        # A write failure between NamedTemporaryFile and the open must still unlink the
        # spilled temp file (it can be multi-GB for a real slide).
        unlinked: list = []
        real_unlink = os.unlink
        monkeypatch.setattr(
            os, "unlink", lambda p: (unlinked.append(p), real_unlink(p))
        )

        class _Boom(io.BytesIO):
            name = "slide.svs"

            def getvalue(self):
                raise OSError("disk full")

        with pytest.raises(OSError, match="disk full"):
            load_wsi_patches(_Boom(b""), 40, max_patches=2)
        assert unlinked and not os.path.exists(unlinked[0])


class TestReadPatch:
    def test_composites_transparent_region_onto_white(self):
        # Out-of-bounds RGBA (alpha 0) must read back as white, not black — a bare
        # .convert("RGB") would blacken it.
        class _Slide:
            def read_region(self, location, level, size):
                return Image.new("RGBA", size, (0, 0, 0, 0))  # fully transparent

        patch = _read_patch(_Slide(), 0, 0, 0, 1.0, 4)
        assert patch.mode == "RGB"
        assert patch.getpixel((0, 0)) == (255, 255, 255)

    def test_scales_location_by_downsample(self):
        # read_region takes a LEVEL-0 location (grid coord * downsample) at the level.
        calls: list = []

        class _Slide:
            def read_region(self, location, level, size):
                calls.append((location, level, size))
                return Image.new("RGBA", size, (10, 20, 30, 255))

        _read_patch(_Slide(), 100, 200, 2, 4.0, 896)
        assert calls == [((400, 800), 2, (896, 896))]  # (100*4, 200*4), level 2


class TestSlideObjectivePower:
    @staticmethod
    def _slide(props):
        return SimpleNamespace(properties=props)

    def test_uses_positive_objective_power(self):
        slide = self._slide({"openslide.objective-power": "20"})
        assert _slide_objective_power(slide) == 20.0

    def test_zero_objective_power_falls_back_to_mpp(self):
        # "0" is a non-positive value some scanners emit for a missing objective power;
        # it must not be trusted (mpp 0.5 -> 20x instead of collapsing pick_level).
        slide = self._slide(
            {"openslide.objective-power": "0", "openslide.mpp-x": "0.5"}
        )
        assert _slide_objective_power(slide) == pytest.approx(20.0)

    def test_malformed_objective_power_falls_back_to_mpp(self):
        slide = self._slide(
            {"openslide.objective-power": "unknown", "openslide.mpp-x": "0.25"}
        )
        assert _slide_objective_power(slide) == pytest.approx(40.0)

    def test_zero_mpp_falls_back_to_default(self):
        assert _slide_objective_power(self._slide({"openslide.mpp-x": "0"})) == 40.0

    def test_negative_mpp_falls_back_to_default(self):
        # mag_from_mpp(-0.5) = -20 -> non-positive -> 40x default.
        assert _slide_objective_power(self._slide({"openslide.mpp-x": "-0.5"})) == 40.0

    def test_no_properties_defaults_to_40(self):
        assert _slide_objective_power(self._slide({})) == 40.0


class TestMlxVlmContract:
    """Guard the mlx-vlm API surface the app depends on.

    Every other test mocks ``mlx_vlm.*`` (AppTest re-execs the script, and a real
    model load is far too heavy for a unit test), so those mocks pass no matter what
    the installed mlx-vlm actually exposes. These introspection checks are the only
    ones that fail when an upgrade drops or renames something ``run_model`` /
    ``load_model`` relies on — caught here instead of at inference time. Imports stay
    inside each test so a moved path fails only that check, not the whole suite.
    """

    def test_public_import_surface_is_callable(self):
        # The exact import paths streamlit_app uses (see its module header).
        from mlx_vlm import load, stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        from mlx_vlm.utils import load_config

        assert callable(stream_generate)
        assert callable(load)
        assert callable(apply_chat_template)
        assert callable(load_config)

    def test_load_returns_model_processor_pair(self):
        # load_model() unpacks `model, processor = load(MODEL_ID)` — a fixed 2-tuple.
        # Guard the arity via load()'s return annotation (no model load).
        from mlx_vlm import load

        ann = inspect.signature(load).return_annotation
        assert ann is not inspect.Signature.empty, "load() lost its return annotation"
        assert not isinstance(ann, str), (
            f"load() return annotation is stringized ({ann!r})"
        )
        assert typing.get_origin(ann) is tuple, (
            f"load() no longer returns a tuple ({ann!r})"
        )
        assert len(typing.get_args(ann)) == 2, (
            f"load() return arity changed; run_model unpacks exactly 2 ({ann!r})"
        )

    def test_stream_generate_accepts_run_model_kwargs(self):
        # run_model() streams via stream_generate(model, processor, prompt, image,
        # max_tokens=/temperature=/repetition_penalty=/repetition_context_size=). Those
        # sampling kwargs ride on **kwargs, so assert exactly that shape (run_model no
        # longer passes `verbose`). Whether the swallowed kwargs are honored is checked
        # by the docstring test below.
        from mlx_vlm import stream_generate

        params = inspect.signature(stream_generate).parameters
        assert any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()), (
            "stream_generate() dropped **kwargs that the sampling kwargs ride on"
        )

    def test_generate_documents_sampling_kwargs(self):
        # The sampling kwargs run_model() rides on stream_generate()'s **kwargs
        # (max_tokens, temperature, and the repetition-loop fix). generate() is just
        # `text += chunk.text` over stream_generate() and is where these kwargs are
        # publicly documented; a signature check can't prove a swallowed kwarg is
        # honored, so their presence in that docstring is the lightweight guard.
        from mlx_vlm import generate

        doc = (generate.__doc__ or "").lower()
        for kw in (
            "max_tokens",
            "temperature",
            "repetition_penalty",
            "repetition_context_size",
        ):
            assert kw in doc, f"generate() docstring no longer mentions {kw!r}"

    def test_generation_result_exposes_text(self):
        # run_model() streams `for chunk in stream_generate(...): yield chunk.text`.
        # The yielded chunks are GenerationResult (top-level importable), so guard that
        # it still exposes `.text` — the attribute the stream accumulation depends on.
        from mlx_vlm import GenerationResult

        if dataclasses.is_dataclass(GenerationResult):
            fields = {f.name for f in dataclasses.fields(GenerationResult)}
        else:
            fields = set(dir(GenerationResult))
        assert "text" in fields

    def test_generation_result_exposes_finish_reason(self):
        # run_model()'s _record_finish reads chunk.finish_reason to decide whether a
        # run stopped at the token cap, which is what drives warn_if_truncated. It
        # reads it via getattr(..., None) -- necessary, because the tests hand
        # run_model MagicMock chunks carrying only .text -- and that tolerance is
        # exactly why this guard has to exist: an upstream rename would make the
        # attribute silently absent, every truncated report would render as a
        # complete one, and no other test would notice (test_app_ui.py sets
        # finish_reason on its own mocks by hand). GenerationResult already carries
        # a dozen diffusion-only fields, so churn on this dataclass is live.
        from mlx_vlm import GenerationResult

        if dataclasses.is_dataclass(GenerationResult):
            fields = {f.name for f in dataclasses.fields(GenerationResult)}
        else:
            fields = set(dir(GenerationResult))
        assert "finish_reason" in fields, (
            "GenerationResult.finish_reason is gone; warn_if_truncated would "
            "silently stop firing on token-capped runs"
        )

    def test_stream_generate_still_reports_a_length_finish_reason(self):
        # The attribute existing is not enough: run_model compares it to the literal
        # "length". Assert that spelling still appears in the dispatch module that
        # yields these chunks, so a rename to e.g. "max_tokens" fails here rather
        # than by quietly never matching.
        from mlx_vlm.generate import dispatch

        source = inspect.getsource(dispatch)
        assert 'finish_reason = "length"' in source or 'finish_reason="length"' in (
            source
        ), "mlx-vlm no longer reports finish_reason == 'length' on a token-cap stop"

    def test_apply_chat_template_accepts_num_images(self):
        # run_model() calls apply_chat_template(..., num_images=).
        from mlx_vlm.prompt_utils import apply_chat_template

        params = inspect.signature(apply_chat_template).parameters
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        assert "num_images" in params or accepts_var_kw


def _wcag_contrast(fg: str, bg: str) -> float:
    """WCAG 2.x contrast ratio between two ``#rrggbb`` colors (1.0 to 21.0)."""

    def luminance(hex_color: str) -> float:
        channels = [int(hex_color.lstrip("#")[i : i + 2], 16) / 255 for i in (0, 2, 4)]
        r, g, b = (
            c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
            for c in channels
        )
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    hi, lo = sorted((luminance(fg), luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _blend(fg: str, bg: str, alpha: float) -> str:
    """``fg`` painted at ``alpha`` opacity over ``bg``, as a ``#rrggbb`` color."""
    pairs = zip(
        (int(fg.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)),
        (int(bg.lstrip("#")[i : i + 2], 16) for i in (0, 2, 4)),
        strict=True,
    )
    return "#" + "".join(f"{round(alpha * f + (1 - alpha) * b):02x}" for f, b in pairs)


class TestThemeConfig:
    """Guard .streamlit/config.toml. Like TestMlxVlmContract, this checks a real asset
    (not a mock): the file must parse, keep BOTH a custom light and dark mode, keep
    every color inside a mode section with the two modes setting the same keys, hold
    the contrast floors the palette was designed to (on the surfaces Streamlit really
    paints), use only theme keys the installed Streamlit recognizes, and keep usage
    telemetry off. The last one is not a theme setting but shares the file and the same
    invariant the theme keys are shaped around: an on-device app makes no outbound
    requests."""

    CONFIG = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"

    def _config(self) -> dict:
        with open(self.CONFIG, "rb") as f:
            return tomllib.load(f)

    def _theme(self) -> dict:
        return self._config()["theme"]

    def test_usage_stats_are_disabled(self):
        # browser.gatherUsageStats defaults to ON, and Streamlit's frontend then posts
        # to data.streamlit.io and a Fivetran webhook on every page load -- by a wide
        # margin the largest outbound flow this app would have, in a project whose
        # README promises nothing leaves the machine. Deleting the key is silent (the
        # default takes over), which is exactly why it is pinned.
        browser = self._config().get("browser", {})
        assert browser.get("gatherUsageStats") is False, (
            "browser.gatherUsageStats must be explicitly false; the default is on"
        )

    def test_config_exists_and_parses(self):
        assert self.CONFIG.is_file()
        assert isinstance(self._theme(), dict)  # raises if not valid TOML / no [theme]

    def test_upload_ceiling_admits_a_real_whole_slide_image(self):
        # Streamlit's default is 200 MB, which rejects real .svs/.ndpi slides (300 MB
        # - 2 GB) at the HTTP layer -- before any app code runs, with a generic
        # message -- in a tab whose loader and docs both reason about multi-GB slides.
        # server.maxUploadSize is the only lever: the limit is enforced server-side
        # against Content-Length, the streamed body and the parsed file, so a widget's
        # max_upload_size= can only narrow it, never raise it. Deleting the key is
        # silent (the 200 MB default takes back over), hence the pin.
        #
        # This lives in its own method rather than in the key walk below, which only
        # ever descends into [theme] and would never see a [server] key.
        server = self._config().get("server", {})
        assert server.get("maxUploadSize") == 2000, (
            "server.maxUploadSize must stay 2000 MB; the 200 MB default rejects "
            "real whole-slide images"
        )

    def test_non_slide_uploaders_narrow_the_raised_ceiling(self):
        # The ceiling above is global, so the three uploaders that are not the slide
        # one have to opt back down. The CT uploader is the one that matters: it is
        # accept_multiple_files=True, carries no type= filter by design, and reads
        # every slice fully into memory, so inheriting 2000 MB gives it by far the
        # widest blast radius of the four. Pinned as a pair -- raising the ceiling
        # without these narrowings is the actual regression.
        #
        # Asserted per widget rather than by occurrence count. A bare
        # `source.count(...) == 3` says nothing about WHICH uploaders carry the kwarg:
        # move it off the CT uploader and onto the slide one and the count is still
        # three, leaving the CT tab silently on 2000 MB and real slides rejected at
        # 200 -- exactly the pair of regressions this test exists to catch. A count is
        # also broken by any unrelated comment that names the constant.
        uploaders = _file_uploader_kwargs()
        assert set(uploaders) == {"cxr_image1", "cxr_image2", "ct_files", "wsi_files"}
        for key in ("cxr_image1", "cxr_image2", "ct_files"):
            assert uploaders[key].get("max_upload_size") == "NON_WSI_MAX_UPLOAD_MB", (
                f"{key} must narrow back to NON_WSI_MAX_UPLOAD_MB"
            )
        assert "max_upload_size" not in uploaders["wsi_files"], (
            "the slide uploader must keep the raised server ceiling; narrowing it "
            "re-rejects the multi-GB slides this config exists to admit"
        )
        source = (
            Path(__file__).resolve().parent.parent / "streamlit_app.py"
        ).read_text()
        assert "NON_WSI_MAX_UPLOAD_MB = 200" in source

    def test_expanders_never_carry_an_icon(self):
        # AppTest classifies an expandable block by whether it carries an icon --
        # `if block.expandable.icon: Status(...) else Expander(...)` in
        # testing/v1/element_tree.py -- because that is how st.status is built. So an
        # icon on a plain expander moves it out of at.expander and into at.status,
        # where .state raises ValueError("Unknown Status state") on any glyph that is
        # not one of st.status's own three.
        #
        # Adding two apt Material glyphs cost five simultaneous failures once already
        # (the "Thinking trace"/"Model settings" label lookups plus the CT/WSI
        # at.status[0].state error assertions), none of which name the cause. This
        # turns that five-test mystery into one message. type="compact" is unaffected
        # and stays.
        for call in _calls_to("st.expander"):
            kwargs = {k.arg for k in call.keywords if k.arg}
            assert "icon" not in kwargs, (
                f"st.expander at line {call.lineno} passes icon=, which reclassifies "
                "it as an st.status under AppTest and breaks the harness's "
                "status/expander discrimination"
            )

    MODES = ("light", "dark")
    # Streamlit's own derived paints (1.64 frontend): st.caption renders at
    # theme.opacities.secondary, and a selected segmented_control segment sits on
    # primaryColor at 10% alpha. The app has ~15 captions, and the WSI magnification
    # control is a segmented_control.
    CAPTION_OPACITY = 0.6
    SEGMENT_TINT = 0.1

    @staticmethod
    def _panel(mode: dict) -> str:
        # The sidebar surface as the frontend resolves it: a mode's sidebar
        # backgroundColor, else that mode's own secondaryBackgroundColor.
        return mode.get("sidebar", {}).get(
            "backgroundColor", mode["secondaryBackgroundColor"]
        )

    def test_offers_both_modes(self):
        # Defining either mode section makes Streamlit offer the System / Light / Dark
        # switch, and a mode whose section is missing is quietly built from the STOCK
        # palette -- the switch stays. So deleting one section fails nothing at
        # runtime; it just reverts that mode to Streamlit's defaults. Pin both.
        theme = self._theme()
        missing = [mode for mode in self.MODES if mode not in theme]
        assert not missing, (
            f"{missing} missing: silently falls back to Streamlit's stock palette"
        )

    def test_modes_define_the_same_palette(self):
        # A color key one mode sets and the other omits leaves the other on
        # Streamlit's stock value for it -- a stock blue st.info in one mode only, say.
        # So the two palettes must set the same keys, sidebars included; and the core
        # plus the load-bearing semantics (yellow: the disclaimer st.warning; red:
        # st.error; blue: the stale-result st.info and the localization :blue-badge)
        # must be among them. A missing sidebar backgroundColor falls back to the
        # mode's secondaryBackgroundColor, a valid color but not the panel that sits
        # one step off the canvas -- so each sidebar must set its own.
        theme = self._theme()
        light, dark = theme["light"], theme["dark"]
        flat = {k for k, v in light.items() if not isinstance(v, dict)}
        assert flat == {k for k, v in dark.items() if not isinstance(v, dict)}, (
            "light and dark set different keys"
        )
        assert set(light.get("sidebar", {})) == set(dark.get("sidebar", {})), (
            "light and dark sidebars set different keys"
        )
        core = {
            "primaryColor",
            "backgroundColor",
            "secondaryBackgroundColor",
            "textColor",
            "linkColor",
            "yellowColor",
            "redColor",
            "blueColor",
        }
        assert core <= flat, f"modes are missing core colors: {core - flat}"
        for mode in self.MODES:
            assert "backgroundColor" in theme[mode].get("sidebar", {}), (
                f"[theme.{mode}.sidebar] must set the app panel's surface"
            )

    def test_colors_live_only_in_mode_sections(self):
        # Keys in [theme] (and [theme.sidebar]) are shared: they apply to every mode
        # that doesn't override them. A color there is therefore one palette's value
        # silently leaking into the other -- a dark textColor over a light canvas the
        # moment the light section forgets to set its own. Shape and type settings are
        # mode-independent and belong there; colors never do.
        theme = self._theme()
        shared = {**theme, **theme.get("sidebar", {})}
        leaked = [k for k in shared if k.endswith(("Color", "Colors"))]
        assert not leaked, f"colors in the shared [theme] block: {leaked}"

    def test_palette_meets_contrast_floors(self):
        # The ratios the palette was designed to, re-derived from the file so a color
        # tweak that breaks one fails here rather than in someone's eyes -- measured on
        # what Streamlit actually paints, not just the raw keys. WCAG 2.x floors: 4.5:1
        # for text, 3:1 for non-text UI.
        theme = self._theme()
        for mode in self.MODES:
            m = theme[mode]
            # The panel paints [theme.<mode>.sidebar] over the mode's own keys, so a
            # sidebar-level textColor or linkColor is what the sidebar really shows.
            surfaces = {
                "canvas": (m, m["backgroundColor"]),
                "sidebar": ({**m, **m.get("sidebar", {})}, self._panel(m)),
            }
            for name, (palette, canvas) in surfaces.items():
                where = f"{mode} {name}"
                text, link = palette["textColor"], palette["linkColor"]
                assert _wcag_contrast(text, canvas) >= 7, where
                # Captions fade the text color to 60%; the light textColor is as dark
                # as it is precisely so this still clears 4.5:1.
                caption = _blend(text, canvas, self.CAPTION_OPACITY)
                assert _wcag_contrast(caption, canvas) >= 4.5, f"{where} caption"
                assert _wcag_contrast(link, canvas) >= 4.5, f"{where} link"
                # The active tab underline, the toggle, the slider track and thumb.
                assert _wcag_contrast(palette["primaryColor"], canvas) >= 3, (
                    f"{where} accent"
                )
            # Streamlit draws primary-button labels in WHITE directly on primaryColor
            # (hover only darkens it), so this is the Run button's label. The nord
            # theme this palette replaced failed it at 2.0:1.
            assert _wcag_contrast("#ffffff", m["primaryColor"]) >= 4.5, (
                f"{mode} Run label"
            )
        # The theme's one accepted trade-off, pinned where it's resolvable: the
        # selected tab label, slider value and selected segment label are TEXT in
        # primaryColor. In dark mode no primary dark enough for the white Run label
        # also reads at 4.5:1 there (the config documents the ~3.5-3.8:1 it gets); in
        # light mode the deep teal clears it in every at-rest role.
        light = theme["light"]
        primary, canvas = light["primaryColor"], light["backgroundColor"]
        assert _wcag_contrast(primary, canvas) >= 4.5, "light tab / slider label"
        segment = _blend(primary, canvas, self.SEGMENT_TINT)
        assert _wcag_contrast(primary, segment) >= 4.5, "light selected segment label"

    def test_links_are_underlined_on_every_surface(self):
        # The underline is the link cue: dark mode has no tint that clears both 4.5:1
        # on the surface and 3:1 against body text. linkUnderline layers like every
        # theme key -- [theme] < [theme.<mode>] for the main area,
        # then [theme.sidebar] < [theme.<mode>.sidebar] for the panel -- so pin the
        # EFFECTIVE value per surface: an override anywhere would silently drop it.
        theme = self._theme()
        for mode in self.MODES:
            main = {**theme, **theme[mode]}
            panel = {
                **main,
                **theme.get("sidebar", {}),
                **theme[mode].get("sidebar", {}),
            }
            assert main.get("linkUnderline") is True, f"{mode} links lost the underline"
            assert panel.get("linkUnderline") is True, (
                f"{mode} sidebar links (the Terms link) lost the underline"
            )

    def test_no_link_sits_inside_a_caption(self):
        # st.caption paints its whole element at 60% opacity, link included, which
        # fades any light-mode link tint short of near-black below 4.5:1 -- the Terms
        # of Use link sat at ~2.6:1 there. Links go in markdown instead (the Terms
        # line is small :gray[] markdown). Checked on the source, so a future caption
        # that grows a [label](url) fails here rather than in a contrast audit.
        for call in _calls_to("st.caption"):
            text = " ".join(ast.unparse(arg) for arg in call.args)
            assert not re.search(r"\]\(", text), (
                f"st.caption at line {call.lineno} carries a markdown link, which "
                "its 60% opacity fades below 4.5:1; use small :gray[] markdown"
            )

    def test_sidebar_gray_text_matches_a_caption(self):
        # The Terms line is :gray[] markdown standing in for a caption, so each
        # sidebar's grayTextColor is set to the exact color a caption paints there
        # (textColor at 60% over the panel): it reads as the captions above it, and
        # inherits their 4.5:1 floor. Derived rather than eyeballed, so retuning
        # textColor without it would fail here instead of drifting.
        theme = self._theme()
        for mode in self.MODES:
            m = theme[mode]
            panel = self._panel(m)
            text = {**m, **m.get("sidebar", {})}["textColor"]
            gray = m.get("sidebar", {}).get("grayTextColor")
            assert gray == _blend(text, panel, self.CAPTION_OPACITY), (
                f"[theme.{mode}.sidebar] grayTextColor {gray} is not the caption "
                f"color {_blend(text, panel, self.CAPTION_OPACITY)}"
            )
            assert _wcag_contrast(gray, panel) >= 4.5, mode

    def test_loads_no_external_assets(self):
        # Inference is fully on-device, so the theme must not be what puts a request
        # on the wire. The bundled theme templates' `font`/`codeFont` (nord's: Inter
        # and JetBrains Mono) load from fonts.googleapis.com on every page load; they
        # are left unset on purpose, and Streamlit's bundled defaults (Source Sans /
        # Source Code, plus the Material Symbols face) are served from its own static
        # bundle instead. A value-level URL check, so re-pasting a template wholesale
        # fails here.
        remote: list[str] = []

        def walk(section: dict, prefix: str) -> None:
            for key, value in section.items():
                path = f"{prefix}.{key}"
                if isinstance(value, dict):
                    walk(value, path)
                elif isinstance(value, str) and "//" in value:
                    remote.append(path)

        walk(self._theme(), "theme")
        assert not remote, f"theme keys fetch remote assets: {remote}"

    def test_only_uses_recognized_theme_keys(self):
        # Cross-check every key against the theme options the installed Streamlit
        # registers, so a typo'd or removed key fails here instead of degrading to a
        # silent startup warning (mirrors the mlx-vlm contract guard's intent). Each
        # leaf is checked at its FULL scoped path (e.g. "theme.sidebar.textColor"),
        # so a top-level-only key misplaced in a subsection is caught, and a valid
        # nested table like [theme.sidebar] is recursed into rather than misread as
        # an unknown leaf.
        from streamlit import config as st_config

        opts = set(st_config.get_config_options())
        unknown: list[str] = []

        def walk(section: dict, prefix: str) -> None:
            for key, value in section.items():
                path = f"{prefix}.{key}"
                if isinstance(value, dict):  # nested table, e.g. [theme.light.sidebar]
                    walk(value, path)
                elif path not in opts:
                    unknown.append(path)

        walk(self._theme(), "theme")
        assert not unknown, f"unrecognized theme keys: {unknown}"


class TestFaviconAsset:
    """Guard the browser-tab favicon. Like TestThemeConfig, this checks a real
    checked-in asset: Streamlit resolves a ``:material/...:`` page_icon to an SVG on
    fonts.gstatic.com and refetches it on EVERY page load, which would be the only
    outbound request an otherwise fully on-device app makes. Vendoring the glyph as a
    local PNG moves it to Streamlit's own /media/ endpoint -- so this pins both halves:
    the file is really there and really a PNG, and set_page_config still points at it
    rather than drifting back to the remote form (which would fail no other test)."""

    ROOT = Path(__file__).resolve().parent.parent

    def test_favicon_file_exists_and_is_a_valid_png(self):
        import streamlit_app

        path = streamlit_app.FAVICON_PATH
        assert path.is_file(), f"favicon missing at {path}"
        # Resolved from __file__, not the CWD: `streamlit run` from another directory
        # must still find it.
        assert path.is_absolute() and self.ROOT in path.parents
        with Image.open(path) as icon:
            icon.load()  # forces the decode, so a truncated commit fails here
            assert icon.format == "PNG"
            assert icon.width == icon.height, "favicon should be square"

    def test_favicon_is_visible(self):
        # A re-render that lands fully transparent (or a stray all-white glyph on the
        # light tab strip) is silently invisible in the browser. Require real opaque
        # pixels, and require them not to be near-black -- the stock Material glyph is
        # #000 and all but disappears against a dark tab strip, which is why the
        # vendored copy is recolored to a teal.
        import streamlit_app

        with Image.open(streamlit_app.FAVICON_PATH) as icon:
            pixels = np.asarray(icon.convert("RGBA"), dtype=np.int16)
        height, width = pixels.shape[:2]
        opaque = pixels[pixels[:, :, 3] > 200]
        assert len(opaque) > 0.02 * height * width, "favicon is ~empty"
        assert int(opaque[:, :3].sum(axis=1).max()) > 200, (
            "favicon glyph is too dark to read on a dark tab strip"
        )
        # The same file is the st.logo mark, drawn on the sidebar of whichever mode is
        # showing -- so the glyph must clear 3:1 (WCAG non-text) on the light AND dark
        # panel. The dark mode's light link tint fails that on light (~1.7:1) and the
        # light primary fails it on dark (~2.7:1); the dark primary passes but leans
        # light-side (4.1:1 on light, 3.5:1 on dark). The glyph sits at the lightness
        # where the two panels' contrasts meet, ~3.8:1 on each.
        glyph = "#{:02x}{:02x}{:02x}".format(
            *np.median(opaque[:, :3], axis=0).astype(int)
        )
        theme = TestThemeConfig()._theme()
        for mode in TestThemeConfig.MODES:
            panel = TestThemeConfig._panel(theme[mode])
            assert _wcag_contrast(glyph, panel) >= 3, (
                f"favicon {glyph} is illegible on the {mode} sidebar {panel}"
            )

    def test_page_icon_points_at_the_local_file(self):
        # The invariant that actually costs something if it regresses. A one-word edit
        # back to ":material/clinical_notes:" restores the per-page-load CDN fetch and
        # looks identical in the browser, so nothing else would catch it.
        source = (self.ROOT / "streamlit_app.py").read_text(encoding="utf-8")
        match = re.search(r"page_icon=(.+?),", source)
        assert match, "no page_icon= argument found in streamlit_app.py"
        assert match.group(1) == "str(FAVICON_PATH)", (
            f"page_icon is {match.group(1)!r}; a :material/...: value refetches the "
            "icon from fonts.gstatic.com on every page load"
        )


class TestHooksConfig:
    """Guard the .claude/settings.json Claude Code hooks. Like TestThemeConfig, this
    checks a real asset (not a mock): the file must parse as JSON, every hook must be a
    well-formed {type: "command", command} entry whose command is valid shell (checked
    against the real interpreter with `sh -n`, mirroring the theme guard's real-key
    check), and the three configured events must stay wired so a dropped or typo'd hook
    fails here instead of silently no-op'ing at runtime. The Edit|Write *matchers* are
    pinned too -- they are an exact tool-name list, so a lower-cased one disables every
    hook under it while leaving all the other checks green -- as is the
    `permissions.deny` block covering the read side of the same secrets.

    Beyond those structural checks, the second half of the class runs the hook commands
    behaviorally — piping mock tool-event JSON on stdin and driving them with a fake
    `uv` on PATH — and asserts their real EXIT CODES and side effects (block vs allow,
    fail closed, run vs skip). Substring checks alone pass through silent
    regressions (an inverted `exit 2`, a mangled `*.py` gate, a dropped secrets.toml
    arm); executing the command is what actually pins the behavior."""

    SETTINGS = Path(__file__).resolve().parent.parent / ".claude" / "settings.json"

    def _settings(self) -> dict:
        # Explicit encoding: the commands embed em-dashes, so the platform default would
        # raise UnicodeDecodeError under a non-UTF-8 locale and error out the class.
        with open(self.SETTINGS, encoding="utf-8") as f:
            return json.load(f)  # raises if not valid JSON

    def _hooks(self) -> dict:
        return self._settings()["hooks"]  # raises if there is no "hooks"

    def test_settings_exists_and_parses(self):
        assert self.SETTINGS.is_file()
        assert isinstance(self._hooks(), dict)

    def test_expected_events_are_wired(self):
        # Deleting an event silently drops that automation (lint/format on edit,
        # the secret guard, or run-tests-on-stop), so pin the three we configured.
        assert {"PreToolUse", "PostToolUse", "Stop"} <= set(self._hooks())

    def test_every_hook_is_a_well_formed_command(self):
        for groups in self._hooks().values():
            assert isinstance(groups, list) and groups
            for group in groups:
                entries = group["hooks"]
                assert isinstance(entries, list) and entries
                for hook in entries:
                    assert hook["type"] == "command"
                    assert isinstance(hook["command"], str) and hook["command"].strip()

    def test_every_command_is_valid_shell(self):
        # Syntax-check each command against a real shell without executing it, so a
        # broken hook (which Claude Code would silently no-op) fails the suite instead.
        for groups in self._hooks().values():
            for group in groups:
                for hook in group["hooks"]:
                    result = subprocess.run(
                        ["sh", "-n", "-c", hook["command"]],
                        capture_output=True,
                        text=True,
                    )
                    assert result.returncode == 0, (
                        f"invalid shell in hook: {hook['command']}\n{result.stderr}"
                    )

    def test_hook_matchers_target_the_edit_tools(self):
        # A matcher of plain letters and "|" is an EXACT tool-name list, so a
        # lower-cased or typo'd matcher ("edit|write") matches nothing and silently
        # disables the hook with no runtime error. Every other check in this class
        # stays green through that mutation -- the commands are still present and
        # still valid shell -- so the matcher itself is what has to be pinned.
        hooks = self._hooks()
        for event in ("PreToolUse", "PostToolUse"):
            for group in hooks[event]:
                assert set(group["matcher"].split("|")) == {"Edit", "Write"}, (
                    f"{event}: matcher {group['matcher']!r} matches no tool"
                )
        for group in hooks["Stop"]:
            # Stop carries no tool name; a matcher here would never match.
            assert "matcher" not in group

    def test_permissions_deny_protects_secret_reads(self):
        # The PreToolUse guard below covers the WRITE side (and only for Edit/Write).
        # Reads are denied declaratively, which also stops `cat .env` through Bash --
        # exfiltration being the worse direction for a token. Protection that moved into
        # settings.json must not quietly move back out.
        deny = set(self._settings()["permissions"]["deny"])
        assert {
            "Read(.env)",
            "Read(.env.local)",
            "Read(.streamlit/secrets.toml)",
        } <= deny
        # Deliberately enumerated, never a `.env.*` wildcard: a Read deny rule ALSO
        # blocks Edit and Write on the same path (Claude Code >= 2.1.228), and an allow
        # rule cannot rescue a denied path -- so a wildcard would silently make
        # .env.example uncreatable, turning the guard's carve-out arm below, and its
        # test pin, into dead configuration that still reports green.
        for rule in deny:
            pattern = rule[rule.index("(") + 1 : rule.rindex(")")]
            for template in (".env.example", ".env.sample", ".env.template"):
                assert not fnmatch.fnmatch(template, pattern), (
                    f"{rule} would also block writes to {template}"
                )

    # --- Behavioral checks: execute the commands and assert real exit codes ---------

    requires_jq = pytest.mark.skipif(
        shutil.which("jq") is None, reason="hook commands parse stdin with jq"
    )

    def _command(self, event: str, needle: str = "") -> str:
        # The lone command for single-hook events, or the one matching `needle`.
        cmds = [h["command"] for g in self._hooks()[event] for h in g["hooks"]]
        if not needle:
            return cmds[0]
        return next(c for c in cmds if needle in c)

    @staticmethod
    def _run(command: str, payload: dict | str, env: dict | None = None):
        # Run a hook as Claude Code does: the tool-event JSON arrives on stdin. A str
        # payload goes through raw, so a malformed event uses this plumbing too.
        return subprocess.run(
            ["/bin/sh", "-c", command],
            input=payload if isinstance(payload, str) else json.dumps(payload),
            text=True,
            capture_output=True,
            env=env,
        )

    @staticmethod
    def _shim(bindir: Path, name: str, body: str) -> None:
        exe = bindir / name
        exe.write_text("#!/bin/sh\n" + body)
        exe.chmod(0o755)

    def _hook_env(self, bindir: Path, root: Path) -> dict:
        # Real env + a fake tool dir on PATH + the project-root the hooks cd into.
        return {
            **os.environ,
            "PATH": f"{bindir}{os.pathsep}{os.environ['PATH']}",
            "CLAUDE_PROJECT_DIR": str(root),
        }

    @requires_jq
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("/proj/.env", 2),  # the HF-token file
            ("/proj/.env.local", 2),  # dotenv variant
            ("/proj/.ENV", 2),  # case-insensitive volume -> same file
            ("/proj/.streamlit/secrets.toml", 2),
            ("/proj/uv.lock", 2),
            ("/proj/.env.example", 0),  # template must stay editable
            ("/proj/streamlit_app.py", 0),
            ("/proj/README.md", 0),
            # Bare names: Edit/Write pass absolute paths today, but a guard whose arms
            # are all anchored "*/" falls OPEN on every relative one, and a suite that
            # only ever sends "/proj/..." can never see it.
            (".env", 2),
            ("uv.lock", 2),
            (".streamlit/secrets.toml", 2),
            (".env.example", 0),
            ("streamlit_app.py", 0),
        ],
    )
    def test_pretooluse_guard_blocks_protected_allows_normal(self, path, expected):
        # Execute the guard with a mock Edit payload; assert it blocks (2) / allows (0).
        r = self._run(self._command("PreToolUse"), {"tool_input": {"file_path": path}})
        assert r.returncode == expected, (
            f"{path}: exit {r.returncode} (stderr: {r.stderr})"
        )
        if expected == 2:
            assert "Blocked" in r.stderr

    def test_pretooluse_guard_fails_closed_without_jq(self, tmp_path):
        # If jq is absent from PATH the guard must BLOCK (exit 2), never fall open.
        empty = tmp_path / "empty"
        empty.mkdir()
        r = self._run(
            self._command("PreToolUse"),
            {"tool_input": {"file_path": "/proj/.env"}},
            env={**os.environ, "PATH": str(empty)},
        )
        assert r.returncode == 2

    @requires_jq
    @pytest.mark.parametrize("payload", ["not json", "{}", '{"tool_input": {}}'])
    def test_pretooluse_guard_fails_closed_on_bad_payload(self, payload):
        # An unparseable event, or one with no file_path, is the SAME "I don't know
        # what is being written" condition as a missing jq, and must reach the same
        # verdict. Left to `jq -r ... // empty` it exits 0 and the write sails through.
        r = self._run(self._command("PreToolUse"), payload)
        assert r.returncode == 2, f"{payload!r}: exit {r.returncode}"

    @requires_jq
    def test_stop_hook_skips_pytest_when_hook_active(self, tmp_path):
        # Loop guard: on a hook-continued stop, exit 0 WITHOUT re-running the suite.
        root = tmp_path / "proj"
        (root / ".claude").mkdir(parents=True)
        (root / ".claude" / ".tests-needed").touch()  # sentinel present...
        bindir = tmp_path / "bin"
        bindir.mkdir()
        self._shim(bindir, "uv", f'touch "{tmp_path}/uv-ran"\n')
        r = self._run(
            self._command("Stop"),
            {"stop_hook_active": True},
            env=self._hook_env(bindir, root),
        )
        assert r.returncode == 0
        assert not (tmp_path / "uv-ran").exists()  # ...but pytest was never invoked

    @requires_jq
    def test_stop_hook_skips_pytest_without_sentinel(self, tmp_path):
        # Change gate: no testable edit this turn -> no sentinel -> skip the ~21s suite.
        root = tmp_path / "proj"
        (root / ".claude").mkdir(parents=True)
        bindir = tmp_path / "bin"
        bindir.mkdir()
        self._shim(bindir, "uv", f'touch "{tmp_path}/uv-ran"\n')
        r = self._run(
            self._command("Stop"),
            {"stop_hook_active": False},
            env=self._hook_env(bindir, root),
        )
        assert r.returncode == 0
        assert not (tmp_path / "uv-ran").exists()

    @requires_jq
    def test_stop_hook_blocks_when_tests_fail(self, tmp_path):
        # Sentinel present + failing suite -> exit 2 with feedback; sentinel kept.
        root = tmp_path / "proj"
        (root / ".claude").mkdir(parents=True)
        sentinel = root / ".claude" / ".tests-needed"
        sentinel.touch()
        bindir = tmp_path / "bin"
        bindir.mkdir()
        self._shim(bindir, "uv", 'echo "1 failed" >&2\nexit 1\n')
        r = self._run(
            self._command("Stop"),
            {"stop_hook_active": False},
            env=self._hook_env(bindir, root),
        )
        assert r.returncode == 2
        assert "Tests are failing" in r.stderr
        assert sentinel.exists()  # kept so the next turn re-checks

    @requires_jq
    def test_stop_hook_passes_and_clears_sentinel(self, tmp_path):
        # Sentinel present + green suite -> exit 0 and the sentinel is cleared.
        root = tmp_path / "proj"
        (root / ".claude").mkdir(parents=True)
        sentinel = root / ".claude" / ".tests-needed"
        sentinel.touch()
        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "uv-args"
        self._shim(bindir, "uv", f'echo "$@" >> "{log}"\nexit 0\n')
        r = self._run(
            self._command("Stop"),
            {"stop_hook_active": False},
            env=self._hook_env(bindir, root),
        )
        assert r.returncode == 0
        assert not sentinel.exists()
        # The gate has to be the SUITE. Every Stop shim here ignores "$@", so without
        # this the command could be swapped for any other tool and stay green.
        assert "pytest" in log.read_text()

    @requires_jq
    @pytest.mark.parametrize(
        ("rel", "marks"),
        [
            ("a.py", True),
            ("pyproject.toml", True),
            # .md / .yml / LICENSE arm too. Nine guard classes -- TestClaudeMd,
            # TestDocsMatchSource, TestLicense, TestReadmeAssets, TestCiWorkflow and
            # the three workflow guards -- test exactly those files and have no other
            # local trigger, so "a docs-only turn is free" was never true here: 15 of
            # the first 122 commits touched a guarded file while arming nothing,
            # including the CLAUDE.md rewrite TestClaudeMd exists to catch.
            ("notes.md", True),
            ("LICENSE", True),
            (".github/workflows/ci.yml", True),
            # A relative path has to resolve against the project root, or the
            # `*/.claude/settings.json` arm never matches and a hooks change ships
            # with the suite unrun -- the same fall-open fixed in the guard.
            (".claude/settings.json", True),
            # TestFaviconAsset guards a binary asset, but Edit/Write cannot produce a
            # valid PNG, so arming here would only ever bill the suite for a write
            # that could not have happened.
            ("assets/favicon.png", False),
        ],
    )
    def test_sentinel_hook_marks_testable_edits(self, tmp_path, rel, marks):
        # The PostToolUse sentinel is what tells Stop a testable file changed this turn.
        root = tmp_path / "proj"
        (root / ".claude").mkdir(parents=True)
        r = self._run(
            self._command("PostToolUse", "tests-needed"),
            {
                "tool_input": {"file_path": str(root / rel)}
                if "/" not in rel
                else {"file_path": rel}
            },
            env={**os.environ, "CLAUDE_PROJECT_DIR": str(root)},
        )
        assert r.returncode == 0
        assert (root / ".claude" / ".tests-needed").exists() is marks

    @requires_jq
    def test_sentinel_arms_when_the_event_cannot_be_read(self, tmp_path):
        # This hook is the ONLY thing that arms the Stop gate, so "I don't know what
        # was written" has to mean ARM, not exit quietly: the old command returned 0
        # having written nothing, leaving the suite silently un-run with no signal
        # anywhere -- and the four Stop tests stay green through it, because each one
        # creates the sentinel by hand. It arms via a builtin redirect rather than
        # `touch`, so an empty PATH cannot stop it either.
        root = tmp_path / "proj"
        (root / ".claude").mkdir(parents=True)
        sentinel = root / ".claude" / ".tests-needed"
        empty = tmp_path / "empty"
        empty.mkdir()
        cmd = self._command("PostToolUse", "tests-needed")
        r = self._run(
            cmd,
            {"tool_input": {"file_path": "a.py"}},
            env={"PATH": str(empty), "CLAUDE_PROJECT_DIR": str(root)},
        )
        assert r.returncode == 0
        assert sentinel.exists(), "a missing jq must arm the gate, not skip it"
        sentinel.unlink()
        # jq present, event unparseable -- the same "I don't know" condition.
        r = self._run(cmd, "not json", env=self._hook_env(empty, root))
        assert r.returncode == 0
        assert sentinel.exists(), "an unreadable event must arm the gate"

    @requires_jq
    def test_post_edit_hooks_ignore_files_outside_the_project(self, tmp_path):
        # A scratch .py outside the repo is not ours: linting it with project rules
        # blocks on unrelated violations, and arming the sentinel bills the ~21s suite
        # at Stop for a file no test covers. Every PostToolUse hook must opt out.
        root = tmp_path / "proj"
        (root / ".claude").mkdir(parents=True)
        outside = tmp_path / "elsewhere" / "scratch.py"
        outside.parent.mkdir()
        outside.write_text("# " + "z" * 300 + "\n")  # unfixable E501
        bindir = tmp_path / "bin"
        bindir.mkdir()
        self._shim(bindir, "uv", f'touch "{tmp_path}/uv-ran"\nexit 1\n')
        env = self._hook_env(bindir, root)
        for needle in ("ruff", "tests-needed"):
            r = self._run(
                self._command("PostToolUse", needle),
                {"tool_input": {"file_path": str(outside)}},
                env=env,
            )
            assert r.returncode == 0, f"{needle}: exit {r.returncode}"
        assert not (tmp_path / "uv-ran").exists()  # no gate was invoked at all
        assert not (root / ".claude" / ".tests-needed").exists()
        # The RELATIVE form of the same escape, which the absolute cases above cannot
        # see: joining "../elsewhere/scratch.py" onto the project root yields a string
        # that still *starts* with that root, so the prefix check alone falls open.
        # Before the `*..*` arm, ruff rewrote this outside file in place -- with this
        # project's rules -- while the hook reported exit 0.
        for needle in ("ruff", "tests-needed"):
            r = self._run(
                self._command("PostToolUse", needle),
                {"tool_input": {"file_path": f"../elsewhere/{outside.name}"}},
                env=env,
            )
            assert r.returncode == 0, f"{needle} (relative): exit {r.returncode}"
        assert not (tmp_path / "uv-ran").exists()
        assert not (root / ".claude" / ".tests-needed").exists()
        assert outside.read_text().startswith("# zzz"), "outside file was rewritten"

    @requires_jq
    def test_ruff_hook_runs_on_py_and_skips_others(self, tmp_path):
        # .py edit -> ruff check + ruff format; non-.py edit -> uv is never invoked.
        root = tmp_path / "proj"
        root.mkdir()
        bindir = tmp_path / "bin"
        bindir.mkdir()
        log = tmp_path / "uv-args"
        self._shim(bindir, "uv", f'echo "$@" >> "{log}"\n')
        env = self._hook_env(bindir, root)
        ruff = self._command("PostToolUse", "ruff")
        assert (
            self._run(
                ruff, {"tool_input": {"file_path": str(root / "a.py")}}, env=env
            ).returncode
            == 0
        )
        calls = log.read_text()
        assert "ruff check" in calls and "ruff format" in calls
        log.unlink()
        assert (
            self._run(
                ruff, {"tool_input": {"file_path": str(root / "a.md")}}, env=env
            ).returncode
            == 0
        )
        assert not log.exists()  # uv not called for a non-.py file

    @requires_jq
    def test_ruff_hook_blocks_on_residual_violation(self, tmp_path):
        # `ruff check --fix` repairs most things but not all -- E501 on a long comment
        # is this repo's most frequent lint failure, and `ruff format` won't rescue it.
        # Fixing silently and discarding the exit code sends it straight to CI, so the
        # hook re-checks afterwards and blocks (exit 2) on whatever survived.
        root = tmp_path / "proj"
        root.mkdir()
        bindir = tmp_path / "bin"
        bindir.mkdir()
        # Fail only the bare re-check (`ruff check FILE`), never the fixing pass, so
        # this pins the re-check specifically rather than "the third uv call".
        self._shim(
            bindir,
            "uv",
            'case "$*" in\n'
            "  *'ruff check --fix'*) exit 0 ;;\n"
            "  *'ruff format'*) exit 0 ;;\n"
            "  *'ruff check'*) echo 'E501 line too long' >&2; exit 1 ;;\n"
            "esac\nexit 0\n",
        )
        r = self._run(
            self._command("PostToolUse", "ruff"),
            {"tool_input": {"file_path": str(root / "a.py")}},
            env=self._hook_env(bindir, root),
        )
        assert r.returncode == 2
        assert "E501" in r.stderr


WORKFLOW_DIR = Path(__file__).resolve().parent.parent / ".github" / "workflows"

# The literal concurrency group release.yml and tag-and-release.yml both declare.
# Groups are repository-wide, so sharing one is what makes the manual and the automatic
# publisher mutually exclusive -- a hand-pushed tag landing while CI's auto-release is
# mid-publish queues behind it instead of racing it.
PUBLISH_GROUP = "publish-release"


class _WorkflowGuard:
    """Shared plumbing for the three checked-in GitHub Actions guards below.

    Each subclass points ``WORKFLOW`` at one file under ``.github/workflows/`` and
    inherits the parsing helpers plus the three properties EVERY workflow in this repo
    must hold: no ``${{ }}`` expression reaches a shell ``run:`` step (the classic
    Actions command-injection sink), no job or step opts out of blocking, and no job
    widens the token the workflow declares. None of the three is specific to one file,
    and a third hand-copied set for the auto-release publisher is exactly how one copy
    drifts from the others.

    Deliberately not named ``Test*``, so pytest collects only the subclasses.
    """

    WORKFLOW: Path

    # GITHUB_TOKEN scope ordering, for the widening comparison below.
    _SCOPE_RANK = {"none": 0, "read": 1, "write": 2}

    def _doc(self) -> dict:
        # Imported inside the test (like the mlx-vlm contract guards) so a missing dep
        # or moved path fails only these checks, not collection of the whole suite.
        import yaml

        with open(self.WORKFLOW) as f:
            return yaml.safe_load(f)  # raises if not valid YAML

    def _on(self, doc: dict) -> dict:
        # PyYAML parses the bare `on:` key as the boolean True (YAML 1.1), so look it up
        # under both spellings rather than assuming a "on" string key.
        return doc.get("on", doc.get(True))

    def _jobs(self, doc: dict) -> dict:
        return doc["jobs"]

    def _steps(self, doc: dict) -> list[dict]:
        # `.get("steps", [])`, not `job["steps"]`: a job that delegates to a reusable
        # workflow via `uses:` has no steps at all, and a KeyError there would replace
        # every assertion message in the class with the same opaque traceback.
        return [
            step for job in self._jobs(doc).values() for step in job.get("steps", [])
        ]

    def _run_commands(self, doc: dict) -> str:
        return "\n".join(s["run"] for s in self._steps(doc) if "run" in s)

    def _step_with(self, doc: dict, action: str) -> dict:
        """The ``with:`` mapping of the first step whose ``uses:`` names ``action``."""
        for step in self._steps(doc):
            if str(step.get("uses", "")).startswith(action):
                return step.get("with") or {}
        raise AssertionError(f"{self.WORKFLOW.name} has no step using {action}")

    def test_workflow_exists_and_parses(self):
        assert self.WORKFLOW.is_file()
        assert isinstance(self._doc(), dict)

    def test_no_expression_flows_into_a_run_step(self):
        # Injection guard: a ${{ }} expression interpolated into a shell `run:` block is
        # the classic GitHub Actions command-injection sink. Expressions stay fine in
        # non-run contexts (concurrency.group, a job `if:`, an `env:` mapping, a `with:`
        # input) -- none may reach a run command, so every value arrives via env.
        offenders = [
            s["run"] for s in self._steps(self._doc()) if "${{" in s.get("run", "")
        ]
        assert not offenders, f"expression flows into run step(s): {offenders}"

    def test_no_gate_neutralizes_itself(self):
        # The other tests prove each gate is PRESENT; this proves none opts out of
        # blocking. A stray `continue-on-error: true` (silence a flake and forget) keeps
        # every substring assertion green while the workflow quietly goes advisory --
        # CI passing on a red suite, or a version-mismatch check that publishes anyway.
        # Conditional `if:` steps stay legitimate, so only the opt-out is banned.
        for name, job in self._jobs(self._doc()).items():
            assert job.get("continue-on-error") is not True, (
                f"job {name} is non-blocking"
            )
            for step in job.get("steps", []):
                assert step.get("continue-on-error") is not True, (
                    f"job {name}: step {step.get('name')!r} is non-blocking"
                )

    def test_no_job_widens_the_workflow_token(self):
        # The workflow-level `permissions:` block is only a DEFAULT: a job-level
        # `permissions:` REPLACES it for that job's GITHUB_TOKEN at runtime. Without
        # this, adding `permissions: {contents: write, id-token: write}` to a job keeps
        # every other assertion in the class green while the token the file advertises
        # quietly escalates -- verified by probe against the real files.
        #
        # NARROWING stays legal on purpose: `permissions: {}` on a job that needs
        # nothing is the hardening this suite should encourage, so compare scope RANKS
        # rather than demanding equality with the workflow default.
        doc = self._doc()
        default = doc.get("permissions") or {}
        for name, job in self._jobs(doc).items():
            granted = job.get("permissions")
            if granted is None:
                continue
            assert isinstance(granted, dict), (
                f"job {name} uses the shorthand {granted!r}; spell the scopes out so "
                "this comparison stays meaningful"
            )
            for scope, level in granted.items():
                have = self._SCOPE_RANK.get(str(level), 2)
                allowed = self._SCOPE_RANK.get(str(default.get(scope, "none")), 0)
                assert have <= allowed, (
                    f"job {name} widens {scope} to {level!r}; the workflow grants "
                    f"{default.get(scope, 'none')!r}"
                )

    def test_checkout_does_not_persist_the_token(self):
        # checkout defaults to writing GITHUB_TOKEN into .git/config. Nothing in this
        # repo pushes with git -- both publishers go through `gh` with GH_TOKEN from
        # env -- so the credential has no reason to outlive the step, and in CI the
        # steps after it execute ~100 third-party packages' build and import code.
        persisted = self._step_with(self._doc(), "actions/checkout").get(
            "persist-credentials"
        )
        assert persisted is False, (
            f"{self.WORKFLOW.name}: checkout leaves the token in .git/config"
        )


class TestCiWorkflow(_WorkflowGuard):
    """Guard the .github/workflows/ci.yml GitHub Actions workflow. Like TestThemeConfig
    and TestHooksConfig, this checks a real checked-in asset (not a mock): the file must
    parse as YAML, run every job on an Apple-Silicon (macOS/arm64) runner matching the
    MLX target, and drive the SAME four gates the local hooks enforce (ruff check, ruff
    format --check, ty check, pytest) via a locked `uv sync`. The injection, blocking
    and token-widening properties come from _WorkflowGuard.

    The reproducibility and blast-radius knobs are pinned here too -- the uv version,
    the cache key, `persist-credentials`, the timeout and the concurrency rule -- an
    audit found all of them removable with a green suite while CLAUDE.md advertised
    them as "Guarded by TestCiWorkflow"."""

    WORKFLOW = WORKFLOW_DIR / "ci.yml"

    def test_triggers_on_push_main_pr_and_dispatch(self):
        # Dropping any of these silently narrows when CI runs (no PR gating, no manual
        # re-run), so pin the three triggers and the main-branch push filter.
        on = self._on(self._doc())
        assert {"push", "pull_request", "workflow_dispatch"} <= set(on)
        assert "main" in on["push"]["branches"]

    def test_runs_on_apple_silicon_runner(self):
        # The app is Apple-Silicon + MLX; every job must run on a macOS (arm64)
        # runner so CI matches dev/prod and the real-mlx-vlm contract test exercises
        # the shipped backend. A stray ubuntu runner would test a different platform.
        for name, job in self._jobs(self._doc()).items():
            assert str(job.get("runs-on", "")).startswith("macos"), (
                f"job {name!r} runs on {job.get('runs-on')!r}, not a macOS runner"
            )

    def test_runs_the_same_four_gates_as_local_hooks(self):
        # CI must enforce exactly the gates the .claude hooks run locally, so a
        # contributor without the hooks can't merge a lint/format/type/test regression.
        cmds = self._run_commands(self._doc())
        assert "ruff check" in cmds
        assert "ruff format --check" in cmds  # verify formatting; never reformat in CI
        assert "ty check" in cmds
        assert "pytest" in cmds

    def test_dependency_install_is_locked(self):
        # `uv sync --locked` both installs deps and fails if uv.lock drifts from
        # pyproject.toml — the reproducible-install + lockfile-drift guard in one step.
        assert "uv sync --locked" in self._run_commands(self._doc())

    def test_uv_itself_is_version_pinned(self):
        # `uv sync --locked` errors unless uv.lock is exactly what the RUNNING uv would
        # produce, so an unpinned uv hands an upstream release the power to turn a green
        # PR red with no repo change -- and that failure is indistinguishable from a
        # genuinely stale lock. Pin the resolver, not just the resolution.
        pinned = str(self._step_with(self._doc(), "astral-sh/setup-uv").get("version"))
        assert re.fullmatch(r"\d+\.\d+\.\d+", pinned), (
            f"setup-uv version is {pinned!r}, not an exact X.Y.Z pin"
        )

    def test_cache_key_is_only_the_lockfile(self):
        # setup-uv's DEFAULT cache-dependency-glob also keys on `**/pyproject.toml`, so
        # reverting to it would miss the key -- and re-save -- on every ruff-rule or
        # [project.urls] edit. Don't overrate the win: setup-uv prunes wheels by default
        # and every dep here IS a wheel, so the restore is ~2 KB and uv re-downloads all
        # 86 packages anyway. `uv sync --locked` catches lock drift regardless of what
        # the cache holds, so narrowing here is purely a (small) speed knob.
        with_ = self._step_with(self._doc(), "astral-sh/setup-uv")
        assert with_.get("cache-dependency-glob") == "uv.lock", (
            f"cache key is {with_.get('cache-dependency-glob')!r}, not the lockfile"
        )
        # CLAUDE.md states enable-cache is deliberately absent because `auto` already
        # resolves to on for a GitHub-hosted runner. Pin the absence, or a future
        # `enable-cache: false` would turn caching off while the doc still claims it on.
        assert "enable-cache" not in with_, (
            "enable-cache is set explicitly; CLAUDE.md documents it as absent"
        )

    def test_token_is_least_privilege(self):
        # CI only reads code and runs tests — it never pushes, comments, or releases,
        # so GITHUB_TOKEN is pinned read-only; dropping it would let a compromised dep
        # escalate via a write-scoped token (same threat model as the injection guard).
        # test_no_job_widens_the_workflow_token covers the job-level override.
        assert self._doc().get("permissions") == {"contents": "read"}

    def test_every_job_is_time_capped(self):
        # Without an explicit cap a hung `uv sync` or a wedged AppTest burns the 6h
        # default on a 10x-multiplier macOS runner. tests/test_app_ui.py sizes
        # APP_RUN_TIMEOUT to report inside this bound, so the two are coupled.
        for name, job in self._jobs(self._doc()).items():
            assert isinstance(job.get("timeout-minutes"), int), (
                f"job {name} has no timeout-minutes"
            )

    def test_main_runs_are_never_cancelled(self):
        # Superseding a stale PR run is right; superseding a main run is not. A
        # cancelled run reports "cancelled", not "failure", so nothing surfaces: the
        # commit lands on main having never passed the gates, and tag-and-release.yml
        # -- which publishes only off a SUCCESSFUL CI run -- silently skips it too.
        concurrency = self._doc()["concurrency"]
        assert "github.ref" in concurrency["group"], (
            "concurrency group is not per-ref, so PRs would cancel each other"
        )
        cancel = concurrency["cancel-in-progress"]
        assert cancel is not True, (
            "cancel-in-progress: true cancels main runs too, leaving commits unverified"
        )
        assert "refs/heads/main" in str(cancel), (
            f"cancel-in-progress is {cancel!r}; it must exempt refs/heads/main"
        )


class TestReleaseWorkflow(_WorkflowGuard):
    """Guard .github/workflows/release.yml -- the MANUAL, tag-driven publisher, and the
    backstop for the automatic one guarded by TestAutoReleaseWorkflow. A checked-in
    asset, so the file must parse as YAML, fire ONLY on a pushed `v*` tag, hold
    `contents: write` (a release needs write -- but nothing more), verify the pushed tag
    matches the pyproject version before publishing (the tag==version check lives here,
    at release time, not as a pytest that would fail between a bump and its tag), and
    create the release with auto-generated notes. Since tag-and-release.yml may already
    have cut the same version, the publish is also pinned as idempotent, and both files
    are pinned to the one repository-wide concurrency group that keeps them exclusive.
    """

    WORKFLOW = WORKFLOW_DIR / "release.yml"

    def test_triggers_only_on_version_tags(self):
        # The release must fire on a pushed vX.Y.Z tag and NOT on a branch push (a
        # branch trigger would republish on every commit to main). Pin the filter to
        # EXACTLY ["v*"]: `any(startswith("v"))` would let an over-broad glob like
        # ["v*", "*"] through, cutting a release for arbitrary non-version tags.
        on = self._on(self._doc())
        assert on["push"].get("tags") == ["v*"], (
            "release.yml tag filter is not exactly ['v*']"
        )
        assert "branches" not in on["push"], (
            "release.yml fires on branch pushes — it would republish on every commit"
        )

    def test_token_can_write_releases_but_no_more(self):
        # A release must CREATE a GitHub Release, so contents:write is required here
        # (unlike CI, which is read-only) — but nothing broader (no packages, id-token,
        # etc.), so a compromised action can't escalate past publishing a release.
        assert self._doc().get("permissions") == {"contents": "write"}

    def test_verifies_tag_matches_pyproject_version(self):
        # The release-time drift guard: the workflow reads pyproject.toml, compares it
        # to the pushed tag, and FAILS the job on a mismatch — so a v0.7.6 tag pushed
        # while pyproject still says 0.7.5 aborts instead of publishing a release whose
        # number lies about the code. Assert on the compare's `v$version` construct,
        # not `$TAG`: `$TAG` also appears in the publish step, so asserting it would
        # pass even if the verify step dropped its tag-vs-version comparison entirely.
        cmds = self._run_commands(self._doc())
        assert "pyproject.toml" in cmds, (
            "release.yml never reads pyproject.toml to verify the tag"
        )
        assert "v$version" in cmds, (
            "release.yml never compares the tag against the pyproject version"
        )
        assert "exit 1" in cmds, (
            "release.yml never fails the job on a tag/version mismatch"
        )

    def test_creates_release_with_generated_notes(self):
        # The publish step itself: `gh release create` with auto-generated notes (from
        # the commit history — no hand-maintained CHANGELOG to drift) and --verify-tag,
        # which refuses to publish against a tag that isn't actually on the remote.
        cmds = self._run_commands(self._doc())
        assert "gh release create" in cmds
        assert "--generate-notes" in cmds
        assert "--verify-tag" in cmds

    def test_publish_is_idempotent(self):
        # tag-and-release.yml normally gets there first, so a hand-pushed tag for an
        # already-published version must be a quiet no-op, not a red job that looks
        # like a real failure.
        assert "gh release view" in self._run_commands(self._doc()), (
            "release.yml would fail instead of skipping an already-published version"
        )

    def test_shares_the_repository_wide_publish_group(self):
        assert self._doc()["concurrency"] == {
            "group": PUBLISH_GROUP,
            "cancel-in-progress": False,
        }

    def test_tag_reaches_shell_via_env(self):
        # The positive half of the injection guard: the tag is not merely kept OUT of
        # the run commands, it is routed IN through env (the safe channel). Check it
        # PER STEP: every $TAG-consuming step must set `TAG: ${{ github.ref_name }}`, so
        # dropping the env wiring from one step (leaving its $TAG unset) fails here
        # instead of passing because another step still routes the tag.
        consumers = [s for s in self._steps(self._doc()) if "$TAG" in s.get("run", "")]
        assert consumers, "no run step consumes $TAG — the tag is never used"
        for step in consumers:
            assert (step.get("env") or {}).get("TAG") == "${{ github.ref_name }}", (
                f"step {step.get('name')!r} uses $TAG but never wires it "
                "from github.ref_name via env"
            )


class TestAutoReleaseWorkflow(_WorkflowGuard):
    """Guard .github/workflows/tag-and-release.yml -- the AUTOMATIC publisher that cuts
    a release when a version bump reaches main.

    The design it pins is forced by one documented GitHub rule: events triggered by the
    default GITHUB_TOKEN do not create new workflow runs. A bot-pushed tag is therefore
    invisible to release.yml's `on: push: tags`, so "tag here, let release.yml publish"
    cannot work without a standing PAT -- which is why this workflow tags AND publishes
    in a single `gh release create --target <sha>` call. Every assertion below protects
    one leg of that: the CI-conclusion gate (nothing publishes off a red or lock-drifted
    commit), the fork filter, the tested-SHA checkout, the state-based idempotency key,
    strict X.Y.Z, published-not-drafted, and the shared concurrency group."""

    WORKFLOW = WORKFLOW_DIR / "tag-and-release.yml"

    # The one version-parsing expression that must be byte-identical in both publishers:
    # if this workflow read the version differently it could cut a tag that the manual
    # publisher's verify step then rejects -- a disagreement that surfaces only at
    # publish time.
    VERSION_PARSE = "grep -m1 '^version = ' pyproject.toml | cut -d'\"' -f2"

    def test_fires_only_on_a_completed_ci_run(self):
        # No `push:` trigger: it would race CI and publish in seconds while the suite
        # was still running, which is precisely the gate this workflow exists to keep.
        on = self._on(self._doc())
        assert set(on) == {"workflow_run", "workflow_dispatch"}, (
            f"unexpected triggers: {sorted(on)}"
        )
        assert on["workflow_run"]["types"] == ["completed"]
        # Without the branch filter every pull-request CI run also starts a (skipped)
        # run here, and a skipped run still occupies a slot in the shared publish group.
        assert on["workflow_run"]["branches"] == ["main"], (
            "workflow_run is not filtered to main, so PR CI runs spawn publish runs"
        )

    def test_gates_on_ci_by_name_not_path(self):
        # `workflows:` matches ci.yml's `name:` VALUE, not its path. Renaming CI would
        # silently un-gate this publisher (fail-safe, but invisible), and worse, a
        # different workflow later named "CI" would gate it instead. Read the name out
        # of ci.yml so a rename fails HERE rather than drifting into a dead trigger.
        import yaml

        with open(TestCiWorkflow.WORKFLOW) as f:
            ci_name = yaml.safe_load(f)["name"]
        assert self._on(self._doc())["workflow_run"]["workflows"] == [ci_name], (
            "tag-and-release.yml no longer gates on ci.yml's workflow name"
        )

    # Pinned as the WHOLE normalised expression rather than as a set of substring
    # checks. A substring guard is satisfied just as happily when the `&&`s are flipped
    # to `||` -- which would disable the CI gate and the fork filter at once -- and it
    # says nothing about the `workflow_dispatch` disjunct. Verified by mutation: with
    # substring checks, both tightening and DELETING the dispatch clause left the suite
    # green. The boolean composition is the property, so assert the composition.
    EXPECTED_IF = (
        "(github.event_name == 'workflow_dispatch' && "
        "github.ref == 'refs/heads/main') || "
        "(github.event.workflow_run.conclusion == 'success' && "
        "github.event.workflow_run.event == 'push' && "
        "github.event.workflow_run.head_branch == 'main' && "
        "github.event.workflow_run.head_repository.full_name == github.repository)"
    )

    def test_publishes_only_from_a_green_same_repo_push_to_main(self):
        # Every conjunct is load-bearing. `head_branch == 'main'` alone is a trap (a
        # fork's default branch is also called main), and the dispatch leg must stay
        # ANDed with a main check: as a bare disjunct it let anyone with write access
        # publish from any branch or tag, because on `workflow_dispatch` github.sha is
        # the tip of the DISPATCHED ref, which both fallbacks below resolve to.
        condition = " ".join(self._jobs(self._doc())["release"]["if"].split())
        assert condition == self.EXPECTED_IF, (
            f"job `if` changed.\n  is:       {condition}\n"
            f"  expected: {self.EXPECTED_IF}"
        )

    def test_checks_out_the_commit_ci_actually_tested(self):
        # On a workflow_run event, github.sha is the DEFAULT BRANCH tip -- not the
        # commit CI verified. Checking out github.sha would let a release describe code
        # that never passed the gates whenever main moved during the run.
        with_ = self._step_with(self._doc(), "actions/checkout")
        assert "workflow_run.head_sha" in str(with_.get("ref")), (
            f"checkout ref is {with_.get('ref')!r}, not the tested SHA"
        )

    def test_skips_a_version_that_is_already_released(self):
        # The idempotency key is repository STATE ("is there a release for the version
        # pyproject.toml claims?"), never a diff. A `git diff HEAD^ HEAD` detector
        # breaks under checkout's default fetch-depth: 1 (HEAD^ does not resolve), on
        # force pushes, on several commits in one push when the bump is not the last,
        # and on job re-runs; a state check survives all four, making a re-run a no-op.
        cmds = self._run_commands(self._doc())
        assert "gh release view" in cmds, (
            "no already-released check — a re-run would try to publish twice"
        )
        assert "exit 0" in cmds, "the already-released check never short-circuits"

    def test_tags_the_tested_sha_and_generates_notes(self):
        # --target <sha> makes `gh release create` mint the tag AND publish in one API
        # call, so there is no window where the tag exists but the release doesn't.
        # That is also why --verify-tag is absent here: it requires a pre-existing tag,
        # and pinning --target to the exact tested SHA is the stronger guarantee.
        cmds = self._run_commands(self._doc())
        assert "gh release create" in cmds
        assert '--target "$SHA"' in cmds, (
            "release is not pinned to the tested SHA via --target"
        )
        assert "--generate-notes" in cmds

    def test_publishes_rather_than_drafting(self):
        # A draft is absent from the public releases API, so README's shields.io release
        # badge would freeze on the previous version -- and a draft does not materialise
        # the git tag until a human clicks Publish.
        assert "--draft" not in self._run_commands(self._doc())

    def test_only_plain_x_y_z_versions_publish(self):
        # A '0.9.0.dev1' or '0.9.0rc1' left in pyproject during work in progress must
        # not become a public release; the workflow skips loudly instead of guessing.
        assert r"^[0-9]+\.[0-9]+\.[0-9]+$" in self._run_commands(self._doc()), (
            "tag-and-release.yml no longer restricts publishing to plain X.Y.Z versions"
        )

    def test_both_publishers_read_the_version_identically(self):
        # A divergence here would let this workflow cut a tag that release.yml's
        # tag==version step rejects -- a disagreement surfacing only at publish time.
        import yaml

        with open(TestReleaseWorkflow.WORKFLOW) as f:
            manual = yaml.safe_load(f)
        manual_cmds = "\n".join(
            s["run"]
            for job in manual["jobs"].values()
            for s in job.get("steps", [])
            if "run" in s
        )
        assert self.VERSION_PARSE in self._run_commands(self._doc())
        assert self.VERSION_PARSE in manual_cmds

    def test_token_can_write_releases_but_no_more(self):
        assert self._doc().get("permissions") == {"contents": "write"}

    def test_publish_group_is_claimed_per_job_not_per_run(self):
        # Same literal as release.yml -- groups are repository-wide, so this is what
        # stops a hand-pushed tag and the automatic path publishing at once. It must sit
        # on the JOB: a workflow-level group is claimed when the RUN starts, before the
        # `if:` is evaluated, so skipped no-op runs would queue in it -- and GitHub
        # cancels a previously *pending* run when a newer one queues, even under
        # `cancel-in-progress: false`, which could silently drop a queued manual
        # release.
        doc = self._doc()
        assert "concurrency" not in doc, (
            "concurrency is declared at workflow level; skipped runs would claim it"
        )
        assert doc["jobs"]["release"]["concurrency"] == {
            "group": PUBLISH_GROUP,
            "cancel-in-progress": False,
        }

    def test_dispatch_path_is_ci_gated_too(self):
        # The job `if:` cannot check a CI conclusion on the dispatch path (there is no
        # upstream run), so the invariant is re-imposed in a step that runs on BOTH
        # paths: the tested SHA must have a successful, push-triggered CI run. Without
        # it, dispatching while main is red -- the very situation the recovery lever
        # exists for -- would publish from a failing commit.
        cmds = self._run_commands(self._doc())
        assert "actions/workflows/ci.yml/runs" in cmds, (
            "nothing verifies the commit has a successful CI run"
        )
        assert "event=push" in cmds, (
            "the CI-run lookup counts pull_request runs, whose head_sha is a PR commit"
        )
        assert "exit 1" in cmds, "the CI-green check never fails the job"

    def test_refuses_a_tag_that_already_exists(self):
        # `--target` is documented as "Unused if the Git tag already exists", so a tag
        # left behind WITHOUT a release (a hand-pushed tag whose release.yml run failed)
        # would silently anchor the release at that tag's commit instead of the tested
        # SHA -- defeating the one guarantee --target is here to provide.
        cmds = self._run_commands(self._doc())
        assert "git/ref/tags/" in cmds, (
            "nothing checks for a pre-existing tag before publishing"
        )

    def test_every_job_is_time_capped(self):
        for name, job in self._jobs(self._doc()).items():
            assert isinstance(job.get("timeout-minutes"), int), (
                f"job {name} has no timeout-minutes"
            )


class TestWorkflowsAreGuarded:
    """Reverse guard over .github/workflows/ — the same shape as TestClaudeMd's reverse
    guard over tests/. Before it existed, `grep workflows tests/*.py` returned only
    hardcoded per-file paths, so a BRAND-NEW workflow — including one holding
    `contents: write` and interpolating an expression straight into a `run:` block —
    could land with zero coverage while the whole suite stayed green. Every workflow
    file must now be named by a guard class in this module."""

    def test_every_workflow_file_has_a_guard_class(self):
        # Resolved through the class objects, not by grepping this file for the
        # filename: a text match is satisfied by a filename mentioned in any passing
        # comment or docstring, which would report a genuinely unguarded workflow as
        # covered. Comparing the two sets also catches the reverse -- a guard class left
        # pointing at a workflow that has been renamed or deleted.
        guarded = {cls.WORKFLOW.name for cls in _WorkflowGuard.__subclasses__()}
        present = {path.name for path in WORKFLOW_DIR.glob("*.y*ml")}
        assert present, f"no workflows under {WORKFLOW_DIR} — has the path moved?"
        assert not present - guarded, (
            f"workflows with no _WorkflowGuard subclass: {sorted(present - guarded)}"
        )
        assert not guarded - present, (
            f"guard classes point at missing workflows: {sorted(guarded - present)}"
        )


class TestAppTestHarness:
    """Guard the AppTest harness in tests/test_app_ui.py against a silent revert.

    ``streamlit.testing.v1.AppTest`` bounds each script run with a 3s
    ``default_timeout``, which the first run in a freshly created venv blows (see the
    comment on ``APP_RUN_TIMEOUT``). tests/test_app_ui.py routes every construction
    through ``_app_test()`` so the whole suite inherits a safe bound -- but a new test
    reaching for the raw constructor gets the 3s default back, passes on a warm dev
    machine, and fails only in CI, where the venv is always cold. Nothing else in the
    suite would catch that, so pin it here.

    This lives beside the other checked-in-asset guards (TestCiWorkflow, TestClaudeMd,
    ...) rather than inside test_app_ui.py both to keep that file to UI flow and so the
    scan covers EVERY module under tests/ -- a second AppTest-based module would
    otherwise reintroduce the flake with the guard still green.
    """

    TESTS_DIR = Path(__file__).resolve().parent

    # Any AppTest constructor, not just from_file: from_string/from_function take the
    # same default_timeout and would reintroduce the same flake.
    CONSTRUCTOR = re.compile(r"AppTest\.from_(?:file|string|function)\s*\(")

    # test_app_ui.py gets exactly one: the call inside _app_test() itself.
    ALLOWED = {"test_app_ui.py": 1}

    def test_apptests_are_built_only_by_the_shared_helper(self):
        offenders = {}
        for path in sorted(self.TESTS_DIR.glob("*.py")):
            found = len(self.CONSTRUCTOR.findall(path.read_text(encoding="utf-8")))
            if found != self.ALLOWED.get(path.name, 0):
                offenders[path.name] = found
        assert not offenders, (
            f"AppTest constructed outside _app_test(): {offenders}; build every "
            "AppTest via tests.test_app_ui._app_test() so its timeout applies"
        )

    def test_helper_applies_a_timeout_clear_of_a_cold_start(self):
        # The helper is only worth pinning if it actually raises the bound: guard both
        # that the per-run budget beats AppTest's 3s default and that the one-time
        # warmup is generous enough for the ~16-24s cold first run.
        from tests.test_app_ui import APP_RUN_TIMEOUT, APP_WARMUP_TIMEOUT, _app_test

        assert APP_RUN_TIMEOUT > 3, "per-run budget no longer beats AppTest's default"
        assert APP_WARMUP_TIMEOUT >= 30, "warmup budget too tight for a cold first run"
        assert _app_test().default_timeout == APP_RUN_TIMEOUT
        assert (
            _app_test(timeout=APP_WARMUP_TIMEOUT).default_timeout == APP_WARMUP_TIMEOUT
        )


class TestClaudeMd:
    """Guard CLAUDE.md, the project context file loaded into every session. Like
    TestThemeConfig/TestHooksConfig/TestCiWorkflow, this checks a real checked-in asset
    (not a mock): the load-bearing files it maps must still exist AND stay named in the
    doc, every test-support module must stay documented, and every code symbol it cites
    from the app's spine must still resolve in streamlit_app. A rename/move/delete that
    leaves the doc stale — the exact currency drift a manual audit turned up
    (tests/dicom_helpers.py existed but the Tests section never mentioned it) — fails
    here instead of silently misleading the next session. (Facts CLAUDE.md shares with
    README.md — model id, WSI extensions, ruff rules — are cross-checked against the
    code by TestDocsMatchSource.)"""

    ROOT = Path(__file__).resolve().parent.parent
    CLAUDE_MD = ROOT / "CLAUDE.md"

    def _text(self) -> str:
        return self.CLAUDE_MD.read_text(encoding="utf-8")

    def test_claude_md_exists_and_is_nonempty(self):
        assert self.CLAUDE_MD.is_file()
        assert self._text().strip(), "CLAUDE.md is empty"

    def test_key_paths_exist_and_are_documented(self):
        # Two-way currency guard for the load-bearing files CLAUDE.md maps: each must
        # (a) still exist on disk, so a rename/move/delete leaving a stale reference
        # fails the exists half, and (b) actually be named in the doc, so dropping the
        # app file or a guarded config asset from the map fails the mention half.
        import re

        text = self._text()
        key_paths = [
            "streamlit_app.py",
            ".claude/settings.json",
            ".github/workflows/ci.yml",
            ".github/workflows/release.yml",
            ".github/workflows/tag-and-release.yml",
            ".streamlit/config.toml",
            "pyproject.toml",
            "uv.lock",
            "README.md",
            "LICENSE",
        ]
        for rel in key_paths:
            assert (self.ROOT / rel).is_file(), (
                f"documented path {rel} no longer exists"
            )
            # Require a STANDALONE reference: the (?<![\w/]) lookbehind keeps a path
            # from being "documented" only as the tail of a longer one — so `README.md`
            # can't be satisfied by `samples/README.md` alone (its one other mention).
            assert re.search(rf"(?<![\w/]){re.escape(rel)}", text), (
                f"{rel} is no longer documented in CLAUDE.md"
            )

    def test_documented_guard_classes_exist(self):
        # CLAUDE.md and README.md now cite guard classes by name, and nothing else
        # resolves them -- so renaming one would leave both docs pointing at a class
        # that is gone, with the suite green. Same drift this class exists to prevent,
        # one level up: test_documented_spine_symbols_exist covers app symbols, this
        # covers the guards. Curated, like the spine list.
        import sys

        module = sys.modules[__name__]
        text = self._text()
        for name in (
            "_WorkflowGuard",
            "TestFaviconAsset",
            "TestCiWorkflow",
            "TestReleaseWorkflow",
            "TestAutoReleaseWorkflow",
            "TestWorkflowsAreGuarded",
        ):
            assert hasattr(module, name), (
                f"CLAUDE.md cites {name}, which no longer exists in "
                f"{Path(__file__).name}"
            )
            assert name in text, f"{name} is no longer documented in CLAUDE.md"

    def test_every_test_module_is_documented(self):
        # Reverse guard over the tests/ dir (small, stable): every non-dunder .py must
        # be named in CLAUDE.md — the generic form of the drift the audit caught, where
        # a test-support module (tests/dicom_helpers.py) existed but was undocumented.
        # conftest.py is skipped: pytest plumbing, not content the doc must inventory.
        text = self._text()
        modules = sorted(
            p.name
            for p in (self.ROOT / "tests").glob("*.py")
            if not p.name.startswith("__") and p.name != "conftest.py"
        )
        undocumented = [m for m in modules if m not in text]
        assert not undocumented, (
            f"tests/ modules missing from CLAUDE.md: {undocumented}"
        )

    def test_documented_spine_symbols_exist(self):
        # Every code symbol CLAUDE.md cites from the app's spine must still resolve in
        # streamlit_app, so a rename not mirrored into the doc fails here. Curated (not
        # scraped from prose) to stay robust: the model/inference path, the four tab
        # renderers, the persistence helper, and the fixed-config constants. This is
        # a deliberate manual sample, not exhaustive — extend it when CLAUDE.md leans
        # on a new load-bearing symbol worth protecting from a silent rename.
        import streamlit_app

        text = self._text()
        spine = [
            "load_model",
            "build_messages",
            "get_generation_params",
            "run_model",
            "load_ct_volume",
            "window_ct_slice",
            "load_wsi_patches",
            "ram_aware_slice_cap",
            "parse_response",
            "parse_boxes",
            "fresh_result_or_hint",
            "tab_settings",
            "render_sidebar",
            "workspace_columns",
            "render_ask_tab",
            "render_cxr_tab",
            "render_ct_tab",
            "render_wsi_tab",
            "CT_WINDOWS",
            "LOCALIZATION_INSTRUCTION",
            "REPETITION_PENALTY",
            "DISCLAIMER_TEXT",
        ]
        for name in spine:
            assert name in text, f"spine symbol {name} dropped from CLAUDE.md"
            assert hasattr(streamlit_app, name), (
                f"CLAUDE.md documents {name}, but it no longer exists in streamlit_app"
            )

    def test_documented_test_harness_symbols_exist(self):
        # test_documented_spine_symbols_exist resolves names against streamlit_app, so
        # it cannot see the harness symbols the Gotchas section leans on. Without this,
        # renaming _app_test or APP_RUN_TIMEOUT leaves CLAUDE.md describing a harness
        # that no longer exists while the whole suite stays green -- exactly the silent
        # drift this class exists to prevent, just on the test side of the fence.
        import tests.test_app_ui as ui

        text = self._text()
        for name in (
            "_app_test",
            "_warm_streamlit_once",
            "APP_RUN_TIMEOUT",
            "APP_WARMUP_TIMEOUT",
        ):
            assert name in text, f"harness symbol {name} dropped from CLAUDE.md"
            assert hasattr(ui, name), (
                f"CLAUDE.md documents {name}, but it no longer exists in test_app_ui"
            )
        # The guard class lives in this module.
        assert "TestAppTestHarness" in text
        assert "TestAppTestHarness" in globals()


class TestDocsMatchSource:
    """Guard the prose docs (CLAUDE.md AND README.md) against the code for the facts
    with a single source of truth: the model id, the accepted WSI extensions, and the
    ruff rule set. Each is checked in every doc, so a code change not mirrored into the
    prose (or a doc that drifts from the code) fails here. This is the generalized fix
    for a real gap — README listed `.tiff` but not the `.tif` that WSI_TYPES accepts,
    and nothing caught it — so both docs stay pinned to the source, not each other."""

    ROOT = Path(__file__).resolve().parent.parent

    def _docs(self) -> list[tuple[str, str]]:
        return [
            (name, (self.ROOT / name).read_text(encoding="utf-8"))
            for name in ("CLAUDE.md", "README.md")
        ]

    def test_model_id_matches_source(self):
        # The model repo id lives in streamlit_app.MODEL_ID; both docs cite it verbatim,
        # so a model swap not mirrored into the prose fails here. MODEL_ID is a long,
        # unique string, so a plain membership check is unambiguous.
        import streamlit_app

        for name, doc in self._docs():
            assert streamlit_app.MODEL_ID in doc, (
                f"{name} does not document MODEL_ID {streamlit_app.MODEL_ID!r}"
            )

    def test_slice_cap_tiers_match_source(self):
        # ram_aware_slice_cap is the single source of truth for the CT/WSI tiers, and
        # both docs quote them in their own formats. Deriving the expected strings
        # here means a retune that forgets either doc fails, the same way a model swap
        # does above -- this closes a gap where the same numbers were hand-maintained
        # in CLAUDE.md, the docstring, and five test comments with nothing pinning them.
        default_32, max_32 = ram_aware_slice_cap(total_ram_gib=32)
        _, max_24 = ram_aware_slice_cap(total_ram_gib=24)
        floor = ram_aware_slice_cap(total_ram_gib=2)[1]
        # First whole GiB tier that clears the 2-slice floor.
        boundary = next(
            g for g in range(2, 129) if ram_aware_slice_cap(total_ram_gib=g)[1] > floor
        )
        docs = dict(self._docs())
        expected = {
            "CLAUDE.md": [f"`({default_32},{max_32})` on 32 GiB"],
            "README.md": [
                f"{floor} below {boundary} GB",
                f"{max_24} at 24 GB",
                f"{max_32} at 32 GB",
            ],
        }
        for name, fragments in expected.items():
            for fragment in fragments:
                assert fragment in docs[name], (
                    f"{name} does not document the slice-cap tier {fragment!r}"
                )

    def test_wsi_extensions_match_source(self):
        # Every accepted WSI extension (streamlit_app.WSI_TYPES) must appear in both
        # docs, matched as a DELIMITED token — `\.ext` not followed by an alphanumeric —
        # because `.tif` is a prefix of `.tiff`. A bare substring check is blind to the
        # gap that shipped once: a doc "contains" `.tif` via `.tiff` while omitting it.
        import re

        import streamlit_app

        for name, doc in self._docs():
            missing = [
                ext
                for ext in streamlit_app.WSI_TYPES
                if not re.search(rf"\.{re.escape(ext)}(?![A-Za-z0-9])", doc)
            ]
            assert not missing, f"{name} does not document WSI extensions {missing}"

    def test_ruff_rule_set_matches_pyproject(self):
        # pyproject's [tool.ruff.lint].select is the source of truth; both docs list it.
        # Codes are matched backtick-wrapped (`E`, `UP`, …) because bare letters like
        # E/F/I/B occur constantly in prose — an unbounded check would pass on any
        # sentence. A rule in pyproject but not the docs (or vice versa) fails here.
        with open(self.ROOT / "pyproject.toml", "rb") as f:
            select = tomllib.load(f)["tool"]["ruff"]["lint"]["select"]
        for name, doc in self._docs():
            missing = [code for code in select if f"`{code}`" not in doc]
            assert not missing, f"{name} does not document ruff rules {missing}"

    def test_license_matches_source(self):
        # pyproject's [project].license is the SPDX source of truth; both docs cite that
        # exact identifier, so a license swap not mirrored into the prose fails here.
        # "Apache-2.0" is distinctive (no .tif/.tiff prefix-collision risk), so a plain
        # membership check is unambiguous.
        with open(self.ROOT / "pyproject.toml", "rb") as f:
            license_id = tomllib.load(f)["project"]["license"]
        for name, doc in self._docs():
            assert license_id in doc, (
                f"{name} does not document the license {license_id!r}"
            )


class TestLicense:
    """Guard the licensing + medical-use assets added with the LICENSE file — a
    checked-in asset like the theme/hooks/CI config. The LICENSE text, the pyproject
    SPDX declaration, and the README's License + Disclaimer sections must stay mutually
    consistent, so swapping one (pyproject flipped to MIT with the Apache text left in
    place, LICENSE deleted out from under `license-files`, or the not-a-medical-device
    notice dropped) fails here instead of shipping a contradiction. The license
    id is also cross-checked against both prose docs by TestDocsMatchSource."""

    ROOT = Path(__file__).resolve().parent.parent

    def _pyproject(self) -> dict[str, typing.Any]:
        with open(self.ROOT / "pyproject.toml", "rb") as f:
            return tomllib.load(f)

    def test_license_file_is_apache_2_0(self):
        # The LICENSE is the source form of the grant; assert it is the Apache-2.0 text
        # and carries the FILLED copyright line, not the bracketed placeholder. A bare
        # "Copyright" check is vacuous: the word is in the Apache body itself.
        text = (self.ROOT / "LICENSE").read_text(encoding="utf-8")
        assert "Apache License" in text
        assert "Version 2.0" in text
        assert "Copyright 2026 Daryl Lim" in text, (
            "LICENSE copyright line is unfilled or the holder/year changed"
        )

    def test_pyproject_declares_apache_2_0(self):
        # The packaging metadata must declare the same license GitHub detects from the
        # file, so the About panel and pyproject never disagree.
        assert self._pyproject()["project"]["license"] == "Apache-2.0"

    def test_pyproject_license_files_resolve(self):
        # Every license-files glob must match a real file, so renaming/deleting LICENSE
        # while leaving the declaration dangling fails here.
        license_files = self._pyproject()["project"]["license-files"]
        assert license_files, "pyproject declares no license-files"
        for pattern in license_files:
            assert list(self.ROOT.glob(pattern)), (
                f"license-files pattern {pattern!r} matches no file"
            )

    def test_readme_documents_license(self):
        # The README must carry a License section that links the LICENSE file, so the
        # code license stays discoverable to a reader (not only to GitHub's scanner).
        readme = (self.ROOT / "README.md").read_text(encoding="utf-8")
        assert "## License" in readme
        # Scope the link check to the section: the top-of-file badge also ends in
        # `](LICENSE)`, so a whole-file check would pass even if the section dropped
        # its link. The `## License` assert above guarantees the split yields 2 parts.
        section = readme.split("## License", 1)[1]
        assert "(LICENSE)" in section, "README License section does not link LICENSE"

    def test_readme_has_medical_use_disclaimer(self):
        # Safety-critical for a medical tool: guard the scope disclaimer from silent
        # removal. The not-a-medical-device / not-medical-advice notice and the pointer
        # to the model's separate HAI-DEF terms must all survive a README edit.
        readme = (self.ROOT / "README.md").read_text(encoding="utf-8")
        assert "## Disclaimer" in readme
        low = readme.lower()
        assert "not a medical device" in low
        assert "not medical advice" in low
        assert "Health AI Developer Foundations" in readme, (
            "README no longer points to the model's HAI-DEF terms"
        )


class TestReadmeAssets:
    """Guard the README's in-repo assets and links — the hero screenshot
    (docs/screenshot.webp) and the relative links it carries (the sample-data guide, the
    LICENSE, the CI/release workflows). Like the other checked-in-asset guards, a
    moved/renamed/deleted target must fail here rather than ship as a broken image or a
    dead link in the rendered README. External (http/mailto) links and in-page anchors
    are out of scope; only repo-relative paths are resolved."""

    ROOT = Path(__file__).resolve().parent.parent

    def _readme(self) -> str:
        return (self.ROOT / "README.md").read_text(encoding="utf-8")

    def _relative_targets(self) -> list[str]:
        # Pull the target from every Markdown `](target)` — plain links, images,
        # and both halves of a nested badge `[![alt](img)](target)`, so a badge
        # pointing at a repo file is validated too, not just its inner image.
        # Keep only repo-relative paths: drop http(s)/mailto and strip any
        # `"title"` / `#fragment`. Deduplicated — one entry per target.
        import re

        targets = []
        for m in re.finditer(r"\]\(([^)]+)\)", self._readme()):
            raw = m.group(1).strip()
            if not raw:
                continue
            target = raw.split()[0].split("#", 1)[0]
            if target and not target.startswith(("http://", "https://", "mailto:")):
                targets.append(target)
        return sorted(set(targets))

    def test_readme_relative_links_resolve(self):
        # Every repo-relative link/image target must exist on disk: the hero image,
        # samples/README.md, LICENSE, and the workflow files the Development section
        # links all fail here if moved, renamed, or deleted.
        missing = [t for t in self._relative_targets() if not (self.ROOT / t).exists()]
        assert not missing, f"README links to missing repo files: {missing}"

    def test_readme_embeds_hero_screenshot(self):
        # Two-way guard for the hero (mirrors TestClaudeMd's exists-AND-documented):
        # the README must embed the docs/screenshot.webp hero AND that file must
        # exist, so it can't be silently dropped from the README or deleted from the
        # repo. Pin the exact path (not just "some docs/ image") to match the framing.
        import re

        heroes = re.findall(r"!\[[^\]]*\]\((docs/[^)\s]+)\)", self._readme())
        assert "docs/screenshot.webp" in heroes, (
            "README no longer embeds the docs/screenshot.webp hero"
        )
        for rel in heroes:
            assert (self.ROOT / rel).is_file(), f"hero image {rel} is missing"
