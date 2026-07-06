import dataclasses
import inspect
import io
import json
import os
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
    LOCALIZATION_INSTRUCTION,
    _detect_total_ram_gib,
    _read_patch,
    _slide_objective_power,
    build_messages,
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
    def test_32gib_yields_8_and_16(self):
        assert ram_aware_slice_cap(total_ram_gib=32) == (8, 16)

    def test_scales_up_with_more_ram(self):
        default, maximum = ram_aware_slice_cap(total_ram_gib=64)
        assert maximum > 16
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

    def test_apply_chat_template_accepts_num_images(self):
        # run_model() calls apply_chat_template(..., num_images=).
        from mlx_vlm.prompt_utils import apply_chat_template

        params = inspect.signature(apply_chat_template).parameters
        accepts_var_kw = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
        )
        assert "num_images" in params or accepts_var_kw


class TestThemeConfig:
    """Guard the .streamlit/config.toml clinical theme. Like TestMlxVlmContract, this
    checks a real asset (not a mock): the file must parse, define BOTH [theme.light]
    and [theme.dark] (Streamlit only offers the light/dark auto-switch when both
    exist), and use only theme keys the installed Streamlit recognizes."""

    CONFIG = Path(__file__).resolve().parent.parent / ".streamlit" / "config.toml"

    def _theme(self) -> dict:
        with open(self.CONFIG, "rb") as f:
            return tomllib.load(f)["theme"]

    def test_config_exists_and_parses(self):
        assert self.CONFIG.is_file()
        assert isinstance(self._theme(), dict)  # raises if not valid TOML / no [theme]

    def test_defines_both_light_and_dark_modes(self):
        # Both subsections are required for the OS/browser auto-switch; dropping
        # either silently locks the app to a single mode.
        theme = self._theme()
        assert "light" in theme and "dark" in theme

    def test_both_modes_define_the_core_palette(self):
        theme = self._theme()
        core = {
            "primaryColor",
            "backgroundColor",
            "secondaryBackgroundColor",
            "textColor",
        }
        for mode in ("light", "dark"):
            assert core <= set(theme[mode]), f"[theme.{mode}] is missing core colors"

    def test_only_uses_recognized_theme_keys(self):
        # Cross-check every key against the theme options the installed Streamlit
        # registers, so a typo'd or removed key fails here instead of degrading to a
        # silent startup warning (mirrors the mlx-vlm contract guard's intent). Each
        # leaf is checked at its FULL scoped path (e.g. "theme.light.primaryColor"),
        # so a top-level-only key misplaced in [theme.light] is caught, and a valid
        # nested table like [theme.light.sidebar] is recursed into rather than
        # misread as an unknown leaf.
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


class TestHooksConfig:
    """Guard the .claude/settings.json Claude Code hooks. Like TestThemeConfig, this
    checks a real asset (not a mock): the file must parse as JSON, every hook must be a
    well-formed {type: "command", command} entry whose command is valid shell (checked
    against the real interpreter with `sh -n`, mirroring the theme guard's real-key
    check), and the three configured events must stay wired so a dropped or typo'd hook
    fails here instead of silently no-op'ing at runtime.

    Beyond those structural checks, the second half of the class runs the hook commands
    behaviorally — piping mock tool-event JSON on stdin and driving them with a fake
    `uv` on PATH — and asserts their real EXIT CODES and side effects (block vs allow,
    fail closed, run vs skip). Substring checks alone pass through silent
    regressions (an inverted `exit 2`, a mangled `*.py` gate, a dropped secrets.toml
    arm); executing the command is what actually pins the behavior."""

    SETTINGS = Path(__file__).resolve().parent.parent / ".claude" / "settings.json"

    def _hooks(self) -> dict:
        with open(self.SETTINGS) as f:
            return json.load(f)["hooks"]  # raises if not valid JSON / no "hooks"

    def _commands_for(self, event: str) -> str:
        # Join every command string configured under one event, for intent checks.
        return " ".join(h["command"] for g in self._hooks()[event] for h in g["hooks"])

    def test_settings_exists_and_parses(self):
        assert self.SETTINGS.is_file()
        assert isinstance(self._hooks(), dict)

    def test_expected_events_are_wired(self):
        # Deleting an event silently drops that automation (format/type-check on edit,
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

    def test_secret_guard_covers_env_and_lockfile(self):
        # The PreToolUse guard must keep protecting the HF token, Streamlit secrets, and
        # the lockfile. (Behaviorally re-verified below; kept as a cheap intent pin.)
        guard = self._commands_for("PreToolUse")
        assert ".env" in guard
        assert "secrets.toml" in guard
        assert "uv.lock" in guard

    def test_edit_hooks_run_formatter_and_type_checker(self):
        # PostToolUse on an edit must keep invoking both ruff and ty.
        post = self._commands_for("PostToolUse")
        assert "ruff" in post
        assert "ty check" in post

    def test_stop_hook_runs_tests_with_loop_guard(self):
        # The Stop hook must run pytest AND guard the infinite stop->fix loop.
        stop = self._commands_for("Stop")
        assert "pytest" in stop
        assert "stop_hook_active" in stop

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
    def _run(command: str, payload: dict, env: dict | None = None):
        # Run a hook as Claude Code does: the tool-event JSON arrives on stdin.
        return subprocess.run(
            ["/bin/sh", "-c", command],
            input=json.dumps(payload),
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
        ("rel", "expected"),
        [
            (".env", 2),  # the HF-token file
            (".env.local", 2),  # dotenv variant
            (".ENV", 2),  # case-insensitive volume -> same file
            (".streamlit/secrets.toml", 2),
            ("uv.lock", 2),
            (".env.example", 0),  # template must stay editable
            ("streamlit_app.py", 0),
            ("README.md", 0),
        ],
    )
    def test_pretooluse_guard_blocks_protected_allows_normal(self, rel, expected):
        # Execute the guard with a mock Edit payload; assert it blocks (2) / allows (0).
        r = self._run(
            self._command("PreToolUse"), {"tool_input": {"file_path": f"/proj/{rel}"}}
        )
        assert r.returncode == expected, (
            f"{rel}: exit {r.returncode} (stderr: {r.stderr})"
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
        # Change gate: no testable edit this turn -> no sentinel -> skip the ~11s suite.
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
        self._shim(bindir, "uv", "exit 0\n")
        r = self._run(
            self._command("Stop"),
            {"stop_hook_active": False},
            env=self._hook_env(bindir, root),
        )
        assert r.returncode == 0
        assert not sentinel.exists()

    @requires_jq
    @pytest.mark.parametrize(
        ("rel", "marks"),
        [("a.py", True), ("pyproject.toml", True), ("notes.md", False)],
    )
    def test_sentinel_hook_marks_testable_edits(self, tmp_path, rel, marks):
        # The PostToolUse sentinel is what tells Stop a testable file changed this turn.
        root = tmp_path / "proj"
        (root / ".claude").mkdir(parents=True)
        r = self._run(
            self._command("PostToolUse", "tests-needed"),
            {"tool_input": {"file_path": f"/x/{rel}"}},
            env={**os.environ, "CLAUDE_PROJECT_DIR": str(root)},
        )
        assert r.returncode == 0
        assert (root / ".claude" / ".tests-needed").exists() is marks

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
                ruff, {"tool_input": {"file_path": "/x/a.py"}}, env=env
            ).returncode
            == 0
        )
        calls = log.read_text()
        assert "ruff check" in calls and "ruff format" in calls
        log.unlink()
        assert (
            self._run(
                ruff, {"tool_input": {"file_path": "/x/a.md"}}, env=env
            ).returncode
            == 0
        )
        assert not log.exists()  # uv not called for a non-.py file

    @requires_jq
    def test_ty_hook_surfaces_errors_on_py_only(self, tmp_path):
        # .py edit + failing type check -> exit 2 (feedback to Claude); .md -> exit 0.
        root = tmp_path / "proj"
        root.mkdir()
        bindir = tmp_path / "bin"
        bindir.mkdir()
        self._shim(bindir, "uv", 'echo "type error" >&2\nexit 1\n')
        env = self._hook_env(bindir, root)
        ty = self._command("PostToolUse", "ty check")
        assert (
            self._run(ty, {"tool_input": {"file_path": "/x/a.py"}}, env=env).returncode
            == 2
        )
        assert (
            self._run(ty, {"tool_input": {"file_path": "/x/a.md"}}, env=env).returncode
            == 0
        )


class TestCiWorkflow:
    """Guard the .github/workflows/ci.yml GitHub Actions workflow. Like TestThemeConfig
    and TestHooksConfig, this checks a real checked-in asset (not a mock): the file must
    parse as YAML, run every job on an Apple-Silicon (macOS/arm64) runner matching the
    MLX target, and drive the SAME four gates the local hooks enforce (ruff check, ruff
    format --check, ty check, pytest) via a locked `uv sync`. It also pins the security
    property that no workflow expression reaches a shell `run:` step, so a future edit
    that introduces a command-injection sink fails here instead of shipping."""

    WORKFLOW = (
        Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    )

    def _doc(self) -> dict:
        # Imported inside the test (like the mlx-vlm contract guards) so a missing dep
        # or moved path fails only this check, not collection of the whole suite.
        import yaml

        with open(self.WORKFLOW) as f:
            return yaml.safe_load(f)  # raises if not valid YAML

    def _on(self, doc: dict) -> dict:
        # PyYAML parses the bare `on:` key as the boolean True (YAML 1.1), so look it up
        # under both spellings rather than assuming a "on" string key.
        return doc.get("on", doc.get(True))

    def _steps(self, doc: dict) -> list[dict]:
        return [step for job in doc["jobs"].values() for step in job["steps"]]

    def _run_commands(self, doc: dict) -> str:
        return "\n".join(s["run"] for s in self._steps(doc) if "run" in s)

    def test_workflow_exists_and_parses(self):
        assert self.WORKFLOW.is_file()
        assert isinstance(self._doc(), dict)

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
        for name, job in self._doc()["jobs"].items():
            assert str(job["runs-on"]).startswith("macos"), (
                f"job {name!r} runs on {job['runs-on']!r}, not a macOS runner"
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

    def test_no_expression_flows_into_a_run_step(self):
        # Injection guard: a ${{ }} expression interpolated into a shell `run:` block is
        # the classic GitHub Actions command-injection sink. Expressions are fine in
        # non-run contexts (e.g. concurrency.group), but none may reach a run command.
        offenders = [
            s["run"] for s in self._steps(self._doc()) if "${{" in s.get("run", "")
        ]
        assert not offenders, f"expression flows into run step(s): {offenders}"

    def test_token_is_least_privilege(self):
        # CI only reads code and runs tests — it never pushes, comments, or releases,
        # so GITHUB_TOKEN is pinned read-only; dropping it would let a compromised dep
        # escalate via a write-scoped token (same threat model as the injection guard).
        assert self._doc().get("permissions") == {"contents": "read"}

    def test_no_gate_neutralizes_itself(self):
        # The tests above prove each gate is PRESENT; this proves none opts out of
        # blocking. A stray `continue-on-error: true` (silence a flaky test and forget)
        # keeps those substrings green while the badge goes advisory — CI passing on a
        # red suite. Conditional `if:` steps are legitimate, so only the non-blocking
        # opt-out is banned, at job and step scope.
        for name, job in self._doc()["jobs"].items():
            assert job.get("continue-on-error") is not True, (
                f"job {name} is non-blocking"
            )
            for step in job["steps"]:
                assert step.get("continue-on-error") is not True, (
                    f"job {name}: step {step.get('name')!r} is non-blocking"
                )


class TestClaudeMd:
    """Guard CLAUDE.md, the project context file loaded into every session. Like
    TestThemeConfig/TestHooksConfig/TestCiWorkflow, this checks a real checked-in asset
    (not a mock): the load-bearing files it maps must still exist AND stay named in the
    doc, every test-support module must stay documented, and every code symbol it cites
    from the app's spine must still resolve in streamlit_app. A rename/move/delete that
    leaves the doc stale — the exact currency drift a manual audit turned up
    (tests/dicom_helpers.py existed but the Tests section never mentioned it) — fails
    here instead of silently misleading the next session."""

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
        text = self._text()
        key_paths = [
            "streamlit_app.py",
            ".claude/settings.json",
            ".github/workflows/ci.yml",
            ".streamlit/config.toml",
            "pyproject.toml",
            "uv.lock",
            "README.md",
        ]
        for rel in key_paths:
            assert (self.ROOT / rel).is_file(), (
                f"documented path {rel} no longer exists"
            )
            assert rel in text, f"{rel} is no longer documented in CLAUDE.md"

    def test_every_test_module_is_documented(self):
        # Reverse guard over the tests/ dir (small and stable): every non-dunder .py must
        # be named in CLAUDE.md. This is the generic form of the drift the audit caught —
        # a test-support module (tests/dicom_helpers.py) that existed but was undocumented
        # — so a future one can't slip in without the Tests section noting it.
        text = self._text()
        modules = sorted(
            p.name
            for p in (self.ROOT / "tests").glob("*.py")
            if not p.name.startswith("__")
        )
        undocumented = [m for m in modules if m not in text]
        assert not undocumented, (
            f"tests/ modules missing from CLAUDE.md: {undocumented}"
        )

    def test_documented_spine_symbols_exist(self):
        # Every code symbol CLAUDE.md cites from the app's spine must still resolve in
        # streamlit_app, so a rename that isn't mirrored into the doc fails here. Curated
        # (not scraped from prose) to stay robust: the model/inference path, the four tab
        # renderers, the persistence helper, and the fixed-config constants.
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
            "render_ask_tab",
            "render_cxr_tab",
            "render_ct_tab",
            "render_wsi_tab",
            "CT_WINDOWS",
            "LOCALIZATION_INSTRUCTION",
            "REPETITION_PENALTY",
        ]
        for name in spine:
            assert name in text, f"spine symbol {name} dropped from CLAUDE.md"
            assert hasattr(streamlit_app, name), (
                f"CLAUDE.md documents {name}, but it no longer exists in streamlit_app"
            )
