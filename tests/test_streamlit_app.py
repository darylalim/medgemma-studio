import os
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from streamlit_app import (
    LOCALIZATION_INSTRUCTION,
    _detect_total_ram_gib,
    build_messages,
    draw_boxes,
    get_generation_params,
    load_ct_volume,
    normalize_hu,
    pad_to_square,
    parse_boxes,
    parse_response,
    ram_aware_slice_cap,
    scale_box,
    subsample_indices,
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
