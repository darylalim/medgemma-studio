import pytest
from PIL import Image

from streamlit_app import (
    LOCALIZATION_INSTRUCTION,
    build_messages,
    draw_boxes,
    get_generation_params,
    pad_to_square,
    parse_boxes,
    parse_response,
    scale_box,
)

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
