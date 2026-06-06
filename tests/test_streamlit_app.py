import pytest
from PIL import Image

from streamlit_app import build_messages, get_generation_params, parse_response

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
        msgs = build_messages("What is a fracture?", "You are a doctor.", image=None)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are a doctor."
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == [{"type": "text", "text": "What is a fracture?"}]

    def test_with_image(self, sample_image):
        msgs = build_messages(
            "Describe this", "You are a radiologist.", image=sample_image
        )
        assert len(msgs) == 2
        assert msgs[0]["content"] == "You are a radiologist."
        user_content = msgs[1]["content"]
        assert len(user_content) == 2
        assert user_content[0] == {"type": "text", "text": "Describe this"}
        assert user_content[1] == {"type": "image"}


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
