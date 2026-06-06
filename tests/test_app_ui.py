import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")

DEFAULT_INSTRUCTION_TEXT = "You are a helpful medical assistant."
DEFAULT_INSTRUCTION_IMAGE = "You are an expert radiologist."


@pytest.fixture
def patched_mlx(monkeypatch):
    """Replace the heavy MLX model load + inference with fast test doubles.

    Patched at the source (`mlx_vlm.*`) rather than on `streamlit_app`, because
    AppTest re-executes the script in a fresh namespace on every `.run()`.
    Returns the mock generation output so tests can set `.text`.
    """
    monkeypatch.setattr("mlx_vlm.load", lambda *a, **k: (MagicMock(), MagicMock()))
    monkeypatch.setattr("mlx_vlm.utils.load_config", lambda *a, **k: {})
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template", lambda *a, **k: "prompt"
    )
    output = MagicMock()
    output.text = "No acute findings."
    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **k: output)
    return output


@pytest.fixture
def app(patched_mlx):
    """A freshly-run AppTest with the model mocked."""
    return AppTest.from_file(APP_PATH).run()


@pytest.fixture
def png_bytes():
    """A minimal valid in-memory PNG for file-upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def test_title_renders(app):
    assert not app.exception
    assert app.title[0].value == "MedGemma Pipeline"


def test_run_button_disabled_without_prompt(app):
    assert app.button[0].label == "Run"
    assert app.button[0].disabled is True


def test_run_button_enabled_with_prompt(app):
    app.text_input[0].set_value("Describe this X-ray").run()
    assert app.button[0].disabled is False


def test_whitespace_prompt_keeps_run_disabled(app):
    app.text_input[0].set_value("   ").run()
    assert app.button[0].disabled is True


def test_default_system_instruction_text_only(app):
    assert app.text_area[0].value == DEFAULT_INSTRUCTION_TEXT


def test_thinking_toggle_defaults_off(app):
    assert app.toggle[0].label == "Thinking"
    assert app.toggle[0].value is False


def test_settings_always_visible(app):
    # Settings were moved out of an expander; nothing is collapsed on first render.
    assert len(app.expander) == 0
    assert len(app.text_area) == 1  # system instruction
    assert len(app.toggle) == 1  # thinking toggle


def test_response_renders(app):
    app.text_input[0].set_value("Describe this X-ray").run()
    app.button[0].click().run()
    assert not app.exception
    markdowns = [m.value for m in app.markdown]
    assert "### Response" in markdowns
    assert "No acute findings." in markdowns


def test_thinking_trace_renders(patched_mlx):
    patched_mlx.text = "<unused94>thought\nLet me reason.<unused95>Final answer."
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Describe this X-ray").run()
    at.toggle[0].set_value(True).run()
    at.button[0].click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "Let me reason." in markdowns  # the thinking trace
    assert "Final answer." in markdowns  # the parsed answer


def test_thinking_on_no_markers_no_expander(patched_mlx):
    patched_mlx.text = "Just a plain reply without markers."
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Describe this X-ray").run()
    at.toggle[0].set_value(True).run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.expander) == 0  # no spurious "Thinking trace" expander
    markdowns = [m.value for m in at.markdown]
    assert "### Response" in markdowns
    assert "Just a plain reply without markers." in markdowns


def test_inference_failure_renders_error(patched_mlx, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("mlx_vlm.generate", _raise)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Describe this X-ray").run()
    at.button[0].click().run()
    assert not at.exception
    assert at.error[0].value == "Inference failed: model exploded"
    assert "### Response" not in [m.value for m in at.markdown]


def test_image_upload_switches_default_instruction(app, png_bytes):
    # No image yet -> generalist default.
    assert app.text_area[0].value == DEFAULT_INSTRUCTION_TEXT
    app.file_uploader[0].upload("xray.png", png_bytes, "image/png").run()
    # Upload succeeded (Image.open did not raise) and the default adapts.
    assert not app.exception
    assert app.text_area[0].value == DEFAULT_INSTRUCTION_IMAGE


def test_edit_then_upload_preserves_instruction(app, png_bytes):
    # A user-edited instruction must survive a later image upload (no data loss).
    app.text_area[0].set_value("MY CUSTOM INSTRUCTION").run()
    app.file_uploader[0].upload("xray.png", png_bytes, "image/png").run()
    assert not app.exception
    assert app.text_area[0].value == "MY CUSTOM INSTRUCTION"


def test_invalid_image_shows_error(app):
    app.file_uploader[0].upload("bad.png", b"not-an-image", "image/png").run()
    assert not app.exception
    assert (
        app.error[0].value == "Failed to load image. Please upload a valid image file."
    )
    # uploaded_image stayed None, so the generalist default is preserved.
    assert app.text_area[0].value == DEFAULT_INSTRUCTION_TEXT


def test_image_inference_passes_image_to_model(patched_mlx, monkeypatch, png_bytes):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "No acute findings."
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_args=a) or out,
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Describe this X-ray").run()
    at.file_uploader[0].upload("xray.png", png_bytes, "image/png").run()
    at.button[0].click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 1
    img_arg = captured["gen_args"][3]  # 4th positional arg to generate
    assert isinstance(img_arg, list) and img_arg
    assert "No acute findings." in [m.value for m in at.markdown]
