import io
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    DEFAULT_INSTRUCTION_COMPARE,
    DEFAULT_INSTRUCTION_IMAGE,
    DEFAULT_INSTRUCTION_TEXT,
)

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")


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
    assert len(app.toggle) == 2  # thinking + localization toggles


def test_localization_toggle_disabled_without_image(app):
    assert app.toggle[1].label == "Locate anatomy (bounding boxes)"
    assert app.toggle[1].disabled is True


def test_localization_caption_discloses_override(app, png_bytes):
    # No disclosure caption until localization is actually active.
    assert not any("ignored in this mode" in c.value for c in app.caption)
    app.file_uploader[0].upload("xray.png", png_bytes, "image/png").run()
    app.toggle[1].set_value(True).run()
    assert any("ignored in this mode" in c.value for c in app.caption)


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


def test_localization_lists_detected_structures(patched_mlx, png_bytes):
    patched_mlx.text = (
        '```json\n[{"box_2d": [100, 100, 500, 500], "label": "right clavicle"}]\n```'
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Where is the right clavicle?").run()
    at.file_uploader[0].upload("xray.png", png_bytes, "image/png").run()
    at.toggle[1].set_value(True).run()  # enable localization (now that an image exists)
    at.button[0].click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" in markdowns
    assert any("right clavicle" in m for m in markdowns)
    assert "### Response" not in markdowns  # localization replaces the text view


def test_localization_passes_square_image_to_model(patched_mlx, monkeypatch, png_bytes):
    captured = {}
    out = MagicMock()
    out.text = '```json\n[{"box_2d": [0, 0, 1000, 1000], "label": "frame"}]\n```'
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_args=a) or out,
    )
    # A non-square PNG so padding is observable.
    buf = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buf, format="PNG")
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Localize the frame").run()
    at.file_uploader[0].upload("wide.png", buf.getvalue(), "image/png").run()
    at.toggle[1].set_value(True).run()
    at.button[0].click().run()
    assert not at.exception
    sent_image = captured["gen_args"][3][0]  # the single image passed to generate
    assert sent_image.size == (20, 20)  # padded to a square


def test_localization_no_boxes_warns(patched_mlx, png_bytes):
    patched_mlx.text = "I could not localize that structure."
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Where is the spine?").run()
    at.file_uploader[0].upload("xray.png", png_bytes, "image/png").run()
    at.toggle[1].set_value(True).run()
    at.button[0].click().run()
    assert not at.exception
    assert at.warning[0].value == "No bounding boxes were returned."


def test_localization_with_thinking_renders_both(patched_mlx, png_bytes):
    # Thinking + localization together: the trace is stripped before parse_boxes,
    # and both the trace and the detected-structures list render.
    patched_mlx.text = (
        "<unused94>thought\nReasoning about anatomy.<unused95>"
        '```json\n[{"box_2d": [100, 100, 500, 500], "label": "bone"}]\n```'
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Locate the bone").run()
    at.file_uploader[0].upload("xray.png", png_bytes, "image/png").run()
    at.toggle[0].set_value(True).run()  # thinking
    at.toggle[1].set_value(True).run()  # localization
    at.button[0].click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "Reasoning about anatomy." in markdowns  # thinking trace
    assert "### Detected structures" in markdowns
    assert any("bone" in m for m in markdowns)


def test_localization_renders_full_frame_box(patched_mlx, png_bytes):
    # A degenerate full-frame box is the model's "not here" fallback (e.g. asking
    # for a femur on a chest X-ray). By design it is treated as a normal detection,
    # not filtered out — the app renders what the model returns.
    patched_mlx.text = (
        '```json\n[{"box_2d": [0, 0, 1000, 1000], "label": "femur"}]\n```'
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Where is the femur?").run()
    at.file_uploader[0].upload("xray.png", png_bytes, "image/png").run()
    at.toggle[1].set_value(True).run()
    at.button[0].click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" in markdowns
    assert any("femur" in m for m in markdowns)
    assert not at.warning  # a full-frame box is a detection, not a "no boxes" case


def test_second_uploader_appears_after_first_image(app, png_bytes):
    # Only one uploader until the first image exists; then a second slot appears.
    assert len(app.file_uploader) == 1
    app.file_uploader[0].upload("first.png", png_bytes, "image/png").run()
    assert len(app.file_uploader) == 2


def test_two_images_switch_to_comparison_instruction(app, png_bytes):
    app.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    app.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    assert not app.exception
    assert app.text_area[0].value == DEFAULT_INSTRUCTION_COMPARE


def test_localization_disabled_with_two_images(app, png_bytes):
    app.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    assert app.toggle[1].disabled is False  # one image: localization available
    app.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    # Two images: localization is single-image only, so it is disabled again.
    assert app.toggle[1].disabled is True


def test_comparison_caption_disclosed(app, png_bytes):
    assert not any("Comparison mode" in c.value for c in app.caption)
    app.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    app.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    assert any("Comparison mode" in c.value for c in app.caption)


def test_two_image_comparison_passes_both_images(patched_mlx, monkeypatch, png_bytes):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "The second study shows interval improvement."
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_args=a) or out,
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Compare these studies").run()
    at.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    at.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    at.button[0].click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 2
    img_arg = captured["gen_args"][3]  # 4th positional arg to generate
    assert isinstance(img_arg, list) and len(img_arg) == 2
    assert "The second study shows interval improvement." in [
        m.value for m in at.markdown
    ]


def test_comparison_uses_larger_token_budget(patched_mlx, monkeypatch, png_bytes):
    # Two images (no thinking) -> the 600-token comparison budget, not the
    # single-image 300, reaches generate().
    captured = {}
    out = MagicMock()
    out.text = "Comparison."
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_kwargs=k) or out,
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Compare these").run()
    at.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    at.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    at.button[0].click().run()
    assert not at.exception
    assert captured["gen_kwargs"]["max_tokens"] == 600


def test_comparison_with_thinking_renders_both(patched_mlx, monkeypatch, png_bytes):
    # Thinking + comparison together: both images are sent, the trace is stripped
    # before show_response, and both the trace (in an expander) and the comparison
    # answer render. Thinking takes precedence, so the 1300-token budget is used.
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    patched_mlx.text = (
        "<unused94>thought\nComparing the two studies.<unused95>"
        "The second study shows interval clearing of the left-base opacity."
    )
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_kwargs=k) or patched_mlx,
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Compare these studies").run()
    at.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    at.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    at.toggle[0].set_value(True).run()  # thinking
    at.button[0].click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 2  # both images still sent
    assert captured["gen_kwargs"]["max_tokens"] == 1600  # thinking+comparison budget
    assert len(at.expander) == 1  # the "Thinking trace" expander
    markdowns = [m.value for m in at.markdown]
    assert "Comparing the two studies." in markdowns  # thinking trace
    assert "### Response" in markdowns
    assert any("interval clearing" in m for m in markdowns)  # comparison answer


def test_text_only_inference_passes_no_image_to_model(patched_mlx, monkeypatch):
    # The zero-image branch: image_for_model = [] or None -> None, and num_images 0.
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
    at.text_input[0].set_value("What is a fracture?").run()
    at.button[0].click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 0
    assert captured["gen_args"][3] is None  # no image -> None passed positionally
    assert "No acute findings." in [m.value for m in at.markdown]


def test_invalid_second_image_falls_back_to_single_image_mode(app, png_bytes):
    # A valid first image plus a failed second upload stays in single-image mode:
    # the error shows, the comparison persona is NOT applied, and localization
    # (single-image only) remains available.
    app.file_uploader[0].upload("good.png", png_bytes, "image/png").run()
    app.file_uploader[1].upload("bad.png", b"not-an-image", "image/png").run()
    assert not app.exception
    assert any(
        e.value == "Failed to load image. Please upload a valid image file."
        for e in app.error
    )
    assert app.text_area[0].value == DEFAULT_INSTRUCTION_IMAGE
    assert not any("Comparison mode" in c.value for c in app.caption)
    assert app.toggle[1].disabled is False  # one valid image -> localization available


def test_edit_then_second_upload_preserves_instruction(app, png_bytes):
    # A user-edited instruction must survive entering comparison mode (the
    # comparison persona must not overwrite a touched value).
    app.text_area[0].set_value("MY CUSTOM INSTRUCTION").run()
    app.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    app.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    assert not app.exception
    assert app.text_area[0].value == "MY CUSTOM INSTRUCTION"
    assert app.text_area[0].value != DEFAULT_INSTRUCTION_COMPARE


def test_removing_first_image_collapses_second_slot(app, png_bytes):
    # The second uploader is nested under `if image1 is not None`; removing the
    # first image collapses it and reverts an untouched default to the text persona.
    app.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    app.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    assert app.text_area[0].value == DEFAULT_INSTRUCTION_COMPARE
    assert len(app.file_uploader) == 2
    app.file_uploader[0].clear().run()
    assert not app.exception
    assert len(app.file_uploader) == 1  # second slot is gone once image1 is None
    assert app.text_area[0].value == DEFAULT_INSTRUCTION_TEXT
    assert not any("Comparison mode" in c.value for c in app.caption)


def test_stale_localization_toggle_runs_comparison_with_two_images(
    patched_mlx, monkeypatch, png_bytes
):
    # Enabling localization with one image then adding a second leaves the toggle
    # disabled but its value stale-True. The run-time guard
    # `localize = is_localizing and len(images) == 1` must force the comparison path
    # (both images sent, 600-token budget, no localization output).
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Both lungs are clear."
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_kwargs=k) or out,
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Compare these").run()
    at.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    at.toggle[1].set_value(True).run()  # enable localization while single image
    at.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    at.button[0].click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 2  # both sent, not a padded single
    assert captured["gen_kwargs"]["max_tokens"] == 600  # comparison budget, not 1000
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" not in markdowns  # localization branch not taken
    assert "### Response" in markdowns
    assert not at.warning


def test_comparison_sends_unpadded_images(patched_mlx, monkeypatch):
    # Comparison must send the original images verbatim; pad-to-square is
    # localization-only. Non-square inputs make any stray padding observable.
    captured = {}
    out = MagicMock()
    out.text = "Comparison."
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_args=a) or out,
    )
    buf = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buf, format="PNG")
    wide_png = buf.getvalue()
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Compare these").run()
    at.file_uploader[0].upload("a.png", wide_png, "image/png").run()
    at.file_uploader[1].upload("b.png", wide_png, "image/png").run()
    at.button[0].click().run()
    assert not at.exception
    img_arg = captured["gen_args"][3]
    assert isinstance(img_arg, list) and len(img_arg) == 2
    assert [im.size for im in img_arg] == [(20, 10), (20, 10)]  # unpadded originals


def test_comparison_labels_images_in_prompt(patched_mlx, monkeypatch, png_bytes):
    # main() passes "First image:"/"Second image:" labels so the comparison
    # persona's ordinal wording binds to specific images.
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(args=a) or "prompt",
    )
    out = MagicMock()
    out.text = "Comparison."
    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **k: out)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input[0].set_value("Compare these").run()
    at.file_uploader[0].upload("before.png", png_bytes, "image/png").run()
    at.file_uploader[1].upload("after.png", png_bytes, "image/png").run()
    at.button[0].click().run()
    assert not at.exception
    messages = captured["args"][2]  # 3rd positional arg to apply_chat_template
    user_texts = [p["text"] for p in messages[1]["content"] if p["type"] == "text"]
    assert "First image:" in user_texts
    assert "Second image:" in user_texts
