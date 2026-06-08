import io
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image
from streamlit.testing.v1 import AppTest

from streamlit_app import (
    DEFAULT_INSTRUCTION_COMPARE,
    DEFAULT_INSTRUCTION_CT,
    DEFAULT_INSTRUCTION_IMAGE,
    DEFAULT_INSTRUCTION_TEXT,
    DEFAULT_INSTRUCTION_WSI,
    REPETITION_CONTEXT_SIZE,
    REPETITION_PENALTY,
)
from tests.dicom_helpers import dicom_bytes

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")


@pytest.fixture
def patched_mlx(monkeypatch):
    """Replace the heavy MLX model load + inference with fast test doubles.

    Patched at the source (`mlx_vlm.*`) rather than on `streamlit_app`, because
    AppTest re-executes the script in a fresh namespace on every `.run()`. Returns
    the mock generation output so tests can set `.text`.
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


def _dicom_bytes(*args, **kwargs):
    """Raw bytes of an in-memory CT DICOM slice, for AppTest's file_uploader.upload()."""
    return dicom_bytes(*args, **kwargs).getvalue()


def _force_ram_gib(monkeypatch, gib):
    """Force the CT slice cap deterministically by making _detect_total_ram_gib's
    os.sysconf reading report a fixed installed-RAM value, regardless of host."""
    real_sysconf = os.sysconf

    def fake(name):
        if name == "SC_PHYS_PAGES":
            return int(gib) * 1024**3 // 4096
        if name == "SC_PAGE_SIZE":
            return 4096
        return real_sysconf(name)

    monkeypatch.setattr(os, "sysconf", fake)


def _upload_ct_pair(at):
    """Upload two DICOM slices to the CT tab (chained: AppTest replaces, not appends)."""
    at.file_uploader(key="ct_files").upload(
        "a.dcm", _dicom_bytes(2, 200), "application/dicom"
    ).upload("b.dcm", _dicom_bytes(1, 100), "application/dicom").run()


class _FakeSlide:
    """OpenSlide stand-in for the WSI tab tests; a saturated thumbnail reads as all
    tissue, so a 3000x3000 single level yields a 3x3 grid of nine patches."""

    def __init__(
        self,
        properties=None,
        thumbnail=None,
        level_dimensions=((3000, 3000),),
        level_downsamples=(1.0,),
    ):
        self.level_dimensions = list(level_dimensions)
        self.level_downsamples = list(level_downsamples)
        self.dimensions = self.level_dimensions[0]
        self.properties = properties or {"openslide.objective-power": "40"}
        self._thumbnail = thumbnail

    def get_thumbnail(self, size):
        if self._thumbnail is not None:
            return self._thumbnail
        arr = np.zeros((800, 800, 3), dtype=np.uint8)
        arr[..., 0], arr[..., 2] = 150, 140
        return Image.fromarray(arr, "RGB")

    def read_region(self, location, level, size):
        return Image.new("RGBA", size, (150, 40, 140, 255))

    def close(self):
        pass


@pytest.fixture
def patched_openslide(monkeypatch):
    """Replace OpenSlide with a fake slide that yields a tissue-filled 3x3 grid."""
    monkeypatch.setattr("openslide.OpenSlide", lambda path: _FakeSlide())


def _upload_slide(at, data=b"slide"):
    at.file_uploader(key="wsi_files").upload(
        "slide.svs", data, "application/octet-stream"
    ).run()


# --------------------------------------------------------------------------- #
# Layout / shared
# --------------------------------------------------------------------------- #


def test_title_renders(app):
    assert not app.exception
    assert app.title[0].value == "MedGemma Pipeline"


def test_four_tabs_render(app):
    assert [t.label for t in app.tabs] == [
        "Ask",
        "Chest X-ray",
        "Computed Tomography",
        "Pathology (WSI)",
    ]


def test_no_expander_on_first_render(app):
    # The "Thinking trace" expander appears only after a thinking response.
    assert len(app.expander) == 0


def test_each_tab_has_independent_settings(app):
    # Per-tab instruction + thinking widgets keyed by tab; one Run button each.
    assert [w.key for w in app.text_area] == [
        "ask_instruction",
        "cxr_instruction",
        "ct_instruction",
        "wsi_instruction",
    ]
    assert {w.key for w in app.toggle} == {
        "ask_thinking",
        "cxr_thinking",
        "cxr_localize",
        "ct_thinking",
        "wsi_thinking",
    }
    assert [w.key for w in app.button] == ["ask_run", "cxr_run", "ct_run", "wsi_run"]


def test_thinking_toggles_are_independent(app):
    app.toggle(key="ask_thinking").set_value(True).run()
    assert app.toggle(key="ask_thinking").value is True
    assert app.toggle(key="cxr_thinking").value is False
    assert app.toggle(key="ct_thinking").value is False


# --------------------------------------------------------------------------- #
# Ask tab (text-only Q&A)
# --------------------------------------------------------------------------- #


def test_ask_default_instruction_is_text_persona(app):
    assert app.text_area(key="ask_instruction").value == DEFAULT_INSTRUCTION_TEXT


def test_ask_thinking_defaults_off(app):
    assert app.toggle(key="ask_thinking").value is False


def test_ask_run_disabled_without_prompt(app):
    assert app.button(key="ask_run").disabled is True


def test_ask_run_enabled_with_prompt(app):
    app.text_input(key="ask_prompt").set_value("What causes effusion?").run()
    assert app.button(key="ask_run").disabled is False


def test_ask_whitespace_prompt_keeps_run_disabled(app):
    app.text_input(key="ask_prompt").set_value("   ").run()
    assert app.button(key="ask_run").disabled is True


def test_ask_response_renders(app):
    app.text_input(key="ask_prompt").set_value("What is a fracture?").run()
    app.button(key="ask_run").click().run()
    assert not app.exception
    markdowns = [m.value for m in app.markdown]
    assert "### Response" in markdowns
    assert "No acute findings." in markdowns


def test_ask_thinking_trace_renders(patched_mlx):
    patched_mlx.text = "<unused94>thought\nLet me reason.<unused95>Final answer."
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ask_prompt").set_value("Why?").run()
    at.toggle(key="ask_thinking").set_value(True).run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "Let me reason." in markdowns  # the thinking trace
    assert "Final answer." in markdowns  # the parsed answer


def test_ask_thinking_no_markers_no_expander(patched_mlx):
    patched_mlx.text = "Just a plain reply without markers."
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ask_prompt").set_value("Why?").run()
    at.toggle(key="ask_thinking").set_value(True).run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    assert len(at.expander) == 0  # no spurious "Thinking trace" expander
    assert "Just a plain reply without markers." in [m.value for m in at.markdown]


def test_ask_inference_failure_renders_error(patched_mlx, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr("mlx_vlm.generate", _raise)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ask_prompt").set_value("Why?").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    assert any(e.value == "Inference failed: model exploded" for e in at.error)
    assert "### Response" not in [m.value for m in at.markdown]


def test_repetition_penalty_passed_to_generate(patched_mlx, monkeypatch):
    # Greedy decoding loops without a repetition penalty; guard that run_model
    # always passes it (and the context size) to generate().
    captured = {}
    out = MagicMock()
    out.text = "ok"
    monkeypatch.setattr(
        "mlx_vlm.generate", lambda *a, **k: captured.update(gen_kwargs=k) or out
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ask_prompt").set_value("Why?").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    assert captured["gen_kwargs"]["repetition_penalty"] == REPETITION_PENALTY
    assert captured["gen_kwargs"]["repetition_context_size"] == REPETITION_CONTEXT_SIZE


def test_ask_passes_no_image_to_model(patched_mlx, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "No acute findings."
    monkeypatch.setattr(
        "mlx_vlm.generate", lambda *a, **k: captured.update(gen_args=a) or out
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ask_prompt").set_value("What is a fracture?").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 0
    assert captured["gen_args"][3] is None  # no image -> None passed positionally


# --------------------------------------------------------------------------- #
# Chest X-ray tab (single image / comparison / localization)
# --------------------------------------------------------------------------- #


def test_cxr_default_instruction_is_image_persona(app):
    # Text-only Q&A now lives in the Ask tab, so the CXR default is the radiologist
    # persona even before an image is attached.
    assert app.text_area(key="cxr_instruction").value == DEFAULT_INSTRUCTION_IMAGE


def test_cxr_localization_toggle_disabled_without_image(app):
    assert app.toggle(key="cxr_localize").label == "Locate anatomy (bounding boxes)"
    assert app.toggle(key="cxr_localize").disabled is True


def test_cxr_localization_caption_discloses_override(app, png_bytes):
    assert not any("ignored in this mode" in c.value for c in app.caption)
    app.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    app.toggle(key="cxr_localize").set_value(True).run()
    assert any("ignored in this mode" in c.value for c in app.caption)


def test_cxr_second_uploader_appears_after_first_image(app, png_bytes):
    assert "cxr_image2" not in [w.key for w in app.file_uploader]
    app.file_uploader(key="cxr_image1").upload(
        "first.png", png_bytes, "image/png"
    ).run()
    assert "cxr_image2" in [w.key for w in app.file_uploader]


def test_cxr_two_images_switch_to_comparison_instruction(app, png_bytes):
    app.file_uploader(key="cxr_image1").upload("a.png", png_bytes, "image/png").run()
    app.file_uploader(key="cxr_image2").upload("b.png", png_bytes, "image/png").run()
    assert not app.exception
    assert app.text_area(key="cxr_instruction").value == DEFAULT_INSTRUCTION_COMPARE


def test_cxr_localization_disabled_with_two_images(app, png_bytes):
    app.file_uploader(key="cxr_image1").upload("a.png", png_bytes, "image/png").run()
    assert app.toggle(key="cxr_localize").disabled is False  # one image
    app.file_uploader(key="cxr_image2").upload("b.png", png_bytes, "image/png").run()
    assert app.toggle(key="cxr_localize").disabled is True  # two images -> single-only


def test_cxr_comparison_caption_disclosed(app, png_bytes):
    assert not any("Comparison mode" in c.value for c in app.caption)
    app.file_uploader(key="cxr_image1").upload("a.png", png_bytes, "image/png").run()
    app.file_uploader(key="cxr_image2").upload("b.png", png_bytes, "image/png").run()
    assert any("Comparison mode" in c.value for c in app.caption)


def test_cxr_edit_then_upload_preserves_instruction(app, png_bytes):
    app.text_area(key="cxr_instruction").set_value("MY CUSTOM INSTRUCTION").run()
    app.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    assert not app.exception
    assert app.text_area(key="cxr_instruction").value == "MY CUSTOM INSTRUCTION"


def test_cxr_invalid_image_shows_error(app):
    app.file_uploader(key="cxr_image1").upload(
        "bad.png", b"not-an-image", "image/png"
    ).run()
    assert not app.exception
    assert any(
        e.value == "Failed to load image. Please upload a valid image file."
        for e in app.error
    )
    # The upload failed, so the persona stays the single-image default.
    assert app.text_area(key="cxr_instruction").value == DEFAULT_INSTRUCTION_IMAGE


def test_cxr_image_inference_passes_image_to_model(patched_mlx, monkeypatch, png_bytes):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "No acute findings."
    monkeypatch.setattr(
        "mlx_vlm.generate", lambda *a, **k: captured.update(gen_args=a) or out
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Describe this X-ray").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 1
    img_arg = captured["gen_args"][3]
    assert isinstance(img_arg, list) and img_arg
    assert "No acute findings." in [m.value for m in at.markdown]


def test_cxr_localization_lists_detected_structures(patched_mlx, png_bytes):
    patched_mlx.text = (
        '```json\n[{"box_2d": [100, 100, 500, 500], "label": "right clavicle"}]\n```'
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Where is the right clavicle?").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" in markdowns
    assert any("right clavicle" in m for m in markdowns)
    assert "### Response" not in markdowns  # localization replaces the text view


def test_cxr_localization_passes_square_image_to_model(patched_mlx, monkeypatch):
    captured = {}
    out = MagicMock()
    out.text = '```json\n[{"box_2d": [0, 0, 1000, 1000], "label": "frame"}]\n```'
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_args=a, gen_kwargs=k) or out,
    )
    buf = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buf, format="PNG")  # non-square -> padding visible
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Localize the frame").run()
    at.file_uploader(key="cxr_image1").upload(
        "wide.png", buf.getvalue(), "image/png"
    ).run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    assert captured["gen_args"][3][0].size == (20, 20)  # padded to a square
    # The loop-guard penalty applies to the localization path too.
    assert captured["gen_kwargs"]["repetition_penalty"] == REPETITION_PENALTY
    assert captured["gen_kwargs"]["repetition_context_size"] == REPETITION_CONTEXT_SIZE


def test_cxr_localization_no_boxes_warns(patched_mlx, png_bytes):
    patched_mlx.text = "I could not localize that structure."
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Where is the spine?").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    assert any(w.value == "No bounding boxes were returned." for w in at.warning)


def test_cxr_localization_with_thinking_renders_both(patched_mlx, png_bytes):
    patched_mlx.text = (
        "<unused94>thought\nReasoning about anatomy.<unused95>"
        '```json\n[{"box_2d": [100, 100, 500, 500], "label": "bone"}]\n```'
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Locate the bone").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_thinking").set_value(True).run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "Reasoning about anatomy." in markdowns  # thinking trace
    assert "### Detected structures" in markdowns
    assert any("bone" in m for m in markdowns)


def test_cxr_localization_renders_full_frame_box(patched_mlx, png_bytes):
    # A degenerate full-frame box is the model's "not here" fallback; by design it is
    # rendered as a normal detection, not filtered out.
    patched_mlx.text = (
        '```json\n[{"box_2d": [0, 0, 1000, 1000], "label": "femur"}]\n```'
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Where is the femur?").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" in markdowns
    assert any("femur" in m for m in markdowns)
    assert not at.warning


def test_cxr_comparison_passes_both_images(patched_mlx, monkeypatch, png_bytes):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "The second study shows interval improvement."
    monkeypatch.setattr(
        "mlx_vlm.generate", lambda *a, **k: captured.update(gen_args=a) or out
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Compare these studies").run()
    at.file_uploader(key="cxr_image1").upload(
        "before.png", png_bytes, "image/png"
    ).run()
    at.file_uploader(key="cxr_image2").upload("after.png", png_bytes, "image/png").run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 2
    img_arg = captured["gen_args"][3]
    assert isinstance(img_arg, list) and len(img_arg) == 2
    assert "The second study shows interval improvement." in [
        m.value for m in at.markdown
    ]


def test_cxr_comparison_uses_larger_token_budget(patched_mlx, monkeypatch, png_bytes):
    captured = {}
    out = MagicMock()
    out.text = "Comparison."
    monkeypatch.setattr(
        "mlx_vlm.generate", lambda *a, **k: captured.update(gen_kwargs=k) or out
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Compare these").run()
    at.file_uploader(key="cxr_image1").upload(
        "before.png", png_bytes, "image/png"
    ).run()
    at.file_uploader(key="cxr_image2").upload("after.png", png_bytes, "image/png").run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    assert captured["gen_kwargs"]["max_tokens"] == 600
    assert captured["gen_kwargs"]["repetition_penalty"] == REPETITION_PENALTY


def test_cxr_comparison_with_thinking_renders_both(patched_mlx, monkeypatch, png_bytes):
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
        "mlx_vlm.generate", lambda *a, **k: captured.update(gen_kwargs=k) or patched_mlx
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Compare these studies").run()
    at.file_uploader(key="cxr_image1").upload(
        "before.png", png_bytes, "image/png"
    ).run()
    at.file_uploader(key="cxr_image2").upload("after.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_thinking").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 2  # both images still sent
    assert captured["gen_kwargs"]["max_tokens"] == 1600  # thinking+comparison budget
    assert len(at.expander) == 1  # the "Thinking trace" expander
    markdowns = [m.value for m in at.markdown]
    assert "Comparing the two studies." in markdowns  # thinking trace
    assert "### Response" in markdowns
    assert any("interval clearing" in m for m in markdowns)


def test_cxr_invalid_second_image_falls_back_to_single_image_mode(app, png_bytes):
    app.file_uploader(key="cxr_image1").upload("good.png", png_bytes, "image/png").run()
    app.file_uploader(key="cxr_image2").upload(
        "bad.png", b"not-an-image", "image/png"
    ).run()
    assert not app.exception
    assert any(
        e.value == "Failed to load image. Please upload a valid image file."
        for e in app.error
    )
    assert app.text_area(key="cxr_instruction").value == DEFAULT_INSTRUCTION_IMAGE
    assert not any("Comparison mode" in c.value for c in app.caption)
    assert app.toggle(key="cxr_localize").disabled is False  # one valid image


def test_cxr_edit_then_second_upload_preserves_instruction(app, png_bytes):
    app.text_area(key="cxr_instruction").set_value("MY CUSTOM INSTRUCTION").run()
    app.file_uploader(key="cxr_image1").upload(
        "before.png", png_bytes, "image/png"
    ).run()
    app.file_uploader(key="cxr_image2").upload(
        "after.png", png_bytes, "image/png"
    ).run()
    assert not app.exception
    assert app.text_area(key="cxr_instruction").value == "MY CUSTOM INSTRUCTION"
    assert app.text_area(key="cxr_instruction").value != DEFAULT_INSTRUCTION_COMPARE


def test_cxr_removing_first_image_collapses_second_slot(app, png_bytes):
    app.file_uploader(key="cxr_image1").upload(
        "before.png", png_bytes, "image/png"
    ).run()
    app.file_uploader(key="cxr_image2").upload(
        "after.png", png_bytes, "image/png"
    ).run()
    assert app.text_area(key="cxr_instruction").value == DEFAULT_INSTRUCTION_COMPARE
    assert "cxr_image2" in [w.key for w in app.file_uploader]
    app.file_uploader(key="cxr_image1").clear().run()
    assert not app.exception
    assert "cxr_image2" not in [w.key for w in app.file_uploader]
    # Untouched default reverts to the single-image persona (Ask owns text-only).
    assert app.text_area(key="cxr_instruction").value == DEFAULT_INSTRUCTION_IMAGE
    assert not any("Comparison mode" in c.value for c in app.caption)


def test_cxr_stale_localization_toggle_runs_comparison_with_two_images(
    patched_mlx, monkeypatch, png_bytes
):
    # Enabling localization with one image then adding a second leaves the toggle
    # disabled but stale-True. The run-time guard `localize = is_localizing and
    # len(images) == 1` must force the comparison path.
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Both lungs are clear."
    monkeypatch.setattr(
        "mlx_vlm.generate", lambda *a, **k: captured.update(gen_kwargs=k) or out
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Compare these").run()
    at.file_uploader(key="cxr_image1").upload(
        "before.png", png_bytes, "image/png"
    ).run()
    at.toggle(key="cxr_localize").set_value(True).run()  # enabled while single image
    at.file_uploader(key="cxr_image2").upload("after.png", png_bytes, "image/png").run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 2  # both sent, not a padded single
    assert captured["gen_kwargs"]["max_tokens"] == 600  # comparison budget, not 1000
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" not in markdowns  # localization branch not taken
    assert "### Response" in markdowns
    assert not at.warning


def test_cxr_comparison_sends_unpadded_images(patched_mlx, monkeypatch):
    captured = {}
    out = MagicMock()
    out.text = "Comparison."
    monkeypatch.setattr(
        "mlx_vlm.generate", lambda *a, **k: captured.update(gen_args=a) or out
    )
    buf = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buf, format="PNG")
    wide_png = buf.getvalue()
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Compare these").run()
    at.file_uploader(key="cxr_image1").upload("a.png", wide_png, "image/png").run()
    at.file_uploader(key="cxr_image2").upload("b.png", wide_png, "image/png").run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    img_arg = captured["gen_args"][3]
    assert [im.size for im in img_arg] == [(20, 10), (20, 10)]  # unpadded originals


def test_cxr_comparison_labels_images_in_prompt(patched_mlx, monkeypatch, png_bytes):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(args=a) or "prompt",
    )
    out = MagicMock()
    out.text = "Comparison."
    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **k: out)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="cxr_prompt").set_value("Compare these").run()
    at.file_uploader(key="cxr_image1").upload(
        "before.png", png_bytes, "image/png"
    ).run()
    at.file_uploader(key="cxr_image2").upload("after.png", png_bytes, "image/png").run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    messages = captured["args"][2]  # 3rd positional arg to apply_chat_template
    user_texts = [p["text"] for p in messages[1]["content"] if p["type"] == "text"]
    assert "First image:" in user_texts
    assert "Second image:" in user_texts


# --------------------------------------------------------------------------- #
# Computed Tomography tab (DICOM -> windowing -> multi-slice)
# --------------------------------------------------------------------------- #


def test_ct_default_instruction_is_ct_persona(app):
    assert app.text_area(key="ct_instruction").value == DEFAULT_INSTRUCTION_CT


def test_ct_caption_describes_dicom_upload(app):
    assert any("DICOM" in c.value for c in app.caption)


def test_ct_slider_present_or_memory_capped(app):
    # The slice slider is RAM-aware; on a very low-memory host it collapses to a
    # fixed 2-slice cap with a caption instead.
    slider_present = "ct_slices" in [w.key for w in app.slider]
    memory_capped = any("Limited memory" in c.value for c in app.caption)
    assert slider_present or memory_capped


def test_ct_run_requires_prompt_and_files(app, png_bytes):
    assert app.button(key="ct_run").disabled is True
    app.text_input(key="ct_prompt").set_value("Any lesions?").run()
    assert app.button(key="ct_run").disabled is True  # prompt but no files
    _upload_ct_pair(app)
    assert app.button(key="ct_run").disabled is False


def test_ct_inference_passes_windowed_slices(patched_mlx, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Two contiguous slices of the liver."
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_args=a, gen_kwargs=k) or out,
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ct_prompt").set_value("Are there hypodense lesions?").run()
    _upload_ct_pair(at)
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 2
    img_arg = captured["gen_args"][3]
    assert isinstance(img_arg, list) and len(img_arg) == 2
    assert all(im.mode == "RGB" for im in img_arg)  # windowed to false-color RGB
    assert captured["gen_kwargs"]["max_tokens"] == 2000  # CT multi-slice budget
    # The repetition penalty must reach the CT path — that's where greedy decoding
    # looped before the fix.
    assert captured["gen_kwargs"]["repetition_penalty"] == REPETITION_PENALTY
    assert captured["gen_kwargs"]["repetition_context_size"] == REPETITION_CONTEXT_SIZE
    assert "Two contiguous slices of the liver." in [m.value for m in at.markdown]


def test_ct_labels_slices_in_prompt(patched_mlx, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(args=a) or "prompt",
    )
    out = MagicMock()
    out.text = "Findings."
    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **k: out)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ct_prompt").set_value("Describe the volume").run()
    _upload_ct_pair(at)
    at.button(key="ct_run").click().run()
    assert not at.exception
    messages = captured["args"][2]
    user_texts = [p["text"] for p in messages[1]["content"] if p["type"] == "text"]
    assert "SLICE 1" in user_texts
    assert "SLICE 2" in user_texts


def test_ct_with_thinking_uses_larger_budget(patched_mlx, monkeypatch):
    captured = {}
    patched_mlx.text = (
        "<unused94>thought\nReviewing each slice.<unused95>No focal lesion."
    )
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_kwargs=k) or patched_mlx,
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ct_prompt").set_value("Any lesions?").run()
    _upload_ct_pair(at)
    at.toggle(key="ct_thinking").set_value(True).run()
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert captured["gen_kwargs"]["max_tokens"] == 2500  # thinking + CT budget
    assert len(at.expander) == 1  # thinking trace
    markdowns = [m.value for m in at.markdown]
    assert "Reviewing each slice." in markdowns
    assert "No focal lesion." in markdowns


def test_ct_invalid_dicom_shows_error(patched_mlx):
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ct_prompt").set_value("Describe this").run()
    at.file_uploader(key="ct_files").upload(
        "bad.dcm", b"not-a-dicom", "application/dicom"
    ).run()
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert any("Failed to read DICOM series" in e.value for e in at.error)
    assert "### Response" not in [m.value for m in at.markdown]


def test_ct_subsamples_to_slider_count(patched_mlx, monkeypatch):
    # The slider value (not the upload count) drives how many windowed slices reach
    # the model: upload 6, request 4, expect 4.
    _force_ram_gib(monkeypatch, 32)  # deterministic slider range (max 16)
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Findings."
    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **k: out)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ct_prompt").set_value("Describe the volume").run()
    uploader = at.file_uploader(key="ct_files")
    for i in range(1, 7):  # six single-series slices
        uploader = uploader.upload(
            f"s{i}.dcm", _dicom_bytes(i, 100 + i), "application/dicom"
        )
    uploader.run()
    at.slider(key="ct_slices").set_value(4).run()
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 4  # subsampled from 6 to 4


def test_ct_slider_default_reflects_ram(patched_mlx, monkeypatch):
    _force_ram_gib(monkeypatch, 32)  # ram_aware_slice_cap -> (default 8, max 16)
    at = AppTest.from_file(APP_PATH).run()
    assert at.slider(key="ct_slices").value == 8


def test_ct_memory_capped_shows_caption_not_slider(patched_mlx, monkeypatch):
    _force_ram_gib(monkeypatch, 16)  # below base + headroom -> (2, 2): no slider
    at = AppTest.from_file(APP_PATH).run()
    assert "ct_slices" not in [w.key for w in at.slider]
    assert any("Limited memory" in c.value for c in at.caption)


def test_ct_rejects_mixed_series_with_error(patched_mlx):
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ct_prompt").set_value("Describe").run()
    at.file_uploader(key="ct_files").upload(
        "a.dcm", _dicom_bytes(1, 100, series_uid="1.2.3"), "application/dicom"
    ).upload(
        "b.dcm", _dicom_bytes(2, 200, series_uid="1.2.4"), "application/dicom"
    ).run()
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert any("Multiple DICOM series" in e.value for e in at.error)
    assert "### Response" not in [m.value for m in at.markdown]


def test_ct_multi_frame_shows_error_not_crash(patched_mlx):
    # A multi-frame DICOM (3D pixel array) must surface the friendly error, not an
    # unhandled traceback from window_ct_slice.
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="ct_prompt").set_value("Describe").run()
    at.file_uploader(key="ct_files").upload(
        "vol.dcm", _dicom_bytes(1, 100, frames=3), "application/dicom"
    ).run()
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert any("single-frame" in e.value for e in at.error)


# --------------------------------------------------------------------------- #
# Pathology (WSI) tab (slide -> tissue patches -> multi-image)
# --------------------------------------------------------------------------- #


def test_wsi_default_instruction_is_pathology_persona(app):
    assert app.text_area(key="wsi_instruction").value == DEFAULT_INSTRUCTION_WSI


def test_wsi_caption_describes_slide_upload(app):
    assert any("whole-slide" in c.value for c in app.caption)


def test_wsi_run_requires_prompt_and_file(app):
    assert app.button(key="wsi_run").disabled is True
    app.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    assert app.button(key="wsi_run").disabled is True  # prompt but no slide
    app.file_uploader(key="wsi_files").upload(
        "slide.svs", b"x", "application/octet-stream"
    ).run()
    assert app.button(key="wsi_run").disabled is False


def test_wsi_inference_passes_patches(patched_mlx, patched_openslide, monkeypatch):
    _force_ram_gib(monkeypatch, 32)  # deterministic slider range (max 16)
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Moderately differentiated adenocarcinoma."
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_args=a, gen_kwargs=k) or out,
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.slider(key="wsi_patches").set_value(4).run()
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 4
    img_arg = captured["gen_args"][3]
    assert isinstance(img_arg, list) and len(img_arg) == 4
    assert all(im.mode == "RGB" for im in img_arg)  # patches read as RGB
    assert captured["gen_kwargs"]["max_tokens"] == 2000  # WSI multi-patch budget
    # The loop-guard penalty must reach the long multi-patch read.
    assert captured["gen_kwargs"]["repetition_penalty"] == REPETITION_PENALTY
    assert captured["gen_kwargs"]["repetition_context_size"] == REPETITION_CONTEXT_SIZE
    assert "Moderately differentiated adenocarcinoma." in [m.value for m in at.markdown]


def test_wsi_labels_patches_in_prompt(patched_mlx, patched_openslide, monkeypatch):
    _force_ram_gib(monkeypatch, 32)
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(args=a) or "prompt",
    )
    out = MagicMock()
    out.text = "Findings."
    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **k: out)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.slider(key="wsi_patches").set_value(3).run()
    at.button(key="wsi_run").click().run()
    assert not at.exception
    messages = captured["args"][2]
    user_texts = [p["text"] for p in messages[1]["content"] if p["type"] == "text"]
    assert "PATCH 1" in user_texts
    assert "PATCH 3" in user_texts


def test_wsi_subsamples_to_slider_count(patched_mlx, patched_openslide, monkeypatch):
    # Nine tissue patches in the grid; the slider (not the grid size) sets how many
    # reach the model.
    _force_ram_gib(monkeypatch, 32)
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Findings."
    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **k: out)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.slider(key="wsi_patches").set_value(6).run()
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 6


def test_wsi_caption_discloses_actual_magnification(
    patched_mlx, patched_openslide, monkeypatch
):
    _force_ram_gib(monkeypatch, 32)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    assert not at.exception
    # A 10x request on a single-level 40x slide is honestly disclosed as ~40x.
    assert any("sampled at ~40.0x" in c.value for c in at.caption)


def test_wsi_with_thinking_uses_larger_budget(
    patched_mlx, patched_openslide, monkeypatch
):
    _force_ram_gib(monkeypatch, 32)
    captured = {}
    patched_mlx.text = (
        "<unused94>thought\nReviewing each patch.<unused95>No malignancy seen."
    )
    monkeypatch.setattr(
        "mlx_vlm.generate",
        lambda *a, **k: captured.update(gen_kwargs=k) or patched_mlx,
    )
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="wsi_prompt").set_value("Any malignancy?").run()
    _upload_slide(at)
    at.toggle(key="wsi_thinking").set_value(True).run()
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert captured["gen_kwargs"]["max_tokens"] == 2500  # thinking + WSI budget
    assert len(at.expander) == 1  # thinking trace
    markdowns = [m.value for m in at.markdown]
    assert "Reviewing each patch." in markdowns
    assert "No malignancy seen." in markdowns


def test_wsi_invalid_slide_shows_error(patched_mlx, monkeypatch):
    def _boom(path):
        raise OSError("not a slide")

    monkeypatch.setattr("openslide.OpenSlide", _boom)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="wsi_prompt").set_value("Describe").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert any("Failed to read slide" in e.value for e in at.error)
    assert "### Response" not in [m.value for m in at.markdown]


def test_wsi_no_tissue_shows_error(patched_mlx, monkeypatch):
    white = Image.new("RGB", (800, 800), (255, 255, 255))
    monkeypatch.setattr("openslide.OpenSlide", lambda path: _FakeSlide(thumbnail=white))
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="wsi_prompt").set_value("Describe").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert any("No tissue" in e.value for e in at.error)
    assert "### Response" not in [m.value for m in at.markdown]


def test_wsi_magnification_selects_pyramid_level(patched_mlx, monkeypatch):
    # Requesting 10x on a 40x two-level slide must pick level 1 (downsample 4) and
    # disclose ~10.0x -> verifies the magnification slider actually switches levels
    # (the single-level fakes elsewhere never exercise this).
    _force_ram_gib(monkeypatch, 32)
    slide = _FakeSlide(
        level_dimensions=((8000, 8000), (2000, 2000)),
        level_downsamples=(1.0, 4.0),
    )
    monkeypatch.setattr("openslide.OpenSlide", lambda path: slide)
    out = MagicMock()
    out.text = "Findings."
    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **k: out)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.select_slider(key="wsi_mag").set_value(10).run()
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert any("~10.0x" in c.value for c in at.caption)


def test_wsi_sparse_tissue_reduces_patch_count(patched_mlx, monkeypatch):
    # Tissue only on the left third -> the nine-tile grid is filtered to three
    # patches, even though eight were requested. Mirrors the live run on the CMU-1
    # slide (8 requested, 3 tissue patches sampled, caption "3 patches sampled").
    _force_ram_gib(monkeypatch, 32)
    thumb = np.full((800, 800, 3), 255, dtype=np.uint8)
    thumb[:, :250, 0], thumb[:, :250, 2] = 150, 140
    monkeypatch.setattr(
        "openslide.OpenSlide",
        lambda path: _FakeSlide(thumbnail=Image.fromarray(thumb, "RGB")),
    )
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Findings."
    monkeypatch.setattr("mlx_vlm.generate", lambda *a, **k: out)
    at = AppTest.from_file(APP_PATH).run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.slider(key="wsi_patches").set_value(8).run()  # request 8, only 3 qualify
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 3
    assert any("3 patches sampled" in c.value for c in at.caption)
