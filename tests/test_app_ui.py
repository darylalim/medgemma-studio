import io
import os
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pydicom
import pytest
import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest
from streamlit.testing.v1.element_tree import Column

from streamlit_app import (
    DEFAULT_INSTRUCTION_COMPARE,
    DEFAULT_INSTRUCTION_CT,
    DEFAULT_INSTRUCTION_IMAGE,
    DEFAULT_INSTRUCTION_TEXT,
    DEFAULT_INSTRUCTION_WSI,
    EMPTY_OUTPUT_HINT,
    MODEL_CARD_URL,
    REPETITION_CONTEXT_SIZE,
    REPETITION_PENALTY,
)
from tests.dicom_helpers import dicom_bytes

APP_PATH = str(Path(__file__).parent.parent / "streamlit_app.py")

# AppTest.run() budgets ONE script run in wall-clock time, and its 3s default is far
# too tight for the first run that renders model output in a freshly created venv.
# Streamlit imports its dataframe stack lazily, so the first st.write_stream pulls in
# pandas + pyarrow -- ~350 modules, measured at 16-20s off a cold page cache against
# ~0.03s once warm. So
#   rm -rf .venv && uv sync --locked && uv run pytest
# (what every CI job does -- the runner builds the venv from scratch each time)
# reproducibly fails whichever test generates first. It reproduces on the previous
# lockfile too, so it is latent rather than newly introduced; GitHub's runners have
# stayed under 3s so far, but that margin is nothing to rely on.
#
# Note the cost rides on the GENERATION path, not the base render: rendering the script
# alone is ~3s cold and does not import pandas, which is why the warmup below has to
# click Run rather than just call .run().
#
# Because the cost is one-time, charge it to a one-time warmup instead of to whichever
# test happens to run first. That keeps the per-run bound tight, which matters because
# the bound is per test, not per suite: at 60s a hang would cost 90 tests x 60s = 90
# minutes and CI would hit ci.yml's 15-minute job cap with no pytest report at all --
# strictly worse than the 3s default it replaced. At 8s that worst case is ~12 minutes
# and still reports, but the margin has thinned: this was sized at 82 tests, and the
# count grows while the cap does not. Re-check it, not just the multiplier, when it
# next moves. 8s is ~3x the slowest test measured here (~2.5s, and that one is
# slow for its own reasons -- decoding a deliberately invalid image -- not from this
# import); a typical warmed run is ~0.03s. Raising it trades that headroom against the
# multiplier above, so re-do the arithmetic before nudging it up.
APP_RUN_TIMEOUT = 8

# The warmup pays the pandas/pyarrow import, so it alone needs the generous bound.
APP_WARMUP_TIMEOUT = 60


def _app_test(timeout: float = APP_RUN_TIMEOUT) -> AppTest:
    """Build this app's AppTest with a cold-start-safe timeout.

    Every AppTest in the suite is constructed here so no call site can silently fall
    back to the 3s default -- a bare ``AppTest.from_file`` passes warm and fails cold,
    i.e. green locally and red in CI. Pinned by ``TestAppTestHarness`` in
    tests/test_streamlit_app.py, which scans every module under tests/.
    """
    return AppTest.from_file(APP_PATH, default_timeout=timeout)


@pytest.fixture(scope="session", autouse=True)
def _warm_streamlit_once():
    """Import Streamlit's dataframe stack once, before any test is timed.

    Drives the cheapest generation path there is (Ask: no image, one mocked chunk) far
    enough to render model output, which is what triggers the pandas/pyarrow import.
    Without this the charge lands on whichever test generates first, which is why a
    cold venv fails it.

    Session-scoped, so it cannot use the function-scoped ``monkeypatch`` fixture;
    ``pytest.MonkeyPatch.context()`` gives the same undo on exit. A hang in this path
    surfaces here as one ``APP_WARMUP_TIMEOUT`` failure that errors the module
    immediately, rather than 82 tests each burning their own timeout. ``AppTest.run``
    captures app exceptions on ``at.exception`` rather than raising, so a timeout is
    the only thing that fails this fixture.
    """
    chunk = MagicMock()
    chunk.text = "warmup"
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("mlx_vlm.load", lambda *a, **k: (MagicMock(), MagicMock()))
        mp.setattr("mlx_vlm.utils.load_config", lambda *a, **k: {})
        mp.setattr("mlx_vlm.prompt_utils.apply_chat_template", lambda *a, **k: "prompt")
        mp.setattr("mlx_vlm.stream_generate", lambda *a, **k: iter([chunk]))
        at = _app_test(timeout=APP_WARMUP_TIMEOUT)
        at.run()
        at.text_input(key="ask_prompt").set_value("warmup").run()
        at.button(key="ask_run").click().run()
    # Drop the warmup's cached mock model and its stored result so neither leaks into
    # the first real test (_clear_caches does this per test, but runs after this one).
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture(autouse=True)
def _clear_caches():
    """Reset Streamlit's resource/data caches before each test. AppTest does not
    clear them between runs, so the @st.cache_data RAM value (which _force_ram_gib
    varies per test) and the @st.cache_resource model would otherwise leak across
    tests in a session and make slice/patch-cap assertions order-dependent."""
    st.cache_data.clear()
    st.cache_resource.clear()
    yield


def _patch_stream(monkeypatch, generate_mock):
    """Patch ``mlx_vlm.stream_generate`` from an old-style ``generate`` mock.

    ``run_model`` now streams via ``stream_generate`` rather than calling
    ``generate``, but the tests keep expressing the mock as the ``generate``
    contract: ``(*a, **k) -> object with .text``. This wraps such a mock as a
    ``stream_generate`` that yields that object as a single chunk, so ``run_model``'s
    ``chunk.text`` + ``st.write_stream`` concatenation reproduces the full text.
    ``stream_generate`` shares ``generate``'s ``(model, processor, prompt, image,
    **kwargs)`` call shape, so the ``gen_args``/``gen_kwargs`` captures still hold; a
    mock that raises still raises (when the generator is consumed by write_stream)."""

    def _stream(*args, **kwargs):
        yield generate_mock(*args, **kwargs)

    monkeypatch.setattr("mlx_vlm.stream_generate", _stream)


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
    _patch_stream(monkeypatch, lambda *a, **k: output)
    return output


@pytest.fixture
def app(patched_mlx):
    """A freshly-run AppTest with the model mocked."""
    return _app_test().run()


@pytest.fixture
def png_bytes():
    """A minimal valid in-memory PNG for file-upload tests."""
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")
    return buf.getvalue()


def _dicom_bytes(*args, **kwargs):
    """Raw bytes of an in-memory CT DICOM slice, for file_uploader.upload()."""
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
    """Upload two slices to the CT tab (chained: AppTest replaces, not appends)."""
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


_TAB_INDEX = {"ask": 0, "cxr": 1, "ct": 2, "wsi": 3}


def _tab(at, tab):
    """One tab's block. Scope content lookups to it rather than the whole app: the
    sidebar panel's captions say "whole-slide" and "Limited memory" too, and the CT
    and WSI tabs share their "Limited memory detected" phrasing, so an app-wide
    ``at.caption`` search passes on another element's text."""
    return at.tabs[_TAB_INDEX[tab]]


def _workspace(at, tab):
    """A tab's ``(inputs, output)`` workspace columns, found by structure.

    The workspace is the first block under the tab (pre-order) whose children are
    all columns -- the ``st.columns`` row itself, so a pair nested inside the inputs
    column (the CXR comparison) is never mistaken for it, and neither is a pair that
    might one day nest in the output column. Weights can't tell them apart: at
    ``WORKSPACE_SPEC = [1, 1]`` the workspace and any nested ``st.columns(2)`` are
    both (0.5, 0.5).
    """
    for node in _tab(at, tab):
        children = list(getattr(node, "children", {}).values())
        if children and all(isinstance(c, Column) for c in children):
            assert len(children) == 2, f"{tab} workspace has {len(children)} columns"
            return children[0], children[1]
    raise AssertionError(f"{tab} tab is no longer an inputs/output workspace")


def _position(block, kind, match=lambda node: True):
    """Document-order index of the first ``kind`` node under ``block`` (``Block``
    iterates pre-order depth-first, i.e. in the order the elements render)."""
    for i, node in enumerate(block):
        if type(node).__name__ == kind and match(node):
            return i
    raise AssertionError(f"no {kind} under this block")


# --------------------------------------------------------------------------- #
# Layout / shared
# --------------------------------------------------------------------------- #


def test_title_renders(app):
    assert not app.exception
    # In the sidebar app panel now -- moved, not duplicated, so the main area opens
    # on the disclaimer and the tabs rather than spending a heading's height on it.
    assert [t.value for t in app.sidebar.title] == ["MedGemma Studio"]
    assert not app.main.title


def test_in_app_disclaimer_renders(app):
    # Safety-critical: a research-only / not-a-medical-device notice must be visible in
    # the app itself, not only in the README (app users never see the README). Guard it
    # from silent removal, mirroring TestLicense's README-disclaimer guard.
    import streamlit_app

    # In the MAIN area specifically: the sidebar collapses, and starts collapsed on a
    # phone, so a notice that migrated there with the rest of the app panel could
    # render and still go unseen.
    warnings = [w.value for w in app.main.warning]
    assert streamlit_app.DISCLAIMER_TEXT in warnings, (
        "in-app research-only disclaimer is missing from the top of the app"
    )
    assert streamlit_app.DISCLAIMER_TEXT not in [w.value for w in app.sidebar.warning]
    low = streamlit_app.DISCLAIMER_TEXT.lower()
    assert "not a medical device" in low
    assert "not medical advice" in low


def test_four_tabs_render(app):
    # Tab labels carry inline Material Symbol icons; AppTest reports the raw label
    # markup (the :material/...: token), not the rendered glyph.
    assert [t.label for t in app.tabs] == [
        ":material/forum: Ask",
        ":material/radiology: Chest X-ray",
        ":material/readiness_score: Computed tomography",
        ":material/biotech: Pathology (WSI)",
    ]


def test_no_expander_on_first_render(app):
    # Each tab has a collapsed "Model settings" expander, but the "Thinking trace"
    # expander appears only after a thinking response.
    assert not any(e.label == "Thinking trace" for e in app.expander)


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


def test_model_settings_live_in_a_collapsed_expander_per_tab(app):
    # The persona + thinking toggle are tucked into a "Model settings" expander so the
    # primary prompt -> upload -> Run flow leads each tab; one such expander per tab.
    assert len([e for e in app.expander if e.label == "Model settings"]) == 4


def test_thinking_toggles_are_independent(app):
    app.toggle(key="ask_thinking").set_value(True).run()
    assert app.toggle(key="ask_thinking").value is True
    assert app.toggle(key="cxr_thinking").value is False
    assert app.toggle(key="ct_thinking").value is False


# --------------------------------------------------------------------------- #
# Workspace layout: sidebar app panel + per-tab inputs | output columns
# --------------------------------------------------------------------------- #


def test_page_uses_the_wide_layout():
    # Each workspace column assumes the wide page. Under "centered" (~730px) each
    # column would get ~340px and each study in the CXR comparison pair ~160px, and no
    # AppTest assertion would notice: AppTest doesn't expose page config, so the
    # call is read from source.
    import ast

    tree = ast.parse(Path(APP_PATH).read_text())
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "attr", None) == "set_page_config"
    ]
    assert len(calls) == 1
    layout = {kw.arg: kw.value for kw in calls[0].keywords}.get("layout")
    assert isinstance(layout, ast.Constant) and layout.value == "wide"


def test_sidebar_is_an_app_panel_without_controls(app):
    assert any(MODEL_CARD_URL in m.value for m in app.sidebar.markdown)
    # Global facts only. st.tabs doesn't tell the server which tab is showing, so a
    # per-tab control moved into the sidebar either shows all four tabs' copies at
    # once or needs on_change="rerun" plus hand-rolled persistence for the hidden
    # ones -- the per-tab controls belong in each tab's inputs column.
    for kind in (
        "button",
        "text_input",
        "text_area",
        "toggle",
        "slider",
        "file_uploader",
        "segmented_control",
    ):
        assert not getattr(app.sidebar, kind), f"the sidebar grew a {kind}"


@pytest.mark.parametrize(
    ("gib", "cap", "caption"),
    [
        (32, "20", "up to 20 slices or patches (default 10)"),
        (16, "2", "Limited memory"),
    ],
)
def test_sidebar_reports_the_ram_derived_cap(
    patched_mlx, monkeypatch, gib, cap, caption
):
    # The one place the cap is stated outright: before, a small Mac saw only a bare
    # "Limited memory detected" caption in the CT/WSI tabs, with nothing naming the
    # memory it was measured against.
    _force_ram_gib(monkeypatch, gib)
    at = _app_test().run()
    assert not at.exception
    assert {m.label: m.value for m in at.sidebar.metric} == {
        "Memory": f"{gib} GiB",
        "Per-run cap": cap,
    }
    assert any(caption in c.value for c in at.sidebar.caption)


@pytest.mark.parametrize("tab", ["ask", "cxr", "ct", "wsi"])
def test_each_tab_is_an_inputs_output_workspace(app, tab):
    inputs, output = _workspace(app, tab)
    assert [b.key for b in inputs.button] == [f"{tab}_run"]
    assert [e.label for e in inputs.expander] == ["Model settings"]
    # Before any run the output column says what will appear there: in the wide
    # layout a blank right half reads as a broken page.
    assert not output.button
    assert [c.value for c in output.caption] == [EMPTY_OUTPUT_HINT]


def test_ask_response_renders_beside_the_inputs(patched_mlx):
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("Why?").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    inputs, output = _workspace(at, "ask")
    assert "### Response" in [m.value for m in output.markdown]
    assert "### Response" not in [m.value for m in inputs.markdown]
    assert EMPTY_OUTPUT_HINT not in [c.value for c in output.caption]


def test_missing_question_warning_renders_under_run(app):
    # Next to the button that raised it, not across the page in the output column.
    app.button(key="ask_run").click().run()
    inputs, output = _workspace(app, "ask")
    assert "Enter a question first." in [w.value for w in inputs.warning]
    assert not output.warning
    # And not beside "the response appears here after you click Run" -- the user
    # just did. empty_hint keys on the raw click, not on the run_requested gate.
    assert EMPTY_OUTPUT_HINT not in [c.value for c in output.caption]


def test_failed_run_shows_its_error_without_the_empty_hint(patched_mlx, monkeypatch):
    # empty_hint=not go: an error followed by "the response appears here after you
    # click Run" would contradict the click that just failed.
    def _raise(*a, **k):
        raise RuntimeError("model exploded")

    _patch_stream(monkeypatch, _raise)
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("Why?").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    _, output = _workspace(at, "ask")
    assert [e.value for e in output.error] == ["Inference failed: model exploded"]
    assert EMPTY_OUTPUT_HINT not in [c.value for c in output.caption]


def test_cxr_preview_renders_under_run_with_the_report_beside_it(
    patched_mlx, png_bytes
):
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Describe.").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    inputs, output = _workspace(at, "cxr")
    assert [i.captions for i in inputs.image] == [["Uploaded image"]]
    # Under Run, not above it: a portrait radiograph is taller than the viewport's
    # spare height, so a preview above the button pushed Run off-screen as soon as a
    # study was attached.
    run_at = _position(inputs, "Button", lambda n: n.key == "cxr_run")
    assert run_at < _position(inputs, "Image")
    assert "### Response" in [m.value for m in output.markdown]
    assert not output.image


def test_cxr_localization_draws_in_the_output_column(patched_mlx, png_bytes):
    # The annotated image IS the answer in this mode, so it goes where answers go;
    # the unannotated study stays in the inputs column beside it for comparison.
    patched_mlx.text = (
        '```json\n[{"box_2d": [100, 100, 500, 500], "label": "right clavicle"}]\n```'
    )
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Where is the right clavicle?").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    inputs, output = _workspace(at, "cxr")
    assert [i.captions for i in output.image] == [["Localized anatomy"]]
    assert "### Detected structures" in [m.value for m in output.markdown]
    assert [i.captions for i in inputs.image] == [["Uploaded image"]]


def test_ct_slices_render_under_the_inputs_with_the_report_beside_them(
    patched_mlx, monkeypatch
):
    # The imagery the model saw lives in the inputs column, so the report column
    # opens directly on the truncation warning and the text it qualifies -- in the
    # single-column layout a full-width slice plus the gallery sat between them.
    _force_ram_gib(monkeypatch, 32)
    out = MagicMock()
    out.text = "Two contiguous slices of the liver."
    out.finish_reason = "length"
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
    at.text_input(key="ct_prompt").set_value("Any lesions?").run()
    _upload_ct_pair(at)
    at.button(key="ct_run").click().run()
    assert not at.exception
    inputs, output = _workspace(at, "ct")
    assert [i.captions for i in inputs.image] == [["Sample windowed slice (1 of 2)"]]
    assert [e.label for e in inputs.expander] == [
        "Model settings",
        "View all 2 windowed slices",
    ]
    assert not output.image
    assert "Two contiguous slices of the liver." in [m.value for m in output.markdown]
    assert _position(output, "Warning") < _position(output, "Markdown")
    assert any("token limit" in w.value for w in output.warning)


def test_wsi_overview_renders_under_the_inputs_with_the_report_beside_it(
    patched_mlx, patched_openslide, monkeypatch
):
    _force_ram_gib(monkeypatch, 32)
    out = MagicMock()
    out.text = "Moderately differentiated adenocarcinoma."
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    assert not at.exception
    inputs, output = _workspace(at, "wsi")
    captions = [i.captions[0] for i in inputs.image]
    assert captions[0] == "Tissue overview"
    assert captions[1].startswith("Sample patch (1 of ")
    assert any("patches sampled at ~" in c.value for c in inputs.caption)
    assert not output.image
    assert "Moderately differentiated adenocarcinoma." in [
        m.value for m in output.markdown
    ]


# --------------------------------------------------------------------------- #
# Ask tab (text-only Q&A)
# --------------------------------------------------------------------------- #


def test_ask_default_instruction_is_text_persona(app):
    assert app.text_area(key="ask_instruction").value == DEFAULT_INSTRUCTION_TEXT


def test_ask_thinking_defaults_off(app):
    assert app.toggle(key="ask_thinking").value is False


def test_ask_run_stays_enabled_without_prompt(app):
    # Deliberately NOT disabled=not prompt. A disabled button dispatches no click
    # event, so with an uncommitted text_input the user's first click is spent
    # blurring the field and the app appears to do nothing. run_requested() validates
    # the prompt at click time instead; this pins the button staying live.
    assert app.button(key="ask_run").disabled is False


def test_ask_run_enabled_with_prompt(app):
    app.text_input(key="ask_prompt").set_value("What causes effusion?").run()
    assert app.button(key="ask_run").disabled is False


def test_ask_empty_prompt_click_warns_instead_of_running(app):
    app.button(key="ask_run").click().run()
    assert not app.exception
    assert any("Enter a question first." in w.value for w in app.warning)
    assert "### Response" not in [m.value for m in app.markdown]


def test_ask_whitespace_prompt_click_warns_instead_of_running(app):
    app.text_input(key="ask_prompt").set_value("   ").run()
    app.button(key="ask_run").click().run()
    assert not app.exception
    assert any("Enter a question first." in w.value for w in app.warning)
    assert "### Response" not in [m.value for m in app.markdown]


def test_ask_response_renders(app):
    app.text_input(key="ask_prompt").set_value("What is a fracture?").run()
    app.button(key="ask_run").click().run()
    assert not app.exception
    markdowns = [m.value for m in app.markdown]
    assert "### Response" in markdowns
    assert "No acute findings." in markdowns


def test_ask_streams_deltas_and_renders_answer_once(patched_mlx, monkeypatch):
    # run_model streams incremental stream_generate deltas and concatenates them; the
    # live stream placeholder is cleared, so the persisted render shows the full answer
    # exactly once (not duplicated by the live stream + the render-outside-the-gate).
    def _stream(*a, **k):
        for delta in ("No ", "acute ", "findings."):
            chunk = MagicMock()
            chunk.text = delta
            yield chunk

    monkeypatch.setattr("mlx_vlm.stream_generate", _stream)
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("What is a fracture?").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "No acute findings." in markdowns  # deltas concatenated
    assert sum(m == "No acute findings." for m in markdowns) == 1  # rendered once


def test_ask_empty_generation_renders_without_crashing(patched_mlx, monkeypatch):
    # A model that emits zero tokens (immediate EOS) must degrade gracefully:
    # st.write_stream returns "" and the persisted render shows an empty answer under
    # "### Response" — not a crash and not an "Inference failed" error.
    def _empty_stream(*a, **k):
        return
        yield  # unreachable — the yield only makes this a zero-chunk generator

    monkeypatch.setattr("mlx_vlm.stream_generate", _empty_stream)
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("hi").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    assert "### Response" in [m.value for m in at.markdown]  # answer section renders
    assert not at.error  # empty output is benign, not an error


def test_truncated_generation_warns_that_output_is_incomplete(patched_mlx):
    # A run cut off at max_new_tokens renders byte-identically to one that finished
    # -- the answer just stops -- so stream_generate's finish_reason on the final
    # chunk is the only thing that can tell the reader. Without this the clipped
    # report reads as a complete one.
    patched_mlx.text = "The lungs are clear and the cardiomediastinal"
    patched_mlx.finish_reason = "length"
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("Describe the film.").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    assert any("token limit" in w.value for w in at.warning)


def test_completed_generation_does_not_warn_about_truncation(patched_mlx):
    # The other half: only "length" warns. A run that stopped on EOS is complete,
    # and a warning on every answer would train the reader to ignore it.
    patched_mlx.text = "No acute findings."
    patched_mlx.finish_reason = "stop"
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("Describe the film.").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    assert not any("token limit" in w.value for w in at.warning)


def test_ask_thinking_trace_renders(patched_mlx):
    patched_mlx.text = "<unused94>thought\nLet me reason.<unused95>Final answer."
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("Why?").run()
    at.toggle(key="ask_thinking").set_value(True).run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "Let me reason." in markdowns  # the thinking trace
    assert "Final answer." in markdowns  # the parsed answer


def test_ask_thinking_no_markers_no_expander(patched_mlx):
    patched_mlx.text = "Just a plain reply without markers."
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("Why?").run()
    at.toggle(key="ask_thinking").set_value(True).run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    # no spurious "Thinking trace" expander (the "Model settings" ones don't count)
    assert not any(e.label == "Thinking trace" for e in at.expander)
    assert "Just a plain reply without markers." in [m.value for m in at.markdown]


def test_ask_inference_failure_renders_error(patched_mlx, monkeypatch):
    def _raise(*a, **k):
        raise RuntimeError("model exploded")

    _patch_stream(monkeypatch, _raise)
    at = _app_test().run()
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
    _patch_stream(monkeypatch, lambda *a, **k: captured.update(gen_kwargs=k) or out)
    at = _app_test().run()
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
    _patch_stream(monkeypatch, lambda *a, **k: captured.update(gen_args=a) or out)
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("What is a fracture?").run()
    at.button(key="ask_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 0
    assert captured["gen_args"][3] is None  # no image -> None passed positionally


def test_ask_result_persists_across_rerun(patched_mlx):
    # Regression for the vanish-on-rerun bug: a result must survive a later widget
    # interaction that does not re-click Run (before the session-state fix, the
    # early-return on a False button wiped the response).
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("What is a fracture?").run()
    at.button(key="ask_run").click().run()
    assert "No acute findings." in [m.value for m in at.markdown]
    at.toggle(key="ask_thinking").set_value(True).run()  # unrelated rerun, no click
    assert not at.exception
    assert "No acute findings." in [m.value for m in at.markdown]


def test_ask_does_not_rerun_inference_on_unrelated_rerun(patched_mlx, monkeypatch):
    # The persisted result must be served from session_state, NOT recomputed: a
    # rerun without a fresh Run click must not call generate() again.
    calls = []
    out = MagicMock()
    out.text = "Cached answer."
    _patch_stream(monkeypatch, lambda *a, **k: calls.append(1) or out)
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("Why?").run()
    at.button(key="ask_run").click().run()
    assert len(calls) == 1
    at.toggle(key="ask_thinking").set_value(True).run()  # no new click
    assert not at.exception
    assert len(calls) == 1  # inference not re-run
    assert "Cached answer." in [m.value for m in at.markdown]


def test_ask_stale_result_cleared_when_prompt_changes(patched_mlx):
    # Persistence must not outlive the inputs: editing the prompt without clicking
    # Run drops the now-stale answer (its stored signature no longer matches).
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("What is a fracture?").run()
    at.button(key="ask_run").click().run()
    assert "No acute findings." in [m.value for m in at.markdown]
    at.text_input(key="ask_prompt").set_value("What is pneumonia?").run()  # no re-run
    assert not at.exception
    assert "No acute findings." not in [m.value for m in at.markdown]
    assert "### Response" not in [m.value for m in at.markdown]
    # The stale result is dropped with a visible hint, not a silent vanish.
    assert any("Inputs changed" in i.value for i in at.info)


def test_ask_result_goes_stale_when_instruction_changes(patched_mlx):
    # The system instruction is fed to the model as the system message, so it is a
    # run-defining input: editing the persona must strand the old answer behind the
    # hint exactly as editing the prompt does. (Before the sig carried it, the answer
    # re-rendered as current under a persona that never produced it.)
    at = _app_test().run()
    at.text_input(key="ask_prompt").set_value("What is a fracture?").run()
    at.button(key="ask_run").click().run()
    assert "No acute findings." in [m.value for m in at.markdown]
    at.text_area(key="ask_instruction").set_value("You are a pediatric radiologist.")
    at.run()  # persona edited, no re-click
    assert not at.exception
    assert "No acute findings." not in [m.value for m in at.markdown]
    assert any("Inputs changed" in i.value for i in at.info)


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
    assert not any("ignored in this mode" in c.value for c in _tab(app, "cxr").caption)
    app.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    app.toggle(key="cxr_localize").set_value(True).run()
    assert any("ignored in this mode" in c.value for c in _tab(app, "cxr").caption)
    # Pin the direction word too: "Model settings" renders *below* this caption, so
    # a substring check on "ignored in this mode" alone would not notice the caption
    # pointing the wrong way after a reorder.
    captions = [c.value for c in _tab(app, "cxr").caption]
    assert any("instruction below is ignored" in c for c in captions)


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
    assert not any("Comparison mode" in c.value for c in _tab(app, "cxr").caption)
    app.file_uploader(key="cxr_image1").upload("a.png", png_bytes, "image/png").run()
    app.file_uploader(key="cxr_image2").upload("b.png", png_bytes, "image/png").run()
    assert any("Comparison mode" in c.value for c in _tab(app, "cxr").caption)


def test_cxr_comparison_previews_studies_side_by_side(app, png_bytes):
    # A single image previews at the inputs column's full width (no nested columns);
    # a second image switches to a side-by-side st.columns(2) pair so a longitudinal
    # pair reads at a glance. Scoped to the inputs column: every tab is itself a
    # two-column workspace now, so a bare `app.columns` is never empty.
    # (A Column's .columns starts with the column itself -- Block.__iter__ yields
    # self first -- so the nested pair is everything after index 0.)
    app.file_uploader(key="cxr_image1").upload("a.png", png_bytes, "image/png").run()
    inputs, _ = _workspace(app, "cxr")
    assert not inputs.columns[1:]
    app.file_uploader(key="cxr_image2").upload("b.png", png_bytes, "image/png").run()
    assert not app.exception
    inputs, _ = _workspace(app, "cxr")
    first, second = inputs.columns[1:]
    assert [i.captions for i in first.image] == [["First image"]]
    assert [i.captions for i in second.image] == [["Second image"]]


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
    _patch_stream(monkeypatch, lambda *a, **k: captured.update(gen_args=a) or out)
    at = _app_test().run()
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
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Where is the right clavicle?").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" in markdowns
    assert any("right clavicle" in m for m in markdowns)
    # The label is listed plainly; the raw normalized coords are no longer shown
    # (the boxes are drawn on the image above, so coords would just be cryptic).
    assert not any("[100, 100, 500, 500]" in m for m in markdowns)
    assert "### Response" not in markdowns  # localization replaces the text view


def test_cxr_localization_passes_square_image_to_model(patched_mlx, monkeypatch):
    captured = {}
    out = MagicMock()
    out.text = '```json\n[{"box_2d": [0, 0, 1000, 1000], "label": "frame"}]\n```'
    _patch_stream(
        monkeypatch,
        lambda *a, **k: captured.update(gen_args=a, gen_kwargs=k) or out,
    )
    buf = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buf, format="PNG")  # non-square -> padding visible
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Localize the frame").run()
    at.file_uploader(key="cxr_image1").upload(
        "wide.png", buf.getvalue(), "image/png"
    ).run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    assert captured["gen_args"][3][0].size == (20, 20)  # padded to a square
    # The localization path opts OUT of the loop-guard penalty: it taxes the structural
    # tokens every extra box repeats, truncating the JSON list. Every other path still
    # asserts the penalty is passed, so this stays a deliberate exception.
    assert "repetition_penalty" not in captured["gen_kwargs"]
    assert "repetition_context_size" not in captured["gen_kwargs"]


def test_cxr_localization_warns_when_the_box_list_is_truncated(patched_mlx, png_bytes):
    # Pins the placement, not just the flag: the localize branch renders boxes and a
    # label legend but never reaches show_response, so a truncation warning attached
    # to the answer view would be invisible on exactly the path where a JSON list
    # clipped at the cap silently costs structures. Hence above the mode branch.
    patched_mlx.text = (
        '```json\n[{"box_2d": [100, 100, 500, 500], "label": "right clavicle"}]\n```'
    )
    patched_mlx.finish_reason = "length"
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Locate everything").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" in markdowns  # the localize view, not the answer
    assert "### Response" not in markdowns
    assert any("token limit" in w.value for w in at.warning)


def test_cxr_localization_no_boxes_warns(patched_mlx, png_bytes):
    patched_mlx.text = "I could not localize that structure."
    at = _app_test().run()
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
    at = _app_test().run()
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
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Where is the femur?").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" in markdowns
    assert any("femur" in m for m in markdowns)
    # (the persistent disclaimer warning is expected; the no-boxes fallback is not)
    assert "No bounding boxes were returned." not in [w.value for w in at.warning]


def test_cxr_comparison_passes_both_images(patched_mlx, monkeypatch, png_bytes):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "The second study shows interval improvement."
    _patch_stream(monkeypatch, lambda *a, **k: captured.update(gen_args=a) or out)
    at = _app_test().run()
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
    _patch_stream(monkeypatch, lambda *a, **k: captured.update(gen_kwargs=k) or out)
    at = _app_test().run()
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
    _patch_stream(
        monkeypatch, lambda *a, **k: captured.update(gen_kwargs=k) or patched_mlx
    )
    at = _app_test().run()
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
    assert any(e.label == "Thinking trace" for e in at.expander)
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
    assert not any("Comparison mode" in c.value for c in _tab(app, "cxr").caption)
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
    assert not any("Comparison mode" in c.value for c in _tab(app, "cxr").caption)


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
    _patch_stream(monkeypatch, lambda *a, **k: captured.update(gen_kwargs=k) or out)
    at = _app_test().run()
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
    # (the persistent disclaimer warning is expected; the no-boxes fallback is not)
    assert "No bounding boxes were returned." not in [w.value for w in at.warning]


def test_cxr_comparison_sends_unpadded_images(patched_mlx, monkeypatch):
    captured = {}
    out = MagicMock()
    out.text = "Comparison."
    _patch_stream(monkeypatch, lambda *a, **k: captured.update(gen_args=a) or out)
    buf = io.BytesIO()
    Image.new("RGB", (20, 10)).save(buf, format="PNG")
    wide_png = buf.getvalue()
    at = _app_test().run()
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
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
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


def test_cxr_result_persists_across_rerun(patched_mlx, monkeypatch, png_bytes):
    calls = []
    out = MagicMock()
    out.text = "No acute findings."
    _patch_stream(monkeypatch, lambda *a, **k: calls.append(1) or out)
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Describe this X-ray").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.button(key="cxr_run").click().run()
    assert "No acute findings." in [m.value for m in at.markdown]
    assert len(calls) == 1
    at.toggle(key="cxr_thinking").set_value(
        True
    ).run()  # sig-preserving rerun, no click
    assert not at.exception
    assert "No acute findings." in [m.value for m in at.markdown]
    assert len(calls) == 1  # served from session_state, not recomputed


def test_cxr_localization_persists_across_rerun(patched_mlx, png_bytes):
    # The drawn annotation + structure list must survive a rerun too (it is stored
    # finished in session_state, so it is not redrawn on every rerun).
    patched_mlx.text = (
        '```json\n[{"box_2d": [100, 100, 500, 500], "label": "right clavicle"}]\n```'
    )
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Where is the right clavicle?").run()
    at.file_uploader(key="cxr_image1").upload("xray.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert "### Detected structures" in [m.value for m in at.markdown]
    at.toggle(key="cxr_thinking").set_value(True).run()  # unrelated rerun, no click
    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert "### Detected structures" in markdowns
    assert any("right clavicle" in m for m in markdowns)


def test_cxr_stale_text_result_cleared_when_second_image_added(patched_mlx, png_bytes):
    # A single-image answer must not linger once a second image switches the tab to
    # comparison mode (the result's signature includes the second upload).
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Describe this X-ray").run()
    at.file_uploader(key="cxr_image1").upload("a.png", png_bytes, "image/png").run()
    at.button(key="cxr_run").click().run()
    assert "### Response" in [m.value for m in at.markdown]
    at.file_uploader(key="cxr_image2").upload("b.png", png_bytes, "image/png").run()
    assert not at.exception
    assert "No acute findings." not in [m.value for m in at.markdown]
    assert "### Response" not in [m.value for m in at.markdown]
    assert any("Inputs changed" in i.value for i in at.info)  # dropped with a hint


def test_cxr_stale_localization_cleared_when_localize_toggled_off(
    patched_mlx, png_bytes
):
    # The drawn annotation is stale once localization is turned off without re-running.
    patched_mlx.text = (
        '```json\n[{"box_2d": [100, 100, 500, 500], "label": "rib"}]\n```'
    )
    at = _app_test().run()
    at.text_input(key="cxr_prompt").set_value("Where is the rib?").run()
    at.file_uploader(key="cxr_image1").upload("x.png", png_bytes, "image/png").run()
    at.toggle(key="cxr_localize").set_value(True).run()
    at.button(key="cxr_run").click().run()
    assert "### Detected structures" in [m.value for m in at.markdown]
    at.toggle(key="cxr_localize").set_value(False).run()  # no re-run
    assert not at.exception
    assert "### Detected structures" not in [m.value for m in at.markdown]


# --------------------------------------------------------------------------- #
# Computed Tomography tab (DICOM -> windowing -> multi-slice)
# --------------------------------------------------------------------------- #


def test_ct_default_instruction_is_ct_persona(app):
    assert app.text_area(key="ct_instruction").value == DEFAULT_INSTRUCTION_CT


def test_ct_caption_describes_dicom_upload(app):
    assert any("DICOM" in c.value for c in _tab(app, "ct").caption)


def test_ct_slider_present_or_memory_capped(app):
    # The slice slider is RAM-aware; on a very low-memory host it collapses to a
    # fixed 2-slice cap with a caption instead.
    slider_present = "ct_slices" in [w.key for w in app.slider]
    memory_capped = any("Limited memory" in c.value for c in _tab(app, "ct").caption)
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
    _patch_stream(
        monkeypatch,
        lambda *a, **k: captured.update(gen_args=a, gen_kwargs=k) or out,
    )
    at = _app_test().run()
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
    # (On success the run st.rerun()s to shed the streamed duplicate, which also
    # discards the transient preprocessing st.status — the error-path status is
    # asserted in test_ct_invalid_dicom_shows_error, which does not rerun.)


def test_ct_labels_slices_in_prompt(patched_mlx, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(args=a) or "prompt",
    )
    out = MagicMock()
    out.text = "Findings."
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
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
    _patch_stream(
        monkeypatch,
        lambda *a, **k: captured.update(gen_kwargs=k) or patched_mlx,
    )
    at = _app_test().run()
    at.text_input(key="ct_prompt").set_value("Any lesions?").run()
    _upload_ct_pair(at)
    at.toggle(key="ct_thinking").set_value(True).run()
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert captured["gen_kwargs"]["max_tokens"] == 2500  # thinking + CT budget
    assert any(e.label == "Thinking trace" for e in at.expander)  # thinking trace
    markdowns = [m.value for m in at.markdown]
    assert "Reviewing each slice." in markdowns
    assert "No focal lesion." in markdowns


def test_ct_invalid_dicom_shows_error(patched_mlx):
    at = _app_test().run()
    at.text_input(key="ct_prompt").set_value("Describe this").run()
    at.file_uploader(key="ct_files").upload(
        "bad.dcm", b"not-a-dicom", "application/dicom"
    ).run()
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert any("Failed to read DICOM series" in e.value for e in at.error)
    assert "### Response" not in [m.value for m in at.markdown]
    # A read failure drives the st.status into its error state (and expands it so
    # the nested error is visible rather than hidden behind a collapsed status).
    assert at.status[0].state == "error"


def test_ct_subsamples_to_slider_count(patched_mlx, monkeypatch):
    # The slider value (not the upload count) drives how many windowed slices reach
    # the model: upload 6, request 4, expect 4.
    _force_ram_gib(monkeypatch, 32)  # deterministic slider range (max 20)
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Findings."
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
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
    _force_ram_gib(monkeypatch, 32)  # ram_aware_slice_cap -> (default 10, max 20)
    at = _app_test().run()
    assert at.slider(key="ct_slices").value == 10


def test_ct_memory_capped_shows_caption_not_slider(patched_mlx, monkeypatch):
    _force_ram_gib(monkeypatch, 16)  # below base + headroom -> (2, 2): no slider
    at = _app_test().run()
    assert "ct_slices" not in [w.key for w in at.slider]
    assert any("Limited memory" in c.value for c in _tab(at, "ct").caption)


def test_ct_rejects_mixed_series_with_error(patched_mlx):
    at = _app_test().run()
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
    at = _app_test().run()
    at.text_input(key="ct_prompt").set_value("Describe").run()
    at.file_uploader(key="ct_files").upload(
        "vol.dcm", _dicom_bytes(1, 100, frames=3), "application/dicom"
    ).run()
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert any("single-frame" in e.value for e in at.error)


def test_ct_accepts_extensionless_dicom_filenames(patched_mlx, monkeypatch):
    # The CT uploader deliberately passes no ``type=`` filter, unlike the CXR and WSI
    # ones: per-slice exports off a PACS or a study CD are routinely extensionless
    # (``IM_0001``, a bare SOP UID). Adding ``type=["dcm"]`` as a
    # consistency tidy-up would silently start rejecting real series -- and every other
    # DICOM fixture here is named ``a.dcm``, so nothing else would go red. Malformed
    # uploads are load_ct_volume's job (test_ct_invalid_dicom_shows_error), not the
    # uploader's.
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Two contiguous slices of the liver."
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
    at.text_input(key="ct_prompt").set_value("Describe").run()
    at.file_uploader(key="ct_files").upload(
        "IM_0001", _dicom_bytes(1, 100), "application/octet-stream"
    ).upload(
        "1.2.840.10008.5.1.4.1.1.2", _dicom_bytes(2, 200), "application/octet-stream"
    ).run()
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert not at.error
    assert captured["act_kwargs"]["num_images"] == 2
    assert "Two contiguous slices of the liver." in [m.value for m in at.markdown]


def test_ct_result_persists_across_rerun(patched_mlx, monkeypatch):
    _force_ram_gib(monkeypatch, 32)
    calls = []
    out = MagicMock()
    out.text = "Two contiguous slices of the liver."
    _patch_stream(monkeypatch, lambda *a, **k: calls.append(1) or out)
    at = _app_test().run()
    at.text_input(key="ct_prompt").set_value("Any lesions?").run()
    _upload_ct_pair(at)
    at.button(key="ct_run").click().run()
    assert "Two contiguous slices of the liver." in [m.value for m in at.markdown]
    assert len(calls) == 1
    at.toggle(key="ct_thinking").set_value(True).run()  # sig-preserving rerun, no click
    assert not at.exception
    assert "Two contiguous slices of the liver." in [m.value for m in at.markdown]
    assert len(calls) == 1  # served from session_state, not recomputed


def test_ct_gallery_exposes_every_windowed_slice(patched_mlx, monkeypatch):
    # The model numbers its findings by slice, so each windowed slice has to be
    # reachable -- the single preview above the gallery only ever shows slice 1.
    # The gallery renders lazily (on_change="rerun" + .open), so the label alone no
    # longer proves the slices are there: drive the expander open through its key,
    # which is what that key is for, and assert the slices themselves.
    _force_ram_gib(monkeypatch, 32)
    out = MagicMock()
    out.text = "Two contiguous slices of the liver."
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
    at.text_input(key="ct_prompt").set_value("Any lesions?").run()
    _upload_ct_pair(at)
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert any("View all 2 windowed slices" in e.label for e in at.expander)
    captions = [i.captions for i in at.image]
    assert ["SLICE 1", "SLICE 2"] not in captions  # collapsed: body not computed
    at.session_state["ct_gallery"] = True
    at.run()
    assert not at.exception
    assert ["SLICE 1", "SLICE 2"] in [i.captions for i in at.image]  # opened


def test_ct_volume_cache_hits_on_a_second_run(patched_mlx, monkeypatch):
    # cached_ct_volume is scope="session", so its entries live under the session id
    # and are dropped when the session disconnects. If that key were unstable across
    # reruns the cache would miss every single Run and nothing else in the suite would
    # notice -- the app would just silently re-read every DICOM each time. Count the
    # reads instead of trusting it. (Same reason `hash_funcs` keys on file_id: under
    # Streamlit's default hasher the CT key mixes in a stream position both loaders
    # leave at EOF, so it changed every Run and never hit.)
    _force_ram_gib(monkeypatch, 32)
    reads = []
    real_dcmread = pydicom.dcmread

    def _counting_dcmread(*a, **k):
        reads.append(1)
        return real_dcmread(*a, **k)

    monkeypatch.setattr("pydicom.dcmread", _counting_dcmread)
    runs = []
    out = MagicMock()
    out.text = "Two contiguous slices of the liver."
    _patch_stream(monkeypatch, lambda *a, **k: runs.append(1) or out)
    at = _app_test().run()
    at.text_input(key="ct_prompt").set_value("Any lesions?").run()
    _upload_ct_pair(at)
    at.button(key="ct_run").click().run()
    assert not at.exception
    assert (len(runs), len(reads)) == (1, 2)  # the two uploaded slices, read once
    at.button(key="ct_run").click().run()  # identical inputs -> served from the cache
    assert not at.exception
    # runs == 2 is what keeps this honest: the second click really did re-enter the
    # Run block and re-infer, so reads staying at 2 is a cache hit rather than a
    # button that never fired.
    assert (len(runs), len(reads)) == (2, 2)


def test_ct_stale_result_cleared_when_slice_count_changes(patched_mlx, monkeypatch):
    _force_ram_gib(monkeypatch, 32)  # slider default 10, max 20
    out = MagicMock()
    out.text = "Liver findings."
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
    at.text_input(key="ct_prompt").set_value("Any lesions?").run()
    _upload_ct_pair(at)
    at.button(key="ct_run").click().run()
    assert "Liver findings." in [m.value for m in at.markdown]
    at.slider(key="ct_slices").set_value(4).run()  # changes n_slices -> stale, no run
    assert not at.exception
    assert "Liver findings." not in [m.value for m in at.markdown]
    assert any("Inputs changed" in i.value for i in at.info)  # dropped with a hint


# --------------------------------------------------------------------------- #
# Pathology (WSI) tab (slide -> tissue patches -> multi-image)
# --------------------------------------------------------------------------- #


def test_wsi_default_instruction_is_pathology_persona(app):
    assert app.text_area(key="wsi_instruction").value == DEFAULT_INSTRUCTION_WSI


def test_wsi_caption_describes_slide_upload(app):
    assert any("whole-slide" in c.value for c in _tab(app, "wsi").caption)


def test_wsi_run_requires_prompt_and_file(app):
    assert app.button(key="wsi_run").disabled is True
    app.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    assert app.button(key="wsi_run").disabled is True  # prompt but no slide
    app.file_uploader(key="wsi_files").upload(
        "slide.svs", b"x", "application/octet-stream"
    ).run()
    assert app.button(key="wsi_run").disabled is False


def test_wsi_inference_passes_patches(patched_mlx, patched_openslide, monkeypatch):
    _force_ram_gib(monkeypatch, 32)  # deterministic slider range (max 20)
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(act_kwargs=k) or "prompt",
    )
    out = MagicMock()
    out.text = "Moderately differentiated adenocarcinoma."
    _patch_stream(
        monkeypatch,
        lambda *a, **k: captured.update(gen_args=a, gen_kwargs=k) or out,
    )
    at = _app_test().run()
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
    # (On success the run st.rerun()s to shed the streamed duplicate, which also
    # discards the transient preprocessing st.status — the error-path status is
    # asserted in test_wsi_no_tissue_shows_error, which does not rerun.)


def test_wsi_labels_patches_in_prompt(patched_mlx, patched_openslide, monkeypatch):
    _force_ram_gib(monkeypatch, 32)
    captured = {}
    monkeypatch.setattr(
        "mlx_vlm.prompt_utils.apply_chat_template",
        lambda *a, **k: captured.update(args=a) or "prompt",
    )
    out = MagicMock()
    out.text = "Findings."
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
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
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
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
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    assert not at.exception
    # A 10x request on a single-level 40x slide is honestly disclosed as ~40x.
    assert any("sampled at ~40.0x" in c.value for c in _tab(at, "wsi").caption)


def test_wsi_with_thinking_uses_larger_budget(
    patched_mlx, patched_openslide, monkeypatch
):
    _force_ram_gib(monkeypatch, 32)
    captured = {}
    patched_mlx.text = (
        "<unused94>thought\nReviewing each patch.<unused95>No malignancy seen."
    )
    _patch_stream(
        monkeypatch,
        lambda *a, **k: captured.update(gen_kwargs=k) or patched_mlx,
    )
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Any malignancy?").run()
    _upload_slide(at)
    at.toggle(key="wsi_thinking").set_value(True).run()
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert captured["gen_kwargs"]["max_tokens"] == 2500  # thinking + WSI budget
    assert any(e.label == "Thinking trace" for e in at.expander)  # thinking trace
    markdowns = [m.value for m in at.markdown]
    assert "Reviewing each patch." in markdowns
    assert "No malignancy seen." in markdowns


def test_wsi_invalid_slide_shows_error(patched_mlx, monkeypatch):
    def _boom(path):
        raise OSError("not a slide")

    monkeypatch.setattr("openslide.OpenSlide", _boom)
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Describe").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert any("Failed to read slide" in e.value for e in at.error)
    assert "### Response" not in [m.value for m in at.markdown]


def test_wsi_no_tissue_shows_error(patched_mlx, monkeypatch):
    white = Image.new("RGB", (800, 800), (255, 255, 255))
    monkeypatch.setattr("openslide.OpenSlide", lambda path: _FakeSlide(thumbnail=white))
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Describe").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert any("No tissue" in e.value for e in at.error)
    assert "### Response" not in [m.value for m in at.markdown]
    # A no-tissue failure drives the st.status into its error state.
    assert at.status[0].state == "error"


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
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.segmented_control(key="wsi_mag").set_value(10).run()
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert any("~10.0x" in c.value for c in _tab(at, "wsi").caption)


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
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.slider(key="wsi_patches").set_value(8).run()  # request 8, only 3 qualify
    at.button(key="wsi_run").click().run()
    assert not at.exception
    assert captured["act_kwargs"]["num_images"] == 3
    assert any("3 patches sampled" in c.value for c in _tab(at, "wsi").caption)


def test_wsi_result_persists_across_rerun(patched_mlx, patched_openslide, monkeypatch):
    _force_ram_gib(monkeypatch, 32)
    calls = []
    out = MagicMock()
    out.text = "Moderately differentiated adenocarcinoma."
    _patch_stream(monkeypatch, lambda *a, **k: calls.append(1) or out)
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    overview = "Moderately differentiated adenocarcinoma."
    assert overview in [m.value for m in at.markdown]
    assert len(calls) == 1
    at.toggle(key="wsi_thinking").set_value(
        True
    ).run()  # sig-preserving rerun, no click
    assert not at.exception
    assert overview in [m.value for m in at.markdown]
    # The tissue-overview + magnification caption persist too.
    assert any("patches sampled at ~" in c.value for c in _tab(at, "wsi").caption)
    assert len(calls) == 1  # served from session_state, not recomputed


def test_wsi_stale_result_cleared_when_magnification_changes(
    patched_mlx, patched_openslide, monkeypatch
):
    # wsi_sig includes target_mag, so changing the magnification slider without a
    # re-run drops the now-stale result (guards the mag branch of the sig).
    _force_ram_gib(monkeypatch, 32)
    out = MagicMock()
    out.text = "Adenocarcinoma."
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    assert "Adenocarcinoma." in [m.value for m in at.markdown]
    at.segmented_control(key="wsi_mag").set_value(
        40
    ).run()  # default 10 -> 40, no re-run
    assert not at.exception
    assert "Adenocarcinoma." not in [m.value for m in at.markdown]
    assert not any("patches sampled at ~" in c.value for c in _tab(at, "wsi").caption)


def test_wsi_stale_result_cleared_when_patch_count_changes(
    patched_mlx, patched_openslide, monkeypatch
):
    # wsi_sig includes n_patches, so moving the patch-count slider without a re-run
    # drops the now-stale result (guards the patch-count branch of the sig).
    _force_ram_gib(monkeypatch, 32)
    out = MagicMock()
    out.text = "Adenocarcinoma."
    _patch_stream(monkeypatch, lambda *a, **k: out)
    at = _app_test().run()
    at.text_input(key="wsi_prompt").set_value("Describe the slide").run()
    _upload_slide(at)
    at.button(key="wsi_run").click().run()
    assert "Adenocarcinoma." in [m.value for m in at.markdown]
    at.slider(key="wsi_patches").set_value(4).run()  # default 10 -> 4, no re-run
    assert not at.exception
    assert "Adenocarcinoma." not in [m.value for m in at.markdown]
    assert any("Inputs changed" in i.value for i in at.info)  # dropped with a hint
