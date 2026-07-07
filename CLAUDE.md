# CLAUDE.md

## Project

**medgemma-studio** — Streamlit app for analyzing medical text and images using Google MedGemma on Apple Silicon with MLX. Single-file app: `streamlit_app.py`.

## Commands

```bash
uv sync                                 # Install dependencies
uv run streamlit run streamlit_app.py   # Run the app
uv run ruff check .                      # Lint
uv run ruff format .                     # Format
uv run ty check                          # Type check
uv run pytest                            # Run tests
```

When working with Python, invoke the relevant `/astral:<skill>` (`/astral:uv`, `/astral:ty`, `/astral:ruff`).

## Architecture

`main()` loads the model once (`@st.cache_resource load_model()` → `mlx_vlm.load()` + `load_config()` → `(model, processor, config)`; model `mlx-community/medgemma-1.5-4b-it-bf16`) and renders a research-only safety disclaimer (`DISCLAIMER_TEXT`, a persistent `st.warning`) above four `st.tabs`, each delegating to a `@st.fragment`-decorated `render_*_tab` with its own keyed widgets:

- **Ask** (`render_ask_tab`) — text-only Q&A. `DEFAULT_INSTRUCTION_TEXT`.
- **Chest X-ray** (`render_cxr_tab`) — single image; two-image **comparison** (both in one prompt labeled "First/Second image:", previewed in `st.columns(2)`, `DEFAULT_INSTRUCTION_COMPARE`, 600-token budget); or anatomy **localization** (single image only — padded to square, `LOCALIZATION_INSTRUCTION` asks for `[y0,x0,y1,x1]` boxes normalized to `[0,1000]`, then parsed → scaled → drawn → cropped back). Comparison and localization are mutually exclusive.
- **Computed Tomography** (`render_ct_tab`) — DICOM multi-file upload → `load_ct_volume` (HU) → `window_ct_slice` (false-color RGB) → `build_messages` with `"SLICE n"` labels → `run_model`. `DEFAULT_INSTRUCTION_CT`.
- **Pathology (WSI)** (`render_wsi_tab`) — single slide (`.svs/.ndpi/.tif/.tiff`) + magnification `segmented_control` (5/10/20/40×) → `load_wsi_patches` (896px tissue patches) → `build_messages` with `"PATCH n"` labels → `run_model`. `DEFAULT_INSTRUCTION_WSI`.

CT and WSI share a 2000-token budget (2500 with thinking) and the same shape: preprocess inside an `st.status` (narrates phases, resolves `complete`/`error`) then stream generation in the main area below. Slice/patch counts are capped by `ram_aware_slice_cap` (multi-image inference is memory-heavy on unified memory).

**Pure helpers** (no Streamlit/model — unit-tested):
- `parse_response` — splits the `<unused94>`/`<unused95>` thinking trace from the answer
- `build_messages` — chat message list; one image placeholder per image, optional per-image label text
- `get_generation_params` — per-mode full instruction + `max_new_tokens` budget
- `pad_to_square`, `parse_boxes`, `scale_box`, `draw_boxes` — localization geometry (top-left pad; parse fenced/bare JSON boxes; scale `[0,1000]` → pixels; draw labeled boxes)
- `normalize_hu`, `window_ct_slice` — CT windowing; `CT_WINDOWS` (wide/soft-tissue/brain → R/G/B) is the model's **trained** CT format, so it is fixed, not user-tunable
- `subsample_indices` — uniform slice/patch indices, endpoints included
- `load_ct_volume` — reads per-slice DICOMs, sorts by InstanceNumber, `seek(0)`-rewinds each upload before `dcmread`, converts to HU via `apply_rescale`
- `ram_aware_slice_cap` — `(default, max)` slice/patch counts scaled to installed RAM (`(8,16)` on 32 GiB, hard max 64); RAM detection memoized via `_cached_total_ram_gib`
- WSI: `mag_from_mpp`, `effective_magnification`, `pick_level`, `patch_grid` (non-overlapping 896px tiles, partial edges dropped), `tissue_mask` (saturation proxy `max−min(RGB)` excludes glass), `tissue_patches` (keep ≥25% tissue), `mark_patches`, `load_wsi_patches` (OpenSlide loader; `read_region` takes a **level-0** location but a target-level size; raises `ValueError` for unreadable / too-small / no-tissue slides)

**UI helpers** (touch Streamlit):
- `load_uploaded_image` — `image.load()` forces the decode so invalid data fails here, not later at `st.image`
- `run_model` — streams via `mlx_vlm.stream_generate` through `st.write_stream`, returns the accumulated string; passes `REPETITION_PENALTY`/`REPETITION_CONTEXT_SIZE`; `temperature=0` (deterministic)
- `render_thought` — thinking trace → expander, returns the answer
- `tab_settings` — per-tab system instruction + thinking toggle (independent session-state keys) in a collapsed "Model settings" expander

**Result persistence** — Each tab runs inference **only** inside its `Run` block, stores output + `is_thinking` + a `sig` of the run-defining inputs (prompt, `_file_sig`=name+size, localize/compare mode, slice/patch count, magnification) in `st.session_state["{ask,cxr,ct,wsi}_result"]`, and renders it **outside** the button gate via `fresh_result_or_hint(key, live_sig)` — returns the stored dict when the `sig` matches, else shows an `st.info` "Inputs changed…" hint and drops it (so a stale result never mismatches the visible inputs). `thinking` is deliberately excluded from `sig`. On a successful run each tab `st.rerun()`s, shedding the streamed raw copy for the clean persisted render.

## Claude Code hooks

`.claude/settings.json` (shared) runs the same gates automatically; guarded structurally **and** behaviorally (executes each command, asserts exit codes) by `TestHooksConfig`:

- **PreToolUse** (`Edit`/`Write`) — blocks writes to `.env`/`.env.*` (except `.env.example`/`.sample`/`.template`), `.streamlit/secrets.toml`, `uv.lock`. Case-insensitive; **fails closed** if `jq` is absent. Accident-guard only (does not intercept `Bash` writes).
- **PostToolUse** (`Edit`/`Write`) — on `.py`: `ruff check --fix` + `ruff format` (silent) then `ty check` (surfaces errors back to fix). Writes a `.claude/.tests-needed` sentinel (gitignored) on any `.py`/`.toml`/`.claude/settings.json` edit.
- **Stop** — if the sentinel exists, runs `uv run pytest`; clears it on pass, blocks + feeds output back on fail. Skipped on docs-only turns; `stop_hook_active` guards against a stop→fix loop.

Personal overrides go in `.claude/settings.local.json` (gitignored).

## Continuous integration

`.github/workflows/ci.yml` runs the same four gates (`ruff check` · `ruff format --check` · `ty check` · `pytest`) via `uv sync --locked` on a `macos-15` (arm64 — matches the MLX target so `TestMlxVlmContract` exercises the shipped backend), on push to `main` / PR / `workflow_dispatch`. Least-privilege `contents: read` token, 15-min timeout, concurrency-cancel. Guarded by `TestCiWorkflow`; badge in `README.md`. A version bump must be lock-synced (`uv lock`), else `uv sync --locked` fails.

## Releases

`.github/workflows/release.yml` publishes a GitHub Release when a `vX.Y.Z` tag is pushed. Unlike CI it runs on `ubuntu-latest` (no MLX — just verify + publish) with a `contents: write` token, 10-min timeout. It **verifies the tag matches `pyproject.toml`'s `version`** (a `v0.7.6` tag pushed while pyproject still says `0.7.5` fails the job — the tag==version check lives at release time, not as a pytest that would fail between a bump and its tag) then runs `gh release create --generate-notes --verify-tag` (`--verify-tag` refuses a dangling tag) — notes come from the conventional-commit history, so there is **no** hand-maintained `CHANGELOG` to drift. `github.ref_name` reaches the shell via `env`, never interpolated into a `run:` step (same injection guard as CI). Guarded by `TestReleaseWorkflow`; a release badge sits in `README.md`, and `pyproject.toml`'s `[project.urls]` `Changelog` points at the Releases page. **To cut a release:** bump `version` in `pyproject.toml` → `uv lock` → commit → create and push a `vX.Y.Z` tag on that commit (don't also run `gh release create` by hand — the workflow already does).

## Tests

- **`tests/test_streamlit_app.py`** — pure helpers (no Streamlit/model): `load_ct_volume` runs against real in-memory DICOMs; `load_wsi_patches` against a `_FakeSlide` mock. Plus real-asset guards that catch upstream/config drift: `TestMlxVlmContract` (introspects the real mlx-vlm API `run_model`/`load_model` depend on — the **only** guard that catches an mlx-vlm upgrade, since every other test mocks it), `TestThemeConfig` (`.streamlit/config.toml`), `TestHooksConfig` (`.claude/settings.json`), `TestCiWorkflow` (`.github/workflows/ci.yml`), `TestReleaseWorkflow` (`.github/workflows/release.yml` — tag-driven publisher), `TestClaudeMd` (this file — asserts every documented path, `tests/` module, and app-spine symbol stays current), `TestLicense` (the `LICENSE` file ↔ its `pyproject` SPDX declaration ↔ the README License + medical-use Disclaimer sections stay mutually consistent), and `TestDocsMatchSource` (cross-checks CLAUDE.md **and** README.md against the code for the model id, WSI extensions, ruff rule set, and license id — the extensions matched as a delimited token so a `.tif`-vs-`.tiff` prefix collision can't hide a gap), and `TestReadmeAssets` (every repo-relative README link/image resolves — the `docs/screenshot.png` hero, the sample-data guide, and the LICENSE/workflow links — and the README keeps embedding that hero) — so neither doc can silently drift from the code.
- **`tests/test_app_ui.py`** — UI flow via `streamlit.testing.v1.AppTest`; asserts per-mode `num_images`/token budgets reach the mocked `stream_generate`, the mutual-exclusivity guards, result persistence vs staleness, and CT/WSI error `st.status` states.
- **`tests/dicom_helpers.py`** — shared in-memory DICOM builder `dicom_bytes`, imported by **both** test files (aliased `_dicom_bytes` in `test_streamlit_app.py`); a test support module, not a test file.
- Manual-test assets live in `samples/` (gitignored except `samples/README.md`); the suite builds its own in-memory fixtures.

## Screenshots

The README hero (`docs/screenshot.png`) shows the **Chest X-ray** tab analyzing the sample radiograph (`samples/cxr/longitudinal_cxr_before.png` — see `samples/README.md`). Regenerate it with headless **Playwright**, run ephemerally (`uv run --with playwright …`) so it stays **out of the project deps** — it is not a dependency:

- Start the app, then drive the CXR tab: attach the sample via the `input[type="file"]`, ask a plain-analysis prompt (localization boxes are unreliable on this 4B model — don't use them for the hero), Run, and wait for the `Response`.
- **Force `color_scheme="dark"`** (headless Chromium defaults to light; the app follows `prefers-color-scheme`) and use a **viewport taller than the whole app** — Streamlit scrolls *inside* `section[data-testid="stMain"]`, so `full_page` otherwise captures one viewport band and clips the title/response.
- Hide the dev chrome (`stToolbar`/`stHeader`/fullscreen buttons), screenshot, then crop to the X-ray→`Response` region (element bounding boxes) and downscale to ~1200px with Pillow. The full-page intermediate (`docs/screenshot-full.png`) is a throwaway — gitignored.
- `TestReadmeAssets` guards that the hero stays embedded and every repo-relative README link resolves.

## Gotchas

- **AppTest re-execs the script each run** — patch `mlx_vlm.*` (and `openslide.OpenSlide`) at the **source**, not `streamlit_app`; select widgets **by key** (tabs render every widget, so position is ambiguous); chain multiple `.upload()` on one element before a single `.run()`.
- **The success path `st.rerun()`s** — the CT/WSI success-path `st.status` is gone by the time AppTest captures the tree, so assert only the *error* state via `at.status[0].state`.
- **Streamlit API** — use `width="stretch"`, not deprecated `use_container_width`.

## Constraints

- **No multi-turn chat** — single Q&A per interaction.
- **Package management** — uv (`pyproject.toml` + `uv.lock`); no `requirements.txt`.
- **OpenSlide** — `openslide-python` + `openslide-bin` (native lib ships as an arm64/universal2 wheel; no Homebrew). Single-file WSI formats only; `.mrxs` (multi-file) excluded.
- **HF token** — optional, from `.env` via `python-dotenv`; the MLX repo is ungated, so a token only avoids download rate limits.
- **Theme** — clinical theme in `.streamlit/config.toml`: shared `[theme]` + `[theme.light]`/`[theme.dark]` palettes (defining both enables OS/browser auto-switch; a lone `[theme]` locks one mode). Keys validated against the installed Streamlit.
- **Linting** — ruff rule set `E`, `F`, `I`, `UP`, `B`, `SIM`, `C4`; see `[tool.ruff.lint]`.
- **License** — app source is Apache-2.0 (`LICENSE`, declared via `license`/`license-files` in `pyproject.toml`); the downloaded MedGemma weights are **not** redistributed here and are separately governed by Google's Health AI Developer Foundations terms. `README.md` carries a research-only, not-a-medical-device disclaimer.
