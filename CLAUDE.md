# CLAUDE.md

## Project

**medgemma-pipeline** — Analyze medical text, 2D images (e.g. chest X-ray), and 3D CT volumes with the Google MedGemma 1.5 model on Apple Silicon with MLX.

## Commands

```bash
uv sync                           # Install dependencies
uv run streamlit run streamlit_app.py  # Run the app
uv run ruff check .               # Lint
uv run ruff format .              # Format
uv run ty check                   # Type check
uv run pytest                     # Run tests
```

## Architecture

Single-file app (`streamlit_app.py`) with the following structure:

- **Model loading** — `@st.cache_resource` loads `mlx-community/medgemma-1.5-4b-it-bf16` via `mlx_vlm.load()` + `load_config()`. Returns `(model, processor, config)`.
- **Pure helpers** — Extracted for testability (no Streamlit or model):
  - `parse_response(response, is_thinking)` — splits `<unused94>`/`<unused95>` thinking trace from response
  - `build_messages(prompt, system_instruction, images=None, image_labels=None)` — constructs chat message list for `apply_chat_template`, appending one image placeholder per image in `images` (supports any count: text-only, single, comparison pair, or a CT slice stack); when `image_labels` is given, a label text part is interleaved before each image
  - `get_generation_params(has_image, is_thinking, system_instruction, is_localizing=False, is_comparing=False, is_ct=False)` — returns `(full_instruction, max_new_tokens)`; comparison uses a 600-token budget, CT a 2000-token budget (2500 with thinking), each keeping the editable instruction
  - `pad_to_square(image)` — top-left pads an image to a square so localization coordinates map back without an offset
  - `parse_boxes(response)` — extracts `[{"box_2d": [y0, x0, y1, x1], "label": ...}]` from a fenced/bare JSON response; drops malformed or non-numeric boxes
  - `scale_box(box_2d, size)` — converts a `[y0, x0, y1, x1]` box normalized to `[0, 1000]` into pixel `(x0, y0, x1, y1)` corners
  - `draw_boxes(image, boxes)` — draws labeled boxes on a copy of a square image
  - `normalize_hu(hu_slice, lo, hi)` — clips a Hounsfield-unit slice to a window and rescales to 0–255
  - `window_ct_slice(hu_slice, windows=CT_WINDOWS)` — packs three fixed HU windows (wide / soft-tissue / brain) into the R/G/B channels of one false-color image; `CT_WINDOWS` is the model's *trained* CT input format, so it is not user-tunable
  - `subsample_indices(n, max_slices)` — uniformly picks slice indices across a volume (endpoints included)
  - `load_ct_volume(dicom_files, max_slices)` — reads per-slice DICOM files, sorts by InstanceNumber, subsamples, and converts to HU via `pydicom`'s `apply_rescale`
  - `ram_aware_slice_cap(total_ram_gib=None)` — returns `(default, max)` CT slice counts scaled to installed memory (`(8, 16)` on a 32 GiB Mac; clamped to a hard max of 64); detection in `_detect_total_ram_gib`
- **UI-layer helpers** (touch Streamlit, not unit-pure): `load_and_preview_image` (open + preview an upload), `run_model` (format → `generate` → return text, error-handling; passes `REPETITION_PENALTY`/`REPETITION_CONTEXT_SIZE` to keep greedy decoding from looping), `render_thought` (split trace into an expander, return the answer), `tab_settings(key_prefix, default_instruction, auto_switch=False)` (per-tab system instruction + thinking toggle with independent session-state keys).
- **Main UI** — `main()` loads the model once and renders three `st.tabs`, each delegating to a `render_*_tab` function with its own keyed widgets:
  - **Ask** (`render_ask_tab`) — text-only Q&A, no image; uses `DEFAULT_INSTRUCTION_TEXT`.
  - **Chest X-ray** (`render_cxr_tab`) — single image, two-image comparison, and anatomy localization. Default persona is `DEFAULT_INSTRUCTION_IMAGE` (or `DEFAULT_INSTRUCTION_COMPARE` for two images); `tab_settings(..., auto_switch=True)` tracks that default until the user edits it.
  - **Computed Tomography** (`render_ct_tab`) — DICOM multi-file uploader + a RAM-aware slice-count slider; uses `DEFAULT_INSTRUCTION_CT`.
- **Localization** (CXR tab) — When "Locate anatomy" is on (enabled only for exactly one image), the image is padded to a square and a dedicated `LOCALIZATION_INSTRUCTION` (which overrides the system instruction) asks MedGemma for boxes as `[y0, x0, y1, x1]` normalized to `[0, 1000]`. The response is parsed, scaled, drawn, and cropped back to the original size.
- **Comparison** (CXR tab) — When two images are attached, both are sent in one prompt (each labeled "First image:"/"Second image:" so the persona's ordinal wording binds to a specific image), the default instruction switches to `DEFAULT_INSTRUCTION_COMPARE`, localization is disabled, and the token budget grows to 600 (1600 with thinking). Mutually exclusive with localization.
- **CT analysis** (CT tab) — Per-slice DICOM files → `load_ct_volume` (HU arrays) → `window_ct_slice` (false-color RGB) → `build_messages` with `"SLICE n"` labels → `generate` at the CT token budget. The slice count is capped by `ram_aware_slice_cap` (multi-image inference is memory-heavy on unified memory). A sample windowed slice is previewed.

## Tests

- `tests/test_streamlit_app.py` — pure helpers (`parse_response`, `build_messages`, `get_generation_params`, `pad_to_square`, `parse_boxes`, `scale_box`, `draw_boxes`, `normalize_hu`, `window_ct_slice`, `subsample_indices`, `load_ct_volume`, `ram_aware_slice_cap`); no Streamlit or model needed. `load_ct_volume` is tested with real in-memory DICOMs built by `_dicom_bytes` (so it exercises `dcmread` → `pixel_array` → `apply_rescale`, including out-of-order InstanceNumber sorting). `ram_aware_slice_cap` is tested via its `total_ram_gib` override so it is host-independent.
- `tests/test_app_ui.py` — UI flow via `streamlit.testing.v1.AppTest`. Patch `mlx_vlm.*` (`load`, `load_config`, `apply_chat_template`, `generate`) at the source, not `streamlit_app`, since AppTest re-execs the script each run. **Select widgets by key** (e.g. `at.button(key="cxr_run")`, `at.text_area(key="ct_instruction")`), not by position, since tabs render every widget. Per-tab keys: `ask_*`, `cxr_*` (incl. `cxr_image1`/`cxr_image2`/`cxr_localize`), `ct_*` (incl. `ct_files`/`ct_slices`). Image tests build in-memory PNGs; CT tests build in-memory DICOMs (`_dicom_bytes`) and must chain multiple `.upload()` calls on one element before a single `.run()` (AppTest re-exec resets a fresh element, so separate `.upload().run()` calls keep only the last). Tests assert per-mode `num_images` / token budgets reach `generate()`, the localization/comparison mutual-exclusivity guards, and the CT windowed-RGB + 2000-token path.
- Optional manual-test assets live in `samples/` (gitignored except `samples/README.md`): a longitudinal chest-X-ray pair in `samples/cxr/` and a CT DICOM series in `samples/ct/`. The automated suite does not use them (it builds in-memory fixtures).

## Constraints

- **No multi-turn chat** — single Q&A per interaction
- **Package management** — uv (`pyproject.toml` + `uv.lock`); no `requirements.txt`
- **HF token** — optional; loaded from `.env` via `python-dotenv`. The MLX model repo is ungated, so a token only helps avoid download rate limits
- **Streamlit API** — use `width="stretch"` (not deprecated `use_container_width`)
