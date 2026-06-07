# CLAUDE.md

## Project

**medgemma-pipeline** — Analyze medical text and images with the Google MedGemma 1.5 model on Apple Silicon with MLX.

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
- **Helper functions** — Pure functions extracted for testability, called by `main()`:
  - `parse_response(response, is_thinking)` — splits `<unused94>`/`<unused95>` thinking trace from response
  - `build_messages(prompt, system_instruction, image)` — constructs chat message list for `apply_chat_template`
  - `get_generation_params(has_image, is_thinking, system_instruction, is_localizing=False)` — returns `(full_instruction, max_new_tokens)`
  - `pad_to_square(image)` — top-left pads an image to a square so localization coordinates map back without an offset
  - `parse_boxes(response)` — extracts `[{"box_2d": [y0, x0, y1, x1], "label": ...}]` from a fenced/bare JSON response; drops malformed or non-numeric boxes
  - `scale_box(box_2d, size)` — converts a `[y0, x0, y1, x1]` box normalized to `[0, 1000]` into pixel `(x0, y0, x1, y1)` corners
  - `draw_boxes(image, boxes)` — draws labeled boxes on a copy of a square image
- **Main UI** — Single-column layout: text input, optional image uploader, always-visible system instruction and thinking toggle, plus a "Locate anatomy" toggle (active only when an image is attached).
- **Inference** — `main()` uses the helpers to build messages, format via `apply_chat_template`, call `generate()`, and parse the response.
- **Localization** — When "Locate anatomy" is on, the image is padded to a square and a dedicated `LOCALIZATION_INSTRUCTION` (which overrides the system instruction) asks MedGemma for boxes as `[y0, x0, y1, x1]` normalized to `[0, 1000]`. The response is parsed, scaled, drawn, and cropped back to the original size.

## Tests

- `tests/test_streamlit_app.py` — pure helpers (`parse_response`, `build_messages`, `get_generation_params`, `pad_to_square`, `parse_boxes`, `scale_box`, `draw_boxes`); no Streamlit or model needed.
- `tests/test_app_ui.py` — UI flow via `streamlit.testing.v1.AppTest`. Patch `mlx_vlm.*` (`load`, `load_config`, `apply_chat_template`, `generate`) at the source, not `streamlit_app`, since AppTest re-execs the script each run. Image-upload and localization tests build a small in-memory PNG with Pillow; localization tests mock a fenced-JSON box response.

## Constraints

- **No multi-turn chat** — single Q&A per interaction
- **Package management** — uv (`pyproject.toml` + `uv.lock`); no `requirements.txt`
- **HF token** — optional; loaded from `.env` via `python-dotenv`. The MLX model repo is ungated, so a token only helps avoid download rate limits
- **Streamlit API** — use `width="stretch"` (not deprecated `use_container_width`)
