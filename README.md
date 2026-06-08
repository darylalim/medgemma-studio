# MedGemma Pipeline

Analyze medical text, 2D images (e.g. chest X-ray), and 3D CT volumes with the Google [MedGemma 1.5](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-bf16) model on Apple Silicon with MLX.

## Features

- Three tabs, each with its own system instruction and thinking toggle:
  - **Ask** — text-only medical Q&A
  - **Chest X-ray** — analyze one image, compare two studies (longitudinal), or draw labeled anatomy bounding boxes ("Locate anatomy")
  - **Computed Tomography** — upload a DICOM series; each slice is windowed into MedGemma's trained false-color (Hounsfield-unit) representation and read as a stack
- RAM-aware CT slice cap (multi-image inference is memory-heavy on unified memory)
- Fully local inference on Apple Silicon via MLX

## Setup

Requires:

- Mac with Apple Silicon
- Python 3.12
- [uv](https://docs.astral.sh/uv/)

The model ([`mlx-community/medgemma-1.5-4b-it-bf16`](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-bf16)) downloads from Hugging Face on first run. The repo is ungated, so no token is required.

```bash
uv sync
```

Optionally, create a `.env` file with a Hugging Face token to avoid download rate limits:

```
HF_TOKEN=your_token_here
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

The app opens with three tabs:

- **Ask** — enter a question and run for a text-only answer.
- **Chest X-ray** — upload an image and run for analysis. To **locate anatomy**, enable the toggle and ask e.g. *"Where is the right clavicle?"*; the app draws labeled bounding boxes (this mode uses a built-in prompt and ignores the system instruction). To **compare** two studies, upload a first image, then a second in the slot that appears — the app sends both in one prompt and describes the changes. (Localization is single-image only and is disabled with two images.)
- **Computed Tomography** — upload a CT series as individual DICOM (`.dcm`) slice files (multi-select), choose how many slices to analyze, enter a question, and run. Each slice is windowed into a false-color image before analysis.

## Development

```bash
uv run ruff check .               # Lint
uv run ruff format .              # Format
uv run ty check                   # Type check
uv run pytest                     # Run tests
```
