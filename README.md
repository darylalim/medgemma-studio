# MedGemma Studio

Analyze medical text, 2D images (e.g. chest X-ray), 3D CT volumes, and whole-slide pathology images with the Google [MedGemma 1.5](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-bf16) model on Apple Silicon with MLX.

## Features

- Four tabs, each with its own system instruction and thinking toggle:
  - **Ask** — text-only medical Q&A
  - **Chest X-ray** — analyze one image, compare two studies (longitudinal), or draw labeled anatomy bounding boxes ("Locate anatomy")
  - **Computed Tomography** — upload a DICOM series; each slice is windowed into MedGemma's trained false-color (Hounsfield-unit) representation and read as a stack
  - **Pathology (WSI)** — upload a whole-slide image (`.svs`/`.ndpi`/`.tiff`); tissue patches are sampled at a chosen magnification and read as the 896px tiles MedGemma is trained on
- RAM-aware cap on CT slices / WSI patches (multi-image inference is memory-heavy on unified memory)
- Results stay visible across reruns and clear automatically when you change the inputs
- Clinical light/dark theme that follows your system (OS/browser) appearance
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

The app opens with four tabs:

- **Ask** — enter a question and run for a text-only answer.
- **Chest X-ray** — upload an image and run for analysis. To **locate anatomy**, enable the toggle and ask e.g. *"Where is the right clavicle?"*; the app draws labeled bounding boxes (this mode uses a built-in prompt and ignores the system instruction). To **compare** two studies, upload a first image, then a second in the slot that appears — the app sends both in one prompt and describes the changes. (Localization is single-image only and is disabled with two images.)
- **Computed Tomography** — upload a CT series as individual DICOM (`.dcm`) slice files (multi-select), choose how many slices to analyze, enter a question, and run. Each slice is windowed into a false-color image before analysis.
- **Pathology (WSI)** — upload a whole-slide image (`.svs`/`.ndpi`/`.tiff`), pick a magnification (5/10/20/40×) and how many tissue patches to analyze, enter a question, and run. A tissue-overview overlay (sampled patches outlined) and a sample patch are shown, with the actual magnification disclosed (clamped to the slide's available pyramid levels).

## Development

```bash
uv run ruff check .               # Lint
uv run ruff format .              # Format
uv run ty check                   # Type check
uv run pytest                     # Run tests
```

Linting uses a curated ruff rule set (`E`, `F`, `I`, `UP`, `B`, `SIM`, `C4`); see `[tool.ruff.lint]` in `pyproject.toml`.
