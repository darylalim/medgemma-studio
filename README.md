# MedGemma Pipeline

Analyze medical text and images with the Google [MedGemma 1.5](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-bf16) model on Apple Silicon with MLX.

## Features

- Single-column UI with text input and optional image upload
- Always-visible system instruction and thinking toggle
- System instruction auto-adjusts based on whether an image is attached
- Optional "Locate anatomy" mode that draws labeled bounding boxes on the image
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

To locate anatomy, upload an image, enable **Locate anatomy**, enter a query (e.g. "Where is the right clavicle?"), and run. The app draws labeled bounding boxes; this mode uses a built-in localization prompt and ignores the system instruction.

## Development

```bash
uv run ruff check .               # Lint
uv run ruff format .              # Format
uv run ty check                   # Type check
uv run pytest                     # Run tests
```
