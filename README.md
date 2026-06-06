# MedGemma Pipeline

Analyze medical text and images with the Google [MedGemma 1.5](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-bf16) model on Apple Silicon with MLX.

## Features

- Single-column UI with text input and optional image upload
- Collapsible settings for thinking trace and system instruction
- System instruction auto-adjusts based on whether an image is attached
- Fully local inference on Apple Silicon via MLX

## Setup

Requires:

- Mac with Apple Silicon
- Python 3.12
- [uv](https://docs.astral.sh/uv/)
- [Hugging Face](https://huggingface.co/) token with access to [`google/medgemma-1.5-4b-it`](https://huggingface.co/google/medgemma-1.5-4b-it)

```bash
uv sync
```

Create a `.env` file:

```
HF_TOKEN=your_token_here
```

## Usage

```bash
uv run streamlit run streamlit_app.py
```

## Development

```bash
uv run ruff check .               # Lint
uv run ruff format .              # Format
uv run ty check                   # Type check
uv run pytest                     # Run tests
```
