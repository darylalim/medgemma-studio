# MedGemma Studio

[![CI](https://github.com/darylalim/medgemma-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/darylalim/medgemma-studio/actions/workflows/ci.yml) [![Release](https://img.shields.io/github/v/release/darylalim/medgemma-studio)](https://github.com/darylalim/medgemma-studio/releases) [![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

Streamlit application for analyzing medical text and images using Google [MedGemma](https://huggingface.co/mlx-community/medgemma-1.5-4b-it-bf16) on Apple Silicon with MLX.

## Disclaimer

> [!WARNING]
> **Research and educational use only — not a medical device.** MedGemma Studio is not for clinical use, diagnosis, or treatment. Its AI-generated outputs may be inaccurate and are not medical advice; always consult a qualified healthcare professional.

Using MedGemma through this app is subject to Google's [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms), which govern the model separately from this project's [license](#license).

## Features

- Four tabs, each with its own settings (system instruction + thinking toggle, in a collapsible "Model settings" panel):
  - **Ask** — text-only medical Q&A
  - **Chest X-ray** — analyze one image, compare two studies side by side (longitudinal), or draw labeled anatomy bounding boxes ("Locate anatomy")
  - **Computed Tomography** — upload a DICOM series; each slice is windowed into MedGemma's trained false-color (Hounsfield-unit) representation and read as a stack
  - **Pathology (WSI)** — upload a whole-slide image (`.svs`/`.ndpi`/`.tif`/`.tiff`); tissue patches are sampled at a chosen magnification and read as the 896px tiles MedGemma is trained on
- The model's answer **streams in live** as it is generated — a blank wait becomes visibly arriving text on a slow local model
- Staged progress feedback while reading a DICOM series or a whole-slide image, before generation begins
- RAM-aware cap on CT slices / WSI patches (multi-image inference is memory-heavy on unified memory)
- Results stay visible across reruns and clear — with a hint — when you change the inputs
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
- **Chest X-ray** — upload an image and run for analysis. To **locate anatomy**, enable the toggle and ask e.g. *"Where is the right clavicle?"*; the app draws labeled bounding boxes (this mode uses a built-in prompt and ignores the system instruction). To **compare** two studies, upload a first image, then a second in the slot that appears — the two are previewed side by side and the app sends both in one prompt and describes the changes. (Localization is single-image only and is disabled with two images.)
- **Computed Tomography** — upload a CT series as individual DICOM (`.dcm`) slice files (multi-select), choose how many slices to analyze, enter a question, and run. Each slice is windowed into a false-color image before analysis.
- **Pathology (WSI)** — upload a whole-slide image (`.svs`/`.ndpi`/`.tif`/`.tiff`), pick a magnification (5/10/20/40×) and how many tissue patches to analyze, enter a question, and run. A tissue-overview overlay (sampled patches outlined) and a sample patch are shown, with the actual magnification disclosed (clamped to the slide's available pyramid levels).

## Development

```bash
uv run ruff check .               # Lint
uv run ruff format .              # Format
uv run ty check                   # Type check
uv run pytest                     # Run tests
```

Linting uses a curated ruff rule set (`E`, `F`, `I`, `UP`, `B`, `SIM`, `C4`); see `[tool.ruff.lint]` in `pyproject.toml`.

Every push to `main` and every pull request runs these same four gates on GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) on an Apple-Silicon runner — the CI badge above reflects the latest run. The workflow's `uv sync --locked` step also fails if `uv.lock` has drifted from `pyproject.toml`.

Pushing a `vX.Y.Z` tag publishes a GitHub Release: a second workflow ([`.github/workflows/release.yml`](.github/workflows/release.yml)) verifies the tag matches the `pyproject.toml` version, then creates the release with notes generated from the commit history (there is no hand-maintained changelog). To cut one: bump `version` in `pyproject.toml`, run `uv lock`, commit, then push a `vX.Y.Z` tag on that commit. Published releases appear on the [Releases page](https://github.com/darylalim/medgemma-studio/releases).

If you use [Claude Code](https://claude.com/claude-code) in this repo, `.claude/settings.json` wires the commands above into hooks: edited Python files are auto-formatted (`ruff`) and type-checked (`ty`), the test suite runs when Claude finishes a turn that touched code or config (docs/chat turns are skipped), and writes to `.env`/`.streamlit/secrets.toml`/`uv.lock` are blocked. The `.claude/settings.json` config is itself guarded by `TestHooksConfig` — as are the repo's other checked-in assets (the theme, the CI and release workflows, and this project's `CLAUDE.md`) via `TestThemeConfig` / `TestCiWorkflow` / `TestReleaseWorkflow` / `TestClaudeMd`, so a config or doc that drifts from the code fails a test.

## License

This project's source code is licensed under the [Apache License 2.0](LICENSE) (`Apache-2.0`).

The MedGemma model is **not** covered by that license. It is distributed under Google's [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms) and downloads separately from Hugging Face at runtime; your use of the model is governed by those terms.
