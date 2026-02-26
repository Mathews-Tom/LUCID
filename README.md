# LUCID

**Linguistic Understanding, Classification, Identification & Defense** — See through the text.

Offline-first AI content detection and humanization engine for LaTeX, Markdown, and plain text documents. Runs entirely on local hardware via [Ollama](https://ollama.ai) for LLM inference and ONNX-optimized models for detection/evaluation.

## Features

- **AI Detection** — RoBERTa classifier + statistical features + ensemble scoring
- **Humanization** — Ollama-powered paraphrasing with adversarial refinement loop
- **Semantic Evaluation** — MiniLM embedding similarity, DeBERTa NLI entailment, BERTScore quality
- **Format-Preserving** — LaTeX byte-position reconstruction, Markdown line-range replacement, plain text paragraph segmentation
- **Checkpoint/Resume** — JSON checkpoints after each chunk, resume interrupted runs
- **Batch Processing** — Process entire directories of documents

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) running locally (for humanization)
- 16GB RAM minimum (32GB recommended for `quality` profile)
- macOS (Apple Silicon optimized) or Linux x86-64

## Installation

```bash
# Clone and install with uv
git clone https://github.com/AetherForge/lucid.git
cd lucid
uv sync

# Verify installation
uv run lucid --version
```

### First-Run Setup

```bash
# Guided setup — checks Ollama, downloads models
uv run lucid setup

# Setup for a specific profile
uv run lucid setup --profile quality
```

## Quick Start

```bash
# Check model availability
uv run lucid models

# Download missing models
uv run lucid models --download

# Detect AI content in a document
uv run lucid detect paper.tex
uv run lucid detect paper.tex --output-format json

# Run full pipeline (detect → humanize → evaluate → reconstruct)
uv run lucid pipeline paper.tex -o paper_humanized.tex

# Humanize a document directly
uv run lucid humanize paper.tex -o paper_humanized.tex

# Process a directory of documents
uv run lucid detect ./papers/

# View current configuration
uv run lucid config
```

## CLI Reference

```
lucid [OPTIONS] COMMAND [ARGS]...

Global Options:
  --profile [fast|balanced|quality]  Quality profile
  --config PATH                      Custom config TOML file
  -v, --verbose                      Verbose output
  -q, --quiet                        Suppress all output
  --version                          Show version

Commands:
  detect     Detect AI-generated content in a document
  humanize   Humanize AI-detected content in a document
  pipeline   Full detect → humanize → validate pipeline
  config     View or modify configuration
  models     Check or download required models
  setup      First-run setup: check Ollama, download models
```

### detect

```bash
lucid detect <INPUT> [OPTIONS]
  --output-format [json|text]   Report format (default: text)
  --threshold FLOAT             Detection threshold override
  -o, --output PATH             Write report to file
```

### humanize

```bash
lucid humanize <INPUT> [OPTIONS]
  -o, --output PATH                  Output file path
  --model TEXT                       Override Ollama model tag
  --adversarial / --no-adversarial   Enable adversarial loop (default: on)
```

### pipeline

```bash
lucid pipeline <INPUT> [OPTIONS]
  -o, --output PATH                  Output file path
  --report PATH                      Write report file
  --output-format [json|text|annotated]  Report format (default: json)
  --resume / --no-resume             Resume from checkpoint (default: on)
  --checkpoint-dir PATH              Checkpoint directory
```

### setup

```bash
lucid setup [OPTIONS]
  --profile [fast|balanced|quality]   Profile to set up (default: balanced)
```

## Configuration

LUCID uses TOML configuration with three built-in profiles:

| Profile | Model Size | Speed | Quality | Use Case |
|---------|-----------|-------|---------|----------|
| `fast` | 3B | Fastest | Good | Quick passes, drafts |
| `balanced` | 7B | Moderate | Better | Default for most documents |
| `quality` | 14B+ | Slow | Best | Final submissions |

```bash
# View config
uv run lucid config

# Override settings
uv run lucid config --set detection.use_binoculars true
```

Configuration files: `config/default.toml`, `config/profiles/`.

### Model Recommendations

| Profile | Default Model | Size | RAM Required | License |
|---------|--------------|------|-------------|---------|
| fast | phi3:3.8b | 2.4GB | 8GB | MIT |
| balanced | qwen2.5:7b | 4.5GB | 12GB | Apache 2.0 |
| quality | llama3.2:8b | 4.9GB | 16GB | Meta Community |

### Profile Comparison

| Feature | fast | balanced | quality |
|---------|------|----------|---------|
| Statistical detection | No | Yes | Yes |
| Binoculars (Tier 3) | No | No | Yes |
| Adversarial iterations | 1 | 3 | 5 |
| LaTeX validation | No | Yes | Yes |
| Embedding threshold | 0.75 | 0.80 | 0.85 |
| BERTScore threshold | 0.82 | 0.88 | 0.90 |

## Web UI

LUCID includes an optional Gradio web interface for browser-based detection and humanization.

```bash
# Install web extras
uv sync --extra web

# Launch web UI
uv run lucid-web
```

The web UI provides two tabs: **Detect** (upload and analyze documents) and **Full Pipeline** (detect, humanize, and download results).

## Architecture

```
Input Document
    │
    ▼
┌─────────┐     ┌──────────┐     ┌────────────┐     ┌───────────┐     ┌──────────────┐
│  Parser  │────▶│ Detector │────▶│ Humanizer  │────▶│ Evaluator │────▶│Reconstructor │
│          │     │          │     │            │     │           │     │              │
│ LaTeX    │     │ RoBERTa  │     │ Ollama LLM │     │ MiniLM    │     │ Position-    │
│ Markdown │     │ Stats    │     │ Adversarial│     │ DeBERTa   │     │ based        │
│ Plain    │     │ Ensemble │     │ Loop       │     │ BERTScore │     │ Replacement  │
└─────────┘     └──────────┘     └────────────┘     └───────────┘     └──────────────┘
    │                                                                         │
    └─────────────── Checkpoint after each chunk ─────────────────────────────┘
```

## Project Structure

```
src/lucid/
├── cli.py              # Click CLI interface
├── pipeline.py         # Pipeline orchestrator
├── checkpoint.py       # Checkpoint/resume system
├── progress.py         # Rich progress reporting
├── output.py           # Output formatting (JSON, text, annotated)
├── config.py           # TOML config with profile merging
├── parser/             # Document parsers (LaTeX, Markdown, plain text)
├── detector/           # AI detection (RoBERTa, statistical, ensemble)
├── humanizer/          # Ollama paraphrasing with adversarial refinement
├── evaluator/          # Semantic evaluation (embedding, NLI, BERTScore)
├── reconstructor/      # Format-preserving document reconstruction
└── models/
    ├── manager.py      # Model lifecycle management
    ├── download.py     # Model availability checker and downloader
    └── results.py      # Result dataclasses
```

## Benchmarks

| Metric | Target |
|--------|--------|
| Detection TPR (AI text) | >85% at 5% FPR |
| Evasion rate (single-pass) | >70% |
| Evasion rate (adversarial) | >85% |
| Semantic similarity | >0.85 embedding, >0.88 BERTScore |

Run benchmarks: `uv run pytest tests/benchmarks/ -m benchmark -v`

Full results: [docs/benchmarks/](docs/benchmarks/README.md)

## Development

```bash
# Install with dev dependencies
uv sync --extra dev

# Run unit tests
uv run pytest

# Run integration tests
uv run pytest -m integration

# Run all tests
uv run pytest -m ""

# Lint
uv run ruff check src/ tests/

# Run example scripts
uv run python examples/detect_latex.py tests/corpus/latex/simple.tex
uv run python examples/full_pipeline.py tests/corpus/markdown/simple.md

# Type check
uv run mypy src/lucid/
```

## License

MIT — See [LICENSE](LICENSE) for details.

See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for the ethical framework and responsible use policy.
