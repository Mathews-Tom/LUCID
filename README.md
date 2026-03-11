# LUCID

**Linguistic Understanding, Classification, Identification & Defense** — See through the text.

Offline-first AI content detection and humanization engine for LaTeX, Markdown, and plain text documents. Runs entirely on local hardware via [Ollama](https://ollama.ai) for LLM inference and ONNX-optimized models for detection/evaluation.

## Features

- **AI Detection** — 12-feature statistical detector + RoBERTa classifier + optional Binoculars cross-perplexity + ensemble scoring with score calibration
- **Transformation** — Ollama-powered paraphrasing with 5 operator strategies, beam search, and term protection (NER + regex)
- **Semantic Evaluation** — Modular metric checkers: MiniLM embedding similarity, DeBERTa NLI entailment, BERTScore quality, readability, structural preservation, semantic drift, term verification
- **Explainability** — Per-chunk detection explanations with feature attribution
- **Format-Preserving** — LaTeX byte-position reconstruction, Markdown line-range replacement, plain text paragraph segmentation
- **Checkpoint/Resume** — JSON checkpoints after each chunk, resume interrupted runs
- **Batch Processing** — Process entire directories of documents
- **Benchmarking** — YAML-manifest-driven experiments with AUROC/AUPRC/ECE metrics and slice-based aggregation

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) running locally (for transformation)
- 16GB RAM minimum (32GB recommended for `quality` profile)
- macOS (Apple Silicon optimized) or Linux x86-64

## Installation

```bash
# From PyPI
uv add lucid-ai

# From source
git clone https://github.com/Mathews-Tom/LUCID.git
cd LUCID
uv sync
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

# Transform AI-detected content
uv run lucid transform paper.tex -o paper_transformed.tex

# Run full pipeline (detect → transform → evaluate → reconstruct)
uv run lucid pipeline paper.tex -o paper_output.tex

# Calibrate detector scores against labeled data
uv run lucid calibrate dataset.jsonl -o calibration.json

# Explain detection results with feature attribution
uv run lucid explain paper.tex

# Run benchmarks
uv run lucid bench run manifests/experiment.yaml
uv run lucid bench report results/ -o report.md

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
  detect      Detect AI-generated content in a document
  transform   Transform AI-detected content in a document
  pipeline    Full detect → transform → evaluate → reconstruct pipeline
  calibrate   Calibrate detector scores against labeled data
  explain     Explain detection results with feature attribution
  bench       Benchmark commands (run, report)
  config      View or modify configuration
  models      Check or download required models
  setup       First-run setup: check Ollama, download models
```

### detect

```bash
lucid detect <INPUT> [OPTIONS]
  --output-format [json|text]   Report format (default: text)
  --threshold FLOAT             Detection threshold override
  -o, --output PATH             Write report to file
```

### transform

```bash
lucid transform <INPUT> [OPTIONS]
  -o, --output PATH             Output file path
  --model TEXT                   Override Ollama model tag
  --search / --no-search         Enable beam search loop (default: on)
  --skip-eval                    Skip semantic evaluation
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

### bench

```bash
lucid bench run <MANIFEST> [OPTIONS]
  -o, --output-dir PATH         Output directory for results
  --detector TEXT                 Detector name override

lucid bench report <RESULTS_DIR> [OPTIONS]
  -o, --output PATH             Report output path
  --format [json|csv|markdown]   Report format (default: markdown)
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
| Search iterations | 1 | 3 | 5 |
| LaTeX validation | No | Yes | Yes |
| Embedding threshold | 0.75 | 0.80 | 0.85 |
| BERTScore threshold | 0.82 | 0.88 | 0.90 |

## Web UI

LUCID includes an optional Gradio web interface for browser-based detection and transformation.

```bash
# Install web extras
uv sync --extra web

# Launch web UI
uv run lucid-web
```

The web UI provides two tabs: **Detect** (upload and analyze documents) and **Full Pipeline** (detect, transform, and download results).

## Architecture

```
Input Document
    │
    ▼
┌─────────┐     ┌──────────┐     ┌─────────────┐     ┌───────────┐     ┌──────────────┐
│  Parser  │────▶│ Detector │────▶│ Transformer │────▶│ Evaluator │────▶│Reconstructor │
│          │     │          │     │             │     │           │     │              │
│ LaTeX    │     │ 12-feat  │     │ Ollama LLM  │     │ Embedding │     │ Position-    │
│ Markdown │     │ RoBERTa  │     │ 5 operators │     │ NLI       │     │ based        │
│ Plain    │     │ Binoculrs│     │ Beam search │     │ BERTScore │     │ Replacement  │
└─────────┘     └──────────┘     └─────────────┘     └───────────┘     └──────────────┘
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
├── core/               # Shared infrastructure
│   ├── errors.py       # Error hierarchy
│   ├── protocols.py    # Detector/Transformer/Evaluator protocols
│   ├── registry.py     # Component registry
│   └── types.py        # Shared type definitions
├── parser/             # Document parsers (LaTeX, Markdown, plain text)
├── detector/           # AI detection engine
│   ├── statistical.py  # 12-feature statistical detector
│   ├── features.py     # Feature extraction (LM, style, structural, discourse)
│   ├── roberta.py      # RoBERTa classifier
│   ├── binoculars.py   # Cross-perplexity detector (Tier 3)
│   ├── ensemble.py     # Score fusion
│   ├── calibrate.py    # Score calibration
│   └── explain.py      # Feature attribution
├── transform/          # Text transformation
│   ├── ollama.py       # Async Ollama HTTP client
│   ├── operators.py    # 5 transformation strategies
│   ├── prompts.py      # Prompt construction
│   ├── search.py       # Beam search over operator space
│   └── term_protect.py # Named entity + regex term protection
├── evaluator/          # Evaluation pipeline orchestrator
├── metrics/            # Modular metric checkers
│   ├── embedding.py    # MiniLM cosine similarity
│   ├── nli.py          # DeBERTa NLI entailment
│   ├── bertscore.py    # BERTScore F1
│   ├── readability.py  # Flesch-Kincaid readability
│   ├── structure.py    # Structural preservation
│   ├── drift.py        # Semantic drift detection
│   └── term_verify.py  # Protected term verification
├── reconstructor/      # Format-preserving document reconstruction
├── bench/              # Benchmarking framework
│   ├── datasets.py     # JSONL/corpus data loading
│   ├── manifests.py    # YAML experiment manifests
│   ├── slices.py       # Slice-based grouping
│   ├── aggregation.py  # AUROC, AUPRC, TPR@FPR5, ECE
│   ├── experiment.py   # Single experiment runner
│   ├── runner.py       # Batch experiment orchestration
│   └── reporting.py    # JSON, CSV, Markdown reports
├── models/
│   ├── manager.py      # Model lifecycle management
│   ├── download.py     # Model availability checker and downloader
│   └── results.py      # Result dataclasses
└── web.py              # Optional Gradio web interface
```

## Benchmarks

| Metric | Target |
|--------|--------|
| Detection TPR (AI text) | >85% at 5% FPR |
| Evasion rate (single-pass) | >70% |
| Evasion rate (adversarial) | >85% |
| Semantic similarity | >0.85 embedding, >0.88 BERTScore |

```bash
# Run a benchmark experiment
uv run lucid bench run benchmarks/manifests/detector_robustness_v1.yaml -o results/

# Generate a report
uv run lucid bench report results/ -o report.md
```

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

# Type check
uv run mypy src/lucid/
```

## License

MIT — See [LICENSE](LICENSE) for details.

See [RESPONSIBLE_USE.md](RESPONSIBLE_USE.md) for the ethical framework and responsible use policy.
