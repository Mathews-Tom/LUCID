# LUCID

**Linguistic Understanding, Classification, Identification & Defense** — See through the text.

Offline-first AI content detection and humanization engine for LaTeX, Markdown, and plain text documents.

## Features

- **AI Detection** — RoBERTa classifier + statistical features + ensemble scoring
- **Humanization** — Ollama-powered paraphrasing with adversarial refinement loop
- **Semantic Evaluation** — MiniLM embedding similarity, DeBERTa NLI, BERTScore quality
- **Format-Preserving** — LaTeX byte-position reconstruction, Markdown line-range replacement
- **Checkpoint/Resume** — JSON checkpoints after each chunk
- **Batch Processing** — Process entire directories

## Quick Start

```bash
git clone https://github.com/AetherForge/lucid.git
cd lucid
uv sync

uv run lucid setup
uv run lucid detect paper.tex
uv run lucid pipeline paper.tex -o paper_humanized.tex
```

## Requirements

- Python 3.12+
- [Ollama](https://ollama.ai) running locally
- 16GB RAM minimum (32GB recommended for `quality` profile)
