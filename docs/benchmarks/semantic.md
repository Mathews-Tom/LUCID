# Semantic Preservation

## Methodology

Semantic preservation is evaluated in three stages:

1. **Embedding similarity** (MiniLM): Cosine similarity between original and humanized text embeddings
2. **NLI entailment** (DeBERTa): Bidirectional natural language inference checking factual consistency
3. **BERTScore F1** (DeBERTa-xlarge): Token-level semantic similarity with baseline rescaling

A paraphrase passes evaluation only if all three stages meet their configured thresholds.

## Results

*Run `LUCID_BENCH_REAL=1 uv run pytest tests/benchmarks/ -m benchmark` to populate with real results.*

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Embedding similarity | — | — | — |
| BERTScore F1 | — | — | — |

| NLI Label | Count |
|-----------|-------|
| entailment | — |
| neutral | — |
| contradiction | — |

**Pass rate**: —

**Targets**: >0.85 embedding similarity, >0.88 BERTScore F1
