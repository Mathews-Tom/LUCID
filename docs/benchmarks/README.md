# Benchmark Results

LUCID includes an automated benchmark suite measuring detection accuracy, humanization evasion rates, semantic preservation, and latency.

## Running Benchmarks

```bash
# CI mode (mocked models, validates code paths)
uv run pytest tests/benchmarks/ -m benchmark -v

# Real mode (actual models, meaningful measurements)
LUCID_BENCH_REAL=1 uv run pytest tests/benchmarks/ -m benchmark -v

# Generate Markdown report from results
uv run python scripts/bench_report.py
```

## Methodology

All benchmarks run against the same test corpus under controlled conditions:

- **Detection accuracy**: TPR/FPR measured at multiple thresholds against labeled AI/human text
- **Evasion rates**: Single-pass and adversarial evasion rates against LUCID's own detector
- **Semantic preservation**: Embedding similarity (MiniLM), BERTScore F1, and NLI entailment
- **Latency**: Per-chunk and full-document timing on reference hardware (Apple M1 Pro)

Results are written to `benchmarks/results/` as timestamped JSON files.

## Quality Metric Targets

| Metric | Target |
|--------|--------|
| Detection TPR (AI text) | >85% at 5% FPR |
| Detection FPR (human text) | <5% |
| Evasion rate (single-pass) | >70% |
| Evasion rate (adversarial) | >85% |
| Embedding similarity | >0.85 |
| BERTScore F1 | >0.88 |
| LaTeX round-trip success | >95% |
| Markdown round-trip success | >98% |
| Detection latency (per paragraph) | <50ms |
| Humanization latency (3B model) | <5s |
| Humanization latency (8B model) | <15s |

## Detailed Results

- [Detection Accuracy](detection.md)
- [Humanization Evasion](humanization.md)
- [Semantic Preservation](semantic.md)
- [Latency](latency.md)
