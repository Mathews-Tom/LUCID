# Latency

## Methodology

Latency is measured on Apple M1 Pro hardware with the balanced profile. Per-chunk latency covers individual pipeline stages; full-document latency covers the complete pipeline from input to output.

## Results

*Run `LUCID_BENCH_REAL=1 uv run pytest tests/benchmarks/ -m benchmark` to populate with real results.*

| Stage | Latency (ms) |
|-------|-------------|
| Detection per chunk | — |
| Humanization per chunk | — |
| Evaluation per chunk | — |
| Full pipeline (document) | — |

## Targets

| Metric | Target |
|--------|--------|
| Detection per paragraph (Tier 1+2) | <50ms |
| Humanization per paragraph (3B model) | <5,000ms |
| Humanization per paragraph (8B model) | <15,000ms |

## Notes

- CI mode measures orchestration overhead only (mocked models)
- Real mode produces meaningful latency measurements
- Latency varies with document length, chunk count, and model size
