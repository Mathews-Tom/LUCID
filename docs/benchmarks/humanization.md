# Humanization Evasion Rates

## Methodology

Evasion rate measures the percentage of AI-detected paragraphs that score below the detection threshold after humanization. Single-pass uses one iteration; adversarial uses the full refinement loop with strategy rotation.

## Strategy Rotation

LUCID cycles through these humanization strategies during adversarial refinement:

1. **standard** — Direct paraphrasing
2. **restructure** — Sentence reordering and restructuring
3. **voice_shift** — Active/passive voice changes
4. **vocabulary** — Synonym substitution and vocabulary diversification
5. **reorder** — Paragraph-level restructuring

## Results

*Run `LUCID_BENCH_REAL=1 uv run pytest tests/benchmarks/ -m benchmark` to populate with real results.*

| Metric | Value |
|--------|-------|
| Single-pass evasion rate | — |
| Adversarial evasion rate | — |
| Mean adversarial iterations | — |

**Targets**: >70% single-pass, >85% adversarial

### Strategy Distribution

| Strategy | Count |
|----------|-------|
| standard | — |
| restructure | — |
| voice_shift | — |
| vocabulary | — |
| reorder | — |
