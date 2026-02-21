# Detection Accuracy

## Methodology

Detection accuracy is measured by running the ensemble detector (RoBERTa + statistical features) against a labeled corpus of AI-generated and human-written text. True positive rate (TPR) and false positive rate (FPR) are computed at multiple score thresholds.

## Tier Activation by Profile

| Profile | RoBERTa (Tier 1) | Statistical (Tier 2) | Binoculars (Tier 3) |
|---------|:-:|:-:|:-:|
| fast | Yes | No | No |
| balanced | Yes | Yes | No |
| quality | Yes | Yes | Yes |

## Results

*Run `LUCID_BENCH_REAL=1 uv run pytest tests/benchmarks/ -m benchmark` to populate with real results.*

| Threshold | TPR | FPR |
|-----------|-----|-----|
| 0.30 | — | — |
| 0.50 | — | — |
| 0.65 | — | — |
| 0.80 | — | — |

**Target**: >85% TPR at 5% FPR
