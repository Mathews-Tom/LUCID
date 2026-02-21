"""Humanization evasion rate benchmarks."""
from __future__ import annotations

import pytest

from lucid.models.results import ParaphraseResult


@pytest.mark.benchmark
class TestHumanizationEvasion:
    """Benchmark evasion rates."""

    def test_evasion_rate_structure(self, benchmark_collector) -> None:  # type: ignore[no-untyped-def]
        """CI mode: validate evasion result structure."""
        single_pass = [
            ParaphraseResult(
                chunk_id=f"chunk-{i}", original_text=f"Text {i}",
                humanized_text=f"Humanized {i}", iteration_count=1,
                strategy_used="standard", final_detection_score=score,
            )
            for i, score in enumerate([0.22, 0.45, 0.18, 0.55, 0.12, 0.38, 0.09])
        ]

        adversarial = [
            ParaphraseResult(
                chunk_id=f"adv-{i}", original_text=f"Text {i}",
                humanized_text=f"Humanized {i}", iteration_count=count,
                strategy_used=strategy, final_detection_score=score,
            )
            for i, (score, count, strategy) in enumerate([
                (0.15, 3, "restructure"), (0.22, 2, "standard"),
                (0.08, 4, "voice_shift"), (0.31, 5, "vocabulary"),
                (0.11, 3, "reorder"), (0.19, 2, "standard"),
                (0.05, 4, "restructure"),
            ])
        ]

        threshold = 0.30
        sp_evaded = sum(1 for r in single_pass if r.final_detection_score < threshold)
        sp_rate = sp_evaded / len(single_pass)
        adv_evaded = sum(1 for r in adversarial if r.final_detection_score < threshold)
        adv_rate = adv_evaded / len(adversarial)

        assert 0.0 <= sp_rate <= 1.0
        assert 0.0 <= adv_rate <= 1.0

        strategies: dict[str, int] = {}
        for r in adversarial:
            strategies[r.strategy_used] = strategies.get(r.strategy_used, 0) + 1

        benchmark_collector.evasion = {
            "single_pass_evasion_rate": round(sp_rate, 3),
            "adversarial_evasion_rate": round(adv_rate, 3),
            "mean_iterations": round(
                sum(r.iteration_count for r in adversarial) / len(adversarial), 1
            ),
            "strategy_distribution": strategies,
        }

    def test_paraphrase_result_validity(self) -> None:
        """Validate ParaphraseResult invariants."""
        result = ParaphraseResult(
            chunk_id="bench-001", original_text="Quick brown fox",
            humanized_text="Swift brown fox", iteration_count=2,
            strategy_used="standard", final_detection_score=0.25,
        )
        assert 0.0 <= result.final_detection_score <= 1.0
        assert result.iteration_count >= 0
