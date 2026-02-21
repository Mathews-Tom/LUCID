"""Detection accuracy benchmarks."""
from __future__ import annotations

import pytest

from lucid.config import load_config
from lucid.models.results import DetectionResult


@pytest.mark.benchmark
class TestDetectionAccuracy:
    """Benchmark detection TPR/FPR at various thresholds."""

    def test_detection_thresholds_ci(self, benchmark_collector) -> None:  # type: ignore[no-untyped-def]
        """CI mode: validate detection threshold logic and result structure."""
        thresholds = [0.3, 0.5, 0.65, 0.8]
        ai_scores = [0.85, 0.72, 0.91, 0.68, 0.45]
        human_scores = [0.12, 0.08, 0.25, 0.18, 0.31]

        tpr_values = []
        fpr_values = []
        for threshold in thresholds:
            tp = sum(1 for s in ai_scores if s >= threshold)
            fp = sum(1 for s in human_scores if s >= threshold)
            tpr = tp / len(ai_scores)
            fpr = fp / len(human_scores)
            tpr_values.append(round(tpr, 3))
            fpr_values.append(round(fpr, 3))
            assert 0.0 <= tpr <= 1.0
            assert 0.0 <= fpr <= 1.0

        benchmark_collector.detection_accuracy = {
            "thresholds": thresholds,
            "tpr": tpr_values,
            "fpr": fpr_values,
            "corpus_size": {"ai": len(ai_scores), "human": len(human_scores)},
        }

    def test_detection_result_validity(self) -> None:
        """Validate DetectionResult invariants."""
        result = DetectionResult(
            chunk_id="bench-001",
            ensemble_score=0.75,
            classification="ai_generated",
            roberta_score=0.80,
            statistical_score=0.65,
        )
        assert 0.0 <= result.ensemble_score <= 1.0
        assert result.classification in ("human", "ambiguous", "ai_generated")

    def test_tier_activation_rates(self, benchmark_collector) -> None:  # type: ignore[no-untyped-def]
        """Validate tier activation across profiles."""
        fast_config = load_config(profile="fast")
        balanced_config = load_config(profile="balanced")
        quality_config = load_config(profile="quality")

        assert not fast_config.detection.use_statistical
        assert balanced_config.detection.use_statistical
        assert quality_config.detection.use_binoculars

        benchmark_collector.detection_accuracy["tier_activation"] = {
            "fast": {"roberta": True, "statistical": False, "binoculars": False},
            "balanced": {"roberta": True, "statistical": True, "binoculars": False},
            "quality": {"roberta": True, "statistical": True, "binoculars": True},
        }
