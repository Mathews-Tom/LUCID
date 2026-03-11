"""Tests for metric aggregation."""

from __future__ import annotations

import pytest

from lucid.bench.aggregation import AggregatedMetrics, SliceAggregator
from lucid.bench.slices import SliceKey
from lucid.core.types import DetectionRecord, SampleRecord


def _sample(sample_id: str, source_class: str = "human") -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        split="test",
        domain="academic",
        source_class=source_class,
        source_model=None,
        document_format="plaintext",
        text="text",
    )


def _detection(sample_id: str, score: float) -> DetectionRecord:
    return DetectionRecord(
        record_id=f"det_{sample_id}",
        sample_id=sample_id,
        transform_id=None,
        detector_name="test",
        score=score,
        confidence=None,
        threshold=None,
        predicted_label="ambiguous",
    )


class TestComputeAuroc:
    def test_perfect_separation(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        result = SliceAggregator.compute_auroc(labels, scores)
        assert result is not None
        assert result == 1.0

    def test_single_class_returns_none(self) -> None:
        labels = [1, 1, 1]
        scores = [0.5, 0.6, 0.7]
        assert SliceAggregator.compute_auroc(labels, scores) is None

    def test_random_scores(self) -> None:
        labels = [0, 1, 0, 1]
        scores = [0.5, 0.5, 0.5, 0.5]
        result = SliceAggregator.compute_auroc(labels, scores)
        assert result is not None
        assert result == 0.5


class TestComputeAuprc:
    def test_two_classes(self) -> None:
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        result = SliceAggregator.compute_auprc(labels, scores)
        assert result is not None
        assert result == 1.0

    def test_single_class_returns_none(self) -> None:
        assert SliceAggregator.compute_auprc([0, 0], [0.1, 0.2]) is None


class TestComputeTprAtFpr:
    def test_perfect_separation(self) -> None:
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        scores = [0.05, 0.1, 0.15, 0.2, 0.8, 0.85, 0.9, 0.95]
        result = SliceAggregator.compute_tpr_at_fpr(labels, scores, target_fpr=0.05)
        assert result is not None
        assert result >= 0.0

    def test_single_class_returns_none(self) -> None:
        assert SliceAggregator.compute_tpr_at_fpr([1, 1], [0.8, 0.9]) is None


class TestComputeEce:
    def test_perfect_calibration(self) -> None:
        # If all predictions match labels perfectly, ECE is 0
        labels = [0, 0, 1, 1]
        confidences = [0.0, 0.0, 1.0, 1.0]
        result = SliceAggregator.compute_ece(labels, confidences)
        assert result is not None
        assert result == pytest.approx(0.0, abs=0.01)

    def test_empty_returns_none(self) -> None:
        assert SliceAggregator.compute_ece([], []) is None

    def test_miscalibrated(self) -> None:
        labels = [0, 0, 0, 0]
        confidences = [0.9, 0.9, 0.9, 0.9]
        result = SliceAggregator.compute_ece(labels, confidences)
        assert result is not None
        assert result > 0.5


class TestSliceAggregator:
    def test_aggregate_overall(self) -> None:
        samples = {
            "s1": _sample("s1", "human"),
            "s2": _sample("s2", "ai_raw"),
        }
        detections = [
            _detection("s1", 0.2),
            _detection("s2", 0.8),
        ]
        agg = SliceAggregator()
        results = agg.aggregate(detections, samples, [])
        # Should have overall only
        assert len(results) == 1
        assert results[0].slice_key.dimension == "overall"
        assert results[0].n_samples == 2
        assert results[0].auroc is not None

    def test_aggregate_with_slices(self) -> None:
        samples = {
            "s1": _sample("s1", "human"),
            "s2": _sample("s2", "ai_raw"),
        }
        detections = [
            _detection("s1", 0.2),
            _detection("s2", 0.8),
        ]
        agg = SliceAggregator()
        results = agg.aggregate(detections, samples, ["source_class"])
        # overall + 2 source_class values
        assert len(results) == 3

    def test_mean_scores(self) -> None:
        samples = {
            "s1": _sample("s1", "human"),
            "s2": _sample("s2", "ai_raw"),
        }
        detections = [
            _detection("s1", 0.2),
            _detection("s2", 0.8),
        ]
        agg = SliceAggregator()
        results = agg.aggregate(detections, samples, [])
        overall = results[0]
        assert overall.mean_score_human == pytest.approx(0.2)
        assert overall.mean_score_ai == pytest.approx(0.8)
