"""Tests for slice grouping and intensity bucketing."""

from __future__ import annotations

import pytest

from lucid.bench.slices import (
    VALID_SLICE_DIMENSIONS,
    SliceKey,
    bucket_intensity,
    group_by_slice,
)
from lucid.core.errors import BenchmarkError
from lucid.core.types import DetectionRecord, SampleRecord


def _sample(
    sample_id: str = "smp_001",
    domain: str = "academic",
    source_class: str = "human",
    source_model: str | None = None,
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        split="test",
        domain=domain,
        source_class=source_class,
        source_model=source_model,
        document_format="plaintext",
        text="text",
    )


def _detection(
    sample_id: str = "smp_001",
    detector_name: str = "roberta",
    score: float = 0.5,
    evidence: dict | None = None,
) -> DetectionRecord:
    return DetectionRecord(
        record_id="det_001",
        sample_id=sample_id,
        transform_id=None,
        detector_name=detector_name,
        score=score,
        confidence=None,
        threshold=None,
        predicted_label="ambiguous",
        evidence=evidence,
    )


class TestGroupBySlice:
    def test_group_by_domain(self) -> None:
        samples = {
            "s1": _sample("s1", domain="academic"),
            "s2": _sample("s2", domain="news"),
            "s3": _sample("s3", domain="academic"),
        }
        dets = [
            _detection("s1"),
            _detection("s2"),
            _detection("s3"),
        ]
        groups = group_by_slice(dets, samples, "domain")
        assert len(groups["academic"]) == 2
        assert len(groups["news"]) == 1

    def test_group_by_detector(self) -> None:
        samples = {"s1": _sample("s1")}
        dets = [
            _detection("s1", detector_name="roberta"),
            _detection("s1", detector_name="statistical"),
        ]
        groups = group_by_slice(dets, samples, "detector")
        assert "roberta" in groups
        assert "statistical" in groups

    def test_group_by_source_class(self) -> None:
        samples = {
            "s1": _sample("s1", source_class="human"),
            "s2": _sample("s2", source_class="ai_raw"),
        }
        dets = [_detection("s1"), _detection("s2")]
        groups = group_by_slice(dets, samples, "source_class")
        assert len(groups) == 2

    def test_invalid_dimension_raises(self) -> None:
        with pytest.raises(BenchmarkError, match="Invalid slice dimension"):
            group_by_slice([], {}, "invalid_dim")

    def test_missing_sample_skipped(self) -> None:
        dets = [_detection("missing_id")]
        groups = group_by_slice(dets, {}, "domain")
        assert len(groups) == 0

    def test_group_by_operator(self) -> None:
        samples = {"s1": _sample("s1")}
        dets = [
            _detection("s1", evidence={"operator": "surface_edit"}),
            _detection("s1", evidence={"operator": "lexical"}),
        ]
        groups = group_by_slice(dets, samples, "operator")
        assert "surface_edit" in groups
        assert "lexical" in groups

    def test_group_by_intensity_bucket(self) -> None:
        samples = {"s1": _sample("s1")}
        dets = [
            _detection("s1", evidence={"intensity": 0.15}),
            _detection("s1", evidence={"intensity": 0.85}),
        ]
        groups = group_by_slice(dets, samples, "intensity_bucket")
        assert len(groups) == 2


class TestBucketIntensity:
    def test_buckets_5(self) -> None:
        assert bucket_intensity(0.0) == "0.00-0.20"
        assert bucket_intensity(0.15) == "0.00-0.20"
        assert bucket_intensity(0.5) == "0.40-0.60"
        assert bucket_intensity(1.0) == "0.80-1.00"

    def test_buckets_10(self) -> None:
        assert bucket_intensity(0.05, n_buckets=10) == "0.00-0.10"
        assert bucket_intensity(0.95, n_buckets=10) == "0.90-1.00"

    def test_zero_buckets_raises(self) -> None:
        with pytest.raises(BenchmarkError, match="n_buckets must be >= 1"):
            bucket_intensity(0.5, n_buckets=0)
