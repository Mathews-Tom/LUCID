"""Tests for lucid.core.types benchmark record dataclasses."""

from __future__ import annotations

import dataclasses

import pytest

from lucid.core.types import (
    DetectionRecord,
    MetricRecord,
    MetricResult,
    SampleRecord,
    TransformationRecord,
)


# -- SampleRecord -----------------------------------------------------------

def _sample_kwargs() -> dict:
    return {
        "sample_id": "s1",
        "split": "train",
        "domain": "academic_stem",
        "source_class": "human",
        "source_model": None,
        "document_format": "latex",
        "text": "Hello world.",
    }


class TestSampleRecord:
    def test_valid_construction(self) -> None:
        rec = SampleRecord(**_sample_kwargs())
        assert rec.sample_id == "s1"
        assert rec.metadata == {}

    @pytest.mark.parametrize("field,value", [
        ("split", "invalid"),
        ("source_class", "robot"),
        ("document_format", "docx"),
    ])
    def test_invalid_enum_fields(self, field: str, value: str) -> None:
        kw = _sample_kwargs()
        kw[field] = value
        with pytest.raises(ValueError):
            SampleRecord(**kw)

    def test_frozen(self) -> None:
        rec = SampleRecord(**_sample_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.text = "changed"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        rec = SampleRecord(**_sample_kwargs())
        assert SampleRecord.from_dict(rec.to_dict()) == rec

    def test_roundtrip_with_metadata(self) -> None:
        kw = _sample_kwargs()
        kw["metadata"] = {"key": "value"}
        rec = SampleRecord(**kw)
        assert SampleRecord.from_dict(rec.to_dict()) == rec


# -- TransformationRecord ---------------------------------------------------

def _transform_kwargs() -> dict:
    return {
        "transform_id": "t1",
        "parent_sample_id": "s1",
        "operator": "paraphrase",
        "intensity": 0.5,
        "config_hash": "abc123",
        "output_text": "Transformed text.",
    }


class TestTransformationRecord:
    def test_valid_construction(self) -> None:
        rec = TransformationRecord(**_transform_kwargs())
        assert rec.intensity == 0.5

    @pytest.mark.parametrize("intensity", [-0.1, 1.1])
    def test_invalid_intensity(self, intensity: float) -> None:
        kw = _transform_kwargs()
        kw["intensity"] = intensity
        with pytest.raises(ValueError):
            TransformationRecord(**kw)

    def test_boundary_intensity(self) -> None:
        for val in (0.0, 1.0):
            kw = _transform_kwargs()
            kw["intensity"] = val
            TransformationRecord(**kw)

    def test_frozen(self) -> None:
        rec = TransformationRecord(**_transform_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.operator = "x"  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        rec = TransformationRecord(**_transform_kwargs())
        assert TransformationRecord.from_dict(rec.to_dict()) == rec


# -- DetectionRecord ---------------------------------------------------------

def _detection_kwargs() -> dict:
    return {
        "record_id": "d1",
        "sample_id": "s1",
        "transform_id": None,
        "detector_name": "roberta",
        "score": 0.85,
        "confidence": 0.9,
        "threshold": 0.5,
        "predicted_label": "ai_generated",
    }


class TestDetectionRecord:
    def test_valid_construction(self) -> None:
        rec = DetectionRecord(**_detection_kwargs())
        assert rec.score == 0.85

    @pytest.mark.parametrize("score", [-0.1, 1.1])
    def test_invalid_score(self, score: float) -> None:
        kw = _detection_kwargs()
        kw["score"] = score
        with pytest.raises(ValueError):
            DetectionRecord(**kw)

    def test_invalid_predicted_label(self) -> None:
        kw = _detection_kwargs()
        kw["predicted_label"] = "unknown"
        with pytest.raises(ValueError):
            DetectionRecord(**kw)

    def test_frozen(self) -> None:
        rec = DetectionRecord(**_detection_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.score = 0.1  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        kw = _detection_kwargs()
        kw["features"] = {"burstiness": 0.3}
        kw["evidence"] = {"key": "val"}
        rec = DetectionRecord(**kw)
        assert DetectionRecord.from_dict(rec.to_dict()) == rec


# -- MetricRecord ------------------------------------------------------------

def _metric_kwargs() -> dict:
    return {
        "record_id": "m1",
        "sample_id": "s1",
        "transform_id": None,
        "metric_name": "bertscore",
        "value": 0.92,
    }


class TestMetricRecord:
    def test_valid_construction(self) -> None:
        rec = MetricRecord(**_metric_kwargs())
        assert rec.value == 0.92

    def test_frozen(self) -> None:
        rec = MetricRecord(**_metric_kwargs())
        with pytest.raises(dataclasses.FrozenInstanceError):
            rec.value = 0.0  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        rec = MetricRecord(**_metric_kwargs())
        assert MetricRecord.from_dict(rec.to_dict()) == rec

    def test_roundtrip_with_metadata(self) -> None:
        kw = _metric_kwargs()
        kw["metadata"] = {"extra": 42}
        rec = MetricRecord(**kw)
        assert MetricRecord.from_dict(rec.to_dict()) == rec


# -- MetricResult ------------------------------------------------------------

class TestMetricResult:
    def test_valid_construction(self) -> None:
        res = MetricResult(metric_name="bleu", value=0.75)
        assert res.metadata == {}

    def test_frozen(self) -> None:
        res = MetricResult(metric_name="bleu", value=0.75)
        with pytest.raises(dataclasses.FrozenInstanceError):
            res.value = 0.0  # type: ignore[misc]

    def test_roundtrip(self) -> None:
        res = MetricResult(metric_name="bleu", value=0.75, metadata={"n": 4})
        assert MetricResult.from_dict(res.to_dict()) == res
