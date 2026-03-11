"""Tests for the unified detection runner."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import MagicMock

import pytest

from lucid.detector.runner import DetectionOutput, DetectorRunner


@dataclass
class _FakeDetectionResult:
    """Minimal stand-in for DetectionResult to avoid heavy imports."""

    chunk_id: str
    ensemble_score: float
    classification: str
    feature_details: dict[str, Any] = field(default_factory=dict)


class TestDetectionOutput:
    def test_to_detection_result_basic(self) -> None:
        output = DetectionOutput(
            detector_name="ensemble",
            label="ai_generated",
            score=0.85,
            features={"lm_perplexity_mean": 30.0},
        )
        result = output.to_detection_result("chunk-1")
        assert result.chunk_id == "chunk-1"
        assert result.ensemble_score == 0.85
        assert result.classification == "ai_generated"
        assert result.feature_details == {"lm_perplexity_mean": 30.0}

    def test_to_detection_result_with_calibration_fields(self) -> None:
        output = DetectionOutput(
            detector_name="ensemble",
            label="human",
            score=0.2,
            confidence=0.15,
            calibration_version="v2",
        )
        result = output.to_detection_result("chunk-2")
        assert result.ensemble_score == 0.2
        assert result.classification == "human"

    def test_frozen(self) -> None:
        output = DetectionOutput(
            detector_name="test", label="human", score=0.1
        )
        with pytest.raises(AttributeError):
            output.score = 0.5  # type: ignore[misc]


class TestDetectorRunner:
    def _make_runner(
        self,
        score: float = 0.7,
        classification: str = "ai_generated",
        features: dict[str, Any] | None = None,
        calibrator: Any = None,
        explainer: Any = None,
    ) -> DetectorRunner:
        detector = MagicMock()
        detector.detect.return_value = _FakeDetectionResult(
            chunk_id="c1",
            ensemble_score=score,
            classification=classification,
            feature_details=features or {},
        )
        return DetectorRunner(
            detector=detector,
            calibrator=calibrator,
            explainer=explainer,
        )

    def test_run_no_calibrator_no_explainer(self) -> None:
        runner = self._make_runner(score=0.6, classification="ambiguous")
        chunk = MagicMock()
        output = runner.run(chunk)
        assert output.score == 0.6
        assert output.label == "ambiguous"
        assert output.confidence is None
        assert output.calibration_version is None
        assert output.evidence == {}

    def test_run_with_calibrator(self) -> None:
        calibrator = MagicMock()
        calibrator.calibrate.return_value = 0.75
        calibrator.version = "v1"
        runner = self._make_runner(score=0.7, calibrator=calibrator)
        chunk = MagicMock()
        output = runner.run(chunk)
        assert output.confidence == 0.75
        assert output.calibration_version == "v1"
        calibrator.calibrate.assert_called_once_with(0.7)

    def test_run_with_explainer(self) -> None:
        explainer = MagicMock()
        explainer.explain.return_value = {
            "top_signals": ["lm_perplexity_mean"],
            "summary": "test",
        }
        runner = self._make_runner(explainer=explainer)
        chunk = MagicMock()
        output = runner.run(chunk)
        assert output.evidence["top_signals"] == ["lm_perplexity_mean"]

    def test_run_with_both(self) -> None:
        calibrator = MagicMock()
        calibrator.calibrate.return_value = 0.9
        calibrator.version = "v2"
        explainer = MagicMock()
        explainer.explain.return_value = {"summary": "all AI"}
        runner = self._make_runner(
            score=0.85,
            classification="ai_generated",
            calibrator=calibrator,
            explainer=explainer,
        )
        chunk = MagicMock()
        output = runner.run(chunk)
        assert output.confidence == 0.9
        assert output.calibration_version == "v2"
        assert output.evidence == {"summary": "all AI"}
        assert output.detector_name == "ensemble"

    def test_run_preserves_features(self) -> None:
        features = {"style_ttr": 0.45, "disc_bigram_entropy": 5.0}
        runner = self._make_runner(features=features)
        chunk = MagicMock()
        output = runner.run(chunk)
        assert output.features == features
