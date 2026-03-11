"""Unified detection runner with structured output."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from lucid.models.results import DetectionResult

if TYPE_CHECKING:
    from lucid.detector.base import LUCIDDetector
    from lucid.detector.calibrate import Calibrator
    from lucid.detector.explain import DetectionExplainer
    from lucid.parser.chunk import ProseChunk


@dataclass(frozen=True, slots=True)
class DetectionOutput:
    """Rich detection output with optional calibration and evidence."""

    detector_name: str
    label: str
    score: float
    confidence: float | None = None
    calibration_version: str | None = None
    features: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_detection_result(self, chunk_id: str) -> DetectionResult:
        """Convert to pipeline-compatible DetectionResult."""
        return DetectionResult(
            chunk_id=chunk_id,
            ensemble_score=self.score,
            classification=self.label,
            feature_details=self.features,
        )


class DetectorRunner:
    """Execute detection through registered backends.

    Wraps LUCIDDetector to produce DetectionOutput with
    optional calibration and evidence attachments.
    """

    def __init__(
        self,
        detector: LUCIDDetector,
        calibrator: Calibrator | None = None,
        explainer: DetectionExplainer | None = None,
    ) -> None:
        self._detector = detector
        self._calibrator = calibrator
        self._explainer = explainer

    def run(self, chunk: ProseChunk) -> DetectionOutput:
        """Run detection with optional calibration and explanation."""
        result = self._detector.detect(chunk)

        confidence = None
        calibration_version = None
        if self._calibrator is not None:
            confidence = self._calibrator.calibrate(result.ensemble_score)
            calibration_version = self._calibrator.version

        evidence: dict[str, Any] = {}
        if self._explainer is not None:
            evidence = self._explainer.explain(result)

        return DetectionOutput(
            detector_name="ensemble",
            label=result.classification,
            score=result.ensemble_score,
            confidence=confidence,
            calibration_version=calibration_version,
            features=result.feature_details,
            evidence=evidence,
        )
