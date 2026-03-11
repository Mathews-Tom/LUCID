"""Core experiment orchestrator for benchmark runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from lucid.bench.aggregation import AggregatedMetrics, SliceAggregator
from lucid.bench.manifests import ExperimentManifest
from lucid.core.types import DetectionRecord, SampleRecord


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """Complete result of a benchmark experiment."""

    manifest_name: str
    detections: tuple[DetectionRecord, ...]
    metrics: tuple[AggregatedMetrics, ...]
    timestamp: str
    duration_seconds: float


class ExperimentRunner:
    """Run benchmark experiments defined by a manifest."""

    def __init__(self, manifest: ExperimentManifest) -> None:
        self._manifest = manifest
        self._aggregator = SliceAggregator()

    def run(self, samples: list[SampleRecord]) -> ExperimentResult:
        """Run the experiment with placeholder detection (no model loading).

        Creates DetectionRecord entries with score=0.5 and label='ambiguous'
        for each sample/detector pair. Use run_with_detector for actual detection.
        """
        start = time.monotonic()
        all_detections: list[DetectionRecord] = []

        for detector_name in self._manifest.detectors:
            for sample in samples:
                record = DetectionRecord(
                    record_id=f"det_{uuid4().hex[:8]}",
                    sample_id=sample.sample_id,
                    transform_id=None,
                    detector_name=detector_name,
                    score=0.5,
                    confidence=None,
                    threshold=None,
                    predicted_label="ambiguous",
                )
                all_detections.append(record)

        samples_map = {s.sample_id: s for s in samples}
        metrics = self._aggregator.aggregate(
            all_detections, samples_map, list(self._manifest.slices)
        )
        duration = time.monotonic() - start

        return ExperimentResult(
            manifest_name=self._manifest.name,
            detections=tuple(all_detections),
            metrics=tuple(metrics),
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=duration,
        )

    def run_with_detector(
        self,
        samples: list[SampleRecord],
        detect_fn: Callable[[str], tuple[float, str, dict[str, float] | None]],
        detector_name: str,
    ) -> list[DetectionRecord]:
        """Run detection on samples using a provided callable.

        Args:
            samples: List of samples to detect.
            detect_fn: Callable taking text, returning (score, label, features).
            detector_name: Name to record for this detector.

        Returns:
            List of DetectionRecord entries.
        """
        detections: list[DetectionRecord] = []

        for sample in samples:
            score, label, features = detect_fn(sample.text)
            record = DetectionRecord(
                record_id=f"det_{uuid4().hex[:8]}",
                sample_id=sample.sample_id,
                transform_id=None,
                detector_name=detector_name,
                score=score,
                confidence=None,
                threshold=None,
                predicted_label=label,
                features=features,
            )
            detections.append(record)

        return detections
