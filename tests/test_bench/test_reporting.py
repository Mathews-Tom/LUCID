"""Tests for benchmark report writing."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from lucid.bench.aggregation import AggregatedMetrics
from lucid.bench.experiment import ExperimentResult
from lucid.bench.reporting import ReportWriter
from lucid.bench.slices import SliceKey
from lucid.core.types import DetectionRecord


def _detection(sample_id: str = "smp_001", score: float = 0.5) -> DetectionRecord:
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


def _metrics() -> list[AggregatedMetrics]:
    return [
        AggregatedMetrics(
            slice_key=SliceKey(dimension="overall", value="all"),
            n_samples=100,
            auroc=0.95,
            auprc=0.90,
            tpr_at_fpr5=0.80,
            human_fpr=0.05,
            mean_score_human=0.2,
            mean_score_ai=0.8,
            calibration_error=0.03,
        ),
        AggregatedMetrics(
            slice_key=SliceKey(dimension="domain", value="academic"),
            n_samples=50,
            auroc=0.92,
            auprc=None,
            tpr_at_fpr5=0.75,
            human_fpr=0.06,
            mean_score_human=0.25,
            mean_score_ai=0.85,
            calibration_error=0.04,
        ),
    ]


def _result() -> ExperimentResult:
    return ExperimentResult(
        manifest_name="test_exp",
        detections=(_detection("s1", 0.3), _detection("s2", 0.8)),
        metrics=tuple(_metrics()),
        timestamp="2026-01-01T00:00:00+00:00",
        duration_seconds=12.5,
    )


class TestWriteDetectionsJsonl:
    def test_writes_valid_jsonl(self, tmp_path: Path) -> None:
        dets = [_detection("s1"), _detection("s2")]
        path = tmp_path / "detections.jsonl"
        ReportWriter.write_detections_jsonl(dets, path)

        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            data = json.loads(line)
            assert "record_id" in data
            assert "detector_name" in data

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "sub" / "dir" / "detections.jsonl"
        ReportWriter.write_detections_jsonl([_detection()], path)
        assert path.exists()


class TestWriteMetricsCsv:
    def test_writes_valid_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "metrics.csv"
        ReportWriter.write_metrics_csv(_metrics(), path)

        with path.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["dimension"] == "overall"
        assert rows[0]["auroc"] == "0.9500"
        assert rows[1]["auprc"] == ""  # None value

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "nested" / "metrics.csv"
        ReportWriter.write_metrics_csv(_metrics(), path)
        assert path.exists()


class TestWriteSummaryMarkdown:
    def test_writes_markdown(self, tmp_path: Path) -> None:
        path = tmp_path / "summary.md"
        ReportWriter.write_summary_markdown(_result(), path)
        content = path.read_text(encoding="utf-8")

        assert "# Benchmark Report: test_exp" in content
        assert "2026-01-01" in content
        assert "12.50s" in content
        assert "Overall Metrics" in content
        assert "domain" in content

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        path = tmp_path / "deep" / "summary.md"
        ReportWriter.write_summary_markdown(_result(), path)
        assert path.exists()
