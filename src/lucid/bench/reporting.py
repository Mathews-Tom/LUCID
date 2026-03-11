"""Output formatters for benchmark results."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from io import StringIO
from pathlib import Path

from lucid.bench.aggregation import AggregatedMetrics
from lucid.bench.experiment import ExperimentResult
from lucid.core.types import DetectionRecord


class ReportWriter:
    """Write benchmark results in various formats."""

    @staticmethod
    def write_detections_jsonl(
        detections: Sequence[DetectionRecord], path: Path
    ) -> None:
        """Write detection records to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for det in detections:
                fh.write(json.dumps(det.to_dict()) + "\n")

    @staticmethod
    def write_metrics_csv(
        metrics: Sequence[AggregatedMetrics], path: Path
    ) -> None:
        """Write aggregated metrics to a CSV file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "dimension",
            "value",
            "n_samples",
            "auroc",
            "auprc",
            "tpr_at_fpr5",
            "human_fpr",
            "mean_score_human",
            "mean_score_ai",
            "calibration_error",
        ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for m in metrics:
                writer.writerow({
                    "dimension": m.slice_key.dimension,
                    "value": m.slice_key.value,
                    "n_samples": m.n_samples,
                    "auroc": _fmt_float(m.auroc),
                    "auprc": _fmt_float(m.auprc),
                    "tpr_at_fpr5": _fmt_float(m.tpr_at_fpr5),
                    "human_fpr": _fmt_float(m.human_fpr),
                    "mean_score_human": _fmt_float(m.mean_score_human),
                    "mean_score_ai": _fmt_float(m.mean_score_ai),
                    "calibration_error": _fmt_float(m.calibration_error),
                })

    @staticmethod
    def write_summary_markdown(
        result: ExperimentResult, path: Path
    ) -> None:
        """Write a markdown summary of experiment results."""
        path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append(f"# Benchmark Report: {result.manifest_name}")
        lines.append("")
        lines.append(f"- **Timestamp**: {result.timestamp}")
        lines.append(f"- **Duration**: {result.duration_seconds:.2f}s")
        lines.append(f"- **Total detections**: {len(result.detections)}")
        lines.append("")

        # Overall metrics
        overall = [m for m in result.metrics if m.slice_key.dimension == "overall"]
        if overall:
            lines.append("## Overall Metrics")
            lines.append("")
            lines.append(_metrics_table(overall))
            lines.append("")

        # Per-slice metrics
        slice_dims: dict[str, list[AggregatedMetrics]] = {}
        for m in result.metrics:
            if m.slice_key.dimension != "overall":
                slice_dims.setdefault(m.slice_key.dimension, []).append(m)

        for dim, slice_metrics in sorted(slice_dims.items()):
            lines.append(f"## Slice: {dim}")
            lines.append("")
            lines.append(_metrics_table(slice_metrics))
            lines.append("")

        path.write_text("\n".join(lines), encoding="utf-8")


def _fmt_float(v: float | None) -> str:
    if v is None:
        return ""
    return f"{v:.4f}"


def _metrics_table(metrics: list[AggregatedMetrics]) -> str:
    """Render a markdown table for a list of AggregatedMetrics."""
    header = "| Value | N | AUROC | AUPRC | TPR@FPR5 | Human FPR | Mean Human | Mean AI | ECE |"
    sep = "|---|---|---|---|---|---|---|---|---|"
    rows = [header, sep]
    for m in metrics:
        rows.append(
            f"| {m.slice_key.value} | {m.n_samples} "
            f"| {_fmt_float(m.auroc)} | {_fmt_float(m.auprc)} "
            f"| {_fmt_float(m.tpr_at_fpr5)} | {_fmt_float(m.human_fpr)} "
            f"| {_fmt_float(m.mean_score_human)} | {_fmt_float(m.mean_score_ai)} "
            f"| {_fmt_float(m.calibration_error)} |"
        )
    return "\n".join(rows)
