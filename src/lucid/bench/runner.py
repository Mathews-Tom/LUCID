"""Batch runner over multiple manifests and profiles."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from lucid.bench.aggregation import SliceAggregator
from lucid.bench.datasets import DatasetLoader
from lucid.bench.experiment import ExperimentResult, ExperimentRunner
from lucid.bench.manifests import ExperimentManifest
from lucid.bench.reporting import ReportWriter


class BenchRunner:
    """Run benchmark experiments from manifest files and persist results."""

    def __init__(self, results_dir: Path) -> None:
        self._results_dir = results_dir

    def run_manifest(
        self,
        manifest_path: Path,
        detect_fn: Callable[[str], tuple[float, str, dict[str, float] | None]],
        detector_name: str,
    ) -> ExperimentResult:
        """Load a manifest, load its dataset, run detection, aggregate, and return results."""
        manifest = ExperimentManifest.from_yaml(manifest_path)
        dataset_path = Path(manifest.dataset)

        if dataset_path.is_dir():
            samples = DatasetLoader.load_corpus(dataset_path)
        else:
            samples = DatasetLoader.load_jsonl(dataset_path)

        runner = ExperimentRunner(manifest)
        detections = runner.run_with_detector(samples, detect_fn, detector_name)

        samples_map = {s.sample_id: s for s in samples}
        aggregator = SliceAggregator()
        metrics = aggregator.aggregate(detections, samples_map, list(manifest.slices))

        return ExperimentResult(
            manifest_name=manifest.name,
            detections=tuple(detections),
            metrics=tuple(metrics),
            timestamp=datetime.now(timezone.utc).isoformat(),
            duration_seconds=0.0,
        )

    def save_results(self, result: ExperimentResult, output_dir: Path) -> None:
        """Save detections.jsonl, metrics.csv, and summary.md to output_dir."""
        output_dir.mkdir(parents=True, exist_ok=True)

        ReportWriter.write_detections_jsonl(
            result.detections, output_dir / "detections.jsonl"
        )
        ReportWriter.write_metrics_csv(
            result.metrics, output_dir / "metrics.csv"
        )
        ReportWriter.write_summary_markdown(
            result, output_dir / "summary.md"
        )
