"""Tests for experiment runner."""

from __future__ import annotations

from lucid.bench.experiment import ExperimentResult, ExperimentRunner
from lucid.bench.manifests import ExperimentManifest, TransformSpec
from lucid.core.types import SampleRecord


def _manifest() -> ExperimentManifest:
    return ExperimentManifest(
        name="test_exp",
        dataset="data.jsonl",
        detectors=("det_a", "det_b"),
        transforms=(TransformSpec(operator="edit", intensities=(0.5,)),),
        metrics=("m1",),
        slices=("domain",),
        seed=42,
    )


def _samples() -> list[SampleRecord]:
    return [
        SampleRecord(
            sample_id="smp_001",
            split="test",
            domain="academic",
            source_class="human",
            source_model=None,
            document_format="plaintext",
            text="Human text content.",
        ),
        SampleRecord(
            sample_id="smp_002",
            split="test",
            domain="academic",
            source_class="ai_raw",
            source_model="gpt-4.1",
            document_format="plaintext",
            text="AI generated content.",
        ),
    ]


class TestExperimentRunner:
    def test_run_creates_detections(self) -> None:
        runner = ExperimentRunner(_manifest())
        result = runner.run(_samples())

        assert isinstance(result, ExperimentResult)
        assert result.manifest_name == "test_exp"
        # 2 detectors * 2 samples = 4 detections
        assert len(result.detections) == 4
        assert result.duration_seconds >= 0.0
        assert result.timestamp != ""

    def test_run_with_detector(self) -> None:
        runner = ExperimentRunner(_manifest())

        def mock_detect(text: str) -> tuple[float, str, dict[str, float] | None]:
            if "AI" in text:
                return (0.9, "ai_generated", {"feature_a": 0.9})
            return (0.1, "human", {"feature_a": 0.1})

        detections = runner.run_with_detector(_samples(), mock_detect, "test_det")
        assert len(detections) == 2
        assert detections[0].score == 0.1
        assert detections[0].predicted_label == "human"
        assert detections[1].score == 0.9
        assert detections[1].predicted_label == "ai_generated"
        assert detections[1].features == {"feature_a": 0.9}

    def test_run_with_detector_sets_detector_name(self) -> None:
        runner = ExperimentRunner(_manifest())

        def noop_detect(text: str) -> tuple[float, str, dict[str, float] | None]:
            return (0.5, "ambiguous", None)

        detections = runner.run_with_detector(_samples(), noop_detect, "my_det")
        assert all(d.detector_name == "my_det" for d in detections)

    def test_run_produces_metrics(self) -> None:
        runner = ExperimentRunner(_manifest())
        result = runner.run(_samples())
        # Should have at least overall metrics
        assert len(result.metrics) >= 1
        assert result.metrics[0].slice_key.dimension == "overall"
