"""Full pipeline latency benchmarks."""
from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lucid.config import load_config
from lucid.models.results import (
    DetectionResult,
    DocumentResult,
    EvaluationResult,
    ParaphraseResult,
)
from lucid.pipeline import LUCIDPipeline, PipelineState


@pytest.mark.benchmark
class TestPipelineLatency:
    """Benchmark pipeline latency with mocked models."""

    @patch("lucid.pipeline.ModelManager")
    @patch("lucid.pipeline.validate_latex")
    @patch("lucid.pipeline.validate_markdown")
    def test_pipeline_latency_ci(
        self,
        mock_validate_md: MagicMock,
        mock_validate_latex: MagicMock,
        mock_model_mgr_cls: MagicMock,
        tmp_path: Path,
        benchmark_collector,  # type: ignore[no-untyped-def]
    ) -> None:
        """CI mode: measure pipeline orchestration overhead."""
        mock_mgr = mock_model_mgr_cls.return_value
        mock_mgr.detector.detect.return_value = DetectionResult(
            chunk_id="c1", ensemble_score=0.8, classification="ai_generated",
        )
        mock_mgr.humanizer.humanize.return_value = ParaphraseResult(
            chunk_id="c1", original_text="Test", humanized_text="Tested",
            iteration_count=1, strategy_used="standard", final_detection_score=0.2,
        )
        mock_mgr.evaluator.evaluate_chunk.return_value = EvaluationResult(
            chunk_id="c1", passed=True, embedding_similarity=0.9,
        )
        mock_validate_md.return_value = MagicMock(valid=True)

        test_file = tmp_path / "bench.md"
        test_file.write_text("# Title\n\nSome AI-generated content.\n", encoding="utf-8")

        config = load_config(profile="balanced")
        pipeline = LUCIDPipeline(config)

        start = time.perf_counter()
        result = pipeline.run(test_file, output_path=tmp_path / "output.md")
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert isinstance(result, DocumentResult)

        benchmark_collector.latency_ms = {
            "pipeline_full_doc": round(elapsed_ms, 1),
            "note": "CI mode with mocked models - orchestration overhead only",
        }

    def test_pipeline_state_machine(self) -> None:
        """Validate pipeline state transitions."""
        states = [s.value for s in PipelineState]
        expected = ["PARSING", "DETECTING", "HUMANIZING", "EVALUATING",
                     "RECONSTRUCTING", "VALIDATING", "COMPLETE", "FAILED"]
        assert states == expected
