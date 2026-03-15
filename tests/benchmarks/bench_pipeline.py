"""Full pipeline latency benchmarks."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from lucid.config import load_config
from lucid.models.results import (
    DetectionResult,
    DocumentResult,
    EvaluationResult,
    TransformResult,
)
from lucid.pipeline import LUCIDPipeline, PipelineState

if TYPE_CHECKING:
    from pathlib import Path


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
        mock_mgr.transformer.transform_batch.side_effect = (
            lambda pairs, on_chunk_done=None: [
                TransformResult(
                    chunk_id=chunk.id,
                    original_text=chunk.text,
                    transformed_text="Tested",
                    iteration_count=1,
                    operator_used="standard",
                    final_detection_score=0.2,
                    diagnostics={
                        "placeholder_failures": 1,
                        "semantic_gate_rejections": 2,
                        "retries_used": 1,
                    },
                )
                for chunk, _det in pairs
            ]
        )
        mock_mgr.evaluator.evaluate_chunk.return_value = EvaluationResult(
            chunk_id="c1", passed=True, embedding_similarity=0.9,
            diagnostics={"terminal_stage": "passed", "rejected_at": None},
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
        benchmark_collector.failure_modes["pipeline_summary_shape"] = {
            "operator_usage_present": "operator_usage" in result.summary_stats,
            "search_diagnostics_present": "search_diagnostics" in result.summary_stats,
            "evaluation_rejection_stages_present": (
                "evaluation_rejection_stages" in result.summary_stats
            ),
        }

    def test_pipeline_state_machine(self) -> None:
        """Validate pipeline state transitions."""
        states = [s.value for s in PipelineState]
        expected = ["PARSING", "DETECTING", "TRANSFORMING", "EVALUATING",
                     "RECONSTRUCTING", "VALIDATING", "COMPLETE", "FAILED"]
        assert states == expected
