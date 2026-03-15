"""Transform effectiveness benchmarks."""
from __future__ import annotations

import pytest

from lucid.models.results import TransformResult


@pytest.mark.benchmark
class TestTransformEffectiveness:
    """Benchmark transform effectiveness rates."""

    def test_effectiveness_rate_structure(self, benchmark_collector) -> None:  # type: ignore[no-untyped-def]
        """CI mode: validate effectiveness result structure."""
        single_pass = [
            TransformResult(
                chunk_id=f"chunk-{i}", original_text=f"Text {i}",
                transformed_text=f"Transformed {i}", iteration_count=1,
                operator_used="standard", final_detection_score=score,
                diagnostics={"placeholder_failures": 0, "semantic_gate_rejections": 0},
            )
            for i, score in enumerate([0.22, 0.45, 0.18, 0.55, 0.12, 0.38, 0.09])
        ]

        search_results = [
            TransformResult(
                chunk_id=f"search-{i}", original_text=f"Text {i}",
                transformed_text=f"Transformed {i}", iteration_count=count,
                operator_used=op_name, final_detection_score=score,
                diagnostics={
                    "placeholder_failures": i % 2,
                    "semantic_gate_rejections": (i + 1) % 3,
                    "retries_used": i % 2,
                },
            )
            for i, (score, count, op_name) in enumerate([
                (0.15, 3, "restructure"), (0.22, 2, "standard"),
                (0.08, 4, "voice_shift"), (0.31, 5, "vocabulary"),
                (0.11, 3, "reorder"), (0.19, 2, "standard"),
                (0.05, 4, "restructure"),
            ])
        ]

        threshold = 0.30
        sp_evaded = sum(1 for r in single_pass if r.final_detection_score < threshold)
        sp_rate = sp_evaded / len(single_pass)
        search_evaded = sum(1 for r in search_results if r.final_detection_score < threshold)
        search_rate = search_evaded / len(search_results)

        assert 0.0 <= sp_rate <= 1.0
        assert 0.0 <= search_rate <= 1.0

        operators: dict[str, int] = {}
        for r in search_results:
            operators[r.operator_used] = operators.get(r.operator_used, 0) + 1

        benchmark_collector.transform_quality = {
            "single_pass_effectiveness_rate": round(sp_rate, 3),
            "search_effectiveness_rate": round(search_rate, 3),
            "mean_iterations": round(
                sum(r.iteration_count for r in search_results) / len(search_results), 1
            ),
            "operator_distribution": operators,
            "placeholder_failures": sum(
                int(r.diagnostics.get("placeholder_failures", 0)) for r in search_results
            ),
            "semantic_gate_rejections": sum(
                int(r.diagnostics.get("semantic_gate_rejections", 0))
                for r in search_results
            ),
        }

        benchmark_collector.failure_modes = {
            "chunks_with_placeholder_failures": sum(
                1
                for r in search_results
                if int(r.diagnostics.get("placeholder_failures", 0)) > 0
            ),
            "retries_used": sum(int(r.diagnostics.get("retries_used", 0)) for r in search_results),
        }

    def test_transform_result_validity(self) -> None:
        """Validate TransformResult invariants."""
        result = TransformResult(
            chunk_id="bench-001", original_text="Quick brown fox",
            transformed_text="Swift brown fox", iteration_count=2,
            operator_used="standard", final_detection_score=0.25,
        )
        assert 0.0 <= result.final_detection_score <= 1.0
        assert result.iteration_count >= 0
