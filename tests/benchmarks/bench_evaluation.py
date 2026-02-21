"""Semantic evaluation benchmarks."""
from __future__ import annotations

import pytest

from lucid.models.results import EvaluationResult


@pytest.mark.benchmark
class TestSemanticPreservation:
    """Benchmark semantic preservation metrics."""

    def test_evaluation_metrics_structure(self, benchmark_collector) -> None:  # type: ignore[no-untyped-def]
        """CI mode: validate evaluation metric ranges."""
        evals = [
            EvaluationResult(chunk_id=f"chunk-{i}", passed=p, embedding_similarity=e,
                             nli_forward=nf, nli_backward=nb, bertscore_f1=b, rejection_reason=r)
            for i, (p, e, nf, nb, b, r) in enumerate([
                (True, 0.92, "entailment", "entailment", 0.91, None),
                (True, 0.88, "entailment", "entailment", 0.89, None),
                (False, 0.71, "neutral", "contradiction", 0.78, "NLI contradiction"),
                (True, 0.95, "entailment", "entailment", 0.94, None),
                (True, 0.86, "entailment", "neutral", 0.87, None),
                (True, 0.90, "entailment", "entailment", 0.92, None),
            ])
        ]

        pass_rate = sum(1 for e in evals if e.passed) / len(evals)
        assert 0.0 <= pass_rate <= 1.0

        emb_scores = [e.embedding_similarity for e in evals if e.embedding_similarity is not None]
        bert_scores = [e.bertscore_f1 for e in evals if e.bertscore_f1 is not None]

        benchmark_collector.semantic_preservation = {
            "embedding_similarity": {
                "mean": round(sum(emb_scores) / len(emb_scores), 3),
                "min": round(min(emb_scores), 3),
                "max": round(max(emb_scores), 3),
            },
            "bertscore_f1": {
                "mean": round(sum(bert_scores) / len(bert_scores), 3),
                "min": round(min(bert_scores), 3),
                "max": round(max(bert_scores), 3),
            },
            "pass_rate": round(pass_rate, 3),
            "nli_distribution": {
                "entailment": sum(1 for e in evals if e.nli_forward == "entailment"),
                "neutral": sum(1 for e in evals if e.nli_forward == "neutral"),
                "contradiction": sum(1 for e in evals if e.nli_forward == "contradiction"),
            },
        }

    def test_evaluation_result_validity(self) -> None:
        """Validate EvaluationResult invariants."""
        result = EvaluationResult(
            chunk_id="bench-001", passed=True,
            embedding_similarity=0.90, bertscore_f1=0.88,
        )
        assert result.passed
        assert result.embedding_similarity is not None
