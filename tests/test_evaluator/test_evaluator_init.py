"""Tests for the LUCIDEvaluator facade."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from lucid.core.protocols import Evaluator
from lucid.evaluator import LUCIDEvaluator
from lucid.models.results import EvaluationResult


def _make_config() -> MagicMock:
    """Build a mock EvaluatorConfig."""
    cfg = MagicMock()
    cfg.embedding_model = "mock-embed"
    cfg.embedding_threshold = 0.80
    cfg.nli_model = "mock-nli"
    cfg.nli_require_bidirectional = True
    cfg.bertscore_model = "mock-bert"
    cfg.bertscore_threshold = 0.88
    return cfg


class TestLUCIDEvaluator:
    """Unit tests for the LUCIDEvaluator facade — pipeline is mocked."""

    @patch("lucid.evaluator.EvaluationPipeline")
    def test_satisfies_evaluator_protocol(self, _mock_pipeline_cls: MagicMock) -> None:
        """LUCIDEvaluator is an instance of the Evaluator protocol."""
        evaluator = LUCIDEvaluator(_make_config())
        assert isinstance(evaluator, Evaluator)

    @patch("lucid.evaluator.EvaluationPipeline")
    def test_evaluate_delegates_to_pipeline(self, mock_pipeline_cls: MagicMock) -> None:
        """evaluate() calls pipeline.run with protocol chunk ID."""
        expected = EvaluationResult(chunk_id="__protocol__", passed=True)
        mock_pipeline_cls.return_value.run.return_value = expected

        evaluator = LUCIDEvaluator(_make_config())
        result = evaluator.evaluate("original", "paraphrase")

        assert result is expected
        mock_pipeline_cls.return_value.run.assert_called_once()
        call_kwargs = mock_pipeline_cls.return_value.run.call_args
        assert call_kwargs[1]["chunk_id"] == "__protocol__"
        assert call_kwargs[1]["original"] == "original"
        assert call_kwargs[1]["paraphrase"] == "paraphrase"

    @patch("lucid.evaluator.EvaluationPipeline")
    def test_evaluate_chunk_propagates_chunk_id(self, mock_pipeline_cls: MagicMock) -> None:
        """evaluate_chunk() passes the provided chunk_id to the pipeline."""
        expected = EvaluationResult(chunk_id="chunk_42", passed=True)
        mock_pipeline_cls.return_value.run.return_value = expected

        evaluator = LUCIDEvaluator(_make_config())
        result = evaluator.evaluate_chunk("chunk_42", "orig", "para")

        assert result is expected
        call_kwargs = mock_pipeline_cls.return_value.run.call_args
        assert call_kwargs[1]["chunk_id"] == "chunk_42"

    @patch("lucid.evaluator.EvaluationPipeline")
    def test_balanced_profile_skips_bertscore(self, mock_pipeline_cls: MagicMock) -> None:
        """Balanced profile sets run_bertscore=False."""
        mock_pipeline_cls.return_value.run.return_value = EvaluationResult(
            chunk_id="__protocol__",
            passed=True,
        )

        evaluator = LUCIDEvaluator(_make_config(), profile="balanced")
        evaluator.evaluate("a", "b")

        opts = mock_pipeline_cls.return_value.run.call_args[1]["options"]
        assert opts.run_bertscore is False

    @patch("lucid.evaluator.EvaluationPipeline")
    def test_quality_profile_enables_bertscore(self, mock_pipeline_cls: MagicMock) -> None:
        """Quality profile sets run_bertscore=True."""
        mock_pipeline_cls.return_value.run.return_value = EvaluationResult(
            chunk_id="__protocol__",
            passed=True,
        )

        evaluator = LUCIDEvaluator(_make_config(), profile="quality")
        evaluator.evaluate("a", "b")

        opts = mock_pipeline_cls.return_value.run.call_args[1]["options"]
        assert opts.run_bertscore is True

    @patch("lucid.evaluator.EvaluationPipeline")
    def test_fast_profile_skips_bertscore(self, mock_pipeline_cls: MagicMock) -> None:
        """Fast profile sets run_bertscore=False."""
        mock_pipeline_cls.return_value.run.return_value = EvaluationResult(
            chunk_id="__protocol__",
            passed=True,
        )

        evaluator = LUCIDEvaluator(_make_config(), profile="fast")
        evaluator.evaluate("a", "b")

        opts = mock_pipeline_cls.return_value.run.call_args[1]["options"]
        assert opts.run_bertscore is False
