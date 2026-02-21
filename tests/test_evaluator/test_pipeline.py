"""Tests for the evaluation pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lucid.evaluator.bertscore import BERTScoreResult
from lucid.evaluator.embedding import EmbeddingResult
from lucid.evaluator.nli import NLIResult
from lucid.evaluator.pipeline import EvaluationPipeline, PipelineOptions
from lucid.evaluator.term_verify import TermVerificationResult


def _make_config(**overrides: object) -> MagicMock:
    """Build a mock EvaluatorConfig with sensible defaults."""
    cfg = MagicMock()
    cfg.embedding_model = overrides.get("embedding_model", "mock-embed")
    cfg.embedding_threshold = overrides.get("embedding_threshold", 0.80)
    cfg.nli_model = overrides.get("nli_model", "mock-nli")
    cfg.nli_require_bidirectional = overrides.get("nli_require_bidirectional", True)
    cfg.bertscore_model = overrides.get("bertscore_model", "mock-bert")
    cfg.bertscore_threshold = overrides.get("bertscore_threshold", 0.88)
    return cfg


class TestEvaluationPipeline:
    """Unit tests — all stages mocked at module level."""

    def setup_method(self) -> None:
        self.config = _make_config()

    @patch("lucid.evaluator.pipeline.BERTScoreChecker")
    @patch("lucid.evaluator.pipeline.NLIChecker")
    @patch("lucid.evaluator.pipeline.EmbeddingSimilarity")
    @patch("lucid.evaluator.pipeline.TermVerifier")
    def test_full_pass_through(
        self,
        mock_tv_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_nli_cls: MagicMock,
        _mock_bert_cls: MagicMock,
    ) -> None:
        """All stages pass → result.passed is True with all scores populated."""
        mock_tv_cls.return_value.verify.return_value = TermVerificationResult(
            passed=True,
            missing_placeholders=(),
            mismatched_numbers=(),
            reason=None,
        )
        mock_emb_cls.return_value.compute.return_value = EmbeddingResult(similarity=0.92)
        mock_nli_cls.return_value.check.return_value = NLIResult(
            forward_label="entailment",
            backward_label="entailment",
            forward_scores={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
            backward_scores={"entailment": 0.90, "neutral": 0.06, "contradiction": 0.04},
        )

        pipeline = EvaluationPipeline(self.config)
        result = pipeline.run("chunk_1", "original text", "paraphrase text")

        assert result.passed is True
        assert result.chunk_id == "chunk_1"
        assert result.embedding_similarity == pytest.approx(0.92)
        assert result.nli_forward == "entailment"
        assert result.nli_backward == "entailment"
        assert result.bertscore_f1 is None  # not run by default
        assert result.rejection_reason is None

    @patch("lucid.evaluator.pipeline.BERTScoreChecker")
    @patch("lucid.evaluator.pipeline.NLIChecker")
    @patch("lucid.evaluator.pipeline.EmbeddingSimilarity")
    @patch("lucid.evaluator.pipeline.TermVerifier")
    def test_early_exit_at_term_verification(
        self,
        mock_tv_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_nli_cls: MagicMock,
        _mock_bert_cls: MagicMock,
    ) -> None:
        """Term verification failure → reject before embedding stage."""
        mock_tv_cls.return_value.verify.return_value = TermVerificationResult(
            passed=False,
            missing_placeholders=("[TERM_000]",),
            mismatched_numbers=(),
            reason="missing placeholders: [TERM_000]",
        )

        pipeline = EvaluationPipeline(self.config)
        result = pipeline.run("chunk_2", "original", "paraphrase")

        assert result.passed is False
        assert "term verification" in (result.rejection_reason or "")
        assert result.embedding_similarity is None
        assert result.nli_forward is None
        mock_emb_cls.return_value.compute.assert_not_called()
        mock_nli_cls.return_value.check.assert_not_called()

    @patch("lucid.evaluator.pipeline.BERTScoreChecker")
    @patch("lucid.evaluator.pipeline.NLIChecker")
    @patch("lucid.evaluator.pipeline.EmbeddingSimilarity")
    @patch("lucid.evaluator.pipeline.TermVerifier")
    def test_early_exit_at_embedding(
        self,
        mock_tv_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_nli_cls: MagicMock,
        _mock_bert_cls: MagicMock,
    ) -> None:
        """Low embedding similarity → reject before NLI stage."""
        mock_tv_cls.return_value.verify.return_value = TermVerificationResult(
            passed=True,
            missing_placeholders=(),
            mismatched_numbers=(),
            reason=None,
        )
        mock_emb_cls.return_value.compute.return_value = EmbeddingResult(similarity=0.55)

        pipeline = EvaluationPipeline(self.config)
        result = pipeline.run("chunk_3", "original", "bad paraphrase")

        assert result.passed is False
        assert result.embedding_similarity == pytest.approx(0.55)
        assert "embedding similarity" in (result.rejection_reason or "")
        assert result.nli_forward is None
        mock_nli_cls.return_value.check.assert_not_called()

    @patch("lucid.evaluator.pipeline.BERTScoreChecker")
    @patch("lucid.evaluator.pipeline.NLIChecker")
    @patch("lucid.evaluator.pipeline.EmbeddingSimilarity")
    @patch("lucid.evaluator.pipeline.TermVerifier")
    def test_early_exit_at_nli_contradiction(
        self,
        mock_tv_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_nli_cls: MagicMock,
        _mock_bert_cls: MagicMock,
    ) -> None:
        """NLI contradiction → reject with embedding and NLI scores preserved."""
        mock_tv_cls.return_value.verify.return_value = TermVerificationResult(
            passed=True,
            missing_placeholders=(),
            mismatched_numbers=(),
            reason=None,
        )
        mock_emb_cls.return_value.compute.return_value = EmbeddingResult(similarity=0.85)
        mock_nli_cls.return_value.check.return_value = NLIResult(
            forward_label="contradiction",
            backward_label="entailment",
            forward_scores={"entailment": 0.05, "neutral": 0.10, "contradiction": 0.85},
            backward_scores={"entailment": 0.90, "neutral": 0.05, "contradiction": 0.05},
        )

        pipeline = EvaluationPipeline(self.config)
        result = pipeline.run("chunk_4", "original", "contradictory")

        assert result.passed is False
        assert result.embedding_similarity == pytest.approx(0.85)
        assert result.nli_forward == "contradiction"
        assert result.nli_backward == "entailment"
        assert "contradiction" in (result.rejection_reason or "")
        assert result.bertscore_f1 is None

    @patch("lucid.evaluator.pipeline.BERTScoreChecker")
    @patch("lucid.evaluator.pipeline.NLIChecker")
    @patch("lucid.evaluator.pipeline.EmbeddingSimilarity")
    @patch("lucid.evaluator.pipeline.TermVerifier")
    def test_nli_bidirectional_required_but_one_neutral(
        self,
        mock_tv_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_nli_cls: MagicMock,
        _mock_bert_cls: MagicMock,
    ) -> None:
        """Bidirectional entailment required but backward is neutral → reject."""
        mock_tv_cls.return_value.verify.return_value = TermVerificationResult(
            passed=True,
            missing_placeholders=(),
            mismatched_numbers=(),
            reason=None,
        )
        mock_emb_cls.return_value.compute.return_value = EmbeddingResult(similarity=0.90)
        mock_nli_cls.return_value.check.return_value = NLIResult(
            forward_label="entailment",
            backward_label="neutral",
            forward_scores={"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05},
            backward_scores={"entailment": 0.20, "neutral": 0.70, "contradiction": 0.10},
        )

        pipeline = EvaluationPipeline(self.config)
        result = pipeline.run("chunk_5", "original", "asymmetric paraphrase")

        assert result.passed is False
        assert "bidirectional" in (result.rejection_reason or "")

    @patch("lucid.evaluator.pipeline.BERTScoreChecker")
    @patch("lucid.evaluator.pipeline.NLIChecker")
    @patch("lucid.evaluator.pipeline.EmbeddingSimilarity")
    @patch("lucid.evaluator.pipeline.TermVerifier")
    def test_nli_unidirectional_mode_forward_only(
        self,
        mock_tv_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_nli_cls: MagicMock,
        _mock_bert_cls: MagicMock,
    ) -> None:
        """With bidirectional=False, forward entailment alone is sufficient."""
        cfg = _make_config(nli_require_bidirectional=False)
        mock_tv_cls.return_value.verify.return_value = TermVerificationResult(
            passed=True,
            missing_placeholders=(),
            mismatched_numbers=(),
            reason=None,
        )
        mock_emb_cls.return_value.compute.return_value = EmbeddingResult(similarity=0.90)
        mock_nli_cls.return_value.check.return_value = NLIResult(
            forward_label="entailment",
            backward_label="neutral",
            forward_scores={"entailment": 0.85, "neutral": 0.10, "contradiction": 0.05},
            backward_scores={"entailment": 0.20, "neutral": 0.70, "contradiction": 0.10},
        )

        pipeline = EvaluationPipeline(cfg)
        result = pipeline.run("chunk_6", "original", "partial paraphrase")

        assert result.passed is True

    @patch("lucid.evaluator.pipeline.BERTScoreChecker")
    @patch("lucid.evaluator.pipeline.NLIChecker")
    @patch("lucid.evaluator.pipeline.EmbeddingSimilarity")
    @patch("lucid.evaluator.pipeline.TermVerifier")
    def test_bertscore_gated_by_options(
        self,
        mock_tv_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_nli_cls: MagicMock,
        mock_bert_cls: MagicMock,
    ) -> None:
        """BERTScore only runs when PipelineOptions.run_bertscore is True."""
        mock_tv_cls.return_value.verify.return_value = TermVerificationResult(
            passed=True,
            missing_placeholders=(),
            mismatched_numbers=(),
            reason=None,
        )
        mock_emb_cls.return_value.compute.return_value = EmbeddingResult(similarity=0.92)
        mock_nli_cls.return_value.check.return_value = NLIResult(
            forward_label="entailment",
            backward_label="entailment",
            forward_scores={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
            backward_scores={"entailment": 0.90, "neutral": 0.06, "contradiction": 0.04},
        )
        mock_bert_cls.return_value.compute.return_value = BERTScoreResult(
            precision=0.91,
            recall=0.89,
            f1=0.90,
        )

        pipeline = EvaluationPipeline(self.config)

        # Default: BERTScore not run
        result_default = pipeline.run("c1", "orig", "para")
        assert result_default.bertscore_f1 is None
        mock_bert_cls.return_value.compute.assert_not_called()

        # Explicit: BERTScore enabled
        opts = PipelineOptions(run_bertscore=True)
        result_bert = pipeline.run("c2", "orig", "para", options=opts)
        assert result_bert.bertscore_f1 == pytest.approx(0.90)
        mock_bert_cls.return_value.compute.assert_called_once()

    @patch("lucid.evaluator.pipeline.BERTScoreChecker")
    @patch("lucid.evaluator.pipeline.NLIChecker")
    @patch("lucid.evaluator.pipeline.EmbeddingSimilarity")
    @patch("lucid.evaluator.pipeline.TermVerifier")
    def test_bertscore_below_threshold_rejects(
        self,
        mock_tv_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_nli_cls: MagicMock,
        mock_bert_cls: MagicMock,
    ) -> None:
        """BERTScore F1 below threshold → reject with all prior scores preserved."""
        mock_tv_cls.return_value.verify.return_value = TermVerificationResult(
            passed=True,
            missing_placeholders=(),
            mismatched_numbers=(),
            reason=None,
        )
        mock_emb_cls.return_value.compute.return_value = EmbeddingResult(similarity=0.92)
        mock_nli_cls.return_value.check.return_value = NLIResult(
            forward_label="entailment",
            backward_label="entailment",
            forward_scores={"entailment": 0.95, "neutral": 0.03, "contradiction": 0.02},
            backward_scores={"entailment": 0.90, "neutral": 0.06, "contradiction": 0.04},
        )
        mock_bert_cls.return_value.compute.return_value = BERTScoreResult(
            precision=0.80,
            recall=0.75,
            f1=0.77,
        )

        pipeline = EvaluationPipeline(self.config)
        opts = PipelineOptions(run_bertscore=True)
        result = pipeline.run("chunk_7", "orig", "para", options=opts)

        assert result.passed is False
        assert result.embedding_similarity == pytest.approx(0.92)
        assert result.nli_forward == "entailment"
        assert result.bertscore_f1 == pytest.approx(0.77)
        assert "BERTScore" in (result.rejection_reason or "")

    @patch("lucid.evaluator.pipeline.BERTScoreChecker")
    @patch("lucid.evaluator.pipeline.NLIChecker")
    @patch("lucid.evaluator.pipeline.EmbeddingSimilarity")
    @patch("lucid.evaluator.pipeline.TermVerifier")
    def test_term_verification_runs_first(
        self,
        mock_tv_cls: MagicMock,
        mock_emb_cls: MagicMock,
        mock_nli_cls: MagicMock,
        mock_bert_cls: MagicMock,
    ) -> None:
        """Term verification failure prevents all model-based stages from running."""
        mock_tv_cls.return_value.verify.return_value = TermVerificationResult(
            passed=False,
            missing_placeholders=("[MATH_001]",),
            mismatched_numbers=("42",),
            reason="missing placeholders: [MATH_001]; missing numbers: 42",
        )

        pipeline = EvaluationPipeline(self.config)
        opts = PipelineOptions(run_bertscore=True)
        result = pipeline.run("chunk_8", "original", "corrupted", opts)

        assert result.passed is False
        mock_emb_cls.return_value.compute.assert_not_called()
        mock_nli_cls.return_value.check.assert_not_called()
        mock_bert_cls.return_value.compute.assert_not_called()
