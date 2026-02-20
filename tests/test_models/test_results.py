"""Tests for result data models."""

from __future__ import annotations

import pytest

from lucid.models.results import (
    DetectionResult,
    DocumentResult,
    EvaluationResult,
    ParaphraseResult,
)
from lucid.parser.chunk import ProseChunk, StructuralChunk


class TestDetectionResult:
    """DetectionResult validation and serialization."""

    def test_valid_result(self) -> None:
        """Valid DetectionResult creates successfully."""
        result = DetectionResult(
            chunk_id="abc",
            ensemble_score=0.75,
            classification="ai_generated",
            roberta_score=0.8,
        )
        assert result.ensemble_score == 0.75
        assert result.classification == "ai_generated"

    def test_score_below_zero_raises(self) -> None:
        """Ensemble score < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match="ensemble_score"):
            DetectionResult(chunk_id="a", ensemble_score=-0.1, classification="human")

    def test_score_above_one_raises(self) -> None:
        """Ensemble score > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match="ensemble_score"):
            DetectionResult(chunk_id="a", ensemble_score=1.1, classification="human")

    def test_invalid_classification_raises(self) -> None:
        """Invalid classification string raises ValueError."""
        with pytest.raises(ValueError, match="classification"):
            DetectionResult(chunk_id="a", ensemble_score=0.5, classification="unknown")

    def test_optional_tier_score_validation(self) -> None:
        """Optional per-tier scores are validated when present."""
        with pytest.raises(ValueError, match="roberta_score"):
            DetectionResult(
                chunk_id="a",
                ensemble_score=0.5,
                classification="ambiguous",
                roberta_score=1.5,
            )

    def test_round_trip(self) -> None:
        """to_dict / from_dict preserves all fields."""
        original = DetectionResult(
            chunk_id="test_id",
            ensemble_score=0.42,
            classification="ambiguous",
            roberta_score=0.5,
            statistical_score=0.3,
            binoculars_score=None,
            feature_details={"burstiness": 0.15},
        )
        restored = DetectionResult.from_dict(original.to_dict())
        assert restored.chunk_id == original.chunk_id
        assert restored.ensemble_score == original.ensemble_score
        assert restored.classification == original.classification
        assert restored.roberta_score == original.roberta_score
        assert restored.binoculars_score is None
        assert restored.feature_details == original.feature_details


class TestParaphraseResult:
    """ParaphraseResult validation and serialization."""

    def test_valid_result(self) -> None:
        """Valid ParaphraseResult creates successfully."""
        result = ParaphraseResult(
            chunk_id="abc",
            original_text="AI text",
            humanized_text="Human text",
            iteration_count=2,
            strategy_used="standard",
            final_detection_score=0.2,
        )
        assert result.iteration_count == 2

    def test_negative_iteration_count_raises(self) -> None:
        """Negative iteration_count raises ValueError."""
        with pytest.raises(ValueError, match="iteration_count"):
            ParaphraseResult(
                chunk_id="a",
                original_text="x",
                humanized_text="y",
                iteration_count=-1,
                strategy_used="s",
                final_detection_score=0.5,
            )

    def test_invalid_detection_score_raises(self) -> None:
        """Detection score out of range raises ValueError."""
        with pytest.raises(ValueError, match="final_detection_score"):
            ParaphraseResult(
                chunk_id="a",
                original_text="x",
                humanized_text="y",
                iteration_count=1,
                strategy_used="s",
                final_detection_score=2.0,
            )

    def test_round_trip(self) -> None:
        """to_dict / from_dict preserves all fields."""
        original = ParaphraseResult(
            chunk_id="id1",
            original_text="original",
            humanized_text="humanized",
            iteration_count=3,
            strategy_used="voice_shift",
            final_detection_score=0.15,
        )
        restored = ParaphraseResult.from_dict(original.to_dict())
        assert restored.chunk_id == original.chunk_id
        assert restored.humanized_text == original.humanized_text
        assert restored.strategy_used == original.strategy_used


class TestEvaluationResult:
    """EvaluationResult validation and serialization."""

    def test_passed_result(self) -> None:
        """Passed result does not require rejection_reason."""
        result = EvaluationResult(
            chunk_id="abc",
            passed=True,
            embedding_similarity=0.92,
        )
        assert result.passed is True
        assert result.rejection_reason is None

    def test_failed_without_reason_raises(self) -> None:
        """Failed result without rejection_reason raises ValueError."""
        with pytest.raises(ValueError, match="rejection_reason is required"):
            EvaluationResult(chunk_id="a", passed=False)

    def test_failed_with_reason(self) -> None:
        """Failed result with rejection_reason is valid."""
        result = EvaluationResult(
            chunk_id="a",
            passed=False,
            rejection_reason="Severe meaning drift",
            embedding_similarity=0.65,
        )
        assert result.rejection_reason == "Severe meaning drift"

    def test_bertscore_range_validation(self) -> None:
        """BERTScore F1 allows [-1.0, 1.0] range."""
        # Valid negative (baseline-rescaled)
        result = EvaluationResult(chunk_id="a", passed=True, bertscore_f1=-0.5)
        assert result.bertscore_f1 == -0.5

        # Invalid below -1.0
        with pytest.raises(ValueError, match="bertscore_f1"):
            EvaluationResult(chunk_id="a", passed=True, bertscore_f1=-1.5)

    def test_round_trip(self) -> None:
        """to_dict / from_dict preserves all fields."""
        original = EvaluationResult(
            chunk_id="eval1",
            passed=True,
            embedding_similarity=0.88,
            nli_forward="entailment",
            nli_backward="entailment",
            bertscore_f1=0.91,
        )
        restored = EvaluationResult.from_dict(original.to_dict())
        assert restored.passed == original.passed
        assert restored.embedding_similarity == original.embedding_similarity
        assert restored.nli_forward == original.nli_forward


class TestDocumentResult:
    """DocumentResult validation and serialization."""

    def test_valid_formats(self) -> None:
        """All three formats are accepted."""
        for fmt in ("latex", "markdown", "plaintext"):
            result = DocumentResult(input_path="test.tex", format=fmt)
            assert result.format == fmt

    def test_invalid_format_raises(self) -> None:
        """Invalid format raises ValueError."""
        with pytest.raises(ValueError, match="format must be one of"):
            DocumentResult(input_path="test.doc", format="word")

    def test_full_round_trip(self) -> None:
        """Full DocumentResult with chunks and results survives round trip."""
        prose = ProseChunk(
            text="AI-generated paragraph",
            start_pos=0,
            end_pos=22,
            domain_hint="stem",
        )
        structural = StructuralChunk(
            text="\\begin{equation}x^2\\end{equation}",
            start_pos=23,
            end_pos=56,
        )
        detection = DetectionResult(
            chunk_id=prose.id,
            ensemble_score=0.85,
            classification="ai_generated",
        )
        paraphrase = ParaphraseResult(
            chunk_id=prose.id,
            original_text=prose.text,
            humanized_text="A naturally written paragraph",
            iteration_count=2,
            strategy_used="standard",
            final_detection_score=0.18,
        )
        evaluation = EvaluationResult(
            chunk_id=prose.id,
            passed=True,
            embedding_similarity=0.89,
        )

        original = DocumentResult(
            input_path="paper.tex",
            format="latex",
            chunks=[prose, structural],
            detections=[detection],
            paraphrases=[paraphrase],
            evaluations=[evaluation],
            compilation_valid=True,
            output_path="paper_humanized.tex",
            summary_stats={"total_chunks": 2, "ai_chunks": 1},
        )

        data = original.to_dict()
        restored = DocumentResult.from_dict(data)

        assert restored.input_path == original.input_path
        assert restored.format == original.format
        assert len(restored.chunks) == 2
        assert isinstance(restored.chunks[0], ProseChunk)
        assert isinstance(restored.chunks[1], StructuralChunk)
        assert len(restored.detections) == 1
        assert restored.detections[0].ensemble_score == 0.85
        assert len(restored.paraphrases) == 1
        assert restored.paraphrases[0].humanized_text == "A naturally written paragraph"
        assert len(restored.evaluations) == 1
        assert restored.evaluations[0].passed is True
        assert restored.compilation_valid is True
        assert restored.summary_stats == {"total_chunks": 2, "ai_chunks": 1}
