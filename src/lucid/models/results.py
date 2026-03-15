"""Result data models for detection, transform, evaluation, and document processing."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from lucid.parser.chunk import Chunk

_VALID_CLASSIFICATIONS = frozenset({"human", "ambiguous", "ai_generated"})
_VALID_FORMATS = frozenset({"latex", "markdown", "plaintext"})


def _validate_score(value: float, name: str, low: float = 0.0, high: float = 1.0) -> None:
    """Validate that a score falls within the expected range."""
    if not (low <= value <= high):
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")


@dataclass
class DetectionResult:
    """Per-chunk AI detection scoring.

    Args:
        chunk_id: ID of the chunk that was scored.
        ensemble_score: Final combined detection score in [0.0, 1.0].
        classification: One of "human", "ambiguous", "ai_generated".
        roberta_score: Optional Tier 1 classifier score.
        statistical_score: Optional Tier 2 statistical features score.
        binoculars_score: Optional Tier 3 cross-perplexity score.
        feature_details: Optional breakdown of statistical features.
    """

    chunk_id: str
    ensemble_score: float
    classification: str
    roberta_score: float | None = None
    statistical_score: float | None = None
    binoculars_score: float | None = None
    feature_details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_score(self.ensemble_score, "ensemble_score")
        if self.classification not in _VALID_CLASSIFICATIONS:
            raise ValueError(
                f"classification must be one of {_VALID_CLASSIFICATIONS}, "
                f"got {self.classification!r}"
            )
        if self.roberta_score is not None:
            _validate_score(self.roberta_score, "roberta_score")
        if self.statistical_score is not None:
            _validate_score(self.statistical_score, "statistical_score")
        if self.binoculars_score is not None:
            _validate_score(self.binoculars_score, "binoculars_score")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "ensemble_score": self.ensemble_score,
            "classification": self.classification,
            "roberta_score": self.roberta_score,
            "statistical_score": self.statistical_score,
            "binoculars_score": self.binoculars_score,
            "feature_details": self.feature_details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectionResult:
        """Deserialize from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            ensemble_score=data["ensemble_score"],
            classification=data["classification"],
            roberta_score=data.get("roberta_score"),
            statistical_score=data.get("statistical_score"),
            binoculars_score=data.get("binoculars_score"),
            feature_details=data.get("feature_details", {}),
        )


@dataclass
class TransformResult:
    """Result of transforming a single chunk.

    Args:
        chunk_id: ID of the chunk that was transformed.
        original_text: The original prose text before transformation.
        transformed_text: The transformed output text.
        iteration_count: Number of search iterations performed.
        operator_used: Name of the final operator that produced the output.
        final_detection_score: Detection score of the transformed text in [0.0, 1.0].
    """

    chunk_id: str
    original_text: str
    transformed_text: str
    iteration_count: int
    operator_used: str
    final_detection_score: float
    semantic_similarity: float | None = None
    fallback_mode: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_score(self.final_detection_score, "final_detection_score")
        if self.iteration_count < 0:
            raise ValueError(f"iteration_count must be >= 0, got {self.iteration_count}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        d: dict[str, Any] = {
            "chunk_id": self.chunk_id,
            "original_text": self.original_text,
            "transformed_text": self.transformed_text,
            "iteration_count": self.iteration_count,
            "operator_used": self.operator_used,
            "final_detection_score": self.final_detection_score,
        }
        if self.semantic_similarity is not None:
            d["semantic_similarity"] = self.semantic_similarity
        if self.fallback_mode is not None:
            d["fallback_mode"] = self.fallback_mode
        if self.diagnostics:
            d["diagnostics"] = self.diagnostics
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformResult:
        """Deserialize from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            original_text=data["original_text"],
            transformed_text=data["transformed_text"],
            iteration_count=data["iteration_count"],
            operator_used=data["operator_used"],
            final_detection_score=data["final_detection_score"],
            semantic_similarity=data.get("semantic_similarity"),
            fallback_mode=data.get("fallback_mode"),
            diagnostics=data.get("diagnostics", {}),
        )


@dataclass
class EvaluationResult:
    """Semantic evaluation result for a paraphrase.

    Args:
        chunk_id: ID of the evaluated chunk.
        passed: Whether the paraphrase passed all evaluation stages.
        embedding_similarity: Stage 1 cosine similarity score.
        nli_forward: Stage 2 forward entailment label.
        nli_backward: Stage 2 backward entailment label.
        bertscore_f1: Stage 3 BERTScore F1 (baseline-rescaled, range [-1.0, 1.0]).
        rejection_reason: Required explanation when passed=False.
    """

    chunk_id: str
    passed: bool
    embedding_similarity: float | None = None
    nli_forward: str | None = None
    nli_backward: str | None = None
    bertscore_f1: float | None = None
    rejection_reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.passed and not self.rejection_reason:
            raise ValueError("rejection_reason is required when passed=False")
        if self.embedding_similarity is not None:
            _validate_score(self.embedding_similarity, "embedding_similarity", low=-1.0)
        if self.bertscore_f1 is not None:
            _validate_score(self.bertscore_f1, "bertscore_f1", low=-1.0)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "passed": self.passed,
            "embedding_similarity": self.embedding_similarity,
            "nli_forward": self.nli_forward,
            "nli_backward": self.nli_backward,
            "bertscore_f1": self.bertscore_f1,
            "rejection_reason": self.rejection_reason,
            "diagnostics": self.diagnostics,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvaluationResult:
        """Deserialize from dictionary."""
        return cls(
            chunk_id=data["chunk_id"],
            passed=data["passed"],
            embedding_similarity=data.get("embedding_similarity"),
            nli_forward=data.get("nli_forward"),
            nli_backward=data.get("nli_backward"),
            bertscore_f1=data.get("bertscore_f1"),
            rejection_reason=data.get("rejection_reason"),
            diagnostics=data.get("diagnostics", {}),
        )


def _chunk_from_dict(data: dict[str, Any]) -> Chunk:
    """Polymorphic chunk deserialization dispatcher."""
    return Chunk.from_dict(data)


@dataclass
class DocumentResult:
    """Aggregate result for an entire document.

    Args:
        input_path: Path to the original input document.
        format: Document format ("latex", "markdown", "plaintext").
        chunks: All parsed chunks in document order.
        detections: Detection results for prose chunks.
        transforms: Transform results for processed chunks.
        evaluations: Evaluation results for transformed chunks.
        compilation_valid: Whether the reconstructed document compiles.
        output_path: Path to the output document (set after reconstruction).
        summary_stats: Aggregate statistics for reporting.
    """

    input_path: str
    format: str
    chunks: list[Chunk] = field(default_factory=list)
    detections: list[DetectionResult] = field(default_factory=list)
    transforms: list[TransformResult] = field(default_factory=list)
    evaluations: list[EvaluationResult] = field(default_factory=list)
    compilation_valid: bool | None = None
    output_path: str | None = None
    summary_stats: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.format not in _VALID_FORMATS:
            raise ValueError(f"format must be one of {_VALID_FORMATS}, got {self.format!r}")

    def to_dict(self) -> dict[str, Any]:
        """Serialize entire document result."""
        return {
            "input_path": self.input_path,
            "format": self.format,
            "chunks": [c.to_dict() for c in self.chunks],
            "detections": [d.to_dict() for d in self.detections],
            "transforms": [p.to_dict() for p in self.transforms],
            "evaluations": [e.to_dict() for e in self.evaluations],
            "compilation_valid": self.compilation_valid,
            "output_path": self.output_path,
            "summary_stats": self.summary_stats,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentResult:
        """Deserialize from dictionary."""
        return cls(
            input_path=data["input_path"],
            format=data["format"],
            chunks=[_chunk_from_dict(c) for c in data.get("chunks", [])],
            detections=[DetectionResult.from_dict(d) for d in data.get("detections", [])],
            transforms=[TransformResult.from_dict(p) for p in data.get("transforms", [])],
            evaluations=[EvaluationResult.from_dict(e) for e in data.get("evaluations", [])],
            compilation_valid=data.get("compilation_valid"),
            output_path=data.get("output_path"),
            summary_stats=data.get("summary_stats", {}),
        )
