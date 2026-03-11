"""Benchmark record dataclasses for LUCID's evaluation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_VALID_SPLITS = frozenset({"train", "val", "test"})
_VALID_SOURCE_CLASSES = frozenset({"human", "ai_raw", "ai_edited_light", "ai_edited_heavy"})
_VALID_DOCUMENT_FORMATS = frozenset({"latex", "markdown", "plaintext"})
_VALID_PREDICTED_LABELS = frozenset({"human", "ambiguous", "ai_generated"})


def _validate_range(value: float, name: str, low: float = 0.0, high: float = 1.0) -> None:
    """Validate that a value falls within [low, high]."""
    if not (low <= value <= high):
        raise ValueError(f"{name} must be in [{low}, {high}], got {value}")


def _validate_enum(value: str, name: str, valid: frozenset[str]) -> None:
    """Validate that a value is in the allowed set."""
    if value not in valid:
        raise ValueError(f"{name} must be one of {valid}, got {value!r}")


@dataclass(frozen=True, slots=True)
class SampleRecord:
    """A single text sample in a benchmark dataset."""

    sample_id: str
    split: str
    domain: str
    source_class: str
    source_model: str | None
    document_format: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_enum(self.split, "split", _VALID_SPLITS)
        _validate_enum(self.source_class, "source_class", _VALID_SOURCE_CLASSES)
        _validate_enum(self.document_format, "document_format", _VALID_DOCUMENT_FORMATS)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sample_id": self.sample_id,
            "split": self.split,
            "domain": self.domain,
            "source_class": self.source_class,
            "source_model": self.source_model,
            "document_format": self.document_format,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SampleRecord:
        """Deserialize from dictionary."""
        return cls(
            sample_id=data["sample_id"],
            split=data["split"],
            domain=data["domain"],
            source_class=data["source_class"],
            source_model=data.get("source_model"),
            document_format=data["document_format"],
            text=data["text"],
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class TransformationRecord:
    """Record of a transformation applied to a sample."""

    transform_id: str
    parent_sample_id: str
    operator: str
    intensity: float
    config_hash: str
    output_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_range(self.intensity, "intensity")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "transform_id": self.transform_id,
            "parent_sample_id": self.parent_sample_id,
            "operator": self.operator,
            "intensity": self.intensity,
            "config_hash": self.config_hash,
            "output_text": self.output_text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TransformationRecord:
        """Deserialize from dictionary."""
        return cls(
            transform_id=data["transform_id"],
            parent_sample_id=data["parent_sample_id"],
            operator=data["operator"],
            intensity=data["intensity"],
            config_hash=data["config_hash"],
            output_text=data["output_text"],
            metadata=data.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class DetectionRecord:
    """Record of a detection run on a sample or transformation."""

    record_id: str
    sample_id: str
    transform_id: str | None
    detector_name: str
    score: float
    confidence: float | None
    threshold: float | None
    predicted_label: str
    features: dict[str, float] | None = None
    evidence: dict[str, str | float] | None = None

    def __post_init__(self) -> None:
        _validate_range(self.score, "score")
        _validate_enum(self.predicted_label, "predicted_label", _VALID_PREDICTED_LABELS)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "sample_id": self.sample_id,
            "transform_id": self.transform_id,
            "detector_name": self.detector_name,
            "score": self.score,
            "confidence": self.confidence,
            "threshold": self.threshold,
            "predicted_label": self.predicted_label,
            "features": self.features,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectionRecord:
        """Deserialize from dictionary."""
        return cls(
            record_id=data["record_id"],
            sample_id=data["sample_id"],
            transform_id=data.get("transform_id"),
            detector_name=data["detector_name"],
            score=data["score"],
            confidence=data.get("confidence"),
            threshold=data.get("threshold"),
            predicted_label=data["predicted_label"],
            features=data.get("features"),
            evidence=data.get("evidence"),
        )


@dataclass(frozen=True, slots=True)
class MetricRecord:
    """Record of a metric computed on a sample or transformation."""

    record_id: str
    sample_id: str
    transform_id: str | None
    metric_name: str
    value: float
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "record_id": self.record_id,
            "sample_id": self.sample_id,
            "transform_id": self.transform_id,
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricRecord:
        """Deserialize from dictionary."""
        return cls(
            record_id=data["record_id"],
            sample_id=data["sample_id"],
            transform_id=data.get("transform_id"),
            metric_name=data["metric_name"],
            value=data["value"],
            metadata=data.get("metadata"),
        )


@dataclass(frozen=True, slots=True)
class MetricResult:
    """Result of computing a single metric."""

    metric_name: str
    value: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "metric_name": self.metric_name,
            "value": self.value,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MetricResult:
        """Deserialize from dictionary."""
        return cls(
            metric_name=data["metric_name"],
            value=data["value"],
            metadata=data.get("metadata", {}),
        )
