"""Interface contracts for LUCID's plugin architecture.

All pipeline components conform to these protocols, enabling
swappable implementations and runtime validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lucid.config import TransformConfig
    from lucid.core.types import MetricResult
    from lucid.models.results import DetectionResult, EvaluationResult, TransformResult
    from lucid.parser.chunk import Chunk, ProseChunk


@runtime_checkable
class DocumentAdapter(Protocol):
    """Parse documents into chunks and reconstruct from modified chunks."""

    def parse(self, content: str) -> list[Chunk]: ...

    def reconstruct(self, original: str, chunks: list[Chunk]) -> str: ...


@runtime_checkable
class Detector(Protocol):
    """Score a prose chunk for AI-generated content probability."""

    def detect(self, chunk: ProseChunk) -> DetectionResult: ...


@runtime_checkable
class Transformer(Protocol):
    """Transform a prose chunk to reduce AI detection score."""

    def transform(self, chunk: ProseChunk, detection: DetectionResult) -> TransformResult: ...


@runtime_checkable
class Evaluator(Protocol):
    """Evaluate semantic preservation between original and paraphrased text."""

    def evaluate(self, original: str, paraphrase: str) -> EvaluationResult: ...


@runtime_checkable
class Metric(Protocol):
    """Compute a metric between original and transformed text."""

    name: str

    def compute(self, original: str, transformed: str) -> MetricResult: ...


@runtime_checkable
class Operator(Protocol):
    """Apply a transformation operator to text."""

    name: str

    def apply(self, text: str, config: TransformConfig) -> TransformResult: ...
