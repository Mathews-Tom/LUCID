"""Interface contracts for LUCID's plugin architecture.

All pipeline components conform to these protocols, enabling
swappable implementations and runtime validation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from lucid.models.results import DetectionResult, EvaluationResult, ParaphraseResult
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
class Humanizer(Protocol):
    """Paraphrase a prose chunk to reduce AI detection score."""

    def humanize(self, chunk: ProseChunk, detection: DetectionResult) -> ParaphraseResult: ...


@runtime_checkable
class Evaluator(Protocol):
    """Evaluate semantic preservation between original and paraphrased text."""

    def evaluate(self, original: str, paraphrase: str) -> EvaluationResult: ...
