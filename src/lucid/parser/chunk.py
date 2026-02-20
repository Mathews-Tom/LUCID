"""Chunk data models for document parsing.

Chunks are the atomic units of document processing. Each chunk carries
position information for lossless document reconstruction.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ChunkType(Enum):
    """Classification of document content regions."""

    PROSE = "prose"
    STRUCTURAL = "structural"


@dataclass
class Chunk:
    """Base chunk with position tracking.

    Args:
        text: The textual content of the chunk.
        chunk_type: Whether this chunk is prose or structural.
        start_pos: Start byte offset in the original document.
        end_pos: End byte offset in the original document.
        metadata: Arbitrary metadata attached during parsing.
        id: Unique identifier (auto-generated uuid4 hex).
    """

    text: str
    chunk_type: ChunkType
    start_pos: int
    end_pos: int
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def __post_init__(self) -> None:
        if self.start_pos < 0:
            raise ValueError(f"start_pos must be >= 0, got {self.start_pos}")
        if self.end_pos < self.start_pos:
            raise ValueError(f"end_pos ({self.end_pos}) must be >= start_pos ({self.start_pos})")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON checkpointing."""
        return {
            "id": self.id,
            "text": self.text,
            "chunk_type": self.chunk_type.value,
            "start_pos": self.start_pos,
            "end_pos": self.end_pos,
            "metadata": self.metadata,
            "_class": type(self).__name__,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """Deserialize from dictionary with polymorphic dispatch."""
        class_name = data.get("_class", "Chunk")
        chunk_type = ChunkType(data["chunk_type"])

        base_kwargs: dict[str, Any] = {
            "id": data["id"],
            "text": data["text"],
            "start_pos": data["start_pos"],
            "end_pos": data["end_pos"],
            "metadata": data.get("metadata", {}),
        }

        if class_name == "ProseChunk":
            return ProseChunk(
                **base_kwargs,
                math_placeholders=data.get("math_placeholders", {}),
                term_placeholders=data.get("term_placeholders", {}),
                protected_text=data.get("protected_text", data["text"]),
                domain_hint=data.get("domain_hint", ""),
            )
        if class_name == "StructuralChunk":
            return StructuralChunk(
                **base_kwargs,
                raw_content=data.get("raw_content", data["text"]),
            )

        return Chunk(chunk_type=chunk_type, **base_kwargs)


@dataclass
class ProseChunk(Chunk):
    """Prose content that can be detected and humanized.

    The chunk_type is always PROSE — enforced in __post_init__.

    Args:
        math_placeholders: Mapping of placeholder tokens to original math expressions.
        term_placeholders: Mapping of placeholder tokens to protected domain terms.
        protected_text: Text with placeholders substituted (defaults to original text).
        domain_hint: Detected content domain (e.g., "stem", "humanities").
    """

    chunk_type: ChunkType = field(init=False, default=ChunkType.PROSE)
    math_placeholders: dict[str, str] = field(default_factory=dict)
    term_placeholders: dict[str, str] = field(default_factory=dict)
    protected_text: str = ""
    domain_hint: str = ""

    def __post_init__(self) -> None:
        self.chunk_type = ChunkType.PROSE
        if not self.protected_text:
            self.protected_text = self.text
        super().__post_init__()

    def to_dict(self) -> dict[str, Any]:
        """Serialize including prose-specific fields."""
        data = super().to_dict()
        data.update(
            {
                "math_placeholders": self.math_placeholders,
                "term_placeholders": self.term_placeholders,
                "protected_text": self.protected_text,
                "domain_hint": self.domain_hint,
            }
        )
        return data


@dataclass
class StructuralChunk(Chunk):
    """Structural content preserved verbatim (math, code, citations, preamble).

    The chunk_type is always STRUCTURAL — enforced in __post_init__.

    Args:
        raw_content: The original raw content bytes for exact reconstruction.
    """

    chunk_type: ChunkType = field(init=False, default=ChunkType.STRUCTURAL)
    raw_content: str = ""

    def __post_init__(self) -> None:
        self.chunk_type = ChunkType.STRUCTURAL
        if not self.raw_content:
            self.raw_content = self.text
        super().__post_init__()

    def to_dict(self) -> dict[str, Any]:
        """Serialize including structural-specific fields."""
        data = super().to_dict()
        data["raw_content"] = self.raw_content
        return data
