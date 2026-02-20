"""Tests for chunk data models."""

from __future__ import annotations

import pytest

from lucid.parser.chunk import Chunk, ChunkType, ProseChunk, StructuralChunk


class TestChunk:
    """Base Chunk construction and validation."""

    def test_valid_chunk(self) -> None:
        """Chunk with valid positions creates successfully."""
        chunk = Chunk(text="hello", chunk_type=ChunkType.PROSE, start_pos=0, end_pos=5)
        assert chunk.text == "hello"
        assert chunk.chunk_type == ChunkType.PROSE
        assert chunk.start_pos == 0
        assert chunk.end_pos == 5

    def test_default_id_generation(self) -> None:
        """Each chunk gets a unique auto-generated ID."""
        c1 = Chunk(text="a", chunk_type=ChunkType.PROSE, start_pos=0, end_pos=1)
        c2 = Chunk(text="b", chunk_type=ChunkType.PROSE, start_pos=0, end_pos=1)
        assert c1.id != c2.id
        assert len(c1.id) == 32  # uuid4 hex

    def test_negative_start_pos_raises(self) -> None:
        """start_pos < 0 raises ValueError."""
        with pytest.raises(ValueError, match="start_pos must be >= 0"):
            Chunk(text="x", chunk_type=ChunkType.PROSE, start_pos=-1, end_pos=5)

    def test_end_before_start_raises(self) -> None:
        """end_pos < start_pos raises ValueError."""
        with pytest.raises(ValueError, match=r"end_pos.*must be >= start_pos"):
            Chunk(text="x", chunk_type=ChunkType.PROSE, start_pos=10, end_pos=5)

    def test_zero_length_chunk_valid(self) -> None:
        """start_pos == end_pos is valid (empty span)."""
        chunk = Chunk(text="", chunk_type=ChunkType.STRUCTURAL, start_pos=5, end_pos=5)
        assert chunk.start_pos == chunk.end_pos

    def test_default_metadata(self) -> None:
        """metadata defaults to empty dict."""
        chunk = Chunk(text="x", chunk_type=ChunkType.PROSE, start_pos=0, end_pos=1)
        assert chunk.metadata == {}


class TestProseChunk:
    """ProseChunk construction and type enforcement."""

    def test_type_is_always_prose(self) -> None:
        """ProseChunk.chunk_type is forced to PROSE regardless of input."""
        chunk = ProseChunk(text="hello world", start_pos=0, end_pos=11)
        assert chunk.chunk_type == ChunkType.PROSE

    def test_protected_text_defaults_to_text(self) -> None:
        """protected_text defaults to the original text."""
        chunk = ProseChunk(text="some text", start_pos=0, end_pos=9)
        assert chunk.protected_text == "some text"

    def test_protected_text_explicit(self) -> None:
        """Explicit protected_text is preserved."""
        chunk = ProseChunk(
            text="The loss $L$ converges",
            start_pos=0,
            end_pos=22,
            protected_text="The loss <MATH_001> converges",
        )
        assert chunk.protected_text == "The loss <MATH_001> converges"

    def test_math_placeholders(self) -> None:
        """Math placeholder dict is stored correctly."""
        placeholders = {"<MATH_001>": "$L$"}
        chunk = ProseChunk(
            text="loss $L$",
            start_pos=0,
            end_pos=8,
            math_placeholders=placeholders,
        )
        assert chunk.math_placeholders == placeholders

    def test_domain_hint_default(self) -> None:
        """domain_hint defaults to empty string."""
        chunk = ProseChunk(text="x", start_pos=0, end_pos=1)
        assert chunk.domain_hint == ""


class TestStructuralChunk:
    """StructuralChunk construction and type enforcement."""

    def test_type_is_always_structural(self) -> None:
        """StructuralChunk.chunk_type is forced to STRUCTURAL."""
        chunk = StructuralChunk(text="\\begin{equation}", start_pos=0, end_pos=16)
        assert chunk.chunk_type == ChunkType.STRUCTURAL

    def test_raw_content_defaults_to_text(self) -> None:
        """raw_content defaults to the text value."""
        chunk = StructuralChunk(text="$$x^2$$", start_pos=0, end_pos=7)
        assert chunk.raw_content == "$$x^2$$"

    def test_raw_content_explicit(self) -> None:
        """Explicit raw_content is preserved."""
        chunk = StructuralChunk(text="rendered", start_pos=0, end_pos=8, raw_content="\\raw{bytes}")
        assert chunk.raw_content == "\\raw{bytes}"


class TestChunkSerialization:
    """Round-trip to_dict / from_dict for all chunk types."""

    def test_base_chunk_round_trip(self) -> None:
        """Base Chunk survives serialization round trip."""
        original = Chunk(
            text="hello",
            chunk_type=ChunkType.PROSE,
            start_pos=0,
            end_pos=5,
            metadata={"key": "value"},
        )
        data = original.to_dict()
        restored = Chunk.from_dict(data)

        assert restored.id == original.id
        assert restored.text == original.text
        assert restored.chunk_type == original.chunk_type
        assert restored.start_pos == original.start_pos
        assert restored.end_pos == original.end_pos
        assert restored.metadata == original.metadata

    def test_prose_chunk_round_trip(self) -> None:
        """ProseChunk preserves all fields through serialization."""
        original = ProseChunk(
            text="The loss function converges",
            start_pos=100,
            end_pos=126,
            math_placeholders={"<M1>": "$L$"},
            term_placeholders={"<T1>": "BERT"},
            protected_text="The loss function converges",
            domain_hint="stem",
        )
        data = original.to_dict()
        restored = Chunk.from_dict(data)

        assert isinstance(restored, ProseChunk)
        assert restored.id == original.id
        assert restored.math_placeholders == original.math_placeholders
        assert restored.term_placeholders == original.term_placeholders
        assert restored.protected_text == original.protected_text
        assert restored.domain_hint == original.domain_hint

    def test_structural_chunk_round_trip(self) -> None:
        """StructuralChunk preserves raw_content through serialization."""
        original = StructuralChunk(
            text="\\begin{equation}x^2\\end{equation}",
            start_pos=50,
            end_pos=83,
            raw_content="\\begin{equation}x^2\\end{equation}",
        )
        data = original.to_dict()
        restored = Chunk.from_dict(data)

        assert isinstance(restored, StructuralChunk)
        assert restored.raw_content == original.raw_content

    def test_from_dict_unknown_class_returns_base(self) -> None:
        """Unknown _class falls back to base Chunk."""
        data = {
            "_class": "UnknownChunk",
            "id": "abc123",
            "text": "test",
            "chunk_type": "prose",
            "start_pos": 0,
            "end_pos": 4,
            "metadata": {},
        }
        chunk = Chunk.from_dict(data)
        assert type(chunk) is Chunk
