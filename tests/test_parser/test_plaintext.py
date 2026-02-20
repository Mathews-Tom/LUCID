"""Tests for the plain text document parser."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lucid.parser.chunk import ChunkType, ProseChunk
from lucid.parser.plaintext import PlainTextDocumentAdapter


@pytest.fixture
def adapter() -> PlainTextDocumentAdapter:
    """Create a default plain text adapter."""
    return PlainTextDocumentAdapter()


class TestParagraphSegmentation:
    """Tests for paragraph splitting."""

    def test_multiple_paragraphs(
        self, adapter: PlainTextDocumentAdapter, corpus_plaintext_dir: Path
    ) -> None:
        """Multiple paragraphs separated by blank lines produce separate chunks."""
        content = (corpus_plaintext_dir / "paragraphs.txt").read_text()
        chunks = adapter.parse(content)
        assert len(chunks) == 4

    def test_single_paragraph(
        self, adapter: PlainTextDocumentAdapter, corpus_plaintext_dir: Path
    ) -> None:
        """Single paragraph without blank lines produces one chunk."""
        content = (corpus_plaintext_dir / "short.txt").read_text()
        chunks = adapter.parse(content)
        assert len(chunks) == 1

    def test_all_chunks_are_prose(
        self, adapter: PlainTextDocumentAdapter, corpus_plaintext_dir: Path
    ) -> None:
        """All chunks in plain text are prose."""
        content = (corpus_plaintext_dir / "mixed.txt").read_text()
        chunks = adapter.parse(content)
        assert all(c.chunk_type == ChunkType.PROSE for c in chunks)
        assert all(isinstance(c, ProseChunk) for c in chunks)

    def test_empty_content(self, adapter: PlainTextDocumentAdapter) -> None:
        """Empty content produces no chunks."""
        chunks = adapter.parse("")
        assert chunks == []

    def test_whitespace_only(self, adapter: PlainTextDocumentAdapter) -> None:
        """Whitespace-only content produces no chunks."""
        chunks = adapter.parse("   \n\n   \n")
        assert chunks == []


class TestPositionAccuracy:
    """Tests for position tracking correctness."""

    def test_positions_match_source(
        self, adapter: PlainTextDocumentAdapter, corpus_plaintext_dir: Path
    ) -> None:
        """chunk.text matches content[start_pos:end_pos]."""
        content = (corpus_plaintext_dir / "paragraphs.txt").read_text()
        chunks = adapter.parse(content)
        for chunk in chunks:
            assert chunk.text == content[chunk.start_pos : chunk.end_pos]

    def test_positions_within_bounds(
        self, adapter: PlainTextDocumentAdapter, corpus_plaintext_dir: Path
    ) -> None:
        """All positions are within document bounds."""
        content = (corpus_plaintext_dir / "mixed.txt").read_text()
        chunks = adapter.parse(content)
        for chunk in chunks:
            assert 0 <= chunk.start_pos <= chunk.end_pos <= len(content)


class TestRoundTrip:
    """Tests for parse+reconstruct identity."""

    @pytest.mark.parametrize("filename", ["paragraphs.txt", "short.txt", "mixed.txt"])
    def test_round_trip_corpus(
        self, adapter: PlainTextDocumentAdapter, filename: str, corpus_plaintext_dir: Path
    ) -> None:
        """Parse and reconstruct produces identical output."""
        content = (corpus_plaintext_dir / filename).read_text()
        chunks = adapter.parse(content)
        result = adapter.reconstruct(content, chunks)
        assert result == content, f"Round-trip failed for {filename}"
