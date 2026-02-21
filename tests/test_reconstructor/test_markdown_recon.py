"""Tests for the Markdown reconstructor."""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid.parser.chunk import Chunk, ChunkType, ProseChunk
from lucid.parser.markdown import MarkdownDocumentAdapter
from lucid.reconstructor.markdown import reconstruct_markdown


@pytest.fixture
def adapter() -> MarkdownDocumentAdapter:
    """Create a default Markdown adapter."""
    return MarkdownDocumentAdapter()


class TestNoModIdentity:
    """Tests that unmodified chunks reproduce the original."""

    def test_identity_simple(self, adapter: MarkdownDocumentAdapter) -> None:
        """No modifications → identical output."""
        content = Path("tests/corpus/markdown/simple.md").read_text()
        chunks = adapter.parse(content)
        result = reconstruct_markdown(content, chunks)
        assert result == content

    def test_identity_complex(self, adapter: MarkdownDocumentAdapter) -> None:
        """No modifications → identical output for complex document."""
        content = Path("tests/corpus/markdown/complex.md").read_text()
        chunks = adapter.parse(content)
        result = reconstruct_markdown(content, chunks)
        assert result == content


class TestBlockReplacement:
    """Tests for replacing prose blocks."""

    def test_replace_paragraph(self, adapter: MarkdownDocumentAdapter) -> None:
        """Replacing a paragraph updates the correct lines."""
        content = Path("tests/corpus/markdown/simple.md").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]

        # Find a body paragraph
        body = [c for c in prose if "first paragraph" in c.text]
        assert len(body) == 1

        body[0].metadata["humanized_text"] = "This is a replaced paragraph."
        result = reconstruct_markdown(content, chunks)

        assert "replaced paragraph" in result
        assert "first paragraph" not in result
        assert "# First Heading" in result  # Other content preserved


class TestLineCountChanges:
    """Tests for replacements that change line counts."""

    def test_fewer_lines(self, adapter: MarkdownDocumentAdapter) -> None:
        """Replacement with fewer lines works correctly."""
        content = Path("tests/corpus/markdown/simple.md").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]

        # Find multi-line paragraph
        multi = [c for c in prose if "\n" in c.text]
        if multi:
            multi[0].metadata["humanized_text"] = "Single line."
            result = reconstruct_markdown(content, chunks)
            assert "Single line." in result

    def test_more_lines(self, adapter: MarkdownDocumentAdapter) -> None:
        """Replacement with more lines works correctly."""
        content = Path("tests/corpus/markdown/simple.md").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]

        # Find the short paragraph
        short = [c for c in prose if "short paragraph" in c.text.lower()]
        if short:
            short[0].metadata["humanized_text"] = "Line one.\nLine two.\nLine three."
            result = reconstruct_markdown(content, chunks)
            assert "Line one.\nLine two.\nLine three." in result


class TestBlankLinePreservation:
    """Tests for blank line handling."""

    def test_blank_lines_preserved(self, adapter: MarkdownDocumentAdapter) -> None:
        """Blank lines between blocks are preserved."""
        content = "# Heading\n\nParagraph one.\n\nParagraph two.\n"
        chunks = adapter.parse(content)
        result = reconstruct_markdown(content, chunks)
        assert result == content


class TestPlaceholderRestoration:
    """Tests for math placeholder restoration during Markdown reconstruction."""

    def test_math_placeholders_restored(self, adapter: MarkdownDocumentAdapter) -> None:
        """Math placeholders in modified text are restored to original expressions."""
        content = Path("tests/corpus/markdown/math.md").read_text()
        chunks = adapter.parse(content)
        prose_with_math = [c for c in chunks if isinstance(c, ProseChunk) and c.math_placeholders]
        assert len(prose_with_math) >= 1

        chunk = prose_with_math[0]
        # Modify text around placeholders
        new_text = chunk.text.replace("equation", "formula")
        chunk.metadata["humanized_text"] = new_text

        result = reconstruct_markdown(content, chunks)
        assert "formula" in result
        # Original math expression restored (not placeholder token)
        assert "[MATH_" not in result

    def test_placeholder_restore_yields_identity_skips(
        self, adapter: MarkdownDocumentAdapter
    ) -> None:
        """Chunk whose modified text equals original after placeholder restore is skipped."""
        content = "Paragraph with $x$ inside.\n"
        chunks = adapter.parse(content)
        prose_with_math = [c for c in chunks if isinstance(c, ProseChunk) and c.math_placeholders]
        assert len(prose_with_math) == 1

        chunk = prose_with_math[0]
        # Set humanized_text to the placeholder form — after restoration it matches original
        chunk.metadata["humanized_text"] = chunk.text

        result = reconstruct_markdown(content, chunks)
        assert result == content


class TestStructuralChunkSkipped:
    """Tests that non-ProseChunk types are skipped during reconstruction."""

    def test_structural_chunks_ignored(self, adapter: MarkdownDocumentAdapter) -> None:
        """Structural chunks are not considered for replacement."""
        content = "# Heading\n\n```python\ncode\n```\n\nParagraph.\n"
        chunks = adapter.parse(content)
        # No modifications — should get identity
        result = reconstruct_markdown(content, chunks)
        assert result == content

    def test_base_chunk_with_prose_type_skipped(self) -> None:
        """A Chunk (not ProseChunk) with chunk_type=PROSE is ignored."""
        original = "Line one\nLine two\n"
        fake_chunk = Chunk(
            text="Line one",
            chunk_type=ChunkType.PROSE,
            start_pos=0,
            end_pos=1,
        )
        result = reconstruct_markdown(original, [fake_chunk])
        assert result == original
