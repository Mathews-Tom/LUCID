"""Tests for the Markdown document parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid.parser.chunk import ChunkType, ProseChunk
from lucid.parser.markdown import MarkdownDocumentAdapter


@pytest.fixture
def adapter() -> MarkdownDocumentAdapter:
    """Create a default Markdown adapter."""
    return MarkdownDocumentAdapter()


class TestParagraphs:
    """Tests for paragraph extraction."""

    def test_paragraphs_are_prose(self, adapter: MarkdownDocumentAdapter) -> None:
        """Paragraphs are classified as prose."""
        content = Path("tests/corpus/markdown/simple.md").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert len(prose) >= 3
        assert any("first paragraph" in c.text for c in prose)

    def test_line_positions(self, adapter: MarkdownDocumentAdapter) -> None:
        """Chunk positions are line indices."""
        content = Path("tests/corpus/markdown/simple.md").read_text()
        lines = content.split("\n")
        chunks = adapter.parse(content)
        for chunk in chunks:
            assert chunk.metadata.get("position_type") == "line"
            text = "\n".join(lines[chunk.start_pos : chunk.end_pos])
            assert text == chunk.text, f"Line range {chunk.start_pos}-{chunk.end_pos} mismatch"


class TestHeadings:
    """Tests for heading extraction."""

    def test_headings_are_prose(self, adapter: MarkdownDocumentAdapter) -> None:
        """Headings are classified as prose."""
        content = Path("tests/corpus/markdown/simple.md").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("# First Heading" in c.text for c in prose)
        assert any("## Second Heading" in c.text for c in prose)
        assert any("### Third Level" in c.text for c in prose)


class TestCodeBlocks:
    """Tests for code block handling."""

    def test_fenced_code_is_structural(self, adapter: MarkdownDocumentAdapter) -> None:
        """Fenced code blocks are structural."""
        content = Path("tests/corpus/markdown/code_blocks.md").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("def hello" in c.text for c in structural)

    def test_plain_code_block_structural(self, adapter: MarkdownDocumentAdapter) -> None:
        """Code blocks without language spec are structural."""
        content = Path("tests/corpus/markdown/code_blocks.md").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("Plain code block" in c.text for c in structural)


class TestTables:
    """Tests for table handling."""

    def test_tables_are_structural(self, adapter: MarkdownDocumentAdapter) -> None:
        """Tables are classified as structural."""
        content = Path("tests/corpus/markdown/complex.md").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("Column A" in c.text for c in structural)


class TestFrontMatter:
    """Tests for YAML front matter."""

    def test_front_matter_is_structural(self, adapter: MarkdownDocumentAdapter) -> None:
        """YAML front matter is structural."""
        content = Path("tests/corpus/markdown/complex.md").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("title:" in c.text for c in structural)


class TestMathBlocks:
    """Tests for math block handling."""

    def test_display_math_is_structural(self, adapter: MarkdownDocumentAdapter) -> None:
        """$$...$$ display math blocks are structural."""
        content = Path("tests/corpus/markdown/math.md").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("\\int" in c.text for c in structural)

    def test_inline_math_placeholders(self, adapter: MarkdownDocumentAdapter) -> None:
        """Inline $...$ math produces placeholders."""
        content = Path("tests/corpus/markdown/math.md").read_text()
        chunks = adapter.parse(content)
        prose_with_math = [c for c in chunks if isinstance(c, ProseChunk) and c.math_placeholders]
        assert len(prose_with_math) >= 1


class TestImages:
    """Tests for image handling."""

    def test_image_only_paragraph_is_structural(self, adapter: MarkdownDocumentAdapter) -> None:
        """Paragraph containing only an image is structural."""
        content = Path("tests/corpus/markdown/complex.md").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("![" in c.text for c in structural)


class TestBlockquotes:
    """Tests for blockquote handling."""

    def test_blockquotes_are_prose(self, adapter: MarkdownDocumentAdapter) -> None:
        """Blockquotes are classified as prose."""
        content = Path("tests/corpus/markdown/complex.md").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("blockquote" in c.text.lower() for c in prose)


class TestLists:
    """Tests for list handling."""

    def test_bullet_list_is_prose(self, adapter: MarkdownDocumentAdapter) -> None:
        """Bullet lists are classified as prose."""
        content = Path("tests/corpus/markdown/lists.md").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("First item" in c.text for c in prose)

    def test_ordered_list_is_prose(self, adapter: MarkdownDocumentAdapter) -> None:
        """Ordered lists are classified as prose."""
        content = Path("tests/corpus/markdown/lists.md").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("First ordered" in c.text for c in prose)

    def test_nested_list_is_prose(self, adapter: MarkdownDocumentAdapter) -> None:
        """Nested lists are classified as prose."""
        content = Path("tests/corpus/markdown/lists.md").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("Inner item" in c.text for c in prose)


class TestChunkOrdering:
    """Tests for chunk ordering."""

    def test_sorted_by_start_pos(self, adapter: MarkdownDocumentAdapter) -> None:
        """Chunks are sorted by start_pos."""
        content = Path("tests/corpus/markdown/complex.md").read_text()
        chunks = adapter.parse(content)
        positions = [c.start_pos for c in chunks]
        assert positions == sorted(positions)


class TestEdgeCases:
    """Tests for edge cases in Markdown parsing."""

    def test_empty_string(self, adapter: MarkdownDocumentAdapter) -> None:
        """Empty string produces no chunks."""
        chunks = adapter.parse("")
        assert chunks == []

    def test_html_block_is_structural(self, adapter: MarkdownDocumentAdapter) -> None:
        """HTML blocks are structural."""
        content = "<div>\n  <p>Hello</p>\n</div>\n\nParagraph.\n"
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("<div>" in c.text for c in structural)

    def test_horizontal_rule_is_structural(self, adapter: MarkdownDocumentAdapter) -> None:
        """Horizontal rules are structural."""
        content = "Paragraph above.\n\n---\n\nParagraph below.\n"
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("---" in c.text for c in structural)


class TestRoundTrip:
    """Tests for parse+reconstruct identity."""

    @pytest.mark.parametrize(
        "filename",
        ["simple.md", "code_blocks.md", "lists.md", "math.md", "complex.md"],
    )
    def test_round_trip_corpus(
        self, adapter: MarkdownDocumentAdapter, filename: str, corpus_markdown_dir: Path
    ) -> None:
        """Parse and reconstruct produces identical output for corpus files."""
        content = (corpus_markdown_dir / filename).read_text()
        chunks = adapter.parse(content)
        result = adapter.reconstruct(content, chunks)
        assert result == content, f"Round-trip failed for {filename}"
