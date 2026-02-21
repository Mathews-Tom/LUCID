"""Tests for the LaTeX reconstructor."""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid.parser.chunk import Chunk, ChunkType, ProseChunk
from lucid.parser.latex import LatexDocumentAdapter
from lucid.reconstructor.latex import reconstruct_latex, restore_placeholders


@pytest.fixture
def adapter() -> LatexDocumentAdapter:
    """Create a default LaTeX adapter."""
    return LatexDocumentAdapter()


class TestNoModIdentity:
    """Tests that unmodified chunks reproduce the original."""

    def test_identity_simple(self, adapter: LatexDocumentAdapter) -> None:
        """No modifications → identical output."""
        content = Path("tests/corpus/latex/simple.tex").read_text()
        chunks = adapter.parse(content)
        result = reconstruct_latex(content, chunks)
        assert result == content

    def test_identity_complex(self, adapter: LatexDocumentAdapter) -> None:
        """No modifications → identical output for complex document."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        result = reconstruct_latex(content, chunks)
        assert result == content


class TestSingleReplacement:
    """Tests for replacing a single prose chunk."""

    def test_replace_one_chunk(self, adapter: LatexDocumentAdapter) -> None:
        """Replacing one prose chunk changes only that region."""
        content = Path("tests/corpus/latex/simple.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        # Find the body prose chunk
        body = [c for c in prose if "simple paragraph" in c.text]
        assert len(body) == 1

        # Modify it
        body[0].metadata["humanized_text"] = "This is a modified paragraph."
        result = reconstruct_latex(content, chunks)

        assert "modified paragraph" in result
        assert "simple paragraph" not in result
        assert "\\documentclass" in result  # structural preserved


class TestMultipleReplacements:
    """Tests for replacing multiple prose chunks."""

    def test_replace_multiple_chunks(self, adapter: LatexDocumentAdapter) -> None:
        """Multiple replacements applied correctly."""
        content = Path("tests/corpus/latex/sections.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]

        # Modify two specific chunks
        modified_count = 0
        for chunk in prose:
            if chunk.text == "First Section":
                chunk.metadata["humanized_text"] = "Modified Title."
                modified_count += 1
            elif "introductory text" in chunk.text:
                chunk.metadata["humanized_text"] = "Modified body text."
                modified_count += 1

        assert modified_count == 2
        result = reconstruct_latex(content, chunks)
        assert "Modified Title." in result
        assert "Modified body text." in result


class TestPlaceholderRestoration:
    """Tests for placeholder restoration."""

    def test_restore_math_placeholders(self) -> None:
        """Math placeholders restored to original expressions."""
        text = "The value [MATH_001] is important."
        placeholders = {"[MATH_001]": "$x = 42$"}
        result = restore_placeholders(text, placeholders)
        assert result == "The value $x = 42$ is important."

    def test_restore_multiple_placeholders(self) -> None:
        """Multiple placeholders restored correctly."""
        text = "[MATH_001] and [MATH_002]"
        placeholders = {
            "[MATH_001]": "$a$",
            "[MATH_002]": "$b$",
        }
        result = restore_placeholders(text, placeholders)
        assert result == "$a$ and $b$"

    def test_restore_empty_placeholders(self) -> None:
        """Empty placeholder dict returns text unchanged."""
        text = "No placeholders here."
        result = restore_placeholders(text, {})
        assert result == text

    def test_roundtrip_with_math_modification(self, adapter: LatexDocumentAdapter) -> None:
        """Modifying text with math placeholders preserves math on reconstruction."""
        content = Path("tests/corpus/latex/math_inline.tex").read_text()
        chunks = adapter.parse(content)
        prose_with_math = [c for c in chunks if isinstance(c, ProseChunk) and c.math_placeholders]
        assert len(prose_with_math) >= 1

        # Simulate modifying text while keeping placeholders
        chunk = prose_with_math[0]
        # Replace the text around the placeholders
        new_text = chunk.text.replace("describes", "represents")
        chunk.metadata["humanized_text"] = new_text

        result = reconstruct_latex(content, chunks)
        assert "represents" in result
        assert "$E = mc^2$" in result  # Math expression preserved


class TestNonProseChunkSkipped:
    """Tests that base Chunk with PROSE type is skipped by isinstance guard."""

    def test_base_chunk_with_prose_type_skipped(self) -> None:
        """A Chunk (not ProseChunk) with chunk_type=PROSE is ignored."""
        original = "Hello world."
        fake_chunk = Chunk(
            text="Hello world.",
            chunk_type=ChunkType.PROSE,
            start_pos=0,
            end_pos=12,
        )
        result = reconstruct_latex(original, [fake_chunk])
        assert result == original


class TestDirectTextModification:
    """Tests for chunks modified via text attribute rather than metadata."""

    def test_chunk_text_differs_from_original(self, adapter: LatexDocumentAdapter) -> None:
        """Chunk with text != original slice triggers replacement without humanized_text."""
        content = Path("tests/corpus/latex/simple.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        body = [c for c in prose if "simple paragraph" in c.text]
        assert len(body) == 1

        # Modify chunk.text directly (no humanized_text metadata)
        body[0].text = "Direct modification."
        result = reconstruct_latex(content, chunks)
        assert "Direct modification." in result
        assert "simple paragraph" not in result
