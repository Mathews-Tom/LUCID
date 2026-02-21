"""Tests for the LaTeX document parser."""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid.parser.chunk import ChunkType, ProseChunk
from lucid.parser.latex import LatexDocumentAdapter


@pytest.fixture
def adapter() -> LatexDocumentAdapter:
    """Create a default LaTeX adapter."""
    return LatexDocumentAdapter()


class TestSimpleDocument:
    """Tests for basic LaTeX document parsing."""

    def test_preamble_is_structural(self, adapter: LatexDocumentAdapter) -> None:
        """Preamble before \\begin{document} is structural."""
        content = "\\documentclass{article}\n\\begin{document}\nHello\n\\end{document}\n"
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("\\documentclass" in c.text for c in structural)

    def test_body_prose_extracted(self, adapter: LatexDocumentAdapter) -> None:
        """Body text inside \\begin{document} is prose."""
        content = "\\documentclass{article}\n\\begin{document}\nHello world.\n\\end{document}\n"
        chunks = adapter.parse(content)
        prose = [c for c in chunks if c.chunk_type == ChunkType.PROSE]
        assert len(prose) >= 1
        assert any("Hello world." in c.text for c in prose)

    def test_chunks_sorted_by_position(self, adapter: LatexDocumentAdapter) -> None:
        """Chunks are sorted by start_pos."""
        content = Path("tests/corpus/latex/sections.tex").read_text()
        chunks = adapter.parse(content)
        positions = [c.start_pos for c in chunks]
        assert positions == sorted(positions)


class TestInlineMath:
    """Tests for inline math placeholder handling."""

    def test_inline_math_produces_placeholders(self, adapter: LatexDocumentAdapter) -> None:
        """Inline $...$ math is replaced with placeholders."""
        content = (
            "\\documentclass{article}\n\\begin{document}\n"
            "The value $x=1$ is given.\n\\end{document}\n"
        )
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk) and c.math_placeholders]
        assert len(prose) >= 1
        chunk = prose[0]
        assert len(chunk.math_placeholders) == 1
        ph = next(iter(chunk.math_placeholders))
        assert ph in chunk.text
        assert "$x=1$" in chunk.math_placeholders[ph]

    def test_multiple_inline_math(self, adapter: LatexDocumentAdapter) -> None:
        """Multiple inline math expressions in one paragraph."""
        content = Path("tests/corpus/latex/math_inline.tex").read_text()
        chunks = adapter.parse(content)
        prose_with_math = [c for c in chunks if isinstance(c, ProseChunk) and c.math_placeholders]
        assert len(prose_with_math) >= 1
        total_phs = sum(len(c.math_placeholders) for c in prose_with_math)
        assert total_phs == 3  # $E = mc^2$, $x > 0$, $f(x) = \sqrt{x}$

    def test_placeholder_format(self, adapter: LatexDocumentAdapter) -> None:
        """Placeholders use ⟨MATH_NNN⟩ format."""
        content = "\\documentclass{article}\n\\begin{document}\nValue $x$.\n\\end{document}\n"
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk) and c.math_placeholders]
        ph = next(iter(prose[0].math_placeholders))
        assert ph.startswith("[MATH_")
        assert ph.endswith("]")


class TestDisplayMath:
    """Tests for display math handling."""

    def test_equation_env_is_structural(self, adapter: LatexDocumentAdapter) -> None:
        """\\begin{equation} is structural."""
        content = Path("tests/corpus/latex/math_display.tex").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("equation" in c.text for c in structural)

    def test_display_dollar_is_structural(self, adapter: LatexDocumentAdapter) -> None:
        """$$...$$ display math is structural."""
        content = Path("tests/corpus/latex/math_display.tex").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("\\int" in c.text for c in structural)


class TestSections:
    """Tests for section/heading handling."""

    def test_section_title_is_prose(self, adapter: LatexDocumentAdapter) -> None:
        """\\section{Title} argument contains prose."""
        content = (
            "\\documentclass{article}\n\\begin{document}\n"
            "\\section{My Title}\nBody.\n\\end{document}\n"
        )
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("My Title" in c.text for c in prose)

    def test_footnote_text_is_prose(self, adapter: LatexDocumentAdapter) -> None:
        """\\footnote{text} argument contains prose."""
        content = Path("tests/corpus/latex/sections.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("footnote" in c.text.lower() for c in prose)


class TestEnvironments:
    """Tests for environment classification."""

    def test_figure_is_structural(self, adapter: LatexDocumentAdapter) -> None:
        """\\begin{figure} is structural."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("figure" in c.text for c in structural)

    def test_caption_prose_extracted(self, adapter: LatexDocumentAdapter) -> None:
        """\\caption inside figure has prose extracted."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("descriptive caption" in c.text for c in prose)

    def test_verbatim_preserved(self, adapter: LatexDocumentAdapter) -> None:
        """\\begin{verbatim} content preserved exactly."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("verbatim" in c.text for c in structural)

    def test_abstract_is_prose(self, adapter: LatexDocumentAdapter) -> None:
        """\\begin{abstract} body is prose."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("summarizes" in c.text for c in prose)

    def test_theorem_body_is_prose(self, adapter: LatexDocumentAdapter) -> None:
        """\\begin{theorem} body is prose."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        # The theorem body contains prose about round-trip
        assert any("round-trip" in c.text for c in prose)

    def test_proof_body_is_prose(self, adapter: LatexDocumentAdapter) -> None:
        """\\begin{proof} body is prose."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("position tracking" in c.text for c in prose)

    def test_itemize_items_are_prose(self, adapter: LatexDocumentAdapter) -> None:
        """\\begin{itemize} item text is prose."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("parsing approach" in c.text for c in prose)

    def test_enumerate_items_are_prose(self, adapter: LatexDocumentAdapter) -> None:
        """\\begin{enumerate} item text is prose."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("Step one" in c.text for c in prose)


class TestCitations:
    """Tests for citation/reference handling."""

    def test_cite_is_structural(self, adapter: LatexDocumentAdapter) -> None:
        """\\cite{} is structural."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("\\cite" in c.text for c in structural)

    def test_ref_is_structural(self, adapter: LatexDocumentAdapter) -> None:
        """\\ref{} is structural."""
        content = Path("tests/corpus/latex/sections.tex").read_text()
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("\\ref" in c.text for c in structural)


class TestPositionAccuracy:
    """Tests for position tracking correctness."""

    def test_positions_match_source(self, adapter: LatexDocumentAdapter) -> None:
        """chunk.text matches content[start_pos:end_pos] for structural chunks."""
        content = Path("tests/corpus/latex/sections.tex").read_text()
        chunks = adapter.parse(content)
        for chunk in chunks:
            if chunk.chunk_type == ChunkType.STRUCTURAL:
                assert chunk.text == content[chunk.start_pos : chunk.end_pos], (
                    f"Position mismatch for structural chunk at {chunk.start_pos}-{chunk.end_pos}"
                )

    def test_prose_positions_valid(self, adapter: LatexDocumentAdapter) -> None:
        """ProseChunk positions are within document bounds."""
        content = Path("tests/corpus/latex/complex.tex").read_text()
        chunks = adapter.parse(content)
        for chunk in chunks:
            assert 0 <= chunk.start_pos <= chunk.end_pos <= len(content)


class TestParagraphSplitting:
    """Tests for paragraph splitting on \\n\\n boundaries."""

    def test_multiple_paragraphs_split(self, adapter: LatexDocumentAdapter) -> None:
        """Multiple paragraphs separated by blank lines produce separate chunks."""
        content = Path("tests/corpus/latex/sections.tex").read_text()
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        # sections.tex has "second section contains multiple paragraphs" then
        # "This is the second paragraph..."
        texts = [c.text for c in prose]
        assert any("second section" in t for t in texts)
        assert any("second paragraph" in t for t in texts)


class TestNoDocument:
    """Tests for documents without \\begin{document}."""

    def test_no_begin_document(self, adapter: LatexDocumentAdapter) -> None:
        """Content without \\begin{document} is treated as prose."""
        content = "Hello world. This is plain LaTeX without document environment."
        chunks = adapter.parse(content)
        assert len(chunks) >= 1
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert len(prose) >= 1


class TestRoundTrip:
    """Tests for parse+reconstruct identity."""

    @pytest.mark.parametrize(
        "filename",
        ["simple.tex", "math_inline.tex", "math_display.tex", "sections.tex", "complex.tex"],
    )
    def test_round_trip_corpus(
        self, adapter: LatexDocumentAdapter, filename: str, corpus_latex_dir: Path
    ) -> None:
        """Parse and reconstruct produces identical output for corpus files."""
        content = (corpus_latex_dir / filename).read_text()
        chunks = adapter.parse(content)
        result = adapter.reconstruct(content, chunks)
        assert result == content, f"Round-trip failed for {filename}"


class TestCustomConfig:
    """Tests for custom environment/macro configuration."""

    def test_custom_prose_env(self) -> None:
        """Custom prose environments are recognized."""
        adapter = LatexDocumentAdapter(custom_prose_envs=("myenv",))
        content = (
            "\\documentclass{article}\n\\begin{document}\n"
            "\\begin{myenv}\nCustom prose.\n\\end{myenv}\n"
            "\\end{document}\n"
        )
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert any("Custom prose" in c.text for c in prose)

    def test_custom_structural_macro(self) -> None:
        """Custom structural macros are recognized."""
        adapter = LatexDocumentAdapter(custom_structural_macros=("mymacro",))
        content = "\\documentclass{article}\n\\begin{document}\n\\mymacro{arg}\n\\end{document}\n"
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("\\mymacro" in c.text for c in structural)


class TestUnknownEnvironment:
    """Tests for unknown environments."""

    def test_unknown_env_is_structural(self, adapter: LatexDocumentAdapter) -> None:
        """Unknown environments default to structural."""
        content = (
            "\\documentclass{article}\n\\begin{document}\n"
            "\\begin{unknownenv}\nContent.\n\\end{unknownenv}\n"
            "\\end{document}\n"
        )
        chunks = adapter.parse(content)
        structural = [c for c in chunks if c.chunk_type == ChunkType.STRUCTURAL]
        assert any("unknownenv" in c.text for c in structural)


class TestEdgeCases:
    """Tests for edge cases and defensive guards."""

    def test_empty_document(self, adapter: LatexDocumentAdapter) -> None:
        """Empty document body produces no prose chunks."""
        content = "\\documentclass{article}\n\\begin{document}\n\\end{document}\n"
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert len(prose) == 0

    def test_whitespace_only_body(self, adapter: LatexDocumentAdapter) -> None:
        """Body with only whitespace produces no prose."""
        content = "\\documentclass{article}\n\\begin{document}\n  \n  \n\\end{document}\n"
        chunks = adapter.parse(content)
        prose = [c for c in chunks if isinstance(c, ProseChunk)]
        assert len(prose) == 0

    def test_empty_string(self, adapter: LatexDocumentAdapter) -> None:
        """Empty string input."""
        chunks = adapter.parse("")
        assert chunks == []
