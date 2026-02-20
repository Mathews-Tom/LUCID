"""Tests for the reconstruction validator."""

from __future__ import annotations

import subprocess
from unittest.mock import patch

from lucid.reconstructor.validator import (
    ValidationResult,
    validate_latex,
    validate_markdown,
)


class TestValidateLatex:
    """Tests for LaTeX validation."""

    def test_valid_document(self) -> None:
        """Valid LaTeX compiles successfully (if compiler available)."""
        content = "\\documentclass{article}\n\\begin{document}\nHello.\n\\end{document}\n"
        result = validate_latex(content)
        # valid=True if compiler installed, valid=None if not
        assert result.valid is not False

    def test_invalid_document(self) -> None:
        """Invalid LaTeX reports errors (if compiler available)."""
        content = "\\documentclass{article}\n\\begin{document}\n\\badcommand\n\\end{document}\n"
        result = validate_latex(content)
        if result.valid is not None:
            # Compiler is available — should report error
            assert result.valid is False
            assert len(result.errors) > 0

    def test_compiler_not_found(self) -> None:
        """Missing compiler returns valid=None."""
        result = validate_latex("test", compiler="nonexistent_latex_compiler_xyz")
        assert result.valid is None
        assert len(result.errors) == 0


class TestValidateMarkdown:
    """Tests for Markdown validation."""

    def test_valid_markdown(self) -> None:
        """Valid Markdown passes validation."""
        content = "# Heading\n\nParagraph with $x=1$ math.\n"
        result = validate_markdown(content)
        assert result.valid is True

    def test_unbalanced_dollars(self) -> None:
        """Unbalanced $ delimiters are detected."""
        content = "# Heading\n\nParagraph with $x=1 unbalanced.\n"
        result = validate_markdown(content)
        assert result.valid is False
        assert any("Unbalanced" in e.message for e in result.errors)

    def test_balanced_dollars_in_code_blocks(self) -> None:
        """Dollar signs inside code blocks are ignored."""
        content = "# Heading\n\n```\n$not_math$\n```\n\nParagraph.\n"
        result = validate_markdown(content)
        assert result.valid is True

    def test_validation_result_dataclass(self) -> None:
        """ValidationResult is immutable (frozen dataclass)."""
        result = ValidationResult(valid=True)
        assert result.valid is True
        assert result.errors == ()

    def test_escaped_dollars_not_counted(self) -> None:
        r"""Escaped \$ are not counted as delimiters."""
        content = r"Price is \$5 and \$10." + "\n"
        result = validate_markdown(content)
        assert result.valid is True


class TestValidateLatexEdgeCases:
    """Tests for LaTeX validation edge cases."""

    def test_compilation_timeout(self) -> None:
        """Compilation timeout returns valid=False with timeout message."""
        doc = r"\documentclass{article}" "\n" r"\begin{document}" "\nHi\n" r"\end{document}" "\n"
        with patch("lucid.reconstructor.validator.subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="pdflatex", timeout=30,
            )
            result = validate_latex(doc)
            assert result.valid is False
            assert any("timed out" in e.message.lower() for e in result.errors)

    def test_log_parse_fallback(self) -> None:
        """When log has no parseable errors, stderr/stdout is used."""
        doc = r"\documentclass{article}" "\n" r"\begin{document}" "\nHi\n" r"\end{document}" "\n"
        mock_result = subprocess.CompletedProcess(
            args=["pdflatex"],
            returncode=1,
            stdout="Some compiler output",
            stderr="",
        )
        with (
            patch("lucid.reconstructor.validator.subprocess.run", return_value=mock_result),
            patch("pathlib.Path.exists", return_value=False),
        ):
            result = validate_latex(doc)
            assert result.valid is False
            assert len(result.errors) >= 1


class TestValidateMarkdownEdgeCases:
    """Tests for Markdown validation edge cases."""

    def test_markdown_parse_exception(self) -> None:
        """MarkdownIt parse exception is captured."""
        with patch("lucid.reconstructor.validator.MarkdownIt") as mock_md_cls:
            mock_md_cls.return_value.parse.side_effect = RuntimeError("Parse failed")
            result = validate_markdown("# Heading\n")
            assert result.valid is False
            assert any("Parse error" in e.message for e in result.errors)
