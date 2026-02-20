"""Tests for parser __init__ module: detect_format and get_adapter."""

from __future__ import annotations

import pytest

from lucid.config import load_config
from lucid.parser import (
    LatexDocumentAdapter,
    MarkdownDocumentAdapter,
    PlainTextDocumentAdapter,
    detect_format,
    get_adapter,
)
from lucid.parser.base import DocumentAdapter


class TestDetectFormat:
    """Tests for file format detection."""

    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            ("document.tex", "latex"),
            ("document.ltx", "latex"),
            ("document.latex", "latex"),
            ("README.md", "markdown"),
            ("notes.markdown", "markdown"),
            ("file.mkd", "markdown"),
            ("file.txt", "plaintext"),
            ("file.text", "plaintext"),
            ("/path/to/deep/file.tex", "latex"),
            ("UPPER.TEX", "latex"),
            ("UPPER.MD", "markdown"),
        ],
    )
    def test_detect_known_formats(self, path: str, expected: str) -> None:
        """Known extensions map to correct format strings."""
        assert detect_format(path) == expected

    def test_detect_unknown_extension(self) -> None:
        """Unrecognized extension raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized file extension"):
            detect_format("document.docx")

    def test_detect_no_extension(self) -> None:
        """File without extension raises ValueError."""
        with pytest.raises(ValueError, match="Unrecognized file extension"):
            detect_format("Makefile")


class TestGetAdapter:
    """Tests for adapter factory."""

    def test_get_latex_adapter(self) -> None:
        """get_adapter('latex') returns LatexDocumentAdapter."""
        adapter = get_adapter("latex")
        assert isinstance(adapter, LatexDocumentAdapter)

    def test_get_markdown_adapter(self) -> None:
        """get_adapter('markdown') returns MarkdownDocumentAdapter."""
        adapter = get_adapter("markdown")
        assert isinstance(adapter, MarkdownDocumentAdapter)

    def test_get_plaintext_adapter(self) -> None:
        """get_adapter('plaintext') returns PlainTextDocumentAdapter."""
        adapter = get_adapter("plaintext")
        assert isinstance(adapter, PlainTextDocumentAdapter)

    def test_get_adapter_with_config(self) -> None:
        """get_adapter with config passes custom settings to LaTeX adapter."""
        config = load_config(profile="balanced")
        adapter = get_adapter("latex", config=config)
        assert isinstance(adapter, LatexDocumentAdapter)

    def test_get_adapter_unknown_format(self) -> None:
        """Unknown format string raises ValueError."""
        with pytest.raises(ValueError, match="Unknown format"):
            get_adapter("html")


class TestBaseExport:
    """Tests for parser.base re-export."""

    def test_document_adapter_importable(self) -> None:
        """DocumentAdapter is importable from parser.base."""
        assert DocumentAdapter is not None
