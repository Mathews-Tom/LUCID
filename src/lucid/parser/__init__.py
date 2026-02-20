"""Document parsing: LaTeX, Markdown, plain text."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from lucid.parser.latex import LatexDocumentAdapter
from lucid.parser.markdown import MarkdownDocumentAdapter
from lucid.parser.plaintext import PlainTextDocumentAdapter

if TYPE_CHECKING:
    from lucid.config import LUCIDConfig
    from lucid.core.protocols import DocumentAdapter

__all__ = [
    "LatexDocumentAdapter",
    "MarkdownDocumentAdapter",
    "PlainTextDocumentAdapter",
    "detect_format",
    "get_adapter",
]

_EXTENSION_MAP: dict[str, str] = {
    ".tex": "latex",
    ".ltx": "latex",
    ".latex": "latex",
    ".md": "markdown",
    ".markdown": "markdown",
    ".mkd": "markdown",
    ".txt": "plaintext",
    ".text": "plaintext",
}


def detect_format(path: str | Path) -> str:
    """Detect document format from file extension.

    Args:
        path: File path (only the extension is used).

    Returns:
        Format string: "latex", "markdown", or "plaintext".

    Raises:
        ValueError: If the extension is not recognized.
    """
    ext = Path(path).suffix.lower()
    fmt = _EXTENSION_MAP.get(ext)
    if fmt is None:
        raise ValueError(f"Unrecognized file extension: {ext!r}")
    return fmt


def get_adapter(
    fmt: str,
    config: LUCIDConfig | None = None,
) -> DocumentAdapter:
    """Create a document adapter for the given format.

    Args:
        fmt: Format string ("latex", "markdown", "plaintext").
        config: LUCID configuration for parser customization.

    Returns:
        A DocumentAdapter instance.

    Raises:
        ValueError: If the format is not recognized.
    """
    if fmt == "latex":
        if config is not None:
            return LatexDocumentAdapter(
                custom_prose_envs=config.parser.custom_prose_environments,
                custom_structural_macros=config.parser.custom_structural_macros,
            )
        return LatexDocumentAdapter()

    if fmt == "markdown":
        return MarkdownDocumentAdapter()

    if fmt == "plaintext":
        return PlainTextDocumentAdapter()

    raise ValueError(f"Unknown format: {fmt!r}")
