"""Chunk-level transform targeting heuristics.

These heuristics intentionally sit above parsing. The parser still classifies
headings and formula-like inline text as prose when they appear in prose
containers, but the transform pipeline can opt them out of paraphrase search.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.parser.chunk import ProseChunk

_MARKDOWN_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")
_BOLD_TITLE_RE = re.compile(r"^\s*\*\*[^*\n]{1,120}\*\*\s*$")
_TITLE_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")
_EQUATION_SYMBOL_RE = re.compile(r"[=|_\\/*+\-()]")
_FORMULA_TOKEN_RE = re.compile(
    r"(?:\b[A-Za-z]+_[A-Za-z]+\b|\b[A-Za-z]\([^)]*\)|\bsum\(|\bprod(?:uct)?\b|\b[DQMP]_[A-Z]+\b)"
)


def is_title_like_chunk(chunk: ProseChunk) -> bool:
    """Return True when a prose chunk behaves like a title or heading."""
    text = chunk.text.strip()
    if not text:
        return False
    if "\n" in text:
        return False
    if _MARKDOWN_HEADING_RE.match(text):
        return True
    if _BOLD_TITLE_RE.match(text):
        return True

    words = _TITLE_WORD_RE.findall(text)
    if not words or len(words) > 12:
        return False

    # Titles tend to be short noun phrases, not sentences.
    if any(text.endswith(mark) for mark in (".", "!", "?")):
        return False
    return ":" in text and len(text) <= 120


def is_equation_like_chunk(chunk: ProseChunk) -> bool:
    """Return True when a prose chunk is primarily symbolic/formulaic."""
    text = chunk.text.strip()
    if not text:
        return False
    if len(text) > 220:
        return False

    symbol_count = len(_EQUATION_SYMBOL_RE.findall(text))
    formula_markers = len(_FORMULA_TOKEN_RE.findall(text))
    has_equals = "=" in text
    newline_count = text.count("\n")

    if has_equals and symbol_count >= 4:
        return True
    if formula_markers >= 2 and symbol_count >= 3:
        return True
    return newline_count == 0 and has_equals and formula_markers >= 1


def skip_transform_reason(
    chunk: ProseChunk,
    *,
    skip_title_like: bool,
    skip_equation_like: bool,
) -> str | None:
    """Return a skip reason when a chunk should not enter transform search."""
    if skip_title_like and is_title_like_chunk(chunk):
        return "title_like"
    if skip_equation_like and is_equation_like_chunk(chunk):
        return "equation_like"
    return None
