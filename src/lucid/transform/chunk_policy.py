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
# Numbered section heading: "2.1.1 Title" or "A.3 Title"
_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:[A-Z]\.|\d+\.)+\d*\s+[A-Z]"
)
_MATH_PLACEHOLDER_RE = re.compile(r"\[MATH_\d{3}\]")
_EQUATION_SYMBOL_RE = re.compile(r"[=|_\\/*+\-()]")
_FORMULA_TOKEN_RE = re.compile(
    r"(?:\b[A-Za-z]+_[A-Za-z]+\b|\b[A-Za-z]\([^)]*\)|\bsum\(|\bprod(?:uct)?\b|\b[DQMP]_[A-Z]+\b)"
)
_PURE_MATH_LINE_RE = re.compile(
    r"^\s*(?:\[MATH_\d{3}\](?:\s*(?:,|and|or)?\s*)?)+[.:;]?\s*$"
)
_MATH_EXPLANATION_RE = re.compile(r"^\s*(?:where|here|with)\s+\[MATH_\d{3}\]\b", re.IGNORECASE)

# Bare citation key: "AuthorYear" or "Author2020" (no spaces)
_BARE_CITATION_KEY_RE = re.compile(r"^[A-Za-z]+\d{4}[a-z]?$")
# LaTeX line break or spacing: just backslashes and whitespace
_LATEX_LINEBREAK_RE = re.compile(r"^\s*\\{1,2}\s*$")

# Default minimum prose length for transformation
DEFAULT_MIN_PROSE_LENGTH = 50


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
    if _NUMBERED_HEADING_RE.match(text) and len(text) <= 120:
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


def is_math_heavy_chunk(chunk: ProseChunk) -> bool:
    """Return True when a prose chunk is dominated by math placeholders."""
    text = chunk.text.strip()
    if not text:
        return False

    math_placeholder_count = len(chunk.math_placeholders)
    if math_placeholder_count == 0:
        math_placeholder_count = len(_MATH_PLACEHOLDER_RE.findall(text))
    if math_placeholder_count == 0:
        return False

    if _PURE_MATH_LINE_RE.fullmatch(text):
        return True
    if math_placeholder_count >= 2 and len(text) <= 340:
        return True
    return len(text) <= 220 and bool(_MATH_EXPLANATION_RE.match(text))


def is_too_short_to_transform(chunk: ProseChunk, min_length: int) -> bool:
    """Return True when a prose chunk is too short for meaningful rewriting."""
    text = chunk.text.strip()
    if not text:
        return True
    if len(text) < min_length:
        return True
    return False


def is_latex_fragment(chunk: ProseChunk) -> bool:
    """Return True for LaTeX structural fragments that are not real prose.

    Catches line breaks (``\\\\``), bare citation keys (``AuthorYear``),
    and single-word fragments that the parser mis-classified as prose.
    """
    text = chunk.text.strip()
    if not text:
        return False
    if _LATEX_LINEBREAK_RE.match(text):
        return True
    if _BARE_CITATION_KEY_RE.match(text):
        return True
    # Single word (no spaces) under 30 chars — not a sentence
    if " " not in text and len(text) < 30:
        return True
    return False


def skip_transform_reason(
    chunk: ProseChunk,
    *,
    skip_title_like: bool,
    skip_equation_like: bool,
    skip_math_heavy: bool,
    min_prose_length: int = DEFAULT_MIN_PROSE_LENGTH,
) -> str | None:
    """Return a skip reason when a chunk should not enter transform search."""
    if is_too_short_to_transform(chunk, min_length=min_prose_length):
        return "too_short"
    if is_latex_fragment(chunk):
        return "latex_fragment"
    if skip_title_like and is_title_like_chunk(chunk):
        return "title_like"
    if skip_math_heavy and is_math_heavy_chunk(chunk):
        return "math_heavy"
    if skip_equation_like and is_equation_like_chunk(chunk):
        return "equation_like"
    return None
