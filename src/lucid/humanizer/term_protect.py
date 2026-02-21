"""Term protection for the LUCID humanization pipeline.

Replaces domain-specific terms, citations, named entities, custom terms,
numbers, and capitalized multi-word phrases with stable placeholders before
LLM rewriting, then restores them afterward.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.config import TermProtectionConfig
    from lucid.parser.chunk import ProseChunk

# ---------------------------------------------------------------------------
# Placeholder constants
# ---------------------------------------------------------------------------

_OPEN = "\u27e8"   # ⟨  mathematical left angle bracket
_CLOSE = "\u27e9"  # ⟩  mathematical right angle bracket

# Match any existing placeholder (MATH or TERM)
_PLACEHOLDER_RE = re.compile(
    r"\u27e8(?:MATH|TERM)_(\d{3})\u27e9"
)

# ---------------------------------------------------------------------------
# Citation regex patterns
# ---------------------------------------------------------------------------

CITATION_PATTERNS: list[str] = [
    # [Author, Year]  or  [Author & Author, Year]
    r"\[([A-Z][a-zA-Z\s&]+(?:,\s*[A-Z][a-zA-Z\s&]+)*,\s*\d{4}[a-z]?)\]",
    # (Author et al., Year)
    r"\(([A-Z][a-zA-Z\s]+\s+et\s+al\.,\s*\d{4}[a-z]?)\)",
    # (Author, Year)  or  (Author & Author, Year)
    r"\(([A-Z][a-zA-Z\s&]+(?:,\s*[A-Z][a-zA-Z\s&]+)*,\s*\d{4}[a-z]?)\)",
    # Author (Year)  or  Author and Author (Year)
    r"([A-Z][a-zA-Z]+(?:\s+(?:and|&)\s+[A-Z][a-zA-Z]+)?)\s+\((\d{4}[a-z]?)\)",
]

# ---------------------------------------------------------------------------
# Number regex
# ---------------------------------------------------------------------------

_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?\b")

# ---------------------------------------------------------------------------
# Capitalized multi-word phrase (2+ consecutive capitalized words)
# ---------------------------------------------------------------------------

_CAP_PHRASE_RE = re.compile(r"\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)+\b")

# ---------------------------------------------------------------------------
# Public data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProtectedText:
    """Result of term protection.

    Attributes:
        text: Text with all placeholders substituted.
        term_placeholders: Mapping of ⟨TERM_NNN⟩ → original term.
        math_placeholders: Pass-through from the source ProseChunk.
    """

    text: str
    term_placeholders: dict[str, str]
    math_placeholders: dict[str, str]


@dataclass(frozen=True, slots=True)
class ValidationResult:
    """Result of placeholder validation.

    Attributes:
        is_valid: True when every expected placeholder is present in text.
        missing_placeholders: Placeholders absent from the text.
    """

    is_valid: bool
    missing_placeholders: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _max_math_index(math_placeholders: dict[str, str]) -> int:
    """Return the highest numeric index in math_placeholders, or -1 if empty."""
    if not math_placeholders:
        return -1
    max_idx = -1
    for key in math_placeholders:
        m = _PLACEHOLDER_RE.fullmatch(key)
        if m:
            idx = int(m.group(1))
            if idx > max_idx:
                max_idx = idx
    return max_idx


def _spans_overlap(
    existing: set[tuple[int, int]], start: int, end: int
) -> bool:
    """Return True if (start, end) overlaps any span in existing."""
    return any(start < e and end > s for s, e in existing)


def _collect_spans_from_patterns(
    text: str,
    patterns: list[str],
    existing: set[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    """Return non-overlapping (start, end, matched_text) for all patterns."""
    found: list[tuple[int, int, str]] = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            start, end = m.start(), m.end()
            matched = m.group(0)
            if not _spans_overlap(existing, start, end):
                found.append((start, end, matched))
                existing.add((start, end))
    return found


# ---------------------------------------------------------------------------
# TermProtector
# ---------------------------------------------------------------------------


class TermProtector:
    """Protect domain terms in prose chunks before LLM humanization.

    Lazy-loads the spaCy NER model on first use when ``config.use_ner`` is
    True. The model is cached on the instance after the first load.
    """

    def __init__(self, config: TermProtectionConfig) -> None:
        self._config = config
        self._nlp: object | None = None  # spaCy Language, loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def protect(self, chunk: ProseChunk) -> ProtectedText:
        """Replace domain terms in chunk.protected_text with placeholders.

        Args:
            chunk: The ProseChunk to protect. Not mutated.

        Returns:
            ProtectedText with placeholder-substituted text and both
            placeholder mappings.
        """
        text = chunk.protected_text
        math_placeholders = dict(chunk.math_placeholders)

        # Build the set of spans already occupied by math placeholders so
        # all other extractors skip over them.
        occupied: set[tuple[int, int]] = set()
        for m in _PLACEHOLDER_RE.finditer(text):
            occupied.add((m.start(), m.end()))

        # Determine starting index for TERM placeholders.
        start_idx = _max_math_index(math_placeholders) + 1
        counter = [start_idx]  # mutable reference shared by inner function

        # Accumulate spans in priority order (highest priority first).
        # Each extractor adds to `occupied` before the next one runs.
        spans: list[tuple[int, int, str]] = []

        # Priority 1: math placeholders — already in `occupied`, skip.

        # Priority 2: citations
        if self._config.protect_citations:
            new = _collect_spans_from_patterns(text, CITATION_PATTERNS, occupied)
            spans.extend(new)

        # Priority 3: named entities via spaCy NER
        if self._config.use_ner:
            new = self._extract_ner_spans(text, occupied)
            spans.extend(new)

        # Priority 4: custom terms
        if self._config.custom_terms:
            new = self._extract_custom_spans(text, occupied)
            spans.extend(new)

        # Priority 5: numbers
        if self._config.protect_numbers:
            new = self._extract_number_spans(text, occupied)
            spans.extend(new)

        # Priority 6: capitalized multi-word phrases
        new = self._extract_cap_phrase_spans(text, occupied)
        spans.extend(new)

        # Sort spans right-to-left so replacements don't shift earlier offsets.
        spans.sort(key=lambda t: t[0], reverse=True)

        term_placeholders: dict[str, str] = {}
        result = text
        for start, end, original in spans:
            placeholder = f"{_OPEN}TERM_{counter[0]:03d}{_CLOSE}"
            counter[0] += 1
            term_placeholders[placeholder] = original
            result = result[:start] + placeholder + result[end:]

        return ProtectedText(
            text=result,
            term_placeholders=term_placeholders,
            math_placeholders=math_placeholders,
        )

    def restore(
        self,
        text: str,
        term_placeholders: dict[str, str],
        math_placeholders: dict[str, str],
    ) -> str:
        """Replace all placeholders in text with their original values.

        Args:
            text: Text containing ⟨TERM_NNN⟩ and/or ⟨MATH_NNN⟩ placeholders.
            term_placeholders: Mapping of term placeholder keys to originals.
            math_placeholders: Mapping of math placeholder keys to originals.

        Returns:
            Fully restored text with no remaining placeholders.

        Raises:
            ValueError: If any placeholder remains in the text after all
                substitutions have been applied.
        """
        combined = {**term_placeholders, **math_placeholders}
        # Longest key first prevents partial-key matches (e.g. TERM_10 vs TERM_100).
        for key in sorted(combined, key=len, reverse=True):
            text = text.replace(key, combined[key])

        remaining = _PLACEHOLDER_RE.search(text)
        if remaining:
            raise ValueError(
                f"Placeholder {remaining.group(0)!r} was not restored. "
                "Ensure term_placeholders and math_placeholders are complete."
            )
        return text

    def validate(
        self,
        text: str,
        term_placeholders: dict[str, str],
        math_placeholders: dict[str, str],
    ) -> ValidationResult:
        """Verify every expected placeholder is present in text.

        Args:
            text: Text to inspect.
            term_placeholders: Expected term placeholders.
            math_placeholders: Expected math placeholders.

        Returns:
            ValidationResult indicating validity and any missing placeholders.
        """
        missing: list[str] = []
        for key in list(term_placeholders) + list(math_placeholders):
            if key not in text:
                missing.append(key)
        return ValidationResult(
            is_valid=len(missing) == 0,
            missing_placeholders=tuple(missing),
        )

    # ------------------------------------------------------------------
    # Private extractors
    # ------------------------------------------------------------------

    def _load_nlp(self) -> object:
        """Load the spaCy model, caching on self._nlp.

        Raises:
            RuntimeError: If spaCy or the model cannot be loaded.
        """
        if self._nlp is not None:
            return self._nlp
        try:
            import spacy  # type: ignore[import-untyped]  # spaCy has no bundled stubs

            self._nlp = spacy.load("en_core_web_sm")
        except Exception as exc:
            raise RuntimeError(
                "Could not load spaCy model 'en_core_web_sm'. "
                "Run: python -m spacy download en_core_web_sm"
            ) from exc
        return self._nlp

    def _extract_ner_spans(
        self, text: str, occupied: set[tuple[int, int]]
    ) -> list[tuple[int, int, str]]:
        """Extract named-entity spans via spaCy NER."""
        nlp = self._load_nlp()
        ner_labels = {"ORG", "PERSON", "GPE", "PRODUCT"}
        doc = nlp(text)  # type: ignore[operator]
        found: list[tuple[int, int, str]] = []
        for ent in doc.ents:
            if ent.label_ not in ner_labels:
                continue
            if len(ent.text) <= 1:
                continue
            start, end = ent.start_char, ent.end_char
            if _spans_overlap(occupied, start, end):
                continue
            found.append((start, end, ent.text))
            occupied.add((start, end))
        return found

    def _extract_custom_spans(
        self, text: str, occupied: set[tuple[int, int]]
    ) -> list[tuple[int, int, str]]:
        """Extract custom term spans using word-boundary matching."""
        found: list[tuple[int, int, str]] = []
        for term in self._config.custom_terms:
            pattern = r"\b" + re.escape(term) + r"\b"
            for m in re.finditer(pattern, text, re.IGNORECASE):
                start, end = m.start(), m.end()
                if _spans_overlap(occupied, start, end):
                    continue
                found.append((start, end, m.group(0)))
                occupied.add((start, end))
        return found

    def _extract_number_spans(
        self, text: str, occupied: set[tuple[int, int]]
    ) -> list[tuple[int, int, str]]:
        """Extract bare number spans."""
        found: list[tuple[int, int, str]] = []
        for m in _NUMBER_RE.finditer(text):
            start, end = m.start(), m.end()
            if _spans_overlap(occupied, start, end):
                continue
            found.append((start, end, m.group(0)))
            occupied.add((start, end))
        return found

    def _extract_cap_phrase_spans(
        self, text: str, occupied: set[tuple[int, int]]
    ) -> list[tuple[int, int, str]]:
        """Extract capitalized multi-word phrase spans."""
        found: list[tuple[int, int, str]] = []
        for m in _CAP_PHRASE_RE.finditer(text):
            start, end = m.start(), m.end()
            if _spans_overlap(occupied, start, end):
                continue
            found.append((start, end, m.group(0)))
            occupied.add((start, end))
        return found
