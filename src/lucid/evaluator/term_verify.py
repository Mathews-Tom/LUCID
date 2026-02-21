"""Cross-cutting term verification for the semantic evaluator.

Ensures paraphrases preserve all placeholders and numerical values
from the original text. Runs as the first check in the evaluation
pipeline — no model loading, regex only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Reuse the same placeholder pattern as term_protect.py
_PLACEHOLDER_RE = re.compile(r"\[(?:MATH|TERM)_(\d{3})\]")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?%?")


@dataclass(frozen=True, slots=True)
class TermVerificationResult:
    """Result of cross-cutting term verification.

    Args:
        passed: True when all placeholders and numbers are preserved.
        missing_placeholders: Placeholder tokens present in original but absent in paraphrase.
        mismatched_numbers: Numerical values present in original but absent in paraphrase.
        reason: Human-readable explanation when passed is False.
    """

    passed: bool
    missing_placeholders: tuple[str, ...]
    mismatched_numbers: tuple[str, ...]
    reason: str | None


class TermVerifier:
    """Verify placeholder and numerical consistency between original and paraphrase.

    This is a stateless, model-free checker that uses regex matching only.
    It runs in microseconds and catches corruption that would be invisible
    to embedding or NLI models.
    """

    def verify(self, original: str, paraphrase: str) -> TermVerificationResult:
        """Check that all placeholders and numbers from original exist in paraphrase.

        Args:
            original: Source text (may contain placeholders and numbers).
            paraphrase: Rewritten text to validate against original.

        Returns:
            TermVerificationResult with pass/fail status and details.
        """
        missing_placeholders = self._check_placeholders(original, paraphrase)
        mismatched_numbers = self._check_numbers(original, paraphrase)

        passed = len(missing_placeholders) == 0 and len(mismatched_numbers) == 0

        reason: str | None = None
        if not passed:
            parts: list[str] = []
            if missing_placeholders:
                parts.append(f"missing placeholders: {', '.join(missing_placeholders)}")
            if mismatched_numbers:
                parts.append(f"missing numbers: {', '.join(mismatched_numbers)}")
            reason = "; ".join(parts)

        return TermVerificationResult(
            passed=passed,
            missing_placeholders=tuple(missing_placeholders),
            mismatched_numbers=tuple(mismatched_numbers),
            reason=reason,
        )

    def _check_placeholders(self, original: str, paraphrase: str) -> list[str]:
        """Return placeholder tokens in original that are absent from paraphrase."""
        original_tokens = set(m.group(0) for m in _PLACEHOLDER_RE.finditer(original))
        paraphrase_tokens = set(m.group(0) for m in _PLACEHOLDER_RE.finditer(paraphrase))
        return sorted(original_tokens - paraphrase_tokens)

    def _check_numbers(self, original: str, paraphrase: str) -> list[str]:
        """Return numerical values in original that are absent from paraphrase."""
        original_numbers = set(_NUMBER_RE.findall(original))
        paraphrase_numbers = set(_NUMBER_RE.findall(paraphrase))
        return sorted(original_numbers - paraphrase_numbers)
