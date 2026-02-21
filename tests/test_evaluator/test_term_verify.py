"""Tests for the cross-cutting term verifier."""

from __future__ import annotations

from lucid.evaluator.term_verify import TermVerifier


class TestTermVerifier:
    """Unit tests for TermVerifier — no model loading, all regex-based."""

    def setup_method(self) -> None:
        self.verifier = TermVerifier()

    # -- Placeholder tests --------------------------------------------------

    def test_all_placeholders_present_passes(self) -> None:
        """All placeholders preserved in paraphrase → pass."""
        original = "The [TERM_000] method achieves [MATH_001] accuracy."
        paraphrase = "The [TERM_000] approach reaches [MATH_001] accuracy."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is True
        assert result.missing_placeholders == ()
        assert result.mismatched_numbers == ()
        assert result.reason is None

    def test_missing_placeholder_fails(self) -> None:
        """Dropped placeholder in paraphrase → fail."""
        original = "Use [TERM_000] and [TERM_001] for analysis."
        paraphrase = "Use [TERM_000] for analysis."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is False
        assert "[TERM_001]" in result.missing_placeholders
        assert result.reason is not None
        assert "missing placeholders" in result.reason

    def test_missing_math_placeholder_fails(self) -> None:
        """Dropped MATH placeholder → fail."""
        original = "Score is [MATH_000] on [MATH_001] samples."
        paraphrase = "Score is [MATH_000] on many samples."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is False
        assert "[MATH_001]" in result.missing_placeholders

    # -- Number tests -------------------------------------------------------

    def test_numbers_preserved_passes(self) -> None:
        """All numbers preserved → pass."""
        original = "We tested 100 samples with 95.5% accuracy."
        paraphrase = "We evaluated 100 samples achieving 95.5% accuracy."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is True
        assert result.mismatched_numbers == ()

    def test_missing_number_fails(self) -> None:
        """Number dropped from paraphrase → fail."""
        original = "The model processed 500 items in 3.2 seconds."
        paraphrase = "The model processed items in 3.2 seconds."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is False
        assert "500" in result.mismatched_numbers
        assert "missing numbers" in (result.reason or "")

    def test_extra_numbers_in_paraphrase_ok(self) -> None:
        """Extra numbers in paraphrase are acceptable."""
        original = "Used 10 layers."
        paraphrase = "Used 10 layers with 256 hidden units."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is True

    # -- Combined tests -----------------------------------------------------

    def test_combined_placeholder_and_number_failure(self) -> None:
        """Both placeholder and number missing → combined failure."""
        original = "The [TERM_000] scored 92% on 50 samples."
        paraphrase = "The approach scored well on samples."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is False
        assert len(result.missing_placeholders) == 1
        assert len(result.mismatched_numbers) >= 1
        assert "missing placeholders" in (result.reason or "")
        assert "missing numbers" in (result.reason or "")

    def test_no_placeholders_no_numbers(self) -> None:
        """Plain text with no placeholders or numbers → pass."""
        original = "This is a simple sentence about machine learning."
        paraphrase = "This sentence discusses machine learning concepts."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is True

    def test_percentage_preserved(self) -> None:
        """Percentage values must be preserved."""
        original = "Achieved 98.7% precision and 95% recall."
        paraphrase = "Obtained 98.7% precision and 95% recall."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is True

    def test_percentage_dropped_fails(self) -> None:
        """Dropped percentage → fail."""
        original = "Improved by 12.5% over baseline."
        paraphrase = "Improved significantly over baseline."
        result = self.verifier.verify(original, paraphrase)
        assert result.passed is False
        assert "12.5%" in result.mismatched_numbers
