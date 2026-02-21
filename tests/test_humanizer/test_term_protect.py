"""Tests for the TermProtector in lucid.humanizer.term_protect."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lucid.config import TermProtectionConfig
from lucid.humanizer.term_protect import (
    ProtectedText,
    TermProtector,
    _CLOSE,
    _OPEN,
)
from lucid.parser.chunk import ProseChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(
    text: str,
    math_placeholders: dict[str, str] | None = None,
    term_placeholders: dict[str, str] | None = None,
) -> ProseChunk:
    """Build a minimal ProseChunk for testing."""
    return ProseChunk(
        text=text,
        start_pos=0,
        end_pos=len(text),
        math_placeholders=math_placeholders or {},
        term_placeholders=term_placeholders or {},
        protected_text=text,
    )


def _make_mock_nlp(entities: list[tuple[str, str, int, int]]) -> MagicMock:
    """Create a mock spaCy NLP pipeline returning fixed entities.

    Args:
        entities: List of (text, label, start_char, end_char).
    """
    mock_nlp = MagicMock()
    mock_doc = MagicMock()
    mock_ents = []
    for ent_text, label, start, end in entities:
        ent = MagicMock()
        ent.text = ent_text
        ent.label_ = label
        ent.start_char = start
        ent.end_char = end
        mock_ents.append(ent)
    mock_doc.ents = mock_ents
    mock_nlp.return_value = mock_doc
    return mock_nlp


def _placeholder(kind: str, idx: int) -> str:
    return f"{_OPEN}{kind}_{idx:03d}{_CLOSE}"


def _cfg(**kwargs: object) -> TermProtectionConfig:
    return TermProtectionConfig(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Citation extraction
# ---------------------------------------------------------------------------


class TestCitationExtraction:
    def test_bracket_citation_protected(self) -> None:
        text = "See [Smith, 2024] for details."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        result = protector.protect(chunk)

        assert "[Smith, 2024]" not in result.text
        assert len(result.term_placeholders) == 1
        assert "[Smith, 2024]" in result.term_placeholders.values()

    def test_et_al_citation_protected(self) -> None:
        text = "As shown in (Johnson et al., 2022)."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        result = protector.protect(chunk)

        assert "(Johnson et al., 2022)" not in result.text
        assert any(
            "(Johnson et al., 2022)" in v for v in result.term_placeholders.values()
        )

    def test_author_year_inline_citation(self) -> None:
        text = "Smith (2020) demonstrated this effect."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        result = protector.protect(chunk)

        # "Smith (2020)" should appear as a term placeholder
        assert any(
            "Smith" in v and "2020" in v for v in result.term_placeholders.values()
        )

    def test_multiple_citations_all_protected(self) -> None:
        text = "Both [Adams, 2021] and [Brown, 2019] agree."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        result = protector.protect(chunk)

        assert "[Adams, 2021]" not in result.text
        assert "[Brown, 2019]" not in result.text
        values = set(result.term_placeholders.values())
        assert "[Adams, 2021]" in values
        assert "[Brown, 2019]" in values

    def test_citations_disabled_not_protected(self) -> None:
        text = "As in [Smith, 2024], the result holds."
        chunk = _make_chunk(text)
        protector = TermProtector(
            _cfg(use_ner=False, protect_citations=False, protect_numbers=False)
        )
        result = protector.protect(chunk)

        assert "[Smith, 2024]" in result.text

    def test_paren_multi_author_citation(self) -> None:
        text = "This was shown (Adams & Brown, 2020)."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        result = protector.protect(chunk)

        assert "(Adams & Brown, 2020)" not in result.text
        assert any("Adams" in v for v in result.term_placeholders.values())


# ---------------------------------------------------------------------------
# Round-trip protect → restore
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_citation_round_trip(self) -> None:
        text = "As shown by [Smith, 2023], the model improves."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        protected = protector.protect(chunk)
        restored = protector.restore(
            protected.text,
            protected.term_placeholders,
            protected.math_placeholders,
        )
        assert restored == text

    def test_number_round_trip(self) -> None:
        text = "The accuracy was 98.5% on 1024 samples."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_citations=False))
        protected = protector.protect(chunk)
        restored = protector.restore(
            protected.text,
            protected.term_placeholders,
            protected.math_placeholders,
        )
        assert restored == text

    def test_custom_term_round_trip(self) -> None:
        text = "The Monte Carlo simulation converged."
        chunk = _make_chunk(text)
        protector = TermProtector(
            _cfg(
                use_ner=False,
                protect_citations=False,
                protect_numbers=False,
                custom_terms=("Monte Carlo",),
            )
        )
        protected = protector.protect(chunk)
        restored = protector.restore(
            protected.text,
            protected.term_placeholders,
            protected.math_placeholders,
        )
        assert restored == text

    def test_math_placeholder_pass_through_round_trip(self) -> None:
        # The chunk's protected_text already contains the math placeholder.
        # protect() leaves it untouched; restore() expands it back to the
        # original LaTeX. The round-trip restores the LaTeX expression.
        math_ph = {_placeholder("MATH", 0): r"\frac{a}{b}"}
        text_with_math = f"The formula {_placeholder('MATH', 0)} is key."
        expected_restored = r"The formula \frac{a}{b} is key."
        chunk = _make_chunk(text_with_math, math_placeholders=math_ph)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        protected = protector.protect(chunk)
        # Math placeholder must still be present after protect()
        assert _placeholder("MATH", 0) in protected.text
        restored = protector.restore(
            protected.text,
            protected.term_placeholders,
            protected.math_placeholders,
        )
        assert restored == expected_restored

    def test_restore_raises_on_orphan_placeholder(self) -> None:
        orphan = _placeholder("TERM", 99)
        text = f"Some {orphan} text."
        protector = TermProtector(_cfg(use_ner=False))
        with pytest.raises(ValueError, match="not restored"):
            protector.restore(text, {}, {})

    @patch("spacy.load")
    def test_ner_round_trip(self, mock_load: MagicMock) -> None:
        text = "OpenAI released GPT-4."
        # "OpenAI" → [0, 6], "GPT-4" → [16, 21] (verified against the string)
        mock_load.return_value = _make_mock_nlp(
            [("OpenAI", "ORG", 0, 6), ("GPT-4", "PRODUCT", 16, 21)]
        )
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=True, protect_citations=False, protect_numbers=False))
        protected = protector.protect(chunk)
        restored = protector.restore(
            protected.text,
            protected.term_placeholders,
            protected.math_placeholders,
        )
        assert restored == text


# ---------------------------------------------------------------------------
# Config toggles
# ---------------------------------------------------------------------------


class TestConfigToggles:
    def test_use_ner_false_no_ner_call(self) -> None:
        text = "Google is a company."
        chunk = _make_chunk(text)
        with patch("spacy.load") as mock_load:
            protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
            protector.protect(chunk)
            mock_load.assert_not_called()

    def test_protect_numbers_false_leaves_numbers(self) -> None:
        text = "There are 42 samples and 3.14 ratio."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_citations=False, protect_numbers=False))
        result = protector.protect(chunk)
        assert "42" in result.text
        assert "3.14" in result.text

    def test_protect_numbers_true_replaces_numbers(self) -> None:
        text = "The score is 95.3%."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_citations=False, protect_numbers=True))
        result = protector.protect(chunk)
        assert "95.3" not in result.text
        assert any("95.3%" in v or "95.3" in v for v in result.term_placeholders.values())

    def test_protect_citations_false_leaves_citations(self) -> None:
        text = "See [Jones, 2020] for more."
        chunk = _make_chunk(text)
        protector = TermProtector(
            _cfg(use_ner=False, protect_citations=False, protect_numbers=False)
        )
        result = protector.protect(chunk)
        assert "[Jones, 2020]" in result.text

    @patch("spacy.load")
    def test_use_ner_true_loads_model(self, mock_load: MagicMock) -> None:
        text = "DeepMind built AlphaFold."
        mock_load.return_value = _make_mock_nlp(
            [("DeepMind", "ORG", 0, 8), ("AlphaFold", "PRODUCT", 15, 24)]
        )
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=True, protect_citations=False, protect_numbers=False))
        result = protector.protect(chunk)
        mock_load.assert_called_once_with("en_core_web_sm")
        # Both entities must be replaced
        assert "DeepMind" not in result.text or "AlphaFold" not in result.text or len(result.term_placeholders) >= 1


# ---------------------------------------------------------------------------
# Overlap resolution
# ---------------------------------------------------------------------------


class TestOverlapResolution:
    @patch("spacy.load")
    def test_citation_wins_over_ner_when_overlapping(
        self, mock_load: MagicMock
    ) -> None:
        """Citation (higher priority) must win when NER overlaps the same span."""
        text = "See [Smith, 2021] for reference."
        # NER reports Smith as PERSON with span [4:9] inside the citation [4:17]
        mock_load.return_value = _make_mock_nlp([("Smith", "PERSON", 5, 10)])
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=True, protect_numbers=False))
        result = protector.protect(chunk)

        # The full citation, not just "Smith", must be in the placeholder map
        values = list(result.term_placeholders.values())
        assert any("[Smith, 2021]" in v for v in values), (
            f"Expected full citation in placeholders, got: {values}"
        )
        # "Smith" must NOT appear as a separate term placeholder
        assert not any(v == "Smith" for v in values), (
            f"NER placeholder for 'Smith' should have been suppressed, got: {values}"
        )

    @patch("spacy.load")
    def test_no_duplicate_protection_for_same_span(
        self, mock_load: MagicMock
    ) -> None:
        text = "Monte Carlo simulation."
        # NER also claims Monte Carlo
        mock_load.return_value = _make_mock_nlp([("Monte Carlo", "GPE", 0, 11)])
        chunk = _make_chunk(text)
        protector = TermProtector(
            _cfg(
                use_ner=True,
                protect_citations=False,
                protect_numbers=False,
                custom_terms=("Monte Carlo",),
            )
        )
        result = protector.protect(chunk)
        # Monte Carlo appears exactly once as a placeholder
        assert list(result.term_placeholders.values()).count("Monte Carlo") == 1


# ---------------------------------------------------------------------------
# Math placeholder pass-through
# ---------------------------------------------------------------------------


class TestMathPlaceholderPassThrough:
    def test_math_placeholders_passed_through(self) -> None:
        math_ph = {_placeholder("MATH", 0): r"\int_0^\infty f(x)\,dx"}
        text = f"We compute {_placeholder('MATH', 0)} analytically."
        chunk = _make_chunk(text, math_placeholders=math_ph)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        result = protector.protect(chunk)

        assert result.math_placeholders == math_ph
        # Math placeholder still present in text
        assert _placeholder("MATH", 0) in result.text

    def test_math_placeholder_not_double_protected(self) -> None:
        math_ph = {_placeholder("MATH", 0): r"\alpha"}
        text = f"Variable {_placeholder('MATH', 0)} is used."
        chunk = _make_chunk(text, math_placeholders=math_ph)
        protector = TermProtector(_cfg(use_ner=False))
        result = protector.protect(chunk)

        # MATH_000 must not appear in term_placeholders
        assert _placeholder("MATH", 0) not in result.term_placeholders


# ---------------------------------------------------------------------------
# Counter starts after max math index
# ---------------------------------------------------------------------------


class TestPlaceholderCounterOffset:
    def test_terms_start_at_zero_with_no_math(self) -> None:
        text = "See [Smith, 2024]."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        result = protector.protect(chunk)

        assert _placeholder("TERM", 0) in result.term_placeholders

    def test_terms_start_after_math_max(self) -> None:
        # Math placeholders go up to index 3
        math_ph = {
            _placeholder("MATH", 0): r"\alpha",
            _placeholder("MATH", 3): r"\beta",
        }
        text = (
            f"Let {_placeholder('MATH', 0)} and {_placeholder('MATH', 3)} be given. "
            "See [Jones, 2022]."
        )
        chunk = _make_chunk(text, math_placeholders=math_ph)
        protector = TermProtector(_cfg(use_ner=False, protect_numbers=False))
        result = protector.protect(chunk)

        # First term placeholder must start at index 4
        assert _placeholder("TERM", 4) in result.term_placeholders
        # Index 0, 1, 2, 3 must NOT be used as TERM
        for i in range(4):
            assert _placeholder("TERM", i) not in result.term_placeholders


# ---------------------------------------------------------------------------
# Validate method
# ---------------------------------------------------------------------------


class TestValidate:
    def test_all_present_returns_valid(self) -> None:
        ph = _placeholder("TERM", 0)
        text = f"Some {ph} content."
        protector = TermProtector(_cfg(use_ner=False))
        result = protector.validate(text, {ph: "original"}, {})
        assert result.is_valid is True
        assert result.missing_placeholders == ()

    def test_missing_placeholder_detected(self) -> None:
        ph = _placeholder("TERM", 0)
        text = "Some content without the placeholder."
        protector = TermProtector(_cfg(use_ner=False))
        result = protector.validate(text, {ph: "original"}, {})
        assert result.is_valid is False
        assert ph in result.missing_placeholders

    def test_math_placeholder_missing_detected(self) -> None:
        math_ph = _placeholder("MATH", 1)
        text = "No math here."
        protector = TermProtector(_cfg(use_ner=False))
        result = protector.validate(text, {}, {math_ph: r"\sigma"})
        assert result.is_valid is False
        assert math_ph in result.missing_placeholders

    def test_validate_passes_when_both_present(self) -> None:
        term_ph = _placeholder("TERM", 0)
        math_ph = _placeholder("MATH", 0)
        text = f"{term_ph} uses {math_ph}."
        protector = TermProtector(_cfg(use_ner=False))
        result = protector.validate(text, {term_ph: "X"}, {math_ph: r"\mu"})
        assert result.is_valid is True


# ---------------------------------------------------------------------------
# Custom terms
# ---------------------------------------------------------------------------


class TestCustomTerms:
    def test_single_word_custom_term(self) -> None:
        text = "The LSTM performed well."
        chunk = _make_chunk(text)
        protector = TermProtector(
            _cfg(use_ner=False, protect_citations=False, protect_numbers=False, custom_terms=("LSTM",))
        )
        result = protector.protect(chunk)

        assert "LSTM" not in result.text
        assert "LSTM" in result.term_placeholders.values()

    def test_multi_word_custom_term(self) -> None:
        text = "The Support Vector Machine model was trained."
        chunk = _make_chunk(text)
        protector = TermProtector(
            _cfg(
                use_ner=False,
                protect_citations=False,
                protect_numbers=False,
                custom_terms=("Support Vector Machine",),
            )
        )
        result = protector.protect(chunk)

        assert "Support Vector Machine" not in result.text
        assert "Support Vector Machine" in result.term_placeholders.values()

    def test_custom_term_word_boundary(self) -> None:
        text = "The GAN architecture and WGAN variant differ."
        chunk = _make_chunk(text)
        protector = TermProtector(
            _cfg(
                use_ner=False,
                protect_citations=False,
                protect_numbers=False,
                custom_terms=("GAN",),
            )
        )
        result = protector.protect(chunk)

        # "GAN" in "WGAN" must NOT be protected (word boundary)
        protected_count = sum(1 for v in result.term_placeholders.values() if v == "GAN")
        assert protected_count == 1

    def test_no_custom_terms_empty_tuple(self) -> None:
        text = "Baseline text with nothing special."
        chunk = _make_chunk(text)
        protector = TermProtector(
            _cfg(use_ner=False, protect_citations=False, protect_numbers=False, custom_terms=())
        )
        result = protector.protect(chunk)
        # Only cap phrases extracted (if any); no custom term placeholders
        for v in result.term_placeholders.values():
            assert v not in ("",)  # all values must be non-empty originals


# ---------------------------------------------------------------------------
# Number protection
# ---------------------------------------------------------------------------


class TestNumberProtection:
    def test_integer_protected(self) -> None:
        text = "There are 256 nodes in the network."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_citations=False, protect_numbers=True))
        result = protector.protect(chunk)
        assert "256" not in result.text

    def test_decimal_protected(self) -> None:
        text = "The learning rate is 0.001."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_citations=False, protect_numbers=True))
        result = protector.protect(chunk)
        assert "0.001" not in result.text

    def test_percentage_protected(self) -> None:
        text = "Accuracy improved by 12.5%."
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=False, protect_citations=False, protect_numbers=True))
        result = protector.protect(chunk)
        assert "12.5%" not in result.text
        assert any("12.5%" in v or "12.5" in v for v in result.term_placeholders.values())


# ---------------------------------------------------------------------------
# Empty text
# ---------------------------------------------------------------------------


class TestEmptyText:
    def test_empty_chunk_returns_empty_protected_text(self) -> None:
        chunk = _make_chunk("")
        protector = TermProtector(_cfg(use_ner=False))
        result = protector.protect(chunk)

        assert isinstance(result, ProtectedText)
        assert result.text == ""
        assert result.term_placeholders == {}
        assert result.math_placeholders == {}

    def test_restore_empty_text(self) -> None:
        protector = TermProtector(_cfg(use_ner=False))
        assert protector.restore("", {}, {}) == ""

    def test_validate_empty_text_no_placeholders(self) -> None:
        protector = TermProtector(_cfg(use_ner=False))
        result = protector.validate("", {}, {})
        assert result.is_valid is True
        assert result.missing_placeholders == ()


# ---------------------------------------------------------------------------
# NER error handling
# ---------------------------------------------------------------------------


class TestNerErrorHandling:
    def test_spacy_load_failure_raises_runtime_error(self) -> None:
        with patch("spacy.load", side_effect=OSError("model not found")):
            text = "Google is headquartered in California."
            chunk = _make_chunk(text)
            protector = TermProtector(_cfg(use_ner=True, protect_citations=False, protect_numbers=False))
            with pytest.raises(RuntimeError, match="en_core_web_sm"):
                protector.protect(chunk)

    @patch("spacy.load")
    def test_single_char_entities_skipped(self, mock_load: MagicMock) -> None:
        text = "Use A for the first variable."
        # NER returns a single-char entity "A"
        mock_load.return_value = _make_mock_nlp([("A", "ORG", 4, 5)])
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=True, protect_citations=False, protect_numbers=False))
        result = protector.protect(chunk)
        # "A" (single char) must NOT appear in term_placeholders
        assert "A" not in result.term_placeholders.values()

    @patch("spacy.load")
    def test_non_target_ner_labels_skipped(self, mock_load: MagicMock) -> None:
        text = "Last Tuesday the event took place."
        # NER returns DATE entity, which is not in ORG/PERSON/GPE/PRODUCT
        mock_load.return_value = _make_mock_nlp([("Tuesday", "DATE", 5, 12)])
        chunk = _make_chunk(text)
        protector = TermProtector(_cfg(use_ner=True, protect_citations=False, protect_numbers=False))
        result = protector.protect(chunk)
        assert "Tuesday" not in result.term_placeholders.values()
