"""Tests for statistical feature extractor (Tier 2 detector)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lucid.detector import DetectorInitError
from lucid.detector.statistical import (
    MIN_WORDS_THRESHOLD,
    StatisticalDetector,
    _combine_features,
    _normalize_feature,
)


# ---------------------------------------------------------------------------
# Unit tests for pure helper functions
# ---------------------------------------------------------------------------


class TestNormalizeFeature:
    """Tests for _normalize_feature pure function."""

    def test_clamps_below_low(self) -> None:
        assert _normalize_feature(-10.0, 0.0, 100.0, invert=False) == 0.0

    def test_clamps_above_high(self) -> None:
        assert _normalize_feature(200.0, 0.0, 100.0, invert=False) == 1.0

    def test_midpoint(self) -> None:
        result = _normalize_feature(50.0, 0.0, 100.0, invert=False)
        assert abs(result - 0.5) < 1e-9

    def test_invert_low_maps_to_one(self) -> None:
        """With invert=True, the low bound maps to 1.0 (most AI-like)."""
        result = _normalize_feature(0.0, 0.0, 100.0, invert=True)
        assert abs(result - 1.0) < 1e-9

    def test_invert_high_maps_to_zero(self) -> None:
        result = _normalize_feature(100.0, 0.0, 100.0, invert=True)
        assert abs(result - 0.0) < 1e-9

    def test_equal_bounds_returns_half(self) -> None:
        result = _normalize_feature(5.0, 5.0, 5.0, invert=False)
        assert result == 0.5


class TestCombineFeatures:
    """Tests for _combine_features aggregation."""

    def test_all_none_returns_half(self) -> None:
        features: dict[str, Any] = {
            "perplexity_proxy": None,
            "burstiness": None,
            "ttr": None,
            "hapax_ratio": None,

            "sentence_length_variance": None,
            "pos_trigram_entropy": None,
        }
        assert _combine_features(features) == 0.5

    def test_result_in_unit_interval(self) -> None:
        features: dict[str, Any] = {
            "perplexity_proxy": 50.0,
            "burstiness": 0.5,
            "ttr": 0.5,
            "hapax_ratio": 0.3,

            "sentence_length_variance": 80.0,
            "pos_trigram_entropy": 3.5,
        }
        score = _combine_features(features)
        assert 0.0 <= score <= 1.0

    def test_very_low_perplexity_high_score(self) -> None:
        """Very low perplexity (AI-like) must contribute to a higher score."""
        features_ai: dict[str, Any] = {
            "perplexity_proxy": 10.0,  # at the AI-bound
            "burstiness": None,
            "ttr": None,
            "hapax_ratio": None,

            "sentence_length_variance": None,
            "pos_trigram_entropy": None,
        }
        features_human: dict[str, Any] = {
            "perplexity_proxy": 200.0,  # at the human-bound
            "burstiness": None,
            "ttr": None,
            "hapax_ratio": None,

            "sentence_length_variance": None,
            "pos_trigram_entropy": None,
        }
        assert _combine_features(features_ai) > _combine_features(features_human)


# ---------------------------------------------------------------------------
# Unit tests with mocked spaCy and sentence structures
# ---------------------------------------------------------------------------


def _make_mock_token(text: str, is_alpha: bool, pos_: str, is_space: bool = False) -> MagicMock:
    """Create a mock spaCy token."""
    token = MagicMock()
    token.text = text
    token.is_alpha = is_alpha
    token.pos_ = pos_
    token.is_space = is_space
    return token


def _make_mock_sent(words: list[str]) -> MagicMock:
    """Create a mock spaCy sentence span with alpha tokens only."""
    sent = MagicMock()
    tokens = [_make_mock_token(w, True, "NOUN") for w in words]
    sent.__iter__ = lambda self: iter(tokens)
    return sent


class TestComputeBurstiness:
    """Tests for StatisticalDetector._compute_burstiness."""

    def test_uniform_lengths_low_burstiness(self) -> None:
        """Identical sentence lengths must give burstiness = 0.0."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        result = detector._compute_burstiness([5, 5, 5, 5])
        assert result is not None
        assert abs(result) < 1e-9

    def test_variable_lengths_higher_burstiness(self) -> None:
        """Variable sentence lengths must give burstiness > 0."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        result = detector._compute_burstiness([3, 3, 15, 3, 3, 20])
        assert result is not None
        assert result > 0.0

    def test_below_min_sentences_returns_none(self) -> None:
        """Fewer than MIN_SENTENCES_THRESHOLD sentences must return None."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        assert detector._compute_burstiness([5, 5]) is None

    def test_zero_mean_returns_zero(self) -> None:
        """All-zero sentence lengths must return 0.0 (not divide-by-zero)."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        result = detector._compute_burstiness([0, 0, 0])
        assert result == 0.0

    def test_exact_min_sentences_threshold(self) -> None:
        """Exactly MIN_SENTENCES_THRESHOLD sentences must return a value."""
        from lucid.detector.statistical import MIN_SENTENCES_THRESHOLD
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        result = detector._compute_burstiness([5] * MIN_SENTENCES_THRESHOLD)
        assert result is not None


class TestComputeTTR:
    """Tests for StatisticalDetector._compute_ttr."""

    def test_all_same_words_low_ttr(self) -> None:
        """Repeated words must give TTR = 1/N (one unique)."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        result = detector._compute_ttr(["the"] * 8)
        assert result is not None
        assert abs(result - 1 / 8) < 1e-9

    def test_all_unique_words_max_ttr(self) -> None:
        """All-unique words must give TTR = 1.0."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        result = detector._compute_ttr(["apple", "banana", "cherry", "date"])
        assert result == 1.0

    def test_empty_words_returns_none(self) -> None:
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        assert detector._compute_ttr([]) is None

    def test_mixed_words(self) -> None:
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        # "the the the the" -> 1 unique / 4 total = 0.25
        result = detector._compute_ttr(["the", "the", "the", "the"])
        assert result is not None
        assert abs(result - 0.25) < 1e-9


class TestComputeHapaxRatio:
    """Tests for StatisticalDetector._compute_hapax_ratio."""

    def test_all_unique_hapax_ratio_one(self) -> None:
        """All unique words must give hapax ratio = 1.0."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        result = detector._compute_hapax_ratio(["apple", "banana", "cherry"])
        assert result == 1.0

    def test_all_repeated_hapax_ratio_zero(self) -> None:
        """No unique words (all repeated >= 2x) must give ratio = 0.0."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        result = detector._compute_hapax_ratio(["the", "the", "a", "a"])
        assert result == 0.0

    def test_empty_returns_none(self) -> None:
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        assert detector._compute_hapax_ratio([]) is None

    def test_partial_hapax(self) -> None:
        """Two of four words unique -> hapax_ratio = 0.5."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        # words: a, a, b, c  (b and c appear once each => 2/4 = 0.5)
        result = detector._compute_hapax_ratio(["a", "a", "b", "c"])
        assert result is not None
        assert abs(result - 0.5) < 1e-9


class TestComputeSentenceStats:
    """Tests for StatisticalDetector._compute_sentence_stats."""

    def test_empty_returns_none_pair(self) -> None:
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        mean, var = detector._compute_sentence_stats([])
        assert mean is None
        assert var is None

    def test_single_length(self) -> None:
        """Single sentence length must give mean=length, variance=0.0."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        mean, var = detector._compute_sentence_stats([10])
        assert mean == 10.0
        assert var == 0.0

    def test_known_values(self) -> None:
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        lengths = [4, 8, 4, 8]
        mean, var = detector._compute_sentence_stats(lengths)
        assert mean is not None
        assert var is not None
        assert abs(mean - 6.0) < 1e-9
        assert abs(var - 4.0) < 1e-9  # population variance


class TestScoreMinWordsThreshold:
    """Tests for the MIN_WORDS_THRESHOLD guard in score()."""

    def test_below_threshold_returns_none(self) -> None:
        """Text with fewer than MIN_WORDS_THRESHOLD words must return None."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        short_text = " ".join(["word"] * (MIN_WORDS_THRESHOLD - 1))
        assert len(short_text.split()) == MIN_WORDS_THRESHOLD - 1
        result = detector.score(short_text)
        assert result is None

    def test_at_threshold_calls_extract_features(self) -> None:
        """Text with exactly MIN_WORDS_THRESHOLD words must not return None."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)

        # Mock extract_features to avoid needing spaCy
        all_none: dict[str, Any] = {
            "perplexity_proxy": None,
            "burstiness": None,
            "ttr": None,
            "hapax_ratio": None,

            "sentence_length_variance": None,
            "pos_trigram_entropy": None,
        }
        with patch.object(detector, "extract_features", return_value=all_none):
            at_threshold_text = " ".join(["word"] * MIN_WORDS_THRESHOLD)
            result = detector.score(at_threshold_text)
            # extract_features called => result is float (may be 0.5 from all-None)
            assert result is not None
            assert isinstance(result, float)

    def test_49_words_returns_none(self) -> None:
        """49 words (one below threshold of 50) must return None."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        text = " ".join([f"word{i}" for i in range(49)])
        assert detector.score(text) is None

    def test_50_words_does_not_return_none(self) -> None:
        """50 words must not return None (may still be 0.5 from all-None features)."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        all_none: dict[str, Any] = {k: None for k in [
            "perplexity_proxy", "burstiness", "ttr", "hapax_ratio",
            "sentence_length_variance", "pos_trigram_entropy"
        ]}
        text = " ".join([f"word{i}" for i in range(50)])
        with patch.object(detector, "extract_features", return_value=all_none):
            result = detector.score(text)
            assert result is not None


class TestExtractFeaturesAllKeys:
    """Tests for extract_features return structure."""

    EXPECTED_KEYS = frozenset({
        "perplexity_proxy",
        "burstiness",
        "ttr",
        "hapax_ratio",
        "sentence_length_variance",
        "pos_trigram_entropy",
    })

    def test_all_seven_keys_present(self) -> None:
        """extract_features must return a dict with all 7 expected keys."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)

        # Fake a minimal spaCy pipeline
        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.has_annotation.return_value = False  # no sents

        # Alpha tokens
        tokens = [
            _make_mock_token("The", True, "DET"),
            _make_mock_token("cat", True, "NOUN"),
            _make_mock_token("sat", True, "VERB"),
        ]
        mock_doc.__iter__ = lambda self: iter(tokens)
        mock_nlp.return_value = mock_doc
        detector._nlp = mock_nlp  # type: ignore[assignment]

        result = detector.extract_features("The cat sat.")
        assert set(result.keys()) == self.EXPECTED_KEYS

    def test_values_are_float_or_none(self) -> None:
        """All feature values must be float or None."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)

        mock_nlp = MagicMock()
        mock_doc = MagicMock()
        mock_doc.has_annotation.return_value = False
        tokens: list[MagicMock] = []
        mock_doc.__iter__ = lambda self: iter(tokens)
        mock_nlp.return_value = mock_doc
        detector._nlp = mock_nlp  # type: ignore[assignment]

        result = detector.extract_features("short")
        for key, value in result.items():
            assert value is None or isinstance(value, float), (
                f"Feature {key!r} must be float | None, got {type(value)}"
            )


class TestFallbackScoringWithoutXGBoost:
    """Tests that scoring works without XGBoost via feature averaging."""

    def test_score_in_unit_interval(self) -> None:
        """score() must always return a float in [0.0, 1.0]."""
        features: dict[str, Any] = {
            "perplexity_proxy": 30.0,
            "burstiness": 0.4,
            "ttr": 0.6,
            "hapax_ratio": 0.35,

            "sentence_length_variance": 50.0,
            "pos_trigram_entropy": 3.0,
        }
        score = _combine_features(features)
        assert 0.0 <= score <= 1.0

    def test_partial_features_score_is_mean_of_available(self) -> None:
        """When some features are None, score averages only available ones."""
        features_partial: dict[str, Any] = {
            "perplexity_proxy": None,
            "burstiness": None,
            "ttr": 0.5,  # mid bound: ttr bounds are (0.2, 0.9), invert=True
            "hapax_ratio": None,

            "sentence_length_variance": None,
            "pos_trigram_entropy": None,
        }
        score = _combine_features(features_partial)
        # ttr=0.5 normalized in [0.2, 0.9] inverted:
        # norm = (0.5-0.2)/(0.9-0.2) = 0.3/0.7 ~= 0.4286
        # inverted = 1 - 0.4286 ~= 0.5714
        expected = 1.0 - (0.5 - 0.2) / (0.9 - 0.2)
        assert abs(score - expected) < 1e-4


@pytest.mark.integration
class TestStatisticalDetectorIntegration:
    """Integration tests requiring spaCy (en_core_web_sm)."""

    def test_extract_features_on_real_text(self) -> None:
        """extract_features must run without errors on real English text."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        text = (
            "The scientist carefully examined the data. "
            "Results were consistent with prior findings. "
            "Further analysis revealed unexpected patterns. "
        )
        result = detector.extract_features(text)
        assert "ttr" in result
        assert "hapax_ratio" in result
        # With real spaCy these should not all be None
        non_none = [v for v in result.values() if v is not None]
        assert len(non_none) >= 3

    def test_score_below_threshold_returns_none(self) -> None:
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        short = "This is a very short text."
        result = detector.score(short)
        assert result is None

    def test_score_above_threshold_returns_float(self) -> None:
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        long_text = " ".join(["The analysis showed consistent results."] * 20)
        result = detector.score(long_text)
        assert result is not None
        assert 0.0 <= result <= 1.0

    def test_pos_trigram_entropy_non_negative(self) -> None:
        """POS trigram entropy must be a non-negative float for valid text."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        detector._ensure_spacy()
        result = detector._compute_pos_trigram_entropy(
            "The quick brown fox jumps over the lazy dog regularly."
        )
        assert result is not None
        assert result >= 0.0

    def test_ttr_on_repetitive_text_lower_than_diverse(self) -> None:
        """Repetitive text must have lower TTR than diverse vocabulary."""
        detector = StatisticalDetector(use_gpt2_perplexity=False)
        detector._ensure_spacy()

        # Run extract_features on repetitive text
        repetitive = "The the the the the the cat cat cat cat cat."
        diverse = "The quick brown fox jumps over the lazy beautiful dog."

        rep_feats = detector.extract_features(repetitive)
        div_feats = detector.extract_features(diverse)

        rep_ttr = rep_feats["ttr"]
        div_ttr = div_feats["ttr"]

        if rep_ttr is not None and div_ttr is not None:
            assert rep_ttr < div_ttr
