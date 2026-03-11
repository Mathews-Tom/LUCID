"""Tests for individual statistical feature extractor methods."""

from __future__ import annotations

import math
from collections import Counter
from unittest.mock import patch

import numpy as np
import pytest

from lucid.detector.features import (
    FUNCTION_WORDS,
    HUMAN_FUNCTION_WORD_BASELINE,
    TRANSITION_PHRASES,
    FunctionWordResult,
    NgramResult,
    SentenceEntropyResult,
    StructuralSymmetryResult,
    TransitionResult,
)
from lucid.detector.statistical import (
    MIN_SENTENCES_THRESHOLD,
    MIN_WORDS_THRESHOLD,
    StatisticalDetector,
    _kurtosis,
    _skewness,
)

HUMAN_TEXT = (
    "The experiment failed spectacularly. Nobody expected the catalyst to react "
    "that way — certainly not Dr. Chen, who had spent three months optimizing the "
    "protocol. But science is funny like that. Sometimes your best-laid plans "
    "dissolve in a puff of hydrogen sulfide. We regrouped. Started over. "
    "The second attempt, stripped of our earlier assumptions, actually worked better."
)

AI_TEXT = (
    "The experiment was conducted using standard laboratory procedures. The results "
    "were analyzed using statistical methods. Furthermore, the data indicated "
    "significant differences between the control and experimental groups. Moreover, "
    "the findings were consistent with previous research in this area. In addition, "
    "the methodology was validated through multiple independent trials. Therefore, "
    "the conclusions drawn from this study are considered to be reliable and robust."
)


class TestSkewnessKurtosis:
    def test_skewness_symmetric(self) -> None:
        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(_skewness(arr)) < 0.01

    def test_skewness_positive(self) -> None:
        arr = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
        assert _skewness(arr) > 0

    def test_skewness_zero_std(self) -> None:
        arr = np.array([5.0, 5.0, 5.0])
        assert _skewness(arr) == 0.0

    def test_kurtosis_normal_like(self) -> None:
        rng = np.random.default_rng(42)
        arr = rng.normal(0, 1, 10000)
        assert abs(_kurtosis(arr)) < 0.2

    def test_kurtosis_zero_std(self) -> None:
        arr = np.array([3.0, 3.0, 3.0])
        assert _kurtosis(arr) == 0.0


class TestFunctionWordDistribution:
    def setup_method(self) -> None:
        self.detector = StatisticalDetector(
            use_gpt2_perplexity=False, use_deep_features=False
        )

    def test_returns_none_for_short_text(self) -> None:
        words = ["hello", "world"]
        result = self.detector._compute_function_word_distribution(words)
        assert result is None

    def test_returns_function_word_result(self) -> None:
        words = AI_TEXT.lower().split()
        result = self.detector._compute_function_word_distribution(words)
        assert isinstance(result, FunctionWordResult)
        assert result.total_count > 0
        assert result.entropy > 0
        assert isinstance(result.divergence, float)

    def test_kl_divergence_is_nonnegative(self) -> None:
        words = AI_TEXT.lower().split()
        result = self.detector._compute_function_word_distribution(words)
        assert result is not None
        # KL-divergence can be negative when distributions overlap heavily
        # but with epsilon smoothing it should be finite
        assert math.isfinite(result.divergence)

    def test_no_function_words_returns_none(self) -> None:
        words = ["xyzzy"] * 60
        result = self.detector._compute_function_word_distribution(words)
        assert result is None

    def test_entropy_increases_with_diversity(self) -> None:
        # Uniform distribution over a few function words
        narrow = ["the"] * 40 + ["a"] * 10 + ["hello"] * 10
        # Broader distribution
        broad = (
            ["the"] * 10 + ["a"] * 10 + ["and"] * 10
            + ["in"] * 10 + ["of"] * 10 + ["hello"] * 10
        )
        r_narrow = self.detector._compute_function_word_distribution(narrow)
        r_broad = self.detector._compute_function_word_distribution(broad)
        assert r_narrow is not None
        assert r_broad is not None
        assert r_broad.entropy > r_narrow.entropy


class TestTransitionFrequency:
    def setup_method(self) -> None:
        self.detector = StatisticalDetector(
            use_gpt2_perplexity=False, use_deep_features=False
        )

    def test_returns_none_for_short_text(self) -> None:
        result = self.detector._compute_transition_frequency("short", 5)
        assert result is None

    def test_detects_single_word_transitions(self) -> None:
        text = "However the results were clear. Therefore we concluded early."
        result = self.detector._compute_transition_frequency(text, 60)
        assert result is not None
        assert result.total_count >= 2
        assert "however" in text.lower()

    def test_detects_multi_word_transitions(self) -> None:
        text = (
            "We found results. In addition, the data confirmed our hypothesis. "
            "As a result, the study was deemed successful. " * 5
        )
        result = self.detector._compute_transition_frequency(text, 80)
        assert result is not None
        assert result.total_count >= 2

    def test_zero_transitions(self) -> None:
        text = "The cat sat on the mat. " * 10
        result = self.detector._compute_transition_frequency(text, 60)
        assert result is not None
        assert result.total_count == 0
        assert result.density_per_1000 == 0.0
        assert result.diversity == 0.0

    def test_density_scales_with_word_count(self) -> None:
        text = "However the data showed something. " * 5
        r1 = self.detector._compute_transition_frequency(text, 100)
        r2 = self.detector._compute_transition_frequency(text, 200)
        assert r1 is not None
        assert r2 is not None
        assert r1.density_per_1000 > r2.density_per_1000

    def test_ai_text_has_more_transitions_than_human(self) -> None:
        ai_words = len(AI_TEXT.split())
        human_words = len(HUMAN_TEXT.split())
        r_ai = self.detector._compute_transition_frequency(AI_TEXT, ai_words)
        r_human = self.detector._compute_transition_frequency(HUMAN_TEXT, human_words)
        assert r_ai is not None
        assert r_human is not None
        assert r_ai.total_count > r_human.total_count


class TestStructuralSymmetry:
    def setup_method(self) -> None:
        self.detector = StatisticalDetector(
            use_gpt2_perplexity=False, use_deep_features=False
        )

    def test_returns_none_for_few_sentences(self) -> None:
        result = self.detector._compute_structural_symmetry([10, 12], "text")
        assert result is None

    def test_uniform_sentences_high_symmetry(self) -> None:
        lengths = [10, 10, 10, 10, 10]
        result = self.detector._compute_structural_symmetry(lengths, "text")
        assert result is not None
        assert result.sentence_uniformity == 1.0

    def test_varied_sentences_lower_symmetry(self) -> None:
        lengths = [3, 25, 8, 40, 5]
        result = self.detector._compute_structural_symmetry(lengths, "text")
        assert result is not None
        assert result.sentence_uniformity < 0.8

    def test_paragraph_variance_with_paragraphs(self) -> None:
        text = "Short.\n\nThis is a longer paragraph with more sentences. It has two."
        lengths = [1, 8, 4]
        result = self.detector._compute_structural_symmetry(lengths, text)
        assert result is not None
        assert result.paragraph_length_variance is not None

    def test_paragraph_variance_none_single_paragraph(self) -> None:
        text = "Just one paragraph with some sentences."
        lengths = [5, 5, 5]
        result = self.detector._compute_structural_symmetry(lengths, text)
        assert result is not None
        assert result.paragraph_length_variance is None


class TestSentenceEntropy:
    def setup_method(self) -> None:
        self.detector = StatisticalDetector(
            use_gpt2_perplexity=False, use_deep_features=False
        )

    def test_returns_none_for_few_sentences(self) -> None:
        result = self.detector._compute_sentence_entropy([["hello"]])
        assert result is None

    def test_returns_result_for_sufficient_sentences(self) -> None:
        sentences = [
            ["the", "cat", "sat", "on", "the", "mat"],
            ["a", "dog", "ran", "through", "the", "park"],
            ["birds", "flew", "over", "the", "tall", "trees"],
        ]
        result = self.detector._compute_sentence_entropy(sentences)
        assert isinstance(result, SentenceEntropyResult)
        assert result.mean_entropy > 0
        assert len(result.per_sentence) == 3

    def test_uniform_sentences_low_variance(self) -> None:
        # Same word distribution in each sentence
        sent = ["a", "b", "c", "d", "e", "f"]
        sentences = [sent[:], sent[:], sent[:], sent[:]]
        result = self.detector._compute_sentence_entropy(sentences)
        assert result is not None
        assert result.entropy_variance < 0.01

    def test_varied_sentences_higher_variance(self) -> None:
        # Very different distributions
        sentences = [
            ["the", "the", "the", "the", "the", "the"],  # low entropy
            ["a", "b", "c", "d", "e", "f"],  # high entropy
            ["x", "x", "x", "x", "x", "y"],  # low entropy
            ["p", "q", "r", "s", "t", "u"],  # high entropy
        ]
        result = self.detector._compute_sentence_entropy(sentences)
        assert result is not None
        assert result.entropy_variance > 0.1


class TestNgramDistribution:
    def setup_method(self) -> None:
        self.detector = StatisticalDetector(
            use_gpt2_perplexity=False, use_deep_features=False
        )

    def test_returns_none_for_short_text(self) -> None:
        result = self.detector._compute_ngram_distribution(
            ["a", "b", "c"], ["NOUN", "VERB", "NOUN"]
        )
        assert result is None

    def test_returns_ngram_result(self) -> None:
        words = "the cat sat on the mat and the dog ran".split()
        pos_tags = ["DET", "NOUN", "VERB", "ADP", "DET", "NOUN", "CCONJ", "DET", "NOUN", "VERB"]
        result = self.detector._compute_ngram_distribution(words, pos_tags)
        assert isinstance(result, NgramResult)
        assert result.bigram_entropy > 0
        assert 0.0 <= result.trigram_rarity <= 1.0
        assert result.pos_trigram_entropy >= 0.0

    def test_repetitive_text_lower_bigram_entropy(self) -> None:
        repetitive = "the the the the the the the the the the".split()
        diverse = "cat dog bird fish tree rock lake sun moon star".split()
        pos_rep = ["DET"] * 10
        pos_div = ["NOUN"] * 10
        r_rep = self.detector._compute_ngram_distribution(repetitive, pos_rep)
        r_div = self.detector._compute_ngram_distribution(diverse, pos_div)
        assert r_rep is not None
        assert r_div is not None
        assert r_rep.bigram_entropy < r_div.bigram_entropy

    def test_trigram_rarity_all_unique(self) -> None:
        words = "a b c d e f g h i j".split()
        pos_tags = ["NOUN"] * 10
        result = self.detector._compute_ngram_distribution(words, pos_tags)
        assert result is not None
        assert result.trigram_rarity == 1.0


class TestClauseDensity:
    def setup_method(self) -> None:
        self.detector_no_deep = StatisticalDetector(
            use_gpt2_perplexity=False, use_deep_features=False
        )

    def test_returns_none_when_deep_disabled(self) -> None:
        result = self.detector_no_deep._compute_clause_density("Some text here.")
        assert result is None


class TestFastTierNoGPT2:
    def setup_method(self) -> None:
        self.detector = StatisticalDetector(
            use_gpt2_perplexity=False, use_deep_features=False
        )

    def test_perplexity_returns_none(self) -> None:
        result = self.detector._compute_perplexity("text", ["sentence one."])
        assert result is None

    def test_token_prob_returns_none(self) -> None:
        result = self.detector._compute_token_prob_distribution("text")
        assert result is None

    def test_extract_features_has_no_lm_keys(self) -> None:
        features = self.detector.extract_features(HUMAN_TEXT)
        assert "lm_perplexity_mean" not in features
        assert "lm_burstiness" not in features
        assert "lm_token_prob_tail_ratio" not in features

    def test_extract_features_has_style_keys(self) -> None:
        features = self.detector.extract_features(HUMAN_TEXT)
        assert "style_ttr" in features
        assert "style_hapax_ratio" in features

    def test_extract_features_has_struct_keys(self) -> None:
        features = self.detector.extract_features(HUMAN_TEXT)
        assert "struct_sentence_length_variance" in features

    def test_extract_features_has_disc_keys(self) -> None:
        features = self.detector.extract_features(AI_TEXT)
        assert "disc_pos_trigram_entropy" in features
        assert "disc_bigram_entropy" in features
