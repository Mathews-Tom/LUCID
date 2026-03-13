"""Tests for the lightweight similarity function."""

from __future__ import annotations

from lucid.transform.similarity import quick_similarity


class TestQuickSimilarity:
    def test_identical_texts(self) -> None:
        assert quick_similarity("hello world", "hello world") == 1.0

    def test_both_empty(self) -> None:
        assert quick_similarity("", "") == 1.0

    def test_one_empty(self) -> None:
        assert quick_similarity("hello", "") == 0.0
        assert quick_similarity("", "hello") == 0.0

    def test_case_insensitive(self) -> None:
        score = quick_similarity("Hello World", "hello world")
        assert score == 1.0

    def test_minor_paraphrase(self) -> None:
        original = "The model achieves state-of-the-art results on this benchmark."
        transformed = "The model attains state-of-the-art results on this benchmark."
        score = quick_similarity(original, transformed)
        assert score > 0.85

    def test_aggressive_rewrite(self) -> None:
        original = "The Cranfield experiments established evaluation methodology."
        transformed = "Scientists developed new ways to assess how well search works."
        score = quick_similarity(original, transformed)
        assert score < 0.5

    def test_moderate_paraphrase(self) -> None:
        original = (
            "BM25 is still a strong contender, even in 2024, when it "
            "beats dense retrievers in some BEIR domains."
        )
        transformed = (
            "Even in 2024, BM25 remains a strong competitor, outperforming "
            "dense retrievers in certain BEIR domains."
        )
        score = quick_similarity(original, transformed)
        assert 0.5 < score < 0.9

    def test_completely_different(self) -> None:
        score = quick_similarity(
            "The cat sat on the mat.",
            "Quantum computing revolutionizes cryptographic protocols.",
        )
        assert score < 0.25
