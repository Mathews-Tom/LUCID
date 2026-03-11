"""Tests for readability metrics."""

from __future__ import annotations

import pytest

from lucid.metrics.readability import LexicalDiversityMetric, SentenceLengthVarianceMetric


class TestSentenceLengthVarianceMetric:
    def setup_method(self) -> None:
        self.metric = SentenceLengthVarianceMetric()

    def test_uniform_sentences(self) -> None:
        transformed = "One two three. Four five six. Seven eight nine."
        result = self.metric.compute("", transformed)
        assert result.value == pytest.approx(0.0)

    def test_varied_sentences(self) -> None:
        transformed = "Short. This is a much longer sentence with many words."
        result = self.metric.compute("", transformed)
        assert result.value > 0.0

    def test_empty_text(self) -> None:
        result = self.metric.compute("", "")
        assert result.value == pytest.approx(0.0)

    def test_single_sentence(self) -> None:
        result = self.metric.compute("", "Just one sentence here.")
        assert result.value == pytest.approx(0.0)


class TestLexicalDiversityMetric:
    def setup_method(self) -> None:
        self.metric = LexicalDiversityMetric()

    def test_all_unique_words(self) -> None:
        result = self.metric.compute("", "the quick brown fox")
        assert result.value == pytest.approx(1.0)

    def test_repeated_words(self) -> None:
        result = self.metric.compute("", "the the the the")
        assert result.value == pytest.approx(0.25)

    def test_empty_text(self) -> None:
        result = self.metric.compute("", "")
        assert result.value == pytest.approx(0.0)

    def test_mixed_case_normalized(self) -> None:
        result = self.metric.compute("", "The the THE")
        assert result.value == pytest.approx(1 / 3)
