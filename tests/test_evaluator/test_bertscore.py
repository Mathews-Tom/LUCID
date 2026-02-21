"""Tests for BERTScore semantic similarity checker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lucid.evaluator.bertscore import BERTScoreChecker, BERTScoreResult


def _mock_tensor(value: float) -> MagicMock:
    """Create a mock tensor that supports ``tensor[0].item()``."""
    item_mock = MagicMock(return_value=value)
    element_mock = MagicMock()
    element_mock.item = item_mock
    tensor = MagicMock()
    tensor.__getitem__ = MagicMock(return_value=element_mock)
    return tensor


class TestBERTScoreChecker:
    """Unit tests for BERTScoreChecker — no model loading."""

    def test_lazy_loading_deferred_until_compute(self) -> None:
        """Scorer is not instantiated at construction time."""
        checker = BERTScoreChecker(model_type="microsoft/deberta-xlarge-mnli")
        assert checker._scorer is None

    @patch("bert_score.BERTScorer", autospec=False)
    def test_compute_extracts_values_correctly(self, mock_scorer_cls: MagicMock) -> None:
        """compute() extracts precision, recall, F1 from scorer output."""
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = (
            _mock_tensor(0.92),
            _mock_tensor(0.89),
            _mock_tensor(0.90),
        )
        mock_scorer_cls.return_value = mock_scorer

        checker = BERTScoreChecker(model_type="microsoft/deberta-xlarge-mnli")
        result = checker.compute("original text", "paraphrase text")

        assert isinstance(result, BERTScoreResult)
        assert result.precision == pytest.approx(0.92)
        assert result.recall == pytest.approx(0.89)
        assert result.f1 == pytest.approx(0.90)

    @patch("bert_score.BERTScorer", autospec=False)
    def test_scorer_loaded_once_across_calls(self, mock_scorer_cls: MagicMock) -> None:
        """BERTScorer is instantiated only on first compute(), reused after."""
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = (
            _mock_tensor(0.85),
            _mock_tensor(0.80),
            _mock_tensor(0.82),
        )
        mock_scorer_cls.return_value = mock_scorer

        checker = BERTScoreChecker(model_type="microsoft/deberta-xlarge-mnli")
        checker.compute("a", "b")
        checker.compute("c", "d")

        mock_scorer_cls.assert_called_once()

    @patch("bert_score.BERTScorer", autospec=False)
    def test_scorer_receives_correct_arguments(self, mock_scorer_cls: MagicMock) -> None:
        """BERTScorer is constructed with the configured model and options."""
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = (
            _mock_tensor(0.5),
            _mock_tensor(0.5),
            _mock_tensor(0.5),
        )
        mock_scorer_cls.return_value = mock_scorer

        checker = BERTScoreChecker(model_type="custom/model")
        checker.compute("x", "y")

        mock_scorer_cls.assert_called_once_with(
            model_type="custom/model",
            rescale_with_baseline=True,
            lang="en",
        )

    @patch("bert_score.BERTScorer", autospec=False)
    def test_score_called_with_cands_and_refs(self, mock_scorer_cls: MagicMock) -> None:
        """scorer.score() receives paraphrase as cands and original as refs."""
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = (
            _mock_tensor(0.7),
            _mock_tensor(0.7),
            _mock_tensor(0.7),
        )
        mock_scorer_cls.return_value = mock_scorer

        checker = BERTScoreChecker(model_type="m")
        checker.compute("the original", "the paraphrase")

        mock_scorer.score.assert_called_once_with(
            cands=["the paraphrase"],
            refs=["the original"],
        )

    @patch("bert_score.BERTScorer", autospec=False)
    def test_negative_values_allowed(self, mock_scorer_cls: MagicMock) -> None:
        """Baseline-rescaled scores can be negative; result must preserve them."""
        mock_scorer = MagicMock()
        mock_scorer.score.return_value = (
            _mock_tensor(-0.15),
            _mock_tensor(-0.20),
            _mock_tensor(-0.18),
        )
        mock_scorer_cls.return_value = mock_scorer

        checker = BERTScoreChecker(model_type="m")
        result = checker.compute("cats sleep", "quantum entanglement theory")

        assert result.precision == pytest.approx(-0.15)
        assert result.recall == pytest.approx(-0.20)
        assert result.f1 == pytest.approx(-0.18)

    def test_result_is_frozen(self) -> None:
        """BERTScoreResult fields cannot be mutated."""
        result = BERTScoreResult(precision=0.9, recall=0.8, f1=0.85)
        with pytest.raises(AttributeError):
            result.f1 = 0.99  # type: ignore[misc]


@pytest.mark.integration
class TestBERTScoreCheckerIntegration:
    """Integration tests — requires model download."""

    def setup_method(self) -> None:
        self.checker = BERTScoreChecker(model_type="microsoft/deberta-xlarge-mnli")

    def test_identical_texts_high_f1(self) -> None:
        """Identical texts produce F1 > 0.9."""
        text = "Transformers have revolutionized natural language processing."
        result = self.checker.compute(text, text)
        assert result.f1 > 0.9

    def test_unrelated_texts_low_f1(self) -> None:
        """Completely unrelated texts produce F1 < 0.3."""
        original = "The mitochondria is the powerhouse of the cell."
        paraphrase = "Stock markets closed higher on Tuesday amid trade optimism."
        result = self.checker.compute(original, paraphrase)
        assert result.f1 < 0.3
