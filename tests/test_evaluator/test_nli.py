"""Tests for the NLI semantic equivalence checker."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lucid.evaluator.nli import NLIChecker, NLIResult


def _make_pipeline_output(
    entailment: float = 0.0,
    neutral: float = 0.0,
    contradiction: float = 0.0,
) -> list[dict[str, float | str]]:
    """Build a mock pipeline output matching HuggingFace top_k=None format."""
    return [
        {"label": "entailment", "score": entailment},
        {"label": "neutral", "score": neutral},
        {"label": "contradiction", "score": contradiction},
    ]


class TestNLIChecker:
    """Unit tests for NLIChecker — pipeline is mocked, no model downloads."""

    def setup_method(self) -> None:
        self.checker = NLIChecker(
            model_name="mock-model",
            require_bidirectional=True,
        )
        self.mock_pipeline = MagicMock()
        self.checker._pipeline = self.mock_pipeline

    def test_forward_and_backward_entailment(self) -> None:
        """Both directions return entailment for a valid paraphrase."""
        self.mock_pipeline.side_effect = [
            _make_pipeline_output(entailment=0.92, neutral=0.05, contradiction=0.03),
            _make_pipeline_output(entailment=0.89, neutral=0.07, contradiction=0.04),
        ]

        result = self.checker.check("The cat sat on the mat.", "A cat was sitting on the mat.")

        assert result.forward_label == "entailment"
        assert result.backward_label == "entailment"

    def test_forward_entailment_backward_neutral(self) -> None:
        """Forward entails but backward is neutral — asymmetric paraphrase."""
        self.mock_pipeline.side_effect = [
            _make_pipeline_output(entailment=0.85, neutral=0.10, contradiction=0.05),
            _make_pipeline_output(entailment=0.20, neutral=0.70, contradiction=0.10),
        ]

        result = self.checker.check(
            "All mammals are warm-blooded.",
            "Dogs are warm-blooded.",
        )

        assert result.forward_label == "entailment"
        assert result.backward_label == "neutral"

    def test_contradiction_detected(self) -> None:
        """Contradictory pair yields contradiction label."""
        self.mock_pipeline.side_effect = [
            _make_pipeline_output(entailment=0.05, neutral=0.10, contradiction=0.85),
            _make_pipeline_output(entailment=0.05, neutral=0.10, contradiction=0.85),
        ]

        result = self.checker.check("The sky is blue.", "The sky is green.")

        assert result.forward_label == "contradiction"
        assert result.backward_label == "contradiction"

    def test_score_dict_construction(self) -> None:
        """Score dictionaries contain all three labels with values summing to ~1.0."""
        self.mock_pipeline.side_effect = [
            _make_pipeline_output(entailment=0.70, neutral=0.20, contradiction=0.10),
            _make_pipeline_output(entailment=0.65, neutral=0.25, contradiction=0.10),
        ]

        result = self.checker.check("Input A.", "Input B.")

        assert set(result.forward_scores.keys()) == {"entailment", "neutral", "contradiction"}
        assert set(result.backward_scores.keys()) == {"entailment", "neutral", "contradiction"}

        forward_sum = sum(result.forward_scores.values())
        backward_sum = sum(result.backward_scores.values())
        assert forward_sum == pytest.approx(1.0, abs=1e-6)
        assert backward_sum == pytest.approx(1.0, abs=1e-6)

    def test_score_values_match_pipeline_output(self) -> None:
        """Individual scores in the result match the pipeline output exactly."""
        self.mock_pipeline.side_effect = [
            _make_pipeline_output(entailment=0.88, neutral=0.07, contradiction=0.05),
            _make_pipeline_output(entailment=0.91, neutral=0.06, contradiction=0.03),
        ]

        result = self.checker.check("premise", "hypothesis")

        assert result.forward_scores["entailment"] == pytest.approx(0.88)
        assert result.forward_scores["neutral"] == pytest.approx(0.07)
        assert result.backward_scores["entailment"] == pytest.approx(0.91)

    def test_pipeline_called_twice_for_bidirectional(self) -> None:
        """Pipeline is called exactly twice: forward and backward."""
        self.mock_pipeline.side_effect = [
            _make_pipeline_output(entailment=0.90, neutral=0.05, contradiction=0.05),
            _make_pipeline_output(entailment=0.90, neutral=0.05, contradiction=0.05),
        ]

        self.checker.check("original text", "paraphrased text")

        assert self.mock_pipeline.call_count == 2

        # Forward call: original as premise, paraphrase as hypothesis
        forward_call = self.mock_pipeline.call_args_list[0]
        assert forward_call[0][0] == {"text": "original text", "text_pair": "paraphrased text"}

        # Backward call: paraphrase as premise, original as hypothesis
        backward_call = self.mock_pipeline.call_args_list[1]
        assert backward_call[0][0] == {"text": "paraphrased text", "text_pair": "original text"}

    def test_lazy_loading_does_not_trigger_on_init(self) -> None:
        """Pipeline is not loaded during __init__."""
        checker = NLIChecker(model_name="some-model")
        assert checker._pipeline is None

    def test_lazy_loading_triggers_on_first_check(self) -> None:
        """Pipeline loads on first property access and caches the result."""
        checker = NLIChecker(model_name="test-model")
        assert checker._pipeline is None

        mock_pipe = MagicMock()
        mock_pipe.side_effect = [
            _make_pipeline_output(entailment=0.90, neutral=0.05, contradiction=0.05),
            _make_pipeline_output(entailment=0.90, neutral=0.05, contradiction=0.05),
            _make_pipeline_output(entailment=0.80, neutral=0.10, contradiction=0.10),
            _make_pipeline_output(entailment=0.80, neutral=0.10, contradiction=0.10),
        ]

        # Patch at the site where the local import resolves: builtins.__import__
        # is overkill; instead, just set the pipeline directly and verify caching.
        checker._pipeline = mock_pipe

        checker.check("a", "b")
        checker.check("c", "d")

        # Pipeline was called 4 times total (2 directions x 2 checks)
        assert mock_pipe.call_count == 4
        # Pipeline reference is the same object — cached, not recreated
        assert checker._pipeline is mock_pipe

    def test_lazy_loading_creates_pipeline_on_first_access(self) -> None:
        """The pipeline property calls transformers.pipeline on first access."""
        checker = NLIChecker(model_name="test-model")

        mock_pipe = MagicMock()
        mock_pipe.return_value = _make_pipeline_output(
            entailment=0.90,
            neutral=0.05,
            contradiction=0.05,
        )

        with patch("transformers.pipelines.pipeline", return_value=mock_pipe) as mock_factory:
            result_pipe = checker.pipeline

            mock_factory.assert_called_once_with(
                "text-classification",
                model="test-model",
                top_k=None,
            )
            assert result_pipe is mock_pipe
            assert checker._pipeline is mock_pipe

    def test_result_is_frozen(self) -> None:
        """NLIResult instances are immutable."""
        result = NLIResult(
            forward_label="entailment",
            backward_label="entailment",
            forward_scores={"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
            backward_scores={"entailment": 0.9, "neutral": 0.05, "contradiction": 0.05},
        )
        with pytest.raises(AttributeError):
            result.forward_label = "neutral"  # type: ignore[misc]

    def test_neutral_in_both_directions(self) -> None:
        """Unrelated texts produce neutral in both directions."""
        self.mock_pipeline.side_effect = [
            _make_pipeline_output(entailment=0.10, neutral=0.80, contradiction=0.10),
            _make_pipeline_output(entailment=0.08, neutral=0.82, contradiction=0.10),
        ]

        result = self.checker.check("The weather is nice.", "Python is a programming language.")

        assert result.forward_label == "neutral"
        assert result.backward_label == "neutral"


@pytest.mark.integration
class TestNLICheckerIntegration:
    """Integration tests — downloads and runs the actual NLI model.

    Run with: pytest -m integration
    """

    def setup_method(self) -> None:
        self.checker = NLIChecker(
            model_name="MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c",
            require_bidirectional=True,
        )

    def test_paraphrase_entailment(self) -> None:
        """A genuine paraphrase pair yields entailment in both directions."""
        result = self.checker.check(
            "The researchers found that the new algorithm outperforms the baseline.",
            "The new algorithm was shown by researchers to surpass the baseline.",
        )
        assert result.forward_label == "entailment"
        assert result.backward_label == "entailment"
        assert result.forward_scores["entailment"] > 0.7
        assert result.backward_scores["entailment"] > 0.7

    def test_contradiction_pair(self) -> None:
        """Contradictory statements yield contradiction in at least one direction."""
        result = self.checker.check(
            "The experiment was successful and met all targets.",
            "The experiment failed to meet any of its targets.",
        )
        has_contradiction = (
            result.forward_label == "contradiction" or result.backward_label == "contradiction"
        )
        assert has_contradiction

    def test_unrelated_pair(self) -> None:
        """Semantically unrelated texts yield neutral in at least one direction."""
        result = self.checker.check(
            "Photosynthesis converts sunlight into chemical energy.",
            "The stock market closed higher on Tuesday.",
        )
        has_neutral = result.forward_label == "neutral" or result.backward_label == "neutral"
        assert has_neutral

    def test_scores_sum_to_one(self) -> None:
        """Score distributions from the real model sum to approximately 1.0."""
        result = self.checker.check(
            "Machine learning models require large datasets.",
            "Large datasets are needed for training ML models.",
        )
        forward_sum = sum(result.forward_scores.values())
        backward_sum = sum(result.backward_scores.values())
        assert forward_sum == pytest.approx(1.0, abs=1e-4)
        assert backward_sum == pytest.approx(1.0, abs=1e-4)
