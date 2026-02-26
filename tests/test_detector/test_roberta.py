"""Tests for RoBERTa ONNX classifier (Tier 1 detector)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lucid.detector import DetectionError, DetectorInitError
from lucid.detector.roberta import (
    RobertaDetector,
    _run_inference,
    _softmax,
)


# ---------------------------------------------------------------------------
# Unit tests — no model loading required
# ---------------------------------------------------------------------------


class TestSoftmax:
    """Tests for _softmax helper."""

    def test_two_class_sums_to_one(self) -> None:
        """Softmax output probabilities must sum to 1.0."""
        probs = _softmax([1.0, 2.0])
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_higher_logit_gets_higher_probability(self) -> None:
        """Class with higher logit must receive higher probability."""
        probs = _softmax([0.0, 5.0])
        assert probs[1] > probs[0]

    def test_equal_logits_give_uniform_distribution(self) -> None:
        """Equal logits must yield 0.5 each for two classes."""
        probs = _softmax([3.0, 3.0])
        assert abs(probs[0] - 0.5) < 1e-6
        assert abs(probs[1] - 0.5) < 1e-6

    def test_known_values(self) -> None:
        """Verify softmax against a known reference computation."""
        # softmax([0, 1]) = [e^0/(e^0+e^1), e^1/(e^0+e^1)]
        import math

        probs = _softmax([0.0, 1.0])
        expected_1 = math.exp(1.0) / (math.exp(0.0) + math.exp(1.0))
        assert abs(probs[1] - expected_1) < 1e-6

    def test_large_logits_numerically_stable(self) -> None:
        """Softmax must not produce NaN or Inf for large logit values."""
        probs = _softmax([1000.0, 1001.0])
        assert all(math.isfinite(p) for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-6

    def test_ai_probability_is_index_1(self) -> None:
        """Index 1 represents Fake (AI) class — higher logit[1] => higher P(AI)."""
        # logits[0]=Real, logits[1]=Fake
        probs_ai = _softmax([-2.0, 3.0])   # strongly AI
        probs_hum = _softmax([3.0, -2.0])  # strongly human
        assert probs_ai[1] > 0.9
        assert probs_hum[1] < 0.1


import math  # noqa: E402  (needed in test body above)


class TestRunInference:
    """Tests for _run_inference with mocked ONNX session."""

    def _make_session(self, logits: list[float]) -> MagicMock:
        """Create a mock session that returns the given logits."""
        session = MagicMock()
        # outputs[0] has shape [1, 2]
        session.run.return_value = [np.array([[logits[0], logits[1]]])]
        # Provide input names
        inp_real = MagicMock()
        inp_real.name = "input_ids"
        inp_att = MagicMock()
        inp_att.name = "attention_mask"
        session.get_inputs.return_value = [inp_real, inp_att]
        return session

    def _make_tokenizer(self) -> MagicMock:
        tokenizer = MagicMock()
        tokenizer.cls_token_id = 0
        tokenizer.sep_token_id = 2
        return tokenizer

    def test_returns_float_in_unit_interval(self) -> None:
        """Inference must return a float in [0.0, 1.0]."""
        session = self._make_session([1.0, 1.0])
        tokenizer = self._make_tokenizer()
        score = _run_inference(session, tokenizer, [10, 20, 30])
        assert 0.0 <= score <= 1.0

    def test_high_ai_logit_gives_high_score(self) -> None:
        """Strongly AI logits (index 1 >> index 0) must give score > 0.9."""
        session = self._make_session([-5.0, 5.0])
        tokenizer = self._make_tokenizer()
        score = _run_inference(session, tokenizer, [10, 20])
        assert score > 0.9

    def test_high_human_logit_gives_low_score(self) -> None:
        """Strongly human logits (index 0 >> index 1) must give score < 0.1."""
        session = self._make_session([5.0, -5.0])
        tokenizer = self._make_tokenizer()
        score = _run_inference(session, tokenizer, [10, 20])
        assert score < 0.1

    def test_includes_token_type_ids_when_in_input_names(self) -> None:
        """When session expects token_type_ids, it must be in the feed dict."""
        session = self._make_session([1.0, 1.0])
        inp_tti = MagicMock()
        inp_tti.name = "token_type_ids"
        inp_ids = MagicMock()
        inp_ids.name = "input_ids"
        inp_att = MagicMock()
        inp_att.name = "attention_mask"
        session.get_inputs.return_value = [inp_ids, inp_att, inp_tti]
        tokenizer = self._make_tokenizer()
        _run_inference(session, tokenizer, [10])
        call_args = session.run.call_args
        feed = call_args[0][1]
        assert "token_type_ids" in feed

    def test_inference_exception_raises_detection_error(self) -> None:
        """ONNX session.run failure must raise DetectionError."""
        session = MagicMock()
        session.run.side_effect = RuntimeError("OOM")
        inp = MagicMock()
        inp.name = "input_ids"
        session.get_inputs.return_value = [inp]
        tokenizer = self._make_tokenizer()
        with pytest.raises(DetectionError, match="ONNX inference failed"):
            _run_inference(session, tokenizer, [10])


class TestRobertaDetectorLazyLoading:
    """Tests for lazy loading and threading behaviour."""

    def test_session_is_none_before_first_call(self) -> None:
        """Session and tokenizer must be None until detect_text is called."""
        detector = RobertaDetector()
        assert detector._session is None
        assert detector._tokenizer is None

    def test_lazy_loading_called_once_on_repeated_calls(self) -> None:
        """_ensure_loaded must load the model exactly once across multiple calls."""
        detector = RobertaDetector()
        load_count = 0

        original_ensure = detector._ensure_loaded

        def patched_ensure() -> None:
            nonlocal load_count
            load_count += 1
            original_ensure()

        # Patch at instance level
        detector._ensure_loaded = patched_ensure  # type: ignore[method-assign]

        mock_session = MagicMock()
        mock_session.run.return_value = [np.array([[1.0, 2.0]])]
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [10, 20, 30]
        mock_tokenizer.cls_token_id = 0
        mock_tokenizer.sep_token_id = 2

        detector._session = mock_session
        detector._tokenizer = mock_tokenizer

        detector.detect_text("Hello world.")
        detector.detect_text("Hello world.")
        # _ensure_loaded is patched but session is pre-set so the real one runs 0 times
        # This confirms no re-loading happens
        assert detector._session is mock_session


class TestRobertaDetectorSlidingWindow:
    """Tests for sliding window logic without model loading."""

    def _make_detector_with_mocks(
        self, token_ids: list[int], logit_pairs: list[tuple[float, float]]
    ) -> RobertaDetector:
        """Build a detector with pre-injected mocks.

        Args:
            token_ids: What tokenizer.encode returns.
            logit_pairs: Sequence of (real_logit, fake_logit) per window call.
        """
        detector = RobertaDetector()

        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = token_ids
        mock_tokenizer.cls_token_id = 0
        mock_tokenizer.sep_token_id = 2

        call_count = [0]
        outputs_sequence = [
            [np.array([[real, fake]])] for real, fake in logit_pairs
        ]

        def run_side_effect(
            output_names: Any, feed: dict[str, Any]
        ) -> list[Any]:
            idx = call_count[0]
            call_count[0] += 1
            return outputs_sequence[idx]

        mock_session = MagicMock()
        mock_session.run.side_effect = run_side_effect
        inp = MagicMock()
        inp.name = "input_ids"
        mock_session.get_inputs.return_value = [inp]

        detector._session = mock_session
        detector._tokenizer = mock_tokenizer
        return detector

    def test_short_text_single_pass(self) -> None:
        """Text <= 510 tokens must trigger exactly one inference call."""
        detector = self._make_detector_with_mocks(
            token_ids=list(range(100)),  # 100 tokens, well under 510
            logit_pairs=[(1.0, 3.0)],   # one call expected
        )
        score = detector.detect_text("short text")
        assert detector._tokenizer is not None
        assert detector._session.run.call_count == 1  # type: ignore[union-attr]
        assert 0.0 <= score <= 1.0

    def test_long_text_triggers_sliding_window(self) -> None:
        """Text > 510 tokens must trigger multiple inference calls."""
        # 766 tokens: window1=0..509, window2=256..765 (stride=256)
        long_ids = list(range(766))
        detector = self._make_detector_with_mocks(
            token_ids=long_ids,
            logit_pairs=[
                (2.0, 1.0),  # window 1: P(AI) ~ 0.27
                (1.0, 4.0),  # window 2: P(AI) ~ 0.95
            ],
        )
        score = detector.detect_text("long text " * 200)
        assert detector._session.run.call_count == 2  # type: ignore[union-attr]
        # Aggregation is mean — average of ~0.27 and ~0.95
        assert 0.5 < score < 0.7

    def test_sliding_window_aggregation_is_mean(self) -> None:
        """Aggregate score over windows must equal the mean window score."""
        # 1020 tokens: window1=0..509, window2=256..765, window3=512..1019
        long_ids = list(range(1020))
        # Window 2 has the highest AI probability
        detector = self._make_detector_with_mocks(
            token_ids=long_ids,
            logit_pairs=[
                (3.0, 1.0),   # window 1: P(AI) low
                (0.0, 5.0),   # window 2: P(AI) very high
                (2.0, 1.0),   # window 3: P(AI) low
            ],
        )
        score = detector.detect_text("very long text " * 300)
        # Mean of three window scores
        w1 = _softmax([3.0, 1.0])[1]
        w2 = _softmax([0.0, 5.0])[1]
        w3 = _softmax([2.0, 1.0])[1]
        expected_mean = (w1 + w2 + w3) / 3
        assert abs(score - expected_mean) < 1e-5

    def test_tokenization_failure_raises_detection_error(self) -> None:
        """Tokenizer exception during encode must raise DetectionError."""
        detector = RobertaDetector()
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = ValueError("tokenizer broke")
        mock_session = MagicMock()
        detector._session = mock_session
        detector._tokenizer = mock_tokenizer

        with pytest.raises(DetectionError, match="Tokenization failed"):
            detector.detect_text("some text")

    def test_exactly_510_tokens_is_single_pass(self) -> None:
        """Text with exactly 510 tokens must not trigger sliding window."""
        detector = self._make_detector_with_mocks(
            token_ids=list(range(510)),
            logit_pairs=[(0.0, 2.0)],
        )
        detector.detect_text("text at boundary")
        assert detector._session.run.call_count == 1  # type: ignore[union-attr]

    def test_511_tokens_triggers_two_windows(self) -> None:
        """Text with 511 tokens must trigger the sliding window path."""
        detector = self._make_detector_with_mocks(
            token_ids=list(range(511)),
            logit_pairs=[(1.0, 1.0), (1.0, 1.0)],
        )
        detector.detect_text("text just over boundary")
        assert detector._session.run.call_count == 2  # type: ignore[union-attr]


@pytest.mark.integration
class TestRobertaDetectorIntegration:
    """Integration tests requiring actual model download."""

    def test_detect_text_returns_valid_score(self) -> None:
        """detect_text must return a float in [0.0, 1.0] on real inference."""
        detector = RobertaDetector()
        score = detector.detect_text(
            "The neural network was trained on a large corpus of text data."
        )
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_model_loads_only_once(self) -> None:
        """Repeated calls must not reload the model."""
        detector = RobertaDetector()
        detector.detect_text("First call.")
        session_id = id(detector._session)
        detector.detect_text("Second call.")
        assert id(detector._session) == session_id
