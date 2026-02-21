"""RoBERTa ONNX-based AI content classifier (Tier 1 detector).

Uses the openai-community/roberta-base-openai-detector model via ONNX Runtime
for CPU-only inference. Long texts are handled via a sliding window strategy.
"""

from __future__ import annotations

import math
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import onnxruntime as ort
    from transformers import AutoTokenizer as _AutoTokenizer

from lucid.detector import DetectionError, DetectorInitError

_MODEL_ID_DEFAULT = "openai-community/roberta-base-openai-detector"
_ONNX_MODEL_ID = "onnx-community/roberta-base-openai-detector-ONNX"

# 512 max tokens - [CLS] - [SEP]
_WINDOW_TOKENS = 510
_STRIDE_TOKENS = 256

_lock = threading.Lock()


def _softmax(logits: list[float]) -> list[float]:
    """Compute softmax over a list of logits.

    Args:
        logits: Raw model output values.

    Returns:
        Probability distribution over classes.
    """
    max_val = max(logits)
    exps = [math.exp(x - max_val) for x in logits]
    total = sum(exps)
    return [e / total for e in exps]


class RobertaDetector:
    """Tier 1 AI content classifier using RoBERTa via ONNX Runtime.

    Lazy-loads the tokenizer and ONNX session on first call to detect().
    Uses a sliding window for texts exceeding 512 tokens.

    Args:
        model_id: HuggingFace model identifier for the detector.
    """

    def __init__(self, model_id: str = _MODEL_ID_DEFAULT) -> None:
        self._model_id = model_id
        self._session: ort.InferenceSession | None = None
        self._tokenizer: _AutoTokenizer | None = None

    def _ensure_loaded(self) -> None:
        """Lazy-load the ONNX session and tokenizer (thread-safe double-checked locking).

        Raises:
            DetectorInitError: If the model or tokenizer cannot be loaded.
        """
        if self._session is not None:
            return
        with _lock:
            if self._session is not None:
                return
            self._session, self._tokenizer = _load_model(self._model_id)

    def detect_text(self, text: str) -> float:
        """Score a raw text string for AI-generated content probability.

        Args:
            text: The input text to classify.

        Returns:
            P(AI-generated) in [0.0, 1.0]. Higher = more likely AI.

        Raises:
            DetectorInitError: If the model fails to load.
            DetectionError: If inference fails at runtime.
        """
        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._session is not None

        try:
            # Tokenize without special tokens to count raw content tokens
            token_ids: list[int] = self._tokenizer.encode(
                text, add_special_tokens=False
            )
        except Exception as exc:
            raise DetectionError(f"Tokenization failed: {exc}") from exc

        if len(token_ids) <= _WINDOW_TOKENS:
            return _run_inference(self._session, self._tokenizer, token_ids)

        # Sliding window for long texts
        window_scores: list[float] = []
        start = 0
        while start < len(token_ids):
            window = token_ids[start : start + _WINDOW_TOKENS]
            score = _run_inference(self._session, self._tokenizer, window)
            window_scores.append(score)
            if start + _WINDOW_TOKENS >= len(token_ids):
                break
            start += _STRIDE_TOKENS

        # Aggregate: max over windows (most suspicious window wins)
        return max(window_scores)


def _load_model(model_id: str) -> tuple[ort.InferenceSession, _AutoTokenizer]:
    """Load ONNX session and tokenizer.

    First tries optimum.onnxruntime for a seamless from_pretrained flow.
    Falls back to direct ONNX file download via HuggingFace Hub.

    Args:
        model_id: HuggingFace model identifier.

    Returns:
        Tuple of (InferenceSession, AutoTokenizer).

    Raises:
        DetectorInitError: If any loading step fails.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise DetectorInitError(
            "transformers is required. Install with: uv add transformers"
        ) from exc

    try:
        tokenizer: _AutoTokenizer = AutoTokenizer.from_pretrained(model_id)
    except Exception as exc:
        raise DetectorInitError(
            f"Failed to load tokenizer for {model_id!r}: {exc}"
        ) from exc

    session = _load_onnx_session(model_id)
    return session, tokenizer


def _load_onnx_session(model_id: str) -> ort.InferenceSession:
    """Load ONNX InferenceSession via optimum or direct hub download.

    Args:
        model_id: HuggingFace model identifier for the detector.

    Returns:
        Configured ONNX InferenceSession (CPUExecutionProvider only).

    Raises:
        DetectorInitError: If session creation fails.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise DetectorInitError(
            "onnxruntime is required. Install with: uv add onnxruntime"
        ) from exc

    # Strategy 1: optimum.onnxruntime (preferred — handles ONNX export automatically)
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification

        ort_model = ORTModelForSequenceClassification.from_pretrained(
            model_id, export=True, provider="CPUExecutionProvider"
        )
        # ORTModelForSequenceClassification wraps a session internally
        session: ort.InferenceSession = ort_model.model  # type: ignore[attr-defined]
        return session
    except ImportError:
        pass  # optimum not installed — fall through to direct strategy
    except Exception as exc:
        raise DetectorInitError(
            f"optimum failed to load model {model_id!r}: {exc}"
        ) from exc

    # Strategy 2: download pre-converted ONNX from hub
    try:
        from huggingface_hub import hf_hub_download

        onnx_path = hf_hub_download(
            repo_id=_ONNX_MODEL_ID,
            filename="onnx/model.onnx",
        )
        session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        return session
    except ImportError as exc:
        raise DetectorInitError(
            "huggingface_hub is required when optimum is unavailable. "
            "Install with: uv add huggingface_hub"
        ) from exc
    except Exception as exc:
        raise DetectorInitError(
            f"Failed to download or load ONNX model from {_ONNX_MODEL_ID!r}: {exc}"
        ) from exc


def _run_inference(
    session: ort.InferenceSession,
    tokenizer: _AutoTokenizer,
    token_ids: list[int],
) -> float:
    """Run a single forward pass on a window of token IDs.

    Prepends [CLS] and appends [SEP], encodes to numpy, runs the session,
    and returns P(AI) via softmax on logits.

    Args:
        session: Active ONNX InferenceSession.
        tokenizer: Tokenizer for special token IDs.
        token_ids: Content token IDs (no special tokens).

    Returns:
        P(AI-generated) in [0.0, 1.0].

    Raises:
        DetectionError: If ONNX inference raises any exception.
    """
    import numpy as np

    cls_id: int = tokenizer.cls_token_id  # type: ignore[assignment]
    sep_id: int = tokenizer.sep_token_id  # type: ignore[assignment]

    ids = [cls_id] + token_ids + [sep_id]
    attention_mask = [1] * len(ids)

    input_ids_arr = np.array([ids], dtype=np.int64)
    attention_mask_arr = np.array([attention_mask], dtype=np.int64)

    # Build feed dict from session input names
    input_names = {inp.name for inp in session.get_inputs()}
    feed: dict[str, Any] = {
        "input_ids": input_ids_arr,
        "attention_mask": attention_mask_arr,
    }
    if "token_type_ids" in input_names:
        feed["token_type_ids"] = np.zeros_like(input_ids_arr)

    try:
        outputs = session.run(None, feed)
    except Exception as exc:
        raise DetectionError(f"ONNX inference failed: {exc}") from exc

    # outputs[0] shape: [1, 2] — logits for [Real, Fake]
    logits: list[float] = outputs[0][0].tolist()
    probs = _softmax(logits)
    # Index 1 = Fake (AI-generated) probability
    return probs[1]
