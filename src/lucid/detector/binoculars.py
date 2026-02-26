"""Binoculars cross-perplexity AI text detection (Tier 3).

Optional deep analysis tier using paired language models to compute
perplexity ratios. Disabled by default due to high memory requirements.
Requires torch to be installed.

Reference: Binoculars paper (ICML 2024).
"""

from __future__ import annotations

import gc
import logging
import math
import threading
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_OBSERVER: str = "tiiuae/falcon-7b-instruct"
DEFAULT_PERFORMER: str = "tiiuae/falcon-7b"
BINOCULARS_THRESHOLD: float = 0.9015
SIGMOID_SCALE: float = 10.0

_lock = threading.Lock()


class BinocularsDetector:
    """Cross-perplexity AI text detector using paired language models.

    Uses two language models (observer and performer) to compute a
    perplexity ratio that distinguishes AI-generated text from human text.
    Models are lazy-loaded on first use and shared across calls.
    """

    def __init__(
        self,
        observer_model: str = DEFAULT_OBSERVER,
        performer_model: str = DEFAULT_PERFORMER,
    ) -> None:
        """Initialize BinocularsDetector with model identifiers.

        Args:
            observer_model: HuggingFace model ID for the observer model.
            performer_model: HuggingFace model ID for the performer model.
        """
        self._observer_id = observer_model
        self._performer_id = performer_model
        self._observer: Any = None  # AutoModelForCausalLM when loaded
        self._performer: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    def score(self, text: str) -> float:
        """Compute normalized Binoculars score.

        Args:
            text: Input text to analyze.

        Returns:
            Score in [0.0, 1.0] where 1.0 indicates AI-generated.

        Raises:
            BinocularsUnavailableError: If torch is missing or models fail to load.
        """
        self._ensure_loaded()
        raw = self._compute_raw_score(text)
        return self._normalize_score(raw)

    def _ensure_loaded(self) -> None:
        """Lazy-load models on first use. Thread-safe via double-checked locking."""
        if self._loaded:
            return
        with _lock:
            if self._loaded:
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as e:
                from lucid.detector import BinocularsUnavailableError

                raise BinocularsUnavailableError(
                    "torch and transformers required for Binoculars tier. "
                    "Install with: uv add torch transformers"
                ) from e
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self._observer_id)
                self._observer = AutoModelForCausalLM.from_pretrained(
                    self._observer_id, dtype=torch.float16
                ).to("cpu")
                self._performer = AutoModelForCausalLM.from_pretrained(
                    self._performer_id, dtype=torch.float16
                ).to("cpu")
                self._observer.eval()
                self._performer.eval()
                self._loaded = True
            except Exception as e:
                from lucid.detector import BinocularsUnavailableError

                raise BinocularsUnavailableError(
                    f"Failed to load Binoculars models: {e}"
                ) from e

    def _compute_raw_score(self, text: str) -> float:
        """Compute raw Binoculars perplexity ratio.

        Args:
            text: Input text to score.

        Returns:
            Raw perplexity / cross-perplexity ratio. Lower values indicate
            AI-generated text (AI text has low perplexity under observer).
        """
        import torch
        import torch.nn.functional as F

        with _lock:
            inputs = self._tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            observer_logits = self._observer(**inputs).logits
            performer_logits = self._performer(**inputs).logits

        # Shift for next-token prediction
        shift_logits_obs = observer_logits[:, :-1, :]
        shift_logits_perf = performer_logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Observer log-probs
        log_probs_obs = F.log_softmax(shift_logits_obs, dim=-1)
        # Performer probs
        probs_perf = F.softmax(shift_logits_perf, dim=-1)

        # Perplexity: -mean(log P_obs(x_i | x_{<i}))
        token_log_probs = log_probs_obs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        perplexity = -token_log_probs.mean().item()

        # Cross-perplexity: -mean(sum_v P_perf(v) * log P_obs(v))
        cross_entropy = -(probs_perf * log_probs_obs).sum(dim=-1).mean().item()

        if cross_entropy == 0:
            return 1.0  # edge case: avoid division by zero
        return perplexity / cross_entropy

    @staticmethod
    def _normalize_score(raw: float) -> float:
        """Map raw Binoculars ratio to [0, 1] where 1.0 = AI-generated.

        AI-generated text has low raw ratios (perplexity/cross-perplexity < threshold).
        The sigmoid is inverted so that low raw values produce scores near 1.0.

        Args:
            raw: Raw perplexity ratio from _compute_raw_score.

        Returns:
            Normalized score clamped to [0.0, 1.0].
        """
        exponent = (raw - BINOCULARS_THRESHOLD) * SIGMOID_SCALE
        # Guard against math.exp overflow for extreme raw values
        if exponent >= 709.0:
            return 0.0
        if exponent <= -709.0:
            return 1.0
        normalized = 1.0 / (1.0 + math.exp(exponent))
        return max(0.0, min(1.0, normalized))

    def unload(self) -> None:
        """Explicitly release model memory.

        Clears all model references and flushes accelerator caches
        (CUDA and MPS) before forcing garbage collection.
        """
        self._observer = None
        self._performer = None
        self._tokenizer = None
        self._loaded = False
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "backends") and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
        except (ImportError, AttributeError):
            pass
        gc.collect()

    @property
    def loaded(self) -> bool:
        """Whether models are currently loaded in memory."""
        return self._loaded
