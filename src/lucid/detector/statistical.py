"""Statistical feature extractor for AI content detection (Tier 2 detector).

Extracts seven linguistic features from prose text and combines them into
a detection score. GPT-2 perplexity proxy is optional (requires torch).
XGBoost classifier is optional; falls back to normalized feature averaging.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

from lucid.detector import DetectionError, DetectorInitError

if TYPE_CHECKING:
    import spacy
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MIN_WORDS_THRESHOLD = 50
MIN_SENTENCES_THRESHOLD = 3

# Empirical bounds for normalisation: (low, high, invert)
# invert=True means low value => AI-like (score maps to 1.0 when feature is at low bound)
_FEATURE_BOUNDS: dict[str, tuple[float, float, bool]] = {
    "perplexity_proxy": (10.0, 200.0, True),   # lower perplexity = more predictable = AI
    "burstiness": (0.0, 1.5, True),             # lower burstiness = more uniform = AI
    "ttr": (0.2, 0.9, True),                    # lower TTR = more repetitive = AI
    "hapax_ratio": (0.1, 0.6, True),            # fewer unique words = AI
    "mean_sentence_length": (5.0, 40.0, False), # no clear direction; use raw
    "sentence_length_variance": (0.0, 200.0, True),  # lower variance = more uniform = AI
    "pos_trigram_entropy": (1.0, 6.0, True),    # lower entropy = more predictable = AI
}

_spacy_lock = threading.Lock()
_gpt2_lock = threading.Lock()


class StatisticalDetector:
    """Tier 2 AI content detector using statistical linguistic features.

    Extracts seven features from prose text and combines them into a score
    in [0.0, 1.0]. Requires spaCy (en_core_web_sm) for POS tagging.
    GPT-2 perplexity proxy is used when torch is available.

    Args:
        use_gpt2_perplexity: Whether to compute GPT-2 perplexity proxy.
            Requires torch and transformers. Defaults to True (lazy-loads).
    """

    def __init__(self, use_gpt2_perplexity: bool = True) -> None:
        self._use_gpt2 = use_gpt2_perplexity
        self._nlp: spacy.language.Language | None = None
        self._gpt2_model: GPT2LMHeadModel | None = None
        self._gpt2_tokenizer: GPT2TokenizerFast | None = None

    def _ensure_spacy(self) -> None:
        """Lazy-load spaCy model (thread-safe).

        Raises:
            DetectorInitError: If spaCy or the en_core_web_sm model is missing.
        """
        if self._nlp is not None:
            return
        with _spacy_lock:
            if self._nlp is not None:
                return
            try:
                import spacy as _spacy
            except ImportError as exc:
                raise DetectorInitError(
                    "spacy is required. Install with: uv add spacy"
                ) from exc
            try:
                self._nlp = _spacy.load(
                    "en_core_web_sm", disable=["parser", "ner", "lemmatizer"]
                )
            except OSError as exc:
                raise DetectorInitError(
                    "spaCy model not found. Install with: "
                    "python -m spacy download en_core_web_sm"
                ) from exc

    def _load_gpt2(self) -> None:
        """Lazy-load GPT-2 model and tokenizer (thread-safe).

        Raises:
            DetectorInitError: If transformers or torch is not available.
        """
        with _gpt2_lock:
            if self._gpt2_model is not None:
                return
            try:
                from transformers import GPT2LMHeadModel as _GPT2Model
                from transformers import GPT2TokenizerFast as _GPT2Tokenizer
            except ImportError as exc:
                raise DetectorInitError(
                    "transformers is required for perplexity feature. "
                    "Install with: uv add transformers"
                ) from exc
            try:
                self._gpt2_tokenizer = _GPT2Tokenizer.from_pretrained("gpt2")
                self._gpt2_model = _GPT2Model.from_pretrained("gpt2")
                self._gpt2_model.eval()  # type: ignore[union-attr]
            except Exception as exc:
                raise DetectorInitError(
                    f"Failed to load GPT-2 model: {exc}"
                ) from exc

    def _compute_perplexity_proxy(self, text: str) -> float | None:
        """Compute mean negative log-likelihood via GPT-2 (perplexity proxy).

        Lower values indicate more predictable, AI-like text.

        Args:
            text: Input prose text.

        Returns:
            Mean NLL loss value, or None if torch is unavailable or text too short.

        Raises:
            DetectionError: If GPT-2 inference fails.
        """
        if not self._use_gpt2:
            return None

        try:
            import torch
        except ImportError:
            return None  # torch not available; skip feature gracefully

        try:
            self._load_gpt2()
        except DetectorInitError:
            return None

        assert self._gpt2_tokenizer is not None
        assert self._gpt2_model is not None

        try:
            tokens = self._gpt2_tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            )
            if tokens["input_ids"].shape[1] < 10:
                return None
            with torch.no_grad():
                outputs = self._gpt2_model(
                    **tokens, labels=tokens["input_ids"]
                )
            return float(outputs.loss.item())
        except Exception as exc:
            raise DetectionError(f"GPT-2 perplexity inference failed: {exc}") from exc

    def _compute_burstiness(self, sentence_lengths: list[int]) -> float | None:
        """Compute burstiness as coefficient of variation of sentence lengths.

        Lower burstiness indicates more uniform pacing, typical of AI text.

        Args:
            sentence_lengths: Word counts per sentence.

        Returns:
            Coefficient of variation (std/mean), or None if too few sentences.
        """
        if len(sentence_lengths) < MIN_SENTENCES_THRESHOLD:
            return None
        arr = np.array(sentence_lengths, dtype=np.float64)
        mean = float(arr.mean())
        if mean == 0.0:
            return 0.0
        return float(arr.std() / mean)

    def _compute_ttr(self, words: list[str]) -> float | None:
        """Compute type-token ratio (unique words / total words).

        Args:
            words: Lowercase word tokens.

        Returns:
            TTR in [0.0, 1.0], or None if word list is empty.
        """
        if not words:
            return None
        return len(set(words)) / len(words)

    def _compute_hapax_ratio(self, words: list[str]) -> float | None:
        """Compute ratio of words appearing exactly once (hapax legomena).

        Args:
            words: Lowercase word tokens.

        Returns:
            Hapax ratio in [0.0, 1.0], or None if word list is empty.
        """
        if not words:
            return None
        counts = Counter(words)
        hapax_count = sum(1 for c in counts.values() if c == 1)
        return hapax_count / len(words)

    def _compute_sentence_stats(
        self, sentence_lengths: list[int]
    ) -> tuple[float | None, float | None]:
        """Compute mean and variance of sentence lengths.

        Args:
            sentence_lengths: Word counts per sentence.

        Returns:
            Tuple of (mean_length, variance), or (None, None) if no sentences.
        """
        if not sentence_lengths:
            return None, None
        arr = np.array(sentence_lengths, dtype=np.float64)
        return float(arr.mean()), float(arr.var())

    def _compute_pos_trigram_entropy(self, text: str) -> float | None:
        """Compute Shannon entropy of POS tag trigrams.

        Lower entropy = more repetitive POS patterns = more AI-like.

        Args:
            text: Input prose text.

        Returns:
            Entropy in bits, or None if text too short for trigrams.

        Raises:
            DetectorInitError: If spaCy model is unavailable.
        """
        self._ensure_spacy()
        assert self._nlp is not None

        doc = self._nlp(text)
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        if len(pos_tags) < 3:
            return None
        trigrams = [
            (pos_tags[i], pos_tags[i + 1], pos_tags[i + 2])
            for i in range(len(pos_tags) - 2)
        ]
        counts = Counter(trigrams)
        total = len(trigrams)
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )
        return entropy

    def extract_features(self, text: str) -> dict[str, Any]:
        """Extract all seven statistical features from text.

        Args:
            text: Input prose text.

        Returns:
            Dict with keys: perplexity_proxy, burstiness, ttr, hapax_ratio,
            mean_sentence_length, sentence_length_variance, pos_trigram_entropy.
            Values are float | None (None when feature cannot be computed).

        Raises:
            DetectorInitError: If spaCy model is unavailable.
            DetectionError: If feature computation raises a runtime error.
        """
        self._ensure_spacy()
        assert self._nlp is not None

        doc = self._nlp(text)

        # Extract words (alphabetic tokens only)
        words = [token.text.lower() for token in doc if token.is_alpha]

        # Extract sentence lengths from spaCy sentence boundaries (via sentencizer)
        # spaCy with parser disabled uses rule-based sentence detection
        sentences = list(doc.sents) if doc.has_annotation("SENT_START") else []
        sentence_lengths: list[int] = [
            len([t for t in sent if t.is_alpha]) for sent in sentences
        ]

        mean_len, variance = self._compute_sentence_stats(sentence_lengths)

        return {
            "perplexity_proxy": self._compute_perplexity_proxy(text),
            "burstiness": self._compute_burstiness(sentence_lengths),
            "ttr": self._compute_ttr(words),
            "hapax_ratio": self._compute_hapax_ratio(words),
            "mean_sentence_length": mean_len,
            "sentence_length_variance": variance,
            "pos_trigram_entropy": self._compute_pos_trigram_entropy(text),
        }

    def score(self, text: str) -> float | None:
        """Compute a combined AI-detection score from statistical features.

        Returns None if the text is below MIN_WORDS_THRESHOLD.
        Falls back to normalized feature averaging (no XGBoost required).

        Args:
            text: Input prose text.

        Returns:
            Score in [0.0, 1.0] where 1.0 = very likely AI, or None if
            text is too short for reliable scoring.

        Raises:
            DetectorInitError: If spaCy model is unavailable.
            DetectionError: If feature computation fails.
        """
        word_count = len(text.split())
        if word_count < MIN_WORDS_THRESHOLD:
            return None

        features = self.extract_features(text)
        return _combine_features(features)


def _normalize_feature(
    value: float, low: float, high: float, invert: bool
) -> float:
    """Normalize a feature value to [0.0, 1.0].

    Args:
        value: Raw feature value.
        low: Minimum expected value (maps to 0.0, or 1.0 if inverted).
        high: Maximum expected value (maps to 1.0, or 0.0 if inverted).
        invert: If True, low value maps to 1.0 (AI-like direction).

    Returns:
        Normalized value clamped to [0.0, 1.0].
    """
    if high == low:
        return 0.5
    normalized = (value - low) / (high - low)
    normalized = max(0.0, min(1.0, normalized))
    return (1.0 - normalized) if invert else normalized


def _combine_features(features: dict[str, Any]) -> float:
    """Combine normalized features into a single score via averaging.

    Args:
        features: Dict of feature name -> raw float value (or None).

    Returns:
        Mean of available normalized feature scores, clamped to [0.0, 1.0].
        Returns 0.5 if no features are available.
    """
    scores: list[float] = []
    for feature_name, (low, high, invert) in _FEATURE_BOUNDS.items():
        raw = features.get(feature_name)
        if raw is None:
            continue
        normalized = _normalize_feature(float(raw), low, high, invert)
        scores.append(normalized)

    if not scores:
        return 0.5
    return float(np.mean(scores))
