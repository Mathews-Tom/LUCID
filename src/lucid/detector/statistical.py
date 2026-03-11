"""Statistical feature extractor for AI content detection (Tier 2 detector).

Extracts 12+ linguistic features across four categories and combines them
into a detection score. GPT-2 perplexity proxy is optional (requires torch).
Deep features (clause density, semantic similarity) require additional models.
"""

from __future__ import annotations

import math
import threading
from collections import Counter
from typing import TYPE_CHECKING, Any

import numpy as np

from lucid.detector import DetectionError, DetectorInitError
from lucid.detector.features import (
    FEATURE_BOUNDS,
    FUNCTION_WORDS,
    HUMAN_FUNCTION_WORD_BASELINE,
    TRANSITION_PHRASES,
    ClauseDensityResult,
    FunctionWordResult,
    InterSentenceSimilarityResult,
    NgramResult,
    PerplexityResult,
    SentenceEntropyResult,
    StructuralSymmetryResult,
    TokenProbDistribution,
    TransitionResult,
)

if TYPE_CHECKING:
    import spacy
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MIN_WORDS_THRESHOLD = 50
MIN_SENTENCES_THRESHOLD = 3
_MIN_SENTENCE_TOKENS = 10

_spacy_lock = threading.Lock()
_gpt2_lock = threading.Lock()
_spacy_deep_lock = threading.Lock()


def _skewness(arr: np.ndarray) -> float:
    """Compute Fisher-Pearson skewness without scipy."""
    std = float(arr.std())
    if std == 0:
        return 0.0
    mean = float(arr.mean())
    return float(((arr - mean) ** 3).mean() / (std ** 3))


def _kurtosis(arr: np.ndarray) -> float:
    """Compute excess kurtosis without scipy."""
    std = float(arr.std())
    if std == 0:
        return 0.0
    mean = float(arr.mean())
    return float(((arr - mean) ** 4).mean() / (std ** 4) - 3.0)


class StatisticalDetector:
    """Tier 2 AI content detector using statistical linguistic features.

    Extracts 12+ features across four categories (language model, stylometric,
    structural, discourse) and combines them into a score in [0.0, 1.0].
    Requires spaCy (en_core_web_sm) for POS tagging. GPT-2 perplexity proxy
    is used when torch is available.

    Args:
        use_gpt2_perplexity: Whether to compute GPT-2 perplexity features.
            Requires torch and transformers. Defaults to True (lazy-loads).
        use_deep_features: Whether to compute deep features (clause density,
            semantic similarity). Requires spaCy with parser enabled.
    """

    def __init__(
        self,
        use_gpt2_perplexity: bool = True,
        use_deep_features: bool = False,
    ) -> None:
        self._use_gpt2 = use_gpt2_perplexity
        self._use_deep_features = use_deep_features
        self._nlp: spacy.language.Language | None = None
        self._nlp_deep: spacy.language.Language | None = None
        self._gpt2_model: GPT2LMHeadModel | None = None
        self._gpt2_tokenizer: GPT2TokenizerFast | None = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_spacy(self) -> None:
        """Lazy-load spaCy model with sentencizer (thread-safe).

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
                self._nlp.add_pipe("sentencizer")
            except OSError as exc:
                raise DetectorInitError(
                    "spaCy model not found. Install with: "
                    "python -m spacy download en_core_web_sm"
                ) from exc

    def _ensure_spacy_with_parser(self) -> None:
        """Lazy-load spaCy model with dependency parser enabled (thread-safe).

        Used for deep-tier features (clause density).

        Raises:
            DetectorInitError: If spaCy or the en_core_web_sm model is missing.
        """
        if self._nlp_deep is not None:
            return
        with _spacy_deep_lock:
            if self._nlp_deep is not None:
                return
            try:
                import spacy as _spacy
            except ImportError as exc:
                raise DetectorInitError(
                    "spacy is required. Install with: uv add spacy"
                ) from exc
            try:
                self._nlp_deep = _spacy.load(
                    "en_core_web_sm", disable=["ner", "lemmatizer"]
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

    # ------------------------------------------------------------------
    # Category 1: Language model statistics
    # ------------------------------------------------------------------

    def _compute_sentence_nll(self, sentence: str) -> float | None:
        """Compute mean NLL for a single sentence via GPT-2.

        Returns None if the sentence is too short (< _MIN_SENTENCE_TOKENS tokens).
        """
        import torch

        assert self._gpt2_tokenizer is not None
        assert self._gpt2_model is not None

        tokens = self._gpt2_tokenizer(
            sentence, return_tensors="pt", truncation=True, max_length=512
        )
        if tokens["input_ids"].shape[1] < _MIN_SENTENCE_TOKENS:
            return None
        with torch.no_grad():
            outputs = self._gpt2_model(**tokens, labels=tokens["input_ids"])
        return float(outputs.loss.item())

    def _compute_perplexity(
        self, text: str, sentence_texts: list[str]
    ) -> PerplexityResult | None:
        """Compute per-sentence and aggregate perplexity via GPT-2.

        Processes each sentence individually to obtain per-sentence NLL values.
        Sentences shorter than _MIN_SENTENCE_TOKENS tokens are skipped.

        Args:
            text: Full input text (unused, kept for API consistency).
            sentence_texts: List of sentence strings.

        Returns:
            PerplexityResult with per-sentence and aggregate stats, or None
            if GPT-2 is unavailable or insufficient sentences are scorable.
        """
        if not self._use_gpt2:
            return None

        try:
            import torch  # noqa: F401
        except ImportError:
            return None

        try:
            self._load_gpt2()
        except DetectorInitError:
            return None

        try:
            per_sentence: list[float] = []
            for sent in sentence_texts:
                nll = self._compute_sentence_nll(sent)
                if nll is not None:
                    per_sentence.append(nll)

            if len(per_sentence) < MIN_SENTENCES_THRESHOLD:
                return None

            arr = np.array(per_sentence)
            return PerplexityResult(
                mean=float(arr.mean()),
                std=float(arr.std()),
                per_sentence=tuple(per_sentence),
                min=float(arr.min()),
                max=float(arr.max()),
            )
        except Exception as exc:
            raise DetectionError(
                f"GPT-2 perplexity inference failed: {exc}"
            ) from exc

    def _compute_token_prob_distribution(
        self, text: str
    ) -> TokenProbDistribution | None:
        """Compute distribution statistics of per-token log-probabilities.

        Uses GPT-2 logits to extract per-token log-probabilities via
        log_softmax, then computes distributional statistics.

        Args:
            text: Input prose text.

        Returns:
            TokenProbDistribution or None if GPT-2 is unavailable or text
            is too short.
        """
        if not self._use_gpt2:
            return None

        try:
            import torch
        except ImportError:
            return None

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
            input_ids = tokens["input_ids"]
            if input_ids.shape[1] < _MIN_SENTENCE_TOKENS:
                return None

            with torch.no_grad():
                outputs = self._gpt2_model(**tokens)

            # logits shape: (1, seq_len, vocab_size)
            # For autoregressive: logits[t] predicts token[t+1]
            logits = outputs.logits[0, :-1, :]  # (seq_len-1, vocab_size)
            target_ids = input_ids[0, 1:]  # (seq_len-1,)

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            token_log_probs = log_probs.gather(
                1, target_ids.unsqueeze(1)
            ).squeeze(1)

            lp = token_log_probs.numpy().astype(np.float64)
            if len(lp) < 2:
                return None

            mean_lp = float(lp.mean())
            std_lp = float(lp.std())
            skew = _skewness(lp)
            kurt = _kurtosis(lp)

            # Tail ratio: fraction below 10th percentile
            threshold = float(np.percentile(lp, 10))
            tail_ratio = float(np.mean(lp < threshold))

            return TokenProbDistribution(
                mean_log_prob=mean_lp,
                std_log_prob=std_lp,
                skewness=skew,
                kurtosis=kurt,
                tail_ratio=tail_ratio,
            )
        except Exception as exc:
            raise DetectionError(
                f"GPT-2 token probability computation failed: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Category 2: Stylometric features
    # ------------------------------------------------------------------

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

    def _compute_function_word_distribution(
        self, words: list[str]
    ) -> FunctionWordResult | None:
        """Compute function word frequency distribution and KL-divergence.

        Args:
            words: Lowercase word tokens.

        Returns:
            FunctionWordResult or None if too few words or no function words found.
        """
        if len(words) < MIN_WORDS_THRESHOLD:
            return None

        counts = {w: 0 for w in FUNCTION_WORDS}
        for word in words:
            if word in counts:
                counts[word] += 1
        total = sum(counts.values())
        if total == 0:
            return None

        freqs = {w: c / total for w, c in counts.items()}

        # Shannon entropy
        eps = 1e-10
        entropy = -sum(
            f * math.log2(f + eps) for f in freqs.values() if f > 0
        )

        # KL-divergence from human baseline
        kl_div = 0.0
        for w in FUNCTION_WORDS:
            p = freqs.get(w, 0.0) + eps
            q = HUMAN_FUNCTION_WORD_BASELINE.get(w, eps) + eps
            kl_div += p * math.log(p / q)

        return FunctionWordResult(
            entropy=entropy, divergence=kl_div, total_count=total
        )

    def _compute_clause_density(self, text: str) -> ClauseDensityResult | None:
        """Compute clause density per sentence using dependency parsing.

        Counts clauses by identifying verbs with clause-root dependency labels
        (ROOT, advcl, relcl, ccomp, xcomp, conj with verb POS).

        Gated behind use_deep_features flag (deep tier only).

        Args:
            text: Input prose text.

        Returns:
            ClauseDensityResult or None if deep features are disabled.
        """
        if not self._use_deep_features:
            return None

        self._ensure_spacy_with_parser()
        assert self._nlp_deep is not None

        doc = self._nlp_deep(text)
        clause_dep_labels = {"ROOT", "advcl", "relcl", "ccomp", "xcomp"}
        verb_pos = {"VERB", "AUX"}

        clause_counts: list[int] = []
        for sent in doc.sents:
            count = 0
            for token in sent:
                if token.dep_ in clause_dep_labels and token.pos_ in verb_pos:
                    count += 1
                elif (
                    token.dep_ == "conj"
                    and token.pos_ in verb_pos
                ):
                    count += 1
            clause_counts.append(count)

        if len(clause_counts) < MIN_SENTENCES_THRESHOLD:
            return None

        arr = np.array(clause_counts, dtype=np.float64)
        return ClauseDensityResult(
            mean=float(arr.mean()), variance=float(arr.var())
        )

    # ------------------------------------------------------------------
    # Category 3: Structural features
    # ------------------------------------------------------------------

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

    def _compute_structural_symmetry(
        self, sentence_lengths: list[int], text: str
    ) -> StructuralSymmetryResult | None:
        """Compute structural regularity metrics.

        Sentence uniformity measures how consistent sentence lengths are
        (high = uniform = AI-like). Paragraph length variance captures
        paragraph-level structural regularity.

        Args:
            sentence_lengths: Word counts per sentence.
            text: Full input text (for paragraph splitting).

        Returns:
            StructuralSymmetryResult or None if too few sentences.
        """
        if len(sentence_lengths) < MIN_SENTENCES_THRESHOLD:
            return None
        arr = np.array(sentence_lengths, dtype=np.float64)
        mean_len = float(arr.mean())
        if mean_len == 0:
            return None
        uniformity = 1.0 - min(1.0, float(arr.std()) / mean_len)

        # Paragraph length variance
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        para_var: float | None = None
        if len(paragraphs) > 1:
            para_lengths = [len(p.split(".")) for p in paragraphs]
            para_var = float(np.var(para_lengths))

        return StructuralSymmetryResult(
            sentence_uniformity=uniformity,
            paragraph_length_variance=para_var,
        )

    # ------------------------------------------------------------------
    # Category 4: Discourse features
    # ------------------------------------------------------------------

    def _compute_pos_trigram_entropy_from_tags(
        self, pos_tags: list[str]
    ) -> float | None:
        """Compute Shannon entropy of POS tag trigrams from pre-computed tags.

        Args:
            pos_tags: List of POS tag strings.

        Returns:
            Entropy in bits, or None if fewer than 3 tags.
        """
        if len(pos_tags) < 3:
            return None
        trigrams = [
            (pos_tags[i], pos_tags[i + 1], pos_tags[i + 2])
            for i in range(len(pos_tags) - 2)
        ]
        counts = Counter(trigrams)
        total = len(trigrams)
        return -sum(
            (c / total) * math.log2(c / total) for c in counts.values()
        )

    def _compute_ngram_distribution(
        self, words: list[str], pos_tags: list[str]
    ) -> NgramResult | None:
        """Compute word-level n-gram and POS trigram statistics.

        Args:
            words: Lowercase word tokens.
            pos_tags: POS tag strings (non-whitespace tokens).

        Returns:
            NgramResult or None if too few words for trigrams.
        """
        if len(words) < 6:
            return None

        # Word bigrams
        bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
        bigram_counts = Counter(bigrams)
        total_bigrams = len(bigrams)
        bigram_entropy = -sum(
            (c / total_bigrams) * math.log2(c / total_bigrams)
            for c in bigram_counts.values()
        )

        # Word trigrams
        trigrams = [
            (words[i], words[i + 1], words[i + 2])
            for i in range(len(words) - 2)
        ]
        trigram_counts = Counter(trigrams)
        trigram_rarity = sum(
            1 for c in trigram_counts.values() if c == 1
        ) / max(len(trigram_counts), 1)

        # POS trigram entropy
        pos_trigram_ent = self._compute_pos_trigram_entropy_from_tags(pos_tags)
        if pos_trigram_ent is None:
            pos_trigram_ent = 0.0

        return NgramResult(
            bigram_entropy=bigram_entropy,
            trigram_rarity=trigram_rarity,
            pos_trigram_entropy=pos_trigram_ent,
        )

    def _compute_sentence_entropy(
        self, sentences: list[list[str]]
    ) -> SentenceEntropyResult | None:
        """Compute per-sentence word distribution entropy.

        Uniform entropy across sentences is characteristic of AI-generated text.

        Args:
            sentences: List of word lists per sentence.

        Returns:
            SentenceEntropyResult or None if too few sentences.
        """
        if len(sentences) < MIN_SENTENCES_THRESHOLD:
            return None
        entropies: list[float] = []
        for sent_words in sentences:
            if not sent_words:
                continue
            counts = Counter(sent_words)
            total = len(sent_words)
            ent = -sum(
                (c / total) * math.log2(c / total) for c in counts.values()
            )
            entropies.append(ent)
        if len(entropies) < MIN_SENTENCES_THRESHOLD:
            return None
        arr = np.array(entropies)
        return SentenceEntropyResult(
            mean_entropy=float(arr.mean()),
            entropy_variance=float(arr.var()),
            per_sentence=tuple(entropies),
        )

    def _compute_inter_sentence_similarity(
        self, sentences: list[str]
    ) -> InterSentenceSimilarityResult | None:
        """Compute cosine similarity between adjacent sentence embeddings.

        Requires sentence-transformers model (deep tier only).
        Implementation deferred to Phase 3 when ModelManager shares embedding model.
        """
        # TODO: implement in Phase 3 with shared sentence-transformers model
        return None

    def _compute_transition_frequency(
        self, text: str, word_count: int
    ) -> TransitionResult | None:
        """Compute transition phrase usage density and diversity.

        Args:
            text: Input prose text.
            word_count: Total word count for density normalization.

        Returns:
            TransitionResult or None if too few words.
        """
        if word_count < MIN_WORDS_THRESHOLD:
            return None

        text_lower = text.lower()
        found: list[str] = []
        for phrase in TRANSITION_PHRASES:
            count = text_lower.count(phrase)
            found.extend([phrase] * count)
        total = len(found)
        if total == 0:
            return TransitionResult(
                density_per_1000=0.0, diversity=0.0, total_count=0
            )
        unique = len(set(found))
        density = (total / word_count) * 1000
        diversity = unique / total
        return TransitionResult(
            density_per_1000=density,
            diversity=diversity,
            total_count=total,
        )

    # ------------------------------------------------------------------
    # Feature extraction and scoring
    # ------------------------------------------------------------------

    def extract_features(self, text: str) -> dict[str, Any]:
        """Extract all statistical features from text.

        Returns dict with keys prefixed by category (lm_, style_, struct_,
        disc_). Values are float or None when a feature cannot be computed.

        Args:
            text: Input prose text.

        Returns:
            Feature dict with keys matching FEATURE_BOUNDS.

        Raises:
            DetectorInitError: If spaCy model is unavailable.
            DetectionError: If feature computation raises a runtime error.
        """
        self._ensure_spacy()
        assert self._nlp is not None

        doc = self._nlp(text)

        words = [token.text.lower() for token in doc if token.is_alpha]
        pos_tags = [token.pos_ for token in doc if not token.is_space]
        sentences = (
            list(doc.sents) if doc.has_annotation("SENT_START") else []
        )
        sentence_lengths = [
            len([t for t in sent if t.is_alpha]) for sent in sentences
        ]
        sentence_texts = [sent.text for sent in sentences]
        sentence_words = [
            [t.text.lower() for t in sent if t.is_alpha] for sent in sentences
        ]

        features: dict[str, Any] = {}

        # Language model statistics
        perplexity = self._compute_perplexity(text, sentence_texts)
        if perplexity is not None:
            features["lm_perplexity_mean"] = perplexity.mean
            features["lm_burstiness"] = perplexity.std

        token_prob = self._compute_token_prob_distribution(text)
        if token_prob is not None:
            features["lm_token_prob_tail_ratio"] = token_prob.tail_ratio

        # Stylometric features
        ttr = self._compute_ttr(words)
        if ttr is not None:
            features["style_ttr"] = ttr
        hapax = self._compute_hapax_ratio(words)
        if hapax is not None:
            features["style_hapax_ratio"] = hapax

        fw = self._compute_function_word_distribution(words)
        if fw is not None:
            features["style_function_word_divergence"] = fw.divergence

        clause = self._compute_clause_density(text)
        if clause is not None:
            features["style_clause_density_variance"] = clause.variance

        # Structural features
        _, variance = self._compute_sentence_stats(sentence_lengths)
        if variance is not None:
            features["struct_sentence_length_variance"] = variance

        symmetry = self._compute_structural_symmetry(sentence_lengths, text)
        if symmetry is not None:
            features["struct_symmetry_score"] = symmetry.sentence_uniformity

        # Discourse features
        sent_ent = self._compute_sentence_entropy(sentence_words)
        if sent_ent is not None:
            features["disc_sentence_entropy_variance"] = sent_ent.entropy_variance

        ngram = self._compute_ngram_distribution(words, pos_tags)
        if ngram is not None:
            features["disc_bigram_entropy"] = ngram.bigram_entropy
            features["disc_trigram_rarity"] = ngram.trigram_rarity
            features["disc_pos_trigram_entropy"] = ngram.pos_trigram_entropy

        transition = self._compute_transition_frequency(text, len(words))
        if transition is not None:
            features["disc_transition_density"] = transition.density_per_1000
            features["disc_transition_diversity"] = transition.diversity

        # Deep features (gated)
        sim = self._compute_inter_sentence_similarity(sentence_texts)
        if sim is not None:
            features["disc_semantic_sim_mean"] = sim.mean_adjacent_similarity
            features["disc_semantic_sim_variance"] = sim.similarity_variance

        return features

    def score(self, text: str) -> float | None:
        """Compute a combined AI-detection score from statistical features.

        Returns None if the text is below MIN_WORDS_THRESHOLD.

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

    Uses FEATURE_BOUNDS from features module for normalization parameters.

    Args:
        features: Dict of feature name -> raw float value (or None).

    Returns:
        Mean of available normalized feature scores, clamped to [0.0, 1.0].
        Returns 0.5 if no features are available.
    """
    scores: list[float] = []
    for feature_name, (low, high, invert) in FEATURE_BOUNDS.items():
        raw = features.get(feature_name)
        if raw is None:
            continue
        normalized = _normalize_feature(float(raw), low, high, invert)
        scores.append(normalized)

    if not scores:
        return 0.5
    return float(np.mean(scores))
