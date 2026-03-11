"""Feature dataclasses, configuration, and reference data for statistical detection.

Defines typed result containers for each feature category, canonical word lists
for distribution analysis, and normalization bounds for score combination.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass


class FeatureTier(enum.Enum):
    """Controls which statistical features are computed."""

    FAST = "fast"  # text-only, no ML models (~10ms)
    STANDARD = "standard"  # + GPT-2 perplexity (~200ms)
    DEEP = "deep"  # + dependency parser + sentence-transformers (~500ms)


@dataclass(frozen=True, slots=True)
class PerplexityResult:
    """Per-sentence and aggregate perplexity statistics."""

    mean: float
    std: float
    per_sentence: tuple[float, ...]
    min: float
    max: float


@dataclass(frozen=True, slots=True)
class TokenProbDistribution:
    """Distribution statistics of per-token log-probabilities."""

    mean_log_prob: float
    std_log_prob: float
    skewness: float
    kurtosis: float
    tail_ratio: float  # fraction of tokens with prob below 10th percentile


@dataclass(frozen=True, slots=True)
class NgramResult:
    """Word-level n-gram frequency statistics."""

    bigram_entropy: float
    trigram_rarity: float  # fraction of trigrams appearing exactly once
    pos_trigram_entropy: float  # existing feature, kept


@dataclass(frozen=True, slots=True)
class FunctionWordResult:
    """Function word distribution analysis."""

    entropy: float  # Shannon entropy of function word frequencies
    divergence: float  # KL-divergence from reference human distribution
    total_count: int


@dataclass(frozen=True, slots=True)
class ClauseDensityResult:
    """Clause density per sentence."""

    mean: float
    variance: float


@dataclass(frozen=True, slots=True)
class StructuralSymmetryResult:
    """Structural regularity metrics."""

    sentence_uniformity: float  # 0=irregular, 1=perfectly uniform
    paragraph_length_variance: float | None


@dataclass(frozen=True, slots=True)
class SentenceEntropyResult:
    """Per-sentence information density."""

    mean_entropy: float
    entropy_variance: float
    per_sentence: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class InterSentenceSimilarityResult:
    """Cosine similarity between adjacent sentence embeddings."""

    mean_adjacent_similarity: float
    similarity_variance: float
    max_similarity: float
    min_similarity: float


@dataclass(frozen=True, slots=True)
class TransitionResult:
    """Transition phrase usage statistics."""

    density_per_1000: float  # transitions per 1000 words
    diversity: float  # unique transitions / total transitions
    total_count: int


# Canonical function word list for distribution analysis
FUNCTION_WORDS: tuple[str, ...] = (
    "the", "a", "an", "and", "but", "or", "nor", "for", "yet", "so",
    "in", "on", "at", "to", "of", "by", "with", "from", "as", "into",
    "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    "because", "although", "however", "therefore", "moreover", "furthermore",
    "if", "then", "that", "which", "who", "whom", "whose",
)

# Canonical transition phrase list for frequency analysis
TRANSITION_PHRASES: tuple[str, ...] = (
    "however", "furthermore", "moreover", "additionally", "consequently",
    "therefore", "nevertheless", "nonetheless", "meanwhile", "subsequently",
    "in addition", "as a result", "on the other hand", "in contrast",
    "for example", "for instance", "in particular", "in fact", "of course",
    "in conclusion", "to summarize", "in summary", "above all", "after all",
    "as such", "that said", "in other words", "to that end", "by contrast",
)

# Reference human function word distribution (empirical baseline)
# Frequencies as proportions (sum to ~1.0) over FUNCTION_WORDS
# Derived from balanced human writing corpora
HUMAN_FUNCTION_WORD_BASELINE: dict[str, float] = {
    "the": 0.180, "a": 0.065, "an": 0.015, "and": 0.085, "but": 0.020,
    "or": 0.015, "nor": 0.001, "for": 0.025, "yet": 0.003, "so": 0.012,
    "in": 0.055, "on": 0.020, "at": 0.015, "to": 0.070, "of": 0.065,
    "by": 0.015, "with": 0.020, "from": 0.015, "as": 0.020, "into": 0.005,
    "is": 0.035, "are": 0.015, "was": 0.020, "were": 0.008, "be": 0.015,
    "been": 0.008, "being": 0.003,
    "have": 0.015, "has": 0.010, "had": 0.010, "do": 0.008, "does": 0.003, "did": 0.005,
    "will": 0.008, "would": 0.010, "shall": 0.001, "should": 0.005,
    "may": 0.005, "might": 0.003, "can": 0.010, "could": 0.008, "must": 0.004,
    "because": 0.008, "although": 0.003, "however": 0.005, "therefore": 0.003,
    "moreover": 0.001, "furthermore": 0.001,
    "if": 0.012, "then": 0.008, "that": 0.050, "which": 0.015,
    "who": 0.008, "whom": 0.001, "whose": 0.002,
}

# Expanded feature normalization bounds: (low, high, invert)
# invert=True means low value => AI-like (score maps to 1.0 when feature is at low bound)
FEATURE_BOUNDS: dict[str, tuple[float, float, bool]] = {
    # Language model statistics
    "lm_perplexity_mean": (10.0, 200.0, True),
    "lm_burstiness": (0.0, 50.0, True),
    "lm_token_prob_tail_ratio": (0.0, 0.3, True),
    # Stylometric features
    "style_ttr": (0.2, 0.9, True),
    "style_hapax_ratio": (0.1, 0.6, True),
    "style_function_word_divergence": (0.0, 0.5, False),
    "style_clause_density_variance": (0.0, 1.5, True),
    # Structural features
    "struct_sentence_length_variance": (0.0, 200.0, True),
    "struct_symmetry_score": (0.0, 1.0, False),
    # Discourse features
    "disc_sentence_entropy_variance": (0.0, 2.0, True),
    "disc_bigram_entropy": (2.0, 8.0, True),
    "disc_trigram_rarity": (0.3, 0.9, True),
    "disc_pos_trigram_entropy": (1.0, 6.0, True),
    "disc_transition_density": (0.0, 15.0, False),
    "disc_transition_diversity": (0.0, 1.0, True),
}
