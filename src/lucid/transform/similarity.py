"""Lightweight text similarity for the semantic gate in the search loop.

Uses stdlib-only methods (difflib + set operations) to approximate
embedding cosine similarity without model inference.  Sub-millisecond
per call, suitable for use inside the hot search loop.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from itertools import pairwise

_TOKEN_RE = re.compile(r"\[(?:MATH|TERM)_\d{3}\]|\d+(?:\.\d+)?%?|[a-z]+(?:-[a-z]+)*")


def _tokenize(text: str) -> list[str]:
    """Tokenize text into normalized lexical units."""
    return _TOKEN_RE.findall(text.lower())


def _jaccard(left: set[str], right: set[str]) -> float:
    """Compute Jaccard overlap, handling empty inputs."""
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def quick_similarity(original: str, transformed: str) -> float:
    """Compute a fast similarity score between original and transformed text.

    Combines structural similarity (SequenceMatcher) with vocabulary
    overlap (word-level Jaccard) to approximate embedding cosine.

    Args:
        original: Source text before transformation.
        transformed: Candidate text after LLM rewriting.

    Returns:
        Similarity score in [0.0, 1.0].  Higher means more similar.
    """
    if not original and not transformed:
        return 1.0
    if not original or not transformed:
        return 0.0

    orig_tokens = _tokenize(original)
    trans_tokens = _tokenize(transformed)
    if not orig_tokens and not trans_tokens:
        return 1.0
    if not orig_tokens or not trans_tokens:
        return 0.0

    normalized_original = " ".join(orig_tokens)
    normalized_transformed = " ".join(trans_tokens)

    # Blend token-order and normalized character similarity to stay robust to
    # punctuation changes while still rewarding close paraphrases.
    raw_char_ratio = SequenceMatcher(None, original.lower(), transformed.lower()).ratio()
    token_seq_ratio = SequenceMatcher(None, orig_tokens, trans_tokens).ratio()
    char_seq_ratio = SequenceMatcher(
        None, normalized_original, normalized_transformed
    ).ratio()

    # Unigram and bigram overlap preserve terminology and local phrasing.
    orig_words = set(orig_tokens)
    trans_words = set(trans_tokens)
    unigram_overlap = _jaccard(orig_words, trans_words)

    orig_bigrams = set(pairwise(orig_tokens))
    trans_bigrams = set(pairwise(trans_tokens))
    bigram_overlap = _jaccard(orig_bigrams, trans_bigrams)

    return (
        0.28 * raw_char_ratio
        + 0.27 * char_seq_ratio
        + 0.18 * token_seq_ratio
        + 0.17 * unigram_overlap
        + 0.10 * bigram_overlap
    )
