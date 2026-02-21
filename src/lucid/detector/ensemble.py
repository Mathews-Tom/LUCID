"""Ensemble scoring and classification for multi-tier detection."""

from __future__ import annotations

from lucid.config import (
    DetectionThresholdsConfig,
    EnsembleWeightsConfig,
    EnsembleWeightsWithBinocularsConfig,
)


def compute_ensemble(
    roberta_score: float,
    statistical_score: float | None,
    binoculars_score: float | None,
    weights: EnsembleWeightsConfig | EnsembleWeightsWithBinocularsConfig,
) -> float:
    """Combine tier scores into a single ensemble detection score.

    Handles missing tiers by re-normalizing weights across active tiers.

    Args:
        roberta_score: Tier 1 RoBERTa classifier probability in [0, 1].
        statistical_score: Tier 2 statistical features probability, or None if unavailable.
        binoculars_score: Tier 3 Binoculars probability, or None if unavailable.
        weights: Weight configuration for score combination.

    Returns:
        Combined score in [0.0, 1.0].
    """
    # Determine which tiers are active
    active_weights: dict[str, float] = {}

    active_weights["roberta"] = weights.roberta
    if statistical_score is not None:
        active_weights["statistical"] = weights.statistical
    if binoculars_score is not None:
        active_weights["binoculars"] = weights.binoculars

    # Re-normalize weights to sum to 1.0
    total_weight = sum(active_weights.values())
    if total_weight <= 0:
        return roberta_score  # Fallback: only roberta available

    normalized_weights = {k: v / total_weight for k, v in active_weights.items()}

    # Compute weighted sum
    ensemble_score = normalized_weights["roberta"] * roberta_score
    if statistical_score is not None:
        ensemble_score += normalized_weights["statistical"] * statistical_score
    if binoculars_score is not None:
        ensemble_score += normalized_weights["binoculars"] * binoculars_score

    return ensemble_score


def classify(score: float, thresholds: DetectionThresholdsConfig) -> str:
    """Classify an ensemble score into human/ambiguous/ai_generated.

    Args:
        score: Ensemble detection score in [0, 1].
        thresholds: Classification boundary configuration.

    Returns:
        One of "human", "ambiguous", "ai_generated".
    """
    if score <= thresholds.human_max:
        return "human"
    if score >= thresholds.ai_min:
        return "ai_generated"
    return "ambiguous"
