"""Detection explainability -- human-readable evidence from feature analysis."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lucid.detector.features import FEATURE_BOUNDS


@dataclass(frozen=True, slots=True)
class FeatureExplanation:
    """Explanation for a single feature's contribution."""

    feature_name: str
    raw_value: float
    normalized_score: float  # 0-1, where 1 = strongly AI-like
    direction: str  # "ai_like" or "human_like"
    description: str  # human-readable explanation
    category: str  # full category name


@dataclass(frozen=True, slots=True)
class DetectionExplanation:
    """Complete explanation of a detection result."""

    overall_score: float
    classification: str
    feature_explanations: tuple[FeatureExplanation, ...]
    top_signals: tuple[str, ...]  # top 3 most contributing features
    summary: str  # natural language summary


# Feature descriptions for human-readable output
_FEATURE_DESCRIPTIONS: dict[str, str] = {
    "lm_perplexity_mean": "Text predictability (low = AI-like)",
    "lm_burstiness": "Variability in predictability across sentences",
    "lm_token_prob_tail_ratio": "Proportion of unexpected word choices",
    "style_ttr": "Vocabulary diversity (unique words / total words)",
    "style_hapax_ratio": "Proportion of words used only once",
    "style_function_word_divergence": "Deviation from typical function word patterns",
    "style_clause_density_variance": "Variation in sentence complexity",
    "struct_sentence_length_variance": "Variation in sentence lengths",
    "struct_symmetry_score": "Structural regularity of the text",
    "disc_sentence_entropy_variance": "Variation in information density across sentences",
    "disc_bigram_entropy": "Diversity of two-word patterns",
    "disc_trigram_rarity": "Proportion of unique three-word patterns",
    "disc_pos_trigram_entropy": "Diversity of grammatical patterns",
    "disc_transition_density": "Frequency of transition phrases (however, moreover, etc.)",
    "disc_transition_diversity": "Variety of transition phrases used",
}

_FEATURE_CATEGORIES: dict[str, str] = {
    "lm_": "Language Model Statistics",
    "style_": "Stylometric Features",
    "struct_": "Structural Features",
    "disc_": "Discourse Features",
}


def _get_category(feature_name: str) -> str:
    for prefix, cat in _FEATURE_CATEGORIES.items():
        if feature_name.startswith(prefix):
            return cat
    return "Unknown"


class DetectionExplainer:
    """Generate human-readable explanations of detection results."""

    def explain(self, result: Any) -> dict[str, Any]:
        """Generate explanation dict from a DetectionResult.

        Args:
            result: DetectionResult with feature_details populated.

        Returns:
            Dict with 'explanations', 'top_signals', and 'summary' keys.
        """
        features = result.feature_details or {}
        if not features:
            return {
                "explanations": [],
                "top_signals": [],
                "summary": "No features available",
            }

        explanations = self._explain_features(features)

        # Sort by normalized score (most AI-like first)
        sorted_expl = sorted(
            explanations, key=lambda e: e.normalized_score, reverse=True
        )
        top_signals = tuple(e.feature_name for e in sorted_expl[:3])

        summary = self._generate_summary(result.classification, sorted_expl)

        full = DetectionExplanation(
            overall_score=result.ensemble_score,
            classification=result.classification,
            feature_explanations=tuple(sorted_expl),
            top_signals=top_signals,
            summary=summary,
        )

        return {
            "explanations": [
                {
                    "feature": e.feature_name,
                    "raw_value": e.raw_value,
                    "normalized_score": e.normalized_score,
                    "direction": e.direction,
                    "description": e.description,
                    "category": e.category,
                }
                for e in full.feature_explanations
            ],
            "top_signals": list(full.top_signals),
            "summary": full.summary,
        }

    def _explain_features(
        self, features: dict[str, Any]
    ) -> list[FeatureExplanation]:
        explanations: list[FeatureExplanation] = []
        for name, raw in features.items():
            if name not in FEATURE_BOUNDS or raw is None:
                continue
            low, high, invert = FEATURE_BOUNDS[name]
            # Normalize
            if high == low:
                normalized = 0.5
            else:
                normalized = (float(raw) - low) / (high - low)
                normalized = max(0.0, min(1.0, normalized))
                if invert:
                    normalized = 1.0 - normalized

            direction = "ai_like" if normalized > 0.5 else "human_like"
            description = _FEATURE_DESCRIPTIONS.get(name, name)
            category = _get_category(name)

            explanations.append(
                FeatureExplanation(
                    feature_name=name,
                    raw_value=float(raw),
                    normalized_score=normalized,
                    direction=direction,
                    description=description,
                    category=category,
                )
            )
        return explanations

    def _generate_summary(
        self, classification: str, explanations: list[FeatureExplanation]
    ) -> str:
        if not explanations:
            return f"Text classified as {classification} with no feature details available."

        ai_count = sum(1 for e in explanations if e.direction == "ai_like")

        top_3 = explanations[:3]
        signal_parts = [
            f"{e.description.lower()} ({e.direction.replace('_', '-')})"
            for e in top_3
        ]

        return (
            f"Text classified as {classification}. "
            f"{ai_count} of {len(explanations)} features indicate AI-generated content. "
            f"Top signals: {', '.join(signal_parts)}."
        )
