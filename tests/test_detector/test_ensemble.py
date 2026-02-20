"""Ensemble scorer tests."""

from __future__ import annotations

import pytest

from lucid.config import (
    DetectionThresholdsConfig,
    EnsembleWeightsConfig,
    EnsembleWeightsWithBinocularsConfig,
)
from lucid.detector.ensemble import classify, compute_ensemble


class TestComputeEnsemble:
    """Tests for compute_ensemble function."""

    def test_two_tier_basic(self) -> None:
        """Test 2-tier ensemble: roberta=0.8, statistical=0.6."""
        weights = EnsembleWeightsConfig(roberta=0.7, statistical=0.3)
        score = compute_ensemble(
            roberta_score=0.8,
            statistical_score=0.6,
            binoculars_score=None,
            weights=weights,
        )
        expected = 0.7 * 0.8 + 0.3 * 0.6
        assert abs(score - expected) < 1e-6

    def test_three_tier_basic(self) -> None:
        """Test 3-tier ensemble: roberta=0.8, statistical=0.6, binoculars=0.9."""
        weights = EnsembleWeightsWithBinocularsConfig(
            roberta=0.4, statistical=0.15, binoculars=0.45
        )
        score = compute_ensemble(
            roberta_score=0.8,
            statistical_score=0.6,
            binoculars_score=0.9,
            weights=weights,
        )
        expected = 0.4 * 0.8 + 0.15 * 0.6 + 0.45 * 0.9
        assert abs(score - expected) < 1e-6

    def test_statistical_none_only_roberta(self) -> None:
        """Test when statistical=None, should return roberta_score."""
        weights = EnsembleWeightsConfig(roberta=0.7, statistical=0.3)
        score = compute_ensemble(
            roberta_score=0.75,
            statistical_score=None,
            binoculars_score=None,
            weights=weights,
        )
        assert score == 0.75

    def test_both_secondary_tiers_none(self) -> None:
        """Test when both statistical and binoculars are None."""
        weights = EnsembleWeightsWithBinocularsConfig(
            roberta=0.4, statistical=0.15, binoculars=0.45
        )
        score = compute_ensemble(
            roberta_score=0.82,
            statistical_score=None,
            binoculars_score=None,
            weights=weights,
        )
        assert score == 0.82

    def test_statistical_none_binoculars_present(self) -> None:
        """Test 2-tier with roberta and binoculars, statistical=None."""
        weights = EnsembleWeightsWithBinocularsConfig(
            roberta=0.4, statistical=0.15, binoculars=0.45
        )
        score = compute_ensemble(
            roberta_score=0.6,
            statistical_score=None,
            binoculars_score=0.8,
            weights=weights,
        )
        # Normalized: roberta=0.4/(0.4+0.45)=0.4705, binoculars=0.45/0.85=0.5294
        expected = (0.4 / 0.85) * 0.6 + (0.45 / 0.85) * 0.8
        assert abs(score - expected) < 1e-6

    def test_extreme_values_zero(self) -> None:
        """Test ensemble with score=0.0."""
        weights = EnsembleWeightsConfig(roberta=0.7, statistical=0.3)
        score = compute_ensemble(
            roberta_score=0.0,
            statistical_score=0.0,
            binoculars_score=None,
            weights=weights,
        )
        assert score == 0.0

    def test_extreme_values_one(self) -> None:
        """Test ensemble with score=1.0."""
        weights = EnsembleWeightsConfig(roberta=0.7, statistical=0.3)
        score = compute_ensemble(
            roberta_score=1.0,
            statistical_score=1.0,
            binoculars_score=None,
            weights=weights,
        )
        assert score == 1.0

    def test_mixed_extreme_values(self) -> None:
        """Test ensemble with mixed extreme values."""
        weights = EnsembleWeightsConfig(roberta=0.5, statistical=0.5)
        score = compute_ensemble(
            roberta_score=1.0,
            statistical_score=0.0,
            binoculars_score=None,
            weights=weights,
        )
        expected = 0.5 * 1.0 + 0.5 * 0.0
        assert score == expected

    def test_weight_normalization_accuracy(self) -> None:
        """Test that normalized weights sum to 1.0 internally."""
        weights = EnsembleWeightsWithBinocularsConfig(
            roberta=0.4, statistical=0.15, binoculars=0.45
        )
        # When all three tiers present, total=1.0
        score = compute_ensemble(
            roberta_score=0.5,
            statistical_score=0.5,
            binoculars_score=0.5,
            weights=weights,
        )
        # All scores identical, result should be same
        assert score == 0.5


class TestClassify:
    """Tests for classify function."""

    def test_classify_human_boundary(self) -> None:
        """Test score <= human_max returns 'human'."""
        thresholds = DetectionThresholdsConfig(human_max=0.30, ai_min=0.65)
        assert classify(0.30, thresholds) == "human"
        assert classify(0.25, thresholds) == "human"
        assert classify(0.0, thresholds) == "human"

    def test_classify_ai_boundary(self) -> None:
        """Test score >= ai_min returns 'ai_generated'."""
        thresholds = DetectionThresholdsConfig(human_max=0.30, ai_min=0.65)
        assert classify(0.65, thresholds) == "ai_generated"
        assert classify(0.70, thresholds) == "ai_generated"
        assert classify(1.0, thresholds) == "ai_generated"

    def test_classify_ambiguous(self) -> None:
        """Test score in (human_max, ai_min) returns 'ambiguous'."""
        thresholds = DetectionThresholdsConfig(human_max=0.30, ai_min=0.65)
        assert classify(0.31, thresholds) == "ambiguous"
        assert classify(0.50, thresholds) == "ambiguous"
        assert classify(0.64, thresholds) == "ambiguous"

    def test_classify_exact_boundaries(self) -> None:
        """Test exact boundary conditions."""
        thresholds = DetectionThresholdsConfig(human_max=0.30, ai_min=0.65)
        # At human_max, still human
        assert classify(0.30, thresholds) == "human"
        # Just above human_max, ambiguous
        assert classify(0.31, thresholds) == "ambiguous"
        # Just below ai_min, ambiguous
        assert classify(0.64, thresholds) == "ambiguous"
        # At ai_min, ai_generated
        assert classify(0.65, thresholds) == "ai_generated"

    def test_classify_custom_thresholds(self) -> None:
        """Test with custom threshold configuration."""
        thresholds = DetectionThresholdsConfig(human_max=0.20, ai_min=0.80)
        assert classify(0.15, thresholds) == "human"
        assert classify(0.50, thresholds) == "ambiguous"
        assert classify(0.85, thresholds) == "ai_generated"

    def test_classify_extreme_values(self) -> None:
        """Test classification at extremes."""
        thresholds = DetectionThresholdsConfig(human_max=0.30, ai_min=0.65)
        assert classify(0.0, thresholds) == "human"
        assert classify(1.0, thresholds) == "ai_generated"


class TestEnsembleWithClassify:
    """Integration tests combining compute_ensemble and classify."""

    def test_end_to_end_two_tier(self) -> None:
        """Test full pipeline: compute ensemble then classify."""
        weights = EnsembleWeightsConfig(roberta=0.7, statistical=0.3)
        thresholds = DetectionThresholdsConfig(human_max=0.30, ai_min=0.65)

        # Case 1: human classification
        score = compute_ensemble(0.2, 0.25, None, weights)
        assert classify(score, thresholds) == "human"

        # Case 2: ambiguous classification
        score = compute_ensemble(0.4, 0.5, None, weights)
        assert classify(score, thresholds) == "ambiguous"

        # Case 3: ai_generated classification
        score = compute_ensemble(0.8, 0.7, None, weights)
        assert classify(score, thresholds) == "ai_generated"

    def test_end_to_end_three_tier(self) -> None:
        """Test full pipeline with three tiers."""
        weights = EnsembleWeightsWithBinocularsConfig(
            roberta=0.4, statistical=0.15, binoculars=0.45
        )
        thresholds = DetectionThresholdsConfig(human_max=0.30, ai_min=0.65)

        # Low ensemble score
        score = compute_ensemble(0.1, 0.2, 0.15, weights)
        assert classify(score, thresholds) == "human"

        # High ensemble score
        score = compute_ensemble(0.8, 0.9, 0.85, weights)
        assert classify(score, thresholds) == "ai_generated"

    def test_score_stability_with_missing_tiers(self) -> None:
        """Test that re-normalization produces stable results."""
        weights = EnsembleWeightsWithBinocularsConfig(
            roberta=0.4, statistical=0.15, binoculars=0.45
        )
        thresholds = DetectionThresholdsConfig(human_max=0.30, ai_min=0.65)

        # Compute with all three tiers
        score_full = compute_ensemble(0.7, 0.7, 0.7, weights)
        classification_full = classify(score_full, thresholds)

        # Compute with only roberta (re-normalized)
        score_roberta = compute_ensemble(0.7, None, None, weights)
        classification_roberta = classify(score_roberta, thresholds)

        # Both should be identical (uniform input)
        assert abs(score_full - score_roberta) < 1e-6
        assert classification_full == classification_roberta
