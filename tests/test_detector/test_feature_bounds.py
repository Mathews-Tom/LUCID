"""Tests for feature normalization and score combination."""

from __future__ import annotations

import pytest

from lucid.detector.features import FEATURE_BOUNDS
from lucid.detector.statistical import _combine_features, _normalize_feature


class TestNormalizeFeature:
    def test_value_at_low_non_inverted(self) -> None:
        assert _normalize_feature(10.0, 10.0, 200.0, invert=False) == 0.0

    def test_value_at_high_non_inverted(self) -> None:
        assert _normalize_feature(200.0, 10.0, 200.0, invert=False) == 1.0

    def test_value_at_midpoint_non_inverted(self) -> None:
        result = _normalize_feature(105.0, 10.0, 200.0, invert=False)
        assert abs(result - 0.5) < 0.01

    def test_value_at_low_inverted(self) -> None:
        assert _normalize_feature(10.0, 10.0, 200.0, invert=True) == 1.0

    def test_value_at_high_inverted(self) -> None:
        assert _normalize_feature(200.0, 10.0, 200.0, invert=True) == 0.0

    def test_value_below_low_clamped(self) -> None:
        result = _normalize_feature(0.0, 10.0, 200.0, invert=False)
        assert result == 0.0

    def test_value_above_high_clamped(self) -> None:
        result = _normalize_feature(300.0, 10.0, 200.0, invert=False)
        assert result == 1.0

    def test_equal_bounds_returns_half(self) -> None:
        assert _normalize_feature(5.0, 5.0, 5.0, invert=False) == 0.5
        assert _normalize_feature(5.0, 5.0, 5.0, invert=True) == 0.5

    def test_value_below_low_inverted_clamped_to_one(self) -> None:
        result = _normalize_feature(0.0, 10.0, 200.0, invert=True)
        assert result == 1.0

    def test_value_above_high_inverted_clamped_to_zero(self) -> None:
        result = _normalize_feature(300.0, 10.0, 200.0, invert=True)
        assert result == 0.0


class TestCombineFeatures:
    def test_empty_features_returns_half(self) -> None:
        assert _combine_features({}) == 0.5

    def test_all_none_returns_half(self) -> None:
        features = {k: None for k in FEATURE_BOUNDS}
        assert _combine_features(features) == 0.5

    def test_single_feature(self) -> None:
        features = {"style_ttr": 0.55}
        result = _combine_features(features)
        assert 0.0 <= result <= 1.0

    def test_result_in_range(self) -> None:
        features = {
            "style_ttr": 0.6,
            "style_hapax_ratio": 0.4,
            "struct_sentence_length_variance": 100.0,
            "disc_pos_trigram_entropy": 3.5,
        }
        result = _combine_features(features)
        assert 0.0 <= result <= 1.0

    def test_unknown_keys_ignored(self) -> None:
        features = {"nonexistent_feature": 42.0}
        assert _combine_features(features) == 0.5


class TestFeatureBoundsValidity:
    def test_all_bounds_have_three_elements(self) -> None:
        for name, bounds in FEATURE_BOUNDS.items():
            assert len(bounds) == 3, f"{name} has {len(bounds)} elements"

    def test_low_less_than_high(self) -> None:
        for name, (low, high, _) in FEATURE_BOUNDS.items():
            assert low < high, f"{name}: low={low} >= high={high}"

    def test_invert_is_bool(self) -> None:
        for name, (_, _, invert) in FEATURE_BOUNDS.items():
            assert isinstance(invert, bool), f"{name}: invert={invert!r}"

    def test_all_keys_have_valid_prefix(self) -> None:
        valid_prefixes = {"lm_", "style_", "struct_", "disc_"}
        for key in FEATURE_BOUNDS:
            prefix = key.split("_")[0] + "_"
            assert prefix in valid_prefixes, f"{key!r} has invalid prefix"

    def test_expected_feature_count(self) -> None:
        assert len(FEATURE_BOUNDS) == 15
