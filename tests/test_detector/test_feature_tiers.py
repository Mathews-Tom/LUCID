"""Tests for feature tier enum and tiered feature extraction."""

from __future__ import annotations

from lucid.detector.features import FEATURE_BOUNDS, FeatureTier
from lucid.detector.statistical import StatisticalDetector

HUMAN_TEXT = (
    "The experiment failed spectacularly. Nobody expected the catalyst to react "
    "that way — certainly not Dr. Chen, who had spent three months optimizing the "
    "protocol. But science is funny like that. Sometimes your best-laid plans "
    "dissolve in a puff of hydrogen sulfide. We regrouped. Started over. "
    "The second attempt, stripped of our earlier assumptions, actually worked better."
)


class TestFeatureTierEnum:
    def test_fast_value(self) -> None:
        assert FeatureTier.FAST.value == "fast"

    def test_standard_value(self) -> None:
        assert FeatureTier.STANDARD.value == "standard"

    def test_deep_value(self) -> None:
        assert FeatureTier.DEEP.value == "deep"

    def test_all_tiers_present(self) -> None:
        assert len(FeatureTier) == 3


class TestFastTierFeatures:
    """Verify fast tier (no GPT-2, no deep) produces only text-based features."""

    def setup_method(self) -> None:
        self.detector = StatisticalDetector(
            use_gpt2_perplexity=False, use_deep_features=False
        )

    def test_no_lm_features(self) -> None:
        features = self.detector.extract_features(HUMAN_TEXT)
        lm_keys = [k for k in features if k.startswith("lm_")]
        assert lm_keys == []

    def test_no_deep_features(self) -> None:
        features = self.detector.extract_features(HUMAN_TEXT)
        assert "style_clause_density_variance" not in features

    def test_has_style_features(self) -> None:
        features = self.detector.extract_features(HUMAN_TEXT)
        assert "style_ttr" in features
        assert "style_hapax_ratio" in features

    def test_has_struct_features(self) -> None:
        features = self.detector.extract_features(HUMAN_TEXT)
        assert "struct_sentence_length_variance" in features

    def test_has_disc_features(self) -> None:
        features = self.detector.extract_features(HUMAN_TEXT)
        disc_keys = [k for k in features if k.startswith("disc_")]
        assert len(disc_keys) > 0


class TestFeatureKeyPrefixes:
    """Verify all feature keys from extract_features match FEATURE_BOUNDS keys."""

    def setup_method(self) -> None:
        self.detector = StatisticalDetector(
            use_gpt2_perplexity=False, use_deep_features=False
        )

    def test_all_keys_in_bounds(self) -> None:
        features = self.detector.extract_features(HUMAN_TEXT)
        for key in features:
            assert key in FEATURE_BOUNDS, (
                f"Feature key {key!r} not found in FEATURE_BOUNDS"
            )

    def test_valid_prefixes(self) -> None:
        valid_prefixes = {"lm_", "style_", "struct_", "disc_"}
        for key in FEATURE_BOUNDS:
            prefix = key.split("_")[0] + "_"
            assert prefix in valid_prefixes, (
                f"FEATURE_BOUNDS key {key!r} has invalid prefix"
            )
