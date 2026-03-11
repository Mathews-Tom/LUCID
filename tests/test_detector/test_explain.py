"""Tests for detection explainability."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from lucid.detector.explain import (
    DetectionExplainer,
    _FEATURE_DESCRIPTIONS,
    _get_category,
)
from lucid.detector.features import FEATURE_BOUNDS


@dataclass
class _FakeResult:
    """Minimal stand-in for DetectionResult."""

    ensemble_score: float
    classification: str
    feature_details: dict[str, Any] = field(default_factory=dict)


class TestDetectionExplainer:
    def test_explain_with_known_features(self) -> None:
        result = _FakeResult(
            ensemble_score=0.75,
            classification="ai_generated",
            feature_details={
                "lm_perplexity_mean": 30.0,  # low = AI-like (inverted)
                "style_ttr": 0.3,  # low = AI-like (inverted)
                "struct_symmetry_score": 0.9,  # high = AI-like (not inverted)
            },
        )
        explainer = DetectionExplainer()
        output = explainer.explain(result)

        assert len(output["explanations"]) == 3
        assert len(output["top_signals"]) == 3
        assert "ai_generated" in output["summary"]

    def test_explanations_sorted_by_normalized_score_desc(self) -> None:
        result = _FakeResult(
            ensemble_score=0.5,
            classification="ambiguous",
            feature_details={
                "lm_perplexity_mean": 10.0,  # at low bound, inverted -> 1.0
                "style_ttr": 0.9,  # at high bound, inverted -> 0.0
            },
        )
        explainer = DetectionExplainer()
        output = explainer.explain(result)
        scores = [e["normalized_score"] for e in output["explanations"]]
        assert scores == sorted(scores, reverse=True)

    def test_top_signals_extracts_top_three(self) -> None:
        features = {
            "lm_perplexity_mean": 10.0,
            "lm_burstiness": 0.0,
            "style_ttr": 0.2,
            "struct_symmetry_score": 1.0,
            "disc_bigram_entropy": 2.0,
        }
        result = _FakeResult(
            ensemble_score=0.8,
            classification="ai_generated",
            feature_details=features,
        )
        explainer = DetectionExplainer()
        output = explainer.explain(result)
        assert len(output["top_signals"]) == 3

    def test_empty_features_returns_defaults(self) -> None:
        result = _FakeResult(
            ensemble_score=0.5,
            classification="ambiguous",
            feature_details={},
        )
        explainer = DetectionExplainer()
        output = explainer.explain(result)
        assert output["explanations"] == []
        assert output["top_signals"] == []
        assert output["summary"] == "No features available"

    def test_none_feature_details_returns_defaults(self) -> None:
        result = _FakeResult(
            ensemble_score=0.5,
            classification="ambiguous",
        )
        result.feature_details = None  # type: ignore[assignment]
        explainer = DetectionExplainer()
        output = explainer.explain(result)
        assert output["explanations"] == []

    def test_summary_contains_classification(self) -> None:
        result = _FakeResult(
            ensemble_score=0.2,
            classification="human",
            feature_details={"style_ttr": 0.8},
        )
        explainer = DetectionExplainer()
        output = explainer.explain(result)
        assert "human" in output["summary"]

    def test_summary_counts_ai_features(self) -> None:
        # lm_perplexity_mean=10 -> inverted, at low bound -> normalized=1.0 -> ai_like
        # style_ttr=0.9 -> inverted, at high bound -> normalized=0.0 -> human_like
        result = _FakeResult(
            ensemble_score=0.6,
            classification="ambiguous",
            feature_details={
                "lm_perplexity_mean": 10.0,
                "style_ttr": 0.9,
            },
        )
        explainer = DetectionExplainer()
        output = explainer.explain(result)
        assert "1 of 2 features indicate AI-generated content" in output["summary"]

    def test_direction_assignment(self) -> None:
        # struct_symmetry_score: (0.0, 1.0, False), value=0.9 -> normalized=0.9 -> ai_like
        # struct_symmetry_score: (0.0, 1.0, False), value=0.1 -> normalized=0.1 -> human_like
        explainer = DetectionExplainer()

        result_high = _FakeResult(
            ensemble_score=0.8,
            classification="ai_generated",
            feature_details={"struct_symmetry_score": 0.9},
        )
        output = explainer.explain(result_high)
        assert output["explanations"][0]["direction"] == "ai_like"

        result_low = _FakeResult(
            ensemble_score=0.2,
            classification="human",
            feature_details={"struct_symmetry_score": 0.1},
        )
        output = explainer.explain(result_low)
        assert output["explanations"][0]["direction"] == "human_like"

    def test_unknown_feature_skipped(self) -> None:
        result = _FakeResult(
            ensemble_score=0.5,
            classification="ambiguous",
            feature_details={"nonexistent_feature": 42.0},
        )
        explainer = DetectionExplainer()
        output = explainer.explain(result)
        assert output["explanations"] == []

    def test_explanation_categories(self) -> None:
        result = _FakeResult(
            ensemble_score=0.5,
            classification="ambiguous",
            feature_details={
                "lm_perplexity_mean": 50.0,
                "style_ttr": 0.5,
                "struct_symmetry_score": 0.5,
                "disc_bigram_entropy": 5.0,
            },
        )
        explainer = DetectionExplainer()
        output = explainer.explain(result)
        categories = {e["category"] for e in output["explanations"]}
        assert "Language Model Statistics" in categories
        assert "Stylometric Features" in categories
        assert "Structural Features" in categories
        assert "Discourse Features" in categories


class TestFeatureDescriptionCoverage:
    def test_all_feature_bounds_have_descriptions(self) -> None:
        missing = set(FEATURE_BOUNDS.keys()) - set(_FEATURE_DESCRIPTIONS.keys())
        assert missing == set(), f"Features missing descriptions: {missing}"


class TestGetCategory:
    def test_known_prefixes(self) -> None:
        assert _get_category("lm_perplexity_mean") == "Language Model Statistics"
        assert _get_category("style_ttr") == "Stylometric Features"
        assert _get_category("struct_symmetry_score") == "Structural Features"
        assert _get_category("disc_bigram_entropy") == "Discourse Features"

    def test_unknown_prefix(self) -> None:
        assert _get_category("foo_bar") == "Unknown"
