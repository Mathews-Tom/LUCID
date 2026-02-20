"""Tests for the layered configuration system."""

from __future__ import annotations

from pathlib import Path

from lucid.config import (
    LUCIDConfig,
    _apply_dot_override,
    _coerce_value,
    _deep_merge,
    load_config,
)


class TestDeepMerge:
    """_deep_merge behavior."""

    def test_flat_override(self) -> None:
        """Scalar values in override replace base."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        """Nested dicts are merged recursively."""
        base = {"outer": {"a": 1, "b": 2}}
        override = {"outer": {"b": 3, "c": 4}}
        result = _deep_merge(base, override)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_lists_replace(self) -> None:
        """Lists in override replace base lists (no extend)."""
        base = {"items": [1, 2, 3]}
        override = {"items": [4, 5]}
        result = _deep_merge(base, override)
        assert result == {"items": [4, 5]}

    def test_base_unmodified(self) -> None:
        """Original base dict is not mutated."""
        base = {"a": {"x": 1}}
        override = {"a": {"y": 2}}
        _deep_merge(base, override)
        assert base == {"a": {"x": 1}}

    def test_deeply_nested(self) -> None:
        """Three levels of nesting merge correctly."""
        base = {"l1": {"l2": {"l3": "old", "keep": True}}}
        override = {"l1": {"l2": {"l3": "new"}}}
        result = _deep_merge(base, override)
        assert result["l1"]["l2"]["l3"] == "new"
        assert result["l1"]["l2"]["keep"] is True


class TestCoerceValue:
    """_coerce_value type detection."""

    def test_true(self) -> None:
        assert _coerce_value("true") is True

    def test_false(self) -> None:
        assert _coerce_value("False") is False

    def test_integer(self) -> None:
        assert _coerce_value("42") == 42
        assert isinstance(_coerce_value("42"), int)

    def test_float(self) -> None:
        assert _coerce_value("3.14") == 3.14
        assert isinstance(_coerce_value("3.14"), float)

    def test_string(self) -> None:
        assert _coerce_value("hello") == "hello"
        assert isinstance(_coerce_value("hello"), str)

    def test_negative_integer(self) -> None:
        assert _coerce_value("-5") == -5


class TestApplyDotOverride:
    """_apply_dot_override path traversal."""

    def test_single_key(self) -> None:
        """Single key sets top-level value."""
        raw: dict[str, object] = {"key": "old"}
        _apply_dot_override(raw, "key", "new")
        assert raw["key"] == "new"

    def test_nested_path(self) -> None:
        """Dotted path traverses nested dicts."""
        raw: dict[str, object] = {"a": {"b": {"c": "old"}}}
        _apply_dot_override(raw, "a.b.c", "new")
        assert raw["a"]["b"]["c"] == "new"  # type: ignore[index]

    def test_creates_missing_keys(self) -> None:
        """Missing intermediate keys are created."""
        raw: dict[str, object] = {}
        _apply_dot_override(raw, "x.y.z", "42")
        assert raw["x"]["y"]["z"] == 42  # type: ignore[index]

    def test_coerces_types(self) -> None:
        """Values are type-coerced from strings."""
        raw: dict[str, object] = {"detection": {}}
        _apply_dot_override(raw, "detection.use_binoculars", "true")
        assert raw["detection"]["use_binoculars"] is True  # type: ignore[index]


class TestLoadConfig:
    """load_config with profile propagation and layered merging."""

    def test_default_config_loads(self) -> None:
        """Default config loads without errors."""
        config = load_config()
        assert isinstance(config, LUCIDConfig)
        assert config.general.language == "en"

    def test_balanced_profile_defaults(self) -> None:
        """Balanced profile is the default."""
        config = load_config()
        assert config.general.profile == "balanced"

    def test_fast_profile(self) -> None:
        """Fast profile overrides apply correctly."""
        config = load_config(profile="fast")
        assert config.general.profile == "fast"
        assert config.ollama.timeout_seconds == 30
        assert config.detection.use_statistical is False
        assert config.humanizer.max_retries == 1

    def test_quality_profile(self) -> None:
        """Quality profile enables binoculars and stricter thresholds."""
        config = load_config(profile="quality")
        assert config.general.profile == "quality"
        assert config.detection.use_binoculars is True
        assert config.ollama.timeout_seconds == 120
        assert config.humanizer.max_retries == 5

    def test_cli_override_scalar(self) -> None:
        """CLI override changes a scalar value."""
        config = load_config(cli_overrides={"ollama.timeout_seconds": "90"})
        assert config.ollama.timeout_seconds == 90

    def test_cli_override_nested(self) -> None:
        """CLI override changes a deeply nested value."""
        config = load_config(cli_overrides={"detection.thresholds.human_max": "0.25"})
        assert config.detection.thresholds.human_max == 0.25

    def test_cli_override_trumps_profile(self) -> None:
        """CLI override has higher priority than profile."""
        config = load_config(
            profile="fast",
            cli_overrides={"humanizer.max_retries": "10"},
        )
        # Fast profile sets max_retries=1, but CLI override sets 10
        assert config.humanizer.max_retries == 10

    def test_nonexistent_user_config_ignored(self) -> None:
        """Missing user config file is silently ignored."""
        config = load_config(user_config_path=Path("/nonexistent/path/config.toml"))
        assert isinstance(config, LUCIDConfig)

    def test_model_tags_default(self) -> None:
        """Default model tags match design spec."""
        config = load_config()
        assert config.ollama.models.fast == "phi3:3.8b"
        assert config.ollama.models.balanced == "qwen2.5:7b"
        assert config.ollama.models.quality == "llama3.1:8b"

    def test_ensemble_weights_default(self) -> None:
        """Default ensemble weights sum to 1.0."""
        config = load_config()
        w = config.detection.ensemble_weights
        assert abs(w.roberta + w.statistical - 1.0) < 1e-9

    def test_ensemble_weights_with_binoculars_default(self) -> None:
        """Binoculars ensemble weights sum to 1.0."""
        config = load_config()
        w = config.detection.ensemble_weights_with_binoculars
        assert abs(w.roberta + w.statistical + w.binoculars - 1.0) < 1e-9


class TestLUCIDConfigDefaults:
    """Verify hardcoded defaults match design spec."""

    def test_detection_thresholds(self) -> None:
        """Detection thresholds match system-design.md."""
        config = LUCIDConfig()
        assert config.detection.thresholds.human_max == 0.30
        assert config.detection.thresholds.ai_min == 0.65

    def test_evaluator_thresholds(self) -> None:
        """Evaluator thresholds match system-design.md."""
        config = LUCIDConfig()
        assert config.evaluator.embedding_threshold == 0.80
        assert config.evaluator.bertscore_threshold == 0.88

    def test_humanizer_defaults(self) -> None:
        """Humanizer defaults match system-design.md."""
        config = LUCIDConfig()
        assert config.humanizer.adversarial_iterations == 5
        assert config.humanizer.adversarial_target_score == 0.25

    def test_temperature_profiles(self) -> None:
        """Temperature per profile matches system-design.md."""
        config = LUCIDConfig()
        assert config.humanizer.temperature.fast == 0.7
        assert config.humanizer.temperature.balanced == 0.6
        assert config.humanizer.temperature.quality == 0.5
