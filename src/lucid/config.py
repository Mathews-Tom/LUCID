"""Layered TOML configuration with typed dataclass mapping.

Priority stack (highest wins):
    1. Hardcoded defaults (LUCIDConfig())
    2. config/default.toml (bundled)
    3. config/profiles/{profile}.toml (profile delta)
    4. ~/.config/lucid/config.toml (user config)
    5. CLI overrides (dot-notation)
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Typed config tree — all frozen, slots for memory efficiency
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GeneralConfig:
    """Top-level general settings."""

    profile: str = "balanced"
    language: str = "en"
    log_level: str = "info"
    output_dir: str = "./lucid_output"


@dataclass(frozen=True, slots=True)
class OllamaModelsConfig:
    """Model tags per quality profile."""

    fast: str = "phi3:3.8b"
    balanced: str = "qwen2.5:7b"
    quality: str = "llama3.1:8b"


@dataclass(frozen=True, slots=True)
class OllamaConfig:
    """Ollama server connection settings."""

    host: str = "http://localhost:11434"
    timeout_seconds: int = 60
    health_check_on_start: bool = True
    models: OllamaModelsConfig = field(default_factory=OllamaModelsConfig)


@dataclass(frozen=True, slots=True)
class EnsembleWeightsConfig:
    """Detection ensemble weights (Tier 1 + Tier 2)."""

    roberta: float = 0.7
    statistical: float = 0.3


@dataclass(frozen=True, slots=True)
class EnsembleWeightsWithBinocularsConfig:
    """Detection ensemble weights (Tier 1 + Tier 2 + Tier 3)."""

    roberta: float = 0.4
    statistical: float = 0.15
    binoculars: float = 0.45


@dataclass(frozen=True, slots=True)
class DetectionThresholdsConfig:
    """Score thresholds for classification decisions."""

    human_max: float = 0.30
    ai_min: float = 0.65
    ambiguity_triggers_binoculars: bool = True


@dataclass(frozen=True, slots=True)
class DetectionConfig:
    """AI detection engine settings."""

    enabled: bool = True
    roberta_model: str = "roberta-base-openai-detector"
    use_statistical: bool = True
    use_binoculars: bool = False
    ensemble_weights: EnsembleWeightsConfig = field(default_factory=EnsembleWeightsConfig)
    ensemble_weights_with_binoculars: EnsembleWeightsWithBinocularsConfig = field(
        default_factory=EnsembleWeightsWithBinocularsConfig
    )
    thresholds: DetectionThresholdsConfig = field(default_factory=DetectionThresholdsConfig)


@dataclass(frozen=True, slots=True)
class TemperatureProfileConfig:
    """LLM temperature per quality profile."""

    fast: float = 0.7
    balanced: float = 0.6
    quality: float = 0.5


@dataclass(frozen=True, slots=True)
class TermProtectionConfig:
    """Term protection settings for humanization."""

    use_ner: bool = True
    custom_terms: tuple[str, ...] = ()
    protect_citations: bool = True
    protect_numbers: bool = True


@dataclass(frozen=True, slots=True)
class HumanizerConfig:
    """Humanization engine settings."""

    max_retries: int = 3
    adversarial_iterations: int = 5
    adversarial_target_score: float = 0.25
    temperature: TemperatureProfileConfig = field(default_factory=TemperatureProfileConfig)
    term_protection: TermProtectionConfig = field(default_factory=TermProtectionConfig)


@dataclass(frozen=True, slots=True)
class EvaluatorConfig:
    """Semantic evaluation settings."""

    embedding_threshold: float = 0.80
    nli_require_bidirectional: bool = True
    bertscore_threshold: float = 0.88
    bertscore_model: str = "microsoft/deberta-xlarge-mnli"


@dataclass(frozen=True, slots=True)
class ParserConfig:
    """Document parser settings."""

    latex_library: str = "pylatexenc"
    markdown_library: str = "markdown-it-py"
    custom_prose_environments: tuple[str, ...] = ()
    custom_structural_macros: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Post-reconstruction validation settings."""

    latex_compile_check: bool = True
    latex_compiler: str = "pdflatex"
    markdown_render_check: bool = True


@dataclass(frozen=True, slots=True)
class LUCIDConfig:
    """Root configuration node."""

    general: GeneralConfig = field(default_factory=GeneralConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    humanizer: HumanizerConfig = field(default_factory=HumanizerConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    parser: ParserConfig = field(default_factory=ParserConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. Lists replace, dicts recurse."""
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _coerce_value(s: str) -> bool | int | float | str:
    """Coerce a CLI string value to its typed equivalent."""
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _apply_dot_override(raw: dict[str, Any], dot_key: str, str_value: str) -> None:
    """Apply a dot-notation CLI override into the raw config dict.

    Example: _apply_dot_override(raw, "detection.use_binoculars", "true")
    sets raw["detection"]["use_binoculars"] = True
    """
    parts = dot_key.split(".")
    target = raw
    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        target = target[part]
    target[parts[-1]] = _coerce_value(str_value)


def _load_toml_file(path: Path) -> dict[str, Any]:
    """Load and parse a TOML file, returning empty dict if not found."""
    if not path.is_file():
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


def _load_bundled_toml(filename: str) -> dict[str, Any]:
    """Load a TOML file bundled in the config/ directory relative to project root."""
    # Walk up from this file to find the project root containing config/
    current = Path(__file__).resolve().parent
    for _ in range(5):
        config_path = current / "config" / filename
        if config_path.is_file():
            with open(config_path, "rb") as f:
                return tomllib.load(f)
        current = current.parent

    # Fallback: try importlib.resources for installed packages
    try:
        config_pkg = resources.files("lucid").joinpath(f"../../../config/{filename}")
        if hasattr(config_pkg, "read_bytes"):
            data = config_pkg.read_bytes()
            return tomllib.loads(data.decode("utf-8"))
    except (FileNotFoundError, TypeError):
        pass

    return {}


def _lists_to_tuples(data: dict[str, Any]) -> dict[str, Any]:
    """Convert lists to tuples in config dicts (for frozen dataclass compatibility)."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _lists_to_tuples(value)
        elif isinstance(value, list):
            result[key] = tuple(value)
        else:
            result[key] = value
    return result


def _build_config(raw: dict[str, Any]) -> LUCIDConfig:
    """Map a merged raw dict to the typed LUCIDConfig tree."""
    raw = _lists_to_tuples(raw)

    general_raw = raw.get("general", {})
    ollama_raw = raw.get("ollama", {})
    detection_raw = raw.get("detection", {})
    humanizer_raw = raw.get("humanizer", {})
    evaluator_raw = raw.get("evaluator", {})
    parser_raw = raw.get("parser", {})
    validation_raw = raw.get("validation", {})

    # Build nested configs bottom-up
    ollama_models = OllamaModelsConfig(**ollama_raw.pop("models", {}))
    ollama = OllamaConfig(**ollama_raw, models=ollama_models)

    ensemble_weights = EnsembleWeightsConfig(**detection_raw.pop("ensemble_weights", {}))
    ensemble_bino = EnsembleWeightsWithBinocularsConfig(
        **detection_raw.pop("ensemble_weights_with_binoculars", {})
    )
    thresholds = DetectionThresholdsConfig(**detection_raw.pop("thresholds", {}))
    detection = DetectionConfig(
        **detection_raw,
        ensemble_weights=ensemble_weights,
        ensemble_weights_with_binoculars=ensemble_bino,
        thresholds=thresholds,
    )

    temp = TemperatureProfileConfig(**humanizer_raw.pop("temperature", {}))
    term_prot = TermProtectionConfig(**humanizer_raw.pop("term_protection", {}))
    humanizer = HumanizerConfig(**humanizer_raw, temperature=temp, term_protection=term_prot)

    return LUCIDConfig(
        general=GeneralConfig(**general_raw),
        ollama=ollama,
        detection=detection,
        humanizer=humanizer,
        evaluator=EvaluatorConfig(**evaluator_raw),
        parser=ParserConfig(**parser_raw),
        validation=ValidationConfig(**validation_raw),
    )


def load_config(
    profile: str | None = None,
    user_config_path: Path | None = None,
    cli_overrides: dict[str, str] | None = None,
) -> LUCIDConfig:
    """Load configuration with 5-layer priority stack.

    Args:
        profile: Quality profile name ("fast", "balanced", "quality").
            If None, uses the value from default.toml.
        user_config_path: Path to user config TOML. Defaults to
            ~/.config/lucid/config.toml.
        cli_overrides: Dot-notation key→value pairs from CLI flags.

    Returns:
        Fully resolved, typed LUCIDConfig.
    """
    # Layer 1: hardcoded defaults (implicit via dataclass defaults)
    # Layer 2: bundled default.toml
    raw = _load_bundled_toml("default.toml")

    # Determine effective profile
    effective_profile = profile
    if effective_profile is None:
        effective_profile = raw.get("general", {}).get("profile", "balanced")

    # Layer 3: profile overrides
    profile_raw = _load_bundled_toml(f"profiles/{effective_profile}.toml")
    raw = _deep_merge(raw, profile_raw)

    # Layer 4: user config
    if user_config_path is None:
        user_config_path = Path.home() / ".config" / "lucid" / "config.toml"
    user_raw = _load_toml_file(user_config_path)
    raw = _deep_merge(raw, user_raw)

    # Layer 5: CLI overrides
    if cli_overrides:
        for dot_key, str_value in cli_overrides.items():
            _apply_dot_override(raw, dot_key, str_value)

    return _build_config(raw)
