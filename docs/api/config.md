# Configuration

LUCID uses a layered TOML configuration system with typed dataclass mapping.

## Priority Stack

1. Hardcoded defaults (dataclass defaults)
2. `config/default.toml` (bundled)
3. `config/profiles/{profile}.toml` (profile delta)
4. `~/.config/lucid/config.toml` (user config)
5. CLI overrides (dot-notation)

## load_config

::: lucid.config.load_config

## LUCIDConfig

::: lucid.config.LUCIDConfig
    options:
      show_source: false
      members: false

## Sub-configurations

### GeneralConfig
::: lucid.config.GeneralConfig
    options:
      show_source: false
      members: false

### OllamaConfig
::: lucid.config.OllamaConfig
    options:
      show_source: false
      members: false

### DetectionConfig
::: lucid.config.DetectionConfig
    options:
      show_source: false
      members: false

### HumanizerConfig
::: lucid.config.HumanizerConfig
    options:
      show_source: false
      members: false

### EvaluatorConfig
::: lucid.config.EvaluatorConfig
    options:
      show_source: false
      members: false
