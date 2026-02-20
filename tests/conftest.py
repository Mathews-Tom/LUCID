"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from lucid.config import LUCIDConfig, load_config

CORPUS_DIR = Path(__file__).parent / "corpus"


@pytest.fixture
def default_config() -> LUCIDConfig:
    """Load default config (balanced profile)."""
    return load_config(profile="balanced")


@pytest.fixture
def fast_config() -> LUCIDConfig:
    """Load fast profile config."""
    return load_config(profile="fast")


@pytest.fixture
def quality_config() -> LUCIDConfig:
    """Load quality profile config."""
    return load_config(profile="quality")


@pytest.fixture
def corpus_latex_dir() -> Path:
    """Path to LaTeX test corpus."""
    return CORPUS_DIR / "latex"


@pytest.fixture
def corpus_markdown_dir() -> Path:
    """Path to Markdown test corpus."""
    return CORPUS_DIR / "markdown"
