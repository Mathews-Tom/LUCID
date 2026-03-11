"""Centralized exception hierarchy for LUCID."""

from __future__ import annotations


class LUCIDError(Exception):
    """Base exception for all LUCID errors."""


class ConfigError(LUCIDError):
    """Configuration loading or validation error."""


class DetectorError(LUCIDError):
    """Base error for detector operations."""


class DetectorInitError(DetectorError):
    """Failed to initialize a detector (model loading, missing dependency)."""


class DetectionError(DetectorError):
    """Failed during detection inference."""


class BinocularsUnavailableError(DetectorError):
    """Binoculars detector cannot be used (missing torch/transformers)."""


class TransformError(LUCIDError):
    """Error during text transformation."""


class MetricError(LUCIDError):
    """Error during metric computation."""


class BenchmarkError(LUCIDError):
    """Error during benchmark execution."""
