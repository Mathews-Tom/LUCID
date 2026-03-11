"""Core protocols and shared infrastructure."""

from __future__ import annotations

from lucid.core.errors import (
    BenchmarkError,
    BinocularsUnavailableError,
    ConfigError,
    DetectionError,
    DetectorError,
    DetectorInitError,
    LUCIDError,
    MetricError,
    TransformError,
)
from lucid.core.registry import Registry
from lucid.core.types import (
    DetectionRecord,
    MetricRecord,
    MetricResult,
    SampleRecord,
    TransformationRecord,
)

__all__ = [
    "BenchmarkError",
    "BinocularsUnavailableError",
    "ConfigError",
    "DetectionError",
    "DetectionRecord",
    "DetectorError",
    "DetectorInitError",
    "LUCIDError",
    "MetricError",
    "MetricRecord",
    "MetricResult",
    "Registry",
    "SampleRecord",
    "TransformError",
    "TransformationRecord",
]
