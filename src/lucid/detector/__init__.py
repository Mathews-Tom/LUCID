"""AI content detection engine."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lucid.detector.ensemble import classify, compute_ensemble

if TYPE_CHECKING:
    from lucid.detector.base import LUCIDDetector
    from lucid.detector.binoculars import BinocularsDetector
    from lucid.detector.roberta import RobertaDetector
    from lucid.detector.statistical import StatisticalDetector


class DetectorError(Exception):
    """Base exception for all detector-related errors."""


class DetectorInitError(DetectorError):
    """Raised when a detector fails to initialize (model load, missing deps)."""


class DetectionError(DetectorError):
    """Raised when a detector fails during inference on a given input."""


class BinocularsUnavailableError(DetectorError):
    """Raised when Binoculars tier cannot run.

    This occurs when torch/transformers are not installed or when
    model loading fails due to OOM or other runtime errors.
    """


def __getattr__(name: str) -> Any:
    """Lazy-load detector classes on first access.

    Supports:
    - LUCIDDetector (orchestrator)
    - RobertaDetector (Tier 1)
    - StatisticalDetector (Tier 2)
    - BinocularsDetector (Tier 3)
    """
    if name == "LUCIDDetector":
        from lucid.detector.base import LUCIDDetector

        return LUCIDDetector
    if name == "RobertaDetector":
        from lucid.detector.roberta import RobertaDetector

        return RobertaDetector
    if name == "StatisticalDetector":
        from lucid.detector.statistical import StatisticalDetector

        return StatisticalDetector
    if name == "BinocularsDetector":
        from lucid.detector.binoculars import BinocularsDetector

        return BinocularsDetector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DetectorError",
    "DetectorInitError",
    "DetectionError",
    "BinocularsUnavailableError",
    "compute_ensemble",
    "classify",
    "LUCIDDetector",
    "RobertaDetector",
    "StatisticalDetector",
    "BinocularsDetector",
]
