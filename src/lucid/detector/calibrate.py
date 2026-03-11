"""Detection score calibration methods."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    """Configuration for a calibration method."""

    method: str  # "temperature_scaling", "isotonic", "slice_thresholds"
    version: str
    parameters: dict[str, Any]


class Calibrator:
    """Calibrate raw detection scores to well-calibrated probabilities."""

    def __init__(self, config: CalibrationConfig) -> None:
        self._config = config
        self._method = config.method
        self._params = config.parameters

    @property
    def version(self) -> str:
        return self._config.version

    def calibrate(self, raw_score: float) -> float:
        """Apply calibration to a raw detection score.

        Returns calibrated probability in [0.0, 1.0].
        """
        if self._method == "temperature_scaling":
            return self._temperature_scale(raw_score)
        if self._method == "isotonic":
            return self._isotonic_interpolate(raw_score)
        if self._method == "slice_thresholds":
            return raw_score  # thresholds don't change the score, just the decision
        raise ValueError(f"Unknown calibration method: {self._method}")

    def _temperature_scale(self, score: float) -> float:
        """Apply temperature scaling: sigmoid(logit(score) / T)."""
        temperature = self._params.get("temperature", 1.0)
        # Clamp to avoid log(0)
        score = max(1e-7, min(1.0 - 1e-7, score))
        logit = math.log(score / (1.0 - score))
        scaled_logit = logit / temperature
        return 1.0 / (1.0 + math.exp(-scaled_logit))

    def _isotonic_interpolate(self, score: float) -> float:
        """Interpolate using isotonic regression breakpoints."""
        breakpoints = self._params.get("breakpoints", [])
        if not breakpoints:
            return score
        # breakpoints is list of (raw, calibrated) pairs, sorted by raw
        xs = [bp[0] for bp in breakpoints]
        ys = [bp[1] for bp in breakpoints]
        return float(np.interp(score, xs, ys))

    @classmethod
    def from_file(cls, path: Path) -> Calibrator:
        """Load calibration config from a JSON file."""
        data = json.loads(path.read_text(encoding="utf-8"))
        config = CalibrationConfig(
            method=data["method"],
            version=data["version"],
            parameters=data.get("parameters", {}),
        )
        return cls(config)

    def save(self, path: Path) -> None:
        """Save calibration config to a JSON file."""
        data = {
            "method": self._config.method,
            "version": self._config.version,
            "parameters": self._config.parameters,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def fit_temperature_scaling(
    scores: list[float], labels: list[int], version: str = "v1"
) -> Calibrator:
    """Fit temperature scaling from scores and binary labels.

    Uses grid search over temperature values to minimize
    negative log-likelihood (calibration loss).

    Args:
        scores: Raw detection scores in [0, 1].
        labels: Binary labels (1 = AI, 0 = human).
        version: Version string for the calibration.

    Returns:
        Calibrator with fitted temperature parameter.
    """
    best_temp = 1.0
    best_nll = float("inf")

    for temp_candidate in np.arange(0.1, 5.0, 0.05):
        nll = 0.0
        for score, label in zip(scores, labels):
            score = max(1e-7, min(1.0 - 1e-7, score))
            logit = math.log(score / (1.0 - score))
            scaled = 1.0 / (1.0 + math.exp(-logit / temp_candidate))
            scaled = max(1e-7, min(1.0 - 1e-7, scaled))
            if label == 1:
                nll -= math.log(scaled)
            else:
                nll -= math.log(1.0 - scaled)
        if nll < best_nll:
            best_nll = nll
            best_temp = float(temp_candidate)

    config = CalibrationConfig(
        method="temperature_scaling",
        version=version,
        parameters={"temperature": best_temp},
    )
    return Calibrator(config)
