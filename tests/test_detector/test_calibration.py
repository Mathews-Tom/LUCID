"""Tests for detection score calibration."""
from __future__ import annotations

import json
import math

import pytest

from lucid.detector.calibrate import (
    CalibrationConfig,
    Calibrator,
    fit_temperature_scaling,
)


class TestTemperatureScaling:
    def test_temperature_one_is_near_identity(self) -> None:
        config = CalibrationConfig(
            method="temperature_scaling",
            version="v1",
            parameters={"temperature": 1.0},
        )
        cal = Calibrator(config)
        for score in [0.1, 0.3, 0.5, 0.7, 0.9]:
            assert cal.calibrate(score) == pytest.approx(score, abs=1e-5)

    def test_temperature_greater_than_one_compresses(self) -> None:
        config = CalibrationConfig(
            method="temperature_scaling",
            version="v1",
            parameters={"temperature": 2.0},
        )
        cal = Calibrator(config)
        # High temperature pushes scores toward 0.5
        assert cal.calibrate(0.9) < 0.9
        assert cal.calibrate(0.1) > 0.1
        # Midpoint stays
        assert cal.calibrate(0.5) == pytest.approx(0.5, abs=1e-5)

    def test_temperature_less_than_one_sharpens(self) -> None:
        config = CalibrationConfig(
            method="temperature_scaling",
            version="v1",
            parameters={"temperature": 0.5},
        )
        cal = Calibrator(config)
        assert cal.calibrate(0.9) > 0.9
        assert cal.calibrate(0.1) < 0.1

    def test_clamps_extreme_values(self) -> None:
        config = CalibrationConfig(
            method="temperature_scaling",
            version="v1",
            parameters={"temperature": 1.0},
        )
        cal = Calibrator(config)
        # Should not raise for 0.0 or 1.0
        result_low = cal.calibrate(0.0)
        result_high = cal.calibrate(1.0)
        assert 0.0 <= result_low <= 1.0
        assert 0.0 <= result_high <= 1.0


class TestIsotonicInterpolation:
    def test_basic_interpolation(self) -> None:
        config = CalibrationConfig(
            method="isotonic",
            version="v1",
            parameters={
                "breakpoints": [[0.0, 0.0], [0.5, 0.3], [1.0, 1.0]]
            },
        )
        cal = Calibrator(config)
        assert cal.calibrate(0.0) == pytest.approx(0.0)
        assert cal.calibrate(0.5) == pytest.approx(0.3)
        assert cal.calibrate(1.0) == pytest.approx(1.0)
        # Midpoint between 0.0 and 0.5 should interpolate
        assert cal.calibrate(0.25) == pytest.approx(0.15)

    def test_empty_breakpoints_returns_score(self) -> None:
        config = CalibrationConfig(
            method="isotonic",
            version="v1",
            parameters={"breakpoints": []},
        )
        cal = Calibrator(config)
        assert cal.calibrate(0.7) == pytest.approx(0.7)


class TestSliceThresholds:
    def test_passthrough(self) -> None:
        config = CalibrationConfig(
            method="slice_thresholds",
            version="v1",
            parameters={},
        )
        cal = Calibrator(config)
        assert cal.calibrate(0.42) == pytest.approx(0.42)


class TestUnknownMethod:
    def test_raises_value_error(self) -> None:
        config = CalibrationConfig(
            method="unknown_method",
            version="v1",
            parameters={},
        )
        cal = Calibrator(config)
        with pytest.raises(ValueError, match="Unknown calibration method"):
            cal.calibrate(0.5)


class TestFitTemperatureScaling:
    def test_well_calibrated_scores_produce_valid_calibrator(self) -> None:
        scores = [0.1, 0.2, 0.8, 0.9]
        labels = [0, 0, 1, 1]
        cal = fit_temperature_scaling(scores, labels, version="test")
        temp = cal._params["temperature"]
        # Temperature must be positive and within search range
        assert 0.1 <= temp < 5.0
        # Calibrated scores should preserve ordering
        c_low = cal.calibrate(0.2)
        c_high = cal.calibrate(0.8)
        assert c_low < c_high

    def test_overconfident_scores_yield_higher_temperature(self) -> None:
        # Overconfident: scores are extreme but labels are mixed
        scores = [0.01, 0.02, 0.98, 0.99]
        labels = [0, 1, 1, 0]  # some wrong
        cal = fit_temperature_scaling(scores, labels, version="test")
        temp = cal._params["temperature"]
        assert temp > 1.0  # needs cooling

    def test_version_preserved(self) -> None:
        scores = [0.3, 0.7]
        labels = [0, 1]
        cal = fit_temperature_scaling(scores, labels, version="v42")
        assert cal.version == "v42"


class TestSaveLoad:
    def test_roundtrip(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "cal.json"
        config = CalibrationConfig(
            method="temperature_scaling",
            version="v3",
            parameters={"temperature": 1.5},
        )
        original = Calibrator(config)
        original.save(path)

        loaded = Calibrator.from_file(path)
        assert loaded.version == "v3"
        assert loaded._method == "temperature_scaling"
        assert loaded._params["temperature"] == 1.5
        # Functional equivalence
        assert loaded.calibrate(0.7) == pytest.approx(
            original.calibrate(0.7)
        )

    def test_from_file_reads_json(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "test_cal.json"
        data = {
            "method": "isotonic",
            "version": "v1",
            "parameters": {
                "breakpoints": [[0.0, 0.1], [1.0, 0.9]]
            },
        }
        path.write_text(json.dumps(data), encoding="utf-8")
        cal = Calibrator.from_file(path)
        assert cal.version == "v1"
        assert cal.calibrate(0.5) == pytest.approx(0.5)

    def test_save_creates_parent_dirs(self, tmp_path: object) -> None:
        from pathlib import Path

        path = Path(str(tmp_path)) / "sub" / "dir" / "cal.json"
        config = CalibrationConfig(
            method="slice_thresholds",
            version="v1",
            parameters={},
        )
        Calibrator(config).save(path)
        assert path.exists()
