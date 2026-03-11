"""Tests for lucid.core.errors exception hierarchy."""

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


class TestExceptionHierarchy:
    def test_all_inherit_from_lucid_error(self) -> None:
        for exc_cls in (
            ConfigError,
            DetectorError,
            DetectorInitError,
            DetectionError,
            BinocularsUnavailableError,
            TransformError,
            MetricError,
            BenchmarkError,
        ):
            assert issubclass(exc_cls, LUCIDError)

    def test_detector_sub_hierarchy(self) -> None:
        assert issubclass(DetectorInitError, DetectorError)
        assert issubclass(DetectionError, DetectorError)
        assert issubclass(BinocularsUnavailableError, DetectorError)

    def test_catch_lucid_error_catches_subtypes(self) -> None:
        for exc_cls in (ConfigError, DetectorInitError, TransformError, MetricError):
            try:
                raise exc_cls("test")
            except LUCIDError:
                pass

    def test_catch_detector_error_catches_subtypes(self) -> None:
        for exc_cls in (DetectorInitError, DetectionError, BinocularsUnavailableError):
            try:
                raise exc_cls("test")
            except DetectorError:
                pass


class TestBackwardCompatibility:
    def test_import_from_detector_package(self) -> None:
        from lucid.detector import (
            BinocularsUnavailableError as Bino,
            DetectionError as DE,
            DetectorError as DErr,
            DetectorInitError as DI,
        )

        assert issubclass(DErr, LUCIDError)
        assert issubclass(DI, DErr)
        assert issubclass(DE, DErr)
        assert issubclass(Bino, DErr)

    def test_same_class_identity(self) -> None:
        from lucid.detector import DetectorError as FromDetector

        assert FromDetector is DetectorError
