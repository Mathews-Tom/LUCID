"""Tests for ModelManager lifecycle management."""

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pytest

from lucid.config import LUCIDConfig
from lucid.models.manager import ModelManager


@pytest.fixture
def manager(default_config: LUCIDConfig) -> ModelManager:
    """ModelManager wired to the default balanced config."""
    return ModelManager(default_config)


class TestInitializeDetector:
    """initialize_detector() creates and caches a LUCIDDetector."""

    @patch("lucid.detector.base.LUCIDDetector")
    def test_creates_and_caches_instance(
        self, mock_cls: MagicMock, manager: ModelManager
    ) -> None:
        """Returned value is the mock instance; internal cache is set."""
        result = manager.initialize_detector()

        mock_cls.assert_called_once_with(manager._config.detection)
        assert result is mock_cls.return_value
        assert manager._detector is mock_cls.return_value


class TestDetectorProperty:
    """detector property accessor."""

    @patch("lucid.detector.base.LUCIDDetector")
    def test_returns_cached_instance(
        self, mock_cls: MagicMock, manager: ModelManager
    ) -> None:
        """Property returns the previously initialized detector."""
        manager.initialize_detector()
        assert manager.detector is mock_cls.return_value

    def test_raises_when_not_initialized(self, manager: ModelManager) -> None:
        """Property raises RuntimeError before initialize_detector()."""
        with pytest.raises(RuntimeError, match="Detector not initialized"):
            _ = manager.detector


class TestInitializeHumanizer:
    """initialize_humanizer() requires detector and creates humanizer."""

    def test_raises_without_detector(self, manager: ModelManager) -> None:
        """RuntimeError when detector has not been initialized first."""
        with pytest.raises(
            RuntimeError, match="Detector must be initialized before humanizer"
        ):
            manager.initialize_humanizer()

    @patch("lucid.humanizer.LUCIDHumanizer")
    @patch("lucid.detector.base.LUCIDDetector")
    def test_succeeds_after_detector_initialized(
        self,
        mock_detector_cls: MagicMock,
        mock_humanizer_cls: MagicMock,
        manager: ModelManager,
    ) -> None:
        """Humanizer is created with the correct arguments after detector init."""
        manager.initialize_detector()
        result = manager.initialize_humanizer()

        mock_humanizer_cls.assert_called_once_with(
            manager._config.humanizer,
            manager._config.ollama,
            mock_detector_cls.return_value,
            manager._config.general.profile,
        )
        assert result is mock_humanizer_cls.return_value
        assert manager._humanizer is mock_humanizer_cls.return_value


class TestHumanizerProperty:
    """humanizer property accessor."""

    def test_raises_when_not_initialized(self, manager: ModelManager) -> None:
        """Property raises RuntimeError before initialize_humanizer()."""
        with pytest.raises(RuntimeError, match="Humanizer not initialized"):
            _ = manager.humanizer


class TestInitializeEvaluator:
    """initialize_evaluator() creates and caches a LUCIDEvaluator."""

    @patch("lucid.evaluator.LUCIDEvaluator")
    def test_creates_and_caches_instance(
        self, mock_cls: MagicMock, manager: ModelManager
    ) -> None:
        """Returned value is the mock instance; internal cache is set."""
        result = manager.initialize_evaluator()

        mock_cls.assert_called_once_with(
            manager._config.evaluator,
            manager._config.general.profile,
        )
        assert result is mock_cls.return_value
        assert manager._evaluator is mock_cls.return_value


class TestEvaluatorProperty:
    """evaluator property accessor."""

    def test_raises_when_not_initialized(self, manager: ModelManager) -> None:
        """Property raises RuntimeError before initialize_evaluator()."""
        with pytest.raises(RuntimeError, match="Evaluator not initialized"):
            _ = manager.evaluator


class TestReleaseDetectionModels:
    """release_detection_models() unloads binoculars and checks memory."""

    @patch("lucid.models.manager.psutil")
    @patch("lucid.detector.base.LUCIDDetector")
    def test_calls_unload_binoculars(
        self,
        mock_detector_cls: MagicMock,
        mock_psutil: MagicMock,
        manager: ModelManager,
    ) -> None:
        """Calls unload_binoculars on the detector."""
        mock_psutil.virtual_memory.return_value.available = 8 * 1024**3
        manager.initialize_detector()
        manager.release_detection_models()
        mock_detector_cls.return_value.unload_binoculars.assert_called_once()

    @patch("lucid.models.manager.psutil")
    def test_noop_without_detector(
        self,
        mock_psutil: MagicMock,
        manager: ModelManager,
    ) -> None:
        """No error when detector not initialized."""
        mock_psutil.virtual_memory.return_value.available = 8 * 1024**3
        manager.release_detection_models()  # Should not raise

    @patch("lucid.models.manager.psutil")
    @patch("lucid.detector.base.LUCIDDetector")
    def test_low_memory_warning(
        self,
        mock_detector_cls: MagicMock,
        mock_psutil: MagicMock,
        manager: ModelManager,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Logs warning when available memory < 4GB."""
        mock_psutil.virtual_memory.return_value.available = 2 * 1024**3  # 2GB
        manager.initialize_detector()
        with caplog.at_level(logging.WARNING):
            manager.release_detection_models()
        assert "Low memory" in caplog.text


class TestShutdownCallsUnload:
    """shutdown() calls unload_binoculars before clearing refs."""

    @patch("lucid.detector.base.LUCIDDetector")
    def test_shutdown_unloads_binoculars(
        self,
        mock_detector_cls: MagicMock,
        manager: ModelManager,
    ) -> None:
        """shutdown() calls unload_binoculars before nulling."""
        manager.initialize_detector()
        manager.shutdown()
        mock_detector_cls.return_value.unload_binoculars.assert_called_once()


class TestShutdown:
    """shutdown() clears all cached references."""

    @patch("lucid.evaluator.LUCIDEvaluator")
    @patch("lucid.humanizer.LUCIDHumanizer")
    @patch("lucid.detector.base.LUCIDDetector")
    def test_clears_all_refs_and_properties_raise(
        self,
        _mock_detector: MagicMock,
        _mock_humanizer: MagicMock,
        _mock_evaluator: MagicMock,
        manager: ModelManager,
    ) -> None:
        """After shutdown, all three properties raise RuntimeError."""
        manager.initialize_detector()
        manager.initialize_humanizer()
        manager.initialize_evaluator()

        manager.shutdown()

        assert manager._detector is None
        assert manager._humanizer is None
        assert manager._evaluator is None

        with pytest.raises(RuntimeError):
            _ = manager.detector
        with pytest.raises(RuntimeError):
            _ = manager.humanizer
        with pytest.raises(RuntimeError):
            _ = manager.evaluator
