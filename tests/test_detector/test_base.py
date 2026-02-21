"""Unit tests for the LUCIDDetector orchestrator.

All tier detectors are mocked — no model downloads, no network access.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from lucid.config import (
    DetectionConfig,
    DetectionThresholdsConfig,
    EnsembleWeightsConfig,
    EnsembleWeightsWithBinocularsConfig,
)
from lucid.core.protocols import Detector
from lucid.detector.base import LUCIDDetector
from lucid.models.results import DetectionResult
from lucid.parser.chunk import ProseChunk


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "The results demonstrate significant improvements.") -> ProseChunk:
    """Create a minimal ProseChunk for testing."""
    return ProseChunk(
        text=text,
        start_pos=0,
        end_pos=len(text),
        protected_text=text,
    )


def _fast_config() -> DetectionConfig:
    """Config with no statistical, no binoculars."""
    return DetectionConfig(
        use_statistical=False,
        use_binoculars=False,
        thresholds=DetectionThresholdsConfig(ambiguity_triggers_binoculars=False),
    )


def _balanced_config() -> DetectionConfig:
    """Config with statistical enabled, no binoculars."""
    return DetectionConfig(
        use_statistical=True,
        use_binoculars=False,
        thresholds=DetectionThresholdsConfig(ambiguity_triggers_binoculars=False),
    )


def _binoculars_config() -> DetectionConfig:
    """Config with all three tiers enabled."""
    return DetectionConfig(
        use_statistical=True,
        use_binoculars=True,
        thresholds=DetectionThresholdsConfig(ambiguity_triggers_binoculars=False),
    )


def _ambiguity_config(
    human_max: float = 0.30, ai_min: float = 0.65
) -> DetectionConfig:
    """Config where ambiguous scores trigger Binoculars."""
    return DetectionConfig(
        use_statistical=True,
        use_binoculars=False,
        thresholds=DetectionThresholdsConfig(
            human_max=human_max,
            ai_min=ai_min,
            ambiguity_triggers_binoculars=True,
        ),
    )


def _patch_tiers(
    roberta_score: float = 0.8,
    statistical_score: float | None = 0.6,
    binoculars_score: float | None = None,
    statistical_raises: bool = False,
    binoculars_raises: bool = False,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    """Return (mock_roberta, mock_statistical, mock_binoculars) with preset return values."""
    mock_roberta = MagicMock()
    mock_roberta.detect_text.return_value = roberta_score

    mock_statistical = MagicMock()
    if statistical_raises:
        mock_statistical.score.side_effect = RuntimeError("stat fail")
    else:
        mock_statistical.score.return_value = statistical_score
    mock_statistical.extract_features.return_value = {"ttr": 0.5}

    mock_binoculars = MagicMock()
    if binoculars_raises:
        mock_binoculars.score.side_effect = RuntimeError("bino fail")
    else:
        mock_binoculars.score.return_value = binoculars_score

    return mock_roberta, mock_statistical, mock_binoculars


# ---------------------------------------------------------------------------
# Protocol compliance
# ---------------------------------------------------------------------------


class TestDetectorProtocol:
    """LUCIDDetector must satisfy the Detector protocol."""

    def test_isinstance_detector_protocol(self) -> None:
        """LUCIDDetector(config) must pass isinstance check against Detector protocol."""
        config = _fast_config()
        with (
            patch("lucid.detector.base.RobertaDetector"),
        ):
            detector = LUCIDDetector(config)
        assert isinstance(detector, Detector)

    def test_has_detect_method(self) -> None:
        """LUCIDDetector must expose a callable detect attribute."""
        config = _fast_config()
        with patch("lucid.detector.base.RobertaDetector"):
            detector = LUCIDDetector(config)
        assert callable(detector.detect)


# ---------------------------------------------------------------------------
# Fast profile (Tier 1 only)
# ---------------------------------------------------------------------------


class TestFastProfile:
    """Fast profile: RoBERTa only, no statistical, no binoculars."""

    def test_detect_returns_detection_result(self) -> None:
        """detect() returns a DetectionResult with correct chunk_id."""
        config = _fast_config()
        chunk = _make_chunk()

        with (
            patch("lucid.detector.base.RobertaDetector") as MockRoberta,
        ):
            mock_roberta = MockRoberta.return_value
            mock_roberta.detect_text.return_value = 0.75
            detector = LUCIDDetector(config)
            result = detector.detect(chunk)

        assert isinstance(result, DetectionResult)
        assert result.chunk_id == chunk.id

    def test_statistical_score_is_none(self) -> None:
        """Fast profile produces statistical_score=None."""
        config = _fast_config()
        chunk = _make_chunk()

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            MockRoberta.return_value.detect_text.return_value = 0.75
            detector = LUCIDDetector(config)
            result = detector.detect(chunk)

        assert result.statistical_score is None

    def test_binoculars_score_is_none(self) -> None:
        """Fast profile produces binoculars_score=None."""
        config = _fast_config()
        chunk = _make_chunk()

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            MockRoberta.return_value.detect_text.return_value = 0.75
            detector = LUCIDDetector(config)
            result = detector.detect(chunk)

        assert result.binoculars_score is None

    def test_roberta_score_propagated(self) -> None:
        """roberta_score in result matches the mocked tier score."""
        config = _fast_config()
        chunk = _make_chunk()

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            MockRoberta.return_value.detect_text.return_value = 0.42
            detector = LUCIDDetector(config)
            result = detector.detect(chunk)

        assert result.roberta_score == pytest.approx(0.42)

    def test_classification_ai_generated(self) -> None:
        """Score above ai_min threshold classifies as ai_generated."""
        config = _fast_config()
        chunk = _make_chunk()

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            MockRoberta.return_value.detect_text.return_value = 0.9
            detector = LUCIDDetector(config)
            result = detector.detect(chunk)

        assert result.classification == "ai_generated"

    def test_classification_human(self) -> None:
        """Score below human_max threshold classifies as human."""
        config = _fast_config()
        chunk = _make_chunk()

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            MockRoberta.return_value.detect_text.return_value = 0.1
            detector = LUCIDDetector(config)
            result = detector.detect(chunk)

        assert result.classification == "human"


# ---------------------------------------------------------------------------
# Balanced profile (Tier 1 + Tier 2)
# ---------------------------------------------------------------------------


class TestBalancedProfile:
    """Balanced profile: RoBERTa + statistical, no binoculars."""

    def test_statistical_score_present(self) -> None:
        """balanced profile produces a non-None statistical_score when tier 2 succeeds."""
        config = _balanced_config()
        chunk = _make_chunk()

        with (
            patch("lucid.detector.base.RobertaDetector") as MockRoberta,
            patch("lucid.detector.base.StatisticalDetector"),
        ):
            MockRoberta.return_value.detect_text.return_value = 0.7
            detector = LUCIDDetector(config)

            mock_stat = MagicMock()
            mock_stat.score.return_value = 0.6
            mock_stat.extract_features.return_value = {"ttr": 0.45}
            detector._statistical = mock_stat

            result = detector.detect(chunk)

        assert result.statistical_score == pytest.approx(0.6)

    def test_feature_details_populated(self) -> None:
        """feature_details dict is set from extract_features when tier 2 runs."""
        config = _balanced_config()
        chunk = _make_chunk()

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            mock_roberta = MockRoberta.return_value
            mock_roberta.detect_text.return_value = 0.5

            detector = LUCIDDetector(config)
            detector._roberta = mock_roberta

            mock_stat = MagicMock()
            mock_stat.score.return_value = 0.55
            mock_stat.extract_features.return_value = {"burstiness": 0.3, "ttr": 0.6}
            detector._statistical = mock_stat

            result = detector.detect(chunk)

        assert result.feature_details == {"burstiness": 0.3, "ttr": 0.6}

    def test_statistical_failure_falls_back_gracefully(self) -> None:
        """When statistical tier raises, result still contains roberta_score."""
        config = _balanced_config()
        chunk = _make_chunk()

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            mock_roberta = MockRoberta.return_value
            mock_roberta.detect_text.return_value = 0.8

            detector = LUCIDDetector(config)
            detector._roberta = mock_roberta

            mock_stat = MagicMock()
            mock_stat.score.side_effect = RuntimeError("spacy unavailable")
            detector._statistical = mock_stat

            result = detector.detect(chunk)

        assert result.roberta_score == pytest.approx(0.8)
        assert result.statistical_score is None

    def test_statistical_none_return_propagated(self) -> None:
        """When statistical.score() returns None (short text), statistical_score is None."""
        config = _balanced_config()
        chunk = _make_chunk("Short.")

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            mock_roberta = MockRoberta.return_value
            mock_roberta.detect_text.return_value = 0.5

            detector = LUCIDDetector(config)
            detector._roberta = mock_roberta

            mock_stat = MagicMock()
            mock_stat.score.return_value = None
            mock_stat.extract_features.return_value = {}
            detector._statistical = mock_stat

            result = detector.detect(chunk)

        assert result.statistical_score is None


# ---------------------------------------------------------------------------
# _should_run_binoculars logic
# ---------------------------------------------------------------------------


class TestShouldRunBinoculars:
    """_should_run_binoculars routing logic.

    All tests construct a detector with all tier constructors mocked, then
    override _binoculars directly on the instance to control the condition.
    """

    def _make_detector(self, config: DetectionConfig) -> LUCIDDetector:
        """Build a LUCIDDetector with all tier classes mocked at module level."""
        with (
            patch("lucid.detector.base.RobertaDetector"),
            patch("lucid.detector.base.StatisticalDetector"),
            patch("lucid.detector.base.BinocularsDetector"),
        ):
            detector = LUCIDDetector(config)
        return detector

    def test_returns_false_when_binoculars_is_none(self) -> None:
        """Returns False when _binoculars is not initialized."""
        detector = self._make_detector(_fast_config())
        detector._binoculars = None
        assert detector._should_run_binoculars(0.5, 0.5) is False

    def test_returns_true_when_use_binoculars_true(self) -> None:
        """Returns True when config.use_binoculars=True regardless of scores."""
        detector = self._make_detector(_binoculars_config())
        detector._binoculars = MagicMock()
        assert detector._should_run_binoculars(0.1, 0.1) is True

    def test_ambiguity_triggers_binoculars_in_band(self) -> None:
        """Returns True when ensemble falls in the ambiguous band and flag is set."""
        detector = self._make_detector(_ambiguity_config(human_max=0.30, ai_min=0.65))
        detector._binoculars = MagicMock()
        # Score of 0.5 is ambiguous (0.30 < 0.5 < 0.65)
        assert detector._should_run_binoculars(0.5, 0.5) is True

    def test_ambiguity_does_not_trigger_below_human_max(self) -> None:
        """Returns False when Tier 1+2 ensemble is clearly human."""
        detector = self._make_detector(_ambiguity_config(human_max=0.30, ai_min=0.65))
        detector._binoculars = MagicMock()
        # Score of 0.1 is human (0.1 <= 0.30)
        assert detector._should_run_binoculars(0.1, 0.1) is False

    def test_ambiguity_does_not_trigger_above_ai_min(self) -> None:
        """Returns False when Tier 1+2 ensemble is clearly AI."""
        detector = self._make_detector(_ambiguity_config(human_max=0.30, ai_min=0.65))
        detector._binoculars = MagicMock()
        # Score of 0.9 is AI (0.9 >= 0.65)
        assert detector._should_run_binoculars(0.9, 0.9) is False

    def test_returns_false_when_flag_off_and_use_binoculars_false(self) -> None:
        """Returns False when ambiguity_triggers_binoculars=False and use_binoculars=False."""
        detector = self._make_detector(_balanced_config())
        detector._binoculars = MagicMock()
        assert detector._should_run_binoculars(0.5, 0.5) is False


# ---------------------------------------------------------------------------
# Binoculars fallback (BinocularsUnavailableError)
# ---------------------------------------------------------------------------


class TestBinocularsUnavailableFallback:
    """Binoculars errors must not propagate — fall back to Tier 1+2."""

    def test_binoculars_error_produces_result_without_bino_score(self) -> None:
        """When Binoculars raises, binoculars_score is None in result."""
        from lucid.detector import BinocularsUnavailableError

        config = _binoculars_config()
        chunk = _make_chunk()

        with (
            patch("lucid.detector.base.RobertaDetector") as MockRoberta,
            patch("lucid.detector.base.StatisticalDetector"),
            patch("lucid.detector.base.BinocularsDetector"),
        ):
            MockRoberta.return_value.detect_text.return_value = 0.8
            detector = LUCIDDetector(config)

            mock_stat = MagicMock()
            mock_stat.score.return_value = 0.7
            mock_stat.extract_features.return_value = {}
            detector._statistical = mock_stat

            mock_bino = MagicMock()
            mock_bino.score.side_effect = BinocularsUnavailableError("OOM")
            detector._binoculars = mock_bino

            result = detector.detect(chunk)

        assert result.binoculars_score is None
        assert result.roberta_score == pytest.approx(0.8)

    def test_binoculars_runtime_error_also_falls_back(self) -> None:
        """Generic RuntimeError from Binoculars also produces binoculars_score=None."""
        config = _binoculars_config()
        chunk = _make_chunk()

        with (
            patch("lucid.detector.base.RobertaDetector") as MockRoberta,
            patch("lucid.detector.base.StatisticalDetector"),
            patch("lucid.detector.base.BinocularsDetector"),
        ):
            MockRoberta.return_value.detect_text.return_value = 0.6
            detector = LUCIDDetector(config)
            detector._statistical = None

            mock_bino = MagicMock()
            mock_bino.score.side_effect = RuntimeError("CUDA OOM")
            detector._binoculars = mock_bino

            result = detector.detect(chunk)

        assert result.binoculars_score is None


# ---------------------------------------------------------------------------
# detect_batch
# ---------------------------------------------------------------------------


class TestDetectBatch:
    """detect_batch returns results in the same order as the input chunks."""

    def test_batch_length_matches_input(self) -> None:
        """detect_batch returns one result per input chunk."""
        config = _fast_config()
        chunks = [_make_chunk(f"Chunk {i}.") for i in range(5)]

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            MockRoberta.return_value.detect_text.return_value = 0.5
            detector = LUCIDDetector(config)
            results = detector.detect_batch(chunks)

        assert len(results) == 5

    def test_batch_chunk_ids_match_order(self) -> None:
        """detect_batch preserves input order by chunk_id."""
        config = _fast_config()
        chunks = [_make_chunk(f"Chunk text number {i}.") for i in range(3)]

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            MockRoberta.return_value.detect_text.return_value = 0.5
            detector = LUCIDDetector(config)
            results = detector.detect_batch(chunks)

        for chunk, result in zip(chunks, results):
            assert result.chunk_id == chunk.id

    def test_empty_batch_returns_empty_list(self) -> None:
        """detect_batch([]) returns []."""
        config = _fast_config()

        with patch("lucid.detector.base.RobertaDetector"):
            detector = LUCIDDetector(config)
            results = detector.detect_batch([])

        assert results == []

    def test_batch_result_types(self) -> None:
        """Every element in detect_batch result is a DetectionResult."""
        config = _fast_config()
        chunks = [_make_chunk("Some AI-generated prose here."), _make_chunk("Human written.")]

        with patch("lucid.detector.base.RobertaDetector") as MockRoberta:
            MockRoberta.return_value.detect_text.return_value = 0.5
            detector = LUCIDDetector(config)
            results = detector.detect_batch(chunks)

        assert all(isinstance(r, DetectionResult) for r in results)
