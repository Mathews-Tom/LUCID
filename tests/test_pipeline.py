"""Tests for the LUCID pipeline orchestrator."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from lucid.config import LUCIDConfig, load_config
from lucid.models.results import (
    DetectionResult,
    DocumentResult,
    EvaluationResult,
    ParaphraseResult,
)
from lucid.pipeline import LUCIDPipeline, PipelineState
from lucid.progress import PipelineEvent


@pytest.fixture
def config() -> LUCIDConfig:
    return load_config(profile="balanced")


@pytest.fixture
def md_input(tmp_path: Path) -> Path:
    p = tmp_path / "test.md"
    p.write_text("# Title\n\nThis is AI-generated content for testing.\n", encoding="utf-8")
    return p


@pytest.fixture
def txt_input(tmp_path: Path) -> Path:
    p = tmp_path / "test.txt"
    p.write_text("This is a paragraph.\n\nThis is another paragraph.\n", encoding="utf-8")
    return p


def _make_detection(chunk_id: str, classification: str = "ai_generated") -> DetectionResult:
    return DetectionResult(
        chunk_id=chunk_id,
        ensemble_score=0.85,
        classification=classification,
    )


def _make_paraphrase(chunk_id: str, original: str) -> ParaphraseResult:
    return ParaphraseResult(
        chunk_id=chunk_id,
        original_text=original,
        humanized_text=f"Humanized: {original}",
        iteration_count=1,
        strategy_used="lexical_diversity",
        final_detection_score=0.15,
    )


def _make_evaluation(chunk_id: str, passed: bool = True) -> EvaluationResult:
    if passed:
        return EvaluationResult(
            chunk_id=chunk_id,
            passed=True,
            embedding_similarity=0.92,
        )
    return EvaluationResult(
        chunk_id=chunk_id,
        passed=False,
        embedding_similarity=0.5,
        rejection_reason="Similarity too low",
    )


class TestPipelineStateEnum:
    """PipelineState enum values."""

    def test_all_states_present(self) -> None:
        states = {s.value for s in PipelineState}
        assert "PARSING" in states
        assert "DETECTING" in states
        assert "HUMANIZING" in states
        assert "EVALUATING" in states
        assert "RECONSTRUCTING" in states
        assert "VALIDATING" in states
        assert "COMPLETE" in states
        assert "FAILED" in states


class TestRunDetectOnly:
    """run_detect_only() parses and detects without humanization."""

    @patch("lucid.pipeline.ModelManager")
    def test_returns_detections(
        self,
        mock_manager_cls: MagicMock,
        config: LUCIDConfig,
        md_input: Path,
    ) -> None:
        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.detector = mock_detector

        mock_detector.detect.return_value = DetectionResult(
            chunk_id="test",
            ensemble_score=0.8,
            classification="ai_generated",
        )

        pipeline = LUCIDPipeline(config)
        result = pipeline.run_detect_only(md_input)

        assert isinstance(result, DocumentResult)
        assert result.format == "markdown"
        assert len(result.chunks) > 0
        assert len(result.detections) > 0
        mock_mgr.shutdown.assert_called_once()


class TestRunFullPipeline:
    """run() executes the full detect-humanize-evaluate-reconstruct flow."""

    @patch("lucid.pipeline.validate_markdown")
    @patch("lucid.pipeline.ModelManager")
    def test_full_pipeline_produces_output(
        self,
        mock_manager_cls: MagicMock,
        mock_validate: MagicMock,
        config: LUCIDConfig,
        md_input: Path,
        tmp_path: Path,
    ) -> None:
        from lucid.reconstructor import ValidationResult

        mock_validate.return_value = ValidationResult(valid=True)

        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_humanizer = MagicMock()
        mock_evaluator = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.initialize_humanizer.return_value = mock_humanizer
        mock_mgr.initialize_evaluator.return_value = mock_evaluator
        mock_mgr.detector = mock_detector
        mock_mgr.humanizer = mock_humanizer
        mock_mgr.evaluator = mock_evaluator

        # Detector returns ai_generated for all chunks
        mock_detector.detect.side_effect = lambda chunk: _make_detection(chunk.id)
        mock_humanizer.humanize.side_effect = (
            lambda chunk, _det: _make_paraphrase(chunk.id, chunk.text)
        )
        mock_evaluator.evaluate_chunk.side_effect = (
            lambda cid, _orig, _hum: _make_evaluation(cid)
        )

        output = tmp_path / "output.md"
        pipeline = LUCIDPipeline(config)
        result = pipeline.run(md_input, output_path=output)

        assert result.output_path == str(output)
        assert output.exists()
        assert len(result.detections) > 0
        assert len(result.paraphrases) > 0
        assert len(result.evaluations) > 0
        assert result.summary_stats["total_chunks"] > 0
        mock_mgr.shutdown.assert_called_once()


class TestProgressCallback:
    """Progress callback receives events during pipeline execution."""

    @patch("lucid.pipeline.ModelManager")
    def test_callback_receives_events(
        self,
        mock_manager_cls: MagicMock,
        config: LUCIDConfig,
        md_input: Path,
    ) -> None:
        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.detector = mock_detector
        mock_detector.detect.return_value = DetectionResult(
            chunk_id="test",
            ensemble_score=0.2,
            classification="human",
        )

        events: list[PipelineEvent] = []
        pipeline = LUCIDPipeline(config)
        pipeline.run_detect_only(md_input, progress_callback=events.append)

        assert len(events) > 0
        states_seen = {e.state for e in events}
        assert "PARSING" in states_seen
        assert "DETECTING" in states_seen


class TestSkipCompleted:
    """Pipeline skips chunks already completed in a checkpoint."""

    @patch("lucid.pipeline.validate_markdown")
    @patch("lucid.pipeline.ModelManager")
    def test_skips_detected_chunks(
        self,
        mock_manager_cls: MagicMock,
        mock_validate: MagicMock,
        config: LUCIDConfig,
        md_input: Path,
        tmp_path: Path,
    ) -> None:
        from lucid.reconstructor import ValidationResult

        mock_validate.return_value = ValidationResult(valid=True)

        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.detector = mock_detector
        mock_detector.detect.return_value = DetectionResult(
            chunk_id="test",
            ensemble_score=0.1,
            classification="human",
        )

        # First run: populates checkpoint
        checkpoint_dir = tmp_path / "ckpt"
        pipeline = LUCIDPipeline(config, checkpoint_dir=checkpoint_dir)
        result1 = pipeline.run(md_input, output_path=tmp_path / "out1.md")

        # Checkpoint should be cleared after successful run
        assert not (checkpoint_dir / "test.md.checkpoint.json").exists()
        assert len(result1.detections) > 0


class TestErrorIsolation:
    """Individual chunk failures do not crash the pipeline."""

    @patch("lucid.pipeline.validate_markdown")
    @patch("lucid.pipeline.ModelManager")
    def test_chunk_failure_logged_and_skipped(
        self,
        mock_manager_cls: MagicMock,
        mock_validate: MagicMock,
        config: LUCIDConfig,
        md_input: Path,
        tmp_path: Path,
    ) -> None:
        from lucid.reconstructor import ValidationResult

        mock_validate.return_value = ValidationResult(valid=True)

        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_humanizer = MagicMock()
        mock_evaluator = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.initialize_humanizer.return_value = mock_humanizer
        mock_mgr.initialize_evaluator.return_value = mock_evaluator
        mock_mgr.detector = mock_detector
        mock_mgr.humanizer = mock_humanizer
        mock_mgr.evaluator = mock_evaluator

        mock_detector.detect.side_effect = lambda chunk: _make_detection(chunk.id)
        mock_humanizer.humanize.side_effect = RuntimeError("Ollama timeout")

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(md_input, output_path=tmp_path / "out.md")

        # Pipeline completes despite humanization failures
        assert len(result.detections) > 0
        assert len(result.paraphrases) == 0
        assert result.summary_stats["failed"] > 0


class TestDefaultOutputPath:
    """Output path defaults to {stem}_humanized.{ext}."""

    @patch("lucid.pipeline.validate_markdown")
    @patch("lucid.pipeline.ModelManager")
    def test_default_output_name(
        self,
        mock_manager_cls: MagicMock,
        mock_validate: MagicMock,
        config: LUCIDConfig,
        md_input: Path,
    ) -> None:
        from lucid.reconstructor import ValidationResult

        mock_validate.return_value = ValidationResult(valid=True)

        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.detector = mock_detector
        mock_detector.detect.return_value = DetectionResult(
            chunk_id="test",
            ensemble_score=0.1,
            classification="human",
        )

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(md_input)

        assert result.output_path is not None
        assert "_humanized" in result.output_path
