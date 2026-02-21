"""End-to-end integration tests for the LUCID pipeline.

All tests mock the ML models (detector, humanizer, evaluator) to avoid
requiring actual model downloads, while exercising the full pipeline
orchestration, checkpoint, output, and CLI layers.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from lucid.cli import main
from lucid.config import LUCIDConfig, load_config
from lucid.models.results import (
    DetectionResult,
    DocumentResult,
    EvaluationResult,
    ParaphraseResult,
)
from lucid.output import OutputFormatter
from lucid.pipeline import LUCIDPipeline

CORPUS_DIR = Path(__file__).parent.parent / "corpus"


@pytest.fixture
def config() -> LUCIDConfig:
    return load_config(profile="balanced")


def _mock_detector_detect(chunk: MagicMock) -> DetectionResult:
    return DetectionResult(
        chunk_id=chunk.id,
        ensemble_score=0.82,
        classification="ai_generated",
    )


def _mock_humanize(chunk: MagicMock, _det: MagicMock) -> ParaphraseResult:
    return ParaphraseResult(
        chunk_id=chunk.id,
        original_text=chunk.text,
        humanized_text=f"[humanized] {chunk.text}",
        iteration_count=2,
        strategy_used="lexical_diversity",
        final_detection_score=0.18,
    )


def _mock_evaluate(chunk_id: str, _orig: str, _hum: str) -> EvaluationResult:
    return EvaluationResult(
        chunk_id=chunk_id,
        passed=True,
        embedding_similarity=0.91,
    )


@pytest.mark.integration
class TestMarkdownPipeline:
    """Full pipeline on a Markdown test file."""

    @patch("lucid.pipeline.validate_markdown")
    @patch("lucid.pipeline.ModelManager")
    def test_markdown_full_flow(
        self,
        mock_manager_cls: MagicMock,
        mock_validate: MagicMock,
        config: LUCIDConfig,
        tmp_path: Path,
    ) -> None:
        from lucid.reconstructor import ValidationResult

        mock_validate.return_value = ValidationResult(valid=True)

        mock_mgr = mock_manager_cls.return_value
        mock_det = MagicMock()
        mock_hum = MagicMock()
        mock_eval = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_det
        mock_mgr.initialize_humanizer.return_value = mock_hum
        mock_mgr.initialize_evaluator.return_value = mock_eval
        mock_mgr.detector = mock_det
        mock_mgr.humanizer = mock_hum
        mock_mgr.evaluator = mock_eval

        mock_det.detect.side_effect = _mock_detector_detect
        mock_hum.humanize.side_effect = _mock_humanize
        mock_eval.evaluate_chunk.side_effect = _mock_evaluate

        input_file = CORPUS_DIR / "markdown" / "simple.md"
        output_file = tmp_path / "simple_humanized.md"

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(input_file, output_path=output_file)

        assert result.format == "markdown"
        assert result.output_path == str(output_file)
        assert output_file.exists()
        assert len(result.detections) > 0
        assert len(result.paraphrases) > 0
        assert all(e.passed for e in result.evaluations)
        assert result.summary_stats["total_chunks"] > 0
        mock_mgr.shutdown.assert_called_once()


@pytest.mark.integration
class TestLatexPipeline:
    """Full pipeline on a LaTeX test file."""

    @patch("lucid.pipeline.validate_latex")
    @patch("lucid.pipeline.ModelManager")
    def test_latex_full_flow(
        self,
        mock_manager_cls: MagicMock,
        mock_validate: MagicMock,
        config: LUCIDConfig,
        tmp_path: Path,
    ) -> None:
        from lucid.reconstructor import ValidationResult

        mock_validate.return_value = ValidationResult(valid=True)

        mock_mgr = mock_manager_cls.return_value
        mock_det = MagicMock()
        mock_hum = MagicMock()
        mock_eval = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_det
        mock_mgr.initialize_humanizer.return_value = mock_hum
        mock_mgr.initialize_evaluator.return_value = mock_eval
        mock_mgr.detector = mock_det
        mock_mgr.humanizer = mock_hum
        mock_mgr.evaluator = mock_eval

        mock_det.detect.side_effect = _mock_detector_detect
        mock_hum.humanize.side_effect = _mock_humanize
        mock_eval.evaluate_chunk.side_effect = _mock_evaluate

        input_file = CORPUS_DIR / "latex" / "simple.tex"
        output_file = tmp_path / "simple_humanized.tex"

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(input_file, output_path=output_file)

        assert result.format == "latex"
        assert output_file.exists()
        output_content = output_file.read_text(encoding="utf-8")
        assert "\\documentclass" in output_content
        assert "\\begin{document}" in output_content


@pytest.mark.integration
class TestDetectOnly:
    """Detect-only mode produces a report without humanization."""

    @patch("lucid.pipeline.ModelManager")
    def test_detect_only_json_report(
        self,
        mock_manager_cls: MagicMock,
        config: LUCIDConfig,
    ) -> None:
        mock_mgr = mock_manager_cls.return_value
        mock_det = MagicMock()
        mock_mgr.initialize_detector.return_value = mock_det
        mock_mgr.detector = mock_det
        mock_det.detect.side_effect = _mock_detector_detect

        input_file = CORPUS_DIR / "markdown" / "simple.md"
        pipeline = LUCIDPipeline(config)
        result = pipeline.run_detect_only(input_file)

        assert len(result.detections) > 0
        assert len(result.paraphrases) == 0

        formatter = OutputFormatter()
        json_output = formatter.format_json(result, config)
        assert '"lucid_version"' in json_output
        assert '"ai_generated"' in json_output


@pytest.mark.integration
class TestCheckpointResume:
    """Checkpoint save/resume across pipeline restarts."""

    @patch("lucid.pipeline.validate_markdown")
    @patch("lucid.pipeline.ModelManager")
    def test_checkpoint_resume_completes(
        self,
        mock_manager_cls: MagicMock,
        mock_validate: MagicMock,
        config: LUCIDConfig,
        tmp_path: Path,
    ) -> None:
        from lucid.reconstructor import ValidationResult

        mock_validate.return_value = ValidationResult(valid=True)

        mock_mgr = mock_manager_cls.return_value
        mock_det = MagicMock()
        mock_hum = MagicMock()
        mock_eval = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_det
        mock_mgr.initialize_humanizer.return_value = mock_hum
        mock_mgr.initialize_evaluator.return_value = mock_eval
        mock_mgr.detector = mock_det
        mock_mgr.humanizer = mock_hum
        mock_mgr.evaluator = mock_eval

        mock_det.detect.side_effect = _mock_detector_detect
        mock_hum.humanize.side_effect = _mock_humanize
        mock_eval.evaluate_chunk.side_effect = _mock_evaluate

        input_file = CORPUS_DIR / "markdown" / "simple.md"
        ckpt_dir = tmp_path / "checkpoints"
        output_file = tmp_path / "output.md"

        # First run: completes and clears checkpoint
        pipeline = LUCIDPipeline(config, checkpoint_dir=ckpt_dir)
        result = pipeline.run(input_file, output_path=output_file)

        assert result.output_path is not None
        assert not (ckpt_dir / "simple.md.checkpoint.json").exists()

        # Second run: no checkpoint to resume from, runs fresh
        pipeline2 = LUCIDPipeline(config, checkpoint_dir=ckpt_dir)
        result2 = pipeline2.run(input_file, output_path=tmp_path / "output2.md")

        assert len(result2.detections) == len(result.detections)


@pytest.mark.integration
class TestBatchProcessing:
    """Batch mode: directory input with mixed file types."""

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_batch_dir_processes_all_files(
        self,
        mock_pipeline_cls: MagicMock,
        tmp_path: Path,
    ) -> None:
        # Create batch directory
        batch_dir = tmp_path / "batch"
        batch_dir.mkdir()
        (batch_dir / "a.md").write_text("# A\n\nContent.\n", encoding="utf-8")
        (batch_dir / "b.tex").write_text(
            "\\documentclass{article}\n\\begin{document}\nText.\n\\end{document}\n",
            encoding="utf-8",
        )
        (batch_dir / "c.txt").write_text("Plain text.\n", encoding="utf-8")
        (batch_dir / "skip.py").write_text("# not a doc\n", encoding="utf-8")

        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_detect_only.return_value = DocumentResult(
            input_path="batch",
            format="markdown",
            summary_stats={},
        )

        cli_runner = CliRunner()
        result = cli_runner.invoke(main, ["detect", str(batch_dir)])

        assert result.exit_code == 0
        # 3 supported files (a.md, b.tex, c.txt) — skip.py ignored
        assert mock_pipeline.run_detect_only.call_count == 3


@pytest.mark.integration
class TestCLIDetectSubcommand:
    """CLI detect subcommand end-to-end."""

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_cli_detect_text(
        self,
        mock_pipeline_cls: MagicMock,
    ) -> None:
        mock_pipeline = mock_pipeline_cls.return_value
        input_file = CORPUS_DIR / "markdown" / "simple.md"
        mock_pipeline.run_detect_only.return_value = DocumentResult(
            input_path=str(input_file),
            format="markdown",
            detections=[
                DetectionResult(
                    chunk_id="x1",
                    ensemble_score=0.75,
                    classification="ai_generated",
                ),
            ],
            summary_stats={"total_chunks": 3, "prose_chunks": 3, "ai_detected": 1},
        )

        runner = CliRunner()
        result = runner.invoke(main, ["detect", str(input_file)])

        assert result.exit_code == 0
        assert "LUCID Pipeline Report" in result.output
