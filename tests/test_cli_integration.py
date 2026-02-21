"""CLI integration tests using Click's CliRunner."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from lucid.cli import main
from lucid.models.results import DetectionResult, DocumentResult


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def md_file(tmp_path: Path) -> Path:
    p = tmp_path / "input.md"
    p.write_text("# Title\n\nSome content to detect.\n", encoding="utf-8")
    return p


@pytest.fixture
def batch_dir(tmp_path: Path) -> Path:
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.md").write_text("# A\n\nContent A.\n", encoding="utf-8")
    (d / "b.txt").write_text("Content B.\n", encoding="utf-8")
    return d


class TestMainGroup:
    """Top-level CLI group."""

    def test_version(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--version"])
        assert result.exit_code == 0
        assert "lucid" in result.output.lower()

    def test_help(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "detect" in result.output
        assert "humanize" in result.output
        assert "pipeline" in result.output
        assert "config" in result.output
        assert "models" in result.output


class TestDetectCommand:
    """detect subcommand."""

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_detect_text_output(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        md_file: Path,
    ) -> None:
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_detect_only.return_value = DocumentResult(
            input_path=str(md_file),
            format="markdown",
            detections=[
                DetectionResult(
                    chunk_id="c1",
                    ensemble_score=0.8,
                    classification="ai_generated",
                ),
            ],
            summary_stats={
                "total_chunks": 1,
                "prose_chunks": 1,
                "ai_detected": 1,
            },
        )

        result = runner.invoke(main, ["detect", str(md_file), "--output-format", "text"])

        assert result.exit_code == 0
        assert "LUCID Pipeline Report" in result.output

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_detect_json_output(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        md_file: Path,
    ) -> None:
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_detect_only.return_value = DocumentResult(
            input_path=str(md_file),
            format="markdown",
            summary_stats={},
        )

        result = runner.invoke(main, ["detect", str(md_file), "--output-format", "json"])

        assert result.exit_code == 0
        assert '"lucid_version"' in result.output


class TestPipelineCommand:
    """pipeline subcommand."""

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_pipeline_basic(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        md_file: Path,
        tmp_path: Path,
    ) -> None:
        output_file = tmp_path / "out.md"
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = DocumentResult(
            input_path=str(md_file),
            format="markdown",
            output_path=str(output_file),
            summary_stats={
                "total_chunks": 1,
                "prose_chunks": 1,
                "ai_detected": 0,
                "humanized": 0,
                "eval_passed": 0,
                "eval_failed": 0,
                "failed": 0,
            },
        )

        result = runner.invoke(
            main,
            ["pipeline", str(md_file), "-o", str(output_file), "--no-resume"],
        )

        assert result.exit_code == 0
        assert "Output:" in result.output

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_pipeline_with_report(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        md_file: Path,
        tmp_path: Path,
    ) -> None:
        report_file = tmp_path / "report.json"
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = DocumentResult(
            input_path=str(md_file),
            format="markdown",
            output_path=str(tmp_path / "out.md"),
            summary_stats={},
        )

        result = runner.invoke(
            main,
            [
                "pipeline",
                str(md_file),
                "--report",
                str(report_file),
                "--output-format",
                "json",
                "--no-resume",
            ],
        )

        assert result.exit_code == 0


class TestConfigCommand:
    """config subcommand."""

    def test_config_show(self, runner: CliRunner) -> None:
        result = runner.invoke(main, ["config"])
        assert result.exit_code == 0
        assert "balanced" in result.output


class TestModelsCommand:
    """models subcommand."""

    @patch("lucid.models.download.ModelDownloader")
    def test_models_check(
        self,
        mock_downloader_cls: MagicMock,
        runner: CliRunner,
    ) -> None:
        from lucid.models.download import ModelStatus

        mock_dl = mock_downloader_cls.return_value
        mock_dl.check_all.return_value = [
            ModelStatus(name="roberta-base", source="huggingface", available=True),
            ModelStatus(name="qwen2.5:7b", source="ollama", available=False),
        ]

        result = runner.invoke(main, ["models"])

        assert result.exit_code == 0
        assert "roberta-base" in result.output
        assert "qwen2.5:7b" in result.output


class TestBatchMode:
    """Directory input processes all supported files."""

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_batch_detect(
        self,
        mock_pipeline_cls: MagicMock,
        runner: CliRunner,
        batch_dir: Path,
    ) -> None:
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_detect_only.return_value = DocumentResult(
            input_path="batch",
            format="markdown",
            summary_stats={},
        )

        result = runner.invoke(main, ["detect", str(batch_dir)])

        assert result.exit_code == 0
        # Should process 2 files (a.md and b.txt)
        assert mock_pipeline.run_detect_only.call_count == 2
