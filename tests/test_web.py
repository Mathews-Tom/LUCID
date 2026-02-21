"""Tests for the Gradio web interface functions."""
from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lucid.models.results import DetectionResult, DocumentResult


class TestRunDetection:
    """Test run_detection function."""

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_returns_text_and_json(self, mock_pipeline_cls: MagicMock, tmp_path: Path) -> None:
        """Detection returns text and JSON reports."""
        from lucid.web import run_detection

        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run_detect_only.return_value = DocumentResult(
            input_path=str(tmp_path / "test.md"),
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

        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\n\nSome content.\n", encoding="utf-8")

        text_report, json_report = run_detection(str(test_file), "balanced")

        assert "LUCID Pipeline Report" in text_report
        assert "lucid_version" in json_report


class TestRunPipeline:
    """Test run_pipeline function."""

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_returns_report_and_path(self, mock_pipeline_cls: MagicMock, tmp_path: Path) -> None:
        """Pipeline returns text report and output path."""
        from lucid.web import run_pipeline

        output_file = tmp_path / "output.md"
        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = DocumentResult(
            input_path=str(tmp_path / "test.md"),
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

        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\n\nSome content.\n", encoding="utf-8")

        text_report, _output_path = run_pipeline(str(test_file), "balanced", True)

        assert "LUCID Pipeline Report" in text_report

    @patch("lucid.pipeline.LUCIDPipeline")
    def test_no_adversarial(self, mock_pipeline_cls: MagicMock, tmp_path: Path) -> None:
        """Pipeline runs with adversarial disabled."""
        from lucid.web import run_pipeline

        mock_pipeline = mock_pipeline_cls.return_value
        mock_pipeline.run.return_value = DocumentResult(
            input_path=str(tmp_path / "test.md"),
            format="markdown",
            summary_stats={},
        )

        test_file = tmp_path / "test.md"
        test_file.write_text("# Test\n\nContent.\n", encoding="utf-8")

        text_report, _ = run_pipeline(str(test_file), "fast", False)
        assert isinstance(text_report, str)


class TestGradioImport:
    """Test gradio import guard."""

    def test_ensure_gradio_error_message(self) -> None:
        """_ensure_gradio raises RuntimeError with install instructions."""
        from unittest.mock import patch as mock_patch

        # Test that the error message is correct when gradio is missing
        with mock_patch.dict("sys.modules", {"gradio": None}):
            from lucid.web import _ensure_gradio
            with pytest.raises(RuntimeError, match="Gradio is not installed"):
                _ensure_gradio()
