"""Tests for output formatting."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from lucid.config import LUCIDConfig, load_config
from lucid.models.results import (
    DetectionResult,
    DocumentResult,
    EvaluationResult,
    TransformResult,
)
from lucid.output import OutputFormatter
from lucid.parser.chunk import ProseChunk

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def config() -> LUCIDConfig:
    return load_config(profile="balanced")


@pytest.fixture
def formatter() -> OutputFormatter:
    return OutputFormatter()


@pytest.fixture
def sample_result() -> DocumentResult:
    """DocumentResult with detections, transforms, and evaluations."""
    chunk = ProseChunk(
        text="AI-generated text here",
        start_pos=0,
        end_pos=22,
        id="abc12345deadbeef",
    )

    doc = DocumentResult(
        input_path="test.md",
        format="markdown",
        chunks=[chunk],
        detections=[
            DetectionResult(
                chunk_id="abc12345deadbeef",
                ensemble_score=0.85,
                classification="ai_generated",
            )
        ],
        transforms=[
            TransformResult(
                chunk_id="abc12345deadbeef",
                original_text="AI-generated text here",
                transformed_text="Naturally written text here",
                iteration_count=3,
                operator_used="lexical_diversity",
                final_detection_score=0.15,
                diagnostics={
                    "placeholder_failures": 1,
                    "semantic_gate_rejections": 2,
                    "retries_used": 1,
                },
            )
        ],
        evaluations=[
            EvaluationResult(
                chunk_id="abc12345deadbeef",
                passed=True,
                embedding_similarity=0.92,
                diagnostics={"terminal_stage": "passed", "rejected_at": None},
            )
        ],
        output_path="test_transformed.md",
        summary_stats={
            "total_chunks": 1,
            "prose_chunks": 1,
            "ai_detected": 1,
            "transformed": 1,
            "unchanged": 0,
            "eval_passed": 1,
            "eval_failed": 0,
            "failed": 0,
            "operator_usage": {"lexical_diversity": 1},
            "search_diagnostics": {
                "placeholder_failures": 1,
                "chunks_with_placeholder_failures": 1,
                "semantic_gate_rejections": 2,
                "restore_failures": 0,
                "retries_used": 1,
            },
            "rejected_chunks": [
                {
                    "chunk_id": "abc12345deadbeef",
                    "rejected_at": "embedding",
                    "rejection_reason": "embedding similarity 0.6700 below threshold 0.68",
                    "operator_used": "lexical_diversity",
                }
            ],
        },
    )
    return doc


class TestFormatJSON:
    """format_json() produces valid JSON with expected schema."""

    def test_valid_json(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
        config: LUCIDConfig,
    ) -> None:
        output = formatter.format_json(sample_result, config)
        data = json.loads(output)

        assert "lucid_version" in data
        assert data["input_path"] == "test.md"
        assert data["format"] == "markdown"
        assert data["profile"] == "balanced"
        assert "timestamp" in data
        assert "summary" in data
        assert "chunks" in data
        assert len(data["chunks"]) == 1

    def test_chunk_detail_includes_detection(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
        config: LUCIDConfig,
    ) -> None:
        output = formatter.format_json(sample_result, config)
        data = json.loads(output)
        chunk = data["chunks"][0]

        assert "detection" in chunk
        assert chunk["detection"]["classification"] == "ai_generated"
        assert "paraphrase" in chunk
        assert "evaluation" in chunk
        assert chunk["paraphrase"]["diagnostics"]["placeholder_failures"] == 1


class TestFormatText:
    """format_text() produces human-readable report."""

    def test_contains_header_and_stats(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
    ) -> None:
        output = formatter.format_text(sample_result)

        assert "LUCID Pipeline Report" in output
        assert "test.md" in output
        assert "markdown" in output
        assert "AI-detected:" in output
        assert "Transformed:" in output

    def test_contains_detection_results(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
    ) -> None:
        output = formatter.format_text(sample_result)

        assert "Detection Results" in output
        assert "ai_generated" in output
        assert "0.850" in output

    def test_contains_evaluation_results(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
    ) -> None:
        output = formatter.format_text(sample_result)

        assert "Evaluation Results" in output
        assert "PASS" in output

    def test_contains_transform_diagnostics(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
    ) -> None:
        output = formatter.format_text(sample_result)

        assert "Transform Diagnostics" in output
        assert "placeholder_failures=1" in output

    def test_contains_rejected_chunks_section(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
    ) -> None:
        output = formatter.format_text(sample_result)

        assert "Rejected Chunks" in output
        assert "embedding similarity 0.6700 below threshold 0.68" in output


class TestFormatAnnotated:
    """format_annotated() inserts LUCID comments into original content."""

    def test_latex_annotation(
        self,
        formatter: OutputFormatter,
    ) -> None:
        result = DocumentResult(
            input_path="test.tex",
            format="latex",
            chunks=[
                ProseChunk(text="Some text", start_pos=50, end_pos=59, id="chunk1234"),
            ],
            detections=[
                DetectionResult(
                    chunk_id="chunk1234",
                    ensemble_score=0.75,
                    classification="ai_generated",
                ),
            ],
        )
        original = "x" * 50 + "Some text" + "y" * 20

        annotated = formatter.format_annotated(result, original)

        assert "%% LUCID:" in annotated
        assert "chunk=chunk123" in annotated
        assert "score=0.750" in annotated

    def test_markdown_annotation(
        self,
        formatter: OutputFormatter,
    ) -> None:
        result = DocumentResult(
            input_path="test.md",
            format="markdown",
            chunks=[
                ProseChunk(text="Content", start_pos=10, end_pos=17, id="md_chunk1"),
            ],
            detections=[
                DetectionResult(
                    chunk_id="md_chunk1",
                    ensemble_score=0.60,
                    classification="ambiguous",
                ),
            ],
        )
        original = "# Header\n\nContent here"

        annotated = formatter.format_annotated(result, original)

        assert "<!-- LUCID:" in annotated
        assert "-->" in annotated

    def test_no_annotation_when_no_detections(
        self,
        formatter: OutputFormatter,
    ) -> None:
        result = DocumentResult(
            input_path="test.md",
            format="markdown",
            chunks=[
                ProseChunk(text="Content", start_pos=0, end_pos=7, id="nochunk"),
            ],
        )
        original = "Content"

        annotated = formatter.format_annotated(result, original)
        assert annotated == original


class TestWrite:
    """write() dispatches to the correct formatter and writes to disk."""

    def test_write_json(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
        config: LUCIDConfig,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "report.json"
        formatter.write(sample_result, out, "json", config=config)

        assert out.exists()
        data = json.loads(out.read_text())
        assert data["format"] == "markdown"

    def test_write_text(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
        tmp_path: Path,
    ) -> None:
        out = tmp_path / "report.txt"
        formatter.write(sample_result, out, "text")

        assert out.exists()
        assert "LUCID Pipeline Report" in out.read_text()

    def test_write_unknown_format_raises(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="Unknown output format"):
            formatter.write(sample_result, tmp_path / "x.txt", "csv")

    def test_write_json_without_config_raises(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="config is required"):
            formatter.write(sample_result, tmp_path / "x.json", "json")

    def test_write_annotated_without_content_raises(
        self,
        formatter: OutputFormatter,
        sample_result: DocumentResult,
        tmp_path: Path,
    ) -> None:
        with pytest.raises(ValueError, match="original_content is required"):
            formatter.write(sample_result, tmp_path / "x.md", "annotated")
