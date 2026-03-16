"""Tests for the LUCID pipeline orchestrator."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from lucid.config import LUCIDConfig, load_config
from lucid.models.results import (
    DetectionResult,
    DocumentResult,
    EvaluationResult,
    TransformResult,
)
from lucid.pipeline import LUCIDPipeline, PipelineState

if TYPE_CHECKING:
    from pathlib import Path

    from lucid.progress import PipelineEvent


@pytest.fixture
def config() -> LUCIDConfig:
    return load_config(profile="balanced")


@pytest.fixture
def md_input(tmp_path: Path) -> Path:
    p = tmp_path / "test.md"
    p.write_text(
        "# Title\n\n"
        "This is AI-generated content for testing the full pipeline execution path.\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def txt_input(tmp_path: Path) -> Path:
    p = tmp_path / "test.txt"
    p.write_text(
        "This is a paragraph that contains enough text for transform testing.\n\n"
        "This is another paragraph with sufficient length for the pipeline.\n",
        encoding="utf-8",
    )
    return p


def _make_detection(chunk_id: str, classification: str = "ai_generated") -> DetectionResult:
    return DetectionResult(
        chunk_id=chunk_id,
        ensemble_score=0.85,
        classification=classification,
    )


def _make_transform(chunk_id: str, original: str) -> TransformResult:
    return TransformResult(
        chunk_id=chunk_id,
        original_text=original,
        transformed_text=f"Transformed: {original}",
        iteration_count=1,
        operator_used="lexical_diversity",
        final_detection_score=0.15,
        diagnostics={
            "placeholder_failures": 1,
            "semantic_gate_rejections": 2,
            "restore_failures": 0,
            "retries_used": 1,
        },
    )


def _make_evaluation(chunk_id: str, passed: bool = True) -> EvaluationResult:
    if passed:
        return EvaluationResult(
            chunk_id=chunk_id,
            passed=True,
            embedding_similarity=0.92,
            diagnostics={"terminal_stage": "passed", "rejected_at": None},
        )
    return EvaluationResult(
        chunk_id=chunk_id,
        passed=False,
        embedding_similarity=0.5,
        rejection_reason="Similarity too low",
        nli_forward="not_entailment",
        nli_backward="not_entailment",
        diagnostics={"terminal_stage": "embedding", "rejected_at": "embedding"},
    )


class TestPipelineStateEnum:
    """PipelineState enum values."""

    def test_all_states_present(self) -> None:
        states = {s.value for s in PipelineState}
        assert "PARSING" in states
        assert "DETECTING" in states
        assert "TRANSFORMING" in states
        assert "EVALUATING" in states
        assert "RECONSTRUCTING" in states
        assert "VALIDATING" in states
        assert "COMPLETE" in states
        assert "FAILED" in states


class TestRunDetectOnly:
    """run_detect_only() parses and detects without transformation."""

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
    """run() executes the full detect-transform-evaluate-reconstruct flow."""

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
        mock_transformer = MagicMock()
        mock_evaluator = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.initialize_transformer.return_value = mock_transformer
        mock_mgr.initialize_evaluator.return_value = mock_evaluator
        mock_mgr.detector = mock_detector
        mock_mgr.transformer = mock_transformer
        mock_mgr.evaluator = mock_evaluator

        # Detector returns ai_generated for all chunks
        mock_detector.detect.side_effect = lambda chunk: _make_detection(chunk.id)
        # Pipeline calls transform_batch, which returns a list of results
        mock_transformer.transform_batch.side_effect = lambda pairs, on_chunk_done=None: [
            _make_transform(chunk.id, chunk.text) for chunk, _det in pairs
        ]
        mock_evaluator.evaluate_chunk.side_effect = (
            lambda cid, _orig, _hum: _make_evaluation(cid)
        )

        output = tmp_path / "output.md"
        pipeline = LUCIDPipeline(config)
        result = pipeline.run(md_input, output_path=output)

        assert result.output_path == str(output)
        assert output.exists()
        assert len(result.detections) > 0
        assert len(result.transforms) > 0
        assert len(result.evaluations) > 0
        assert result.summary_stats["total_chunks"] > 0
        assert result.summary_stats["transformed"] == len(result.transforms)
        assert result.summary_stats["operator_usage"] == {
            "lexical_diversity": len(result.transforms)
        }
        assert result.summary_stats["search_diagnostics"]["placeholder_failures"] == len(
            result.transforms
        )
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
        mock_transformer = MagicMock()
        mock_evaluator = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.initialize_transformer.return_value = mock_transformer
        mock_mgr.initialize_evaluator.return_value = mock_evaluator
        mock_mgr.detector = mock_detector
        mock_mgr.transformer = mock_transformer
        mock_mgr.evaluator = mock_evaluator

        mock_detector.detect.side_effect = lambda chunk: _make_detection(chunk.id)
        # Pipeline calls transform_batch; return exceptions to simulate failures
        mock_transformer.transform_batch.side_effect = lambda pairs, on_chunk_done=None: [
            RuntimeError("Ollama timeout") for _ in pairs
        ]

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(md_input, output_path=tmp_path / "out.md")

        # Pipeline completes despite transformation failures
        assert len(result.detections) > 0
        assert len(result.transforms) == 0
        assert result.summary_stats["failed"] > 0
        mock_mgr.initialize_evaluator.assert_not_called()

    @patch("lucid.pipeline.validate_markdown")
    @patch("lucid.pipeline.ModelManager")
    def test_identity_fallback_not_counted_as_transformed(
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
        mock_transformer = MagicMock()
        mock_evaluator = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.initialize_transformer.return_value = mock_transformer
        mock_mgr.initialize_evaluator.return_value = mock_evaluator
        mock_mgr.detector = mock_detector
        mock_mgr.transformer = mock_transformer
        mock_mgr.evaluator = mock_evaluator

        mock_detector.detect.side_effect = lambda chunk: _make_detection(chunk.id)
        mock_transformer.transform_batch.side_effect = lambda pairs, on_chunk_done=None: [
            TransformResult(
                chunk_id=chunk.id,
                original_text=chunk.text,
                transformed_text=chunk.text,
                iteration_count=2,
                operator_used="identity_keep_original",
                final_detection_score=0.85,
                semantic_similarity=1.0,
                fallback_mode="keep_original",
            )
            for chunk, _det in pairs
        ]
        mock_evaluator.evaluate_chunk.side_effect = (
            lambda cid, _orig, _hum: _make_evaluation(cid)
        )

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(md_input, output_path=tmp_path / "identity.md")

        assert result.summary_stats["transformed"] == 0
        assert result.summary_stats["unchanged"] == len(result.transforms)
        assert result.summary_stats["fallback_modes"] == {"keep_original": len(result.transforms)}

    def test_summary_tracks_evaluation_rejection_stages(
        self,
        config: LUCIDConfig,
    ) -> None:
        from lucid.parser.chunk import ProseChunk

        prose = ProseChunk(text="Original", start_pos=0, end_pos=8)
        result = DocumentResult(
            input_path="test.md",
            format="markdown",
            chunks=[prose],
            detections=[_make_detection(prose.id)],
            transforms=[_make_transform(prose.id, prose.text)],
            evaluations=[_make_evaluation(prose.id, passed=False)],
        )

        summary = LUCIDPipeline._compute_summary(result, failed_ids={})

        assert summary["evaluation_rejection_stages"] == {"embedding": 1}
        assert summary["search_diagnostics"]["semantic_gate_rejections"] == 2
        assert summary["rejected_chunks"][0]["chunk_id"] == prose.id
        assert summary["rejected_chunks"][0]["operator_used"] == "lexical_diversity"

    @patch("lucid.pipeline.validate_markdown")
    @patch("lucid.pipeline.ModelManager")
    def test_skips_title_like_chunks_from_transform(
        self,
        mock_manager_cls: MagicMock,
        mock_validate: MagicMock,
        config: LUCIDConfig,
        tmp_path: Path,
    ) -> None:
        from lucid.reconstructor import ValidationResult

        mock_validate.return_value = ValidationResult(valid=True)
        md_input = tmp_path / "titles.md"
        md_input.write_text(
            "### 2.1.1 Foundational Retrieval-Augmented Generation\n\n"
            "This paragraph explains the section in full sentences.\n",
            encoding="utf-8",
        )

        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_transformer = MagicMock()
        mock_evaluator = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.initialize_transformer.return_value = mock_transformer
        mock_mgr.initialize_evaluator.return_value = mock_evaluator
        mock_mgr.detector = mock_detector
        mock_mgr.transformer = mock_transformer
        mock_mgr.evaluator = mock_evaluator

        mock_detector.detect.side_effect = lambda chunk: _make_detection(chunk.id)
        mock_transformer.transform_batch.side_effect = lambda pairs, on_chunk_done=None: [
            _make_transform(chunk.id, chunk.text) for chunk, _det in pairs
        ]
        mock_evaluator.evaluate_chunk.side_effect = (
            lambda cid, _orig, _hum: _make_evaluation(cid)
        )

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(md_input, output_path=tmp_path / "titles_out.md")

        batch_pairs = mock_transformer.transform_batch.call_args.args[0]
        assert len(batch_pairs) == 1
        assert batch_pairs[0][0].text == "This paragraph explains the section in full sentences."
        assert result.summary_stats["skipped_non_transformable"] == {"title_like": 1}

    @patch("lucid.pipeline.ModelManager")
    def test_skips_equation_like_chunks_from_transform(
        self,
        mock_manager_cls: MagicMock,
        config: LUCIDConfig,
        tmp_path: Path,
    ) -> None:
        txt_input = tmp_path / "equations.txt"
        txt_input.write_text(
            "P(q | M_d) = product over t in q of P(t | M_d) where P(t) = count(t)\n\n"
            "This paragraph explains the model behavior in normal prose.",
            encoding="utf-8",
        )

        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_transformer = MagicMock()
        mock_evaluator = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.initialize_transformer.return_value = mock_transformer
        mock_mgr.initialize_evaluator.return_value = mock_evaluator
        mock_mgr.detector = mock_detector
        mock_mgr.transformer = mock_transformer
        mock_mgr.evaluator = mock_evaluator

        mock_detector.detect.side_effect = lambda chunk: _make_detection(chunk.id)
        mock_transformer.transform_batch.side_effect = lambda pairs, on_chunk_done=None: [
            _make_transform(chunk.id, chunk.text) for chunk, _det in pairs
        ]
        mock_evaluator.evaluate_chunk.side_effect = (
            lambda cid, _orig, _hum: _make_evaluation(cid)
        )

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(txt_input, output_path=tmp_path / "equations_out.txt")

        batch_pairs = mock_transformer.transform_batch.call_args.args[0]
        assert len(batch_pairs) == 1
        assert (
            batch_pairs[0][0].text
            == "This paragraph explains the model behavior in normal prose."
        )
        assert "equation_like" in result.summary_stats["skipped_non_transformable"]

    @patch("lucid.pipeline.ModelManager")
    def test_skips_math_heavy_chunks_from_transform(
        self,
        mock_manager_cls: MagicMock,
        config: LUCIDConfig,
        tmp_path: Path,
    ) -> None:
        txt_input = tmp_path / "math_heavy.txt"
        txt_input.write_text(
            "[MATH_001], [MATH_002], [MATH_003], [MATH_004], [MATH_005].\n\n"
            "This paragraph explains the retrieval procedure in normal prose.",
            encoding="utf-8",
        )

        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_transformer = MagicMock()
        mock_evaluator = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.initialize_transformer.return_value = mock_transformer
        mock_mgr.initialize_evaluator.return_value = mock_evaluator
        mock_mgr.detector = mock_detector
        mock_mgr.transformer = mock_transformer
        mock_mgr.evaluator = mock_evaluator

        mock_detector.detect.side_effect = lambda chunk: _make_detection(chunk.id)
        mock_transformer.transform_batch.side_effect = lambda pairs, on_chunk_done=None: [
            _make_transform(chunk.id, chunk.text) for chunk, _det in pairs
        ]
        mock_evaluator.evaluate_chunk.side_effect = (
            lambda cid, _orig, _hum: _make_evaluation(cid)
        )

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(txt_input, output_path=tmp_path / "math_heavy_out.txt")

        batch_pairs = mock_transformer.transform_batch.call_args.args[0]
        assert len(batch_pairs) == 1
        assert (
            batch_pairs[0][0].text
            == "This paragraph explains the retrieval procedure in normal prose."
        )
        assert result.summary_stats["skipped_non_transformable"] == {"math_heavy": 1}

    @patch("lucid.pipeline.validate_markdown")
    @patch("lucid.pipeline.ModelManager")
    def test_policy_skips_structural_like_prose_and_evaluates_only_remaining_chunk(
        self,
        mock_manager_cls: MagicMock,
        mock_validate: MagicMock,
        config: LUCIDConfig,
        tmp_path: Path,
    ) -> None:
        from lucid.reconstructor import ValidationResult

        mock_validate.return_value = ValidationResult(valid=True)
        md_input = tmp_path / "mixed.md"
        md_input.write_text(
            "### Foundational Retrieval-Augmented Generation Overview\n\n"
            "P(q | M_d) = product over t in q of P(t | M_d) where P(t) = count(t)\n\n"
            "This paragraph explains the section in ordinary prose.\n",
            encoding="utf-8",
        )

        mock_mgr = mock_manager_cls.return_value
        mock_detector = MagicMock()
        mock_transformer = MagicMock()
        mock_evaluator = MagicMock()

        mock_mgr.initialize_detector.return_value = mock_detector
        mock_mgr.initialize_transformer.return_value = mock_transformer
        mock_mgr.initialize_evaluator.return_value = mock_evaluator
        mock_mgr.detector = mock_detector
        mock_mgr.transformer = mock_transformer
        mock_mgr.evaluator = mock_evaluator

        mock_detector.detect.side_effect = lambda chunk: _make_detection(chunk.id)
        mock_transformer.transform_batch.side_effect = lambda pairs, on_chunk_done=None: [
            _make_transform(chunk.id, chunk.text) for chunk, _det in pairs
        ]
        mock_evaluator.evaluate_chunk.side_effect = (
            lambda cid, _orig, _hum: _make_evaluation(cid)
        )

        pipeline = LUCIDPipeline(config)
        result = pipeline.run(md_input, output_path=tmp_path / "mixed_out.md")

        batch_pairs = mock_transformer.transform_batch.call_args.args[0]
        assert len(batch_pairs) == 1
        assert batch_pairs[0][0].text == "This paragraph explains the section in ordinary prose."
        assert len(result.evaluations) == 1
        assert result.summary_stats["skipped_non_transformable"] == {
            "title_like": 1,
            "equation_like": 1,
        }


class TestDefaultOutputPath:
    """Output path defaults to {stem}_transformed.{ext}."""

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
        assert "_transformed" in result.output_path
