"""Pipeline orchestrator for the LUCID detect-humanize-evaluate-reconstruct flow."""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lucid.checkpoint import CheckpointManager
from lucid.models.manager import ModelManager
from lucid.models.results import DocumentResult
from lucid.parser import detect_format, get_adapter
from lucid.parser.chunk import ProseChunk
from lucid.progress import PipelineEvent
from lucid.reconstructor import validate_latex, validate_markdown

if TYPE_CHECKING:
    from lucid.config import LUCIDConfig

logger = logging.getLogger(__name__)


class PipelineState(enum.Enum):
    """Stages of the LUCID processing pipeline."""

    PARSING = "PARSING"
    DETECTING = "DETECTING"
    HUMANIZING = "HUMANIZING"
    EVALUATING = "EVALUATING"
    RECONSTRUCTING = "RECONSTRUCTING"
    VALIDATING = "VALIDATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class LUCIDPipeline:
    """Orchestrates the full LUCID detect-humanize-evaluate-reconstruct pipeline.

    Args:
        config: Fully resolved LUCID configuration.
        checkpoint_dir: Directory for checkpoint persistence. None disables checkpointing.
    """

    def __init__(
        self,
        config: LUCIDConfig,
        checkpoint_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._checkpoint_dir = checkpoint_dir

    def run(
        self,
        input_path: Path,
        output_path: Path | None = None,
        progress_callback: Callable[[PipelineEvent], None] | None = None,
    ) -> DocumentResult:
        """Run the full pipeline: parse -> detect -> humanize -> evaluate -> reconstruct.

        Args:
            input_path: Path to the input document.
            output_path: Where to write the output. Defaults to {stem}_humanized.{ext}.
            progress_callback: Optional callback for progress events.

        Returns:
            DocumentResult with all accumulated results.
        """
        content = input_path.read_text(encoding="utf-8")
        fmt = detect_format(input_path)
        adapter = get_adapter(fmt, self._config)

        doc_result = DocumentResult(input_path=str(input_path), format=fmt)

        # Checkpoint setup
        checkpoint_mgr: CheckpointManager | None = None
        completed_ids: dict[str, list[str]] = {
            "detection": [],
            "humanization": [],
            "evaluation": [],
        }
        failed_ids: dict[str, str] = {}

        if self._checkpoint_dir is not None:
            checkpoint_mgr = CheckpointManager(self._checkpoint_dir, input_path)
            checkpoint_data = checkpoint_mgr.load()
            if checkpoint_data is not None:
                doc_result = checkpoint_data.document_result
                completed_ids = checkpoint_data.completed_chunk_ids
                failed_ids = dict(checkpoint_data.failed_chunk_ids)
                logger.info(
                    "Resumed from checkpoint at state=%s with %d completed detections",
                    checkpoint_data.state,
                    len(completed_ids.get("detection", [])),
                )

        # PARSING
        if not doc_result.chunks:
            self._emit(
                progress_callback, PipelineState.PARSING, None, 0, 1, "Parsing document"
            )
            doc_result.chunks = adapter.parse(content)

        prose_chunks = [c for c in doc_result.chunks if isinstance(c, ProseChunk)]
        total_prose = len(prose_chunks)

        # DETECTING
        model_mgr = ModelManager(self._config)
        model_mgr.initialize_detector()

        for i, chunk in enumerate(prose_chunks):
            if chunk.id in completed_ids["detection"]:
                continue
            self._emit(
                progress_callback,
                PipelineState.DETECTING,
                chunk.id,
                i,
                total_prose,
                f"Detecting chunk {chunk.id[:8]}",
            )
            result = model_mgr.detector.detect(chunk)
            doc_result.detections.append(result)
            completed_ids["detection"].append(chunk.id)
            if checkpoint_mgr is not None:
                checkpoint_mgr.save(
                    doc_result,
                    PipelineState.DETECTING.value,
                    completed_ids,
                    failed_ids,
                )

        # Build detection map
        detection_map = {d.chunk_id: d for d in doc_result.detections}
        target_chunks = [
            c
            for c in prose_chunks
            if detection_map.get(c.id) is not None
            and detection_map[c.id].classification in ("ai_generated", "ambiguous")
        ]
        total_targets = len(target_chunks)

        # HUMANIZING + EVALUATING
        if target_chunks:
            model_mgr.initialize_humanizer()
            model_mgr.initialize_evaluator()

            for i, chunk in enumerate(target_chunks):
                if chunk.id in completed_ids.get("humanization", []):
                    continue
                if chunk.id in failed_ids:
                    continue

                self._emit(
                    progress_callback,
                    PipelineState.HUMANIZING,
                    chunk.id,
                    i,
                    total_targets,
                    f"Humanizing chunk {chunk.id[:8]}",
                )
                try:
                    paraphrase = model_mgr.humanizer.humanize(
                        chunk, detection_map[chunk.id]
                    )
                    doc_result.paraphrases.append(paraphrase)

                    self._emit(
                        progress_callback,
                        PipelineState.EVALUATING,
                        chunk.id,
                        i,
                        total_targets,
                        f"Evaluating chunk {chunk.id[:8]}",
                    )
                    evaluation = model_mgr.evaluator.evaluate_chunk(
                        chunk.id, chunk.text, paraphrase.humanized_text
                    )
                    doc_result.evaluations.append(evaluation)

                    if evaluation.passed:
                        chunk.metadata["humanized_text"] = paraphrase.humanized_text

                    completed_ids["humanization"].append(chunk.id)
                    completed_ids["evaluation"].append(chunk.id)
                except Exception as exc:
                    logger.error("Chunk %s failed: %s", chunk.id, exc)
                    failed_ids[chunk.id] = str(exc)

                if checkpoint_mgr is not None:
                    checkpoint_mgr.save(
                        doc_result,
                        PipelineState.HUMANIZING.value,
                        completed_ids,
                        failed_ids,
                    )

        # RECONSTRUCTING
        self._emit(
            progress_callback,
            PipelineState.RECONSTRUCTING,
            None,
            0,
            1,
            "Reconstructing document",
        )
        output_content = adapter.reconstruct(content, doc_result.chunks)

        # VALIDATING
        self._emit(
            progress_callback,
            PipelineState.VALIDATING,
            None,
            0,
            1,
            "Validating output",
        )
        if fmt == "latex":
            validation = validate_latex(
                output_content, self._config.validation.latex_compiler
            )
            doc_result.compilation_valid = validation.valid
        elif fmt == "markdown":
            validation = validate_markdown(output_content)
            doc_result.compilation_valid = validation.valid

        # Write output
        if output_path is None:
            output_path = input_path.with_stem(input_path.stem + "_humanized")
        output_path.write_text(output_content, encoding="utf-8")
        doc_result.output_path = str(output_path)

        # Summary stats
        doc_result.summary_stats = self._compute_summary(doc_result, failed_ids)

        # Cleanup
        model_mgr.shutdown()
        if checkpoint_mgr is not None:
            checkpoint_mgr.clear()

        self._emit(
            progress_callback,
            PipelineState.COMPLETE,
            None,
            0,
            1,
            "Pipeline complete",
        )

        return doc_result

    def run_detect_only(
        self,
        input_path: Path,
        progress_callback: Callable[[PipelineEvent], None] | None = None,
    ) -> DocumentResult:
        """Run detection only — no humanization, evaluation, or reconstruction.

        Args:
            input_path: Path to the input document.
            progress_callback: Optional callback for progress events.

        Returns:
            DocumentResult with chunks and detections populated.
        """
        content = input_path.read_text(encoding="utf-8")
        fmt = detect_format(input_path)
        adapter = get_adapter(fmt, self._config)

        doc_result = DocumentResult(input_path=str(input_path), format=fmt)

        self._emit(
            progress_callback, PipelineState.PARSING, None, 0, 1, "Parsing document"
        )
        doc_result.chunks = adapter.parse(content)

        prose_chunks = [c for c in doc_result.chunks if isinstance(c, ProseChunk)]
        total_prose = len(prose_chunks)

        model_mgr = ModelManager(self._config)
        model_mgr.initialize_detector()

        for i, chunk in enumerate(prose_chunks):
            self._emit(
                progress_callback,
                PipelineState.DETECTING,
                chunk.id,
                i,
                total_prose,
                f"Detecting chunk {chunk.id[:8]}",
            )
            result = model_mgr.detector.detect(chunk)
            doc_result.detections.append(result)

        model_mgr.shutdown()

        doc_result.summary_stats = {
            "total_chunks": len(doc_result.chunks),
            "prose_chunks": total_prose,
            "ai_detected": sum(
                1 for d in doc_result.detections if d.classification == "ai_generated"
            ),
            "ambiguous": sum(
                1 for d in doc_result.detections if d.classification == "ambiguous"
            ),
            "human": sum(
                1 for d in doc_result.detections if d.classification == "human"
            ),
        }

        return doc_result

    @staticmethod
    def _emit(
        callback: Callable[[PipelineEvent], None] | None,
        state: PipelineState,
        chunk_id: str | None,
        chunk_index: int,
        total_chunks: int,
        detail: str,
    ) -> None:
        """Emit a progress event if a callback is registered."""
        if callback is not None:
            callback(
                PipelineEvent(
                    state=state.value,
                    chunk_id=chunk_id,
                    chunk_index=chunk_index,
                    total_chunks=total_chunks,
                    detail=detail,
                )
            )

    @staticmethod
    def _compute_summary(
        doc_result: DocumentResult,
        failed_ids: dict[str, str],
    ) -> dict[str, Any]:
        """Compute summary statistics from accumulated results."""
        prose_count = sum(
            1 for c in doc_result.chunks if isinstance(c, ProseChunk)
        )
        ai_detected = sum(
            1
            for d in doc_result.detections
            if d.classification in ("ai_generated", "ambiguous")
        )
        eval_passed = sum(1 for e in doc_result.evaluations if e.passed)
        eval_failed = sum(1 for e in doc_result.evaluations if not e.passed)

        return {
            "total_chunks": len(doc_result.chunks),
            "prose_chunks": prose_count,
            "ai_detected": ai_detected,
            "humanized": len(doc_result.paraphrases),
            "eval_passed": eval_passed,
            "eval_failed": eval_failed,
            "failed": len(failed_ids),
        }
