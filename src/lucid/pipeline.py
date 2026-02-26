"""Pipeline orchestrator for the LUCID detect-humanize-evaluate-reconstruct flow."""

from __future__ import annotations

import enum
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lucid.checkpoint import CheckpointManager
from lucid.models.manager import ModelManager
from lucid.models.results import DetectionResult, DocumentResult, ParaphraseResult
from lucid.parser import detect_format, get_adapter
from lucid.parser.chunk import ProseChunk
from lucid.parser.merge import merge_prose_for_detection
from lucid.detector.statistical import MIN_WORDS_THRESHOLD
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
        skip_eval: bool = False,
    ) -> None:
        self._config = config
        self._checkpoint_dir = checkpoint_dir
        self._skip_eval = skip_eval

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

        # DETECTING (with merge for fragmented prose)
        model_mgr = ModelManager(self._config)
        model_mgr.initialize_detector()

        detection_groups = merge_prose_for_detection(
            doc_result.chunks, MIN_WORDS_THRESHOLD
        )

        # Filter groups that still need detection
        pending_groups: list[tuple[int, str, list[ProseChunk]]] = []
        for i, (merged_text, constituent_chunks) in enumerate(detection_groups):
            constituent_ids = [c.id for c in constituent_chunks]
            if all(cid in completed_ids["detection"] for cid in constituent_ids):
                continue
            pending_groups.append((i, merged_text, constituent_chunks))

        self._emit(
            progress_callback,
            PipelineState.DETECTING,
            None,
            0,
            len(detection_groups),
            f"Detecting {len(pending_groups)} groups (parallel)",
        )

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {}
            for i, merged_text, constituent_chunks in pending_groups:
                temp_chunk = ProseChunk(
                    text=merged_text,
                    start_pos=constituent_chunks[0].start_pos,
                    end_pos=constituent_chunks[-1].end_pos,
                )
                temp_chunk.protected_text = merged_text
                future = executor.submit(model_mgr.detector.detect, temp_chunk)
                futures[future] = (i, merged_text, constituent_chunks)

            for future in as_completed(futures):
                i, merged_text, constituent_chunks = futures[future]
                result = future.result()

                # Fan out the detection result to all constituent chunks
                for constituent in constituent_chunks:
                    doc_result.detections.append(
                        DetectionResult(
                            chunk_id=constituent.id,
                            ensemble_score=result.ensemble_score,
                            classification=result.classification,
                            roberta_score=result.roberta_score,
                            statistical_score=result.statistical_score,
                            binoculars_score=result.binoculars_score,
                            feature_details=result.feature_details,
                        )
                    )
                    completed_ids["detection"].append(constituent.id)

                if checkpoint_mgr is not None:
                    checkpoint_mgr.save(
                        doc_result,
                        PipelineState.DETECTING.value,
                        completed_ids,
                        failed_ids,
                    )

        # Build detection map
        detection_map = {d.chunk_id: d for d in doc_result.detections}
        target_classifications = {"ai_generated"}
        if self._config.humanizer.humanize_ambiguous:
            target_classifications.add("ambiguous")
        else:
            ambiguous_count = sum(
                1 for c in prose_chunks
                if detection_map.get(c.id) is not None
                and detection_map[c.id].classification == "ambiguous"
            )
            if ambiguous_count:
                logger.info(
                    "Skipping %d ambiguous chunks (humanize_ambiguous=false)",
                    ambiguous_count,
                )
        target_chunks = [
            c
            for c in prose_chunks
            if detection_map.get(c.id) is not None
            and detection_map[c.id].classification in target_classifications
        ]
        total_targets = len(target_chunks)

        # HUMANIZING + EVALUATING
        if target_chunks:
            model_mgr.release_detection_models()
            model_mgr.initialize_humanizer()
            if not self._skip_eval:
                model_mgr.initialize_evaluator()

            # Filter chunks that still need humanization
            pending_chunks = [
                chunk
                for chunk in target_chunks
                if chunk.id not in completed_ids.get("humanization", [])
                and chunk.id not in failed_ids
            ]

            # --- Batch humanization ---
            if pending_chunks:
                self._emit(
                    progress_callback,
                    PipelineState.HUMANIZING,
                    None,
                    0,
                    total_targets,
                    f"Humanizing {len(pending_chunks)} chunks (batch)",
                )

                chunks_and_detections = [
                    (chunk, detection_map[chunk.id]) for chunk in pending_chunks
                ]

                # humanize_batch returns results positionally; failures raise per-chunk
                # Use individual error handling by processing via gather with return_exceptions
                batch_results: list[ParaphraseResult | BaseException] = (
                    model_mgr.humanizer.humanize_batch(chunks_and_detections)
                )

                paraphrase_map: dict[str, ParaphraseResult] = {}
                for chunk, result in zip(pending_chunks, batch_results):
                    if isinstance(result, BaseException):
                        logger.error("Chunk %s humanization failed: %s", chunk.id, result)
                        failed_ids[chunk.id] = str(result)
                    else:
                        doc_result.paraphrases.append(result)
                        paraphrase_map[chunk.id] = result
                        completed_ids["humanization"].append(chunk.id)

                if checkpoint_mgr is not None:
                    checkpoint_mgr.save(
                        doc_result,
                        PipelineState.HUMANIZING.value,
                        completed_ids,
                        failed_ids,
                    )

                if self._skip_eval:
                    # Apply humanized text directly without evaluation
                    for chunk in pending_chunks:
                        if chunk.id in paraphrase_map:
                            chunk.metadata["humanized_text"] = (
                                paraphrase_map[chunk.id].humanized_text
                            )
                            completed_ids["evaluation"].append(chunk.id)
                else:
                    # --- Parallel evaluation ---
                    eval_chunks = [c for c in pending_chunks if c.id in paraphrase_map]
                    if eval_chunks:
                        self._emit(
                            progress_callback,
                            PipelineState.EVALUATING,
                            None,
                            0,
                            total_targets,
                            f"Evaluating {len(eval_chunks)} chunks (parallel)",
                        )

                        with ThreadPoolExecutor(max_workers=4) as executor:
                            eval_futures = {}
                            for chunk in eval_chunks:
                                p = paraphrase_map[chunk.id]
                                future = executor.submit(
                                    model_mgr.evaluator.evaluate_chunk,
                                    chunk.id,
                                    chunk.text,
                                    p.humanized_text,
                                )
                                eval_futures[future] = chunk

                            for future in as_completed(eval_futures):
                                chunk = eval_futures[future]
                                try:
                                    evaluation = future.result()
                                    doc_result.evaluations.append(evaluation)
                                    if evaluation.passed:
                                        chunk.metadata["humanized_text"] = (
                                            paraphrase_map[chunk.id].humanized_text
                                        )
                                    completed_ids["evaluation"].append(chunk.id)
                                except Exception as exc:
                                    logger.error(
                                        "Chunk %s evaluation failed: %s", chunk.id, exc
                                    )
                                    failed_ids[chunk.id] = str(exc)

                        if checkpoint_mgr is not None:
                            checkpoint_mgr.save(
                                doc_result,
                                PipelineState.EVALUATING.value,
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

        detection_groups = merge_prose_for_detection(
            doc_result.chunks, MIN_WORDS_THRESHOLD
        )

        for i, (merged_text, constituent_chunks) in enumerate(detection_groups):
            self._emit(
                progress_callback,
                PipelineState.DETECTING,
                constituent_chunks[0].id,
                i,
                len(detection_groups),
                f"Detecting chunk {constituent_chunks[0].id[:8]}",
            )

            temp_chunk = ProseChunk(
                text=merged_text,
                start_pos=constituent_chunks[0].start_pos,
                end_pos=constituent_chunks[-1].end_pos,
            )
            temp_chunk.protected_text = merged_text

            result = model_mgr.detector.detect(temp_chunk)

            for constituent in constituent_chunks:
                doc_result.detections.append(
                    DetectionResult(
                        chunk_id=constituent.id,
                        ensemble_score=result.ensemble_score,
                        classification=result.classification,
                        roberta_score=result.roberta_score,
                        statistical_score=result.statistical_score,
                        binoculars_score=result.binoculars_score,
                        feature_details=result.feature_details,
                    )
                )

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
