"""Pipeline orchestrator for the LUCID detect-transform-evaluate-reconstruct flow."""

from __future__ import annotations

import enum
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any

from lucid.checkpoint import CheckpointManager
from lucid.detector.statistical import MIN_WORDS_THRESHOLD
from lucid.models.manager import ModelManager
from lucid.models.results import DetectionResult, DocumentResult, TransformResult
from lucid.parser import detect_format, get_adapter
from lucid.parser.chunk import ProseChunk
from lucid.parser.merge import merge_prose_for_detection
from lucid.progress import PipelineEvent
from lucid.reconstructor import validate_latex, validate_markdown
from lucid.transform.chunk_policy import skip_transform_reason

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from lucid.config import LUCIDConfig

logger = logging.getLogger(__name__)


class PipelineState(enum.Enum):
    """Stages of the LUCID processing pipeline."""

    PARSING = "PARSING"
    DETECTING = "DETECTING"
    TRANSFORMING = "TRANSFORMING"
    EVALUATING = "EVALUATING"
    RECONSTRUCTING = "RECONSTRUCTING"
    VALIDATING = "VALIDATING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class LUCIDPipeline:
    """Orchestrates the full LUCID detect-transform-evaluate-reconstruct pipeline.

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
        """Run the full pipeline: parse -> detect -> transform -> evaluate -> reconstruct.

        Args:
            input_path: Path to the input document.
            output_path: Where to write the output. Defaults to {stem}_transformed.{ext}.
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
            "transformation": [],
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

        with ThreadPoolExecutor(max_workers=self._config.detection.max_concurrent) as executor:
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
        if self._config.transform.transform_ambiguous:
            target_classifications.add("ambiguous")
        else:
            ambiguous_count = sum(
                1 for c in prose_chunks
                if detection_map.get(c.id) is not None
                and detection_map[c.id].classification == "ambiguous"
            )
            if ambiguous_count:
                logger.info(
                    "Skipping %d ambiguous chunks (transform_ambiguous=false)",
                    ambiguous_count,
                )
        target_chunks = [
            c for c in prose_chunks
            if detection_map.get(c.id) is not None
            and detection_map[c.id].classification in target_classifications
        ]
        skipped_non_transformable: dict[str, int] = {}
        filtered_target_chunks: list[ProseChunk] = []
        for chunk in target_chunks:
            reason = skip_transform_reason(
                chunk,
                skip_title_like=self._config.transform.skip_title_like_chunks,
                skip_equation_like=self._config.transform.skip_equation_like_chunks,
                skip_math_heavy=self._config.transform.skip_math_heavy_chunks,
                min_prose_length=self._config.transform.min_prose_length,
            )
            if reason is None:
                filtered_target_chunks.append(chunk)
                continue
            chunk.metadata["skip_transform_reason"] = reason
            skipped_non_transformable[reason] = skipped_non_transformable.get(reason, 0) + 1

        if skipped_non_transformable:
            logger.info(
                "Skipping %d non-transformable chunks: %s",
                sum(skipped_non_transformable.values()),
                skipped_non_transformable,
            )
        target_chunks = filtered_target_chunks
        total_targets = len(target_chunks)

        # TRANSFORMING + EVALUATING
        if target_chunks:
            model_mgr.release_detection_models()
            model_mgr.initialize_transformer()

            # Filter chunks that still need transformation
            pending_chunks = [
                chunk
                for chunk in target_chunks
                if chunk.id not in completed_ids.get("transformation", [])
                and chunk.id not in failed_ids
            ]

            # --- Batch transformation ---
            if pending_chunks:
                self._emit(
                    progress_callback,
                    PipelineState.TRANSFORMING,
                    None,
                    0,
                    total_targets,
                    f"Transforming {len(pending_chunks)} chunks (batch)",
                )

                chunks_and_detections = [
                    (chunk, detection_map[chunk.id]) for chunk in pending_chunks
                ]

                def _on_chunk_done(done: int, total: int) -> None:
                    self._emit(
                        progress_callback,
                        PipelineState.TRANSFORMING,
                        None,
                        done,
                        total_targets,
                        f"Transformed {done}/{total} chunks",
                    )

                # transform_batch returns results positionally; failures raise per-chunk
                # Use individual error handling by processing via gather with return_exceptions
                batch_results: list[TransformResult | BaseException] = (
                    model_mgr.transformer.transform_batch(
                        chunks_and_detections, on_chunk_done=_on_chunk_done,
                    )
                )

                transform_map: dict[str, TransformResult] = {}
                for chunk, result in zip(pending_chunks, batch_results, strict=True):
                    if isinstance(result, BaseException):
                        logger.error("Chunk %s transformation failed: %s", chunk.id, result)
                        failed_ids[chunk.id] = str(result)
                    else:
                        doc_result.transforms.append(result)
                        transform_map[chunk.id] = result
                        completed_ids["transformation"].append(chunk.id)

                if checkpoint_mgr is not None:
                    checkpoint_mgr.save(
                        doc_result,
                        PipelineState.TRANSFORMING.value,
                        completed_ids,
                        failed_ids,
                    )

                if self._skip_eval:
                    # Apply transformed text directly without evaluation
                    for chunk in pending_chunks:
                        if chunk.id in transform_map:
                            chunk.metadata["transformed_text"] = (
                                transform_map[chunk.id].transformed_text
                            )
                            completed_ids["evaluation"].append(chunk.id)
                else:
                    # --- Parallel evaluation ---
                    eval_chunks = [c for c in pending_chunks if c.id in transform_map]
                    if eval_chunks:
                        model_mgr.initialize_evaluator()
                        self._emit(
                            progress_callback,
                            PipelineState.EVALUATING,
                            None,
                            0,
                            total_targets,
                            f"Evaluating {len(eval_chunks)} chunks (parallel)",
                        )

                        with ThreadPoolExecutor(
                            max_workers=self._config.evaluator.max_concurrent
                        ) as executor:
                            eval_futures = {}
                            for chunk in eval_chunks:
                                p = transform_map[chunk.id]
                                future = executor.submit(
                                    model_mgr.evaluator.evaluate_chunk,
                                    chunk.id,
                                    chunk.text,
                                    p.transformed_text,
                                )
                                eval_futures[future] = chunk

                            for future in as_completed(eval_futures):
                                chunk = eval_futures[future]
                                try:
                                    evaluation = future.result()
                                    doc_result.evaluations.append(evaluation)
                                    if evaluation.passed:
                                        chunk.metadata["transformed_text"] = (
                                            transform_map[chunk.id].transformed_text
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
            output_path = input_path.with_stem(input_path.stem + "_transformed")
        output_path.write_text(output_content, encoding="utf-8")
        doc_result.output_path = str(output_path)

        # Summary stats
        doc_result.summary_stats = self._compute_summary(
            doc_result,
            failed_ids,
            skipped_non_transformable=skipped_non_transformable,
        )

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
        """Run detection only -- no transformation, evaluation, or reconstruction.

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
        skipped_non_transformable: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        """Compute summary statistics from accumulated results."""
        prose_count = sum(
            1 for c in doc_result.chunks if isinstance(c, ProseChunk)
        )
        changed_transforms = sum(
            1 for t in doc_result.transforms if t.transformed_text != t.original_text
        )
        ai_detected = sum(
            1
            for d in doc_result.detections
            if d.classification in ("ai_generated", "ambiguous")
        )
        eval_passed = sum(1 for e in doc_result.evaluations if e.passed)
        eval_failed = sum(1 for e in doc_result.evaluations if not e.passed)
        fallback_modes: dict[str, int] = {}
        operator_usage: dict[str, int] = {}
        evaluation_rejection_stages: dict[str, int] = {}
        rejected_chunks: list[dict[str, Any]] = []
        search_diagnostics = {
            "chunks_with_placeholder_failures": 0,
            "placeholder_failures": 0,
            "semantic_gate_rejections": 0,
            "low_similarity_rejections": 0,
            "prompt_echo_rejections": 0,
            "restore_failures": 0,
            "retries_used": 0,
        }
        for transform in doc_result.transforms:
            operator_usage[transform.operator_used] = (
                operator_usage.get(transform.operator_used, 0) + 1
            )
            if transform.fallback_mode is not None:
                fallback_modes[transform.fallback_mode] = (
                    fallback_modes.get(transform.fallback_mode, 0) + 1
                )
            diagnostics = transform.diagnostics
            placeholder_failures = int(diagnostics.get("placeholder_failures", 0))
            if placeholder_failures > 0:
                search_diagnostics["chunks_with_placeholder_failures"] += 1
                search_diagnostics["placeholder_failures"] += placeholder_failures
            search_diagnostics["semantic_gate_rejections"] += int(
                diagnostics.get("semantic_gate_rejections", 0)
            )
            search_diagnostics["low_similarity_rejections"] += int(
                diagnostics.get("low_similarity_rejections", 0)
            )
            search_diagnostics["prompt_echo_rejections"] += int(
                diagnostics.get("prompt_echo_rejections", 0)
            )
            search_diagnostics["restore_failures"] += int(
                diagnostics.get("restore_failures", 0)
            )
            search_diagnostics["retries_used"] += int(diagnostics.get("retries_used", 0))

        for evaluation in doc_result.evaluations:
            stage = evaluation.diagnostics.get("rejected_at")
            if stage:
                evaluation_rejection_stages[str(stage)] = (
                    evaluation_rejection_stages.get(str(stage), 0) + 1
                )
                transform = next(
                    (
                        item
                        for item in doc_result.transforms
                        if item.chunk_id == evaluation.chunk_id
                    ),
                    None,
                )
                rejected_chunks.append(
                    {
                        "chunk_id": evaluation.chunk_id,
                        "rejected_at": stage,
                        "rejection_reason": evaluation.rejection_reason,
                        "embedding_similarity": evaluation.embedding_similarity,
                        "nli_forward": evaluation.nli_forward,
                        "nli_backward": evaluation.nli_backward,
                        "bertscore_f1": evaluation.bertscore_f1,
                        "operator_used": transform.operator_used if transform else None,
                        "fallback_mode": transform.fallback_mode if transform else None,
                    }
                )

        # Aggregate failure reasons for summary display
        failure_reasons: dict[str, int] = {}
        for reason in failed_ids.values():
            # Normalize reasons to categories
            if "Placeholder" in reason or "placeholder" in reason:
                key = "placeholder preservation failed"
            elif "Ollama" in reason or "connection" in reason.lower():
                key = "LLM connection/generation error"
            elif "timeout" in reason.lower():
                key = "LLM timeout"
            else:
                key = reason[:80]
            failure_reasons[key] = failure_reasons.get(key, 0) + 1

        return {
            "total_chunks": len(doc_result.chunks),
            "prose_chunks": prose_count,
            "ai_detected": ai_detected,
            "transformed": changed_transforms,
            "unchanged": len(doc_result.transforms) - changed_transforms,
            "eval_passed": eval_passed,
            "eval_failed": eval_failed,
            "failed": len(failed_ids),
            "failure_reasons": failure_reasons,
            "fallback_modes": fallback_modes,
            "operator_usage": operator_usage,
            "evaluation_rejection_stages": evaluation_rejection_stages,
            "rejected_chunks": rejected_chunks,
            "search_diagnostics": search_diagnostics,
            "skipped_non_transformable": skipped_non_transformable or {},
        }
