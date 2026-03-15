"""Transformation search loop for the LUCID pipeline.

Iterates through transform operators, re-scoring each candidate
with the detection engine, and returns the best (lowest-scoring) result.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from lucid.models.results import TransformResult
from lucid.parser.chunk import ProseChunk
from lucid.transform.ollama import GenerateOptions, OllamaError
from lucid.transform.operators import select_operator
from lucid.transform.similarity import quick_similarity

if TYPE_CHECKING:
    from lucid.config import TransformConfig
    from lucid.detector.base import LUCIDDetector
    from lucid.models.results import DetectionResult
    from lucid.transform.ollama import OllamaClient
    from lucid.transform.prompts import PromptBuilder
    from lucid.transform.term_protect import TermProtector

logger = logging.getLogger(__name__)

# After this many consecutive placeholder failures, abort the search loop.
_CONSECUTIVE_FAIL_ABORT = 3

# Candidates scoring below this floor are likely over-rewritten.  The semantic
# gate (quick_similarity) handles most cases; this is a safety net.
_SCORE_FLOOR = 0.05
_SEMANTIC_PRIORITY_MARGIN = 0.03
_PROMPT_ECHO_PATTERNS = (
    "the output for this example should be",
    "upon inspection of your specific input text",
    "i must inform you that",
    "i will proceed with rewriting",
    "please note that your instructions",
    "your instructions explicitly state",
    "i am providing a natural rewrite",
    "critical rules",
    "specific input string",
)


def _build_search_diagnostics(
    *,
    chunk: ProseChunk,
    placeholder_count: int,
    attempted_operators: list[str],
    configured_iterations: int,
    iterations_attempted: int,
    placeholder_failures: int,
    semantic_gate_rejections: int,
    low_similarity_rejections: int,
    prompt_echo_rejections: int,
    restore_failures: int,
    retries_used: int,
    aborted_on_placeholder_failures: bool,
    fallback_policy: str,
    fallback_mode: str | None,
    final_status: str,
) -> dict[str, object]:
    """Build a stable machine-readable view of transform search behavior."""
    return {
        "chunk_length": len(chunk.text),
        "placeholder_count": placeholder_count,
        "attempted_operators": attempted_operators,
        "search_iterations_configured": configured_iterations,
        "iterations_attempted": iterations_attempted,
        "placeholder_failures": placeholder_failures,
        "semantic_gate_rejections": semantic_gate_rejections,
        "low_similarity_rejections": low_similarity_rejections,
        "prompt_echo_rejections": prompt_echo_rejections,
        "restore_failures": restore_failures,
        "retries_used": retries_used,
        "aborted_on_placeholder_failures": aborted_on_placeholder_failures,
        "fallback_policy": fallback_policy,
        "fallback_mode": fallback_mode,
        "final_status": final_status,
    }


def _should_adaptive_keep_original(
    config: TransformConfig,
    *,
    placeholder_count: int,
    placeholder_failures: int,
    aborted_on_placeholder_failures: bool,
) -> bool:
    """Decide whether placeholder pressure should degrade to keep-original."""
    if not config.adaptive_placeholder_fallback:
        return False
    if placeholder_count >= config.adaptive_keep_original_min_placeholders:
        return True
    if placeholder_failures >= config.adaptive_keep_original_after_failures:
        return True
    return aborted_on_placeholder_failures and placeholder_failures > 0


def _prefers_candidate(candidate: TransformResult, incumbent: TransformResult) -> bool:
    """Return True when candidate is preferable to incumbent."""
    candidate_sim = (
        candidate.semantic_similarity if candidate.semantic_similarity is not None else 0.0
    )
    incumbent_sim = (
        incumbent.semantic_similarity if incumbent.semantic_similarity is not None else 0.0
    )
    if candidate_sim > incumbent_sim + _SEMANTIC_PRIORITY_MARGIN:
        return True
    if incumbent_sim > candidate_sim + _SEMANTIC_PRIORITY_MARGIN:
        return False

    candidate_above_floor = candidate.final_detection_score >= _SCORE_FLOOR
    incumbent_above_floor = incumbent.final_detection_score >= _SCORE_FLOOR
    if candidate_above_floor == incumbent_above_floor:
        if candidate.final_detection_score != incumbent.final_detection_score:
            return candidate.final_detection_score < incumbent.final_detection_score
        return candidate_sim > incumbent_sim
    return candidate_above_floor and not incumbent_above_floor


def _looks_like_prompt_echo(text: str) -> bool:
    """Detect obvious instruction leakage in generated output."""
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    return any(pattern in lowered for pattern in _PROMPT_ECHO_PATTERNS)


async def transformation_search(
    chunk: ProseChunk,
    detection: DetectionResult,
    client: OllamaClient,
    detector: LUCIDDetector,
    term_protector: TermProtector,
    prompt_builder: PromptBuilder,
    config: TransformConfig,
    model: str,
    profile: str,
) -> TransformResult:
    """Run the transformation search loop.

    Iterates through operators, generating candidates and re-scoring
    each with the detector. Tracks the best candidate (lowest detection
    score) and exits early when the score drops below the target.

    Args:
        chunk: Prose chunk to transform.
        detection: Initial detection result.
        client: Active OllamaClient (already in context manager).
        detector: Detection engine for re-scoring.
        term_protector: Handles placeholder protection and restoration.
        prompt_builder: Constructs prompts per operator/domain/profile.
        config: Transform configuration.
        model: Ollama model tag.
        profile: Quality profile name.

    Returns:
        TransformResult from the best candidate across all iterations.

    Raises:
        RuntimeError: If no valid candidate is produced across all iterations.
    """
    protected = term_protector.protect(chunk)
    temperature = getattr(config.temperature, profile)
    all_placeholders = protected.all_placeholders()
    placeholder_count = len(all_placeholders)

    # Scale num_predict with input length to avoid truncation on long chunks
    input_token_estimate = len(protected.text.split()) * 2
    num_predict = max(512, min(2048, int(input_token_estimate * 1.3)))

    # Build a test prompt to measure actual token overhead
    test_operator = select_operator(0, placeholder_count=placeholder_count)
    test_prompt = prompt_builder.build(
        protected.text,
        test_operator,
        chunk.domain_hint or "general",
        profile,
        placeholders=protected.all_placeholders(),
    )
    prompt_token_estimate = len(test_prompt.split()) * 2
    num_ctx = prompt_token_estimate + num_predict

    options = GenerateOptions(
        temperature=temperature, num_predict=num_predict, num_ctx=num_ctx,
    )

    best: TransformResult | None = None
    domain = chunk.domain_hint or "general"
    consecutive_failures = 0
    placeholder_failure_count = 0
    semantic_gate_rejections = 0
    low_similarity_rejections = 0
    prompt_echo_rejections = 0
    restore_failures = 0
    retries_used = 0
    attempted_operators: list[str] = []
    iterations_attempted = 0
    aborted_on_placeholder_failures = False

    for i in range(config.search_iterations):
        operator = select_operator(
            i,
            placeholder_count=placeholder_count,
            placeholder_failures=placeholder_failure_count,
            narrow_after_failures=config.operator_narrowing_after_failures,
            single_operator_after_failures=config.operator_single_mode_after_failures,
        )
        attempted_operators.append(operator.name)
        iterations_attempted = i + 1

        prompt = prompt_builder.build(
            protected.text,
            operator,
            domain,
            profile,
            placeholders=protected.all_placeholders(),
        )

        try:
            result = await client.generate(prompt, model, options=options)
        except OllamaError:
            logger.warning(
                "Ollama generation failed on iteration %d (operator=%s)",
                i,
                operator.name,
                exc_info=True,
            )
            if best is not None:
                return best
            raise

        # Validate placeholders
        output_text = result.text
        validation = term_protector.validate(
            output_text, protected.term_placeholders, protected.math_placeholders
        )
        if not validation.is_valid:
            # Try repair: LLM may have written original values instead of placeholders
            repaired_text, repair_ok = term_protector.repair(
                output_text, protected.term_placeholders, protected.math_placeholders
            )
            if repair_ok:
                logger.info(
                    "Iteration %d: repaired %d dropped placeholders",
                    i,
                    len(validation.missing_placeholders),
                )
                output_text = repaired_text
                consecutive_failures = 0
            else:
                logger.warning(
                    "Iteration %d: LLM dropped placeholders %s, retrying with lower temp",
                    i,
                    validation.missing_placeholders,
                )
                placeholder_failure_count += 1
                # Retry once with reduced temperature
                retry_temp = max(0.1, temperature - 0.2)
                retry_options = GenerateOptions(
                    temperature=retry_temp, num_predict=options.num_predict,
                    num_ctx=options.num_ctx,
                )
                retries_used += 1
                try:
                    retry_result = await client.generate(prompt, model, options=retry_options)
                except OllamaError:
                    logger.warning("Iteration %d: retry generation also failed", i)
                    consecutive_failures += 1
                    if consecutive_failures >= _CONSECUTIVE_FAIL_ABORT:
                        aborted_on_placeholder_failures = True
                        logger.warning(
                            "Aborting search after %d consecutive placeholder failures",
                            consecutive_failures,
                        )
                        break
                    continue

                # Try repair on retry output too
                retry_text = retry_result.text
                retry_validation = term_protector.validate(
                    retry_text, protected.term_placeholders, protected.math_placeholders
                )
                if not retry_validation.is_valid:
                    retry_text, retry_repair_ok = term_protector.repair(
                        retry_text, protected.term_placeholders, protected.math_placeholders
                    )
                    if not retry_repair_ok:
                        logger.warning(
                            "Iteration %d: retry also dropped placeholders, skipping", i
                        )
                        placeholder_failure_count += 1
                        consecutive_failures += 1
                        if consecutive_failures >= _CONSECUTIVE_FAIL_ABORT:
                            aborted_on_placeholder_failures = True
                            logger.warning(
                                "Aborting search after %d consecutive placeholder failures",
                                consecutive_failures,
                            )
                            break
                        continue

                output_text = retry_text
                consecutive_failures = 0
        else:
            consecutive_failures = 0

        # Restore placeholders and re-score
        try:
            restored = term_protector.restore(
                output_text, protected.term_placeholders, protected.math_placeholders
            )
        except ValueError:
            logger.warning(
                "Iteration %d: placeholder restore failed (LLM hallucinated tokens), skipping",
                i,
            )
            restore_failures += 1
            consecutive_failures += 1
            if consecutive_failures >= _CONSECUTIVE_FAIL_ABORT:
                aborted_on_placeholder_failures = True
                logger.warning(
                    "Aborting search after %d consecutive failures", consecutive_failures,
                )
                break
            continue

        # Semantic gate: reject candidates that diverge too far from original.
        # Runs before the expensive detect_fast() call to save time.
        sim = quick_similarity(chunk.text, restored)
        if config.reject_prompt_echo and _looks_like_prompt_echo(restored):
            prompt_echo_rejections += 1
            logger.warning(
                "Iteration %d: generated output leaked prompt/instruction text, skipping",
                i,
            )
            continue
        if sim < config.minimum_candidate_similarity:
            low_similarity_rejections += 1
            logger.warning(
                "Iteration %d: semantic similarity %.3f below hard floor %.3f, skipping",
                i,
                sim,
                config.minimum_candidate_similarity,
            )
            continue
        if sim < config.semantic_gate_threshold:
            semantic_gate_rejections += 1
            logger.warning(
                "Iteration %d: semantic similarity %.3f below gate %.3f, skipping",
                i,
                sim,
                config.semantic_gate_threshold,
            )
            continue

        temp_chunk = ProseChunk(
            text=restored,
            start_pos=chunk.start_pos,
            end_pos=chunk.start_pos + len(restored),
        )
        score = detector.detect_fast(temp_chunk).ensemble_score

        candidate = TransformResult(
            chunk_id=chunk.id,
            original_text=chunk.text,
            transformed_text=restored,
            iteration_count=i + 1,
            operator_used=operator.name,
            final_detection_score=score,
            semantic_similarity=sim,
            diagnostics={},
        )

        # Track best using semantic fidelity first, then detection evasion.
        if best is None or _prefers_candidate(candidate, best):
            best = candidate

        # Early exit
        if _SCORE_FLOOR <= score < config.search_target_score:
            logger.info(
                "Search converged at iteration %d (score=%.3f, target=%.3f)",
                i + 1,
                score,
                config.search_target_score,
            )
            candidate.diagnostics = _build_search_diagnostics(
                chunk=chunk,
                placeholder_count=placeholder_count,
                attempted_operators=attempted_operators,
                configured_iterations=config.search_iterations,
                iterations_attempted=iterations_attempted,
                placeholder_failures=placeholder_failure_count,
                semantic_gate_rejections=semantic_gate_rejections,
                low_similarity_rejections=low_similarity_rejections,
                prompt_echo_rejections=prompt_echo_rejections,
                restore_failures=restore_failures,
                retries_used=retries_used,
                aborted_on_placeholder_failures=aborted_on_placeholder_failures,
                fallback_policy=config.fallback_policy,
                fallback_mode=None,
                final_status="target_reached",
            )
            return candidate

    if best is not None:
        best.diagnostics = _build_search_diagnostics(
            chunk=chunk,
            placeholder_count=placeholder_count,
            attempted_operators=attempted_operators,
            configured_iterations=config.search_iterations,
            iterations_attempted=iterations_attempted,
            placeholder_failures=placeholder_failure_count,
            semantic_gate_rejections=semantic_gate_rejections,
            low_similarity_rejections=low_similarity_rejections,
            prompt_echo_rejections=prompt_echo_rejections,
            restore_failures=restore_failures,
            retries_used=retries_used,
            aborted_on_placeholder_failures=aborted_on_placeholder_failures,
            fallback_policy=config.fallback_policy,
            fallback_mode=best.fallback_mode,
            final_status="best_candidate",
        )
        return best

    logger.warning(
        "All %d iterations failed validation for chunk %s; fallback_policy=%s",
        config.search_iterations,
        chunk.id,
        config.fallback_policy,
    )
    if (
        config.fallback_policy == "mark_failed"
        and _should_adaptive_keep_original(
            config,
            placeholder_count=placeholder_count,
            placeholder_failures=placeholder_failure_count,
            aborted_on_placeholder_failures=aborted_on_placeholder_failures,
        )
    ):
        logger.warning(
            "Adaptive placeholder fallback keeping original text for chunk %s "
            "(placeholders=%d, placeholder_failures=%d)",
            chunk.id,
            placeholder_count,
            placeholder_failure_count,
        )
        return TransformResult(
            chunk_id=chunk.id,
            original_text=chunk.text,
            transformed_text=chunk.text,
            iteration_count=config.search_iterations,
            operator_used="identity_keep_original",
            final_detection_score=detection.ensemble_score,
            semantic_similarity=1.0,
            fallback_mode="keep_original",
            diagnostics=_build_search_diagnostics(
                chunk=chunk,
                placeholder_count=placeholder_count,
                attempted_operators=attempted_operators,
                configured_iterations=config.search_iterations,
                iterations_attempted=iterations_attempted,
                placeholder_failures=placeholder_failure_count,
                semantic_gate_rejections=semantic_gate_rejections,
                low_similarity_rejections=low_similarity_rejections,
                prompt_echo_rejections=prompt_echo_rejections,
                restore_failures=restore_failures,
                retries_used=retries_used,
                aborted_on_placeholder_failures=aborted_on_placeholder_failures,
                fallback_policy=config.fallback_policy,
                fallback_mode="keep_original",
                final_status="adaptive_keep_original",
            ),
        )
    if config.fallback_policy == "keep_original":
        logger.warning(
            "Keeping original text for chunk %s after %d placeholder failures",
            chunk.id,
            placeholder_failure_count,
        )
        return TransformResult(
            chunk_id=chunk.id,
            original_text=chunk.text,
            transformed_text=chunk.text,
            iteration_count=config.search_iterations,
            operator_used="identity_keep_original",
            final_detection_score=detection.ensemble_score,
            semantic_similarity=1.0,
            fallback_mode="keep_original",
            diagnostics=_build_search_diagnostics(
                chunk=chunk,
                placeholder_count=placeholder_count,
                attempted_operators=attempted_operators,
                configured_iterations=config.search_iterations,
                iterations_attempted=iterations_attempted,
                placeholder_failures=placeholder_failure_count,
                semantic_gate_rejections=semantic_gate_rejections,
                low_similarity_rejections=low_similarity_rejections,
                prompt_echo_rejections=prompt_echo_rejections,
                restore_failures=restore_failures,
                retries_used=retries_used,
                aborted_on_placeholder_failures=aborted_on_placeholder_failures,
                fallback_policy=config.fallback_policy,
                fallback_mode="keep_original",
                final_status="fallback_keep_original",
            ),
        )
    if config.fallback_policy == "unsafe_unprotected":
        logger.warning(
            "Attempting unprotected fallback for chunk %s after %d placeholder failures",
            chunk.id,
            placeholder_failure_count,
        )
        fallback_operator = select_operator(0, placeholder_count=0)
        fallback_prompt = prompt_builder.build(
            chunk.text,
            fallback_operator,
            domain,
            profile,
            placeholders=None,
        )
        fallback_options = GenerateOptions(
            temperature=max(0.1, temperature - 0.2),
            num_predict=num_predict,
            num_ctx=num_ctx,
        )
        try:
            fallback_result = await client.generate(
                fallback_prompt, model, options=fallback_options,
            )
            fallback_text = fallback_result.text
            fallback_similarity = quick_similarity(chunk.text, fallback_text)
            fallback_chunk = ProseChunk(
                text=fallback_text,
                start_pos=chunk.start_pos,
                end_pos=chunk.start_pos + len(fallback_text),
            )
            fallback_score = detector.detect_fast(fallback_chunk).ensemble_score
            logger.info(
                "Unprotected fallback produced score %.3f for chunk %s",
                fallback_score,
                chunk.id,
            )
            return TransformResult(
                chunk_id=chunk.id,
                original_text=chunk.text,
                transformed_text=fallback_text,
                iteration_count=config.search_iterations + 1,
                operator_used=fallback_operator.name,
                final_detection_score=fallback_score,
                semantic_similarity=fallback_similarity,
                fallback_mode="unsafe_unprotected",
                diagnostics=_build_search_diagnostics(
                    chunk=chunk,
                    placeholder_count=placeholder_count,
                    attempted_operators=attempted_operators,
                    configured_iterations=config.search_iterations,
                    iterations_attempted=iterations_attempted,
                    placeholder_failures=placeholder_failure_count,
                    semantic_gate_rejections=semantic_gate_rejections,
                    low_similarity_rejections=low_similarity_rejections,
                    prompt_echo_rejections=prompt_echo_rejections,
                    restore_failures=restore_failures,
                    retries_used=retries_used,
                    aborted_on_placeholder_failures=aborted_on_placeholder_failures,
                    fallback_policy=config.fallback_policy,
                    fallback_mode="unsafe_unprotected",
                    final_status="fallback_unsafe_unprotected",
                ),
            )
        except OllamaError as exc:
            raise RuntimeError(
                f"No valid transform produced for chunk {chunk.id} "
                f"after {config.search_iterations} iterations "
                f"(unprotected fallback also failed)"
            ) from exc

    raise RuntimeError(
        f"No valid transform produced for chunk {chunk.id} "
        f"after {config.search_iterations} iterations "
        f"(fallback_policy={config.fallback_policy})"
    )
