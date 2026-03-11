"""Transformation search loop for the LUCID pipeline.

Iterates through transform operators, re-scoring each candidate
with the detection engine, and returns the best (lowest-scoring) result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lucid.transform.ollama import GenerateOptions, OllamaError
from lucid.transform.operators import select_operator
from lucid.models.results import TransformResult
from lucid.parser.chunk import ProseChunk

if TYPE_CHECKING:
    from lucid.config import HumanizerConfig
    from lucid.detector.base import LUCIDDetector
    from lucid.transform.ollama import OllamaClient
    from lucid.transform.prompts import PromptBuilder
    from lucid.transform.term_protect import TermProtector
    from lucid.models.results import DetectionResult

logger = logging.getLogger(__name__)


async def transformation_search(
    chunk: ProseChunk,
    detection: DetectionResult,
    client: OllamaClient,
    detector: LUCIDDetector,
    term_protector: TermProtector,
    prompt_builder: PromptBuilder,
    config: HumanizerConfig,
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
    options = GenerateOptions(temperature=temperature, num_predict=num_predict)

    best: TransformResult | None = None
    domain = chunk.domain_hint or "general"

    for i in range(config.adversarial_iterations):
        operator = select_operator(i, placeholder_count=placeholder_count)

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
        validation = term_protector.validate(
            result.text, protected.term_placeholders, protected.math_placeholders
        )
        if not validation.is_valid:
            logger.warning(
                "Iteration %d: LLM dropped placeholders %s, retrying with lower temp",
                i,
                validation.missing_placeholders,
            )
            # Retry once with reduced temperature (less creativity, more literal copying)
            retry_temp = max(0.1, temperature - 0.2)
            retry_options = GenerateOptions(
                temperature=retry_temp, num_predict=options.num_predict
            )
            try:
                retry_result = await client.generate(prompt, model, options=retry_options)
            except OllamaError:
                logger.warning("Iteration %d: retry generation also failed", i)
                continue
            retry_validation = term_protector.validate(
                retry_result.text,
                protected.term_placeholders,
                protected.math_placeholders,
            )
            if not retry_validation.is_valid:
                logger.warning("Iteration %d: retry also dropped placeholders, skipping", i)
                continue
            result = retry_result
            validation = retry_validation

        # Restore placeholders and re-score
        restored = term_protector.restore(
            result.text, protected.term_placeholders, protected.math_placeholders
        )

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
        )

        # Track best
        if best is None or score < best.final_detection_score:
            best = candidate

        # Early exit
        if score < config.adversarial_target_score:
            logger.info(
                "Search converged at iteration %d (score=%.3f, target=%.3f)",
                i + 1,
                score,
                config.adversarial_target_score,
            )
            return candidate

    if best is not None:
        return best

    raise RuntimeError(
        f"No valid transform produced for chunk {chunk.id} "
        f"after {config.adversarial_iterations} iterations"
    )
