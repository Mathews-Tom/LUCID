"""Adversarial refinement loop for the LUCID humanization pipeline.

Iterates through humanization strategies, re-scoring each candidate
with the detection engine, and returns the best (lowest-scoring) result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lucid.humanizer.ollama import GenerateOptions, OllamaError
from lucid.humanizer.strategies import select_strategy
from lucid.models.results import ParaphraseResult
from lucid.parser.chunk import ProseChunk

if TYPE_CHECKING:
    from lucid.config import HumanizerConfig
    from lucid.detector.base import LUCIDDetector
    from lucid.humanizer.ollama import OllamaClient
    from lucid.humanizer.prompts import PromptBuilder
    from lucid.humanizer.term_protect import TermProtector
    from lucid.models.results import DetectionResult

logger = logging.getLogger(__name__)


async def adversarial_humanize(
    chunk: ProseChunk,
    detection: DetectionResult,
    client: OllamaClient,
    detector: LUCIDDetector,
    term_protector: TermProtector,
    prompt_builder: PromptBuilder,
    config: HumanizerConfig,
    model: str,
    profile: str,
) -> ParaphraseResult:
    """Run the adversarial humanization loop.

    Iterates through strategies, generating candidates and re-scoring
    each with the detector. Tracks the best candidate (lowest detection
    score) and exits early when the score drops below the target.

    Args:
        chunk: Prose chunk to humanize.
        detection: Initial detection result.
        client: Active OllamaClient (already in context manager).
        detector: Detection engine for re-scoring.
        term_protector: Handles placeholder protection and restoration.
        prompt_builder: Constructs prompts per strategy/domain/profile.
        config: Humanizer configuration.
        model: Ollama model tag.
        profile: Quality profile name.

    Returns:
        ParaphraseResult from the best candidate across all iterations.

    Raises:
        RuntimeError: If no valid candidate is produced across all iterations.
    """
    protected = term_protector.protect(chunk)
    temperature = getattr(config.temperature, profile)
    options = GenerateOptions(temperature=temperature)

    best: ParaphraseResult | None = None
    domain = chunk.domain_hint or "general"

    for i in range(config.adversarial_iterations):
        strategy = select_strategy(i)

        prompt = prompt_builder.build(protected.text, strategy, domain, profile)

        try:
            result = await client.generate(prompt, model, options=options)
        except OllamaError:
            logger.warning(
                "Ollama generation failed on iteration %d (strategy=%s)",
                i,
                strategy.name,
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
                "Iteration %d: LLM dropped placeholders %s, skipping",
                i,
                validation.missing_placeholders,
            )
            continue

        # Restore placeholders and re-score
        restored = term_protector.restore(
            result.text, protected.term_placeholders, protected.math_placeholders
        )

        temp_chunk = ProseChunk(
            text=restored,
            start_pos=chunk.start_pos,
            end_pos=chunk.start_pos + len(restored),
        )
        score = detector.detect(temp_chunk).ensemble_score

        candidate = ParaphraseResult(
            chunk_id=chunk.id,
            original_text=chunk.text,
            humanized_text=restored,
            iteration_count=i + 1,
            strategy_used=strategy.name,
            final_detection_score=score,
        )

        # Track best
        if best is None or score < best.final_detection_score:
            best = candidate

        # Early exit
        if score < config.adversarial_target_score:
            logger.info(
                "Adversarial converged at iteration %d (score=%.3f, target=%.3f)",
                i + 1,
                score,
                config.adversarial_target_score,
            )
            return candidate

    if best is not None:
        return best

    raise RuntimeError(
        f"No valid paraphrase produced for chunk {chunk.id} "
        f"after {config.adversarial_iterations} iterations"
    )
