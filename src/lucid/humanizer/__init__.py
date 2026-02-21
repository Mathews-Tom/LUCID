"""Content humanization via Ollama LLM inference.

The :class:`LUCIDHumanizer` satisfies the :class:`~lucid.core.protocols.Humanizer`
protocol, wrapping the async :class:`OllamaClient` behind a synchronous
``humanize()`` call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from lucid.humanizer.ollama import GenerateOptions, OllamaClient
from lucid.humanizer.prompts import PromptBuilder
from lucid.humanizer.strategies import select_strategy
from lucid.humanizer.term_protect import TermProtector
from lucid.models.results import ParaphraseResult

if TYPE_CHECKING:
    from lucid.config import HumanizerConfig, OllamaConfig
    from lucid.detector.base import LUCIDDetector
    from lucid.models.results import DetectionResult
    from lucid.parser.chunk import ProseChunk

logger = logging.getLogger(__name__)

__all__ = ["LUCIDHumanizer"]


class LUCIDHumanizer:
    """Synchronous humanizer satisfying the ``Humanizer`` protocol.

    Wraps :class:`OllamaClient` (async) behind ``asyncio.run()`` so that
    callers can use a plain ``humanize(chunk, detection)`` call.

    Args:
        humanizer_config: Humanizer settings (retries, adversarial iterations, etc.).
        ollama_config: Ollama connection and model settings.
        detector: Detection engine used for adversarial re-scoring.
        profile: Quality profile (``"fast"``, ``"balanced"``, ``"quality"``).
    """

    def __init__(
        self,
        humanizer_config: HumanizerConfig,
        ollama_config: OllamaConfig,
        detector: LUCIDDetector,
        profile: str = "balanced",
    ) -> None:
        self._config = humanizer_config
        self._ollama_config = ollama_config
        self._detector = detector
        self._profile = profile

        self._term_protector = TermProtector(humanizer_config.term_protection)
        self._prompt_builder = PromptBuilder()

        self._model: str = getattr(ollama_config.models, profile)
        self._temperature: float = getattr(humanizer_config.temperature, profile)

    def humanize(
        self,
        chunk: ProseChunk,
        detection: DetectionResult,
    ) -> ParaphraseResult:
        """Paraphrase a prose chunk to reduce its AI detection score.

        Runs the full pipeline: term protection → prompt construction →
        LLM generation → placeholder validation → restoration. Delegates
        to :func:`~lucid.humanizer.adversarial.adversarial_humanize` when
        ``adversarial_iterations > 1``.

        Args:
            chunk: Prose chunk to humanize.
            detection: Detection result that triggered humanization.

        Returns:
            ParaphraseResult with the humanized text and metadata.
        """
        return asyncio.run(self._humanize_async(chunk, detection))

    async def _humanize_async(
        self,
        chunk: ProseChunk,
        detection: DetectionResult,
    ) -> ParaphraseResult:
        """Async implementation of the humanization pipeline."""
        async with OllamaClient(
            host=self._ollama_config.host,
            timeout=float(self._ollama_config.timeout_seconds),
        ) as client:
            if self._config.adversarial_iterations > 1:
                from lucid.humanizer.adversarial import adversarial_humanize

                return await adversarial_humanize(
                    chunk=chunk,
                    detection=detection,
                    client=client,
                    detector=self._detector,
                    term_protector=self._term_protector,
                    prompt_builder=self._prompt_builder,
                    config=self._config,
                    model=self._model,
                    profile=self._profile,
                )

            return await self._single_pass(chunk, detection, client)

    async def _single_pass(
        self,
        chunk: ProseChunk,
        detection: DetectionResult,
        client: OllamaClient,
    ) -> ParaphraseResult:
        """Execute a single-pass humanization (no adversarial loop)."""
        protected = self._term_protector.protect(chunk)

        strategy = select_strategy(0)
        prompt = self._prompt_builder.build(
            protected.text, strategy, chunk.domain_hint, self._profile
        )

        options = GenerateOptions(temperature=self._temperature)
        result = await client.generate(prompt, self._model, options=options)

        # Validate placeholders
        validation = self._term_protector.validate(
            result.text, protected.term_placeholders, protected.math_placeholders
        )
        if not validation.is_valid:
            raise ValueError(
                f"LLM dropped placeholders: {validation.missing_placeholders}"
            )

        # Restore original terms
        restored = self._term_protector.restore(
            result.text, protected.term_placeholders, protected.math_placeholders
        )

        return ParaphraseResult(
            chunk_id=chunk.id,
            original_text=chunk.text,
            humanized_text=restored,
            iteration_count=1,
            strategy_used=strategy.name,
            final_detection_score=detection.ensemble_score,
        )
