"""Content transformation via Ollama LLM inference.

The :class:`LUCIDTransformer` satisfies the :class:`~lucid.core.protocols.Transformer`
protocol, wrapping the async :class:`OllamaClient` behind a synchronous
``transform()`` call.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from lucid.transform.ollama import GenerateOptions, OllamaClient
from lucid.transform.prompts import PromptBuilder
from lucid.transform.operators import select_operator
from lucid.transform.term_protect import TermProtector
from lucid.models.results import TransformResult

if TYPE_CHECKING:
    from lucid.config import TransformConfig, OllamaConfig
    from lucid.detector.base import LUCIDDetector
    from lucid.models.results import DetectionResult
    from lucid.parser.chunk import ProseChunk

logger = logging.getLogger(__name__)

__all__ = ["LUCIDTransformer"]


class LUCIDTransformer:
    """Synchronous transformer satisfying the ``Transformer`` protocol.

    Wraps :class:`OllamaClient` (async) behind ``asyncio.run()`` so that
    callers can use a plain ``transform(chunk, detection)`` call.

    Args:
        transform_config: Transform settings (retries, search iterations, etc.).
        ollama_config: Ollama connection and model settings.
        detector: Detection engine used for search re-scoring.
        profile: Quality profile (``"fast"``, ``"balanced"``, ``"quality"``).
    """

    def __init__(
        self,
        transform_config: TransformConfig,
        ollama_config: OllamaConfig,
        detector: LUCIDDetector,
        profile: str = "balanced",
    ) -> None:
        self._config = transform_config
        self._ollama_config = ollama_config
        self._detector = detector
        self._profile = profile

        self._term_protector = TermProtector(transform_config.term_protection)
        self._prompt_builder = PromptBuilder()

        self._model: str = getattr(ollama_config.models, profile)
        self._temperature: float = getattr(transform_config.temperature, profile)
        self._model_resolved: bool = False

    def transform_batch(
        self,
        chunks_and_detections: list[tuple[ProseChunk, DetectionResult]],
        max_concurrent: int = 4,
    ) -> list[TransformResult | BaseException]:
        """Transform multiple chunks concurrently via a single event loop.

        Uses asyncio.Semaphore to limit concurrent Ollama requests.
        Individual chunk failures are returned as BaseException instances
        at the corresponding index (via ``return_exceptions=True``).

        Args:
            chunks_and_detections: Pairs of (chunk, detection) to process.
            max_concurrent: Maximum concurrent Ollama requests.

        Returns:
            List of TransformResult or BaseException, positionally matching input.
        """
        return asyncio.run(
            self._transform_batch_async(chunks_and_detections, max_concurrent)
        )

    async def _transform_batch_async(
        self,
        chunks_and_detections: list[tuple[ProseChunk, DetectionResult]],
        max_concurrent: int,
    ) -> list[TransformResult | BaseException]:
        """Async implementation of batch transformation."""
        semaphore = asyncio.Semaphore(max_concurrent)
        async with OllamaClient(
            host=self._ollama_config.host,
            timeout=float(self._ollama_config.timeout_seconds),
        ) as client:
            await self._resolve_model(client)

            async def process_one(
                chunk: ProseChunk, detection: DetectionResult
            ) -> TransformResult:
                async with semaphore:
                    if self._config.search_iterations > 1:
                        from lucid.transform.search import transformation_search

                        return await transformation_search(
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

            tasks = [
                process_one(chunk, detection)
                for chunk, detection in chunks_and_detections
            ]
            results: list[TransformResult | BaseException] = await asyncio.gather(
                *tasks, return_exceptions=True
            )
            return results

    def transform(
        self,
        chunk: ProseChunk,
        detection: DetectionResult,
    ) -> TransformResult:
        """Transform a prose chunk to reduce its AI detection score.

        Runs the full pipeline: term protection -> prompt construction ->
        LLM generation -> placeholder validation -> restoration. Delegates
        to :func:`~lucid.transform.search.transformation_search` when
        ``search_iterations > 1``.

        Args:
            chunk: Prose chunk to transform.
            detection: Detection result that triggered transformation.

        Returns:
            TransformResult with the transformed text and metadata.
        """
        return asyncio.run(self._transform_async(chunk, detection))

    async def _resolve_model(self, client: OllamaClient) -> None:
        """Resolve configured model tag to an available model (once)."""
        if self._model_resolved:
            return
        resolved = await client.resolve_model(self._model)
        if resolved != self._model:
            logger.warning(
                "Configured model %r not found; resolved to %r",
                self._model,
                resolved,
            )
            self._model = resolved
        self._model_resolved = True

    async def _transform_async(
        self,
        chunk: ProseChunk,
        detection: DetectionResult,
    ) -> TransformResult:
        """Async implementation of the transformation pipeline."""
        async with OllamaClient(
            host=self._ollama_config.host,
            timeout=float(self._ollama_config.timeout_seconds),
        ) as client:
            await self._resolve_model(client)
            if self._config.search_iterations > 1:
                from lucid.transform.search import transformation_search

                return await transformation_search(
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
    ) -> TransformResult:
        """Execute a single-pass transformation (no search loop)."""
        protected = self._term_protector.protect(chunk)
        all_placeholders = protected.all_placeholders()

        strategy = select_operator(0, placeholder_count=len(all_placeholders))
        prompt = self._prompt_builder.build(
            protected.text, strategy, chunk.domain_hint, self._profile,
            placeholders=all_placeholders,
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

        return TransformResult(
            chunk_id=chunk.id,
            original_text=chunk.text,
            transformed_text=restored,
            iteration_count=1,
            operator_used=strategy.name,
            final_detection_score=detection.ensemble_score,
        )
