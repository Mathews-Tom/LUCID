"""Semantic evaluation: embedding similarity, NLI, BERTScore.

The :class:`LUCIDEvaluator` satisfies the :class:`~lucid.core.protocols.Evaluator`
protocol, wrapping the tiered :class:`EvaluationPipeline` behind a synchronous
``evaluate()`` call.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from lucid.evaluator.pipeline import EvaluationPipeline, PipelineOptions

if TYPE_CHECKING:
    from lucid.config import EvaluatorConfig
    from lucid.models.results import EvaluationResult

logger = logging.getLogger(__name__)

__all__ = ["LUCIDEvaluator"]


class LUCIDEvaluator:
    """Synchronous evaluator satisfying the ``Evaluator`` protocol.

    Determines pipeline options from the quality profile: BERTScore is
    activated only in ``quality`` mode to avoid unnecessary latency in
    ``fast`` and ``balanced`` profiles.

    Args:
        config: Evaluator settings (thresholds, model names).
        profile: Quality profile (``"fast"``, ``"balanced"``, ``"quality"``).
    """

    def __init__(self, config: EvaluatorConfig, profile: str = "balanced") -> None:
        self._config = config
        self._profile = profile
        self._pipeline = EvaluationPipeline(config)
        self._options = PipelineOptions(run_bertscore=(profile == "quality"))

    def evaluate(self, original: str, paraphrase: str) -> EvaluationResult:
        """Evaluate semantic preservation between original and paraphrased text.

        Satisfies the ``Evaluator`` protocol. Uses a synthetic chunk ID
        since the protocol does not expose chunk identifiers.

        Args:
            original: Source text before paraphrasing.
            paraphrase: Rewritten text to evaluate.

        Returns:
            EvaluationResult with pass/fail status and intermediate scores.
        """
        return self._pipeline.run(
            chunk_id="__protocol__",
            original=original,
            paraphrase=paraphrase,
            options=self._options,
        )

    def evaluate_chunk(
        self,
        chunk_id: str,
        original: str,
        paraphrase: str,
    ) -> EvaluationResult:
        """Evaluate a specific chunk by ID — for pipeline integration.

        Args:
            chunk_id: Identifier of the chunk being evaluated.
            original: Source text before paraphrasing.
            paraphrase: Rewritten text to evaluate.

        Returns:
            EvaluationResult with the provided chunk_id.
        """
        return self._pipeline.run(
            chunk_id=chunk_id,
            original=original,
            paraphrase=paraphrase,
            options=self._options,
        )
