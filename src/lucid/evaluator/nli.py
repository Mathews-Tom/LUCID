"""NLI-based semantic equivalence checker for the evaluation pipeline.

Uses a DeBERTa-v3 model fine-tuned on NLI tasks to verify that a paraphrase
entails — and is entailed by — the original text. Bidirectional entailment
confirms semantic preservation; contradiction or neutrality flags meaning drift.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from transformers.pipelines.text_classification import TextClassificationPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class NLIResult:
    """Result of bidirectional NLI inference between original and paraphrase.

    Args:
        forward_label: Highest-scoring label for original → paraphrase.
        backward_label: Highest-scoring label for paraphrase → original.
        forward_scores: Full label → probability mapping for the forward direction.
        backward_scores: Full label → probability mapping for the backward direction.
    """

    forward_label: str
    backward_label: str
    forward_scores: dict[str, float]
    backward_scores: dict[str, float]


class NLIChecker:
    """Bidirectional NLI checker using a HuggingFace text-classification pipeline.

    The model is lazy-loaded on first call to ``check()`` and cached for reuse.

    Args:
        model_name: HuggingFace model identifier for NLI classification.
        require_bidirectional: When True, both forward and backward directions
            must yield entailment for a paraphrase to be considered valid.
    """

    def __init__(self, model_name: str, require_bidirectional: bool = True) -> None:
        self._model_name = model_name
        self._require_bidirectional = require_bidirectional
        self._pipeline: TextClassificationPipeline | None = None

    @property
    def pipeline(self) -> TextClassificationPipeline:
        """Return the cached pipeline, loading it on first access."""
        if self._pipeline is None:
            logger.info("Loading NLI model: %s", self._model_name)
            from transformers import pipeline as hf_pipeline

            self._pipeline = hf_pipeline(
                "text-classification",
                model=self._model_name,
                top_k=None,
            )
            logger.info("NLI model loaded successfully")
        return self._pipeline

    def check(self, original: str, paraphrase: str) -> NLIResult:
        """Run bidirectional NLI inference between original and paraphrase.

        Args:
            original: Source text (premise in forward direction).
            paraphrase: Rewritten text (hypothesis in forward direction).

        Returns:
            NLIResult with labels and score distributions for both directions.
        """
        forward_scores = self._classify(original, paraphrase)
        backward_scores = self._classify(paraphrase, original)

        forward_label = self._top_label(forward_scores)
        backward_label = self._top_label(backward_scores)

        logger.debug(
            "NLI forward=%s (%.3f), backward=%s (%.3f)",
            forward_label,
            forward_scores[forward_label],
            backward_label,
            backward_scores[backward_label],
        )

        return NLIResult(
            forward_label=forward_label,
            backward_label=backward_label,
            forward_scores=forward_scores,
            backward_scores=backward_scores,
        )

    def _classify(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Run single-direction NLI classification.

        Args:
            premise: The text treated as the premise.
            hypothesis: The text treated as the hypothesis.

        Returns:
            Mapping of label → probability score.
        """
        raw: list[dict[str, Any]] = self.pipeline(  # type: ignore[assignment]
            {"text": premise, "text_pair": hypothesis},
            top_k=None,
        )
        return {item["label"]: float(item["score"]) for item in raw}

    @staticmethod
    def _top_label(scores: dict[str, float]) -> str:
        """Return the label with the highest score."""
        return max(scores, key=scores.__getitem__)
