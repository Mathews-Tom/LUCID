"""NLI-based semantic equivalence metric.

Uses a DeBERTa-v3 model fine-tuned on NLI tasks to verify that a paraphrase
entails -- and is entailed by -- the original text. Bidirectional entailment
confirms semantic preservation; contradiction or neutrality flags meaning drift.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from lucid.core.types import MetricResult
from lucid.metrics import metric_registry

if TYPE_CHECKING:
    from transformers.pipelines.text_classification import TextClassificationPipeline

logger = logging.getLogger(__name__)

_nli_lock = threading.Lock()
_nli_inference_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class NLIResult:
    """Result of bidirectional NLI inference between original and paraphrase.

    Args:
        forward_label: Highest-scoring label for original -> paraphrase.
        backward_label: Highest-scoring label for paraphrase -> original.
        forward_scores: Full label -> probability mapping for the forward direction.
        backward_scores: Full label -> probability mapping for the backward direction.
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
            with _nli_lock:
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
        """Run single-direction NLI classification."""
        with _nli_inference_lock:
            raw: list[dict[str, Any]] = self.pipeline(  # type: ignore[assignment]
                {"text": premise, "text_pair": hypothesis},
                top_k=None,
            )
        return {item["label"]: float(item["score"]) for item in raw}

    def close(self) -> None:
        """Release cached pipeline references for cleaner shutdown."""
        self._pipeline = None

    @staticmethod
    def _top_label(scores: dict[str, float]) -> str:
        """Return the label with the highest score."""
        return max(scores, key=scores.__getitem__)


@metric_registry.register("nli_entailment")
class NLIEntailmentMetric:
    """Metric protocol wrapper for NLI entailment checking."""

    name: str = "nli_entailment"

    def __init__(
        self,
        model_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-docnli-ling-2c",
    ) -> None:
        self._inner = NLIChecker(model_name=model_name, require_bidirectional=True)

    def compute(self, original: str, transformed: str) -> MetricResult:
        result = self._inner.check(original, transformed)
        # Score: 1.0 if bidirectional entailment, 0.5 if forward-only, 0.0 otherwise
        if result.forward_label == "entailment" and result.backward_label == "entailment":
            score = 1.0
        elif result.forward_label == "entailment":
            score = 0.5
        elif result.forward_label == "contradiction" or result.backward_label == "contradiction":
            score = 0.0
        else:
            score = 0.25
        return MetricResult(
            metric_name=self.name,
            value=score,
            metadata={
                "forward_label": result.forward_label,
                "backward_label": result.backward_label,
            },
        )
