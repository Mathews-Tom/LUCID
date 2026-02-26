"""BERTScore-based semantic similarity for the evaluator pipeline."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bert_score import BERTScorer

logger = logging.getLogger(__name__)

_bertscore_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class BERTScoreResult:
    """Precision, recall, and F1 from BERTScore evaluation."""

    precision: float
    recall: float
    f1: float


class BERTScoreChecker:
    """Compute BERTScore between original and paraphrased text.

    Lazily loads the ``BERTScorer`` model on first call to ``compute``.
    """

    def __init__(self, model_type: str) -> None:
        self._model_type = model_type
        self._scorer: BERTScorer | None = None

    def _load_scorer(self) -> BERTScorer:
        if self._scorer is None:
            with _bertscore_lock:
                if self._scorer is None:
                    from bert_score import BERTScorer

                    logger.info("Loading BERTScore model: %s", self._model_type)
                    self._scorer = BERTScorer(
                        model_type=self._model_type,
                        rescale_with_baseline=True,
                        lang="en",
                    )
        return self._scorer

    def compute(self, original: str, paraphrase: str) -> BERTScoreResult:
        """Score semantic similarity between original and paraphrase.

        Args:
            original: Source text.
            paraphrase: Rewritten text to compare against original.

        Returns:
            BERTScoreResult with precision, recall, and F1 in approx [-1, 1]
            (rescaled with baseline).
        """
        scorer = self._load_scorer()
        p, r, f1 = scorer.score(cands=[paraphrase], refs=[original])
        result = BERTScoreResult(
            precision=p[0].item(),
            recall=r[0].item(),
            f1=f1[0].item(),
        )
        logger.debug(
            "BERTScore — P: %.4f  R: %.4f  F1: %.4f",
            result.precision,
            result.recall,
            result.f1,
        )
        return result
