"""Evaluation pipeline orchestrator with tiered early exit.

Execution order:
    1. Term verification (regex, microseconds) → REJECT on failure
    2. Embedding similarity (cosine, ~1ms) → REJECT below threshold
    3. NLI entailment (bidirectional, ~30ms) → REJECT on contradiction/missing entailment
    4. BERTScore (optional, ~100ms) → REJECT below F1 threshold

Each stage populates its scores into the result before checking thresholds.
Early exit preserves all intermediate scores computed up to the rejection point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lucid.evaluator.bertscore import BERTScoreChecker
from lucid.evaluator.embedding import EmbeddingSimilarity
from lucid.evaluator.nli import NLIChecker
from lucid.evaluator.term_verify import TermVerifier
from lucid.models.results import EvaluationResult

if TYPE_CHECKING:
    from lucid.config import EvaluatorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PipelineOptions:
    """Runtime options controlling which stages execute.

    Args:
        run_bertscore: Enable the BERTScore stage (Stage 3). Typically
            activated only in the ``quality`` profile.
    """

    run_bertscore: bool = False


class EvaluationPipeline:
    """Tiered semantic evaluation with early exit.

    Owns all stage instances. Lazy loading propagates through each
    stage — models are loaded on first evaluation, not at construction.

    Args:
        config: Evaluator configuration with thresholds and model names.
    """

    def __init__(self, config: EvaluatorConfig) -> None:
        self._config = config
        self._term_verifier = TermVerifier()
        self._embedding = EmbeddingSimilarity(model_name=config.embedding_model)
        self._nli = NLIChecker(
            model_name=config.nli_model,
            require_bidirectional=config.nli_require_bidirectional,
        )
        self._bertscore = BERTScoreChecker(model_type=config.bertscore_model)

    def run(
        self,
        chunk_id: str,
        original: str,
        paraphrase: str,
        options: PipelineOptions | None = None,
    ) -> EvaluationResult:
        """Execute the evaluation pipeline with early exit on failure.

        Args:
            chunk_id: Identifier for the chunk being evaluated.
            original: Source text before paraphrasing.
            paraphrase: Rewritten text to evaluate.
            options: Runtime options. Defaults to BERTScore disabled.

        Returns:
            EvaluationResult with all intermediate scores populated up to
            the point of rejection (or all stages if passed).
        """
        if options is None:
            options = PipelineOptions()

        # Stage 0: Term verification (cheapest, most critical)
        term_result = self._term_verifier.verify(original, paraphrase)
        if not term_result.passed:
            logger.info("Chunk %s rejected at term verification: %s", chunk_id, term_result.reason)
            return EvaluationResult(
                chunk_id=chunk_id,
                passed=False,
                rejection_reason=f"term verification failed: {term_result.reason}",
            )

        # Stage 1: Embedding similarity
        emb_result = self._embedding.compute(original, paraphrase)
        embedding_similarity = emb_result.similarity

        if embedding_similarity < self._config.embedding_threshold:
            logger.info(
                "Chunk %s rejected at embedding: %.4f < %.4f",
                chunk_id,
                embedding_similarity,
                self._config.embedding_threshold,
            )
            return EvaluationResult(
                chunk_id=chunk_id,
                passed=False,
                embedding_similarity=embedding_similarity,
                rejection_reason=(
                    f"embedding similarity {embedding_similarity:.4f} "
                    f"below threshold {self._config.embedding_threshold}"
                ),
            )

        # Stage 2: NLI entailment
        nli_result = self._nli.check(original, paraphrase)
        nli_forward = nli_result.forward_label
        nli_backward = nli_result.backward_label

        nli_passed = self._check_nli_pass(nli_forward, nli_backward)
        if not nli_passed:
            reason = self._nli_rejection_reason(nli_forward, nli_backward)
            logger.info("Chunk %s rejected at NLI: %s", chunk_id, reason)
            return EvaluationResult(
                chunk_id=chunk_id,
                passed=False,
                embedding_similarity=embedding_similarity,
                nli_forward=nli_forward,
                nli_backward=nli_backward,
                rejection_reason=reason,
            )

        # Stage 3: BERTScore (optional)
        bertscore_f1: float | None = None
        if options.run_bertscore:
            bert_result = self._bertscore.compute(original, paraphrase)
            f1_score: float = bert_result.f1
            bertscore_f1 = f1_score

            if f1_score < self._config.bertscore_threshold:
                logger.info(
                    "Chunk %s rejected at BERTScore: %.4f < %.4f",
                    chunk_id,
                    bertscore_f1,
                    self._config.bertscore_threshold,
                )
                return EvaluationResult(
                    chunk_id=chunk_id,
                    passed=False,
                    embedding_similarity=embedding_similarity,
                    nli_forward=nli_forward,
                    nli_backward=nli_backward,
                    bertscore_f1=bertscore_f1,
                    rejection_reason=(
                        f"BERTScore F1 {bertscore_f1:.4f} "
                        f"below threshold {self._config.bertscore_threshold}"
                    ),
                )

        # All stages passed
        logger.info("Chunk %s passed all evaluation stages", chunk_id)
        return EvaluationResult(
            chunk_id=chunk_id,
            passed=True,
            embedding_similarity=embedding_similarity,
            nli_forward=nli_forward,
            nli_backward=nli_backward,
            bertscore_f1=bertscore_f1,
        )

    def _check_nli_pass(self, forward: str, backward: str) -> bool:
        """Determine whether NLI labels constitute a pass."""
        if forward == "contradiction" or backward == "contradiction":
            return False
        if self._config.nli_require_bidirectional:
            return forward == "entailment" and backward == "entailment"
        return forward == "entailment"

    def _nli_rejection_reason(self, forward: str, backward: str) -> str:
        """Build a human-readable NLI rejection reason."""
        if forward == "contradiction" or backward == "contradiction":
            return f"NLI detected contradiction (forward={forward}, backward={backward})"
        if self._config.nli_require_bidirectional:
            return (
                f"NLI bidirectional entailment required but got "
                f"forward={forward}, backward={backward}"
            )
        return f"NLI forward direction is {forward}, expected entailment"
