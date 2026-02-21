"""Main detection orchestrator satisfying the Detector protocol."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from lucid.config import DetectionConfig
from lucid.detector.binoculars import BinocularsDetector
from lucid.detector.ensemble import classify, compute_ensemble
from lucid.detector.roberta import RobertaDetector
from lucid.detector.statistical import StatisticalDetector
from lucid.models.results import DetectionResult

if TYPE_CHECKING:
    from lucid.parser.chunk import ProseChunk

logger = logging.getLogger(__name__)


class LUCIDDetector:
    """Multi-tier AI text detection engine.

    Orchestrates RoBERTa (Tier 1), statistical features (Tier 2),
    and optional Binoculars (Tier 3) into an ensemble detection result.
    Satisfies the Detector protocol from lucid.core.protocols.

    Args:
        config: Detection configuration controlling which tiers are active
            and the ensemble weight/threshold settings.
    """

    def __init__(self, config: DetectionConfig) -> None:
        self._config = config
        self._roberta = RobertaDetector(model_id=config.roberta_model)

        self._statistical: StatisticalDetector | None = None
        if config.use_statistical:
            self._statistical = StatisticalDetector()

        self._binoculars: BinocularsDetector | None = None
        if config.use_binoculars or config.thresholds.ambiguity_triggers_binoculars:
            self._binoculars = BinocularsDetector()

    def detect(self, chunk: ProseChunk) -> DetectionResult:
        """Score a prose chunk for AI-generated content probability.

        Runs Tier 1 (RoBERTa) unconditionally. Tier 2 (statistical) and
        Tier 3 (Binoculars) are run conditionally based on config and the
        ambiguity of earlier tier results.

        Args:
            chunk: Prose chunk to analyze. Uses chunk.protected_text for scoring.

        Returns:
            DetectionResult with ensemble score, classification, and per-tier scores.
        """
        text = chunk.protected_text

        # Tier 1: always runs
        roberta_score = self._roberta.detect_text(text)

        # Tier 2: conditional on config.use_statistical
        statistical_score: float | None = None
        feature_details: dict[str, Any] = {}
        if self._statistical is not None:
            try:
                raw_score = self._statistical.score(text)
                # score() returns None for texts below MIN_WORDS_THRESHOLD
                statistical_score = raw_score
                feature_details = self._statistical.extract_features(text)
            except Exception:
                logger.warning(
                    "Statistical detection failed, continuing with Tier 1 only",
                    exc_info=True,
                )

        # Tier 3: conditional on config and ambiguity
        binoculars_score: float | None = None
        if self._should_run_binoculars(roberta_score, statistical_score):
            try:
                assert self._binoculars is not None
                binoculars_score = self._binoculars.score(text)
            except Exception:
                logger.warning(
                    "Binoculars detection failed, falling back to Tier 1+2",
                    exc_info=True,
                )

        # Select ensemble weights based on whether Tier 3 contributed
        weights = (
            self._config.ensemble_weights_with_binoculars
            if binoculars_score is not None
            else self._config.ensemble_weights
        )
        ensemble_score = compute_ensemble(
            roberta_score, statistical_score, binoculars_score, weights
        )
        classification = classify(ensemble_score, self._config.thresholds)

        return DetectionResult(
            chunk_id=chunk.id,
            ensemble_score=ensemble_score,
            classification=classification,
            roberta_score=roberta_score,
            statistical_score=statistical_score,
            binoculars_score=binoculars_score,
            feature_details=feature_details,
        )

    def _should_run_binoculars(
        self, roberta_score: float, statistical_score: float | None
    ) -> bool:
        """Determine whether the Binoculars tier should activate.

        Returns True when:
        - Binoculars detector is initialized, AND
        - Either use_binoculars=True (always run), OR
        - ambiguity_triggers_binoculars=True AND the Tier 1+2 ensemble
          falls in the ambiguous band (human_max < score < ai_min).

        Args:
            roberta_score: Tier 1 score.
            statistical_score: Tier 2 score, or None if unavailable.

        Returns:
            True if Binoculars should run, False otherwise.
        """
        if self._binoculars is None:
            return False
        if self._config.use_binoculars:
            return True
        if self._config.thresholds.ambiguity_triggers_binoculars:
            partial = compute_ensemble(
                roberta_score,
                statistical_score,
                None,
                self._config.ensemble_weights,
            )
            t = self._config.thresholds
            return t.human_max < partial < t.ai_min
        return False

    def unload_binoculars(self) -> None:
        """Explicitly release Binoculars model memory."""
        if self._binoculars is not None:
            self._binoculars.unload()

    def detect_batch(self, chunks: list[ProseChunk]) -> list[DetectionResult]:
        """Detect AI content in multiple chunks, preserving input order.

        Args:
            chunks: Sequence of prose chunks to analyze.

        Returns:
            List of DetectionResult in the same order as the input chunks.
        """
        return [self.detect(chunk) for chunk in chunks]
