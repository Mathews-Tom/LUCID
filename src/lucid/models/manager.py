"""Model lifecycle management for the LUCID pipeline.

Centralizes initialization, caching, and teardown of the three heavy
ML components: detector, transformer, and evaluator.
"""

from __future__ import annotations

import gc
import logging
from typing import TYPE_CHECKING

import psutil

from lucid.config import LUCIDConfig

if TYPE_CHECKING:
    from lucid.detector.base import LUCIDDetector
    from lucid.evaluator import LUCIDEvaluator
    from lucid.transform import LUCIDTransformer

logger = logging.getLogger(__name__)


class ModelManager:
    """Owns the lifecycle of LUCID's ML components.

    Each component is lazily initialized via its ``initialize_*`` method
    and cached for reuse.  Property accessors enforce that callers cannot
    reach an uninitialized component.

    Args:
        config: Fully resolved LUCID configuration tree.
    """

    def __init__(self, config: LUCIDConfig) -> None:
        self._config = config
        self._detector: LUCIDDetector | None = None
        self._transformer: LUCIDTransformer | None = None
        self._evaluator: LUCIDEvaluator | None = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize_detector(self) -> LUCIDDetector:
        """Create and cache a LUCIDDetector instance."""
        from lucid.detector.base import LUCIDDetector

        self._detector = LUCIDDetector(self._config.detection)
        return self._detector

    def initialize_transformer(self) -> LUCIDTransformer:
        """Create and cache LUCIDTransformer. Requires detector already initialized."""
        if self._detector is None:
            raise RuntimeError("Detector must be initialized before transformer")

        from lucid.transform import LUCIDTransformer

        self._transformer = LUCIDTransformer(
            self._config.humanizer,
            self._config.ollama,
            self._detector,
            self._config.general.profile,
        )
        return self._transformer

    def initialize_evaluator(self) -> LUCIDEvaluator:
        """Create and cache LUCIDEvaluator."""
        from lucid.evaluator import LUCIDEvaluator

        self._evaluator = LUCIDEvaluator(
            self._config.evaluator,
            self._config.general.profile,
        )
        return self._evaluator

    # ------------------------------------------------------------------
    # Property accessors (fail-fast on uninitialized)
    # ------------------------------------------------------------------

    @property
    def detector(self) -> LUCIDDetector:
        """Return the cached detector or raise if not initialized."""
        if self._detector is None:
            raise RuntimeError(
                "Detector not initialized — call initialize_detector() first"
            )
        return self._detector

    @property
    def transformer(self) -> LUCIDTransformer:
        """Return the cached transformer or raise if not initialized."""
        if self._transformer is None:
            raise RuntimeError(
                "Transformer not initialized — call initialize_transformer() first"
            )
        return self._transformer

    @property
    def evaluator(self) -> LUCIDEvaluator:
        """Return the cached evaluator or raise if not initialized."""
        if self._evaluator is None:
            raise RuntimeError(
                "Evaluator not initialized — call initialize_evaluator() first"
            )
        return self._evaluator

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------

    def release_detection_models(self) -> None:
        """Unload heavy detection models to free memory for transformation.

        Call between detection and transformation phases.
        Logs a warning if available memory is below 4GB after release.
        """
        if self._detector is not None:
            self._detector.unload_binoculars()
        gc.collect()

        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < 4.0:
            logger.warning(
                "Low memory after detector release: %.1fGB available. "
                "Transformation may be slow or fail.",
                available_gb,
            )

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """Release all component references and collect garbage."""
        if self._detector is not None:
            self._detector.unload_binoculars()
        self._detector = None
        self._transformer = None
        self._evaluator = None
        gc.collect()
