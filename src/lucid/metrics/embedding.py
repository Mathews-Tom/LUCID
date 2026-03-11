"""Embedding-based semantic similarity metric."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from lucid.core.types import MetricResult
from lucid.metrics import metric_registry

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_embedding_lock = threading.Lock()


@dataclass(frozen=True, slots=True)
class EmbeddingResult:
    """Cosine similarity between original and paraphrase embeddings."""

    similarity: float


class EmbeddingSimilarity:
    """Compute semantic similarity via sentence embeddings.

    Lazily loads the SentenceTransformer model on first call to ``compute``.
    """

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        if self._model is None:
            with _embedding_lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer

                    logger.info("Loading embedding model: %s", self._model_name)
                    self._model = SentenceTransformer(self._model_name)
        return self._model

    def compute(self, original: str, paraphrase: str) -> EmbeddingResult:
        """Encode both texts and return cosine similarity.

        Args:
            original: Source text.
            paraphrase: Rewritten text to compare against original.

        Returns:
            EmbeddingResult with cosine similarity in [-1, 1].
        """
        model = self._load_model()
        vectors: NDArray[np.float32] = model.encode([original, paraphrase])
        a, b = vectors[0], vectors[1]
        similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        logger.debug("Embedding similarity: %.4f", similarity)
        return EmbeddingResult(similarity=similarity)


@metric_registry.register("embedding_cosine")
class EmbeddingSimilarityMetric:
    """Metric protocol wrapper for embedding cosine similarity."""

    name: str = "embedding_cosine"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self._inner = EmbeddingSimilarity(model_name=model_name)

    def compute(self, original: str, transformed: str) -> MetricResult:
        result = self._inner.compute(original, transformed)
        return MetricResult(
            metric_name=self.name,
            value=result.similarity,
        )
