"""Embedding-based semantic similarity for the evaluator pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


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
