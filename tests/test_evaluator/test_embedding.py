"""Tests for embedding-based semantic similarity."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from lucid.evaluator.embedding import EmbeddingSimilarity


class TestEmbeddingSimilarity:
    """Unit tests — model is mocked, no downloads."""

    def setup_method(self) -> None:
        self.scorer = EmbeddingSimilarity(model_name="mock-model")

    def test_lazy_loading_deferred(self) -> None:
        """Model is not loaded at construction time."""
        assert self.scorer._model is None

    @patch("sentence_transformers.SentenceTransformer")
    def test_lazy_loading_triggered_on_compute(self, mock_cls: MagicMock) -> None:
        """First call to compute triggers model loading."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        mock_cls.return_value = mock_model

        self.scorer.compute("a", "b")
        mock_cls.assert_called_once_with("mock-model")

    @patch("sentence_transformers.SentenceTransformer")
    def test_identical_vectors_similarity_one(self, mock_cls: MagicMock) -> None:
        """Identical vectors yield cosine similarity of 1.0."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[3.0, 4.0], [3.0, 4.0]], dtype=np.float32)
        mock_cls.return_value = mock_model

        result = self.scorer.compute("same", "same")
        assert result.similarity == pytest.approx(1.0)

    @patch("sentence_transformers.SentenceTransformer")
    def test_orthogonal_vectors_similarity_zero(self, mock_cls: MagicMock) -> None:
        """Orthogonal vectors yield cosine similarity of 0.0."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mock_cls.return_value = mock_model

        result = self.scorer.compute("a", "b")
        assert result.similarity == pytest.approx(0.0)

    @patch("sentence_transformers.SentenceTransformer")
    def test_opposite_vectors_similarity_negative(self, mock_cls: MagicMock) -> None:
        """Opposite vectors yield cosine similarity of -1.0."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [-1.0, 0.0]], dtype=np.float32)
        mock_cls.return_value = mock_model

        result = self.scorer.compute("a", "b")
        assert result.similarity == pytest.approx(-1.0)

    @patch("sentence_transformers.SentenceTransformer")
    def test_known_angle_similarity(self, mock_cls: MagicMock) -> None:
        """45-degree vectors yield cosine similarity of ~0.7071."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        mock_cls.return_value = mock_model

        result = self.scorer.compute("a", "b")
        assert result.similarity == pytest.approx(1.0 / np.sqrt(2.0))

    @patch("sentence_transformers.SentenceTransformer")
    def test_model_loaded_once(self, mock_cls: MagicMock) -> None:
        """Repeated compute calls load the model only once."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        mock_cls.return_value = mock_model

        self.scorer.compute("a", "b")
        self.scorer.compute("c", "d")
        mock_cls.assert_called_once()

    @patch("sentence_transformers.SentenceTransformer")
    def test_result_is_frozen(self, mock_cls: MagicMock) -> None:
        """EmbeddingResult is immutable."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        mock_cls.return_value = mock_model

        result = self.scorer.compute("a", "b")
        with pytest.raises(AttributeError):
            result.similarity = 0.5  # type: ignore[misc]


@pytest.mark.integration
class TestEmbeddingSimilarityIntegration:
    """Integration tests — downloads and runs the real model."""

    def setup_method(self) -> None:
        self.scorer = EmbeddingSimilarity(model_name="sentence-transformers/all-MiniLM-L6-v2")

    def test_identical_texts(self) -> None:
        """Identical strings produce similarity near 1.0."""
        text = "The transformer architecture revolutionized natural language processing."
        result = self.scorer.compute(text, text)
        assert result.similarity == pytest.approx(1.0, abs=0.01)

    def test_clear_paraphrase_above_threshold(self) -> None:
        """Semantically equivalent paraphrase scores above 0.8."""
        original = "Deep learning models require large amounts of training data."
        paraphrase = "Large datasets are necessary to train deep learning systems."
        result = self.scorer.compute(original, paraphrase)
        assert result.similarity > 0.8

    def test_unrelated_texts_below_threshold(self) -> None:
        """Completely unrelated texts score below 0.5."""
        original = "Quantum computing uses qubits for parallel computation."
        unrelated = "The recipe calls for two cups of flour and one egg."
        result = self.scorer.compute(original, unrelated)
        assert result.similarity < 0.5
