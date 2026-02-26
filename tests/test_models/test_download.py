"""Tests for ModelDownloader availability checks."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lucid.config import load_config
from lucid.models.download import ModelDownloader


@pytest.fixture()
def config() -> object:
    """Load balanced profile config for test fixtures."""
    return load_config(profile="balanced")


@pytest.fixture()
def downloader(config: object) -> ModelDownloader:
    return ModelDownloader(config)  # type: ignore[arg-type]


class TestCheckOllama:
    """Tests for Ollama server reachability."""

    @patch("httpx.get")
    def test_returns_true_on_200(self, mock_get: MagicMock, downloader: ModelDownloader) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        result = downloader.check_ollama()

        assert result is True
        mock_get.assert_called_once()

    @patch("httpx.get")
    def test_returns_false_on_connection_error(
        self, mock_get: MagicMock, downloader: ModelDownloader
    ) -> None:
        import httpx

        mock_get.side_effect = httpx.ConnectError("connection refused")

        result = downloader.check_ollama()

        assert result is False


class TestCheckOllamaModel:
    """Tests for Ollama model availability."""

    @patch("httpx.get")
    def test_returns_true_when_model_present(
        self, mock_get: MagicMock, downloader: ModelDownloader
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "qwen2.5:7b", "size": "4.4 GB"},
                {"name": "llama3.2:8b", "size": "4.7 GB"},
            ]
        }
        mock_get.return_value = mock_resp

        result = downloader.check_ollama_model("qwen2.5:7b")

        assert result is True

    @patch("httpx.get")
    def test_returns_false_when_model_absent(
        self, mock_get: MagicMock, downloader: ModelDownloader
    ) -> None:
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "llama3.2:8b", "size": "4.7 GB"},
            ]
        }
        mock_get.return_value = mock_resp

        result = downloader.check_ollama_model("qwen2.5:7b")

        assert result is False


class TestCheckHuggingfaceCache:
    """Tests for HuggingFace local cache detection."""

    @patch("huggingface_hub.try_to_load_from_cache")
    def test_returns_true_when_cached(
        self, mock_load: MagicMock, downloader: ModelDownloader
    ) -> None:
        mock_load.return_value = "/home/user/.cache/huggingface/hub/models--roberta/config.json"

        result = downloader._check_huggingface_cache("roberta-base-openai-detector")

        assert result is True

    @patch("huggingface_hub.try_to_load_from_cache")
    def test_returns_false_when_not_cached(
        self, mock_load: MagicMock, downloader: ModelDownloader
    ) -> None:
        mock_load.return_value = None

        result = downloader._check_huggingface_cache("roberta-base-openai-detector")

        assert result is False


class TestCheckAll:
    """Tests for full model availability sweep."""

    @patch("httpx.get")
    @patch("huggingface_hub.try_to_load_from_cache")
    def test_returns_correct_statuses(
        self,
        mock_hf_cache: MagicMock,
        mock_get: MagicMock,
        downloader: ModelDownloader,
    ) -> None:
        # All HuggingFace models cached
        mock_hf_cache.return_value = "/some/path/config.json"

        # Ollama model available
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [{"name": "qwen2.5:7b", "size": "4.4 GB"}]
        }
        mock_get.return_value = mock_resp

        statuses = downloader.check_all()

        assert len(statuses) == 5

        hf_statuses = [s for s in statuses if s.source == "huggingface"]
        assert len(hf_statuses) == 4
        assert all(s.available for s in hf_statuses)

        ollama_statuses = [s for s in statuses if s.source == "ollama"]
        assert len(ollama_statuses) == 1
        assert ollama_statuses[0].available is True
        assert ollama_statuses[0].name == "qwen2.5:7b"
