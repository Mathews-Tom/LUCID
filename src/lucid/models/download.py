"""Model availability checking and downloading for the LUCID pipeline.

Supports two model sources:
- HuggingFace Hub: transformer models cached locally
- Ollama: LLM models served via local HTTP API
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from lucid.config import LUCIDConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ModelStatus:
    """Availability status for a single model dependency."""

    name: str
    source: str  # "huggingface" | "ollama"
    available: bool
    size_estimate: str | None = None


class ModelDownloader:
    """Check availability and fetch models required by the LUCID pipeline."""

    def __init__(self, config: LUCIDConfig) -> None:
        self._config = config

    def check_all(self) -> list[ModelStatus]:
        """Check availability of all required models.

        Returns list of ModelStatus for:
        - detection.roberta_model (huggingface)
        - evaluator.embedding_model (huggingface)
        - evaluator.nli_model (huggingface)
        - evaluator.bertscore_model (huggingface)
        - ollama model for current profile (ollama)
        """
        statuses: list[ModelStatus] = []

        # HuggingFace models
        hf_models = [
            self._config.detection.roberta_model,
            self._config.evaluator.embedding_model,
            self._config.evaluator.nli_model,
            self._config.evaluator.bertscore_model,
        ]
        for model_id in hf_models:
            available = self._check_huggingface_cache(model_id)
            statuses.append(
                ModelStatus(name=model_id, source="huggingface", available=available)
            )

        # Ollama model for active profile
        profile = self._config.general.profile
        ollama_model: str = getattr(self._config.ollama.models, profile)
        ollama_available = self.check_ollama_model(ollama_model)
        statuses.append(
            ModelStatus(name=ollama_model, source="ollama", available=ollama_available)
        )

        return statuses

    def check_ollama(self) -> bool:
        """Check if Ollama server is reachable via GET {host}/api/tags."""
        import httpx

        try:
            resp = httpx.get(
                f"{self._config.ollama.host}/api/tags",
                timeout=5.0,
            )
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def check_ollama_model(self, model_tag: str) -> bool:
        """Check if a specific model is available in Ollama."""
        import httpx

        try:
            resp = httpx.get(
                f"{self._config.ollama.host}/api/tags",
                timeout=5.0,
            )
            if resp.status_code != 200:
                return False
            data: dict[str, list[dict[str, str]]] = resp.json()
            models = data.get("models", [])
            return any(
                m.get("name", "").startswith(model_tag.split(":")[0]) for m in models
            )
        except httpx.HTTPError:
            return False

    def _check_huggingface_cache(self, model_id: str) -> bool:
        """Check if a HuggingFace model is cached locally."""
        try:
            from huggingface_hub import try_to_load_from_cache

            result = try_to_load_from_cache(model_id, "config.json")
            # Returns a path string if cached, _CACHED_NO_EXIST sentinel
            # if explicitly marked non-existent, None if not in cache.
            return isinstance(result, str)
        except Exception:
            return False

    def download_huggingface(self, model_id: str) -> None:
        """Download a HuggingFace model to local cache."""
        from transformers import AutoModel

        AutoModel.from_pretrained(model_id)

    def pull_ollama_model(self, model_tag: str) -> None:
        """Pull an Ollama model via POST {host}/api/pull."""
        import httpx

        resp = httpx.post(
            f"{self._config.ollama.host}/api/pull",
            json={"name": model_tag},
            timeout=None,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"Failed to pull Ollama model {model_tag}: {resp.status_code}"
            )
