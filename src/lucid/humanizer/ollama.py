"""Async Ollama HTTP client with retry logic and streaming support."""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Self

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import httpx

# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------


class OllamaError(Exception):
    """Base exception for Ollama client errors."""


class OllamaConnectionError(OllamaError):
    """Ollama server is unreachable."""


class OllamaTimeoutError(OllamaError):
    """Request exceeded configured timeout."""


class OllamaModelNotFoundError(OllamaError):
    """Requested model is not available on the server.

    Attributes:
        model: The model that was requested.
        available_models: Models currently pulled on the server.
    """

    def __init__(self, model: str, available_models: list[str]) -> None:
        self.model = model
        self.available_models = available_models
        super().__init__(
            f"Model {model!r} not found. Available: {', '.join(available_models) or 'none'}"
        )


class OllamaEmptyResponseError(OllamaError):
    """Ollama returned an empty completion."""


class OllamaMalformedResponseError(OllamaError):
    """Ollama response had invalid JSON or missing required fields."""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GenerateOptions:
    """Parameters for Ollama generation requests."""

    temperature: float = 0.6
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    num_predict: int = 512
    num_ctx: int = 2048

    def to_dict(self) -> dict[str, Any]:
        """Convert to Ollama API options dict."""
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }


@dataclass(frozen=True, slots=True)
class ModelInfo:
    """Metadata for a model available on the Ollama server."""

    name: str
    size: int
    digest: str
    modified_at: str


@dataclass(frozen=True, slots=True)
class GenerateResult:
    """Result from a non-streaming generate request."""

    text: str
    model: str
    total_duration_ns: int
    eval_count: int


# ---------------------------------------------------------------------------
# Model resolution helpers
# ---------------------------------------------------------------------------

# Map common size suffixes to approximate byte sizes for comparison.
_SIZE_SUFFIX_BYTES: dict[str, int] = {
    "1b": 1_000_000_000,
    "3b": 3_000_000_000,
    "3.8b": 3_800_000_000,
    "7b": 7_000_000_000,
    "8b": 8_000_000_000,
    "13b": 13_000_000_000,
    "14b": 14_000_000_000,
    "32b": 32_000_000_000,
    "70b": 70_000_000_000,
}


def _extract_model_family(tag: str) -> str:
    """Extract the base model family from an Ollama tag.

    Examples:
        ``llama3.1:8b``   → ``llama3``
        ``llama3.2:latest`` → ``llama3``
        ``qwen2.5:7b``    → ``qwen2``
        ``phi3:3.8b``     → ``phi3``
        ``gemma3:latest``  → ``gemma3``

    Strips the version suffix (digits/dots after the family name) and the
    size/tag portion after the colon.
    """
    # Remove everything after ':'
    base = tag.split(":")[0]
    # Strip trailing version numbers: "llama3.1" → "llama3", "qwen2.5" → "qwen2"
    # Keep the first digit sequence attached to the name (e.g., "llama3", "phi3")
    # but strip subsequent .N version suffixes.
    match = re.match(r"^([a-zA-Z]+-?[a-zA-Z]*\d*)(?:\.\d+)*$", base)
    if match:
        return match.group(1)
    return base


def _extract_size_bytes(tag: str, models: list[ModelInfo]) -> int:
    """Estimate model size in bytes from the tag suffix or model metadata.

    Checks the tag suffix first (e.g., ``:8b``), then falls back to the
    ``ModelInfo.size`` field if the model is in the list.
    """
    if ":" in tag:
        suffix = tag.split(":", 1)[1].lower()
        if suffix in _SIZE_SUFFIX_BYTES:
            return _SIZE_SUFFIX_BYTES[suffix]

    # Fall back to metadata
    for m in models:
        if m.name == tag:
            return m.size
    return 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Async HTTP client for the Ollama REST API.

    Usage::

        async with OllamaClient() as client:
            result = await client.generate("Hello", model="phi3:3.8b")
            print(result.text)
    """

    def __init__(
        self,
        host: str = "http://localhost:11434",
        timeout: float = 60.0,
    ) -> None:
        self._host = host.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> Self:
        self._client = httpx.AsyncClient(
            base_url=self._host,
            timeout=httpx.Timeout(self._timeout, connect=10.0),
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        """Return the underlying httpx client, raising if not in context manager."""
        if self._client is None:
            raise RuntimeError("OllamaClient must be used as an async context manager")
        return self._client

    # -- Health & model listing ------------------------------------------------

    async def health_check(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            resp = await self.client.get("/")
            return resp.status_code == 200
        except httpx.ConnectError:
            return False
        except httpx.TimeoutException:
            return False

    async def list_models(self) -> list[ModelInfo]:
        """List all models pulled on the server."""
        try:
            resp = await self.client.get("/api/tags")
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Cannot connect to {self._host}") from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Timed out listing models") from exc

        data = resp.json()
        models: list[ModelInfo] = []
        for m in data.get("models", []):
            models.append(
                ModelInfo(
                    name=m["name"],
                    size=m.get("size", 0),
                    digest=m.get("digest", ""),
                    modified_at=m.get("modified_at", ""),
                )
            )
        return models

    async def list_running_models(self) -> list[str]:
        """List models currently loaded in memory."""
        try:
            resp = await self.client.get("/api/ps")
            resp.raise_for_status()
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Cannot connect to {self._host}") from exc

        data = resp.json()
        return [m["name"] for m in data.get("models", [])]

    async def is_model_available(self, model: str) -> bool:
        """Check if a model is pulled, trying with :latest suffix as fallback."""
        models = await self.list_models()
        names = {m.name for m in models}
        if model in names:
            return True
        return ":" not in model and f"{model}:latest" in names

    async def resolve_model(self, requested: str) -> str:
        """Resolve a requested model tag to an available model.

        Resolution order:
        1. Exact match → return as-is.
        2. Family match → find available models sharing the same base family
           (e.g., ``llama3.1:8b`` matches ``llama3.2:latest``), prefer the
           closest size match.
        3. No match → raise OllamaModelNotFoundError.

        Args:
            requested: The configured model tag (e.g., ``"llama3.1:8b"``).

        Returns:
            An available model tag on the server.

        Raises:
            OllamaModelNotFoundError: No exact or family match found.
        """
        models = await self.list_models()
        names = {m.name for m in models}

        # 1. Exact match
        if requested in names:
            return requested
        # Also try :latest suffix
        if ":" not in requested and f"{requested}:latest" in names:
            return f"{requested}:latest"

        # 2. Family match
        family = _extract_model_family(requested)
        requested_size = _extract_size_bytes(requested, models)

        candidates: list[ModelInfo] = []
        for m in models:
            if _extract_model_family(m.name) == family:
                candidates.append(m)

        if candidates:
            # Sort by size proximity to the requested model's expected size
            if requested_size > 0:
                candidates.sort(key=lambda m: abs(m.size - requested_size))
            else:
                # No size info — prefer models with larger size (higher quality)
                candidates.sort(key=lambda m: m.size, reverse=True)
            return candidates[0].name

        # 3. No match
        raise OllamaModelNotFoundError(requested, [m.name for m in models])

    # -- Generation ------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        model: str,
        options: GenerateOptions | None = None,
        max_retries: int = 3,
    ) -> GenerateResult:
        """Generate a non-streaming completion with retry logic.

        Retries with exponential backoff (1s, 2s, 4s) on connection/timeout
        errors. On empty response, retries once with temperature + 0.1.

        Args:
            prompt: The input prompt text.
            model: Ollama model tag (e.g., "phi3:3.8b").
            options: Generation parameters.
            max_retries: Maximum retry attempts for transient failures.

        Returns:
            GenerateResult with the completion text and metadata.

        Raises:
            OllamaConnectionError: Server unreachable after all retries.
            OllamaTimeoutError: Request timed out after all retries.
            OllamaModelNotFoundError: Model not pulled on server.
            OllamaEmptyResponseError: Empty response after retry.
            OllamaMalformedResponseError: Invalid response format.
        """
        if options is None:
            options = GenerateOptions()

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": options.to_dict(),
        }

        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                resp = await self.client.post("/api/generate", json=payload)

                if resp.status_code == 404:
                    available = await self.list_models()
                    raise OllamaModelNotFoundError(model, [m.name for m in available])

                resp.raise_for_status()
                data = resp.json()

            except httpx.ConnectError:
                last_error = OllamaConnectionError(f"Cannot connect to {self._host}")
                await asyncio.sleep(2**attempt)
                continue
            except httpx.TimeoutException:
                last_error = OllamaTimeoutError(
                    f"Request timed out (attempt {attempt + 1}/{max_retries})"
                )
                await asyncio.sleep(2**attempt)
                continue
            except (json.JSONDecodeError, KeyError) as exc:
                raise OllamaMalformedResponseError(f"Invalid response from Ollama: {exc}") from exc

            text = data.get("response", "").strip()

            # Empty response: retry once with slightly higher temperature
            if not text:
                if attempt < max_retries - 1:
                    current_temp = payload["options"]["temperature"]
                    payload["options"]["temperature"] = min(current_temp + 0.1, 2.0)
                    last_error = OllamaEmptyResponseError("Empty completion received")
                    await asyncio.sleep(1)
                    continue
                raise OllamaEmptyResponseError("Empty completion after all retries")

            return GenerateResult(
                text=text,
                model=data.get("model", model),
                total_duration_ns=data.get("total_duration", 0),
                eval_count=data.get("eval_count", 0),
            )

        # All retries exhausted
        if last_error is not None:
            raise last_error
        raise OllamaConnectionError("All retries exhausted")  # pragma: no cover

    async def generate_stream(
        self,
        prompt: str,
        model: str,
        options: GenerateOptions | None = None,
    ) -> AsyncIterator[str]:
        """Stream generation tokens as an async iterator.

        Yields text tokens as they arrive via NDJSON streaming.

        Args:
            prompt: The input prompt text.
            model: Ollama model tag.
            options: Generation parameters.

        Yields:
            Individual text tokens from the completion.

        Raises:
            OllamaConnectionError: Server unreachable.
            OllamaModelNotFoundError: Model not pulled.
        """
        if options is None:
            options = GenerateOptions()

        payload: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": options.to_dict(),
        }

        try:
            async with self.client.stream("POST", "/api/generate", json=payload) as resp:
                if resp.status_code == 404:
                    available = await self.list_models()
                    raise OllamaModelNotFoundError(model, [m.name for m in available])
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = data.get("response", "")
                    if token:
                        yield token

                    if data.get("done", False):
                        return

        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Cannot connect to {self._host}") from exc
        except httpx.TimeoutException as exc:
            raise OllamaTimeoutError("Stream timed out") from exc

    async def stop_model(self, model: str) -> None:
        """Unload a model from memory via keep_alive=0.

        Args:
            model: Ollama model tag to unload.
        """
        try:
            await self.client.post(
                "/api/generate",
                json={"model": model, "keep_alive": 0},
            )
        except httpx.ConnectError as exc:
            raise OllamaConnectionError(f"Cannot connect to {self._host}") from exc
