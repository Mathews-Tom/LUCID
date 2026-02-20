"""Tests for Ollama async HTTP client."""

from __future__ import annotations

import json

import httpx
import pytest

from lucid.humanizer.ollama import (
    GenerateOptions,
    OllamaClient,
    OllamaConnectionError,
    OllamaEmptyResponseError,
    OllamaModelNotFoundError,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _model_list_json(*names: str) -> dict[str, list[dict[str, object]]]:
    """Build a /api/tags response JSON with the given model names."""
    return {"models": [{"name": n, "size": 0, "digest": "", "modified_at": ""} for n in names]}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_transport() -> httpx.MockTransport:
    """Create a mock transport for httpx."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Default handler that returns 200 for health checks."""
        if request.url.path == "/":
            return httpx.Response(200, text="Ollama is running")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def _make_client(transport: httpx.MockTransport) -> OllamaClient:
    """Create an OllamaClient with injected transport."""
    client = OllamaClient.__new__(OllamaClient)
    client._host = "http://localhost:11434"
    client._timeout = 60.0
    client._client = httpx.AsyncClient(
        base_url="http://localhost:11434",
        transport=transport,
    )
    return client


# ---------------------------------------------------------------------------
# Health check tests
# ---------------------------------------------------------------------------


class TestHealthCheck:
    """OllamaClient.health_check tests."""

    async def test_healthy_server(self) -> None:
        """health_check returns True for 200 response."""
        transport = httpx.MockTransport(lambda _: httpx.Response(200, text="OK"))
        client = _make_client(transport)
        assert await client.health_check() is True
        await client._client.aclose()  # type: ignore[union-attr]

    async def test_unreachable_server(self) -> None:
        """health_check returns False on connection error."""

        def raise_connect(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        transport = httpx.MockTransport(raise_connect)
        client = _make_client(transport)
        assert await client.health_check() is False
        await client._client.aclose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# List models tests
# ---------------------------------------------------------------------------


class TestListModels:
    """OllamaClient.list_models tests."""

    async def test_list_models_success(self) -> None:
        """list_models returns parsed ModelInfo list."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/tags":
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {
                                "name": "phi3:3.8b",
                                "size": 2400000000,
                                "digest": "abc123",
                                "modified_at": "2025-01-01T00:00:00Z",
                            },
                            {
                                "name": "qwen2.5:7b",
                                "size": 4500000000,
                                "digest": "def456",
                                "modified_at": "2025-01-02T00:00:00Z",
                            },
                        ]
                    },
                )
            return httpx.Response(404)

        client = _make_client(httpx.MockTransport(handler))
        models = await client.list_models()
        assert len(models) == 2
        assert models[0].name == "phi3:3.8b"
        assert models[1].name == "qwen2.5:7b"
        assert models[0].size == 2400000000
        await client._client.aclose()  # type: ignore[union-attr]

    async def test_list_models_connection_error(self) -> None:
        """list_models raises OllamaConnectionError on connect failure."""

        def raise_connect(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused")

        client = _make_client(httpx.MockTransport(raise_connect))
        with pytest.raises(OllamaConnectionError):
            await client.list_models()
        await client._client.aclose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Model availability tests
# ---------------------------------------------------------------------------


class TestModelAvailability:
    """OllamaClient.is_model_available tests."""

    async def test_exact_match(self) -> None:
        """Exact model name match returns True."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json=_model_list_json("phi3:3.8b"),
            )

        client = _make_client(httpx.MockTransport(handler))
        assert await client.is_model_available("phi3:3.8b") is True
        await client._client.aclose()  # type: ignore[union-attr]

    async def test_latest_suffix_fallback(self) -> None:
        """Model without tag matches name:latest."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_model_list_json("phi3:latest"))

        client = _make_client(httpx.MockTransport(handler))
        assert await client.is_model_available("phi3") is True
        await client._client.aclose()  # type: ignore[union-attr]

    async def test_model_not_found(self) -> None:
        """Missing model returns False."""

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"models": []})

        client = _make_client(httpx.MockTransport(handler))
        assert await client.is_model_available("nonexistent:7b") is False
        await client._client.aclose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# Generate tests
# ---------------------------------------------------------------------------


class TestGenerate:
    """OllamaClient.generate tests."""

    async def test_successful_generate(self) -> None:
        """Successful generate returns text and metadata."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/generate":
                return httpx.Response(
                    200,
                    json={
                        "response": "Hello! How can I help?",
                        "model": "phi3:3.8b",
                        "total_duration": 1500000000,
                        "eval_count": 10,
                        "done": True,
                    },
                )
            return httpx.Response(404)

        client = _make_client(httpx.MockTransport(handler))
        result = await client.generate("Hello", model="phi3:3.8b")
        assert result.text == "Hello! How can I help?"
        assert result.model == "phi3:3.8b"
        assert result.total_duration_ns == 1500000000
        assert result.eval_count == 10
        await client._client.aclose()  # type: ignore[union-attr]

    async def test_model_not_found_raises(self) -> None:
        """404 response raises OllamaModelNotFoundError."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/generate":
                return httpx.Response(404, json={"error": "model not found"})
            if request.url.path == "/api/tags":
                return httpx.Response(200, json=_model_list_json("phi3:3.8b"))
            return httpx.Response(404)

        client = _make_client(httpx.MockTransport(handler))
        with pytest.raises(OllamaModelNotFoundError) as exc_info:
            await client.generate("test", model="nonexistent:7b")
        assert "nonexistent:7b" in str(exc_info.value)
        assert exc_info.value.available_models == ["phi3:3.8b"]
        await client._client.aclose()  # type: ignore[union-attr]

    async def test_empty_response_retries_with_higher_temp(self) -> None:
        """Empty response triggers retry with temperature + 0.1."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            if request.url.path == "/api/generate":
                call_count += 1
                body = json.loads(request.content)
                if call_count == 1:
                    # First call: empty response
                    return httpx.Response(
                        200,
                        json={"response": "", "model": "phi3:3.8b", "done": True},
                    )
                # Second call: should have higher temperature
                assert body["options"]["temperature"] > 0.6
                return httpx.Response(
                    200,
                    json={
                        "response": "Now I respond!",
                        "model": "phi3:3.8b",
                        "total_duration": 100,
                        "eval_count": 5,
                        "done": True,
                    },
                )
            return httpx.Response(404)

        client = _make_client(httpx.MockTransport(handler))
        result = await client.generate("test", model="phi3:3.8b", max_retries=3)
        assert result.text == "Now I respond!"
        assert call_count == 2
        await client._client.aclose()  # type: ignore[union-attr]

    async def test_all_empty_responses_raises(self) -> None:
        """All-empty responses exhaust retries and raise."""

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path == "/api/generate":
                return httpx.Response(
                    200,
                    json={"response": "", "model": "phi3:3.8b", "done": True},
                )
            return httpx.Response(404)

        client = _make_client(httpx.MockTransport(handler))
        with pytest.raises(OllamaEmptyResponseError, match="all retries"):
            await client.generate("test", model="phi3:3.8b", max_retries=2)
        await client._client.aclose()  # type: ignore[union-attr]

    async def test_connection_error_retries(self) -> None:
        """Connection errors trigger exponential backoff retries."""
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise httpx.ConnectError("refused")
            return httpx.Response(
                200,
                json={
                    "response": "recovered",
                    "model": "phi3:3.8b",
                    "total_duration": 100,
                    "eval_count": 1,
                    "done": True,
                },
            )

        client = _make_client(httpx.MockTransport(handler))
        result = await client.generate("test", model="phi3:3.8b", max_retries=3)
        assert result.text == "recovered"
        assert call_count == 3
        await client._client.aclose()  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# GenerateOptions tests
# ---------------------------------------------------------------------------


class TestGenerateOptions:
    """GenerateOptions defaults and serialization."""

    def test_defaults(self) -> None:
        """Default options match design spec values."""
        opts = GenerateOptions()
        assert opts.temperature == 0.6
        assert opts.top_p == 0.9
        assert opts.top_k == 40
        assert opts.num_ctx == 4096

    def test_to_dict(self) -> None:
        """to_dict produces complete options dict."""
        opts = GenerateOptions(temperature=0.8, num_predict=256)
        d = opts.to_dict()
        assert d["temperature"] == 0.8
        assert d["num_predict"] == 256
        assert "top_p" in d


# ---------------------------------------------------------------------------
# Context manager tests
# ---------------------------------------------------------------------------


class TestContextManager:
    """Async context manager protocol."""

    async def test_context_manager_creates_client(self) -> None:
        """Entering context creates httpx client."""
        async with OllamaClient(host="http://localhost:11434") as client:
            assert client._client is not None
        assert client._client is None

    def test_client_property_outside_context_raises(self) -> None:
        """Accessing .client outside context manager raises RuntimeError."""
        client = OllamaClient()
        with pytest.raises(RuntimeError, match="async context manager"):
            _ = client.client
