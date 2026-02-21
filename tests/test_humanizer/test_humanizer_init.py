"""Tests for the LUCIDHumanizer orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock

import httpx
import pytest

from lucid.config import (
    HumanizerConfig,
    OllamaConfig,
    OllamaModelsConfig,
    TemperatureProfileConfig,
    TermProtectionConfig,
)
from lucid.humanizer import LUCIDHumanizer
from lucid.models.results import DetectionResult, ParaphraseResult
from lucid.parser.chunk import ProseChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str, domain: str = "stem") -> ProseChunk:
    return ProseChunk(
        text=text,
        start_pos=0,
        end_pos=len(text),
        domain_hint=domain,
    )


def _make_detection(chunk_id: str, score: float = 0.85) -> DetectionResult:
    return DetectionResult(
        chunk_id=chunk_id,
        ensemble_score=score,
        classification="ai_generated",
    )


def _make_configs(
    adversarial_iterations: int = 1,
) -> tuple[HumanizerConfig, OllamaConfig]:
    hconfig = HumanizerConfig(
        adversarial_iterations=adversarial_iterations,
        temperature=TemperatureProfileConfig(fast=0.7, balanced=0.6, quality=0.5),
        term_protection=TermProtectionConfig(
            use_ner=False, protect_citations=False, protect_numbers=False
        ),
    )
    oconfig = OllamaConfig(
        host="http://localhost:11434",
        timeout_seconds=30,
        models=OllamaModelsConfig(fast="phi3:3.8b", balanced="qwen2.5:7b", quality="llama3.1:8b"),
    )
    return hconfig, oconfig


def _mock_transport(response_text: str) -> httpx.MockTransport:
    """Create a mock transport that returns the given text for generate requests."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/generate":
            return httpx.Response(
                200,
                json={
                    "response": response_text,
                    "model": "phi3:3.8b",
                    "total_duration": 1000000,
                    "eval_count": 10,
                    "done": True,
                },
            )
        if request.url.path == "/":
            return httpx.Response(200, text="Ollama is running")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_satisfies_humanizer_protocol(self) -> None:
        from lucid.core.protocols import Humanizer

        hconfig, oconfig = _make_configs()
        detector = MagicMock()
        humanizer = LUCIDHumanizer(hconfig, oconfig, detector, profile="fast")
        assert isinstance(humanizer, Humanizer)

    def test_has_humanize_method(self) -> None:
        hconfig, oconfig = _make_configs()
        detector = MagicMock()
        humanizer = LUCIDHumanizer(hconfig, oconfig, detector)
        assert callable(getattr(humanizer, "humanize", None))


# ---------------------------------------------------------------------------
# Single-pass tests
# ---------------------------------------------------------------------------


class TestSinglePass:
    def test_single_pass_returns_paraphrase_result(self) -> None:
        """Single-pass humanize returns a valid ParaphraseResult."""
        hconfig, oconfig = _make_configs(adversarial_iterations=1)
        detector = MagicMock()

        chunk = _make_chunk("The model performs well on benchmarks.")
        detection = _make_detection(chunk.id)

        # The LLM will return the same text (no placeholders to preserve)
        humanized_text = "The model does well on benchmarks."
        humanizer = LUCIDHumanizer(hconfig, oconfig, detector, profile="fast")

        transport = _mock_transport(humanized_text)
        import asyncio

        async def run_with_mock() -> ParaphraseResult:
            from lucid.humanizer.ollama import OllamaClient

            client = OllamaClient.__new__(OllamaClient)
            client._host = "http://localhost:11434"
            client._timeout = 30.0
            client._client = httpx.AsyncClient(
                base_url="http://localhost:11434",
                transport=transport,
            )
            try:
                result = await humanizer._single_pass(chunk, detection, client)
                return result
            finally:
                await client._client.aclose()

        result = asyncio.run(run_with_mock())

        assert isinstance(result, ParaphraseResult)
        assert result.chunk_id == chunk.id
        assert result.original_text == chunk.text
        assert result.humanized_text == humanized_text
        assert result.iteration_count == 1
        assert result.strategy_used == "STANDARD"

    def test_single_pass_raises_on_missing_placeholder(self) -> None:
        """If LLM drops a placeholder, ValueError is raised."""
        hconfig, oconfig = _make_configs(adversarial_iterations=1)
        # Enable citations so we get placeholders
        hconfig_with_cite = HumanizerConfig(
            adversarial_iterations=1,
            temperature=hconfig.temperature,
            term_protection=TermProtectionConfig(
                use_ner=False, protect_citations=True, protect_numbers=False
            ),
        )
        detector = MagicMock()

        chunk = _make_chunk("See [Smith, 2024] for details.")
        detection = _make_detection(chunk.id)

        # LLM response that drops the placeholder
        transport = _mock_transport("See Smith 2024 for details.")
        humanizer = LUCIDHumanizer(hconfig_with_cite, oconfig, detector, profile="fast")

        import asyncio

        async def run_with_mock() -> ParaphraseResult:
            from lucid.humanizer.ollama import OllamaClient

            client = OllamaClient.__new__(OllamaClient)
            client._host = "http://localhost:11434"
            client._timeout = 30.0
            client._client = httpx.AsyncClient(
                base_url="http://localhost:11434",
                transport=transport,
            )
            try:
                return await humanizer._single_pass(chunk, detection, client)
            finally:
                await client._client.aclose()

        with pytest.raises(ValueError, match="dropped placeholders"):
            asyncio.run(run_with_mock())


# ---------------------------------------------------------------------------
# Model and temperature selection
# ---------------------------------------------------------------------------


class TestModelSelection:
    def test_fast_profile_uses_fast_model(self) -> None:
        hconfig, oconfig = _make_configs()
        detector = MagicMock()
        humanizer = LUCIDHumanizer(hconfig, oconfig, detector, profile="fast")
        assert humanizer._model == "phi3:3.8b"

    def test_balanced_profile_uses_balanced_model(self) -> None:
        hconfig, oconfig = _make_configs()
        detector = MagicMock()
        humanizer = LUCIDHumanizer(hconfig, oconfig, detector, profile="balanced")
        assert humanizer._model == "qwen2.5:7b"

    def test_quality_profile_uses_quality_model(self) -> None:
        hconfig, oconfig = _make_configs()
        detector = MagicMock()
        humanizer = LUCIDHumanizer(hconfig, oconfig, detector, profile="quality")
        assert humanizer._model == "llama3.1:8b"

    def test_fast_temperature(self) -> None:
        hconfig, oconfig = _make_configs()
        detector = MagicMock()
        humanizer = LUCIDHumanizer(hconfig, oconfig, detector, profile="fast")
        assert humanizer._temperature == 0.7

    def test_quality_temperature(self) -> None:
        hconfig, oconfig = _make_configs()
        detector = MagicMock()
        humanizer = LUCIDHumanizer(hconfig, oconfig, detector, profile="quality")
        assert humanizer._temperature == 0.5
