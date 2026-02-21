"""Tests for the adversarial refinement loop."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from lucid.config import (
    HumanizerConfig,
    TemperatureProfileConfig,
    TermProtectionConfig,
)
from lucid.humanizer.adversarial import adversarial_humanize
from lucid.humanizer.ollama import (
    GenerateResult,
    OllamaClient,
    OllamaConnectionError,
)
from lucid.humanizer.prompts import PromptBuilder
from lucid.humanizer.strategies import Strategy
from lucid.humanizer.term_protect import TermProtector
from lucid.models.results import DetectionResult, ParaphraseResult
from lucid.parser.chunk import ProseChunk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "The model achieves good results.") -> ProseChunk:
    return ProseChunk(text=text, start_pos=0, end_pos=len(text), domain_hint="stem")


def _make_detection(chunk_id: str, score: float = 0.85) -> DetectionResult:
    return DetectionResult(
        chunk_id=chunk_id, ensemble_score=score, classification="ai_generated"
    )


def _make_config(iterations: int = 3, target: float = 0.25) -> HumanizerConfig:
    return HumanizerConfig(
        adversarial_iterations=iterations,
        adversarial_target_score=target,
        temperature=TemperatureProfileConfig(fast=0.7, balanced=0.6, quality=0.5),
        term_protection=TermProtectionConfig(
            use_ner=False, protect_citations=False, protect_numbers=False
        ),
    )


def _make_mock_client(responses: list[str]) -> MagicMock:
    """Create a mock OllamaClient that returns specified texts in sequence."""
    client = MagicMock(spec=OllamaClient)
    call_count = 0

    async def mock_generate(
        prompt: str, model: str, options: object = None, max_retries: int = 3
    ) -> GenerateResult:
        nonlocal call_count
        idx = min(call_count, len(responses) - 1)
        text = responses[idx]
        call_count += 1
        return GenerateResult(
            text=text, model=model, total_duration_ns=1000000, eval_count=10
        )

    client.generate = mock_generate
    return client


def _make_mock_detector(scores: list[float]) -> MagicMock:
    """Create a mock detector that returns scores in sequence."""
    detector = MagicMock()
    call_count = 0

    def mock_detect(chunk: ProseChunk) -> DetectionResult:
        nonlocal call_count
        idx = min(call_count, len(scores) - 1)
        score = scores[idx]
        call_count += 1
        return DetectionResult(
            chunk_id=chunk.id,
            ensemble_score=score,
            classification="ai_generated" if score >= 0.65 else "human",
        )

    detector.detect = mock_detect
    return detector


# ---------------------------------------------------------------------------
# Core loop tests
# ---------------------------------------------------------------------------


class TestAdversarialLoop:
    def test_early_exit_on_low_score(self) -> None:
        """Loop exits early when score drops below target."""
        chunk = _make_chunk()
        detection = _make_detection(chunk.id)
        config = _make_config(iterations=5, target=0.25)

        # First iteration scores 0.60, second 0.20 (below target)
        client = _make_mock_client(["Rewritten text v1.", "Rewritten text v2."])
        detector = _make_mock_detector([0.60, 0.20])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            adversarial_humanize(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config, "phi3:3.8b", "fast",
            )
        )

        assert isinstance(result, ParaphraseResult)
        assert result.final_detection_score == 0.20
        assert result.iteration_count == 2  # Early exit at iteration 2

    def test_returns_best_when_target_not_met(self) -> None:
        """Returns best candidate even when target score is not reached."""
        chunk = _make_chunk()
        detection = _make_detection(chunk.id)
        config = _make_config(iterations=3, target=0.10)

        # Scores: 0.70, 0.40, 0.50 — best is 0.40
        client = _make_mock_client(["v1", "v2", "v3"])
        detector = _make_mock_detector([0.70, 0.40, 0.50])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            adversarial_humanize(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config, "phi3:3.8b", "fast",
            )
        )

        assert result.final_detection_score == 0.40
        assert result.iteration_count == 2  # Best was iteration 2

    def test_strategy_rotation_across_iterations(self) -> None:
        """Each iteration uses a different strategy (round-robin)."""
        chunk = _make_chunk()
        detection = _make_detection(chunk.id)
        config = _make_config(iterations=5, target=0.01)  # Never reached

        prompts_seen: list[str] = []

        client = MagicMock(spec=OllamaClient)

        async def capture_generate(
            prompt: str, model: str, options: object = None, max_retries: int = 3
        ) -> GenerateResult:
            prompts_seen.append(prompt)
            return GenerateResult(
                text="rewritten", model=model, total_duration_ns=100, eval_count=1
            )

        client.generate = capture_generate
        detector = _make_mock_detector([0.80, 0.70, 0.60, 0.50, 0.40])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        asyncio.run(
            adversarial_humanize(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config, "phi3:3.8b", "fast",
            )
        )

        # 5 iterations → 5 prompts
        assert len(prompts_seen) == 5

        # Iteration 0 = STANDARD (no modifier), iteration 1 = RESTRUCTURE, etc.
        assert Strategy.RESTRUCTURE.prompt_modifier in prompts_seen[1]
        assert Strategy.VOICE_SHIFT.prompt_modifier in prompts_seen[2]

    def test_placeholder_validation_failure_skips_iteration(self) -> None:
        """Iterations where LLM drops placeholders are skipped."""
        text = "See [Smith, 2024] for details."
        chunk = _make_chunk(text)
        detection = _make_detection(chunk.id)
        config = _make_config(iterations=3, target=0.25)
        config_with_cite = HumanizerConfig(
            adversarial_iterations=3,
            adversarial_target_score=0.25,
            temperature=config.temperature,
            term_protection=TermProtectionConfig(
                use_ner=False, protect_citations=True, protect_numbers=False
            ),
        )

        term_protector = TermProtector(config_with_cite.term_protection)
        protected = term_protector.protect(chunk)

        # First response drops placeholder, second preserves it
        responses = [
            "See Smith 2024 for details.",  # Missing placeholder → skip
            f"See {list(protected.term_placeholders.keys())[0]} for reference.",  # Valid
            "Another bad response.",  # Missing → skip
        ]
        client = _make_mock_client(responses)
        detector = _make_mock_detector([0.20])  # Only called once (iteration 2)

        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            adversarial_humanize(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config_with_cite, "phi3:3.8b", "fast",
            )
        )

        # Only the valid iteration counts
        assert result.final_detection_score == 0.20

    def test_ollama_error_returns_best_if_available(self) -> None:
        """OllamaError on later iteration returns best so far."""
        chunk = _make_chunk()
        detection = _make_detection(chunk.id)
        config = _make_config(iterations=3, target=0.10)

        call_count = 0

        client = MagicMock(spec=OllamaClient)

        async def fail_on_second(
            prompt: str, model: str, options: object = None, max_retries: int = 3
        ) -> GenerateResult:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise OllamaConnectionError("connection lost")
            return GenerateResult(
                text="rewritten", model=model, total_duration_ns=100, eval_count=1
            )

        client.generate = fail_on_second
        detector = _make_mock_detector([0.50])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            adversarial_humanize(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config, "phi3:3.8b", "fast",
            )
        )

        assert result.final_detection_score == 0.50

    def test_ollama_error_on_first_iteration_raises(self) -> None:
        """OllamaError on first iteration with no best candidate raises."""
        chunk = _make_chunk()
        detection = _make_detection(chunk.id)
        config = _make_config(iterations=3)

        client = MagicMock(spec=OllamaClient)

        async def always_fail(
            prompt: str, model: str, options: object = None, max_retries: int = 3
        ) -> GenerateResult:
            raise OllamaConnectionError("unreachable")

        client.generate = always_fail
        detector = MagicMock()

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        with pytest.raises(OllamaConnectionError):
            asyncio.run(
                adversarial_humanize(
                    chunk, detection, client, detector, term_protector,
                    prompt_builder, config, "phi3:3.8b", "fast",
                )
            )

    def test_all_iterations_invalid_raises_runtime_error(self) -> None:
        """If every iteration fails validation, RuntimeError is raised."""
        text = "See [Smith, 2024] for details."
        chunk = _make_chunk(text)
        detection = _make_detection(chunk.id)
        config_with_cite = HumanizerConfig(
            adversarial_iterations=2,
            adversarial_target_score=0.25,
            temperature=TemperatureProfileConfig(),
            term_protection=TermProtectionConfig(
                use_ner=False, protect_citations=True, protect_numbers=False
            ),
        )

        # Both responses drop the placeholder
        client = _make_mock_client(["Bad response 1.", "Bad response 2."])
        detector = MagicMock()

        term_protector = TermProtector(config_with_cite.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        with pytest.raises(RuntimeError, match="No valid paraphrase"):
            asyncio.run(
                adversarial_humanize(
                    chunk, detection, client, detector, term_protector,
                    prompt_builder, config_with_cite, "phi3:3.8b", "fast",
                )
            )
