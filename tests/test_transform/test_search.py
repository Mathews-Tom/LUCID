"""Tests for the transformation search loop."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from lucid.config import (
    TemperatureProfileConfig,
    TermProtectionConfig,
    TransformConfig,
)
from lucid.models.results import DetectionResult, TransformResult
from lucid.parser.chunk import ProseChunk
from lucid.transform.ollama import (
    GenerateResult,
    OllamaClient,
    OllamaConnectionError,
)
from lucid.transform.operators import Operator
from lucid.transform.prompts import PromptBuilder
from lucid.transform.search import transformation_search
from lucid.transform.term_protect import TermProtector

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chunk(text: str = "The model achieves good results.") -> ProseChunk:
    return ProseChunk(text=text, start_pos=0, end_pos=len(text), domain_hint="stem")


def _make_detection(chunk_id: str, score: float = 0.85) -> DetectionResult:
    return DetectionResult(
        chunk_id=chunk_id, ensemble_score=score, classification="ai_generated"
    )


def _make_config(iterations: int = 3, target: float = 0.25) -> TransformConfig:
    return TransformConfig(
        search_iterations=iterations,
        search_target_score=target,
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
    detector.detect_fast = mock_detect
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

        # Responses must be similar enough to pass the semantic gate
        client = _make_mock_client([
            "The model achieves strong results.",
            "The model attains good results.",
        ])
        detector = _make_mock_detector([0.60, 0.20])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            transformation_search(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config, "phi3:3.8b", "fast",
            )
        )

        assert isinstance(result, TransformResult)
        assert result.final_detection_score == 0.20
        assert result.iteration_count == 2  # Early exit at iteration 2

    def test_returns_best_when_target_not_met(self) -> None:
        """Returns best candidate even when target score is not reached."""
        chunk = _make_chunk()
        detection = _make_detection(chunk.id)
        config = _make_config(iterations=3, target=0.10)

        # Scores: 0.70, 0.40, 0.50 — best is 0.40
        client = _make_mock_client([
            "The model achieves strong results.",
            "The model attains good results.",
            "The model obtains good results.",
        ])
        detector = _make_mock_detector([0.70, 0.40, 0.50])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            transformation_search(
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
                text="The model achieves good results.",
                model=model, total_duration_ns=100, eval_count=1,
            )

        client.generate = capture_generate
        detector = _make_mock_detector([0.80, 0.70, 0.60, 0.50, 0.40])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        asyncio.run(
            transformation_search(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config, "phi3:3.8b", "fast",
            )
        )

        # 5 iterations → 5 prompts
        assert len(prompts_seen) == 5

        # Iteration 0 = STANDARD (no modifier), iteration 1 = RESTRUCTURE, etc.
        assert Operator.RESTRUCTURE.prompt_modifier in prompts_seen[1]
        assert Operator.VOICE_SHIFT.prompt_modifier in prompts_seen[2]

    def test_placeholder_validation_failure_skips_iteration(self) -> None:
        """Iterations where LLM drops placeholders are skipped."""
        text = "See [Smith, 2024] for details."
        chunk = _make_chunk(text)
        detection = _make_detection(chunk.id)
        config = _make_config(iterations=3, target=0.25)
        config_with_cite = TransformConfig(
            search_iterations=3,
            search_target_score=0.25,
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
            f"See {next(iter(protected.term_placeholders.keys()))} for reference.",  # Valid
            "Another bad response.",  # Missing → skip
        ]
        client = _make_mock_client(responses)
        detector = _make_mock_detector([0.20])  # Only called once (iteration 2)

        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            transformation_search(
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
                text="The model achieves good results.",
                model=model, total_duration_ns=100, eval_count=1,
            )

        client.generate = fail_on_second
        detector = _make_mock_detector([0.50])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            transformation_search(
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
                transformation_search(
                    chunk, detection, client, detector, term_protector,
                    prompt_builder, config, "phi3:3.8b", "fast",
                )
            )

    def test_all_iterations_invalid_uses_unprotected_fallback(self) -> None:
        """Unsafe unprotected fallback remains available when explicitly enabled."""
        text = "See [Smith, 2024] for details."
        chunk = _make_chunk(text)
        detection = _make_detection(chunk.id)
        config_with_cite = TransformConfig(
            search_iterations=2,
            search_target_score=0.25,
            fallback_policy="unsafe_unprotected",
            temperature=TemperatureProfileConfig(),
            term_protection=TermProtectionConfig(
                use_ner=False, protect_citations=True, protect_numbers=False
            ),
        )

        # All responses drop the placeholder; fallback uses raw text (no placeholders)
        client = _make_mock_client(["Bad response 1.", "Bad response 2.", "Fallback output."])
        detector = _make_mock_detector([0.50])  # Score for fallback re-scoring

        term_protector = TermProtector(config_with_cite.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            transformation_search(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config_with_cite, "phi3:3.8b", "fast",
            )
        )

        assert result.transformed_text == "Fallback output."
        assert result.iteration_count == config_with_cite.search_iterations + 1
        assert result.fallback_mode == "unsafe_unprotected"

    def test_all_iterations_invalid_mark_failed_raises_runtime_error(self) -> None:
        """Default fallback policy fails closed instead of dropping protection."""
        text = "See [Smith, 2024] for details."
        chunk = _make_chunk(text)
        detection = _make_detection(chunk.id)
        config_with_cite = TransformConfig(
            search_iterations=2,
            search_target_score=0.25,
            temperature=TemperatureProfileConfig(),
            term_protection=TermProtectionConfig(
                use_ner=False, protect_citations=True, protect_numbers=False
            ),
        )

        client = _make_mock_client(["Bad response 1.", "Bad response 2."])
        detector = MagicMock()

        term_protector = TermProtector(config_with_cite.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        with pytest.raises(RuntimeError, match="fallback_policy=mark_failed"):
            asyncio.run(
                transformation_search(
                    chunk, detection, client, detector, term_protector,
                    prompt_builder, config_with_cite, "phi3:3.8b", "fast",
                )
            )

    def test_all_iterations_invalid_keep_original_returns_identity(self) -> None:
        """Identity fallback preserves the source text without pretending to rewrite it."""
        text = "See [Smith, 2024] for details."
        chunk = _make_chunk(text)
        detection = _make_detection(chunk.id)
        config_with_cite = TransformConfig(
            search_iterations=2,
            search_target_score=0.25,
            fallback_policy="keep_original",
            temperature=TemperatureProfileConfig(),
            term_protection=TermProtectionConfig(
                use_ner=False, protect_citations=True, protect_numbers=False
            ),
        )

        client = _make_mock_client(["Bad response 1.", "Bad response 2."])
        detector = MagicMock()

        term_protector = TermProtector(config_with_cite.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            transformation_search(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config_with_cite, "phi3:3.8b", "fast",
            )
        )

        assert result.transformed_text == chunk.text
        assert result.operator_used == "identity_keep_original"
        assert result.fallback_mode == "keep_original"
        assert result.final_detection_score == detection.ensemble_score

    def test_all_iterations_and_unprotected_fallback_fail_raises_runtime_error(self) -> None:
        """RuntimeError raised when explicit unsafe fallback also fails."""
        text = "See [Smith, 2024] for details."
        chunk = _make_chunk(text)
        detection = _make_detection(chunk.id)
        config_with_cite = TransformConfig(
            search_iterations=2,
            search_target_score=0.25,
            fallback_policy="unsafe_unprotected",
            temperature=TemperatureProfileConfig(),
            term_protection=TermProtectionConfig(
                use_ner=False, protect_citations=True, protect_numbers=False
            ),
        )

        client = MagicMock(spec=OllamaClient)
        call_count = 0

        async def fail_on_fallback(
            prompt: str, model: str, options: object = None, max_retries: int = 3
        ) -> GenerateResult:
            nonlocal call_count
            call_count += 1
            if call_count > 4:
                raise OllamaConnectionError("unreachable")
            return GenerateResult(
                text="Bad response.", model=model,
                total_duration_ns=1000000, eval_count=10,
            )

        client.generate = fail_on_fallback
        detector = MagicMock()

        term_protector = TermProtector(config_with_cite.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        with pytest.raises(RuntimeError, match="unprotected fallback also failed"):
            asyncio.run(
                transformation_search(
                    chunk, detection, client, detector, term_protector,
                    prompt_builder, config_with_cite, "phi3:3.8b", "fast",
                )
            )


# ---------------------------------------------------------------------------
# Semantic gate tests
# ---------------------------------------------------------------------------


class TestSemanticGate:
    def test_gate_rejects_low_similarity_picks_better_candidate(self) -> None:
        """Search skips aggressively rewritten text and picks a faithful candidate."""
        original = "The model achieves good results on benchmarks."
        chunk = _make_chunk(original)
        detection = _make_detection(chunk.id)
        config = TransformConfig(
            search_iterations=2,
            search_target_score=0.10,
            semantic_gate_threshold=0.40,
            temperature=TemperatureProfileConfig(fast=0.7, balanced=0.6, quality=0.5),
            term_protection=TermProtectionConfig(
                use_ner=False, protect_citations=False, protect_numbers=False,
            ),
        )

        # Iteration 0: totally different text (low similarity, low detection score)
        # Iteration 1: minor paraphrase (high similarity, higher detection score)
        client = _make_mock_client([
            "Quantum computing reshapes modern cryptography.",  # sim < 0.55
            "The model gets good results on benchmarks.",  # sim > 0.55
        ])
        # Only iteration 1 passes the gate, so only one detect_fast call
        detector = _make_mock_detector([0.40])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            transformation_search(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config, "phi3:3.8b", "fast",
            )
        )

        # Should pick iteration 1 (faithful) despite iteration 0 having lower detection score
        assert "gets good results" in result.transformed_text
        assert result.semantic_similarity is not None
        assert result.semantic_similarity >= 0.55

    def test_gate_allows_good_candidates(self) -> None:
        """Candidates above the threshold pass through normally."""
        original = "The model achieves good results."
        chunk = _make_chunk(original)
        detection = _make_detection(chunk.id)
        config = _make_config(iterations=2, target=0.10)

        # Both responses are minor paraphrases — both should pass the gate
        client = _make_mock_client([
            "The model attains good results.",
            "The model reaches good results.",
        ])
        detector = _make_mock_detector([0.50, 0.30])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            transformation_search(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config, "phi3:3.8b", "fast",
            )
        )

        # Best should be the lower score (0.30)
        assert result.final_detection_score == 0.30

    def test_higher_similarity_beats_slightly_lower_detection_score(self) -> None:
        """Search favors candidates more likely to survive semantic evaluation."""
        original = "The model achieves good results on benchmarks."
        chunk = _make_chunk(original)
        detection = _make_detection(chunk.id)
        config = TransformConfig(
            search_iterations=2,
            search_target_score=0.10,
            semantic_gate_threshold=0.55,
            temperature=TemperatureProfileConfig(fast=0.7, balanced=0.6, quality=0.5),
            term_protection=TermProtectionConfig(
                use_ner=False, protect_citations=False, protect_numbers=False,
            ),
        )

        client = _make_mock_client([
            "The model gets good benchmark results.",
            "Benchmarks show the model achieves good results.",
        ])
        detector = _make_mock_detector([0.29, 0.31])

        term_protector = TermProtector(config.term_protection)
        prompt_builder = PromptBuilder(examples_dir=None)

        result = asyncio.run(
            transformation_search(
                chunk, detection, client, detector, term_protector,
                prompt_builder, config, "phi3:3.8b", "fast",
            )
        )

        assert result.transformed_text == "Benchmarks show the model achieves good results."
        assert result.final_detection_score == 0.31
