"""Tests for the PromptBuilder in lucid.transform.prompts."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from lucid.transform.prompts import ExamplePair, PromptBuilder
from lucid.transform.operators import Operator

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def builder(tmp_path: Path) -> PromptBuilder:
    """PromptBuilder with a temporary examples directory."""
    examples_dir = tmp_path / "examples"
    examples_dir.mkdir()

    # stem.toml — 4 pairs
    (examples_dir / "stem.toml").write_text(
        'domain = "stem"\n'
        "[[pairs]]\n"
        'input = "AI stem input 1."\n'
        'output = "Human stem output 1."\n'
        "[[pairs]]\n"
        'input = "AI stem input 2."\n'
        'output = "Human stem output 2."\n'
        "[[pairs]]\n"
        'input = "AI stem input 3."\n'
        'output = "Human stem output 3."\n'
        "[[pairs]]\n"
        'input = "AI stem input 4."\n'
        'output = "Human stem output 4."\n'
    )

    # general.toml — 4 pairs (fallback)
    (examples_dir / "general.toml").write_text(
        'domain = "general"\n'
        "[[pairs]]\n"
        'input = "AI general input 1."\n'
        'output = "Human general output 1."\n'
        "[[pairs]]\n"
        'input = "AI general input 2."\n'
        'output = "Human general output 2."\n'
        "[[pairs]]\n"
        'input = "AI general input 3."\n'
        'output = "Human general output 3."\n'
        "[[pairs]]\n"
        'input = "AI general input 4."\n'
        'output = "Human general output 4."\n'
    )

    return PromptBuilder(examples_dir=examples_dir)


@pytest.fixture
def builder_no_examples(tmp_path: Path) -> PromptBuilder:
    """PromptBuilder with an empty examples directory."""
    examples_dir = tmp_path / "empty_examples"
    examples_dir.mkdir()
    return PromptBuilder(examples_dir=examples_dir)


# ---------------------------------------------------------------------------
# Prompt structure tests
# ---------------------------------------------------------------------------


class TestPromptStructure:
    def test_contains_system_prompt(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Test text.", Operator.STANDARD, "stem", "balanced")
        assert "expert academic editor" in prompt

    def test_contains_rules(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Test text.", Operator.STANDARD, "stem", "balanced")
        assert "RULES:" in prompt
        assert "[TERM_NNN]" in prompt
        assert "[MATH_NNN]" in prompt

    def test_contains_input_and_output_markers(self, builder: PromptBuilder) -> None:
        prompt = builder.build("My protected text.", Operator.STANDARD, "stem", "balanced")
        assert "INPUT:\nMy protected text." in prompt
        assert prompt.rstrip().endswith("OUTPUT:")

    def test_operator_modifier_included(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.RESTRUCTURE, "stem", "balanced")
        assert Operator.RESTRUCTURE.prompt_modifier in prompt

    def test_standard_operator_no_extra_modifier(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.STANDARD, "stem", "balanced")
        # No operator modifier section for STANDARD (empty string)
        assert "ADDITIONAL INSTRUCTION" not in prompt


# ---------------------------------------------------------------------------
# Few-shot example tests
# ---------------------------------------------------------------------------


class TestFewShotExamples:
    def test_balanced_profile_includes_two_examples(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.STANDARD, "stem", "balanced")
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "Example 3:" not in prompt

    def test_quality_profile_includes_three_examples(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.STANDARD, "stem", "quality")
        assert "Example 1:" in prompt
        assert "Example 2:" in prompt
        assert "Example 3:" in prompt

    def test_fast_profile_no_examples(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.STANDARD, "stem", "fast")
        assert "EXAMPLES:" not in prompt
        assert "Example 1:" not in prompt

    def test_unknown_domain_falls_back_to_general(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.STANDARD, "unknown_domain", "balanced")
        assert "general input" in prompt

    def test_missing_example_file_no_examples(
        self, builder_no_examples: PromptBuilder
    ) -> None:
        prompt = builder_no_examples.build("Text.", Operator.STANDARD, "stem", "balanced")
        assert "EXAMPLES:" not in prompt

    def test_stem_examples_loaded(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.STANDARD, "stem", "balanced")
        assert "stem input 1" in prompt
        assert "stem output 1" in prompt


# ---------------------------------------------------------------------------
# Operator modifier integration
# ---------------------------------------------------------------------------


class TestOperatorModifiers:
    def test_all_strategies_produce_valid_prompts(self, builder: PromptBuilder) -> None:
        for operator in Operator:
            prompt = builder.build("Text.", operator, "stem", "balanced")
            assert "INPUT:" in prompt
            assert "OUTPUT:" in prompt
            if operator.prompt_modifier:
                assert operator.prompt_modifier in prompt

    def test_voice_shift_modifier_in_prompt(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.VOICE_SHIFT, "stem", "balanced")
        assert "active voice" in prompt

    def test_vocabulary_modifier_in_prompt(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.VOCABULARY, "stem", "balanced")
        assert "less common synonyms" in prompt

    def test_reorder_modifier_in_prompt(self, builder: PromptBuilder) -> None:
        prompt = builder.build("Text.", Operator.REORDER, "stem", "balanced")
        assert "Reorder the points" in prompt


# ---------------------------------------------------------------------------
# ExamplePair dataclass
# ---------------------------------------------------------------------------


class TestExamplePair:
    def test_frozen(self) -> None:
        pair = ExamplePair(input="in", output="out")
        with pytest.raises(AttributeError):
            pair.input = "changed"  # type: ignore[misc]

    def test_fields(self) -> None:
        pair = ExamplePair(input="hello", output="world")
        assert pair.input == "hello"
        assert pair.output == "world"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_text_produces_valid_prompt(self, builder: PromptBuilder) -> None:
        prompt = builder.build("", Operator.STANDARD, "stem", "balanced")
        assert "INPUT:\n" in prompt
        assert "OUTPUT:" in prompt

    def test_nonexistent_examples_dir(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "no_such_dir"
        pb = PromptBuilder(examples_dir=nonexistent)
        prompt = pb.build("Text.", Operator.STANDARD, "stem", "balanced")
        # No crash, no examples
        assert "EXAMPLES:" not in prompt
