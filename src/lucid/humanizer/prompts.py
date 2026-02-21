"""Prompt construction for the LUCID humanization pipeline.

Assembles system prompt + strategy modifier + few-shot examples + input
text into a complete prompt string for Ollama generation.
"""

from __future__ import annotations

import logging
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lucid.humanizer.strategies import Strategy

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are an expert academic editor specializing in improving the naturalness "
    "and readability of technical writing. Rewrite the following paragraph to "
    "sound more natural and human-written while preserving all technical meaning, "
    "factual claims, and logical structure."
)

_RULES = (
    "RULES:\n"
    "- Preserve all terms marked with [TERM_NNN] exactly as-is (never modify)\n"
    "- Preserve all mathematical placeholders [MATH_NNN] exactly as-is\n"
    "- Do NOT add new information or remove existing claims\n"
    "- Do NOT introduce self-contradictions\n"
    "- Do NOT significantly reorder paragraph structure (preserve logical flow)\n"
    "- Maintain the same level of technical precision as the original\n"
    "- Return only the rewritten paragraph; no commentary or explanation"
)

_PROFILE_EXAMPLE_COUNTS: dict[str, int] = {
    "fast": 0,
    "balanced": 2,
    "quality": 3,
}

# ---------------------------------------------------------------------------
# Example pair
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExamplePair:
    """A single input/output few-shot example."""

    input: str
    output: str


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """Construct prompts for humanization using strategy + few-shot examples.

    Loads few-shot examples from TOML files in ``examples_dir``. Each TOML
    file is named ``{domain}.toml`` and contains a ``[[pairs]]`` array.

    Args:
        examples_dir: Directory containing domain TOML files. Defaults to
            ``config/examples/`` relative to the project root.
    """

    def __init__(self, examples_dir: Path | None = None) -> None:
        if examples_dir is None:
            examples_dir = self._find_examples_dir()
        self._examples_dir = examples_dir
        self._cache: dict[str, list[ExamplePair]] = {}
        self._load_all()

    def build(
        self,
        protected_text: str,
        strategy: Strategy,
        domain: str,
        profile: str,
    ) -> str:
        """Assemble a complete prompt for Ollama generation.

        Args:
            protected_text: Prose text with placeholders substituted.
            strategy: Current humanization strategy.
            domain: Content domain (``"stem"``, ``"humanities"``, ``"business"``,
                ``"general"``).
            profile: Quality profile (``"fast"``, ``"balanced"``, ``"quality"``).

        Returns:
            Complete prompt string ready for ``OllamaClient.generate()``.
        """
        parts: list[str] = [_SYSTEM_PROMPT, "", _RULES]

        # Strategy modifier
        modifier = strategy.prompt_modifier
        if modifier:
            parts.append("")
            parts.append(modifier)

        # Few-shot examples
        count = _PROFILE_EXAMPLE_COUNTS.get(profile, 2)
        examples = self._get_examples(domain, count)
        if examples:
            parts.append("")
            parts.append("EXAMPLES:")
            for i, ex in enumerate(examples, 1):
                parts.append(f"\nExample {i}:")
                parts.append(f"INPUT: {ex.input}")
                parts.append(f"OUTPUT: {ex.output}")

        # Input text
        parts.append("")
        parts.append(f"INPUT:\n{protected_text}")
        parts.append("")
        parts.append("OUTPUT:")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_examples(self, domain: str, count: int) -> list[ExamplePair]:
        """Return up to ``count`` examples for domain, falling back to general."""
        if count <= 0:
            return []
        examples = self._cache.get(domain, [])
        if len(examples) < count:
            examples = self._cache.get("general", [])
        return examples[:count]

    def _load_all(self) -> None:
        """Load all TOML example files from the examples directory."""
        if not self._examples_dir.is_dir():
            logger.warning("Examples directory not found: %s", self._examples_dir)
            return

        for path in sorted(self._examples_dir.glob("*.toml")):
            domain = path.stem
            try:
                with open(path, "rb") as f:
                    data = tomllib.load(f)
                pairs = [
                    ExamplePair(input=p["input"], output=p["output"])
                    for p in data.get("pairs", [])
                ]
                self._cache[domain] = pairs
            except Exception:
                logger.warning("Failed to load examples from %s", path, exc_info=True)

    @staticmethod
    def _find_examples_dir() -> Path:
        """Locate config/examples/ by walking up from this file."""
        current = Path(__file__).resolve().parent
        for _ in range(5):
            candidate = current / "config" / "examples"
            if candidate.is_dir():
                return candidate
            current = current.parent
        return Path("config/examples")
