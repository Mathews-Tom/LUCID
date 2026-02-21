"""Humanization strategies for adversarial paraphrase generation."""

from __future__ import annotations

from enum import Enum


class Strategy(Enum):
    """Humanization strategy applied per adversarial iteration."""

    STANDARD = ""
    RESTRUCTURE = (
        "Significantly vary sentence lengths. Use some very short sentences."
        " Combine some into longer compound sentences."
    )
    VOICE_SHIFT = (
        "Rewrite using more active voice."
        " Add hedging language like 'it seems' or 'arguably'."
    )
    VOCABULARY = "Use less common synonyms. Avoid overly predictable word choices."
    REORDER = "Reorder the points in this paragraph while maintaining logical flow."

    @property
    def prompt_modifier(self) -> str:
        """Return the strategy-specific prompt modifier string."""
        return self.value


def select_strategy(iteration: int) -> Strategy:
    """Select a strategy via round-robin over the iteration index."""
    members = list(Strategy)
    return members[iteration % len(members)]
