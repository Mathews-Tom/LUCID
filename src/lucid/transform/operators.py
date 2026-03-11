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
        "Rewrite using more active voice constructions."
        " Vary between formal and conversational register within the same passage."
    )
    VOCABULARY = "Use less common synonyms. Avoid overly predictable word choices."
    REORDER = "Reorder the points in this paragraph while maintaining logical flow."

    @property
    def prompt_modifier(self) -> str:
        """Return the strategy-specific prompt modifier string."""
        return self.value


def select_strategy(iteration: int, placeholder_count: int = 0) -> Strategy:
    """Select a strategy via round-robin over the iteration index.

    Skips REORDER when placeholder_count > 3 to reduce placeholder drops.
    """
    members = list(Strategy)
    if placeholder_count > 3:
        members = [s for s in members if s != Strategy.REORDER]
    return members[iteration % len(members)]
