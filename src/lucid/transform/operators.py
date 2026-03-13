"""Transform operators for search-based paraphrase generation."""

from __future__ import annotations

from enum import Enum


class Operator(Enum):
    """Transform operator applied per search iteration."""

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
        """Return the operator-specific prompt modifier string."""
        return self.value


def select_operator(iteration: int, placeholder_count: int = 0) -> Operator:
    """Select an operator via round-robin over the iteration index.

    Becomes more conservative as placeholder pressure rises:
    - >3 placeholders: skip REORDER
    - >5 placeholders: use only STANDARD and RESTRUCTURE
    """
    members = list(Operator)
    if placeholder_count > 5:
        members = [Operator.STANDARD, Operator.RESTRUCTURE]
    elif placeholder_count > 3:
        members = [s for s in members if s != Operator.REORDER]
    return members[iteration % len(members)]
