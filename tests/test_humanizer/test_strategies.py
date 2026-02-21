from __future__ import annotations

from lucid.humanizer.strategies import Strategy, select_strategy


def test_all_five_strategies_exist() -> None:
    members = list(Strategy)
    assert len(members) == 5
    names = {m.name for m in members}
    assert names == {"STANDARD", "RESTRUCTURE", "VOICE_SHIFT", "VOCABULARY", "REORDER"}


def test_standard_has_empty_modifier() -> None:
    assert Strategy.STANDARD.prompt_modifier == ""


def test_restructure_modifier() -> None:
    assert Strategy.RESTRUCTURE.prompt_modifier == (
        "Significantly vary sentence lengths. Use some very short sentences. "
        "Combine some into longer compound sentences."
    )


def test_voice_shift_modifier() -> None:
    assert Strategy.VOICE_SHIFT.prompt_modifier == (
        "Rewrite using more active voice. Add hedging language like 'it seems' or 'arguably'."
    )


def test_vocabulary_modifier() -> None:
    assert Strategy.VOCABULARY.prompt_modifier == (
        "Use less common synonyms. Avoid overly predictable word choices."
    )


def test_reorder_modifier() -> None:
    assert Strategy.REORDER.prompt_modifier == (
        "Reorder the points in this paragraph while maintaining logical flow."
    )


def test_select_strategy_round_robin() -> None:
    expected = [
        Strategy.STANDARD,
        Strategy.RESTRUCTURE,
        Strategy.VOICE_SHIFT,
        Strategy.VOCABULARY,
        Strategy.REORDER,
        Strategy.STANDARD,
        Strategy.RESTRUCTURE,
        Strategy.VOICE_SHIFT,
        Strategy.VOCABULARY,
        Strategy.REORDER,
    ]
    for i, exp in enumerate(expected):
        assert select_strategy(i) == exp, f"iteration {i}: expected {exp}, got {select_strategy(i)}"


def test_select_strategy_wraps_at_five() -> None:
    assert select_strategy(0) == select_strategy(5)
    assert select_strategy(4) == select_strategy(9)


def test_prompt_modifier_returns_value() -> None:
    for strategy in Strategy:
        assert strategy.prompt_modifier == strategy.value
