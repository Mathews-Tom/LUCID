from __future__ import annotations

from lucid.transform.operators import Operator, select_operator


def test_all_five_strategies_exist() -> None:
    members = list(Operator)
    assert len(members) == 5
    names = {m.name for m in members}
    assert names == {"STANDARD", "RESTRUCTURE", "VOICE_SHIFT", "VOCABULARY", "REORDER"}


def test_standard_has_empty_modifier() -> None:
    assert Operator.STANDARD.prompt_modifier == ""


def test_restructure_modifier() -> None:
    assert Operator.RESTRUCTURE.prompt_modifier == (
        "Significantly vary sentence lengths. Use some very short sentences. "
        "Combine some into longer compound sentences."
    )


def test_voice_shift_modifier() -> None:
    assert Operator.VOICE_SHIFT.prompt_modifier == (
        "Rewrite using more active voice constructions."
        " Vary between formal and conversational register within the same passage."
    )


def test_vocabulary_modifier() -> None:
    assert Operator.VOCABULARY.prompt_modifier == (
        "Use less common synonyms. Avoid overly predictable word choices."
    )


def test_reorder_modifier() -> None:
    assert Operator.REORDER.prompt_modifier == (
        "Reorder the points in this paragraph while maintaining logical flow."
    )


def test_select_operator_round_robin() -> None:
    expected = [
        Operator.STANDARD,
        Operator.RESTRUCTURE,
        Operator.VOICE_SHIFT,
        Operator.VOCABULARY,
        Operator.REORDER,
        Operator.STANDARD,
        Operator.RESTRUCTURE,
        Operator.VOICE_SHIFT,
        Operator.VOCABULARY,
        Operator.REORDER,
    ]
    for i, exp in enumerate(expected):
        assert select_operator(i) == exp, f"iteration {i}: expected {exp}, got {select_operator(i)}"


def test_select_operator_wraps_at_five() -> None:
    assert select_operator(0) == select_operator(5)
    assert select_operator(4) == select_operator(9)


def test_prompt_modifier_returns_value() -> None:
    for operator in Operator:
        assert operator.prompt_modifier == operator.value
