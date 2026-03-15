from __future__ import annotations

from lucid.parser.chunk import ProseChunk
from lucid.transform.chunk_policy import (
    is_equation_like_chunk,
    is_title_like_chunk,
    skip_transform_reason,
)


def _chunk(text: str) -> ProseChunk:
    return ProseChunk(text=text, start_pos=0, end_pos=len(text))


def test_markdown_heading_is_title_like() -> None:
    assert is_title_like_chunk(
        _chunk("### 2.1.1 Foundational Retrieval-Augmented Generation")
    )


def test_bold_heading_is_title_like() -> None:
    assert is_title_like_chunk(
        _chunk("**The Language Model Revolution: Ponte and Croft (1998)**")
    )


def test_normal_sentence_is_not_title_like() -> None:
    assert not is_title_like_chunk(_chunk("This section explains the core retrieval pipeline."))


def test_symbolic_formula_is_equation_like() -> None:
    assert is_equation_like_chunk(_chunk("P(q | M_d) = product over t in q of P(t | M_d)"))


def test_rocchio_update_is_equation_like() -> None:
    text = (
        "q_new = alpha * q_orig + beta * (1/|D_R|) * sum(d in D_R) - "
        "gamma * (1/|D_NR|) * sum(d in D_NR)"
    )
    assert is_equation_like_chunk(_chunk(text))


def test_plain_sentence_is_not_equation_like() -> None:
    assert not is_equation_like_chunk(
        _chunk("The method updates the query using relevant documents.")
    )


def test_skip_transform_reason_prioritizes_title_like() -> None:
    reason = skip_transform_reason(
        _chunk("### Retrieval-Augmented Generation"),
        skip_title_like=True,
        skip_equation_like=True,
    )
    assert reason == "title_like"


def test_skip_transform_reason_returns_equation_like() -> None:
    reason = skip_transform_reason(
        _chunk("P(q | M_d) = product over t in q of P(t | M_d)"),
        skip_title_like=True,
        skip_equation_like=True,
    )
    assert reason == "equation_like"
