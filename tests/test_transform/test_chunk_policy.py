from __future__ import annotations

from lucid.parser.chunk import ProseChunk
from lucid.transform.chunk_policy import (
    is_equation_like_chunk,
    is_latex_fragment,
    is_math_heavy_chunk,
    is_title_like_chunk,
    is_too_short_to_transform,
    skip_transform_reason,
)


def _chunk(text: str) -> ProseChunk:
    return ProseChunk(text=text, start_pos=0, end_pos=len(text))


# ---------------------------------------------------------------------------
# Title-like detection
# ---------------------------------------------------------------------------


def test_markdown_heading_is_title_like() -> None:
    assert is_title_like_chunk(
        _chunk("### 2.1.1 Foundational Retrieval-Augmented Generation")
    )


def test_bold_heading_is_title_like() -> None:
    assert is_title_like_chunk(
        _chunk("**The Language Model Revolution: Ponte and Croft (1998)**")
    )


def test_numbered_section_heading_is_title_like() -> None:
    assert is_title_like_chunk(
        _chunk("2.1.1 Foundational Retrieval-Augmented Generation")
    )


def test_lettered_section_heading_is_title_like() -> None:
    assert is_title_like_chunk(_chunk("A.3 Appendix Details"))


def test_normal_sentence_is_not_title_like() -> None:
    assert not is_title_like_chunk(_chunk("This section explains the core retrieval pipeline."))


# ---------------------------------------------------------------------------
# Equation-like detection
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Math-heavy detection
# ---------------------------------------------------------------------------


def test_single_math_placeholder_line_is_math_heavy() -> None:
    chunk = _chunk("[MATH_001]")
    chunk.math_placeholders = {"[MATH_001]": "$x$"}
    assert is_math_heavy_chunk(chunk)


def test_multiple_math_placeholders_in_short_line_are_math_heavy() -> None:
    text = "where [MATH_002] is the total number of documents and [MATH_003] counts the term."
    chunk = _chunk(text)
    chunk.math_placeholders = {
        "[MATH_002]": "$N$",
        "[MATH_003]": "$df_t$",
    }
    assert is_math_heavy_chunk(chunk)


def test_plain_sentence_with_no_math_is_not_math_heavy() -> None:
    assert not is_math_heavy_chunk(_chunk("This section discusses retrieval trade-offs."))


# ---------------------------------------------------------------------------
# Too-short detection
# ---------------------------------------------------------------------------


def test_empty_chunk_is_too_short() -> None:
    assert is_too_short_to_transform(_chunk(""), min_length=50)


def test_short_chunk_is_too_short() -> None:
    assert is_too_short_to_transform(_chunk("precision"), min_length=50)


def test_whitespace_only_is_too_short() -> None:
    assert is_too_short_to_transform(_chunk("   "), min_length=50)


def test_adequate_chunk_is_not_too_short() -> None:
    text = "The retrieval pipeline processes documents through several stages."
    assert not is_too_short_to_transform(_chunk(text), min_length=50)


# ---------------------------------------------------------------------------
# LaTeX fragment detection
# ---------------------------------------------------------------------------


def test_latex_linebreak_is_fragment() -> None:
    assert is_latex_fragment(_chunk(" \\\\"))


def test_bare_citation_key_is_fragment() -> None:
    assert is_latex_fragment(_chunk("Lewis2020"))
    assert is_latex_fragment(_chunk("Deerwester1990"))


def test_single_word_is_fragment() -> None:
    assert is_latex_fragment(_chunk("precision"))
    assert is_latex_fragment(_chunk("recall"))


def test_conjunction_is_fragment() -> None:
    assert is_latex_fragment(_chunk("and"))


def test_normal_sentence_is_not_fragment() -> None:
    assert not is_latex_fragment(
        _chunk("The retrieval pipeline processes documents through several stages.")
    )


def test_multi_word_phrase_is_not_fragment() -> None:
    assert not is_latex_fragment(_chunk("Karen Sparck Jones"))


# ---------------------------------------------------------------------------
# skip_transform_reason integration
# ---------------------------------------------------------------------------


def test_skip_transform_reason_returns_too_short() -> None:
    reason = skip_transform_reason(
        _chunk("recall"),
        skip_title_like=True,
        skip_equation_like=True,
        skip_math_heavy=True,
        min_prose_length=50,
    )
    assert reason == "too_short"


def test_skip_transform_reason_returns_latex_fragment() -> None:
    reason = skip_transform_reason(
        _chunk(" \\\\"),
        skip_title_like=True,
        skip_equation_like=True,
        skip_math_heavy=True,
        min_prose_length=2,  # below stripped length so too_short doesn't fire first
    )
    assert reason == "latex_fragment"


def test_skip_transform_reason_prioritizes_title_like() -> None:
    reason = skip_transform_reason(
        _chunk("### Retrieval-Augmented Generation for Document Processing"),
        skip_title_like=True,
        skip_equation_like=True,
        skip_math_heavy=True,
    )
    assert reason == "title_like"


def test_skip_transform_reason_returns_equation_like() -> None:
    reason = skip_transform_reason(
        _chunk("P(q | M_d) = product over t in q of P(t | M_d) where P(t) = count(t)"),
        skip_title_like=True,
        skip_equation_like=True,
        skip_math_heavy=True,
    )
    assert reason == "equation_like"


def test_skip_transform_reason_returns_math_heavy() -> None:
    chunk = _chunk(
        "[MATH_001] and [MATH_002] are the relevant and non-relevant document sets."
    )
    chunk.math_placeholders = {
        "[MATH_001]": "$D_R$",
        "[MATH_002]": "$D_{NR}$",
    }
    reason = skip_transform_reason(
        chunk,
        skip_title_like=True,
        skip_equation_like=True,
        skip_math_heavy=True,
    )
    assert reason == "math_heavy"


def test_skip_transform_reason_returns_none_for_valid_prose() -> None:
    reason = skip_transform_reason(
        _chunk("The retrieval pipeline processes documents through several stages of analysis."),
        skip_title_like=True,
        skip_equation_like=True,
        skip_math_heavy=True,
    )
    assert reason is None


def test_numbered_heading_caught_by_skip_transform() -> None:
    reason = skip_transform_reason(
        _chunk("2.1.1 Foundational Retrieval-Augmented Generation Overview"),
        skip_title_like=True,
        skip_equation_like=True,
        skip_math_heavy=True,
    )
    assert reason == "title_like"
