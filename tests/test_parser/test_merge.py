"""Tests for detection-only prose chunk merging."""

from __future__ import annotations

from lucid.detector.statistical import MIN_WORDS_THRESHOLD
from lucid.parser.chunk import ProseChunk, StructuralChunk
from lucid.parser.merge import merge_prose_for_detection


def _make_prose(word_count: int, start: int = 0, label: str = "word") -> ProseChunk:
    """Create a ProseChunk with exactly word_count words."""
    text = " ".join(f"{label}{i}" for i in range(word_count))
    return ProseChunk(text=text, start_pos=start, end_pos=start + len(text))


def _make_structural(start: int = 0) -> StructuralChunk:
    """Create a StructuralChunk."""
    return StructuralChunk(
        text="\\section{Title}", start_pos=start, end_pos=start + 15
    )


class TestMergeProseForDetection:
    """Tests for merge_prose_for_detection."""

    def test_single_long_chunk(self) -> None:
        """One ProseChunk with 60 words forms a single group without merging."""
        chunk = _make_prose(60)
        groups = merge_prose_for_detection([chunk], min_words=MIN_WORDS_THRESHOLD)

        assert len(groups) == 1
        merged_text, constituents = groups[0]
        assert constituents == [chunk]
        assert merged_text == chunk.text

    def test_three_short_chunks_merged(self) -> None:
        """Three consecutive short ProseChunks merge into one group."""
        c1 = _make_prose(15, start=0, label="a")
        c2 = _make_prose(15, start=100, label="b")
        c3 = _make_prose(15, start=200, label="c")

        groups = merge_prose_for_detection([c1, c2, c3], min_words=MIN_WORDS_THRESHOLD)

        assert len(groups) == 1
        merged_text, constituents = groups[0]
        assert constituents == [c1, c2, c3]
        assert merged_text == f"{c1.text} {c2.text} {c3.text}"

    def test_structural_breaks_merge(self) -> None:
        """A StructuralChunk between short prose creates two separate groups."""
        c1 = _make_prose(15, start=0, label="a")
        s = _make_structural(start=100)
        c2 = _make_prose(15, start=200, label="b")

        groups = merge_prose_for_detection([c1, s, c2], min_words=MIN_WORDS_THRESHOLD)

        assert len(groups) == 2
        assert groups[0][1] == [c1]
        assert groups[1][1] == [c2]

    def test_mixed_short_and_long(self) -> None:
        """Long prose gets own group; subsequent short chunks merge separately."""
        long_chunk = _make_prose(60, start=0, label="long")
        s1 = _make_prose(15, start=500, label="s1")
        s2 = _make_prose(15, start=600, label="s2")

        groups = merge_prose_for_detection(
            [long_chunk, s1, s2], min_words=MIN_WORDS_THRESHOLD
        )

        assert len(groups) == 2
        # Long chunk stands alone
        assert groups[0][1] == [long_chunk]
        assert groups[0][0] == long_chunk.text
        # Short chunks merged
        assert groups[1][1] == [s1, s2]

    def test_empty_input(self) -> None:
        """Empty chunk list produces empty output."""
        groups = merge_prose_for_detection([], min_words=MIN_WORDS_THRESHOLD)
        assert groups == []

    def test_single_short_chunk_emitted(self) -> None:
        """A single short chunk is still emitted as a group (not lost)."""
        chunk = _make_prose(10)
        groups = merge_prose_for_detection([chunk], min_words=MIN_WORDS_THRESHOLD)

        assert len(groups) == 1
        merged_text, constituents = groups[0]
        assert constituents == [chunk]
        assert merged_text == chunk.text

    def test_flush_at_threshold(self) -> None:
        """Accumulator flushes when accumulated words reach min_words."""
        c1 = _make_prose(25, start=0, label="a")
        c2 = _make_prose(25, start=200, label="b")
        c3 = _make_prose(10, start=400, label="c")

        groups = merge_prose_for_detection([c1, c2, c3], min_words=MIN_WORDS_THRESHOLD)

        # c1+c2 = 50 words -> flush. c3 = 10 words -> separate group
        assert len(groups) == 2
        assert groups[0][1] == [c1, c2]
        assert groups[1][1] == [c3]
