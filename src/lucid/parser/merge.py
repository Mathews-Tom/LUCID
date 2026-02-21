"""Detection-only chunk merging for fragmented prose.

Groups consecutive short ProseChunks so statistical detectors
have sufficient text for reliable scoring.
"""

from __future__ import annotations

from lucid.parser.chunk import Chunk, ProseChunk


def merge_prose_for_detection(
    chunks: list[Chunk],
    min_words: int = 50,
) -> list[tuple[str, list[ProseChunk]]]:
    """Group consecutive short ProseChunks for joint detection.

    Returns list of (merged_text, constituent_chunks) tuples.
    Chunks already >= min_words are returned as single-element groups.
    Non-prose chunks break merge groups.

    Args:
        chunks: Full chunk list (prose + structural, in document order).
        min_words: Minimum word count for a chunk to stand alone.

    Returns:
        List of (merged_text, [constituent_chunks]) tuples.
    """
    groups: list[tuple[str, list[ProseChunk]]] = []
    accumulator: list[ProseChunk] = []
    accumulated_words: int = 0

    def _flush() -> None:
        nonlocal accumulated_words
        if accumulator:
            merged_text = " ".join(c.text for c in accumulator)
            groups.append((merged_text, list(accumulator)))
            accumulator.clear()
            accumulated_words = 0

    for chunk in chunks:
        if not isinstance(chunk, ProseChunk):
            _flush()
            continue

        word_count = len(chunk.text.split())

        if word_count >= min_words:
            _flush()
            groups.append((chunk.text, [chunk]))
            continue

        accumulator.append(chunk)
        accumulated_words += word_count

        if accumulated_words >= min_words:
            _flush()

    _flush()
    return groups
