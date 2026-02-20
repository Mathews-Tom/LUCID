"""Markdown document reconstruction via line-based replacement.

Replaces modified prose chunks at their exact line ranges while
preserving all structural content.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lucid.parser.chunk import ChunkType, ProseChunk
from lucid.reconstructor.latex import restore_placeholders

if TYPE_CHECKING:
    from lucid.parser.chunk import Chunk


def reconstruct_markdown(original: str, chunks: list[Chunk]) -> str:
    """Reconstruct a Markdown document from (possibly modified) chunks.

    Operates on line indices. Modified chunks have their lines replaced
    in the original document.

    Args:
        original: The original document string.
        chunks: List of chunks from parsing (possibly with modifications).

    Returns:
        Reconstructed document string. Identical to original if no
        chunks were modified.
    """
    lines = original.split("\n")
    replacements: list[tuple[int, int, list[str]]] = []

    for chunk in chunks:
        if chunk.chunk_type != ChunkType.PROSE:
            continue
        if not isinstance(chunk, ProseChunk):
            continue

        # Determine if this chunk was modified
        original_text = "\n".join(lines[chunk.start_pos : chunk.end_pos])
        modified_text = chunk.metadata.get("humanized_text", "")

        if not modified_text and chunk.text == original_text:
            continue

        new_text = modified_text if modified_text else chunk.text

        # Restore placeholders
        all_placeholders = {**chunk.math_placeholders, **chunk.term_placeholders}
        if all_placeholders:
            new_text = restore_placeholders(new_text, all_placeholders)

        if new_text == original_text:
            continue

        new_lines = new_text.split("\n")
        replacements.append((chunk.start_pos, chunk.end_pos, new_lines))

    if not replacements:
        return original

    # Sort by start_pos descending — replace from end to preserve line offsets
    replacements.sort(key=lambda r: r[0], reverse=True)

    for start, end, new_lines in replacements:
        lines[start:end] = new_lines

    return "\n".join(lines)
