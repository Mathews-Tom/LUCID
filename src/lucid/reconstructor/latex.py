"""LaTeX document reconstruction via position-based replacement.

Replaces modified prose chunks at their exact string positions while
preserving all structural content byte-for-byte.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lucid.parser.chunk import ChunkType, ProseChunk

if TYPE_CHECKING:
    from lucid.parser.chunk import Chunk


def restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
    """Replace placeholder tokens with their original content.

    Handles both math (⟨MATH_NNN⟩) and term (⟨TERM_NNN⟩) placeholders.

    Args:
        text: Text containing placeholder tokens.
        placeholders: Mapping of placeholder → original content.

    Returns:
        Text with all placeholders restored.
    """
    result = text
    for placeholder, original in placeholders.items():
        result = result.replace(placeholder, original)
    return result


def reconstruct_latex(original: str, chunks: list[Chunk]) -> str:
    """Reconstruct a LaTeX document from (possibly modified) chunks.

    Identifies modified ProseChunks, restores their placeholders, and
    applies replacements in reverse position order to maintain offset
    validity.

    Args:
        original: The original document string.
        chunks: List of chunks from parsing (possibly with modifications).

    Returns:
        Reconstructed document string. Identical to original if no
        chunks were modified.
    """
    replacements: list[tuple[int, int, str]] = []

    for chunk in chunks:
        if chunk.chunk_type != ChunkType.PROSE:
            continue
        if not isinstance(chunk, ProseChunk):
            continue

        # Determine if this chunk was modified
        original_slice = original[chunk.start_pos : chunk.end_pos]
        modified_text = chunk.metadata.get("humanized_text", "")

        if not modified_text and chunk.text == original_slice:
            continue

        # Use humanized_text from metadata if present, otherwise chunk.text
        new_text = modified_text if modified_text else chunk.text

        # Restore placeholders
        all_placeholders = {**chunk.math_placeholders, **chunk.term_placeholders}
        if all_placeholders:
            new_text = restore_placeholders(new_text, all_placeholders)

        if new_text == original_slice:
            continue

        replacements.append((chunk.start_pos, chunk.end_pos, new_text))

    if not replacements:
        return original

    # Sort by start_pos descending — replace from end to preserve offsets
    replacements.sort(key=lambda r: r[0], reverse=True)

    result = original
    for start, end, new_text in replacements:
        result = result[:start] + new_text + result[end:]

    return result
