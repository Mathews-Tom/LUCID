"""Plain text document parser.

Splits text into paragraphs on blank-line boundaries.
Positions are Python str indices.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from lucid.parser.chunk import ProseChunk

if TYPE_CHECKING:
    from lucid.parser.chunk import Chunk

_PARA_SPLIT_RE = re.compile(r"\n\n+")


class PlainTextDocumentAdapter:
    """Parse plain text documents into paragraph chunks.

    All chunks are ProseChunks since plain text has no structural markup.
    Positions are character offsets into the original string.
    """

    def parse(self, content: str) -> list[Chunk]:
        """Parse plain text into paragraph chunks.

        Args:
            content: Full plain text document as a string.

        Returns:
            List of ProseChunk objects sorted by start_pos.
        """
        if not content.strip():
            return []

        chunks: list[Chunk] = []
        # Split on paragraph boundaries while tracking positions
        parts = _PARA_SPLIT_RE.split(content)
        offset = 0

        for part in parts:
            # Find the actual position of this part in the content
            idx = content.find(part, offset)
            if idx == -1:
                continue
            if part.strip():
                chunks.append(
                    ProseChunk(
                        text=part,
                        start_pos=idx,
                        end_pos=idx + len(part),
                    )
                )
            offset = idx + len(part)

        return chunks

    def reconstruct(self, original: str, chunks: list[Chunk]) -> str:
        """Reconstruct plain text from modified chunks.

        Uses the same position-based replacement as the LaTeX reconstructor.

        Args:
            original: The original document string.
            chunks: List of chunks (possibly modified).

        Returns:
            Reconstructed document string.
        """
        from lucid.reconstructor.latex import reconstruct_latex

        return reconstruct_latex(original, chunks)
