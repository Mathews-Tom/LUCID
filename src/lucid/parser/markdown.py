"""Markdown document parser using markdown-it-py.

Parses Markdown into line-based chunks. Positions are line indices
(0-indexed, exclusive end) rather than character offsets.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.front_matter import front_matter_plugin

from lucid.parser.chunk import ProseChunk, StructuralChunk

if TYPE_CHECKING:
    from lucid.parser.chunk import Chunk

# Token types that are structural (not prose)
_STRUCTURAL_TOKENS: frozenset[str] = frozenset(
    {
        "fence",
        "code_block",
        "html_block",
        "hr",
        "math_block",
        "math_block_dollarmath",
        "front_matter",
    }
)


class MarkdownDocumentAdapter:
    """Parse Markdown documents into line-based chunks.

    Uses markdown-it-py with dollarmath and front_matter plugins.
    All chunk positions are line indices (0-indexed, exclusive end).
    """

    def __init__(self) -> None:
        self._md = MarkdownIt().enable("table")
        dollarmath_plugin(self._md)
        front_matter_plugin(self._md)
        self._math_counter = 0

    def parse(self, content: str) -> list[Chunk]:
        """Parse Markdown content into prose and structural chunks.

        Args:
            content: Full Markdown document as a string.

        Returns:
            List of Chunk objects sorted by start_pos (line index).
        """
        self._math_counter = 0
        tokens = self._md.parse(content)
        lines = content.split("\n")
        chunks: list[Chunk] = []

        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Skip tokens without line mapping
            if token.map is None:
                i += 1
                continue

            start_line, end_line = token.map

            if token.type in _STRUCTURAL_TOKENS:
                text = "\n".join(lines[start_line:end_line])
                chunks.append(
                    StructuralChunk(
                        text=text,
                        start_pos=start_line,
                        end_pos=end_line,
                        metadata={"position_type": "line"},
                    )
                )
                i += 1

            elif token.type == "heading_open":
                # Heading: collect up to heading_close
                _, math_phs, close_idx = self._collect_block_text(
                    tokens,
                    i,
                    "heading_close",
                )
                text = "\n".join(lines[start_line:end_line])
                chunks.append(
                    ProseChunk(
                        text=text,
                        start_pos=start_line,
                        end_pos=end_line,
                        math_placeholders=math_phs,
                        metadata={"position_type": "line"},
                    )
                )
                i = close_idx + 1

            elif token.type == "paragraph_open":
                # Check if this is an image-only paragraph
                _, math_phs, close_idx = self._collect_block_text(
                    tokens,
                    i,
                    "paragraph_close",
                )
                text = "\n".join(lines[start_line:end_line])

                if self._is_image_only(tokens, i, close_idx):
                    chunks.append(
                        StructuralChunk(
                            text=text,
                            start_pos=start_line,
                            end_pos=end_line,
                            metadata={"position_type": "line"},
                        )
                    )
                else:
                    chunks.append(
                        ProseChunk(
                            text=text,
                            start_pos=start_line,
                            end_pos=end_line,
                            math_placeholders=math_phs,
                            metadata={"position_type": "line"},
                        )
                    )
                i = close_idx + 1

            elif token.type in ("bullet_list_open", "ordered_list_open"):
                close_type = token.type.replace("_open", "_close")
                _, math_phs, close_idx = self._collect_block_text(
                    tokens,
                    i,
                    close_type,
                )
                text = "\n".join(lines[start_line:end_line])
                chunks.append(
                    ProseChunk(
                        text=text,
                        start_pos=start_line,
                        end_pos=end_line,
                        math_placeholders=math_phs,
                        metadata={"position_type": "line"},
                    )
                )
                i = close_idx + 1

            elif token.type == "blockquote_open":
                _, math_phs, close_idx = self._collect_block_text(
                    tokens,
                    i,
                    "blockquote_close",
                )
                text = "\n".join(lines[start_line:end_line])
                chunks.append(
                    ProseChunk(
                        text=text,
                        start_pos=start_line,
                        end_pos=end_line,
                        math_placeholders=math_phs,
                        metadata={"position_type": "line"},
                    )
                )
                i = close_idx + 1

            elif token.type == "table_open":
                close_idx = self._find_close(tokens, i, "table_close")
                text = "\n".join(lines[start_line:end_line])
                chunks.append(
                    StructuralChunk(
                        text=text,
                        start_pos=start_line,
                        end_pos=end_line,
                        metadata={"position_type": "line"},
                    )
                )
                i = close_idx + 1

            else:
                i += 1

        chunks.sort(key=lambda c: c.start_pos)
        return chunks

    def reconstruct(self, original: str, chunks: list[Chunk]) -> str:
        """Reconstruct Markdown document from modified chunks.

        Args:
            original: The original document string.
            chunks: List of chunks (possibly modified).

        Returns:
            Reconstructed document string.
        """
        from lucid.reconstructor.markdown import reconstruct_markdown

        return reconstruct_markdown(original, chunks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _collect_block_text(
        self,
        tokens: list[Any],
        open_idx: int,
        close_type: str,
    ) -> tuple[str, dict[str, str], int]:
        """Collect text content and math placeholders from a block.

        Args:
            tokens: Full token list.
            open_idx: Index of the opening token.
            close_type: Token type of the closing token.

        Returns:
            Tuple of (extracted_text, math_placeholders, close_index).
        """
        math_placeholders: dict[str, str] = {}
        text_parts: list[str] = []
        close_idx = open_idx + 1

        nesting = 1
        open_type = tokens[open_idx].type

        j = open_idx + 1
        while j < len(tokens):
            t = tokens[j]
            if t.type == open_type:
                nesting += 1
            elif t.type == close_type:
                nesting -= 1
                if nesting == 0:
                    close_idx = j
                    break

            if t.type == "inline" and t.children:
                for child in t.children:
                    if child.type == "math_inline" or child.type == "math_inline_double":
                        self._math_counter += 1
                        ph = f"\u27e8MATH_{self._math_counter:03d}\u27e9"
                        math_placeholders[ph] = child.markup + child.content + child.markup
                        text_parts.append(ph)
                    elif child.type == "text":
                        text_parts.append(child.content)
                    elif child.type == "softbreak":
                        text_parts.append("\n")
                    elif child.type == "code_inline":
                        text_parts.append(child.markup + child.content + child.markup)
                    else:
                        text_parts.append(child.content or "")

            j += 1

        return "".join(text_parts), math_placeholders, close_idx

    @staticmethod
    def _is_image_only(tokens: list[Any], open_idx: int, close_idx: int) -> bool:
        """Check if a paragraph block contains only an image.

        Args:
            tokens: Full token list.
            open_idx: Index of paragraph_open.
            close_idx: Index of paragraph_close.

        Returns:
            True if the paragraph contains only an image token.
        """
        for j in range(open_idx + 1, close_idx):
            t = tokens[j]
            if t.type == "inline" and t.children:
                non_empty = [c for c in t.children if c.type != "softbreak"]
                if len(non_empty) == 1 and non_empty[0].type == "image":
                    return True
        return False

    @staticmethod
    def _find_close(tokens: list[Any], open_idx: int, close_type: str) -> int:
        """Find the matching close token.

        Args:
            tokens: Full token list.
            open_idx: Index of the opening token.
            close_type: Token type to find.

        Returns:
            Index of the closing token.
        """
        open_type = tokens[open_idx].type
        nesting = 1
        j = open_idx + 1
        while j < len(tokens):
            if tokens[j].type == open_type:
                nesting += 1
            elif tokens[j].type == close_type:
                nesting -= 1
                if nesting == 0:
                    return j
            j += 1
        return len(tokens) - 1
