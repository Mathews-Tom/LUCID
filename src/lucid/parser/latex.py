"""LaTeX document parser using pylatexenc.

Walks the pylatexenc AST to classify content as prose or structural.
Position tracking uses Python str indices (not byte offsets).
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from pylatexenc.latexwalker import (
    LatexCharsNode,
    LatexCommentNode,
    LatexEnvironmentNode,
    LatexGroupNode,
    LatexMacroNode,
    LatexMathNode,
    LatexWalker,
    get_default_latex_context_db,
)

from lucid.parser.chunk import ProseChunk, StructuralChunk

if TYPE_CHECKING:
    from lucid.parser.chunk import Chunk

# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------

STRUCTURAL_ENVS: frozenset[str] = frozenset(
    {
        "equation",
        "equation*",
        "align",
        "align*",
        "alignat",
        "alignat*",
        "gather",
        "gather*",
        "multline",
        "multline*",
        "flalign",
        "flalign*",
        "eqnarray",
        "eqnarray*",
        "table",
        "table*",
        "tabular",
        "tabular*",
        "longtable",
        "figure",
        "figure*",
        "verbatim",
        "lstlisting",
        "minted",
        "tikzpicture",
        "pgfpicture",
        "algorithmic",
        "algorithm",
        "algorithm2e",
        "array",
        "matrix",
        "pmatrix",
        "bmatrix",
        "vmatrix",
        "Vmatrix",
        "Bmatrix",
        "cases",
        "split",
        "subequations",
        "thebibliography",
    }
)

PROSE_ENVS: frozenset[str] = frozenset(
    {
        "document",
        "abstract",
        "theorem",
        "lemma",
        "proof",
        "corollary",
        "proposition",
        "definition",
        "remark",
        "example",
        "exercise",
        "note",
        "itemize",
        "enumerate",
        "description",
        "quote",
        "quotation",
        "verse",
        "center",
        "flushleft",
        "flushright",
        "minipage",
        "frame",
    }
)

STRUCTURAL_MACROS: frozenset[str] = frozenset(
    {
        "cite",
        "citep",
        "citet",
        "citeauthor",
        "citeyear",
        "nocite",
        "ref",
        "label",
        "eqref",
        "pageref",
        "autoref",
        "nameref",
        "hyperref",
        "includegraphics",
        "input",
        "include",
        "documentclass",
        "usepackage",
        "RequirePackage",
        "newcommand",
        "renewcommand",
        "providecommand",
        "newenvironment",
        "renewenvironment",
        "DeclareMathOperator",
        "bibliography",
        "bibliographystyle",
        "addbibresource",
        "maketitle",
        "tableofcontents",
        "listoffigures",
        "listoftables",
        "newtheorem",
        "theoremstyle",
        "setlength",
        "setcounter",
        "addtocounter",
        "hspace",
        "vspace",
        "hfill",
        "vfill",
        "centering",
        "raggedright",
        "raggedleft",
        "pagestyle",
        "thispagestyle",
    }
)

PROSE_ARG_MACROS: frozenset[str] = frozenset(
    {
        "section",
        "section*",
        "subsection",
        "subsection*",
        "subsubsection",
        "subsubsection*",
        "paragraph",
        "paragraph*",
        "subparagraph",
        "subparagraph*",
        "chapter",
        "chapter*",
        "part",
        "part*",
        "title",
        "author",
        "date",
        "caption",
        "textbf",
        "textit",
        "emph",
        "texttt",
        "textsc",
        "textsf",
        "textrm",
        "underline",
        "mbox",
        "text",
        "footnote",
        "footnotetext",
        "marginpar",
    }
)

_PARA_SPLIT_RE = re.compile(r"\n\n+")


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class LatexDocumentAdapter:
    """Parse LaTeX documents into position-tracked chunks.

    Args:
        custom_prose_envs: Additional environment names to treat as prose containers.
        custom_structural_macros: Additional macro names to treat as structural.
    """

    def __init__(
        self,
        custom_prose_envs: tuple[str, ...] = (),
        custom_structural_macros: tuple[str, ...] = (),
    ) -> None:
        self._prose_envs = PROSE_ENVS | frozenset(custom_prose_envs)
        self._structural_macros = STRUCTURAL_MACROS | frozenset(custom_structural_macros)
        self._math_counter = 0

    def parse(self, content: str) -> list[Chunk]:
        """Parse LaTeX content into prose and structural chunks.

        Args:
            content: Full LaTeX document as a string.

        Returns:
            List of Chunk objects sorted by start_pos.
        """
        self._math_counter = 0
        lw = LatexWalker(content, latex_context=get_default_latex_context_db())
        nodelist, _, _ = lw.get_latex_nodes()

        # Find \begin{document} node
        doc_node: LatexEnvironmentNode | None = None
        doc_index = -1
        for i, node in enumerate(nodelist):
            if isinstance(node, LatexEnvironmentNode) and node.environmentname == "document":
                doc_node = node
                doc_index = i
                break

        chunks: list[Chunk] = []

        if doc_node is None:
            # No \begin{document} — treat entire content as prose
            chunks.extend(self._walk_nodes(nodelist, content, in_prose=True))
        else:
            # Everything before \begin{document} is preamble (structural)
            if doc_index > 0:
                preamble_start = nodelist[0].pos
                preamble_end = doc_node.pos
                if preamble_start is not None and preamble_end is not None:
                    preamble_text = content[preamble_start:preamble_end]
                    if preamble_text.strip():
                        chunks.append(
                            StructuralChunk(
                                text=preamble_text,
                                start_pos=preamble_start,
                                end_pos=preamble_end,
                            )
                        )

            # Walk inside \begin{document}...\end{document}
            body_nodes = doc_node.nodelist or []
            chunks.extend(self._walk_nodes(body_nodes, content, in_prose=True))

            # The \begin{document} and \end{document} tags themselves are structural
            # but are already outside the body nodelist, handled by position gaps

        chunks.sort(key=lambda c: c.start_pos)
        return chunks

    def reconstruct(self, original: str, chunks: list[Chunk]) -> str:
        """Reconstruct document from modified chunks.

        Args:
            original: The original document string.
            chunks: List of chunks (possibly modified).

        Returns:
            Reconstructed document string.
        """
        from lucid.reconstructor.latex import reconstruct_latex

        return reconstruct_latex(original, chunks)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _walk_nodes(
        self,
        nodes: list[Any] | tuple[Any, ...],
        source: str,
        *,
        in_prose: bool,
    ) -> list[Chunk]:
        """Recursively walk nodes, classifying as prose or structural.

        Args:
            nodes: List of pylatexenc AST nodes.
            source: Original document string.
            in_prose: Whether the current context is a prose container.

        Returns:
            List of chunks extracted from these nodes.
        """
        chunks: list[Chunk] = []

        # Collect prose runs (adjacent chars + inline math) when in prose context
        if in_prose:
            runs = self._collect_prose_runs(nodes, source)
            for run in runs:
                chunks.extend(run)
            return chunks

        # Not in prose context — everything is structural
        for node in nodes:
            if node is None or node.pos is None:
                continue
            chunk = self._node_to_structural(node, source)
            if chunk is not None:
                chunks.append(chunk)

        return chunks

    def _collect_prose_runs(
        self,
        nodes: list[Any] | tuple[Any, ...],
        source: str,
    ) -> list[list[Chunk]]:
        """Group nodes into prose runs and structural singletons.

        A prose run is a sequence of LatexCharsNode and inline LatexMathNode
        that form contiguous prose text. Structural nodes break runs.

        Args:
            nodes: Flat list of sibling nodes.
            source: Original document string.

        Returns:
            List of chunk-lists. Each inner list is either a single-element
            structural chunk or a multi-element prose run converted to chunks.
        """
        result: list[list[Chunk]] = []
        current_run: list[Any] = []

        def flush_run() -> None:
            if not current_run:
                return
            prose_chunks = self._build_prose_chunks(current_run, source)
            if prose_chunks:
                result.append(prose_chunks)
            current_run.clear()

        for node in nodes:
            if node is None or node.pos is None:
                continue

            is_inline_math = isinstance(node, LatexMathNode) and node.displaytype == "inline"
            if isinstance(node, LatexCharsNode) or is_inline_math:
                current_run.append(node)

            elif isinstance(node, LatexCommentNode):
                # Comments are structural but don't break prose runs
                flush_run()
                chunk = self._node_to_structural(node, source)
                if chunk is not None:
                    result.append([chunk])

            elif isinstance(node, LatexMacroNode):
                macro_name = node.macroname
                # Starred macro names: pylatexenc strips the *, check with it
                macro_check = macro_name.rstrip("*")

                if macro_check in self._structural_macros:
                    flush_run()
                    chunk = self._node_to_structural(node, source)
                    if chunk is not None:
                        result.append([chunk])

                elif macro_check in PROSE_ARG_MACROS:
                    flush_run()
                    # The macro itself is structural, but its arguments contain prose
                    chunks = self._handle_prose_macro(node, source)
                    if chunks:
                        result.append(chunks)

                elif macro_name == "item":
                    flush_run()
                    # \item itself is structural; prose follows as siblings
                    chunk = self._node_to_structural(node, source)
                    if chunk is not None:
                        result.append([chunk])

                else:
                    # Unknown macro — treat as inline prose element
                    current_run.append(node)

            elif isinstance(node, LatexEnvironmentNode):
                flush_run()
                env_chunks = self._handle_environment(node, source)
                if env_chunks:
                    result.append(env_chunks)

            elif isinstance(node, LatexGroupNode):
                # Bare groups: descend if in prose
                flush_run()
                if node.nodelist:
                    inner = self._walk_nodes(node.nodelist, source, in_prose=True)
                    if inner:
                        result.append(inner)

            elif isinstance(node, LatexMathNode):
                # Display math — structural
                flush_run()
                chunk = self._node_to_structural(node, source)
                if chunk is not None:
                    result.append([chunk])

            else:
                flush_run()
                chunk = self._node_to_structural(node, source)
                if chunk is not None:
                    result.append([chunk])

        flush_run()
        return result

    def _handle_environment(
        self,
        node: LatexEnvironmentNode,
        source: str,
    ) -> list[Chunk]:
        """Handle an environment node — structural or prose container.

        Args:
            node: The environment node.
            source: Original document string.

        Returns:
            List of chunks from this environment.
        """
        env_name = node.environmentname or ""

        if env_name in STRUCTURAL_ENVS:
            # Structural environment — but descend into caption if present
            chunks: list[Chunk] = []

            # Check for \caption inside structural envs (figure, table)
            caption_chunks = self._extract_captions(node, source)

            if caption_chunks:
                # Emit the whole env as structural, plus caption prose separately
                # The structural chunk covers the full env span
                struct = self._node_to_structural(node, source)
                if struct is not None:
                    chunks.append(struct)
                # Caption chunks are emitted as overlapping prose — reconstructor
                # handles this via caption-specific replacement
                # Actually, for clean separation: emit structural parts before/after
                # caption, and caption prose inline. But that's complex.
                # Simpler: just note captions in metadata.
                chunks.extend(caption_chunks)
            else:
                struct = self._node_to_structural(node, source)
                if struct is not None:
                    chunks.append(struct)
            return chunks

        if env_name in self._prose_envs:
            # Prose container — descend into body
            body_nodes = node.nodelist or []
            return self._walk_nodes(body_nodes, source, in_prose=True)

        # Unknown environment — treat as structural
        chunk = self._node_to_structural(node, source)
        return [chunk] if chunk is not None else []

    def _extract_captions(
        self,
        node: LatexEnvironmentNode,
        source: str,
    ) -> list[Chunk]:
        """Extract prose from \\caption macros inside structural environments.

        Args:
            node: A structural environment node.
            source: Original document string.

        Returns:
            List of ProseChunks from caption arguments.
        """
        chunks: list[Chunk] = []
        body_nodes = list(node.nodelist or [])
        for idx, child in enumerate(body_nodes):
            if child is None:
                continue
            if isinstance(child, LatexMacroNode) and child.macroname == "caption":
                # First try: args attached to the macro node
                caption_chunks = self._handle_prose_macro(child, source)
                if caption_chunks:
                    chunks.extend(caption_chunks)
                    continue
                # Fallback: pylatexenc may not associate {group} with \caption.
                # Look for a following LatexGroupNode sibling.
                if idx + 1 < len(body_nodes):
                    sibling = body_nodes[idx + 1]
                    if isinstance(sibling, LatexGroupNode) and sibling.nodelist:
                        inner = self._walk_nodes(
                            sibling.nodelist,
                            source,
                            in_prose=True,
                        )
                        chunks.extend(inner)
        return chunks

    def _handle_prose_macro(
        self,
        node: LatexMacroNode,
        source: str,
    ) -> list[Chunk]:
        """Handle a macro whose arguments contain prose (section, caption, etc.).

        The macro command itself is structural; its arguments contain prose.

        Args:
            node: The macro node.
            source: Original document string.

        Returns:
            List of chunks — structural for the macro, prose for args.
        """
        chunks: list[Chunk] = []

        if node.nodeargd and node.nodeargd.argnlist:
            for arg in node.nodeargd.argnlist:
                if arg is None or arg.pos is None:
                    continue
                if isinstance(arg, LatexGroupNode) and arg.nodelist:
                    inner_chunks = self._walk_nodes(
                        arg.nodelist,
                        source,
                        in_prose=True,
                    )
                    chunks.extend(inner_chunks)

        return chunks

    def _build_prose_chunks(
        self,
        run: list[Any],
        source: str,
    ) -> list[Chunk]:
        """Convert a prose run (chars + inline math) into ProseChunks.

        Splits on paragraph breaks (\\n\\n) within the original source span.
        Positions always reference the original document string.

        Args:
            run: List of LatexCharsNode and inline LatexMathNode nodes.
            source: Original document string.

        Returns:
            List of ProseChunk objects.
        """
        if not run:
            return []

        # Determine the overall span in the original source
        valid_nodes = [n for n in run if n.pos is not None and n.len is not None]
        if not valid_nodes:
            return []

        span_start = valid_nodes[0].pos
        span_end = valid_nodes[-1].pos + valid_nodes[-1].len
        original_text = source[span_start:span_end]

        # Build math placeholder mapping keyed by source position
        math_by_pos: dict[tuple[int, int], str] = {}
        math_originals: dict[str, str] = {}
        for node in valid_nodes:
            if isinstance(node, LatexMathNode):
                self._math_counter += 1
                placeholder = f"\u27e8MATH_{self._math_counter:03d}\u27e9"
                math_source = source[node.pos : node.pos + node.len]
                math_by_pos[(node.pos, node.pos + node.len)] = placeholder
                math_originals[placeholder] = math_source

        # Split original text on paragraph breaks
        paragraphs = self._split_paragraphs(original_text, span_start)

        chunks: list[Chunk] = []
        for p_start, p_end, p_text in paragraphs:
            if not p_text.strip():
                continue

            # Build the placeholder version of this paragraph
            # Replace math expressions within this paragraph's span
            placeholder_text = p_text
            for (m_start, m_end), ph in sorted(
                math_by_pos.items(), key=lambda x: x[0], reverse=True
            ):
                if m_start >= p_start and m_end <= p_end:
                    # This math expression falls within this paragraph
                    rel_start = m_start - p_start
                    rel_end = m_end - p_start
                    placeholder_text = (
                        placeholder_text[:rel_start] + ph + placeholder_text[rel_end:]
                    )

            # Filter math placeholders to only those in this paragraph
            seg_math = {k: v for k, v in math_originals.items() if k in placeholder_text}

            chunks.append(
                ProseChunk(
                    text=placeholder_text,
                    start_pos=p_start,
                    end_pos=p_end,
                    math_placeholders=seg_math,
                )
            )

        return chunks

    @staticmethod
    def _split_paragraphs(
        text: str,
        base_pos: int,
    ) -> list[tuple[int, int, str]]:
        """Split text on \\n\\n boundaries, tracking positions.

        Args:
            text: The text to split.
            base_pos: The start position of the text in the original document.

        Returns:
            List of (start_pos, end_pos, text) tuples.
        """
        result: list[tuple[int, int, str]] = []
        parts = _PARA_SPLIT_RE.split(text)
        offset = 0

        for part in parts:
            # Find actual position of this part in the original text
            idx = text.find(part, offset)
            if idx == -1:
                continue
            start = base_pos + idx
            end = start + len(part)
            result.append((start, end, part))
            offset = idx + len(part)

        return result

    @staticmethod
    def _node_to_structural(node: Any, source: str) -> StructuralChunk | None:
        """Convert any node to a StructuralChunk using its source span.

        Args:
            node: A pylatexenc AST node.
            source: Original document string.

        Returns:
            StructuralChunk or None if node has no position.
        """
        if node.pos is None or node.len is None:
            return None
        text = source[node.pos : node.pos + node.len]
        if not text:
            return None
        return StructuralChunk(
            text=text,
            start_pos=node.pos,
            end_pos=node.pos + node.len,
        )
