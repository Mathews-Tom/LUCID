"""Output formatting for LUCID pipeline results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from lucid import __version__

if TYPE_CHECKING:
    from lucid.config import LUCIDConfig
    from lucid.models.results import DocumentResult


class OutputFormatter:
    """Format and write pipeline results in multiple output formats."""

    def format_json(self, result: DocumentResult, config: LUCIDConfig) -> str:
        """Serialize pipeline results as a JSON report.

        Args:
            result: Completed pipeline results.
            config: Configuration used for the run.

        Returns:
            JSON string with schema version, metadata, summary, and per-chunk detail.
        """
        detection_map = {d.chunk_id: d.to_dict() for d in result.detections}
        paraphrase_map = {p.chunk_id: p.to_dict() for p in result.paraphrases}
        evaluation_map = {e.chunk_id: e.to_dict() for e in result.evaluations}

        chunks_detail: list[dict[str, Any]] = []
        for chunk in result.chunks:
            entry: dict[str, Any] = {
                "id": chunk.id,
                "type": chunk.chunk_type.value,
                "text_preview": chunk.text[:100],
            }
            if chunk.id in detection_map:
                entry["detection"] = detection_map[chunk.id]
            if chunk.id in paraphrase_map:
                entry["paraphrase"] = paraphrase_map[chunk.id]
            if chunk.id in evaluation_map:
                entry["evaluation"] = evaluation_map[chunk.id]
            chunks_detail.append(entry)

        report: dict[str, Any] = {
            "lucid_version": __version__,
            "input_path": result.input_path,
            "format": result.format,
            "profile": config.general.profile,
            "timestamp": datetime.now(UTC).isoformat(),
            "summary": result.summary_stats,
            "compilation_valid": result.compilation_valid,
            "output_path": result.output_path,
            "chunks": chunks_detail,
        }

        return json.dumps(report, indent=2)

    def format_text(self, result: DocumentResult) -> str:
        """Format pipeline results as a human-readable text report.

        Args:
            result: Completed pipeline results.

        Returns:
            Multi-line text report string.
        """
        lines: list[str] = []
        lines.append("LUCID Pipeline Report")
        lines.append("=" * 50)
        lines.append(f"Input:  {result.input_path}")
        lines.append(f"Format: {result.format}")
        if result.output_path:
            lines.append(f"Output: {result.output_path}")
        lines.append("")

        stats = result.summary_stats
        lines.append("Summary")
        lines.append("-" * 30)
        lines.append(f"  Total chunks:     {stats.get('total_chunks', 0)}")
        lines.append(f"  Prose chunks:     {stats.get('prose_chunks', 0)}")
        lines.append(f"  AI-detected:      {stats.get('ai_detected', 0)}")
        lines.append(f"  Humanized:        {stats.get('humanized', 0)}")
        lines.append(f"  Eval passed:      {stats.get('eval_passed', 0)}")
        lines.append(f"  Eval failed:      {stats.get('eval_failed', 0)}")
        lines.append(f"  Failed:           {stats.get('failed', 0)}")
        lines.append("")

        if result.detections:
            lines.append("Detection Results")
            lines.append("-" * 30)
            for det in result.detections:
                lines.append(
                    f"  [{det.chunk_id[:8]}] "
                    f"score={det.ensemble_score:.3f} "
                    f"class={det.classification}"
                )
            lines.append("")

        if result.evaluations:
            lines.append("Evaluation Results")
            lines.append("-" * 30)
            for ev in result.evaluations:
                status = "PASS" if ev.passed else "FAIL"
                reason = f" ({ev.rejection_reason})" if ev.rejection_reason else ""
                lines.append(f"  [{ev.chunk_id[:8]}] {status}{reason}")
            lines.append("")

        return "\n".join(lines)

    def format_annotated(
        self, result: DocumentResult, original_content: str
    ) -> str:
        """Insert LUCID annotation comments into the original document.

        Args:
            result: Completed pipeline results.
            original_content: The original document text.

        Returns:
            Annotated document string with LUCID comments before each processed chunk.
        """
        detection_map = {d.chunk_id: d for d in result.detections}

        is_latex = result.format == "latex"
        comment_prefix = "%% LUCID:" if is_latex else "<!-- LUCID:"
        comment_suffix = "" if is_latex else " -->"

        annotations: list[tuple[int, str]] = []
        for chunk in result.chunks:
            if chunk.id not in detection_map:
                continue
            det = detection_map[chunk.id]
            comment = (
                f"{comment_prefix} chunk={chunk.id[:8]} "
                f"score={det.ensemble_score:.3f} "
                f"class={det.classification}{comment_suffix}"
            )
            annotations.append((chunk.start_pos, comment))

        if not annotations:
            return original_content

        # Insert annotations in reverse order to preserve positions
        annotations.sort(key=lambda x: x[0], reverse=True)

        annotated = original_content
        for pos, comment in annotations:
            annotated = annotated[:pos] + comment + "\n" + annotated[pos:]

        return annotated

    def write(
        self,
        result: DocumentResult,
        path: Path,
        output_format: str,
        config: LUCIDConfig | None = None,
        original_content: str | None = None,
    ) -> None:
        """Write formatted output to a file.

        Args:
            result: Completed pipeline results.
            path: Output file path.
            output_format: One of "json", "text", "annotated".
            config: Required for JSON format.
            original_content: Required for annotated format.

        Raises:
            ValueError: If required arguments are missing for the chosen format.
        """
        if output_format == "json":
            if config is None:
                raise ValueError("config is required for JSON output format")
            content = self.format_json(result, config)
        elif output_format == "text":
            content = self.format_text(result)
        elif output_format == "annotated":
            if original_content is None:
                raise ValueError(
                    "original_content is required for annotated output format"
                )
            content = self.format_annotated(result, original_content)
        else:
            raise ValueError(f"Unknown output format: {output_format!r}")

        path.write_text(content, encoding="utf-8")
