"""Progress reporting for the LUCID pipeline."""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

if TYPE_CHECKING:
    from rich.console import Console
    from rich.progress import TaskID

    from lucid.models.results import DocumentResult


@dataclass(frozen=True, slots=True)
class PipelineEvent:
    """Immutable event emitted by each pipeline stage.

    Attributes:
        state: Pipeline state name (e.g. "PARSING", "DETECTING").
        chunk_id: ID of the chunk being processed, if applicable.
        chunk_index: Zero-based index of the current chunk.
        total_chunks: Total number of chunks in the document.
        detail: Human-readable detail string for verbose output.
    """

    state: str
    chunk_id: str | None
    chunk_index: int
    total_chunks: int
    detail: str


def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {secs:.1f}s"
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours}h {int(minutes)}m {secs:.1f}s"


class ProgressReporter:
    """Rich-based progress display for the LUCID pipeline.

    Renders a live progress bar on TTY stderr. Falls back to structured
    log messages when stderr is not a terminal.
    """

    def __init__(
        self,
        console: Console,
        verbose: bool = False,
        quiet: bool = False,
    ) -> None:
        self._console = console
        self._verbose = verbose
        self._quiet = quiet
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._current_state: str = ""
        self._is_tty: bool = sys.stderr.isatty()
        self._logger: logging.Logger = logging.getLogger("lucid.progress")
        self._start_time: float = 0.0

    def callback(self, event: PipelineEvent) -> None:
        """Handle a pipeline event -- update progress display."""
        if self._quiet:
            return

        if event.state != self._current_state:
            self._current_state = event.state
            if self._progress is not None and self._task_id is not None:
                self._progress.update(self._task_id, description=f"[cyan]{event.state}")

        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=event.chunk_index + 1,
                total=event.total_chunks,
            )

        if self._verbose and event.detail:
            if self._is_tty:
                self._console.print(f"  [dim]{event.detail}[/dim]")
            else:
                self._logger.info(event.detail)

        if not self._is_tty and not self._verbose:
            self._logger.info(
                "%s [%d/%d] %s",
                event.state,
                event.chunk_index + 1,
                event.total_chunks,
                event.detail,
            )

    def start(self, total_chunks: int) -> None:
        """Start the progress display."""
        if self._quiet:
            return

        self._start_time = time.perf_counter()

        if self._is_tty:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                console=self._console,
            )
            self._progress.start()
            self._task_id = self._progress.add_task("[cyan]STARTING", total=total_chunks)
        else:
            self._logger.info("Pipeline started")

    def finish(self, document_result: DocumentResult) -> None:
        """Stop progress and print summary table."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

        if self._quiet:
            return

        stats = document_result.summary_stats
        table = Table(title="Pipeline Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total chunks", str(stats.get("total_chunks", 0)))
        table.add_row("Prose chunks", str(stats.get("prose_chunks", 0)))
        table.add_row("AI-detected", str(stats.get("ai_detected", 0)))
        transformed = stats.get("transformed", 0)
        table.add_row("Transformed", str(transformed))
        unchanged = stats.get("unchanged", 0)
        if unchanged > 0:
            table.add_row("Unchanged", str(unchanged))
        table.add_row("Eval passed", str(stats.get("eval_passed", 0)))
        table.add_row("Eval failed", str(stats.get("eval_failed", 0)))
        failed = stats.get("failed", 0)
        if failed > 0:
            table.add_row(
                "Skipped (failed)",
                f"[red]{failed}[/red]",
            )
        else:
            table.add_row("Skipped (failed)", "0")

        skipped_non_transformable = stats.get("skipped_non_transformable", {})
        if skipped_non_transformable:
            skip_parts = [
                f"{reason} ({count})"
                for reason, count in skipped_non_transformable.items()
            ]
            table.add_row("Skipped (policy)", "; ".join(skip_parts))

        ai_detected = stats.get("ai_detected", 0)
        if ai_detected > 0 and transformed == 0 and failed > 0:
            table.add_row(
                "[bold red]WARNING[/bold red]",
                "[red]All transformations failed — output is identical to input[/red]",
            )

        # Surface transformation failure reasons
        failure_reasons = stats.get("failure_reasons", {})
        if failure_reasons:
            reason_parts = [f"{reason} ({count})" for reason, count in failure_reasons.items()]
            table.add_row("Failure reasons", "; ".join(reason_parts))

        fallback_modes = stats.get("fallback_modes", {})
        if fallback_modes:
            mode_parts = [f"{mode} ({count})" for mode, count in fallback_modes.items()]
            table.add_row("Fallback modes", "; ".join(mode_parts))

        operator_usage = stats.get("operator_usage", {})
        if operator_usage:
            operator_parts = [f"{name} ({count})" for name, count in operator_usage.items()]
            table.add_row("Operators", "; ".join(operator_parts))

        search_diagnostics = stats.get("search_diagnostics", {})
        if search_diagnostics:
            summary = (
                f"placeholder_failures={search_diagnostics.get('placeholder_failures', 0)}, "
                f"chunks_with_placeholder_failures="
                f"{search_diagnostics.get('chunks_with_placeholder_failures', 0)}, "
                "semantic_gate_rejections="
                f"{search_diagnostics.get('semantic_gate_rejections', 0)}, "
                f"low_similarity_rejections="
                f"{search_diagnostics.get('low_similarity_rejections', 0)}, "
                f"prompt_echo_rejections="
                f"{search_diagnostics.get('prompt_echo_rejections', 0)}, "
                f"restore_failures={search_diagnostics.get('restore_failures', 0)}, "
                f"retries_used={search_diagnostics.get('retries_used', 0)}"
            )
            table.add_row("Search diagnostics", summary)

        # Surface evaluation rejection reasons
        rejected = [e for e in document_result.evaluations if not e.passed]
        if rejected:
            reasons: dict[str, int] = {}
            for e in rejected:
                key = e.rejection_reason or "unknown"
                # Normalize variable scores to category
                if key.startswith("embedding similarity"):
                    key = "embedding similarity below threshold"
                elif key.startswith("BERTScore F1"):
                    key = "BERTScore F1 below threshold"
                reasons[key] = reasons.get(key, 0) + 1
            summary_parts = [f"{reason} ({count})" for reason, count in reasons.items()]
            table.add_row("Rejection reasons", "; ".join(summary_parts))

        rejection_stages = stats.get("evaluation_rejection_stages", {})
        if rejection_stages:
            stage_parts = [f"{stage} ({count})" for stage, count in rejection_stages.items()]
            table.add_row("Rejection stages", "; ".join(stage_parts))

        if document_result.output_path:
            table.add_row("Output", document_result.output_path)

        elapsed = time.perf_counter() - self._start_time
        table.add_row("Elapsed", _format_duration(elapsed))

        self._console.print(table)
