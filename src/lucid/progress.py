"""Progress reporting for the LUCID pipeline."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console
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
            self._logger.info("Pipeline started -- %d chunks to process", total_chunks)

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
        table.add_row("Humanized", str(stats.get("humanized", 0)))
        table.add_row("Eval passed", str(stats.get("eval_passed", 0)))
        table.add_row("Eval failed", str(stats.get("eval_failed", 0)))
        table.add_row("Skipped (failed)", str(stats.get("failed", 0)))

        if document_result.output_path:
            table.add_row("Output", document_result.output_path)

        self._console.print(table)
