"""Checkpoint persistence for resumable pipeline execution."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from lucid.models.results import DocumentResult

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CheckpointData:
    """Immutable snapshot of pipeline progress.

    Args:
        input_hash: SHA-256 hex digest of the input file content.
        format: Document format (latex, markdown, plaintext).
        profile: Quality profile name (fast, balanced, quality).
        state: Current pipeline state (e.g. DETECTING, HUMANIZING).
        document_result: Accumulated document processing results.
        completed_chunk_ids: Chunk IDs completed per stage.
        failed_chunk_ids: Chunk ID to error message mapping.
        timestamp: ISO 8601 timestamp of checkpoint creation.
    """

    input_hash: str
    format: str
    profile: str
    state: str
    document_result: DocumentResult
    completed_chunk_ids: dict[str, list[str]]
    failed_chunk_ids: dict[str, str]
    timestamp: str


class CheckpointManager:
    """Manages atomic checkpoint save/load for pipeline resumption."""

    def __init__(self, checkpoint_dir: Path, input_path: Path) -> None:
        self._checkpoint_dir = checkpoint_dir
        self._input_path = input_path
        self._input_hash = self._compute_hash(input_path)
        self._checkpoint_file = checkpoint_dir / f"{input_path.name}.checkpoint.json"

    @staticmethod
    def _compute_hash(path: Path) -> str:
        """Compute SHA-256 hex digest of file content."""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8192), b""):
                h.update(block)
        return h.hexdigest()

    def save(
        self,
        document_result: DocumentResult,
        state: str,
        completed_ids: dict[str, list[str]],
        failed_ids: dict[str, str],
    ) -> None:
        """Atomically write checkpoint to disk.

        Writes to a temporary file first, then replaces the target
        to avoid partial writes on crash.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "input_hash": self._input_hash,
            "format": document_result.format,
            "profile": "balanced",
            "state": state,
            "document_result": document_result.to_dict(),
            "completed_chunk_ids": completed_ids,
            "failed_chunk_ids": failed_ids,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        tmp_path = self._checkpoint_file.with_suffix(".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp_path, self._checkpoint_file)

    def load(self) -> CheckpointData | None:
        """Load checkpoint from disk.

        Returns:
            CheckpointData if a valid checkpoint exists, None otherwise.
            Logs warnings for corrupt JSON or input hash mismatch.
        """
        if not self._checkpoint_file.is_file():
            return None

        try:
            with open(self._checkpoint_file, encoding="utf-8") as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Corrupt checkpoint file %s: %s", self._checkpoint_file, exc)
            return None

        stored_hash = raw.get("input_hash", "")
        if stored_hash != self._input_hash:
            logger.warning(
                "Checkpoint hash mismatch for %s (stored=%s, current=%s). "
                "Input file changed since last run.",
                self._checkpoint_file,
                stored_hash,
                self._input_hash,
            )
            return None

        try:
            document_result = DocumentResult.from_dict(raw["document_result"])
        except (KeyError, ValueError, TypeError) as exc:
            logger.warning("Failed to deserialize document_result from checkpoint: %s", exc)
            return None

        return CheckpointData(
            input_hash=raw["input_hash"],
            format=raw["format"],
            profile=raw["profile"],
            state=raw["state"],
            document_result=document_result,
            completed_chunk_ids=raw["completed_chunk_ids"],
            failed_chunk_ids=raw["failed_chunk_ids"],
            timestamp=raw["timestamp"],
        )

    def exists(self) -> bool:
        """Return True if a checkpoint file exists on disk."""
        return self._checkpoint_file.is_file()

    def clear(self) -> None:
        """Delete the checkpoint file if it exists."""
        if self._checkpoint_file.is_file():
            self._checkpoint_file.unlink()
