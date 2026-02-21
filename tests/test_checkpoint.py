"""Tests for checkpoint save/load/clear lifecycle."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from lucid.checkpoint import CheckpointData, CheckpointManager
from lucid.models.results import DocumentResult


@pytest.fixture
def input_file(tmp_path: Path) -> Path:
    """Create a minimal input file for checkpoint hashing."""
    p = tmp_path / "test.md"
    p.write_text("# Hello\n\nSome content here.", encoding="utf-8")
    return p


@pytest.fixture
def checkpoint_dir(tmp_path: Path) -> Path:
    """Dedicated checkpoint directory."""
    return tmp_path / "checkpoints"


@pytest.fixture
def manager(checkpoint_dir: Path, input_file: Path) -> CheckpointManager:
    """Pre-configured CheckpointManager."""
    return CheckpointManager(checkpoint_dir=checkpoint_dir, input_path=input_file)


@pytest.fixture
def sample_doc() -> DocumentResult:
    """Minimal DocumentResult for testing."""
    return DocumentResult(input_path="test.md", format="markdown")


class TestRoundTrip:
    """Save then load produces equivalent CheckpointData."""

    def test_round_trip(
        self, manager: CheckpointManager, sample_doc: DocumentResult
    ) -> None:
        """Saved checkpoint loads back with matching fields."""
        completed = {"detection": ["c1", "c2"], "humanization": [], "evaluation": []}
        failed = {"c3": "timeout"}

        manager.save(sample_doc, state="DETECTING", completed_ids=completed, failed_ids=failed)

        loaded = manager.load()
        assert loaded is not None
        assert isinstance(loaded, CheckpointData)
        assert loaded.state == "DETECTING"
        assert loaded.format == "markdown"
        assert loaded.completed_chunk_ids == completed
        assert loaded.failed_chunk_ids == failed
        assert loaded.document_result.input_path == "test.md"
        assert loaded.document_result.format == "markdown"
        assert loaded.timestamp != ""


class TestLoadMissingFile:
    """load() returns None when no checkpoint exists."""

    def test_missing_file(self, manager: CheckpointManager) -> None:
        """No checkpoint file on disk returns None."""
        assert manager.load() is None


class TestLoadCorruptJSON:
    """load() returns None and logs warning for corrupt JSON."""

    def test_corrupt_json(
        self,
        manager: CheckpointManager,
        sample_doc: DocumentResult,
        checkpoint_dir: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Corrupt JSON triggers warning and returns None."""
        manager.save(sample_doc, state="DETECTING", completed_ids={}, failed_ids={})

        # Corrupt the file
        checkpoint_file = checkpoint_dir / "test.md.checkpoint.json"
        checkpoint_file.write_text("{invalid json", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            result = manager.load()

        assert result is None
        assert "Corrupt checkpoint file" in caplog.text


class TestLoadHashMismatch:
    """load() returns None and logs warning when input file changed."""

    def test_hash_mismatch(
        self,
        manager: CheckpointManager,
        sample_doc: DocumentResult,
        input_file: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Changed input file triggers hash mismatch warning."""
        manager.save(sample_doc, state="DETECTING", completed_ids={}, failed_ids={})

        # Modify the input file (changes hash)
        input_file.write_text("completely different content", encoding="utf-8")

        # Create new manager with updated hash
        new_manager = CheckpointManager(
            checkpoint_dir=manager._checkpoint_dir, input_path=input_file
        )

        with caplog.at_level(logging.WARNING):
            result = new_manager.load()

        assert result is None
        assert "hash mismatch" in caplog.text


class TestExists:
    """exists() reflects checkpoint file presence."""

    def test_before_save(self, manager: CheckpointManager) -> None:
        """No checkpoint file before first save."""
        assert manager.exists() is False

    def test_after_save(
        self, manager: CheckpointManager, sample_doc: DocumentResult
    ) -> None:
        """Checkpoint file present after save."""
        manager.save(sample_doc, state="DETECTING", completed_ids={}, failed_ids={})
        assert manager.exists() is True


class TestClear:
    """clear() removes checkpoint file."""

    def test_clear_removes_file(
        self, manager: CheckpointManager, sample_doc: DocumentResult
    ) -> None:
        """clear() deletes the checkpoint and exists() returns False."""
        manager.save(sample_doc, state="DETECTING", completed_ids={}, failed_ids={})
        assert manager.exists() is True

        manager.clear()
        assert manager.exists() is False

    def test_clear_noop_when_missing(self, manager: CheckpointManager) -> None:
        """clear() does not raise when no checkpoint exists."""
        manager.clear()  # Should not raise


class TestAtomicWrite:
    """Atomic write leaves no .tmp file on success."""

    def test_no_tmp_file_lingers(
        self,
        manager: CheckpointManager,
        sample_doc: DocumentResult,
        checkpoint_dir: Path,
    ) -> None:
        """After save, no .tmp file remains in the checkpoint directory."""
        manager.save(sample_doc, state="DETECTING", completed_ids={}, failed_ids={})

        tmp_files = list(checkpoint_dir.glob("*.tmp"))
        assert tmp_files == []
