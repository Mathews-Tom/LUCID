"""Tests for benchmark dataset loading and writing."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lucid.bench.datasets import DatasetLoader
from lucid.core.errors import BenchmarkError
from lucid.core.types import SampleRecord


def _make_sample(
    sample_id: str = "smp_test0001",
    source_class: str = "human",
    domain: str = "academic",
    text: str = "This is test text.",
) -> SampleRecord:
    return SampleRecord(
        sample_id=sample_id,
        split="test",
        domain=domain,
        source_class=source_class,
        source_model=None,
        document_format="plaintext",
        text=text,
    )


class TestLoadJsonl:
    def test_round_trip(self, tmp_path: Path) -> None:
        samples = [
            _make_sample("smp_00000001"),
            _make_sample("smp_00000002", source_class="ai_raw", text="AI text."),
        ]
        path = tmp_path / "dataset.jsonl"
        DatasetLoader.write_jsonl(samples, path)
        loaded = DatasetLoader.load_jsonl(path)

        assert len(loaded) == 2
        assert loaded[0].sample_id == "smp_00000001"
        assert loaded[1].source_class == "ai_raw"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(BenchmarkError, match="not found"):
            DatasetLoader.load_jsonl(tmp_path / "nonexistent.jsonl")

    def test_invalid_json_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(BenchmarkError, match="Invalid JSON"):
            DatasetLoader.load_jsonl(path)

    def test_invalid_record_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad_record.jsonl"
        path.write_text(json.dumps({"sample_id": "x"}) + "\n", encoding="utf-8")
        with pytest.raises(BenchmarkError, match="Invalid SampleRecord"):
            DatasetLoader.load_jsonl(path)

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        sample = _make_sample()
        path = tmp_path / "with_blanks.jsonl"
        lines = ["", json.dumps(sample.to_dict()), "", ""]
        path.write_text("\n".join(lines), encoding="utf-8")
        loaded = DatasetLoader.load_jsonl(path)
        assert len(loaded) == 1

    def test_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(BenchmarkError, match="not a file"):
            DatasetLoader.load_jsonl(tmp_path)


class TestLoadCorpus:
    def test_load_corpus(self, tmp_path: Path) -> None:
        # Create corpus structure: domain/source_class/*.txt
        (tmp_path / "academic" / "human").mkdir(parents=True)
        (tmp_path / "academic" / "ai_raw").mkdir(parents=True)
        (tmp_path / "academic" / "human" / "doc1.txt").write_text("Human text.", encoding="utf-8")
        (tmp_path / "academic" / "ai_raw" / "doc2.txt").write_text("AI text.", encoding="utf-8")

        records = DatasetLoader.load_corpus(tmp_path)
        assert len(records) == 2
        domains = {r.domain for r in records}
        assert domains == {"academic"}
        classes = {r.source_class for r in records}
        assert classes == {"human", "ai_raw"}

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(BenchmarkError, match="not found"):
            DatasetLoader.load_corpus(tmp_path / "nonexistent")

    def test_empty_corpus_raises(self, tmp_path: Path) -> None:
        (tmp_path / "empty_corpus").mkdir()
        with pytest.raises(BenchmarkError, match="No .txt files"):
            DatasetLoader.load_corpus(tmp_path / "empty_corpus")

    def test_file_instead_of_dir_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "notadir.txt"
        path.write_text("x", encoding="utf-8")
        with pytest.raises(BenchmarkError, match="not a directory"):
            DatasetLoader.load_corpus(path)

    def test_invalid_source_class_raises(self, tmp_path: Path) -> None:
        (tmp_path / "domain" / "invalid_class").mkdir(parents=True)
        (tmp_path / "domain" / "invalid_class" / "doc.txt").write_text("text", encoding="utf-8")
        with pytest.raises(BenchmarkError, match="Invalid record"):
            DatasetLoader.load_corpus(tmp_path)


class TestWriteJsonl:
    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        samples = [_make_sample()]
        path = tmp_path / "sub" / "dir" / "out.jsonl"
        DatasetLoader.write_jsonl(samples, path)
        assert path.exists()
        loaded = DatasetLoader.load_jsonl(path)
        assert len(loaded) == 1
