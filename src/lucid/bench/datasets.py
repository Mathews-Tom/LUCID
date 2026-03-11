"""Dataset loaders for benchmark samples."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from lucid.core.errors import BenchmarkError
from lucid.core.types import SampleRecord


class DatasetLoader:
    """Load and write benchmark datasets in JSONL and corpus-directory formats."""

    @staticmethod
    def load_jsonl(path: Path) -> list[SampleRecord]:
        """Load SampleRecord objects from a JSONL file.

        Each line must be a valid JSON object deserializable via SampleRecord.from_dict.
        Raises BenchmarkError on I/O or validation failures.
        """
        if not path.exists():
            raise BenchmarkError(f"Dataset file not found: {path}")
        if not path.is_file():
            raise BenchmarkError(f"Dataset path is not a file: {path}")

        records: list[SampleRecord] = []
        with path.open(encoding="utf-8") as fh:
            for line_num, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                except json.JSONDecodeError as exc:
                    raise BenchmarkError(
                        f"Invalid JSON on line {line_num} of {path}: {exc}"
                    ) from exc
                try:
                    records.append(SampleRecord.from_dict(data))
                except (KeyError, ValueError) as exc:
                    raise BenchmarkError(
                        f"Invalid SampleRecord on line {line_num} of {path}: {exc}"
                    ) from exc
        return records

    @staticmethod
    def load_corpus(
        corpus_dir: Path,
        split: str = "test",
        document_format: str = "plaintext",
    ) -> list[SampleRecord]:
        """Load text files from a corpus directory structure.

        Expected layout: corpus_dir/domain/source_class/*.txt
        The domain and source_class are inferred from directory names.
        source_model is None for all loaded records.
        """
        if not corpus_dir.exists():
            raise BenchmarkError(f"Corpus directory not found: {corpus_dir}")
        if not corpus_dir.is_dir():
            raise BenchmarkError(f"Corpus path is not a directory: {corpus_dir}")

        records: list[SampleRecord] = []
        for domain_dir in sorted(corpus_dir.iterdir()):
            if not domain_dir.is_dir():
                continue
            domain = domain_dir.name
            for source_class_dir in sorted(domain_dir.iterdir()):
                if not source_class_dir.is_dir():
                    continue
                source_class = source_class_dir.name
                for txt_file in sorted(source_class_dir.glob("*.txt")):
                    text = txt_file.read_text(encoding="utf-8")
                    sample_id = f"smp_{uuid4().hex[:8]}"
                    try:
                        record = SampleRecord(
                            sample_id=sample_id,
                            split=split,
                            domain=domain,
                            source_class=source_class,
                            source_model=None,
                            document_format=document_format,
                            text=text,
                        )
                    except ValueError as exc:
                        raise BenchmarkError(
                            f"Invalid record from {txt_file}: {exc}"
                        ) from exc
                    records.append(record)

        if not records:
            raise BenchmarkError(f"No .txt files found in corpus directory: {corpus_dir}")
        return records

    @staticmethod
    def write_jsonl(records: list[SampleRecord], path: Path) -> None:
        """Write SampleRecord objects to a JSONL file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            for record in records:
                fh.write(json.dumps(record.to_dict()) + "\n")
