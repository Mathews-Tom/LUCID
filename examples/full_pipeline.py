#!/usr/bin/env python3
"""Run the full LUCID pipeline with progress tracking and all output formats.

Demonstrates:
- Custom progress callback
- Checkpoint directory for resume support
- All three output formats (text, JSON, annotated)
- Accessing detailed result data

Usage:
    uv run python examples/full_pipeline.py <path-to-document>
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from lucid.config import load_config
from lucid.output import OutputFormatter
from lucid.pipeline import LUCIDPipeline

if TYPE_CHECKING:
    from lucid.progress import PipelineEvent


def progress_callback(event: PipelineEvent) -> None:
    """Print each pipeline event to stdout."""
    prefix = f"[{event.state}]"
    progress = f" ({event.chunk_index + 1}/{event.total_chunks})" if event.total_chunks > 0 else ""
    print(f"  {prefix}{progress} {event.detail}")


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/full_pipeline.py <path-to-document>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    # Load config with quality profile
    config = load_config(profile="quality")

    # Create pipeline with checkpoint support
    checkpoint_dir = Path(".lucid_checkpoints")
    pipeline = LUCIDPipeline(config, checkpoint_dir=checkpoint_dir)

    # Run pipeline with progress callback
    print(f"Processing: {input_path}")
    print("-" * 50)
    result = pipeline.run(input_path, progress_callback=progress_callback)
    print("-" * 50)

    # Generate all three output formats
    formatter = OutputFormatter()
    original_content = input_path.read_text(encoding="utf-8")

    # Text report
    print("\n" + formatter.format_text(result))

    # JSON report
    json_path = input_path.with_suffix(".report.json")
    json_path.write_text(formatter.format_json(result, config), encoding="utf-8")
    print(f"JSON report: {json_path}")

    # Annotated document
    annotated_path = input_path.with_stem(input_path.stem + "_annotated")
    annotated_path.write_text(
        formatter.format_annotated(result, original_content), encoding="utf-8"
    )
    print(f"Annotated output: {annotated_path}")

    # Summary
    stats = result.summary_stats
    print("\nPipeline complete:")
    print(f"  Total chunks: {stats.get('total_chunks', 0)}")
    print(f"  AI-detected: {stats.get('ai_detected', 0)}")
    print(f"  Humanized: {stats.get('humanized', 0)}")
    print(f"  Eval passed: {stats.get('eval_passed', 0)}")
    print(f"  Failed: {stats.get('failed', 0)}")


if __name__ == "__main__":
    main()
