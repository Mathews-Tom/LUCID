#!/usr/bin/env python3
"""Humanize AI-detected content in a Markdown document.

Demonstrates:
- Running the full pipeline with an explicit output path
- Overriding the Ollama model via cli_overrides
- Accessing result output path and compilation status

Usage:
    uv run python examples/humanize_markdown.py <path-to-md-file>
"""
from __future__ import annotations

import sys
from pathlib import Path

from lucid.config import load_config
from lucid.pipeline import LUCIDPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/humanize_markdown.py <path-to-md-file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    # Load config with model override
    config = load_config(
        profile="balanced",
        cli_overrides={"ollama.models.balanced": "mistral:7b"},
    )

    # Run full pipeline
    pipeline = LUCIDPipeline(config)
    output_path = input_path.with_stem(input_path.stem + "_humanized")
    result = pipeline.run(input_path, output_path=output_path)

    # Report results
    print(f"Output written to: {result.output_path}")
    print(f"Compilation valid: {result.compilation_valid}")

    stats = result.summary_stats
    print(f"Chunks humanized: {stats.get('humanized', 0)}")
    print(f"Evaluation passed: {stats.get('eval_passed', 0)}")
    print(f"Evaluation failed: {stats.get('eval_failed', 0)}")


if __name__ == "__main__":
    main()
