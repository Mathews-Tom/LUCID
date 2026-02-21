#!/usr/bin/env python3
"""Detect AI-generated content in a LaTeX document.

Demonstrates:
- Loading config with a specific profile
- Running detection-only pipeline
- Formatting output as text and JSON
- Accessing summary statistics

Usage:
    uv run python examples/detect_latex.py <path-to-tex-file>
"""
from __future__ import annotations

import sys
from pathlib import Path

from lucid.config import load_config
from lucid.output import OutputFormatter
from lucid.pipeline import LUCIDPipeline


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python examples/detect_latex.py <path-to-tex-file>")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    if not input_path.exists():
        print(f"File not found: {input_path}")
        sys.exit(1)

    # Load config with balanced profile
    config = load_config(profile="balanced")

    # Create pipeline and run detection only
    pipeline = LUCIDPipeline(config)
    result = pipeline.run_detect_only(input_path)

    # Format as human-readable text
    formatter = OutputFormatter()
    print(formatter.format_text(result))

    # Write JSON report to file
    report_path = input_path.with_suffix(".report.json")
    report_json = formatter.format_json(result, config)
    report_path.write_text(report_json, encoding="utf-8")
    print(f"JSON report written to: {report_path}")

    # Access summary statistics
    stats = result.summary_stats
    print(f"\nSummary: {stats.get('prose_chunks', 0)} prose chunks analyzed, "
          f"{stats.get('ai_detected', 0)} flagged as AI-generated")


if __name__ == "__main__":
    main()
