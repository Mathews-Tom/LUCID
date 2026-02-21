#!/usr/bin/env python3
"""Generate Markdown benchmark report from JSON results.

Usage:
    uv run python scripts/bench_report.py [path-to-json]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def find_latest_results() -> Path:
    """Find the most recent benchmark results file."""
    results_dir = Path(__file__).resolve().parent.parent / "benchmarks" / "results"
    if not results_dir.exists():
        print(f"No results directory at {results_dir}")
        sys.exit(1)
    files = sorted(results_dir.glob("*.json"), reverse=True)
    if not files:
        print("No benchmark result files found.")
        sys.exit(1)
    return files[0]


def format_detection_table(data: dict[str, Any]) -> str:
    """Format detection accuracy as Markdown table."""
    if not data:
        return "No detection data available.\n"
    lines = ["## Detection Accuracy\n",
             "| Threshold | TPR | FPR |",
             "|-----------|-----|-----|"]
    for i, t in enumerate(data.get("thresholds", [])):
        tpr = data.get("tpr", [])[i] if i < len(data.get("tpr", [])) else 0
        fpr = data.get("fpr", [])[i] if i < len(data.get("fpr", [])) else 0
        lines.append(f"| {t:.2f} | {tpr:.3f} | {fpr:.3f} |")
    corpus = data.get("corpus_size", {})
    lines.append(f"\nCorpus: {corpus.get('ai', 0)} AI, {corpus.get('human', 0)} human samples\n")
    return "\n".join(lines)


def format_evasion_table(data: dict[str, Any]) -> str:
    """Format evasion rates as Markdown table."""
    if not data:
        return "No evasion data available.\n"
    lines = ["## Evasion Rates\n",
             "| Metric | Value |",
             "|--------|-------|",
             f"| Single-pass evasion | {data.get('single_pass_evasion_rate', 0):.1%} |",
             f"| Adversarial evasion | {data.get('adversarial_evasion_rate', 0):.1%} |",
             f"| Mean iterations | {data.get('mean_iterations', 0):.1f} |"]
    strategies = data.get("strategy_distribution", {})
    if strategies:
        lines.extend(["\n### Strategy Distribution\n",
                       "| Strategy | Count |", "|----------|-------|"])
        for s, c in sorted(strategies.items()):
            lines.append(f"| {s} | {c} |")
    return "\n".join(lines)


def format_semantic_table(data: dict[str, Any]) -> str:
    """Format semantic preservation as Markdown table."""
    if not data:
        return "No semantic data available.\n"
    emb = data.get("embedding_similarity", {})
    bert = data.get("bertscore_f1", {})
    lines = ["## Semantic Preservation\n",
             "| Metric | Mean | Min | Max |",
             "|--------|------|-----|-----|",
             f"| Embedding similarity | {emb.get('mean', 0):.3f} | "
             f"{emb.get('min', 0):.3f} | {emb.get('max', 0):.3f} |",
             f"| BERTScore F1 | {bert.get('mean', 0):.3f} | "
             f"{bert.get('min', 0):.3f} | {bert.get('max', 0):.3f} |",
             f"\nPass rate: {data.get('pass_rate', 0):.1%}\n"]
    return "\n".join(lines)


def format_latency_table(data: dict[str, Any]) -> str:
    """Format latency as Markdown table."""
    if not data:
        return "No latency data available.\n"
    lines = ["## Latency\n",
             "| Metric | Value (ms) |",
             "|--------|------------|"]
    for key, value in data.items():
        if key == "note":
            continue
        lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
    note = data.get("note")
    if note:
        lines.append(f"\n*{note}*\n")
    return "\n".join(lines)


def main() -> None:
    """Generate and print benchmark report."""
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else find_latest_results()
    data = json.loads(path.read_text(encoding="utf-8"))

    meta = data.get("metadata", {})
    print("# LUCID Benchmark Report\n")
    print(f"- **Version:** {meta.get('lucid_version', 'unknown')}")
    print(f"- **Date:** {meta.get('timestamp', 'unknown')}")
    print(f"- **Commit:** {meta.get('git_commit', 'unknown')}")
    print(f"- **Profile:** {meta.get('profile', 'unknown')}")
    print(f"- **Real models:** {meta.get('real_models', False)}")
    print(f"- **Platform:** {meta.get('platform', 'unknown')}")
    print()
    print(format_detection_table(data.get("detection_accuracy", {})))
    print(format_evasion_table(data.get("evasion", {})))
    print(format_semantic_table(data.get("semantic_preservation", {})))
    print(format_latency_table(data.get("latency_ms", {})))


if __name__ == "__main__":
    main()
