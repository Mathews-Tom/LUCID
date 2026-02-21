"""Benchmark suite fixtures and result collection."""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from lucid import __version__


def _git_commit() -> str:
    """Get current git short commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent.parent,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


@dataclass
class BenchmarkCollector:
    """Accumulates benchmark results across a pytest session."""

    metadata: dict[str, Any] = field(default_factory=dict)
    detection_accuracy: dict[str, Any] = field(default_factory=dict)
    evasion: dict[str, Any] = field(default_factory=dict)
    semantic_preservation: dict[str, Any] = field(default_factory=dict)
    latency_ms: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all results."""
        return {
            "metadata": self.metadata,
            "detection_accuracy": self.detection_accuracy,
            "evasion": self.evasion,
            "semantic_preservation": self.semantic_preservation,
            "latency_ms": self.latency_ms,
        }

    def write(self, output_dir: Path) -> Path:
        """Write results to a timestamped JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(UTC).strftime("%Y%m%d")
        commit = self.metadata.get("git_commit", "unknown")
        path = output_dir / f"{timestamp}-{commit}.json"
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path


@pytest.fixture(scope="session")
def real_models() -> bool:
    """Return True if real models should be used (LUCID_BENCH_REAL=1)."""
    return os.environ.get("LUCID_BENCH_REAL") == "1"


@pytest.fixture(scope="session")
def benchmark_collector() -> BenchmarkCollector:
    """Session-scoped benchmark result collector."""
    import platform

    collector = BenchmarkCollector()
    collector.metadata = {
        "lucid_version": __version__,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": _git_commit(),
        "profile": "balanced",
        "real_models": os.environ.get("LUCID_BENCH_REAL") == "1",
        "platform": f"{platform.system()} {platform.machine()}",
    }
    return collector


@pytest.fixture(scope="session")
def benchmark_output_path() -> Path:
    """Output directory for benchmark results."""
    return Path(__file__).resolve().parent.parent.parent / "benchmarks" / "results"


@pytest.fixture(autouse=True, scope="session")
def _write_benchmark_results(
    benchmark_collector: BenchmarkCollector,
    benchmark_output_path: Path,
) -> Any:
    """Write collected benchmark results at session end."""
    yield
    has_data = any([
        benchmark_collector.detection_accuracy,
        benchmark_collector.evasion,
        benchmark_collector.semantic_preservation,
        benchmark_collector.latency_ms,
    ])
    if has_data:
        path = benchmark_collector.write(benchmark_output_path)
        print(f"\nBenchmark results written to: {path}")
