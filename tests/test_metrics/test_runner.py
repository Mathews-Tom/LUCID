"""Tests for MetricRunner."""

from __future__ import annotations

import pytest

from lucid.core.registry import Registry
from lucid.core.types import MetricResult
from lucid.metrics.base import MetricRunner


class _FakeMetricA:
    name: str = "fake_a"

    def compute(self, original: str, transformed: str) -> MetricResult:
        return MetricResult(metric_name=self.name, value=0.9)


class _FakeMetricB:
    name: str = "fake_b"

    def compute(self, original: str, transformed: str) -> MetricResult:
        return MetricResult(metric_name=self.name, value=0.5)


def _make_registry() -> Registry:
    registry: Registry = Registry("test_metric")
    registry.register("fake_a")(_FakeMetricA)
    registry.register("fake_b")(_FakeMetricB)
    return registry


class TestMetricRunner:
    """Unit tests for MetricRunner."""

    def test_run_single_metric(self) -> None:
        runner = MetricRunner(_make_registry())
        results = runner.run(["fake_a"], "orig", "trans")
        assert len(results) == 1
        assert results[0].metric_name == "fake_a"
        assert results[0].value == pytest.approx(0.9)

    def test_run_multiple_metrics(self) -> None:
        runner = MetricRunner(_make_registry())
        results = runner.run(["fake_a", "fake_b"], "orig", "trans")
        assert len(results) == 2
        names = [r.metric_name for r in results]
        assert names == ["fake_a", "fake_b"]

    def test_run_caches_instances(self) -> None:
        runner = MetricRunner(_make_registry())
        runner.run(["fake_a"], "orig", "trans")
        runner.run(["fake_a"], "orig2", "trans2")
        assert len(runner._instances) == 1

    def test_run_unknown_metric_raises(self) -> None:
        runner = MetricRunner(_make_registry())
        with pytest.raises(KeyError, match="nonexistent"):
            runner.run(["nonexistent"], "orig", "trans")

    def test_run_empty_list(self) -> None:
        runner = MetricRunner(_make_registry())
        results = runner.run([], "orig", "trans")
        assert results == []
