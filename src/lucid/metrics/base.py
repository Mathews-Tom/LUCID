"""Base classes and runner for metric computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from lucid.core.types import MetricResult

if TYPE_CHECKING:
    from lucid.core.protocols import Metric
    from lucid.core.registry import Registry


class MetricRunner:
    """Execute multiple metrics via registry lookup."""

    def __init__(self, registry: Registry[Metric]) -> None:
        self._registry = registry
        self._instances: dict[str, Metric] = {}

    def run(
        self, metric_names: list[str], original: str, transformed: str
    ) -> list[MetricResult]:
        """Run named metrics and collect results."""
        results: list[MetricResult] = []
        for name in metric_names:
            metric = self._get_or_create(name)
            result = metric.compute(original, transformed)
            results.append(result)
        return results

    def _get_or_create(self, name: str) -> Metric:
        if name not in self._instances:
            cls = self._registry.get(name)
            self._instances[name] = cls()
        return self._instances[name]
