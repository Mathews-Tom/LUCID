"""Structural preservation metrics."""

from __future__ import annotations

import re

from lucid.core.types import MetricResult
from lucid.metrics import metric_registry


@metric_registry.register("heading_preservation")
class HeadingPreservationMetric:
    """Check that markdown/latex headings are preserved."""

    name: str = "heading_preservation"

    _HEADING_RE = re.compile(r"(?m)^#{1,6}\s+.+$|\\(?:sub)*section\{[^}]+\}")

    def compute(self, original: str, transformed: str) -> MetricResult:
        orig_headings = self._HEADING_RE.findall(original)
        trans_headings = self._HEADING_RE.findall(transformed)
        if not orig_headings:
            return MetricResult(metric_name=self.name, value=1.0)
        preserved = sum(1 for h in orig_headings if h in trans_headings)
        return MetricResult(
            metric_name=self.name,
            value=preserved / len(orig_headings),
            metadata={"original_count": len(orig_headings), "preserved": preserved},
        )
