"""Readability and structural metrics."""

from __future__ import annotations

import re

import numpy as np

from lucid.core.types import MetricResult
from lucid.metrics import metric_registry


@metric_registry.register("sentence_length_variance")
class SentenceLengthVarianceMetric:
    """Measure variance in sentence lengths."""

    name: str = "sentence_length_variance"

    def compute(self, original: str, transformed: str) -> MetricResult:
        sentences = [s.strip() for s in re.split(r"[.!?]+", transformed) if s.strip()]
        if not sentences:
            return MetricResult(metric_name=self.name, value=0.0)
        lengths = [len(s.split()) for s in sentences]
        variance = float(np.var(lengths))
        return MetricResult(metric_name=self.name, value=variance)


@metric_registry.register("lexical_diversity")
class LexicalDiversityMetric:
    """Type-token ratio of transformed text."""

    name: str = "lexical_diversity"

    def compute(self, original: str, transformed: str) -> MetricResult:
        words = transformed.lower().split()
        if not words:
            return MetricResult(metric_name=self.name, value=0.0)
        ttr = len(set(words)) / len(words)
        return MetricResult(metric_name=self.name, value=ttr)
