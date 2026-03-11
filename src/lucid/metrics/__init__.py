"""Metric computation for semantic preservation evaluation."""

from __future__ import annotations

from typing import Any

from lucid.core.protocols import Metric
from lucid.core.registry import Registry

metric_registry = Registry[Metric]("metric")


def __getattr__(name: str) -> Any:
    """Lazy-load metric classes on first access."""
    if name == "MetricRunner":
        from lucid.metrics.base import MetricRunner

        return MetricRunner
    if name == "EmbeddingSimilarity":
        from lucid.metrics.embedding import EmbeddingSimilarity

        return EmbeddingSimilarity
    if name == "EmbeddingSimilarityMetric":
        from lucid.metrics.embedding import EmbeddingSimilarityMetric

        return EmbeddingSimilarityMetric
    if name == "NLIChecker":
        from lucid.metrics.nli import NLIChecker

        return NLIChecker
    if name == "NLIEntailmentMetric":
        from lucid.metrics.nli import NLIEntailmentMetric

        return NLIEntailmentMetric
    if name == "BERTScoreChecker":
        from lucid.metrics.bertscore import BERTScoreChecker

        return BERTScoreChecker
    if name == "BERTScoreMetric":
        from lucid.metrics.bertscore import BERTScoreMetric

        return BERTScoreMetric
    if name == "TermVerifier":
        from lucid.metrics.term_verify import TermVerifier

        return TermVerifier
    if name == "TermPreservationMetric":
        from lucid.metrics.term_verify import TermPreservationMetric

        return TermPreservationMetric
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "metric_registry",
    "MetricRunner",
    "EmbeddingSimilarity",
    "EmbeddingSimilarityMetric",
    "NLIChecker",
    "NLIEntailmentMetric",
    "BERTScoreChecker",
    "BERTScoreMetric",
    "TermVerifier",
    "TermPreservationMetric",
]
