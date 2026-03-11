"""Domain-specific drift metrics."""

from __future__ import annotations

import re

from lucid.core.types import MetricResult
from lucid.metrics import metric_registry


@metric_registry.register("numeric_drift")
class NumericDriftMetric:
    """Detect numeric value changes between original and transformed text."""

    name: str = "numeric_drift"

    def compute(self, original: str, transformed: str) -> MetricResult:
        original_nums = set(re.findall(r"\b\d+\.?\d*\b", original))
        transformed_nums = set(re.findall(r"\b\d+\.?\d*\b", transformed))
        preserved = len(original_nums & transformed_nums)
        total = len(original_nums) if original_nums else 1
        score = preserved / total
        return MetricResult(
            metric_name=self.name,
            value=score,
            metadata={"original_count": len(original_nums), "preserved": preserved},
        )


@metric_registry.register("entity_drift")
class EntityDriftMetric:
    """Detect entity (capitalized word) changes between texts."""

    name: str = "entity_drift"

    def compute(self, original: str, transformed: str) -> MetricResult:
        original_entities = self._extract_entities(original)
        transformed_entities = self._extract_entities(transformed)
        if not original_entities:
            return MetricResult(metric_name=self.name, value=1.0)
        preserved = len(original_entities & transformed_entities)
        score = preserved / len(original_entities)
        return MetricResult(
            metric_name=self.name,
            value=score,
            metadata={
                "original_count": len(original_entities),
                "preserved": preserved,
            },
        )

    @staticmethod
    def _extract_entities(text: str) -> set[str]:
        """Extract capitalized words that are not sentence-initial."""
        words = text.split()
        entities: set[str] = set()
        for i, word in enumerate(words):
            clean = word.strip(".,;:!?\"'()[]")
            if not clean:
                continue
            # Skip sentence-initial words (index 0 or preceded by sentence-ending punctuation)
            if i == 0:
                continue
            if i > 0 and words[i - 1].rstrip().endswith((".", "!", "?")):
                continue
            if clean[0].isupper() and clean.isalpha():
                entities.add(clean)
        return entities


@metric_registry.register("citation_drift")
class CitationDriftMetric:
    """Detect changes in bracketed references like [1], [2]."""

    name: str = "citation_drift"

    _CITATION_RE = re.compile(r"\[\d+\]")

    def compute(self, original: str, transformed: str) -> MetricResult:
        original_citations = set(self._CITATION_RE.findall(original))
        transformed_citations = set(self._CITATION_RE.findall(transformed))
        if not original_citations:
            return MetricResult(metric_name=self.name, value=1.0)
        preserved = len(original_citations & transformed_citations)
        score = preserved / len(original_citations)
        return MetricResult(
            metric_name=self.name,
            value=score,
            metadata={
                "original_count": len(original_citations),
                "preserved": preserved,
            },
        )
