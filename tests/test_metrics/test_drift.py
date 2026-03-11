"""Tests for drift metrics."""

from __future__ import annotations

import pytest

from lucid.metrics.drift import CitationDriftMetric, EntityDriftMetric, NumericDriftMetric


class TestNumericDriftMetric:
    def setup_method(self) -> None:
        self.metric = NumericDriftMetric()

    def test_all_numbers_preserved(self) -> None:
        original = "There are 42 items and 3.14 ratio."
        transformed = "There are 42 items and 3.14 ratio."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(1.0)

    def test_some_numbers_missing(self) -> None:
        original = "Values 10, 20, 30 were recorded."
        transformed = "Values 10 and 30 were recorded."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(2 / 3)

    def test_no_numbers_in_original(self) -> None:
        original = "No numbers here."
        transformed = "Still no numbers."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(0.0)

    def test_all_numbers_dropped(self) -> None:
        original = "The value is 100."
        transformed = "The value is large."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(0.0)


class TestEntityDriftMetric:
    def setup_method(self) -> None:
        self.metric = EntityDriftMetric()

    def test_entities_preserved(self) -> None:
        original = "The study by Smith and Johnson found results."
        transformed = "The research by Smith and Johnson revealed results."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(1.0)

    def test_entity_dropped(self) -> None:
        original = "The study by Smith and Johnson found results."
        transformed = "The study by Smith found results."
        result = self.metric.compute(original, transformed)
        assert result.value < 1.0

    def test_no_entities(self) -> None:
        original = "no capitalized words here at all."
        transformed = "still no capitalized words."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(1.0)


class TestCitationDriftMetric:
    def setup_method(self) -> None:
        self.metric = CitationDriftMetric()

    def test_all_citations_preserved(self) -> None:
        original = "As shown in [1] and [2], the results hold."
        transformed = "The results hold as shown in [1] and [2]."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(1.0)

    def test_citation_dropped(self) -> None:
        original = "References [1], [2], and [3] confirm this."
        transformed = "References [1] and [3] confirm this."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(2 / 3)

    def test_no_citations(self) -> None:
        original = "No citations in this text."
        transformed = "This text has no citations."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(1.0)

    def test_all_citations_dropped(self) -> None:
        original = "See [1] and [2] for details."
        transformed = "See the references for details."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(0.0)
