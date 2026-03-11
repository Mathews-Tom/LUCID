"""Tests for structural preservation metrics."""

from __future__ import annotations

import pytest

from lucid.metrics.structure import HeadingPreservationMetric


class TestHeadingPreservationMetric:
    def setup_method(self) -> None:
        self.metric = HeadingPreservationMetric()

    def test_markdown_headings_preserved(self) -> None:
        original = "# Introduction\nSome text.\n## Methods\nMore text."
        transformed = "# Introduction\nRewritten text.\n## Methods\nRewritten more."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(1.0)

    def test_markdown_heading_dropped(self) -> None:
        original = "# Introduction\n## Methods\n## Results"
        transformed = "# Introduction\n## Results"
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(2 / 3)

    def test_latex_sections_preserved(self) -> None:
        original = r"\section{Introduction}" "\n" r"\subsection{Background}"
        transformed = r"\section{Introduction}" "\n" r"\subsection{Background}"
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(1.0)

    def test_no_headings_returns_one(self) -> None:
        original = "Just plain text without headings."
        transformed = "Rewritten plain text."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(1.0)

    def test_all_headings_dropped(self) -> None:
        original = "# Title\n## Subtitle"
        transformed = "Title text. Subtitle text."
        result = self.metric.compute(original, transformed)
        assert result.value == pytest.approx(0.0)
