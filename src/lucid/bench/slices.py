"""Slice dimensions and grouping logic for benchmark analysis."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from lucid.core.errors import BenchmarkError
from lucid.core.types import DetectionRecord, SampleRecord

VALID_SLICE_DIMENSIONS: frozenset[str] = frozenset({
    "domain",
    "source_class",
    "source_model",
    "detector",
    "operator",
    "intensity_bucket",
    "document_format",
})


@dataclass(frozen=True, slots=True)
class SliceKey:
    """Identifies a slice dimension and value."""

    dimension: str
    value: str


def group_by_slice(
    records: Sequence[DetectionRecord],
    samples: dict[str, SampleRecord],
    dimension: str,
) -> dict[str, list[DetectionRecord]]:
    """Group detection records by a slice dimension.

    Supported dimensions:
    - domain, source_class, source_model, document_format: from SampleRecord
    - detector: from DetectionRecord.detector_name
    - operator: from DetectionRecord metadata (transform_id prefix convention)
    - intensity_bucket: bucketed intensity from metadata
    """
    if dimension not in VALID_SLICE_DIMENSIONS:
        raise BenchmarkError(
            f"Invalid slice dimension {dimension!r}. "
            f"Valid dimensions: {', '.join(sorted(VALID_SLICE_DIMENSIONS))}"
        )

    groups: dict[str, list[DetectionRecord]] = defaultdict(list)

    for rec in records:
        value = _extract_dimension_value(rec, samples, dimension)
        if value is not None:
            groups[value].append(rec)

    return dict(groups)


def _extract_dimension_value(
    rec: DetectionRecord,
    samples: dict[str, SampleRecord],
    dimension: str,
) -> str | None:
    """Extract a dimension value from a detection record."""
    if dimension == "detector":
        return rec.detector_name

    sample = samples.get(rec.sample_id)
    if sample is None:
        return None

    if dimension == "domain":
        return sample.domain
    if dimension == "source_class":
        return sample.source_class
    if dimension == "source_model":
        return sample.source_model
    if dimension == "document_format":
        return sample.document_format

    # Metadata-based dimensions
    if dimension == "operator":
        if rec.evidence and "operator" in rec.evidence:
            return str(rec.evidence["operator"])
        return None

    if dimension == "intensity_bucket":
        if rec.evidence and "intensity" in rec.evidence:
            return bucket_intensity(float(rec.evidence["intensity"]))
        return None

    return None


def bucket_intensity(intensity: float, n_buckets: int = 5) -> str:
    """Assign an intensity value to a named bucket.

    Divides [0, 1] into n_buckets equal-width bins.
    Returns a string like '0.00-0.20'.
    """
    if n_buckets < 1:
        raise BenchmarkError(f"n_buckets must be >= 1, got {n_buckets}")

    bucket_width = 1.0 / n_buckets
    bucket_idx = min(int(intensity / bucket_width), n_buckets - 1)
    low = bucket_idx * bucket_width
    high = (bucket_idx + 1) * bucket_width
    return f"{low:.2f}-{high:.2f}"
