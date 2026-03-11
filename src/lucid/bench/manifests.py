"""YAML experiment manifest parser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from lucid.core.errors import BenchmarkError


@dataclass(frozen=True, slots=True)
class TransformSpec:
    """Specification for a transform operator and its intensity levels."""

    operator: str
    intensities: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Parsed experiment manifest defining a benchmark run."""

    name: str
    dataset: str
    detectors: tuple[str, ...]
    transforms: tuple[TransformSpec, ...]
    metrics: tuple[str, ...]
    slices: tuple[str, ...]
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentManifest:
        """Parse an experiment manifest from a YAML file."""
        if not path.exists():
            raise BenchmarkError(f"Manifest file not found: {path}")

        raw = path.read_text(encoding="utf-8")
        try:
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            raise BenchmarkError(f"Invalid YAML in manifest {path}: {exc}") from exc

        if not isinstance(data, dict):
            raise BenchmarkError(f"Manifest must be a YAML mapping, got {type(data).__name__}")

        _require_keys(data, ("name", "dataset", "detectors", "metrics", "slices"), path)

        transforms: list[TransformSpec] = []
        for t in data.get("transforms", []):
            if not isinstance(t, dict) or "operator" not in t or "intensities" not in t:
                raise BenchmarkError(
                    f"Each transform must have 'operator' and 'intensities' keys in {path}"
                )
            transforms.append(
                TransformSpec(
                    operator=t["operator"],
                    intensities=tuple(float(i) for i in t["intensities"]),
                )
            )

        return cls(
            name=data["name"],
            dataset=data["dataset"],
            detectors=tuple(data["detectors"]),
            transforms=tuple(transforms),
            metrics=tuple(data["metrics"]),
            slices=tuple(data["slices"]),
            seed=int(data.get("seed", 42)),
        )

    def to_yaml(self, path: Path) -> None:
        """Serialize the manifest to a YAML file."""
        data: dict[str, object] = {
            "name": self.name,
            "dataset": self.dataset,
            "detectors": list(self.detectors),
            "transforms": [
                {"operator": t.operator, "intensities": list(t.intensities)}
                for t in self.transforms
            ],
            "metrics": list(self.metrics),
            "slices": list(self.slices),
            "seed": self.seed,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _require_keys(data: dict[str, object], keys: tuple[str, ...], path: Path) -> None:
    missing = [k for k in keys if k not in data]
    if missing:
        raise BenchmarkError(
            f"Manifest {path} missing required keys: {', '.join(missing)}"
        )
