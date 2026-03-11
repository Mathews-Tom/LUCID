"""Tests for experiment manifest parsing."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from lucid.bench.manifests import ExperimentManifest, TransformSpec
from lucid.core.errors import BenchmarkError


def _minimal_manifest_data() -> dict:
    return {
        "name": "test_experiment",
        "dataset": "data/test.jsonl",
        "detectors": ["roberta"],
        "transforms": [
            {"operator": "surface_edit", "intensities": [0.3, 0.7]},
        ],
        "metrics": ["embedding_cosine"],
        "slices": ["domain"],
        "seed": 123,
    }


class TestFromYaml:
    def test_parse_valid(self, tmp_path: Path) -> None:
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(_minimal_manifest_data()), encoding="utf-8")

        manifest = ExperimentManifest.from_yaml(path)
        assert manifest.name == "test_experiment"
        assert manifest.dataset == "data/test.jsonl"
        assert manifest.detectors == ("roberta",)
        assert len(manifest.transforms) == 1
        assert manifest.transforms[0].operator == "surface_edit"
        assert manifest.transforms[0].intensities == (0.3, 0.7)
        assert manifest.metrics == ("embedding_cosine",)
        assert manifest.slices == ("domain",)
        assert manifest.seed == 123

    def test_default_seed(self, tmp_path: Path) -> None:
        data = _minimal_manifest_data()
        del data["seed"]
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")

        manifest = ExperimentManifest.from_yaml(path)
        assert manifest.seed == 42

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(BenchmarkError, match="not found"):
            ExperimentManifest.from_yaml(tmp_path / "missing.yaml")

    def test_missing_required_key_raises(self, tmp_path: Path) -> None:
        data = _minimal_manifest_data()
        del data["name"]
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")

        with pytest.raises(BenchmarkError, match="missing required keys.*name"):
            ExperimentManifest.from_yaml(path)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("{{invalid yaml", encoding="utf-8")

        with pytest.raises(BenchmarkError, match="Invalid YAML"):
            ExperimentManifest.from_yaml(path)

    def test_non_mapping_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "list.yaml"
        path.write_text("- item1\n- item2\n", encoding="utf-8")

        with pytest.raises(BenchmarkError, match="must be a YAML mapping"):
            ExperimentManifest.from_yaml(path)

    def test_bad_transform_raises(self, tmp_path: Path) -> None:
        data = _minimal_manifest_data()
        data["transforms"] = [{"operator": "edit"}]  # missing intensities
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")

        with pytest.raises(BenchmarkError, match="'operator' and 'intensities'"):
            ExperimentManifest.from_yaml(path)

    def test_no_transforms_ok(self, tmp_path: Path) -> None:
        data = _minimal_manifest_data()
        del data["transforms"]
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")

        manifest = ExperimentManifest.from_yaml(path)
        assert manifest.transforms == ()


class TestToYaml:
    def test_round_trip(self, tmp_path: Path) -> None:
        manifest = ExperimentManifest(
            name="roundtrip",
            dataset="data.jsonl",
            detectors=("det1", "det2"),
            transforms=(TransformSpec(operator="op1", intensities=(0.5,)),),
            metrics=("m1",),
            slices=("domain",),
            seed=99,
        )
        path = tmp_path / "out.yaml"
        manifest.to_yaml(path)

        loaded = ExperimentManifest.from_yaml(path)
        assert loaded.name == manifest.name
        assert loaded.detectors == manifest.detectors
        assert loaded.transforms[0].operator == "op1"
        assert loaded.seed == 99
