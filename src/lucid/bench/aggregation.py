"""Metric aggregation per slice for benchmark evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from lucid.core.types import DetectionRecord, SampleRecord
from lucid.bench.slices import SliceKey, group_by_slice

_AI_SOURCE_CLASSES = frozenset({"ai_raw", "ai_edited_light", "ai_edited_heavy"})


@dataclass(frozen=True, slots=True)
class AggregatedMetrics:
    """Aggregated benchmark metrics for a single slice."""

    slice_key: SliceKey
    n_samples: int
    auroc: float | None
    auprc: float | None
    tpr_at_fpr5: float | None
    human_fpr: float | None
    mean_score_human: float | None
    mean_score_ai: float | None
    calibration_error: float | None


class SliceAggregator:
    """Compute benchmark metrics across detection records grouped by slice."""

    def aggregate(
        self,
        detections: list[DetectionRecord],
        samples: dict[str, SampleRecord],
        slices: Sequence[str],
    ) -> list[AggregatedMetrics]:
        """Aggregate metrics for each requested slice dimension."""
        results: list[AggregatedMetrics] = []

        # Overall metrics
        results.append(self._compute_slice_metrics(
            SliceKey(dimension="overall", value="all"),
            detections,
            samples,
        ))

        for dimension in slices:
            groups = group_by_slice(detections, samples, dimension)
            for value, group_records in sorted(groups.items()):
                results.append(self._compute_slice_metrics(
                    SliceKey(dimension=dimension, value=value),
                    group_records,
                    samples,
                ))

        return results

    def _compute_slice_metrics(
        self,
        slice_key: SliceKey,
        detections: list[DetectionRecord],
        samples: dict[str, SampleRecord],
    ) -> AggregatedMetrics:
        """Compute all metrics for a single slice."""
        labels: list[int] = []
        scores: list[float] = []
        human_scores: list[float] = []
        ai_scores: list[float] = []

        for det in detections:
            sample = samples.get(det.sample_id)
            if sample is None:
                continue
            label = 1 if sample.source_class in _AI_SOURCE_CLASSES else 0
            labels.append(label)
            scores.append(det.score)
            if label == 0:
                human_scores.append(det.score)
            else:
                ai_scores.append(det.score)

        auroc = self.compute_auroc(labels, scores)
        auprc = self.compute_auprc(labels, scores)
        tpr_at_fpr5 = self.compute_tpr_at_fpr(labels, scores, target_fpr=0.05)
        ece = self.compute_ece(labels, scores)

        # Human FPR: fraction of human samples scored above 0.5
        human_fpr: float | None = None
        if human_scores:
            human_fpr = sum(1 for s in human_scores if s > 0.5) / len(human_scores)

        mean_human = float(np.mean(human_scores)) if human_scores else None
        mean_ai = float(np.mean(ai_scores)) if ai_scores else None

        return AggregatedMetrics(
            slice_key=slice_key,
            n_samples=len(labels),
            auroc=auroc,
            auprc=auprc,
            tpr_at_fpr5=tpr_at_fpr5,
            human_fpr=human_fpr,
            mean_score_human=mean_human,
            mean_score_ai=mean_ai,
            calibration_error=ece,
        )

    @staticmethod
    def compute_auroc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
        """Compute Area Under ROC Curve. Returns None if fewer than 2 classes."""
        unique = set(labels)
        if len(unique) < 2:
            return None
        return float(roc_auc_score(labels, scores))

    @staticmethod
    def compute_auprc(labels: Sequence[int], scores: Sequence[float]) -> float | None:
        """Compute Area Under Precision-Recall Curve. Returns None if fewer than 2 classes."""
        unique = set(labels)
        if len(unique) < 2:
            return None
        return float(average_precision_score(labels, scores))

    @staticmethod
    def compute_tpr_at_fpr(
        labels: Sequence[int],
        scores: Sequence[float],
        target_fpr: float = 0.05,
    ) -> float | None:
        """Compute TPR at a given FPR threshold via interpolation.

        Returns None if fewer than 2 classes.
        """
        unique = set(labels)
        if len(unique) < 2:
            return None
        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        return float(np.interp(target_fpr, fpr_arr, tpr_arr))

    @staticmethod
    def compute_ece(
        labels: Sequence[int],
        confidences: Sequence[float],
        n_bins: int = 10,
    ) -> float | None:
        """Compute Expected Calibration Error.

        Bins predictions by confidence and computes weighted average
        of |accuracy - confidence| per bin.
        Returns None if no samples.
        """
        if not labels:
            return None

        labels_arr = np.array(labels, dtype=np.float64)
        conf_arr = np.array(confidences, dtype=np.float64)
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        ece = 0.0
        total = len(labels_arr)

        for i in range(n_bins):
            low, high = bin_edges[i], bin_edges[i + 1]
            if i == n_bins - 1:
                mask = (conf_arr >= low) & (conf_arr <= high)
            else:
                mask = (conf_arr >= low) & (conf_arr < high)
            bin_size = int(mask.sum())
            if bin_size == 0:
                continue
            bin_acc = float(labels_arr[mask].mean())
            bin_conf = float(conf_arr[mask].mean())
            ece += (bin_size / total) * abs(bin_acc - bin_conf)

        return ece
