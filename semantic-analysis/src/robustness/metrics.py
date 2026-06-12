from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, normalized_mutual_info_score


@dataclass(frozen=True)
class LabelSeries:
    """A label vector aligned with stable item ids."""

    ids: list[str]
    labels: np.ndarray

    def __post_init__(self) -> None:
        if len(self.ids) != int(self.labels.shape[0]):
            raise ValueError("ids and labels must have the same length")

    def as_dict(self) -> dict[str, int]:
        return {str(i): int(l) for i, l in zip(self.ids, self.labels)}


def align_label_series(left: LabelSeries, right: LabelSeries) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Align two label vectors on their common ids, preserving left-side id order."""

    right_by_id = right.as_dict()
    common_ids = [item_id for item_id in left.ids if item_id in right_by_id]
    if not common_ids:
        raise ValueError("No common ids between label series")

    left_by_id = left.as_dict()
    left_labels = np.array([left_by_id[item_id] for item_id in common_ids], dtype=int)
    right_labels = np.array([right_by_id[item_id] for item_id in common_ids], dtype=int)
    return common_ids, left_labels, right_labels


def _same_cluster_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    """Return index-pairs that are co-clustered in non-noise clusters."""

    pairs: set[tuple[int, int]] = set()
    for label in sorted(set(int(x) for x in labels)):
        if label == -1:
            continue
        idx = np.where(labels == label)[0].tolist()
        if len(idx) < 2:
            continue
        pairs.update(combinations(idx, 2))
    return pairs


def co_cluster_jaccard(left_labels: np.ndarray, right_labels: np.ndarray) -> float:
    """Jaccard similarity between non-noise co-cluster pair sets.

    Noise-labelled observations do not form co-cluster pairs. If both assignments have no
    non-noise co-cluster pairs, return 1.0 because they agree on the absence of pairwise
    cluster structure over the aligned universe.
    """

    left_pairs = _same_cluster_pairs(left_labels)
    right_pairs = _same_cluster_pairs(right_labels)
    union = left_pairs | right_pairs
    if not union:
        return 1.0
    return len(left_pairs & right_pairs) / len(union)


def noise_status_agreement(left_labels: np.ndarray, right_labels: np.ndarray) -> float:
    """Fraction of aligned items with the same noise/non-noise status."""

    if len(left_labels) == 0:
        return float("nan")
    return float(np.mean((left_labels == -1) == (right_labels == -1)))


def compare_labels(
    baseline: LabelSeries,
    variant: LabelSeries,
    *,
    baseline_name: str = "baseline",
    variant_name: str = "variant",
    experiment: str = "unknown",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute the standard robustness metrics for two cluster assignments."""

    common_ids, left_labels, right_labels = align_label_series(baseline, variant)
    result: dict[str, Any] = {
        "experiment": experiment,
        "baseline": baseline_name,
        "variant": variant_name,
        "n_common": len(common_ids),
        "n_baseline_total": len(baseline.ids),
        "n_variant_total": len(variant.ids),
        "baseline_frac_noise_common": float(np.mean(left_labels == -1)),
        "variant_frac_noise_common": float(np.mean(right_labels == -1)),
        "baseline_n_clusters_common": int(len(set(left_labels)) - (1 if -1 in left_labels else 0)),
        "variant_n_clusters_common": int(len(set(right_labels)) - (1 if -1 in right_labels else 0)),
        "ari": float(adjusted_rand_score(left_labels, right_labels)),
        "ami": float(adjusted_mutual_info_score(left_labels, right_labels)),
        "nmi": float(normalized_mutual_info_score(left_labels, right_labels)),
        "noise_status_agreement": noise_status_agreement(left_labels, right_labels),
        "co_cluster_jaccard": co_cluster_jaccard(left_labels, right_labels),
    }
    if extra:
        result.update(extra)
    return result


def records_to_markdown_table(records: Iterable[dict[str, Any]], columns: list[str]) -> str:
    """Render selected record columns as a compact Markdown table."""

    rows = list(records)
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    body: list[str] = []
    for row in rows:
        vals: list[str] = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}")
            else:
                vals.append(str(val))
        body.append("| " + " | ".join(vals) + " |")
    return "\n".join([header, sep, *body]) + "\n"
