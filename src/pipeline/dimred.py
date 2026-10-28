from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
import trimap

from .config import PipelineConfig


# --- Dimensionality reduction stage ---
# Runs TriMAP to obtain a 2D embedding and reports a lightweight goodness-of-fit diagnostic (trustworthiness).


@dataclass
class TriMAPResult:
    # Container returned by run_trimap(): 2D coords plus metrics/params for logging.
    Y: np.ndarray  # 2D embedding (n, 2)
    metrics: Dict[str, float]
    params: Dict[str, float]


def _ensure_unique_ids(ids: list[str]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            duplicates.add(item_id)
        seen.add(item_id)
    if duplicates:
        sample = ", ".join(sorted(duplicates)[:5])
        raise ValueError(f"Dimensionality-reduction map requires unique ids; duplicate ids include: {sample}")


def save_dimred_map(
    path: Path,
    Y: np.ndarray,
    ids: list[str],
    *,
    model_key: str,
    tag: str,
    config_path: str | None = None,
    metrics: dict[str, Any] | None = None,
    params: dict[str, Any] | None = None,
    note: str | None = None,
) -> None:
    """Persist a fixed 2D coordinate map keyed by stable item ids.

    This is intentionally a coordinate map, not a reusable TriMAP estimator. It is
    appropriate for replotting alternate cluster labels over the same topology when
    the underlying item universe is shared.
    """

    ids = [str(item_id) for item_id in ids]
    _ensure_unique_ids(ids)
    if Y.ndim != 2 or Y.shape[1] != 2:
        raise ValueError(f"Expected 2D coordinates with shape (n, 2); got {Y.shape}")
    if Y.shape[0] != len(ids):
        raise ValueError(f"Coordinate row count ({Y.shape[0]}) does not match id count ({len(ids)})")

    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema": "semantic-analysis.dimred-map.v1",
        "kind": "fixed_2d_coordinate_map",
        "ids": ids,
        "coords": Y.astype(float).tolist(),
        "model_key": str(model_key),
        "tag": str(tag),
        "config_path": config_path,
        "metrics": metrics or {},
        "params": params or {},
        "note": note
        or (
            "This file stores fixed 2D coordinates keyed by item id. It is not a reusable TriMAP transform; "
            "it should be used to compare alternate labels on a shared point layout."
        ),
    }
    import json

    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_dimred_map(path: Path, target_ids: list[str]) -> tuple[np.ndarray, dict[str, Any]]:
    """Load and align a fixed coordinate map to `target_ids`.

    The returned coordinate array preserves the order of `target_ids`. Extra ids in
    the map are ignored; missing target ids are an error because the plot would be
    partially undefined.
    """

    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "semantic-analysis.dimred-map.v1":
        raise ValueError(f"Unsupported dimensionality-reduction map schema in {path}: {payload.get('schema')}")

    map_ids = [str(item_id) for item_id in payload.get("ids", [])]
    _ensure_unique_ids(map_ids)
    coords = np.array(payload.get("coords", []), dtype=np.float32)
    if coords.ndim != 2 or coords.shape[1] != 2 or coords.shape[0] != len(map_ids):
        raise ValueError(f"Invalid coordinate payload in {path}: coords shape {coords.shape}, ids {len(map_ids)}")

    target_ids = [str(item_id) for item_id in target_ids]
    _ensure_unique_ids(target_ids)
    by_id = {item_id: coords[i] for i, item_id in enumerate(map_ids)}
    missing = [item_id for item_id in target_ids if item_id not in by_id]
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"Dimensionality-reduction map {path} is missing {len(missing)} target ids, including: {sample}")

    aligned = np.vstack([by_id[item_id] for item_id in target_ids]).astype(np.float32)
    return aligned, payload


def run_trimap(
    cfg: PipelineConfig,
    X_unit: np.ndarray,
    n_inliers: int | None = None,
    n_outliers: int | None = None,
    n_random: int | None = None,
    n_iters: int | None = None,
    metric: str | None = None,
    seed: int | None = None,
) -> TriMAPResult:
    # Resolve TriMAP hyperparameters from the config unless explicitly overridden.
    n_inliers = n_inliers if n_inliers is not None else cfg.trimap_n_inliers
    n_outliers = n_outliers if n_outliers is not None else cfg.trimap_n_outliers
    n_random = n_random if n_random is not None else cfg.trimap_n_random
    n_iters = n_iters if n_iters is not None else cfg.trimap_n_iters
    metric = metric if metric is not None else cfg.metric
    seed = seed if seed is not None else cfg.seed

    # Ensure n_inliers is less than n - 1
    n_samples = X_unit.shape[0]
    if n_inliers >= n_samples - 1:
        # TriMAP needs at least two other points for inlier selection.
        n_inliers = n_samples - 2

    # Precompute distance matrix to avoid annoy estimation.
    D = pairwise_distances(X_unit, metric=metric)

    # Note: the trimap library doesn't expose a seed parameter here; reproducibility is typically handled
    # by setting global RNG state upstream (see set_global_seed in utils.py).
    reducer = trimap.TRIMAP(
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        n_iters=n_iters,
        use_dist_matrix=True,
        verbose=True,
    )

    Y = reducer.fit_transform(D)

    # Goodness-of-fit metric: trustworthiness (higher is better).
    # trustworthiness requires n_neighbors < n_samples, and behaves best with a modest neighborhood size.
    n_neighbors = min(10, (n_samples - 1) // 2)
    tw = float(trustworthiness(X_unit, Y, n_neighbors=n_neighbors, metric=metric))

    metrics = {"trustworthiness@10": tw}
    params = {
        "n_inliers": n_inliers,
        "n_outliers": n_outliers,
        "n_random": n_random,
        "n_iters": n_iters,
        "metric": metric,
        "seed": seed,
    }

    return TriMAPResult(Y=Y, metrics=metrics, params=params)
