from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
import trimap

from .config import PipelineConfig


@dataclass
class TriMAPResult:
    Y: np.ndarray  # 2D embedding (n, 2)
    metrics: Dict[str, float]
    params: Dict[str, float]


def run_trimap(
    cfg: PipelineConfig,
    X_unit: np.ndarray,
    n_inliers: int | None = None,
    n_outliers: int | None = None,
    n_random: int | None = None,
    metric: str | None = None,
    seed: int | None = None,
) -> TriMAPResult:
    n_inliers = n_inliers if n_inliers is not None else cfg.trimap_n_inliers
    n_outliers = n_outliers if n_outliers is not None else cfg.trimap_n_outliers
    n_random = n_random if n_random is not None else cfg.trimap_n_random
    metric = metric if metric is not None else cfg.metric
    seed = seed if seed is not None else cfg.seed

    # Ensure n_inliers is less than n - 1
    n_samples = X_unit.shape[0]
    if n_inliers >= n_samples - 1:
        n_inliers = n_samples - 2

    reducer = trimap.TRIMAP(
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        distance=metric,
        verbose=True,
    )

    Y = reducer.fit_transform(X_unit)

    # Goodness-of-fit metric: trustworthiness (higher is better)
    # .. ensure n_neighbors is less than n_samples / 2
    n_neighbors = min(10, (n_samples - 1) // 2)
    tw = float(trustworthiness(X_unit, Y, n_neighbors=n_neighbors, metric=metric))

    metrics = {"trustworthiness@10": tw}
    params = {
        "n_inliers": n_inliers,
        "n_outliers": n_outliers,
        "n_random": n_random,
        "metric": metric,
        "seed": seed,
    }

    return TriMAPResult(Y=Y, metrics=metrics, params=params)
