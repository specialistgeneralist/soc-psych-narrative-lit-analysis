from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.manifold import trustworthiness
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

    # Note: the trimap library doesn't expose a seed parameter here; reproducibility is typically handled
    # by setting global RNG state upstream (see set_global_seed in utils.py).
    reducer = trimap.TRIMAP(
        n_inliers=n_inliers,
        n_outliers=n_outliers,
        n_random=n_random,
        n_iters=n_iters,
        distance=metric,
        verbose=True,
    )

    Y = reducer.fit_transform(X_unit)

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
