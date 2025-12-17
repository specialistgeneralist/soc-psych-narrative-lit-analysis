from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from .config import PipelineConfig


@dataclass
class ClusterResult:
    labels: np.ndarray
    probabilities: np.ndarray
    metrics: Dict[str, float]
    params: Dict[str, float]
    model: hdbscan.HDBSCAN


def _maybe_pca(X: np.ndarray, pca_dim: Optional[int], seed: int) -> np.ndarray:
    if pca_dim is None:
        return X
    if pca_dim >= X.shape[1]:
        return X
    pca = PCA(n_components=pca_dim, random_state=seed)
    Xr = pca.fit_transform(X)
    # L2-normalize again to keep distances well-scaled
    norms = np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12
    Xr = Xr / norms
    return Xr.astype(np.float32)


def run_hdbscan(
    cfg: PipelineConfig,
    X_unit: np.ndarray,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    metric: Optional[str] = None,
    cluster_selection_method: Optional[str] = None,  # 'eom' or 'leaf'
    cluster_selection_epsilon: float = 0.0,
    allow_single_cluster: bool = False,
    p: Optional[int] = None,  # for Minkowski metrics
    n_jobs: int = -1,
    pca_dim: Optional[int] = None,
) -> ClusterResult:
    
    # Set up from config
    min_cluster_size = cfg.hdbscan_min_cluster_size if min_cluster_size is None else min_cluster_size
    min_samples =  cfg.hdbscan_min_samples if min_samples is None else min_samples
    pca_dim = cfg.hdbscan_pca_dim if pca_dim is None else pca_dim
    selection_method =cfg.hdbscan_selection_method if cluster_selection_method is None else "eom" 
    selection_epsilon = cfg.hdbscan_cluster_selection_epsilon if cluster_selection_epsilon is None else cluster_selection_epsilon
    metric = cfg.hdbscan_metric if metric is None else metric

    # PCA (optional)
    X_proc = _maybe_pca(X_unit, pca_dim=pca_dim, seed=cfg.seed)

    kwargs = dict(
        min_cluster_size=min_cluster_size,
        metric=metric,
        cluster_selection_method=selection_method,
        prediction_data=True,
        core_dist_n_jobs=n_jobs,
        gen_min_span_tree=True,
        cluster_selection_epsilon=selection_epsilon,
        allow_single_cluster=allow_single_cluster,
    )
    if min_samples is not None:
        kwargs["min_samples"] = min_samples
    if p is not None:
        kwargs["p"] = p

    clusterer = hdbscan.HDBSCAN(**kwargs)

    labels = clusterer.fit_predict(X_proc)
    probs = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))

    # Metrics
    n_noise = int((labels == -1).sum())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    frac_noise = n_noise / len(labels)

    # Silhouette on labeled points only (if more than 1 cluster)
    mask = labels != -1
    if mask.sum() > 1 and n_clusters > 1:
        try:
            sil = float(silhouette_score(X_proc[mask], labels[mask], metric=metric))
        except Exception:
            sil = float("nan")
    else:
        sil = float("nan")

    metrics = {
        "n_clusters": n_clusters,
        "frac_noise": frac_noise,
        "silhouette": sil,
    }
    params = {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples if min_samples is not None else -1,
        "metric": metric,
        "cluster_selection_method": selection_method,
        "cluster_selection_epsilon": cluster_selection_epsilon,
        "allow_single_cluster": allow_single_cluster,
        "p": -1 if p is None else p,
        "n_jobs": n_jobs,
        "pca_dim": -1 if pca_dim is None else pca_dim,
    }

    return ClusterResult(labels=labels, probabilities=probs, metrics=metrics, params=params, model=clusterer)
