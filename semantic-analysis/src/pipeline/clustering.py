from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Any

import numpy as np
import hdbscan
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from .config import PipelineConfig


# --- Clustering stage ---
# Wraps HDBSCAN with optional PCA preprocessing and a few simple diagnostics (noise fraction, silhouette scores).


_USE_CFG: Any = object()  # sentinel for "use cfg default"


@dataclass
class ClusterResult:
    # Output bundle for downstream logging/visualization: labels + membership probs + metrics + the fitted model.
    labels: np.ndarray
    probabilities: np.ndarray
    metrics: Dict[str, float]
    params: Dict[str, float]
    model: hdbscan.HDBSCAN


def _maybe_pca(X: np.ndarray, pca_dim: Optional[int], seed: int) -> np.ndarray:
    """Optionally reduce dimensionality with PCA; when PCA is applied we L2-normalize rows.

    The normalization keeps the projected vectors on a comparable scale and is helpful when using
    distance metrics sensitive to magnitude.
    """
    if pca_dim is None:
        return X.astype(np.float32)
    if pca_dim >= X.shape[1]:
        return X.astype(np.float32)
    pca = PCA(n_components=pca_dim, random_state=seed)
    Xr = pca.fit_transform(X)
    norms = np.linalg.norm(Xr, axis=1, keepdims=True) + 1e-12
    Xr = Xr / norms
    return Xr.astype(np.float32)


def _safe_silhouette(X: np.ndarray, labels: np.ndarray, metric: str) -> float:
    """Compute silhouette on non-noise points only; return NaN if undefined."""
    # Silhouette is undefined for a single cluster, and noise points (-1) should not be treated as a cluster.
    mask = labels != -1
    if mask.sum() <= 1:
        return float("nan")
    n_clusters = int(len(set(labels[mask])))
    if n_clusters <= 1:
        return float("nan")
    try:
        return float(silhouette_score(X[mask], labels[mask], metric=metric))
    except Exception:
        return float("nan")


def run_hdbscan(
    cfg: PipelineConfig,
    X_unit: np.ndarray,
    min_cluster_size: Optional[int] = None,
    min_samples: Optional[int] = None,
    metric: Optional[str] = None,
    cluster_selection_method: Optional[str] = None,  # 'eom' or 'leaf'
    cluster_selection_epsilon: Optional[float] = None,
    allow_single_cluster: bool = False,
    p: Optional[int] = None,  # for Minkowski metrics
    n_jobs: int = -1,
    pca_dim: Any = _USE_CFG,
    *,
    silhouette_eval_pca_dim: Any = _USE_CFG,
    silhouette_eval_metric: Any = _USE_CFG,
) -> ClusterResult:

    # --- Resolve defaults from config (unless explicitly overridden) ---
    min_cluster_size = cfg.hdbscan_min_cluster_size if min_cluster_size is None else min_cluster_size
    min_samples = cfg.hdbscan_min_samples if min_samples is None else min_samples
    selection_method = cfg.hdbscan_selection_method if cluster_selection_method is None else cluster_selection_method
    selection_epsilon = cfg.hdbscan_cluster_selection_epsilon if cluster_selection_epsilon is None else cluster_selection_epsilon
    metric = cfg.hdbscan_metric if metric is None else metric

    # Important: distinguish "not provided" (use cfg default) vs "explicit None" (disable PCA).
    if pca_dim is _USE_CFG:
        pca_dim_eff: Optional[int] = cfg.hdbscan_pca_dim
    else:
        pca_dim_eff = pca_dim  # may be None to mean "no PCA"

    # --- Build the feature space used for clustering ---
    X_proc = _maybe_pca(X_unit, pca_dim=pca_dim_eff, seed=cfg.seed)

    # --- Fit HDBSCAN ---
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
    # probabilities_ is available when prediction_data=True; keep a safe fallback for older/edge cases.
    probs = getattr(clusterer, "probabilities_", np.ones_like(labels, dtype=float))

    # --- Primary metrics (computed in the clustering space) ---
    n_noise = int((labels == -1).sum())
    n_clusters = int(len(set(labels)) - (1 if -1 in labels else 0))
    frac_noise = n_noise / len(labels)

    sil_cluster_space = _safe_silhouette(X_proc, labels, metric=metric)

    # --- Optional evaluation space for silhouette (can differ from clustering space) ---
    if silhouette_eval_pca_dim is _USE_CFG:
        eval_pca_dim = cfg.hdbscan_silhouette_eval_pca_dim
    else:
        eval_pca_dim = silhouette_eval_pca_dim  # may be None to disable eval PCA (i.e., use clustering space)

    if silhouette_eval_metric is _USE_CFG:
        eval_metric = cfg.hdbscan_silhouette_eval_metric or metric
    else:
        eval_metric = silhouette_eval_metric

    # Build eval space
    if eval_pca_dim is None:
        # No separate evaluation PCA requested: evaluate silhouette in clustering space.
        X_eval = X_proc
        eval_pca_dim_effective: Optional[int] = pca_dim_eff
    else:
        # If eval dim equals clustering dim, reuse X_proc; otherwise compute separate PCA.
        if pca_dim_eff is not None and eval_pca_dim == pca_dim_eff:
            X_eval = X_proc
        else:
            X_eval = _maybe_pca(X_unit, pca_dim=eval_pca_dim, seed=cfg.seed)
        eval_pca_dim_effective = eval_pca_dim

    sil_eval_space = _safe_silhouette(X_eval, labels, metric=eval_metric)

    metrics: Dict[str, float] = {
        "n_clusters": float(n_clusters),
        "frac_noise": float(frac_noise),
        # Backwards-compatible key: silhouette computed in clustering space.
        "silhouette": float(sil_cluster_space),
        # New keys:
        "silhouette_eval": float(sil_eval_space),
        "silhouette_eval_pca_dim": float(-1 if eval_pca_dim_effective is None else eval_pca_dim_effective),
    }

    params: Dict[str, float] = {
        "min_cluster_size": float(min_cluster_size),
        "min_samples": float(min_samples if min_samples is not None else -1),
        "metric": metric,  # type: ignore[assignment]
        "cluster_selection_method": selection_method,  # type: ignore[assignment]
        "cluster_selection_epsilon": float(selection_epsilon),
        "allow_single_cluster": float(allow_single_cluster),
        "p": float(-1 if p is None else p),
        "n_jobs": float(n_jobs),
        # Store effective pca_dim used for clustering
        "pca_dim": float(-1 if pca_dim_eff is None else pca_dim_eff),
        "silhouette_eval_metric": eval_metric,  # type: ignore[assignment]
    }

    return ClusterResult(labels=labels, probabilities=probs, metrics=metrics, params=params, model=clusterer)
