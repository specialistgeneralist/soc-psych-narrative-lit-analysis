from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np

from .config import PipelineConfig
from .dimred import run_trimap
from .clustering import run_hdbscan


@dataclass
class ParamSearchResult:
    records: List[Dict]


def grid_search_trimap(
    cfg: PipelineConfig,
    X_unit: np.ndarray,
    n_inliers_grid: Iterable[int] = (5, 10, 15),
    n_outliers_grid: Iterable[int] = (3, 5, 10),
    n_random_grid: Iterable[int] = (3, 5, 10),
) -> ParamSearchResult:
    recs: List[Dict] = []
    for ni in n_inliers_grid:
        for no in n_outliers_grid:
            for nr in n_random_grid:
                res = run_trimap(cfg, X_unit, n_inliers=ni, n_outliers=no, n_random=nr)
                recs.append(
                    {
                        "n_inliers": ni,
                        "n_outliers": no,
                        "n_random": nr,
                        **res.metrics,
                    }
                )
    return ParamSearchResult(records=recs)


def grid_search_hdbscan(
    cfg: PipelineConfig,
    X_unit: np.ndarray,
    min_cluster_size_grid: Iterable[int] | None = None,
    min_samples_grid: Iterable[int | None] | None = None,
    cluster_selection_method_grid: Iterable[str] | None = None,
    cluster_selection_epsilon_grid: Iterable[float] | None = None,
    allow_single_cluster_grid: Iterable[bool] | None = None,
    metric_grid: Iterable[str] | None = None,
    pca_dim_grid: Iterable[int | None] | None = None,
) -> ParamSearchResult:
    # Use grids from config if not provided
    if min_cluster_size_grid is None:
        min_cluster_size_grid = cfg.hdbscan_grid_min_cluster_size
    if min_samples_grid is None:
        min_samples_grid = cfg.hdbscan_grid_min_samples
    if cluster_selection_method_grid is None:
        cluster_selection_method_grid = cfg.hdbscan_grid_selection_method
    if cluster_selection_epsilon_grid is None:
        cluster_selection_epsilon_grid = cfg.hdbscan_grid_selection_epsilon
    if allow_single_cluster_grid is None:
        allow_single_cluster_grid = cfg.hdbscan_grid_allow_single_cluster
    if metric_grid is None:
        metric_grid = cfg.hdbscan_grid_metric
    if pca_dim_grid is None:
        pca_dim_grid = cfg.hdbscan_grid_pca_dim

    recs: List[Dict] = []
    for pca_dim in pca_dim_grid:
        for metric in metric_grid:
            for mcs in min_cluster_size_grid:
                for ms in min_samples_grid:
                    for csm in cluster_selection_method_grid:
                        for eps in cluster_selection_epsilon_grid:
                            for asc in allow_single_cluster_grid:
                                res = run_hdbscan(
                                    cfg,
                                    X_unit,
                                    min_cluster_size=mcs,
                                    min_samples=ms,
                                    metric=metric,
                                    cluster_selection_method=csm,
                                    cluster_selection_epsilon=eps,
                                    allow_single_cluster=asc,
                                    pca_dim=pca_dim,
                                )
                                recs.append(
                                    {
                                        "pca_dim": -1 if pca_dim is None else pca_dim,
                                        "metric": metric,
                                        "min_cluster_size": mcs,
                                        "min_samples": -1 if ms is None else ms,
                                        "cluster_selection_method": csm,
                                        "cluster_selection_epsilon": eps,
                                        "allow_single_cluster": asc,
                                        **res.metrics,
                                    }
                                )
    return ParamSearchResult(records=recs)
