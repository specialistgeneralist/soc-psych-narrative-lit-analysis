from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from packaging import version


def ensure_dirs(*paths: Path) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def _json_default(o: Any):
    # Handle common non-serializable types (numpy, Path, etc.)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    # Fallback: try string representation
    return str(o)


def save_json(obj: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int) -> None:
    import random

    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def package_versions(packages: Iterable[str]) -> Dict[str, str]:
    vers: Dict[str, str] = {}
    for pkg in packages:
        try:
            import importlib.metadata as importlib_metadata

            vers[pkg] = importlib_metadata.version(pkg)
        except Exception:
            vers[pkg] = "unknown"
    return vers


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def nearest_to_centroid(X: np.ndarray, indices: np.ndarray, k: int = 3, metric: str = "cosine") -> np.ndarray:
    if indices.size == 0:
        return np.array([], dtype=int)
    cluster_X = X[indices]
    centroid = cluster_X.mean(axis=0, keepdims=True)
    if metric == "cosine":
        centroid = normalize_rows(centroid)
        cluster_X_n = normalize_rows(cluster_X)
        sims = (cluster_X_n @ centroid.T).ravel()
        order = np.argsort(-sims)
    else:
        # Euclidean distance
        dists = np.linalg.norm(cluster_X - centroid, axis=1)
        order = np.argsort(dists)
    k = min(k, len(order))
    return indices[order[:k]]
