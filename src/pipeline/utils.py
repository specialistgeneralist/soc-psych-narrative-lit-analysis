from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
from packaging import version

# Shared utility helpers used across pipeline stages.


def ensure_dirs(*paths: Path) -> None:
    # Convenience wrapper to create a set of directories (including parents).
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def text_hash(text: str) -> str:
    # Short stable hash used to detect text changes between runs without storing full text in filenames.
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    # Small UTC timestamp helper for metadata/logging.
    return datetime.utcnow().isoformat() + "Z"


def _json_default(o: Any):
    # Handle common non-serializable types (numpy, Path, etc.).
    # This keeps `save_json` ergonomic for metrics dicts that contain numpy scalars/arrays.
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
    # Always create parent directories so callers can treat this as "write JSON here".
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=_json_default)


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def set_global_seed(seed: int) -> None:
    # Best-effort reproducibility across common libraries.
    import random

    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # Torch is optional in this project; ignore if missing or misconfigured.
        pass


def package_versions(packages: Iterable[str]) -> Dict[str, str]:
    # Record package versions in run metadata to make results auditable/reproducible.
    vers: Dict[str, str] = {}
    for pkg in packages:
        try:
            import importlib.metadata as importlib_metadata

            vers[pkg] = importlib_metadata.version(pkg)
        except Exception:
            vers[pkg] = "unknown"
    return vers


def normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # L2-normalize each row vector. `eps` prevents division-by-zero for empty/degenerate vectors.
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return X / norms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # Cosine similarity matrix between rows of `a` and rows of `b`.
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T


def nearest_to_centroid(X: np.ndarray, indices: np.ndarray, k: int = 3, metric: str = "cosine") -> np.ndarray:
    # Select `k` representative points from a cluster by proximity to the cluster centroid.
    if indices.size == 0:
        return np.array([], dtype=int)
    cluster_X = X[indices]
    centroid = cluster_X.mean(axis=0, keepdims=True)
    if metric == "cosine":
        # For cosine, normalize vectors first so dot-product equals cosine similarity.
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
