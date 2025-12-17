from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def load_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text())
    # Support either {"records": [...]} or a plain list
    if isinstance(data, dict) and "records" in data:
        recs = data["records"]
    elif isinstance(data, list):
        recs = data
    else:
        raise ValueError("Unrecognized JSON structure: expected list or {'records': [...]}.")
    # Normalize common sentinel values
    for r in recs:
        if "pca_dim" in r and r["pca_dim"] == -1:
            r["pca_dim"] = None
        if "min_samples" in r and r["min_samples"] == -1:
            r["min_samples"] = None
    return recs


def composite_score(r: Dict[str, Any], w_noise: float, w_clusters: float, cap_clusters: int) -> float:
    sil = r.get("silhouette", float("nan"))
    noise = r.get("frac_noise", float("nan"))
    nclu = r.get("n_clusters", 0)
    # Guard NaNs: treat missing silhouette as very poor
    if not isinstance(sil, (int, float)):
        sil = float("nan")
    if not isinstance(noise, (int, float)):
        noise = float("nan")
    # Normalize cluster count to [0,1] by capping
    nclu_norm = min(max(int(nclu), 0), cap_clusters) / float(cap_clusters)
    # Higher is better: silhouette up, noise down, clusters up
    # If sil or noise is NaN, push score very low
    if sil != sil or noise != noise:  # NaN checks
        return -1e9
    return float(sil) - w_noise * float(noise) + w_clusters * nclu_norm


def rank_records(
    recs: List[Dict[str, Any]],
    primary: str,
    secondary: str,
    tertiary: str,
    use_composite: bool,
    w_noise: float,
    w_clusters: float,
    cap_clusters: int,
) -> List[Tuple[float, Dict[str, Any]]]:
    ranked: List[Tuple[float, Dict[str, Any]]] = []
    if use_composite:
        for r in recs:
            score = composite_score(r, w_noise=w_noise, w_clusters=w_clusters, cap_clusters=cap_clusters)
            ranked.append((score, r))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    # Key-based lexicographic ranking
    def key_func(r: Dict[str, Any]):
        # Map direction: silhouette desc, frac_noise asc, n_clusters desc by default
        def val(name: str):
            v = r.get(name)
            # Treat NaN silhouette as worst
            if name == "silhouette" and (v is None or not isinstance(v, (int, float)) or v != v):
                return -1e9
            return v
        k1 = val(primary)
        k2 = val(secondary)
        k3 = val(tertiary)
        # Prepare for sort: we will invert signs per metric semantics
        # silhouette: higher better -> negative used? We'll sort by tuple with signs handled below
        return k1, k2, k3

    # Build list and sort with correct directions
    # Directions: silhouette desc, frac_noise asc, n_clusters desc
    DIR = {"silhouette": -1, "frac_noise": +1, "n_clusters": -1}

    def norm_tuple(r: Dict[str, Any]):
        k1, k2, k3 = r["__k"]
        def norm(v, name):
            try:
                vv = float(v)
            except Exception:
                vv = float("nan")
            if name == "silhouette" and (vv != vv):  # NaN
                vv = -1e9
            return DIR.get(name, -1) * vv
        return (
            norm(k1, primary),
            norm(k2, secondary),
            norm(k3, tertiary),
        )

    tmp: List[Dict[str, Any]] = []
    for r in recs:
        rr = dict(r)
        rr["__k"] = key_func(r)
        tmp.append(rr)
    tmp.sort(key=norm_tuple)
    return [(0.0, {k: v for k, v in r.items() if k != "__k"}) for r in tmp]


def print_table(items: List[Dict[str, Any]], max_rows: int) -> None:
    cols = [
        "score", "pca_dim", "metric", "min_cluster_size", "min_samples",
        "cluster_selection_method", "cluster_selection_epsilon",
        "n_clusters", "frac_noise", "silhouette",
    ]
    # Header
    print("\t".join(cols))
    for r in items[:max_rows]:
        row = [
            f"{r.get('score', ''):>8.5f}" if isinstance(r.get('score', None), (int, float)) else "",
            str(r.get("pca_dim", "")),
            str(r.get("metric", "")),
            str(r.get("min_cluster_size", "")),
            str(r.get("min_samples", "")),
            str(r.get("cluster_selection_method", "")),
            str(r.get("cluster_selection_epsilon", "")),
            str(r.get("n_clusters", "")),
            f"{r.get('frac_noise', float('nan')):.6f}" if isinstance(r.get('frac_noise', None), (int, float)) else "",
            f"{r.get('silhouette', float('nan')):.6f}" if isinstance(r.get('silhouette', None), (int, float)) else "",
        ]
        print("\t".join(row))


def main():
    ap = argparse.ArgumentParser(description="Rank HDBSCAN tuning results from JSON and output top-N")
    ap.add_argument("input", type=Path, help="Path to JSON with records or {'records': [...]} ")
    ap.add_argument("--top", type=int, default=20, help="Number of top rows to print")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save top-N as JSON")
    ap.add_argument("--composite", action="store_true", help="Use composite score = silhouette - w_noise*frac_noise + w_clusters*norm(n_clusters)")
    ap.add_argument("--w-noise", type=float, default=0.5, help="Weight for frac_noise in composite score")
    ap.add_argument("--w-clusters", type=float, default=0.1, help="Weight for cluster count term in composite score")
    ap.add_argument("--cap-clusters", type=int, default=50, help="Cap for cluster count normalization in composite score")
    ap.add_argument("--primary", type=str, default="silhouette", choices=["silhouette", "frac_noise", "n_clusters"], help="Primary sort key (silhouette desc, frac_noise asc, n_clusters desc)")
    ap.add_argument("--secondary", type=str, default="frac_noise", choices=["silhouette", "frac_noise", "n_clusters"], help="Secondary sort key")
    ap.add_argument("--tertiary", type=str, default="n_clusters", choices=["silhouette", "frac_noise", "n_clusters"], help="Tertiary sort key")
    args = ap.parse_args()

    recs = load_records(args.input)

    ranked = rank_records(
        recs,
        primary=args.primary,
        secondary=args.secondary,
        tertiary=args.tertiary,
        use_composite=args.composite,
        w_noise=args.w_noise,
        w_clusters=args.w_clusters,
        cap_clusters=args.cap_clusters,
    )

    # Prepare top-N output
    top_items: List[Dict[str, Any]] = []
    for score, r in ranked[: args.top]:
        rr = dict(r)
        rr["score"] = score if args.composite else None
        top_items.append(rr)

    print_table(top_items, max_rows=args.top)

    if args.out is not None:
        payload = {"records": top_items}
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved top-{args.top} records to {args.out}")


if __name__ == "__main__":
    main()
