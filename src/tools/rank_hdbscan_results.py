from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- CLI: rank HDBSCAN grid-search outputs ---
# This script is meant to consume `grid_search_hdbscan` JSON output and print a TSV table.
# It supports both:
# - records saved as {"records": [...]} (preferred), and
# - records saved as a raw JSON list (legacy/convenience).


# Metrics that can be used for sorting
SORT_KEYS = [
    "silhouette",
    "frac_noise",
]


def _is_nan(x: Any) -> bool:
    # Fast NaN check (works for floats): NaN is the only value where x != x.
    return isinstance(x, float) and x != x


def _as_float(x: Any, default: float) -> float:
    # Best-effort float conversion for JSON payloads that may contain strings/ints/nulls.
    try:
        return float(x)
    except Exception:
        return default


def load_records(path: Path) -> List[Dict[str, Any]]:
    # Read and normalize records so downstream ranking is stable across schema versions.
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
        # pca_dim: preserve -1 sentinel; normalize legacy null->-1
        if r.get("pca_dim", None) is None:
            r["pca_dim"] = -1
        else:
            try:
                r["pca_dim"] = int(float(r["pca_dim"]))
            except Exception:
                # leave as-is
                pass

        # min_samples: keep None for readability
        if "min_samples" in r and r["min_samples"] == -1:
            r["min_samples"] = None

        # eval pca dim: preserve -1 sentinel; normalize legacy null->-1
        if "silhouette_eval_pca_dim" in r:
            if r.get("silhouette_eval_pca_dim", None) is None:
                r["silhouette_eval_pca_dim"] = -1
            else:
                try:
                    r["silhouette_eval_pca_dim"] = int(float(r["silhouette_eval_pca_dim"]))
                except Exception:
                    pass

        # Backwards-compat: if silhouette_eval missing, set it to silhouette.
        if "silhouette_eval" not in r:
            r["silhouette_eval"] = r.get("silhouette", float("nan"))

    return recs


def composite_score(r: Dict[str, Any], w_noise: float) -> float:
    """Composite score used when --composite is enabled.

    Uses silhouette (higher is better), frac_noise (lower is better).
    """
    sil_f = _as_float(r.get("silhouette", float("nan")), float("nan"))
    noise_f = _as_float(r.get("frac_noise", float("nan")), float("nan"))

    # Guard NaNs: treat missing silhouette/noise as very poor
    if _is_nan(sil_f) or _is_nan(noise_f):
        return -1e9

    return float(sil_f) - w_noise * float(noise_f)


def rank_records(
    recs: List[Dict[str, Any]],
    primary: str,
    secondary: str,
    use_composite: bool,
    w_noise: float,
) -> List[Tuple[float, Dict[str, Any]]]:
    # Return (score, record) pairs; in non-composite mode score is a placeholder.
    ranked: List[Tuple[float, Dict[str, Any]]] = []

    if use_composite:
        for r in recs:
            score = composite_score(r, w_noise=w_noise)
            ranked.append((score, r))
        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    # Directions: silhouette desc, frac_noise asc
    DIR = {
        "silhouette": -1,
        "frac_noise": +1,
    }

    def _val(r: Dict[str, Any], name: str) -> float:
        vv = _as_float(r.get(name, float("nan")), float("nan"))

        # Treat NaN silhouettes as worst
        if name == "silhouette" and _is_nan(vv):
            vv = -1e9

        # Treat NaN frac_noise as worst (high)
        if name == "frac_noise" and _is_nan(vv):
            vv = 1e9

        return vv

    def sort_key(r: Dict[str, Any]):
        # Multiply by direction so we can always sort ascending.
        k1 = DIR.get(primary, -1) * _val(r, primary)
        k2 = DIR.get(secondary, -1) * _val(r, secondary)
        return (k1, k2)

    tmp = list(recs)
    tmp.sort(key=sort_key)
    return [(0.0, r) for r in tmp]


def print_table(items: List[Dict[str, Any]], max_rows: int) -> None:
    # Print a TSV table so it can be piped into pandas/Excel easily.
    cols = [
        "score",
        "pca_dim",
        "metric",
        "min_cluster_size",
        "min_samples",
        "cluster_selection_method",
        "cluster_selection_epsilon",
        "n_clusters",
        "frac_noise",
        "silhouette",
        "silhouette_eval",
        "silhouette_eval_pca_dim",
    ]

    print("\t".join(cols))
    for r in items[:max_rows]:
        def fnum(name: str) -> str:
            v = r.get(name, None)
            if isinstance(v, (int, float)):
                try:
                    return f"{float(v):.6f}"
                except Exception:
                    return ""
            return ""

        row = [
            f"{r.get('score', ''):>8.5f}" if isinstance(r.get("score", None), (int, float)) else "",
            str(r.get("pca_dim", "")),
            str(r.get("metric", "")),
            str(r.get("min_cluster_size", "")),
            str(r.get("min_samples", "")),
            str(r.get("cluster_selection_method", "")),
            str(r.get("cluster_selection_epsilon", "")),
            str(r.get("n_clusters", "")),
            fnum("frac_noise"),
            fnum("silhouette"),
            fnum("silhouette_eval"),
            str(r.get("silhouette_eval_pca_dim", "")),
        ]
        print("\t".join(row))


def main():
    ap = argparse.ArgumentParser(description="Rank HDBSCAN tuning results from JSON and output top-N")
    ap.add_argument("input", type=Path, help="Path to JSON with records or {'records': [...]} ")
    ap.add_argument("--top", type=int, default=20, help="Number of top rows to print")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save top-N as JSON")

    ap.add_argument(
        "--composite",
        action="store_true",
        help="Use composite score = silhouette - w_noise*frac_noise",
    )
    ap.add_argument("--w-noise", type=float, default=0.5, help="Weight for frac_noise in composite score")

    ap.add_argument(
        "--primary",
        type=str,
        default="silhouette",
        choices=SORT_KEYS,
        help="Primary sort key (silhouette desc, frac_noise asc)",
    )
    ap.add_argument("--secondary", type=str, default="frac_noise", choices=SORT_KEYS, help="Secondary sort key")

    args = ap.parse_args()

    recs = load_records(args.input)

    ranked = rank_records(
        recs,
        primary=args.primary,
        secondary=args.secondary,
        use_composite=args.composite,
        w_noise=args.w_noise,
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
