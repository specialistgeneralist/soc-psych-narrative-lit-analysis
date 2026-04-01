from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

# --- CLI: rank TriMAP grid-search outputs ---
# Reads JSON produced by `grid_search_trimap` and prints a TSV table of the top-N settings.
# The JSON can be either {"records": [...]} or a raw list for convenience.

DEFAULT_METRIC = "trustworthiness@10"


def _as_float(x: Any, default: float) -> float:
    # Best-effort float conversion for JSON payloads that may contain strings/ints/nulls.
    try:
        return float(x)
    except Exception:
        return default


def _is_nan(x: Any) -> bool:
    # Fast NaN check (works for floats): NaN is the only value where x != x.
    return isinstance(x, float) and x != x


def load_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text())

    # Support either {"records": [...]} or a plain list
    if isinstance(data, dict) and "records" in data:
        recs = data["records"]
    elif isinstance(data, list):
        recs = data
    else:
        raise ValueError("Unrecognized JSON structure: expected list or {'records': [...]}.")

    if not isinstance(recs, list):
        raise ValueError("Expected 'records' to be a list.")

    return recs


def rank_records(
    recs: List[Dict[str, Any]],
    metric: str = DEFAULT_METRIC,
) -> List[Tuple[float, Dict[str, Any]]]:
    # Returns (metric_value, record) pairs sorted descending by metric.
    def metric_value(r: Dict[str, Any]) -> float:
        v = _as_float(r.get(metric, float("nan")), float("nan"))
        if _is_nan(v):
            # Missing/invalid metrics are always ranked last.
            return -1e9
        return v

    ranked = [(metric_value(r), r) for r in recs]

    # Sort by metric descending, then by params ascending for stable/readable output
    ranked.sort(
        key=lambda x: (
            -x[0],
            int(x[1].get("n_inliers", 10**9)),
            int(x[1].get("n_outliers", 10**9)),
            int(x[1].get("n_random", 10**9)),
        )
    )
    return ranked


def print_table(items: List[Dict[str, Any]], metric: str, max_rows: int) -> None:
    # Print a TSV table so it can be piped into pandas/Excel easily.
    cols = [
        "rank",
        "n_inliers",
        "n_outliers",
        "n_random",
        metric,
    ]

    print("\t".join(cols))
    for i, r in enumerate(items[:max_rows], start=1):
        metric_val = _as_float(r.get(metric, float("nan")), float("nan"))
        row = [
            str(i),
            str(r.get("n_inliers", "")),
            str(r.get("n_outliers", "")),
            str(r.get("n_random", "")),
            "" if _is_nan(metric_val) else f"{metric_val:.6f}",
        ]
        print("\t".join(row))


def main() -> None:
    ap = argparse.ArgumentParser(description="Rank TriMAP tuning results from JSON and output top-N")
    ap.add_argument("input", type=Path, help="Path to JSON with records or {'records': [...]} ")
    ap.add_argument("--metric", type=str, default=DEFAULT_METRIC, help="Metric field to rank by")
    ap.add_argument("--top", type=int, default=20, help="Number of top rows to print")
    ap.add_argument("--out", type=Path, default=None, help="Optional path to save top-N as JSON")
    args = ap.parse_args()

    recs = load_records(args.input)
    ranked = rank_records(recs, metric=args.metric)

    top_items: List[Dict[str, Any]] = []
    for score, r in ranked[: args.top]:
        rr = dict(r)
        rr["rank"] = len(top_items) + 1
        rr["score"] = score
        top_items.append(rr)

    print_table(top_items, metric=args.metric, max_rows=args.top)

    if args.out is not None:
        payload = {"records": top_items}
        args.out.write_text(json.dumps(payload, indent=2))
        print(f"\nSaved top-{args.top} records to {args.out}")


if __name__ == "__main__":
    main()
