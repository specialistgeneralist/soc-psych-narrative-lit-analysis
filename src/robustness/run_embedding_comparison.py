from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import yaml

from src.pipeline.config import PipelineConfig, load_config
from src.pipeline.embeddings import load_embeddings
from src.pipeline.utils import ensure_dirs, nearest_to_centroid, package_versions, save_json
from src.robustness.metrics import LabelSeries, align_label_series, compare_labels, records_to_markdown_table


def _normalize_representative_top_k(value: Any) -> Optional[int]:
    """Normalize YAML/CLI representative top-k values.

    YAML `none` is parsed by PyYAML as the string "none", not Python None, so
    normalise the user-facing all-data aliases explicitly.
    """
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "all", "none", "null"}:
            return None
        value = normalized
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("representative_top_k_per_cluster must be a positive integer, or one of: all, none, null.") from exc
    if parsed <= 0:
        raise ValueError("representative_top_k_per_cluster must be positive, or use all/none/null for all eligible data.")
    return parsed


@dataclass
class EmbeddingComparisonSpec:
    """YAML-backed, load-only embedding comparison specification."""

    baseline_config: str = "configs/baseline.yaml"
    baseline_tag: str = "baseline"
    baseline_model_key: str = "large"
    baseline_name: str = "openai_large_baseline"

    variant_config: str = "configs/baseline_gemini.yaml"
    variant_tag: str = "baseline_gemini"
    variant_model_key: str = "google_gemini_embedding_001"
    variant_name: str = "google_gemini_baseline"

    output_dir: str = "outputs/embedding_comparison"
    reuse_existing_only: bool = True

    representative_top_k_per_cluster: Optional[int] = None
    representative_metric: str = "cosine"
    include_noise_representatives: bool = False
    write_representative_text: bool = True

    def __post_init__(self) -> None:
        self.representative_top_k_per_cluster = _normalize_representative_top_k(self.representative_top_k_per_cluster)


KEY_RESULT_COLUMNS = [
    "scope",
    "experiment",
    "baseline",
    "variant",
    "n_common",
    "baseline_n_clusters_common",
    "variant_n_clusters_common",
    "baseline_frac_noise_common",
    "variant_frac_noise_common",
    "ari",
    "ami",
    "nmi",
    "noise_status_agreement",
    "co_cluster_jaccard",
    "baseline_pair_count",
    "variant_pair_count",
    "pair_intersection",
    "pair_union",
]


def _read_spec(path: Path) -> EmbeddingComparisonSpec:
    if not path.exists():
        raise FileNotFoundError(f"Embedding comparison spec not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return EmbeddingComparisonSpec(**raw)


def _git_commit_or_unknown() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return proc.stdout.strip()
    except Exception:
        return "unknown"


def _embedding_path(cfg: PipelineConfig, model_key: str) -> Path:
    return cfg.paths.embeddings_dir / f"embeddings_{model_key}.parquet"


def _labels_path(cfg: PipelineConfig, tag: str, model_key: str) -> Path:
    return cfg.paths.results_dir / f"{tag}_labels_{model_key}.npy"


def _load_required_embeddings(cfg: PipelineConfig, model_key: str, *, role: str) -> pd.DataFrame:
    path = _embedding_path(cfg, model_key)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {role} embeddings parquet: {path}. This comparison runner is load-only and will not recompute embeddings."
        )
    df = load_embeddings(path)
    if "id" not in df.columns:
        raise ValueError(f"{role} embeddings parquet must contain an 'id' column: {path}")
    if "vector_unit" not in df.columns:
        raise ValueError(f"{role} embeddings parquet must contain a 'vector_unit' column: {path}")
    return df


def _load_required_labels(cfg: PipelineConfig, tag: str, model_key: str, ids: list[str], *, role: str) -> LabelSeries:
    path = _labels_path(cfg, tag, model_key)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {role} labels: {path}. Run the corresponding baseline clustering first; this runner is load-only."
        )
    labels = np.load(path).astype(int)
    return LabelSeries(ids=ids, labels=labels)


def _same_cluster_pairs(labels: np.ndarray) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for label in sorted(set(int(x) for x in labels)):
        if label == -1:
            continue
        idx = np.where(labels == label)[0].tolist()
        if len(idx) >= 2:
            pairs.update(combinations(idx, 2))
    return pairs


def _pair_count_metrics(baseline: LabelSeries, variant: LabelSeries) -> dict[str, int]:
    _common_ids, left_labels, right_labels = align_label_series(baseline, variant)
    left_pairs = _same_cluster_pairs(left_labels)
    right_pairs = _same_cluster_pairs(right_labels)
    union = left_pairs | right_pairs
    intersection = left_pairs & right_pairs
    return {
        "baseline_pair_count": len(left_pairs),
        "variant_pair_count": len(right_pairs),
        "pair_intersection": len(intersection),
        "pair_union": len(union),
    }


def _compare_scope(
    baseline: LabelSeries,
    variant: LabelSeries,
    *,
    scope: str,
    baseline_name: str,
    variant_name: str,
) -> dict[str, Any]:
    record = compare_labels(
        baseline,
        variant,
        baseline_name=baseline_name,
        variant_name=variant_name,
        experiment="embedding_representation_comparison",
        extra={"scope": scope},
    )
    record.update(_pair_count_metrics(baseline, variant))
    return record


def _filter_label_series(series: LabelSeries, selected_ids: list[str]) -> LabelSeries:
    by_id = series.as_dict()
    kept_ids = [item_id for item_id in selected_ids if item_id in by_id]
    if not kept_ids:
        raise ValueError("Representative filter produced no ids present in the label series")
    return LabelSeries(ids=kept_ids, labels=np.array([by_id[item_id] for item_id in kept_ids], dtype=int))


def _select_baseline_representatives(
    spec: EmbeddingComparisonSpec,
    df_baseline: pd.DataFrame,
    baseline: LabelSeries,
) -> tuple[list[str], pd.DataFrame]:
    """Select baseline items for the representative-subpopulation comparison.

    If `representative_top_k_per_cluster` is None, select all rows from the
    eligible baseline clusters. Otherwise, recompute top-k centroid-nearest
    representatives from current labels/embeddings using the same helper as the
    summarization pipeline, rather than trusting a possibly stale summaries_index.json.
    """

    X_unit = np.array(df_baseline["vector_unit"].tolist(), dtype=np.float32)
    labels = baseline.labels
    ids = df_baseline["id"].astype(str).tolist()
    texts = df_baseline["text"].astype(str).tolist() if "text" in df_baseline.columns else [""] * len(ids)
    top_k = spec.representative_top_k_per_cluster

    selected_ids: list[str] = []
    rows: list[dict[str, Any]] = []
    cluster_ids = sorted(int(c) for c in set(labels) if spec.include_noise_representatives or int(c) != -1)
    for cluster_id in cluster_ids:
        idx = np.where(labels == cluster_id)[0]
        if idx.size == 0:
            continue

        if top_k is None:
            rep_indices = idx
            selection_mode = "all_cluster_items"
        else:
            if int(top_k) <= 0:
                raise ValueError("representative_top_k_per_cluster must be a positive integer, or null/none for all data.")
            rep_indices = nearest_to_centroid(
                X_unit,
                idx,
                k=int(top_k),
                metric=spec.representative_metric,
            )
            selection_mode = "centroid_nearest_top_k"

        for rank, row_index in enumerate(rep_indices.tolist(), start=1):
            item_id = ids[row_index]
            selected_ids.append(item_id)
            row = {
                "baseline_cluster": cluster_id,
                "representative_rank": rank if top_k is not None else None,
                "selection_rank": rank,
                "selection_mode": selection_mode,
                "row_index": int(row_index),
                "id": item_id,
            }
            if spec.write_representative_text:
                row["text"] = texts[row_index]
            rows.append(row)

    # Preserve first occurrence order while guarding against accidental duplicate ids.
    deduped_ids = list(dict.fromkeys(selected_ids))
    return deduped_ids, pd.DataFrame(rows)


def _contingency_frame(baseline: LabelSeries, variant: LabelSeries, *, scope: str) -> pd.DataFrame:
    common_ids, left_labels, right_labels = align_label_series(baseline, variant)
    df = pd.DataFrame(
        {
            "id": common_ids,
            "baseline_label": left_labels,
            "variant_label": right_labels,
        }
    )
    table = (
        df.groupby(["baseline_label", "variant_label"], dropna=False)
        .size()
        .reset_index(name="n")
        .sort_values(["baseline_label", "variant_label"])
    )
    table.insert(0, "scope", scope)
    return table


def _write_records(out_dir: Path, records: list[dict[str, Any]]) -> None:
    ensure_dirs(out_dir)
    save_json({"records": records}, out_dir / "embedding_comparison_metrics.json")
    pd.DataFrame(records).to_csv(out_dir / "embedding_comparison_metrics.csv", index=False)

    md = "# Embedding representation comparison\n\n"
    md += "This report is generated by `python -m src.robustness.run_embedding_comparison`.\n\n"
    md += records_to_markdown_table(records, KEY_RESULT_COLUMNS)
    (out_dir / "embedding_comparison_report.md").write_text(md, encoding="utf-8")


def _write_manifest(
    out_dir: Path,
    spec: EmbeddingComparisonSpec,
    baseline_cfg: PipelineConfig,
    variant_cfg: PipelineConfig,
) -> None:
    manifest = {
        "spec": asdict(spec),
        "baseline_config_effective": baseline_cfg.to_dict(),
        "variant_config_effective": variant_cfg.to_dict(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit_or_unknown(),
        "package_versions": package_versions(
            [
                "pandas",
                "numpy",
                "scikit-learn",
                "pyarrow",
                "pyyaml",
            ]
        ),
    }
    save_json(manifest, out_dir / "run_manifest.json")


def _run_spec(spec: EmbeddingComparisonSpec) -> Path:
    baseline_cfg = load_config(spec.baseline_config)
    variant_cfg = load_config(spec.variant_config)
    out_dir = Path(spec.output_dir)
    ensure_dirs(out_dir)

    # If k given, append k{spec.representative_top_k_per_cluster} to the output dir for clarity when comparing multiple runs with different k.
    if spec.representative_top_k_per_cluster is not None:
        out_dir = out_dir / f"_k{spec.representative_top_k_per_cluster}"
        ensure_dirs(out_dir)
    else:
        out_dir = out_dir / "_all_data"
        ensure_dirs(out_dir)

    df_baseline = _load_required_embeddings(baseline_cfg, spec.baseline_model_key, role="baseline")
    df_variant = _load_required_embeddings(variant_cfg, spec.variant_model_key, role="variant")

    baseline_ids = df_baseline["id"].astype(str).tolist()
    variant_ids = df_variant["id"].astype(str).tolist()
    baseline = _load_required_labels(
        baseline_cfg,
        spec.baseline_tag,
        spec.baseline_model_key,
        baseline_ids,
        role="baseline",
    )
    variant = _load_required_labels(
        variant_cfg,
        spec.variant_tag,
        spec.variant_model_key,
        variant_ids,
        role="variant",
    )

    selected_ids, representative_df = _select_baseline_representatives(spec, df_baseline, baseline)
    representative_df.to_csv(out_dir / "baseline_representative_items.csv", index=False)

    baseline_rep = _filter_label_series(baseline, selected_ids)
    variant_rep = _filter_label_series(variant, selected_ids)

    records = [
        _compare_scope(
            baseline,
            variant,
            scope="all_points",
            baseline_name=spec.baseline_name,
            variant_name=spec.variant_name,
        ),
        _compare_scope(
            baseline_rep,
            variant_rep,
            scope="baseline_representatives",
            baseline_name=spec.baseline_name,
            variant_name=spec.variant_name,
        ),
    ]

    _write_records(out_dir, records)
    _contingency_frame(baseline, variant, scope="all_points").to_csv(
        out_dir / "contingency_all_points.csv", index=False
    )
    _contingency_frame(baseline_rep, variant_rep, scope="baseline_representatives").to_csv(
        out_dir / "contingency_baseline_representatives.csv", index=False
    )
    _write_manifest(out_dir, spec, baseline_cfg, variant_cfg)
    return out_dir


def run(spec_path: Path) -> Path:
    spec = _read_spec(spec_path)
    return _run_spec(spec)


def _parse_representative_top_k(value: str) -> Optional[int]:
    try:
        return _normalize_representative_top_k(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _apply_cli_overrides(spec: EmbeddingComparisonSpec, args: argparse.Namespace) -> EmbeddingComparisonSpec:
    if hasattr(args, "representative_top_k"):
        spec.representative_top_k_per_cluster = args.representative_top_k
    return spec


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load-only comparison of baseline OpenAI-large and Gemini cluster assignments."
    )
    parser.add_argument("--spec", default="configs/embedding_comparison.yaml", help="Path to embedding comparison YAML spec")
    parser.add_argument(
        "--representative-top-k",
        type=_parse_representative_top_k,
        default=argparse.SUPPRESS,
        help=(
            "Override representative_top_k_per_cluster from the YAML. "
            "Use --representative-top-k all (or none/null) for all eligible baseline-cluster items, "
            "or e.g. --representative-top-k 5 for five centroid-nearest representatives per baseline cluster."
        ),
    )
    args = parser.parse_args()

    spec = _apply_cli_overrides(_read_spec(Path(args.spec)), args)
    out_dir = _run_spec(spec)
    print(f"Embedding comparison outputs written to {out_dir}")


if __name__ == "__main__":
    main()
