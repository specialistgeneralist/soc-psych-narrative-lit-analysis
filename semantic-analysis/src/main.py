from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.pipeline.config import PipelineConfig, load_config
from src.pipeline.utils import ensure_dirs, load_json, package_versions, save_json, set_global_seed
from src.pipeline.embeddings import compute_and_store_embeddings, load_embeddings
from src.pipeline.dimred import run_trimap
from src.pipeline.clustering import run_hdbscan
from src.pipeline.parameter_search import grid_search_hdbscan, grid_search_trimap
from src.pipeline.visualize import plot_embedding
from src.pipeline.summarize import summarize_clusters


def read_input_tsv(path: Path, text_column: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    assert text_column in df.columns, f"Column '{text_column}' not found in {path}"
    return df


def stage_embeddings(tag: str, cfg: PipelineConfig, df: pd.DataFrame, which: List[str]) -> Dict[str, Path]:
    paths: Dict[str, Path] = {}
    for key in which:
        p = compute_and_store_embeddings(tag,cfg, df, model_key=key, overwrite=True)
        paths[key] = p
    return paths


def stage_dimred(cfg: PipelineConfig, df_emb: pd.DataFrame) -> Dict:
    X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
    res = run_trimap(cfg, X_unit)
    return {"Y": res.Y, "metrics": res.metrics, "params": res.params}


def stage_clustering(cfg: PipelineConfig, df_emb: pd.DataFrame) -> Dict:
    X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
    res = run_hdbscan(cfg, X_unit)
    return {"labels": res.labels, "probs": res.probabilities, "metrics": res.metrics, "params": res.params}

def _resolve_existing_key_for_visualize(cfg: PipelineConfig, results_dir: Path, tag: str, preferred_order: List[str]) -> Optional[str]:
    """Pick a model key ('large' or 'small') that has all required artifacts for visualization.

    We require: embeddings parquet, coords npy, labels npy.
    """

    # Try in preferred order first
    for key in preferred_order:
        coords_path = results_dir / f"{tag}_coords_{key}.npy"
        labels_path = results_dir / f"{tag}_labels_{key}.npy"
        emb_path = cfg.paths.embeddings_dir / f"embeddings_{key}.parquet"
        if coords_path.exists() and labels_path.exists() and emb_path.exists():
            return key

    # Fallback: try both regardless of preference
    for key in ["large", "small"]:
        if key in preferred_order:
            continue  # already checked
        coords_path = results_dir / f"{tag}_coords_{key}.npy"
        labels_path = results_dir / f"{tag}_labels_{key}.npy"
        emb_path = cfg.paths.embeddings_dir / f"embeddings_{key}.parquet"
        if coords_path.exists() and labels_path.exists() and emb_path.exists():
            return key

    return None


def _load_experiment(path: Path) -> Dict[str, Any]:
    exp = load_json(path)
    if not isinstance(exp, dict):
        raise ValueError(f"Experiment file must be a JSON object at top-level: {path}")
    return exp


def _apply_overrides(cfg: PipelineConfig, exp: Dict[str, Any]) -> Tuple[PipelineConfig, Dict[str, Any]]:
    """Apply experiment overrides to a config instance.

    Supported schema (recommended):
        {
          "tag": "exp_1",
          "seed": 42,
          "trimap": {"n_inliers": 3, "n_outliers": 10, "n_random": 5, "metric": "euclidean"},
          "hdbscan": {
            "min_cluster_size": 10,
            "min_samples": 2,
            "pca_dim": 5,
            "metric": "manhattan",
            "selection_method": "leaf",
            "cluster_selection_epsilon": 0.0
          }
        }

    Notes:
    - TriMAP metric overrides cfg.metric (the global distance used for dimred).
    - HDBSCAN metric overrides cfg.hdbscan_metric (clustering distance).
    """

    applied: Dict[str, Any] = {}

    if "seed" in exp and exp["seed"] is not None:
        cfg.seed = int(exp["seed"])
        applied["seed"] = cfg.seed

    trimap = exp.get("trimap")
    if isinstance(trimap, dict):
        if "n_inliers" in trimap and trimap["n_inliers"] is not None:
            cfg.trimap_n_inliers = int(trimap["n_inliers"])
            applied.setdefault("trimap", {})["n_inliers"] = cfg.trimap_n_inliers
        if "n_outliers" in trimap and trimap["n_outliers"] is not None:
            cfg.trimap_n_outliers = int(trimap["n_outliers"])
            applied.setdefault("trimap", {})["n_outliers"] = cfg.trimap_n_outliers
        if "n_random" in trimap and trimap["n_random"] is not None:
            cfg.trimap_n_random = int(trimap["n_random"])
            applied.setdefault("trimap", {})["n_random"] = cfg.trimap_n_random
        if "metric" in trimap and trimap["metric"] is not None:
            cfg.metric = str(trimap["metric"])
            applied.setdefault("trimap", {})["metric"] = cfg.metric

    hdb = exp.get("hdbscan")
    if isinstance(hdb, dict):
        if "min_cluster_size" in hdb and hdb["min_cluster_size"] is not None:
            cfg.hdbscan_min_cluster_size = int(hdb["min_cluster_size"])
            applied.setdefault("hdbscan", {})["min_cluster_size"] = cfg.hdbscan_min_cluster_size
        if "min_samples" in hdb:
            # allow explicit null in json to mean "None"
            cfg.hdbscan_min_samples = None if hdb["min_samples"] is None else int(hdb["min_samples"])
            applied.setdefault("hdbscan", {})["min_samples"] = cfg.hdbscan_min_samples
        if "pca_dim" in hdb:
            cfg.hdbscan_pca_dim = None if hdb["pca_dim"] is None else int(hdb["pca_dim"])
            applied.setdefault("hdbscan", {})["pca_dim"] = cfg.hdbscan_pca_dim
        if "metric" in hdb and hdb["metric"] is not None:
            cfg.hdbscan_metric = str(hdb["metric"])
            applied.setdefault("hdbscan", {})["metric"] = cfg.hdbscan_metric
        if "selection_method" in hdb and hdb["selection_method"] is not None:
            cfg.hdbscan_selection_method = str(hdb["selection_method"])
            applied.setdefault("hdbscan", {})["selection_method"] = cfg.hdbscan_selection_method
        if "cluster_selection_epsilon" in hdb and hdb["cluster_selection_epsilon"] is not None:
            cfg.hdbscan_cluster_selection_epsilon = float(hdb["cluster_selection_epsilon"])
            applied.setdefault("hdbscan", {})["cluster_selection_epsilon"] = cfg.hdbscan_cluster_selection_epsilon

    # Optional: allow overriding summarization selection breadth
    if "top_k_per_cluster" in exp and exp["top_k_per_cluster"] is not None:
        cfg.top_k_per_cluster = int(exp["top_k_per_cluster"])
        applied["top_k_per_cluster"] = cfg.top_k_per_cluster

    return cfg, applied


def main():
    parser = argparse.ArgumentParser(description="Semantic clustering and synthesis pipeline")
    parser.add_argument("--env", type=str, default=None, help="Path to .env (optional)")
    parser.add_argument("--input", type=str, default=None, help="Override input TSV path")
    parser.add_argument("--text-col", type=str, default=None, help="Text column name")
    parser.add_argument("--id-col", type=str, default=None, help="ID column name (optional)")

    parser.add_argument("--models", type=str, default="small,large", help="Which embeddings to compute or load: small,large or both")

    # NEW: experiment / parameter override file
    parser.add_argument(
        "--exp",
        "--params",
        dest="exp",
        type=str,
        default=None,
        help=(
            "Path to a JSON experiment file that overrides TriMAP/HDBSCAN params. "
            "This lets you run multiple near-parameter variants and compare outputs."
        ),
    )

    # Stage toggles
    parser.add_argument(
        "--do",
        type=str,
        default="all",
        help=(
            "Stages to run: embeddings,dimred,cluster,summarize,visualize (or 'visualise'),all. "
            "Use 'visualize' to render plots from existing coords/labels without recomputing."
        ),
    )

    # Tuning toggles
    parser.add_argument("--tune-trimap", action="store_true", help="Grid search TriMAP params and save report")
    parser.add_argument("--tune-hdbscan", action="store_true", help="Grid search HDBSCAN params and save report")

    # Visualization and outputs
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help=(
            "Tag for output filenames. If omitted and --exp is provided, we use exp['tag'] if present; "
            "otherwise default to 'run'."
        ),
    )

    args = parser.parse_args()

    cfg = load_config(args.env)

    # CLI overrides first
    if args.input:
        cfg.paths.input_tsv = Path(args.input)
    if args.text_col:
        cfg.data.text_column = args.text_col
    if args.id_col is not None:
        cfg.data.id_column = args.id_col if args.id_col != "" else None

    # Experiment overrides
    exp_path: Optional[Path] = Path(args.exp) if args.exp else None
    exp: Optional[Dict[str, Any]] = None
    applied_overrides: Dict[str, Any] = {}

    if exp_path is not None:
        exp = _load_experiment(exp_path)
        cfg, applied_overrides = _apply_overrides(cfg, exp)

    # Resolve tag (prefer CLI, then experiment file, then default)
    tag = args.tag or ((exp or {}).get("tag") if exp else None) or "run"

    # Seeding after all overrides
    set_global_seed(cfg.seed)

    ensure_dirs(
        cfg.paths.output_dir,
        cfg.paths.embeddings_dir,
        cfg.paths.plots_dir,
        cfg.paths.results_dir,
        cfg.paths.summaries_dir,
    )

    df_input = read_input_tsv(cfg.paths.input_tsv, cfg.data.text_column)

    selected_models = [m.strip() for m in args.models.split(",") if m.strip() in ("small", "large")]
    if not selected_models:
        selected_models = ["small"]

    # Stage selection (support British spelling 'visualise' as well)
    run_embeddings = run_dimred = run_cluster = run_summarize = run_visualize = False
    if args.do == "all":
        run_embeddings = run_dimred = run_cluster = run_summarize = True
        # visualize happens implicitly if both dimred+cluster are run
    else:
        stages = {s.strip().lower() for s in args.do.split(",")}
        run_embeddings = "embeddings" in stages
        run_dimred = "dimred" in stages
        run_cluster = "cluster" in stages
        run_summarize = "summarize" in stages
        run_visualize = ("visualize" in stages) or ("visualise" in stages)

    meta: Dict[str, Any] = {
        "config": cfg.to_dict(),
        "experiment": {
            "path": str(exp_path) if exp_path else None,
            "raw": exp,
            "applied_overrides": applied_overrides,
        },
        "package_versions": package_versions(
            [
                "pandas",
                "numpy",
                "openai",
                "anthropic",
                "scikit-learn",
                "hdbscan",
                "trimap",
                "matplotlib",
                "seaborn",
                "scipy",
                "pyarrow",
            ]
        ),
        "tag": tag,
    }

    results_dir = cfg.paths.results_dir
    save_json(meta, results_dir / f"{tag}_meta.json")

    emb_paths: Dict[str, Path] = {}
    df_emb: Optional[pd.DataFrame] = None

    if run_embeddings:
        emb_paths = stage_embeddings(tag, cfg, df_input, which=selected_models)
        save_json({k: str(v) for k, v in emb_paths.items()}, results_dir / f"{tag}_embedding_paths.json")

    # Decide which embedding key to use
    chosen_key = "large" if "large" in selected_models else selected_models[0]

    # If we need embeddings downstream or just to visualize, try to use existing files
    if not emb_paths and (run_dimred or run_cluster or run_summarize or run_visualize):
        # Prefer the chosen_key, but allow fallback to the other if needed (esp. for visualize-only)
        candidate = cfg.paths.embeddings_dir / f"embeddings_{chosen_key}.parquet"
        if candidate.exists():
            emb_paths[chosen_key] = candidate
        else:
            # Try the alternative key
            alt_key = "small" if chosen_key == "large" else "large"
            alt_candidate = cfg.paths.embeddings_dir / f"embeddings_{alt_key}.parquet"
            if alt_candidate.exists():
                chosen_key = alt_key
                emb_paths[chosen_key] = alt_candidate
            else:
                # For visualize-only, we may be able to determine the correct key from saved coords/labels
                if run_visualize:
                    resolved = _resolve_existing_key_for_visualize(
                        cfg, results_dir, tag, preferred_order=(selected_models if selected_models else [chosen_key])
                    )
                    if resolved is not None:
                        chosen_key = resolved
                        resolved_path = cfg.paths.embeddings_dir / f"embeddings_{chosen_key}.parquet"
                        if resolved_path.exists():
                            emb_paths[chosen_key] = resolved_path
                if not emb_paths:
                    raise FileNotFoundError("No embeddings found. Run with --do embeddings or ensure an embeddings parquet exists.")

    # Load embeddings (needed for ids even when only visualizing)
    df_emb = load_embeddings(emb_paths[chosen_key])

    # TriMAP coordinates
    Y: Optional[np.ndarray] = None
    if run_dimred:
        dim_res = stage_dimred(cfg, df_emb)
        Y = dim_res["Y"]
        save_json({"metrics": dim_res["metrics"], "params": dim_res["params"]}, results_dir / f"{tag}_trimap.json")
        coords_path = results_dir / f"{tag}_coords_{chosen_key}.npy"
        np.save(coords_path, Y)
    elif run_visualize:
        coords_path = results_dir / f"{tag}_coords_{chosen_key}.npy"
        if not coords_path.exists():
            # Try alternate key if available
            for key in (["large", "small"] if chosen_key == "small" else ["large", "small"]):
                alt = results_dir / f"{tag}_coords_{key}.npy"
                if alt.exists():
                    chosen_key = key
                    coords_path = alt
                    break
        if not coords_path.exists():
            raise FileNotFoundError(
                f"No existing TriMAP coords found for tag '{tag}'. Expected {results_dir / f'{tag}_coords_{{large|small}}.npy'}"
            )
        Y = np.load(coords_path)

    # HDBSCAN labels
    labels: Optional[np.ndarray] = None
    if run_cluster:
        clus_res = stage_clustering(cfg, df_emb)
        labels = clus_res["labels"]
        save_json(
            {"metrics": {k: float(v) for k, v in clus_res["metrics"].items()}, "params": clus_res["params"]},
            results_dir / f"{tag}_hdbscan.json",
        )
        labels_path = results_dir / f"{tag}_labels_{chosen_key}.npy"
        np.save(labels_path, labels)
    elif run_visualize:
        labels_path = results_dir / f"{tag}_labels_{chosen_key}.npy"
        if not labels_path.exists():
            for key in (["large", "small"] if chosen_key == "small" else ["large", "small"]):
                alt = results_dir / f"{tag}_labels_{key}.npy"
                if alt.exists():
                    chosen_key = key
                    labels_path = alt
                    break
        if not labels_path.exists():
            raise FileNotFoundError(
                f"No existing HDBSCAN labels found for tag '{tag}'. Expected {results_dir / f'{tag}_labels_{{large|small}}.npy'}"
            )
        labels = np.load(labels_path)

    # Visualization
    if run_visualize:
        if Y is None or labels is None:
            raise RuntimeError("Visualization requested but missing coords or labels.")
        ids = df_emb["id"].astype(str).tolist()
        plot_embedding(cfg, Y, labels, ids, cfg.paths.plots_dir / f"{tag}_trimap_hdbscan.png")

    # Visualization also when both stages were run in this invocation
    if (run_dimred and run_cluster) and (Y is not None and labels is not None) and not run_visualize:
        ids = df_emb["id"].astype(str).tolist()
        plot_embedding(cfg, Y, labels, ids, cfg.paths.plots_dir / f"{tag}_trimap_hdbscan.png")

    # Summarization
    if run_summarize:
        if labels is None:
            labels_path = results_dir / f"{tag}_labels_{chosen_key}.npy"
            if labels_path.exists():
                labels = np.load(labels_path)
            else:
                raise FileNotFoundError("No labels available for summarization. Run clustering first or provide labels.")

        summaries = summarize_clusters(
            tag,
            cfg,
            np.array(df_emb["vector_unit"].tolist(), dtype=np.float32),
            df_emb,
            labels,
            prompt_path=Path("prompts/synthesis_prompt.md"),
            out_dir=cfg.paths.summaries_dir,
        )

        index = [
            {
                "cluster_id": s.cluster_id,
                "n_items": len(s.indices),
                "representative_indices": s.representative_indices,
                "summary_path": str(cfg.paths.summaries_dir / f"{s.cluster_id:02d}_summary.md"),
            }
            for s in summaries
        ]
        save_json({"summaries": index}, results_dir / f"{tag}_summaries_index.json")

    # Optional tuning
    if args.tune_trimap:
        X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
        g = grid_search_trimap(cfg, X_unit)
        save_json({"records": g.records}, results_dir / f"{tag}_tuning_trimap.json")

    if args.tune_hdbscan:
        X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
        g = grid_search_hdbscan(cfg, X_unit)
        save_json({"records": g.records}, results_dir / f"{tag}_tuning_hdbscan.json")


if __name__ == "__main__":
    main()
