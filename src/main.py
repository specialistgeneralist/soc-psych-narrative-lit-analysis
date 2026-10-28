from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.pipeline.config import PipelineConfig, load_config
from src.pipeline.utils import ensure_dirs, package_versions, save_json, set_global_seed
from src.pipeline.embeddings import compute_and_store_embeddings, load_embeddings
from src.pipeline.dimred import load_dimred_map, run_trimap, save_dimred_map
from src.pipeline.clustering import run_hdbscan
from src.pipeline.parameter_search import grid_search_hdbscan, grid_search_trimap
from src.pipeline.visualize import plot_embedding
from src.pipeline.summarize import summarize_clusters

# This file is the CLI entry point for the end-to-end pipeline:
# 1) read input TSV  2) compute/load embeddings  3) TriMAP  4) HDBSCAN
# 5) plot embedding  6) summarize clusters via an LLM (Anthropic).


# --- I/O helpers ---


def read_input_tsv(path: Path, text_column: str) -> pd.DataFrame:
    # Read everything as string to avoid dtype surprises (IDs, empty fields, etc.).
    df = pd.read_csv(path, sep="\t", dtype=str)
    assert text_column in df.columns, f"Column '{text_column}' not found in {path}"
    return df


# --- Stage wrappers ---
# These small wrappers standardize data extraction (e.g., pulling `vector_unit`) and
# make it easier to run stages independently from `main()`.


def stage_embeddings(cfg: PipelineConfig, df: pd.DataFrame, which: List[str]) -> Dict[str, Path]:
    # Compute embeddings for the requested model(s) and return the output parquet path per model key.
    paths: Dict[str, Path] = {}
    for key in which:
        p = compute_and_store_embeddings(cfg, df, model_key=key, overwrite=True)
        paths[key] = p
    return paths


def _embedding_ids(df_emb: pd.DataFrame) -> List[str]:
    if "id" not in df_emb.columns:
        raise ValueError("Embeddings parquet must contain an 'id' column for dimensionality-reduction map alignment.")
    ids = df_emb["id"].astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("Embeddings parquet contains duplicate ids; fixed dimensionality-reduction maps require unique ids.")
    return ids


def stage_dimred(cfg: PipelineConfig, df_emb: pd.DataFrame, *, use_dimred_map: Path | None = None) -> Dict:
    # TriMAP expects an array of vectors; we store unit-normalized vectors in `vector_unit`.
    if use_dimred_map is not None:
        Y, map_meta = load_dimred_map(use_dimred_map, _embedding_ids(df_emb))
        return {"Y": Y, "metrics": map_meta.get("metrics", {}), "params": map_meta.get("params", {}), "map_meta": map_meta}
    X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
    res = run_trimap(cfg, X_unit)
    return {"Y": res.Y, "metrics": res.metrics, "params": res.params, "map_meta": None}


def stage_clustering(cfg: PipelineConfig, df_emb: pd.DataFrame) -> Dict:
    # Cluster in embedding space; `run_hdbscan` handles optional PCA and silhouette reporting.
    X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
    res = run_hdbscan(cfg, X_unit)
    return {"labels": res.labels, "probs": res.probabilities, "metrics": res.metrics, "params": res.params}


def stage_summarize(
    cfg: PipelineConfig,
    df_emb: pd.DataFrame,
    labels: np.ndarray,
    prompt_path: Path,
    out_dir: Path,
    tag: str = "run",
):
    # Summarization needs both the raw texts (from the embeddings parquet) and the vectors for nearest-centroid selection.
    X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
    return summarize_clusters(cfg, X_unit, df_emb, labels, prompt_path, out_dir, tag=tag)


def _resolve_existing_key_for_visualize(cfg: PipelineConfig, results_dir: Path, tag: str, preferred_order: List[str]) -> Optional[str]:
    """Pick a model key ('large' or 'small') that has all required artifacts for visualization."""
    # `--do visualize` is intentionally "read-only": we want to plot from existing coords/labels/embeddings
    # even if the user didn't run embeddings in this invocation.
    for key in preferred_order:
        coords_path = results_dir / f"{tag}_coords_{key}.npy"
        labels_path = results_dir / f"{tag}_labels_{key}.npy"
        emb_path = cfg.paths.embeddings_dir / f"embeddings_{key}.parquet"
        if coords_path.exists() and labels_path.exists() and emb_path.exists():
            return key
    for key in ["large", "small"]:
        if key in preferred_order:
            continue
        coords_path = results_dir / f"{tag}_coords_{key}.npy"
        labels_path = results_dir / f"{tag}_labels_{key}.npy"
        emb_path = cfg.paths.embeddings_dir / f"embeddings_{key}.parquet"
        if coords_path.exists() and labels_path.exists() and emb_path.exists():
            return key
    return None


def main():
    # --- CLI / config ---
    parser = argparse.ArgumentParser(description="Semantic clustering and synthesis pipeline")
    parser.add_argument("--config", type=str, default="configs/baseline.yaml", help="Path to experiment YAML config")
    parser.add_argument("--env", type=str, default=None, help="Path to .env with secrets (optional)")
    parser.add_argument("--input", type=str, default=None, help="Override input TSV path")
    parser.add_argument("--text-col", type=str, default=None, help="Override text column name")
    parser.add_argument("--id-col", type=str, default=None, help="Override ID column name (optional)")
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help=(
            "Override which embeddings to compute/load, comma-separated. OpenAI keys small/large can be computed; "
            "other keys are treated as cached parquet keys, e.g. google_gemini_embedding_001. Defaults to config `use_models`."
        ),
    )

    parser.add_argument(
        "--do",
        type=str,
        default="all",
        help=(
            "Stages to run: embeddings,dimred,cluster,summarize,visualize (or 'visualise'),none,all. "
            "Use 'visualize' to render plots from existing coords/labels without recomputing."
        ),
    )

    parser.add_argument("--tune-trimap", action="store_true", help="Grid search TriMAP params and save report")
    parser.add_argument("--tune-hdbscan", action="store_true", help="Grid search HDBSCAN params and save report")
    parser.add_argument(
        "--trimap-n-iters",
        type=int,
        default=None,
        help="Optional override for TriMAP optimization iterations. Library default is 400; final runs may use e.g. 1000.",
    )
    parser.add_argument(
        "--save-dimred-map",
        nargs="?",
        const="__default__",
        default=None,
        help=(
            "Save the 2D dimensionality-reduction coordinate map keyed by item id. "
            "If no path is supplied, writes outputs/results/{tag}_dimred_map_{model_key}.json."
        ),
    )
    parser.add_argument(
        "--use-dimred-map",
        type=str,
        default=None,
        help="Load an existing fixed 2D coordinate map instead of running TriMAP, aligning coordinates by item id.",
    )
    parser.add_argument(
        "--vis-percentile",
        type=float,
        default=75.0,
        help=(
            "Density contour mass percentile for visualization shading. "
            "Accepts either 0-1 (e.g. 0.75) or 0-100 (e.g. 75). "
            "Only applies when --vis-boundary-style=density."
        ),
    )
    parser.add_argument(
        "--vis-boundary-style",
        type=str,
        choices=["density", "hull", "none"],
        default="density",
        help="Cluster overlay style: shaded KDE density region (default), legacy convex hull, or none.",
    )
    parser.add_argument(
        "--vis-marker-size",
        type=float,
        default=None,
        help="Override visualization scatter marker size. Defaults to config visualization.marker_size.",
    )
    parser.add_argument(
        "--label-clusters",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw cluster integer labels at cluster centroids (default: enabled).",
    )
    parser.add_argument("--tag", type=str, default="run", help="Tag for output filenames, default to config file if exists")

    args = parser.parse_args()

    # Load config and apply CLI overrides (CLI is meant for quick experiments; YAML remains the source of truth).
    cfg = load_config(config_path=args.config, env_path=args.env)

    # If tag not provided, default to config filename without extension if possible.
    # .. example "configs/baseline.yaml" -> "baseline"
    if args.tag == "run":
        config_stem = Path(args.config).stem
        args.tag = config_stem if config_stem else "run"

    # Apply "simple" overrides directly to cfg; downstream functions read from `cfg`.
    if args.input:
        cfg.paths.input_tsv = Path(args.input)
    if args.text_col:
        cfg.data.text_column = args.text_col
    if args.id_col is not None:
        cfg.data.id_column = args.id_col if args.id_col != "" else None
    if args.trimap_n_iters is not None:
        cfg.trimap_n_iters = int(args.trimap_n_iters)

    # --- Reproducibility + output layout ---
    set_global_seed(cfg.seed)
    ensure_dirs(
        cfg.paths.output_dir,
        cfg.paths.embeddings_dir,
        cfg.paths.plots_dir,
        cfg.paths.results_dir,
        cfg.paths.summaries_dir,
    )

    print(f" --> Loading input data from {cfg.paths.input_tsv} with text column '{cfg.data.text_column}' and id column '{cfg.data.id_column}'")
    df_input = read_input_tsv(cfg.paths.input_tsv, cfg.data.text_column)

    # Determine which embedding keys to use. OpenAI keys ("small"/"large") can be computed;
    # provider-specific keys such as "google_gemini_embedding_001" are load-only cached parquet keys.
    if args.models:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        selected_models = [str(m) for m in cfg.use_models if str(m).strip()]
    if not selected_models:
        selected_models = ["small"]

    # --- Stage selection ---
    # `--do all` intentionally does not include visualization-only mode; plotting happens either:
    # - as part of the normal dimred+cluster run, or
    # - explicitly via `--do visualize` (read existing artifacts and plot).
    run_embeddings = run_dimred = run_cluster = run_summarize = run_visualize = False
    if args.do == "all" and not (args.tune_trimap or args.tune_hdbscan):
        run_embeddings = run_dimred = run_cluster = run_summarize = True
    else:
        stages = {s.strip().lower() for s in args.do.split(",")}
        run_embeddings = "embeddings" in stages
        run_dimred = "dimred" in stages
        run_cluster = "cluster" in stages
        run_summarize = "summarize" in stages
        run_visualize = ("visualize" in stages) or ("visualise" in stages)
        # "none" intentionally leaves all stage flags false; tuning switches still load cached embeddings below.

    # Record config and versions to make results self-describing/reproducible.
    meta: Dict = {
        "config": cfg.to_dict(),
        "config_path": args.config,
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
                "pyyaml",
            ]
        ),
        "tag": args.tag,
        "save_dimred_map": args.save_dimred_map,
        "use_dimred_map": args.use_dimred_map,
    }

    results_dir = cfg.paths.results_dir
    save_json(meta, results_dir / f"{args.tag}_meta.json")

    emb_paths: Dict[str, Path] = {}
    df_emb: Optional[pd.DataFrame] = None

    if run_embeddings:
        unsupported_compute_keys = [key for key in selected_models if key not in ("small", "large")]
        if unsupported_compute_keys:
            raise ValueError(
                "The embeddings stage can only compute OpenAI keys 'small' and 'large'. "
                f"These keys are load-only and must already exist as parquet files: {unsupported_compute_keys}. "
                "For cached embeddings, omit the embeddings stage, e.g. use --do none with --tune-trimap/--tune-hdbscan, "
                "or use --do dimred,cluster,visualize for a full cached run."
            )
        print(f" --> Computing/loading embeddings for models: {selected_models}")
        emb_paths = stage_embeddings(cfg, df_input, which=selected_models)
        save_json({k: str(v) for k, v in emb_paths.items()}, results_dir / f"{args.tag}_embedding_paths.json")

    # `chosen_key` is the model whose embedding parquet we will load for downstream stages.
    # Preference: large (if requested), otherwise the first selected model.
    chosen_key = "large" if "large" in selected_models else selected_models[0]

    # If we didn't (re)compute embeddings in this run, try to locate an existing parquet on disk.
    # This supports workflows like: run embeddings once, then iterate on dimred/cluster/visualize/tuning.
    needs_embeddings = run_dimred or run_cluster or run_summarize or run_visualize or args.tune_trimap or args.tune_hdbscan
    if not emb_paths and needs_embeddings:
        candidate = cfg.paths.embeddings_dir / f"embeddings_{chosen_key}.parquet"
        if candidate.exists():
            emb_paths[chosen_key] = candidate
        else:
            # Fallback to the other model key if present.
            alt_key = "small" if chosen_key == "large" else "large"
            alt_candidate = cfg.paths.embeddings_dir / f"embeddings_{alt_key}.parquet"
            if alt_candidate.exists():
                chosen_key = alt_key
                emb_paths[chosen_key] = alt_candidate
            else:
                # Visualization is the one mode where we do an exhaustive scan for "any consistent set"
                # of coords/labels/embeddings, because the user explicitly asked to plot existing artifacts.
                if run_visualize:
                    resolved = _resolve_existing_key_for_visualize(
                        cfg, results_dir, args.tag, preferred_order=(selected_models if selected_models else [chosen_key])
                    )
                    if resolved is not None:
                        chosen_key = resolved
                        resolved_path = cfg.paths.embeddings_dir / f"embeddings_{chosen_key}.parquet"
                        if resolved_path.exists():
                            emb_paths[chosen_key] = resolved_path
                if not emb_paths:
                    raise FileNotFoundError(
                        "No embeddings found. Run with --do embeddings for OpenAI keys, or ensure the cached embeddings parquet "
                        f"exists at {candidate}. For provider-specific keys such as Google Vertex/Gemini, embeddings are "
                        "intentionally load-only in this CLI path."
                    )

    df_emb = load_embeddings(emb_paths[chosen_key])

    # --- Dimensionality reduction (TriMAP) ---
    Y: Optional[np.ndarray] = None
    dim_res: Optional[Dict] = None
    dimred_map_meta: Optional[Dict] = None
    dimred_map_path = Path(args.use_dimred_map) if args.use_dimred_map else None
    if run_dimred:
        dim_res = stage_dimred(cfg, df_emb, use_dimred_map=dimred_map_path)
        Y = dim_res["Y"]
        dimred_map_meta = dim_res.get("map_meta")
        if dimred_map_meta is not None:
            map_model_key = str(dimred_map_meta.get("model_key", ""))
            if map_model_key and map_model_key != chosen_key:
                print(
                    "WARNING: using fixed dimensionality-reduction coordinates saved for model key "
                    f"'{map_model_key}' with current model key '{chosen_key}'. This is useful for visual alignment, "
                    "but it is not a projection of the current embedding topology."
                )
            save_json(
                {
                    "source": "fixed_dimred_map",
                    "map_path": str(dimred_map_path),
                    "map_model_key": dimred_map_meta.get("model_key"),
                    "map_tag": dimred_map_meta.get("tag"),
                    "metrics": dim_res["metrics"],
                    "params": dim_res["params"],
                    "note": (
                        "Coordinates were loaded from a fixed item-id map rather than recomputed. "
                        "Use this for comparing alternate labels on a shared layout; cross-embedding use is visual alignment only."
                    ),
                },
                results_dir / f"{args.tag}_trimap.json",
            )
        else:
            save_json({"source": "trimap", "metrics": dim_res["metrics"], "params": dim_res["params"]}, results_dir / f"{args.tag}_trimap.json")
        coords_path = results_dir / f"{args.tag}_coords_{chosen_key}.npy"
        np.save(coords_path, Y)
    elif run_visualize:
        if dimred_map_path is not None:
            Y, dimred_map_meta = load_dimred_map(dimred_map_path, _embedding_ids(df_emb))
            map_model_key = str(dimred_map_meta.get("model_key", ""))
            if map_model_key and map_model_key != chosen_key:
                print(
                    "WARNING: using fixed dimensionality-reduction coordinates saved for model key "
                    f"'{map_model_key}' with current model key '{chosen_key}'. This is useful for visual alignment, "
                    "but it is not a projection of the current embedding topology."
                )
        else:
            # Read existing coordinates from disk, trying the other model key if the preferred one is missing.
            coords_path = results_dir / f"{args.tag}_coords_{chosen_key}.npy"
            if not coords_path.exists():
                for key in (["large", "small"] if chosen_key == "small" else ["large", "small"]):
                    alt = results_dir / f"{args.tag}_coords_{key}.npy"
                    if alt.exists():
                        chosen_key = key
                        coords_path = alt
                        break
            if not coords_path.exists():
                raise FileNotFoundError(
                    f"No existing TriMAP coords found for tag '{args.tag}'. Expected {results_dir / f'{args.tag}_coords_{{large|small}}.npy'}"
                )
            Y = np.load(coords_path)

    if args.save_dimred_map is not None:
        if Y is None:
            raise RuntimeError("--save-dimred-map requires dimensionality-reduction coordinates. Include --do dimred or --do visualize.")
        if args.save_dimred_map == "__default__":
            save_map_path = results_dir / f"{args.tag}_dimred_map_{chosen_key}.json"
        else:
            save_map_path = Path(args.save_dimred_map)
        source_metrics = dim_res.get("metrics", {}) if dim_res else {}
        source_params = dim_res.get("params", {}) if dim_res else {}
        save_dimred_map(
            save_map_path,
            Y,
            _embedding_ids(df_emb),
            model_key=chosen_key,
            tag=args.tag,
            config_path=args.config,
            metrics=source_metrics,
            params=source_params,
        )
        print(f" --> Saved fixed dimensionality-reduction coordinate map to {save_map_path}")

    # --- Clustering (HDBSCAN) ---
    labels: Optional[np.ndarray] = None
    if run_cluster:
        clus_res = stage_clustering(cfg, df_emb)
        labels = clus_res["labels"]
        save_json(
            {"metrics": {k: float(v) for k, v in clus_res["metrics"].items()}, "params": clus_res["params"]},
            results_dir / f"{args.tag}_hdbscan.json",
        )
        labels_path = results_dir / f"{args.tag}_labels_{chosen_key}.npy"
        np.save(labels_path, labels)
    elif run_visualize:
        # Read existing labels from disk, trying the other model key if the preferred one is missing.
        labels_path = results_dir / f"{args.tag}_labels_{chosen_key}.npy"
        if not labels_path.exists():
            for key in (["large", "small"] if chosen_key == "small" else ["large", "small"]):
                alt = results_dir / f"{args.tag}_labels_{key}.npy"
                if alt.exists():
                    chosen_key = key
                    labels_path = alt
                    break
        if not labels_path.exists():
            raise FileNotFoundError(
                f"No existing HDBSCAN labels found for tag '{args.tag}'. Expected {results_dir / f'{args.tag}_labels_{{large|small}}.npy'}"
            )
        labels = np.load(labels_path)

    # --- Visualization ---
    if run_visualize:
        if Y is None or labels is None:
            raise RuntimeError("Visualization requested but missing coords or labels.")
        ids = df_emb["id"].astype(str).tolist()
        plot_embedding(
            cfg,
            Y,
            labels,
            ids,
            cfg.paths.plots_dir / f"{args.tag}_trimap_hdbscan.png",
            boundary_style=args.vis_boundary_style,
            density_percentile=args.vis_percentile,
            marker_size=args.vis_marker_size,
            label_clusters=args.label_clusters,
        )

    # Default behavior: if we ran both dimred and clustering, plot once at the end.
    if (run_dimred and run_cluster) and (Y is not None and labels is not None) and not run_visualize:
        ids = df_emb["id"].astype(str).tolist()
        plot_embedding(
            cfg,
            Y,
            labels,
            ids,
            cfg.paths.plots_dir / f"{args.tag}_trimap_hdbscan.png",
            boundary_style=args.vis_boundary_style,
            density_percentile=args.vis_percentile,
            marker_size=args.vis_marker_size,
            label_clusters=args.label_clusters,
        )

    # --- Summarization (LLM synthesis) ---
    if run_summarize:
        # Allow summarization to run even if this invocation didn't run clustering, as long as labels exist on disk.
        if labels is None:
            labels_path = results_dir / f"{args.tag}_labels_{chosen_key}.npy"
            if labels_path.exists():
                labels = np.load(labels_path)
            else:
                raise FileNotFoundError("No labels available for summarization. Run clustering first or provide labels.")
        summaries = summarize_clusters(
            cfg,
            np.array(df_emb["vector_unit"].tolist(), dtype=np.float32),
            df_emb,
            labels,
            prompt_path=Path("prompts/synthesis_prompt.md"),
            out_dir=cfg.paths.summaries_dir,
            tag=args.tag,
        )
        index = [
            {
                "cluster_id": s.cluster_id,
                "n_items": len(s.indices),
                "representative_indices": s.representative_indices,
                "summary_path": str(cfg.paths.summaries_dir / f"{args.tag}_cluster_{s.cluster_id:02d}_summary.md"),
            }
            for s in summaries
        ]
        save_json(
            {
                "summaries": index,
                "meta_summary_path": str(cfg.paths.summaries_dir / f"{args.tag}_meta_summary.md"),
            },
            results_dir / f"{args.tag}_summaries_index.json",
        )

    # --- Hyperparameter tuning ---
    if args.tune_trimap:
        X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
        g = grid_search_trimap(cfg, X_unit)
        save_json({"records": g.records}, results_dir / f"{args.tag}_tuning_trimap.json")

    if args.tune_hdbscan:
        X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
        g = grid_search_hdbscan(cfg, X_unit)
        save_json({"records": g.records}, results_dir / f"{args.tag}_tuning_hdbscan.json")


if __name__ == "__main__":
    main()
