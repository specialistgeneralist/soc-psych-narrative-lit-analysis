from __future__ import annotations

import argparse
import copy
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from src.main import read_input_tsv
from src.pipeline.clustering import run_hdbscan
from src.pipeline.config import PipelineConfig, load_config
from src.pipeline.embeddings import compute_and_store_embeddings, load_embeddings
from src.pipeline.google_embeddings import VertexEmbeddingSettings, compute_and_store_vertex_embeddings, infer_google_project
from src.pipeline.summarize import summarize_clusters
from src.pipeline.utils import ensure_dirs, load_json, package_versions, save_json, set_global_seed
from src.robustness.metrics import LabelSeries, compare_labels, records_to_markdown_table


@dataclass
class RobustnessSpec:
    """YAML-backed robustness experiment specification."""

    baseline_config: str = "configs/baseline.yaml"
    baseline_tag: str = "baseline"
    baseline_model_key: str = "large"
    output_dir: str = "outputs/robustness_mvp"
    trimap_n_iters: int = 1000

    run_vector_openai_small: bool = True
    run_vector_google_vertex: bool = False
    run_seed_variants: bool = True
    run_hdbscan_regime_comparison: bool = True
    run_labelling_variants: bool = False
    run_knockout_trials: bool = True

    seed_variants: list[int] = field(default_factory=lambda: [7, 123, 2026, 31415, 8675309])
    hdbscan_variant_tags: list[str] = field(default_factory=lambda: ["variant_1", "variant_2"])
    hdbscan_variant_configs: list[str] = field(default_factory=lambda: ["configs/variant_1.yaml", "configs/variant_2.yaml"])
    google_vertex_project: str | None = None
    google_vertex_location: str = "us-central1"
    google_vertex_model: str = "gemini-embedding-001"
    google_vertex_task_type: str = "CLUSTERING"
    google_vertex_output_dimensionality: int | None = 3072
    google_vertex_model_key: str = "google_gemini_embedding_001"
    google_vertex_tag: str = "robust_google_gemini_embedding_001"
    knockout_fractions: list[float] = field(default_factory=lambda: [0.9, 0.8, 0.7])
    knockout_trials_per_fraction: int = 10
    knockout_seed: int = 424242

    # Labelling variants are intentionally opt-in because exact model availability and cost vary.
    # Example item: {provider: anthropic, model: claude-haiku-4-5, tag: label_haiku}
    labelling_variants: list[dict[str, str]] = field(default_factory=list)


KEY_RESULT_COLUMNS = [
    "experiment",
    "variant",
    "n_common",
    "variant_n_clusters_common",
    "variant_frac_noise_common",
    "ari",
    "ami",
    "nmi",
    "noise_status_agreement",
    "co_cluster_jaccard",
]


def _read_spec(path: Path) -> RobustnessSpec:
    if not path.exists():
        raise FileNotFoundError(f"Robustness spec not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return RobustnessSpec(**raw)


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


def _load_ids_for_embeddings(cfg: PipelineConfig, df_emb: pd.DataFrame) -> list[str]:
    if "id" not in df_emb.columns:
        raise ValueError("Embeddings parquet must contain an 'id' column")
    return df_emb["id"].astype(str).tolist()


def _load_label_series(results_dir: Path, tag: str, model_key: str, ids: list[str]) -> LabelSeries:
    labels_path = results_dir / f"{tag}_labels_{model_key}.npy"
    if not labels_path.exists():
        raise FileNotFoundError(f"Label file not found: {labels_path}")
    return LabelSeries(ids=ids, labels=np.load(labels_path).astype(int))


def _load_or_run_baseline(
    cfg: PipelineConfig,
    df_emb: pd.DataFrame,
    out_dir: Path,
    tag: str,
    model_key: str,
) -> LabelSeries:
    """Load canonical baseline labels if available, otherwise compute them in the run output dir."""

    ids = _load_ids_for_embeddings(cfg, df_emb)
    labels_path = cfg.paths.results_dir / f"{tag}_labels_{model_key}.npy"
    if labels_path.exists():
        return _load_label_series(cfg.paths.results_dir, tag, model_key, ids)

    return _save_cluster_run(
        cfg,
        df_emb,
        out_dir,
        tag,
        model_key,
        extra_meta={"source_of_variation": "baseline_recomputed", "reason": "baseline labels absent"},
    )


def _save_cluster_run(
    cfg: PipelineConfig,
    df_emb: pd.DataFrame,
    out_dir: Path,
    tag: str,
    model_key: str,
    *,
    extra_meta: dict[str, Any] | None = None,
) -> LabelSeries:
    set_global_seed(cfg.seed)
    X_unit = np.array(df_emb["vector_unit"].tolist(), dtype=np.float32)
    result = run_hdbscan(cfg, X_unit)
    ensure_dirs(out_dir)

    labels_path = out_dir / f"{tag}_labels_{model_key}.npy"
    hdbscan_path = out_dir / f"{tag}_hdbscan.json"
    np.save(labels_path, result.labels.astype(int))
    save_json(
        {
            "metrics": result.metrics,
            "params": result.params,
            "tag": tag,
            "model_key": model_key,
            "seed": cfg.seed,
            "extra_meta": extra_meta or {},
        },
        hdbscan_path,
    )
    return LabelSeries(ids=_load_ids_for_embeddings(cfg, df_emb), labels=result.labels.astype(int))


def _subset_embeddings(df_emb: pd.DataFrame, selected_ids: set[str]) -> pd.DataFrame:
    mask = df_emb["id"].astype(str).isin(selected_ids)
    subset = df_emb.loc[mask].reset_index(drop=True)
    if subset.empty:
        raise ValueError("Subsample produced no rows")
    return subset


def _write_records(out_dir: Path, records: list[dict[str, Any]]) -> None:
    ensure_dirs(out_dir)
    save_json({"records": records}, out_dir / "robustness_metrics.json")
    pd.DataFrame(records).to_csv(out_dir / "robustness_metrics.csv", index=False)

    md = "# Robustness MVP results\n\n"
    md += "This report is generated by `python -m src.robustness.run_robustness`.\n\n"
    md += records_to_markdown_table(records, KEY_RESULT_COLUMNS)
    (out_dir / "robustness_report.md").write_text(md, encoding="utf-8")


def _write_manifest(out_dir: Path, spec: RobustnessSpec, baseline_cfg: PipelineConfig) -> None:
    manifest = {
        "spec": asdict(spec),
        "baseline_config_effective": baseline_cfg.to_dict(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": _git_commit_or_unknown(),
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
    }
    save_json(manifest, out_dir / "run_manifest.json")


def run_vector_representation(
    spec: RobustnessSpec,
    cfg: PipelineConfig,
    df_input: pd.DataFrame,
    baseline: LabelSeries,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """A. Compare baseline large embeddings against OpenAI small embeddings."""

    if not spec.run_vector_openai_small:
        return []

    small_path = compute_and_store_embeddings(cfg, df_input, model_key="small", overwrite=False)
    df_small = load_embeddings(small_path)
    variant = _save_cluster_run(
        cfg,
        df_small,
        out_dir,
        "robust_openai_small",
        "small",
        extra_meta={"source_of_variation": "vector_representation", "embedding_model_key": "small"},
    )
    return [
        compare_labels(
            baseline,
            variant,
            variant_name="openai_small_embeddings",
            experiment="A_vector_representation",
            extra={"embedding_model_key": "small"},
        )
    ]


def run_google_vertex_representation(
    spec: RobustnessSpec,
    cfg: PipelineConfig,
    df_input: pd.DataFrame,
    baseline: LabelSeries,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """A. Compare baseline OpenAI large embeddings against Google Vertex embeddings."""

    if not spec.run_vector_google_vertex:
        return []

    settings = VertexEmbeddingSettings(
        project=infer_google_project(spec.google_vertex_project),
        location=spec.google_vertex_location,
        model=spec.google_vertex_model,
        task_type=spec.google_vertex_task_type,
        output_dimensionality=spec.google_vertex_output_dimensionality,
        model_key=spec.google_vertex_model_key,
    )
    emb_path = compute_and_store_vertex_embeddings(cfg, df_input, settings, overwrite=False)
    df_google = load_embeddings(emb_path)
    variant = _save_cluster_run(
        cfg,
        df_google,
        out_dir,
        spec.google_vertex_tag,
        spec.google_vertex_model_key,
        extra_meta={
            "source_of_variation": "vector_representation",
            "provider": "google_vertex",
            "embedding_model": settings.model,
            "task_type": settings.task_type,
            "output_dimensionality": settings.output_dimensionality,
            "location": settings.location,
        },
    )
    return [
        compare_labels(
            baseline,
            variant,
            variant_name="google_vertex_gemini_embedding_001",
            experiment="A_vector_representation",
            extra={
                "embedding_provider": "google_vertex",
                "embedding_model": settings.model,
                "embedding_task_type": settings.task_type,
                "embedding_output_dimensionality": settings.output_dimensionality,
            },
        )
    ]


def run_seed_variants(
    spec: RobustnessSpec,
    cfg: PipelineConfig,
    df_large: pd.DataFrame,
    baseline: LabelSeries,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """B. Compare alternate PCA/random seeds against the baseline spec."""

    if not spec.run_seed_variants:
        return []

    records: list[dict[str, Any]] = []
    for seed in spec.seed_variants:
        cfg_seed = copy.deepcopy(cfg)
        cfg_seed.seed = int(seed)
        variant = _save_cluster_run(
            cfg_seed,
            df_large,
            out_dir,
            f"robust_seed_{seed}",
            spec.baseline_model_key,
            extra_meta={"source_of_variation": "seed", "seed": int(seed)},
        )
        records.append(
            compare_labels(
                baseline,
                variant,
                variant_name=f"seed_{seed}",
                experiment="B_seed",
                extra={"seed": int(seed)},
            )
        )
    return records


def run_hdbscan_regime_comparison(
    spec: RobustnessSpec,
    baseline: LabelSeries,
    baseline_ids: list[str],
    baseline_cfg: PipelineConfig,
) -> list[dict[str, Any]]:
    """C. Compare existing HDBSCAN regime runs, rerunning only if labels are absent."""

    if not spec.run_hdbscan_regime_comparison:
        return []

    records: list[dict[str, Any]] = []
    for config_path, tag in zip(spec.hdbscan_variant_configs, spec.hdbscan_variant_tags):
        cfg_variant = load_config(config_path)
        emb_path = cfg_variant.paths.embeddings_dir / f"embeddings_{spec.baseline_model_key}.parquet"
        df_emb = load_embeddings(emb_path)
        ids = _load_ids_for_embeddings(cfg_variant, df_emb)
        labels_path = cfg_variant.paths.results_dir / f"{tag}_labels_{spec.baseline_model_key}.npy"
        if labels_path.exists():
            variant = _load_label_series(cfg_variant.paths.results_dir, tag, spec.baseline_model_key, ids)
        else:
            variant = _save_cluster_run(cfg_variant, df_emb, cfg_variant.paths.results_dir, tag, spec.baseline_model_key)
        records.append(
            compare_labels(
                baseline,
                variant,
                variant_name=tag,
                experiment="C_hdbscan_regime",
                extra={"config_path": config_path},
            )
        )
    return records


def run_knockout_trials(
    spec: RobustnessSpec,
    cfg: PipelineConfig,
    df_large: pd.DataFrame,
    baseline: LabelSeries,
    out_dir: Path,
) -> list[dict[str, Any]]:
    """E. Subsample rows and compare surviving ids against the full baseline assignment."""

    if not spec.run_knockout_trials:
        return []

    rng = np.random.default_rng(spec.knockout_seed)
    all_ids = np.array(_load_ids_for_embeddings(cfg, df_large), dtype=str)
    records: list[dict[str, Any]] = []

    for fraction in spec.knockout_fractions:
        if not (0.0 < float(fraction) < 1.0):
            raise ValueError(f"Knockout fraction must be between 0 and 1: {fraction}")
        n_keep = max(2, int(round(len(all_ids) * float(fraction))))
        for trial in range(1, int(spec.knockout_trials_per_fraction) + 1):
            selected = set(rng.choice(all_ids, size=n_keep, replace=False).tolist())
            df_subset = _subset_embeddings(df_large, selected)
            tag = f"robust_knockout_{int(round(fraction * 100)):02d}_trial_{trial:02d}"
            variant = _save_cluster_run(
                cfg,
                df_subset,
                out_dir,
                tag,
                spec.baseline_model_key,
                extra_meta={
                    "source_of_variation": "knockout",
                    "fraction_kept": float(fraction),
                    "trial": int(trial),
                    "n_keep": int(n_keep),
                },
            )
            (out_dir / f"{tag}_ids.txt").write_text("\n".join(variant.ids) + "\n", encoding="utf-8")
            records.append(
                compare_labels(
                    baseline,
                    variant,
                    variant_name=tag,
                    experiment="E_knockout",
                    extra={"fraction_kept": float(fraction), "trial": int(trial), "n_keep": int(n_keep)},
                )
            )
    return records


def run_labelling_variants(
    spec: RobustnessSpec,
    cfg: PipelineConfig,
    df_large: pd.DataFrame,
    baseline: LabelSeries,
    out_dir: Path,
) -> None:
    """D. Optional Anthropic labelling variants for baseline clusters.

    This stage writes alternate summaries, but does not add rows to the clustering metric table because
    the underlying cluster assignments are unchanged. Text-comparison/evaluation can be layered on top
    of these generated summaries in a later analysis step.
    """

    if not spec.run_labelling_variants or not spec.labelling_variants:
        return

    labels = baseline.labels
    X_unit = np.array(df_large["vector_unit"].tolist(), dtype=np.float32)
    label_out_dir = out_dir / "labelling_variants"
    ensure_dirs(label_out_dir)

    for variant in spec.labelling_variants:
        provider = variant.get("provider", "anthropic")
        model = variant.get("model")
        tag = variant.get("tag") or f"label_{provider}_{model}".replace("/", "_").replace(":", "_")
        if provider != "anthropic":
            raise NotImplementedError("MVP labelling variants currently support provider='anthropic' only")
        if not model:
            raise ValueError(f"Labelling variant requires a model: {variant}")

        cfg_label = copy.deepcopy(cfg)
        cfg_label.models.anthropic_model = model
        summaries = summarize_clusters(
            cfg_label,
            X_unit,
            df_large,
            labels,
            prompt_path=Path("prompts/synthesis_prompt.md"),
            out_dir=label_out_dir,
            tag=tag,
        )
        save_json(
            {
                "provider": provider,
                "model": model,
                "tag": tag,
                "n_summaries": len(summaries),
                "summary_paths": [str(label_out_dir / f"{tag}_cluster_{s.cluster_id:02d}_summary.md") for s in summaries],
                "meta_summary_path": str(label_out_dir / f"{tag}_meta_summary.md"),
            },
            label_out_dir / f"{tag}_index.json",
        )


def run(spec_path: Path) -> Path:
    spec = _read_spec(spec_path)
    cfg = load_config(spec.baseline_config)
    cfg.trimap_n_iters = int(spec.trimap_n_iters)
    out_dir = Path(spec.output_dir)
    ensure_dirs(out_dir)
    _write_manifest(out_dir, spec, cfg)

    df_input = read_input_tsv(cfg.paths.input_tsv, cfg.data.text_column)
    emb_path = cfg.paths.embeddings_dir / f"embeddings_{spec.baseline_model_key}.parquet"
    if not emb_path.exists():
        emb_path = compute_and_store_embeddings(cfg, df_input, model_key=spec.baseline_model_key, overwrite=False)
    df_large = load_embeddings(emb_path)
    baseline_ids = _load_ids_for_embeddings(cfg, df_large)
    baseline = _load_or_run_baseline(cfg, df_large, out_dir, spec.baseline_tag, spec.baseline_model_key)

    records: list[dict[str, Any]] = []
    records.extend(run_vector_representation(spec, cfg, df_input, baseline, out_dir))
    records.extend(run_google_vertex_representation(spec, cfg, df_input, baseline, out_dir))
    records.extend(run_seed_variants(spec, cfg, df_large, baseline, out_dir))
    records.extend(run_hdbscan_regime_comparison(spec, baseline, baseline_ids, cfg))
    records.extend(run_knockout_trials(spec, cfg, df_large, baseline, out_dir))
    _write_records(out_dir, records)

    run_labelling_variants(spec, cfg, df_large, baseline, out_dir)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MVP robustness experiments for semantic-analysis.")
    parser.add_argument("--spec", default="configs/robustness_mvp.yaml", help="Path to robustness YAML spec")
    args = parser.parse_args()
    out_dir = run(Path(args.spec))
    print(f"Robustness outputs written to {out_dir}")


if __name__ == "__main__":
    main()
