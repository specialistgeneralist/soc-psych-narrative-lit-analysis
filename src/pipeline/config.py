from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Mapping, Optional, Tuple

from dotenv import load_dotenv

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "PyYAML is required to load experiment config files. Install with `pip install pyyaml`."
    ) from e


# YAML-backed configuration for the pipeline.
#
# Design goal: keep experiments reproducible and self-contained in YAML, while allowing
# `.env` to provide secrets (API keys) without ever committing them to the repo.


@dataclass
class Paths:
    # Where to read data from and where to write all artifacts (embeddings, plots, JSON metrics, summaries).
    input_tsv: Path
    output_dir: Path
    embeddings_dir: Path
    plots_dir: Path
    results_dir: Path
    summaries_dir: Path
    cache_dir: Path


@dataclass
class ModelConfig:
    # Model identifiers (strings) are passed directly into the respective SDKs.
    openai_small: str = "text-embedding-3-small"
    openai_large: str = "text-embedding-3-large"
    anthropic_model: str = "claude-sonnet-4-5-sonnet-20250929"


@dataclass
class DataConfig:
    # Column names in the input TSV/CSV.
    text_column: str = "text_normalised"
    id_column: Optional[str] = "uid"


@dataclass
class VisualizationConfig:
    # Matplotlib scatter marker size for cluster plots.
    marker_size: float = 30.0


@dataclass
class PipelineConfig:
    # This is the authoritative in-memory config object. Most stages accept `cfg` and read fields directly.
    seed: int = 42
    paths: Paths = None  # type: ignore
    models: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    # Defaults/toggles
    use_models: tuple[str, ...] = ("small", "large")
    metric: str = "euclidean"

    # TriMAP defaults
    trimap_n_inliers: int = 10
    trimap_n_outliers: int = 10
    trimap_n_random: int = 5
    trimap_n_iters: int = 400

    # TriMAP tuning grids
    trimap_grid_n_inliers: Tuple[int, ...] = (5, 10, 15)
    trimap_grid_n_outliers: Tuple[int, ...] = (3, 5, 10)
    trimap_grid_n_random: Tuple[int, ...] = (3, 5, 10)

    # HDBSCAN defaults
    hdbscan_min_cluster_size: int = 10
    hdbscan_min_samples: Optional[int] = 2
    hdbscan_pca_dim: Optional[int] = 2
    hdbscan_metric: str = "manhattan"
    hdbscan_selection_method: str = "leaf"
    hdbscan_cluster_selection_epsilon: float = 0.0

    # Supplementary eval metric space
    hdbscan_silhouette_eval_pca_dim: Optional[int] = 10
    hdbscan_silhouette_eval_metric: Optional[str] = None

    # HDBSCAN tuning grids
    hdbscan_grid_min_cluster_size: Tuple[int, ...] = (5, 10, 15, 20, 25)
    hdbscan_grid_min_samples: Tuple[Optional[int], ...] = (1, 2, 4)
    hdbscan_grid_selection_method: Tuple[str, ...] = ("leaf", "eom")
    hdbscan_grid_selection_epsilon: Tuple[float, ...] = (0.0, 0.01, 0.1)
    hdbscan_grid_allow_single_cluster: Tuple[bool, ...] = (False,)
    hdbscan_grid_metric: Tuple[str, ...] = ("euclidean", "manhattan")
    hdbscan_grid_pca_dim: Tuple[Optional[int], ...] = (None, 2, 3, 4, 5)

    # Summarization
    top_k_per_cluster: int = 3

    def to_dict(self):
        # `asdict` doesn't know how to serialize `Path` nicely; convert paths to strings explicitly.
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in self.paths.__dict__.items()}
        return d


def _as_tuple(value: Any, default: tuple) -> tuple:
    # Helper for YAML sections where values may be expressed as lists or tuples.
    if value is None:
        return default
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return tuple(value)


def _as_path(value: Any, default: str | Path) -> Path:
    # Normalize YAML string/path-like values to `Path`, with a default fallback.
    if value is None:
        return Path(default)
    return Path(value)


def _load_yaml_config(path: str | Path) -> dict[str, Any]:
    # Load YAML into a top-level dict. Missing files are treated as fatal so experiments are explicit.
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a top-level mapping/object: {p}")
    return data


def _section(data: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    # Pull a subsection from the YAML. Missing/empty sections are treated as `{}`.
    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Config section '{key}' must be a mapping/object")
    return value


def load_config(config_path: Optional[str] = None, env_path: Optional[str] = None) -> PipelineConfig:
    """Load pipeline config.

    Precedence philosophy:
    - `.env` is for secrets / environment only.
    - YAML is the authoritative experimental specification.
    - Python dataclass defaults act only as final fallbacks.
    """
    # Load environment variables first so SDKs can find API keys (without putting them in YAML).
    if env_path is None:
        load_dotenv(override=False)
    else:
        load_dotenv(env_path, override=False)

    # [Comment This: corrected — was "Load config path from YAML, with fallback ..."; missing files raise
    # FileNotFoundError in `_load_yaml_config`, while an *empty* YAML file yields `{}` and falls back to defaults.]
    config_path = config_path or "configs/baseline.yaml"
    raw = _load_yaml_config(config_path)
    if raw:
        print(f" --> Loaded config from {config_path}")
    else:
        # This branch means the YAML file existed but was empty (or contained only null).
        print(f" --> No config found at {config_path}, using defaults")

    paths_cfg = _section(raw, "paths")
    models_cfg = _section(raw, "models")
    data_cfg = _section(raw, "data")
    visualization_cfg = _section(raw, "visualization")
    trimap_cfg = _section(raw, "trimap")
    trimap_grid_cfg = _section(raw, "trimap_grid")
    hdbscan_cfg = _section(raw, "hdbscan")
    hdbscan_grid_cfg = _section(raw, "hdbscan_grid")
    summarization_cfg = _section(raw, "summarization")

    # Populate dataclasses, using YAML where provided and dataclass defaults as fallbacks.
    cfg = PipelineConfig(
        seed=int(raw.get("seed", 42)),
        paths=Paths(
            input_tsv=_as_path(paths_cfg.get("input_tsv"), "data/input.tsv"),
            output_dir=_as_path(paths_cfg.get("output_dir"), "outputs"),
            embeddings_dir=_as_path(paths_cfg.get("embeddings_dir"), "outputs/embeddings"),
            plots_dir=_as_path(paths_cfg.get("plots_dir"), "outputs/plots"),
            results_dir=_as_path(paths_cfg.get("results_dir"), "outputs/results"),
            summaries_dir=_as_path(paths_cfg.get("summaries_dir"), "outputs/summaries"),
            cache_dir=_as_path(paths_cfg.get("cache_dir"), "outputs/cache"),
        ),
        models=ModelConfig(
            openai_small=str(models_cfg.get("openai_small", ModelConfig.openai_small)),
            openai_large=str(models_cfg.get("openai_large", ModelConfig.openai_large)),
            anthropic_model=str(models_cfg.get("anthropic_model", ModelConfig.anthropic_model)),
        ),
        data=DataConfig(
            text_column=str(data_cfg.get("text_column", DataConfig.text_column)),
            id_column=data_cfg.get("id_column", DataConfig.id_column),
        ),
        visualization=VisualizationConfig(
            marker_size=float(visualization_cfg.get("marker_size", VisualizationConfig.marker_size)),
        ),
    )

    cfg.use_models = _as_tuple(raw.get("use_models"), cfg.use_models)
    cfg.metric = str(raw.get("metric", cfg.metric))

    cfg.trimap_n_inliers = int(trimap_cfg.get("n_inliers", cfg.trimap_n_inliers))
    cfg.trimap_n_outliers = int(trimap_cfg.get("n_outliers", cfg.trimap_n_outliers))
    cfg.trimap_n_random = int(trimap_cfg.get("n_random", cfg.trimap_n_random))
    cfg.trimap_n_iters = int(trimap_cfg.get("n_iters", cfg.trimap_n_iters))

    cfg.trimap_grid_n_inliers = _as_tuple(trimap_grid_cfg.get("n_inliers"), cfg.trimap_grid_n_inliers)
    cfg.trimap_grid_n_outliers = _as_tuple(trimap_grid_cfg.get("n_outliers"), cfg.trimap_grid_n_outliers)
    cfg.trimap_grid_n_random = _as_tuple(trimap_grid_cfg.get("n_random"), cfg.trimap_grid_n_random)

    cfg.hdbscan_min_cluster_size = int(hdbscan_cfg.get("min_cluster_size", cfg.hdbscan_min_cluster_size))
    cfg.hdbscan_min_samples = hdbscan_cfg.get("min_samples", cfg.hdbscan_min_samples)
    cfg.hdbscan_pca_dim = hdbscan_cfg.get("pca_dim", cfg.hdbscan_pca_dim)
    cfg.hdbscan_metric = str(hdbscan_cfg.get("metric", cfg.hdbscan_metric))
    cfg.hdbscan_selection_method = str(hdbscan_cfg.get("selection_method", cfg.hdbscan_selection_method))
    cfg.hdbscan_cluster_selection_epsilon = float(
        hdbscan_cfg.get("cluster_selection_epsilon", cfg.hdbscan_cluster_selection_epsilon)
    )
    cfg.hdbscan_silhouette_eval_pca_dim = hdbscan_cfg.get(
        "silhouette_eval_pca_dim", cfg.hdbscan_silhouette_eval_pca_dim
    )
    cfg.hdbscan_silhouette_eval_metric = hdbscan_cfg.get(
        "silhouette_eval_metric", cfg.hdbscan_silhouette_eval_metric
    )

    cfg.hdbscan_grid_min_cluster_size = _as_tuple(
        hdbscan_grid_cfg.get("min_cluster_size"), cfg.hdbscan_grid_min_cluster_size
    )
    cfg.hdbscan_grid_min_samples = _as_tuple(
        hdbscan_grid_cfg.get("min_samples"), cfg.hdbscan_grid_min_samples
    )
    cfg.hdbscan_grid_selection_method = _as_tuple(
        hdbscan_grid_cfg.get("selection_method"), cfg.hdbscan_grid_selection_method
    )
    cfg.hdbscan_grid_selection_epsilon = _as_tuple(
        hdbscan_grid_cfg.get("cluster_selection_epsilon"), cfg.hdbscan_grid_selection_epsilon
    )
    cfg.hdbscan_grid_allow_single_cluster = _as_tuple(
        hdbscan_grid_cfg.get("allow_single_cluster"), cfg.hdbscan_grid_allow_single_cluster
    )
    cfg.hdbscan_grid_metric = _as_tuple(hdbscan_grid_cfg.get("metric"), cfg.hdbscan_grid_metric)
    cfg.hdbscan_grid_pca_dim = _as_tuple(hdbscan_grid_cfg.get("pca_dim"), cfg.hdbscan_grid_pca_dim)

    cfg.top_k_per_cluster = int(summarization_cfg.get("top_k_per_cluster", cfg.top_k_per_cluster))

    return cfg
