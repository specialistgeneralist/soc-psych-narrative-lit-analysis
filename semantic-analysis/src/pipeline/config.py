from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Tuple

from dotenv import load_dotenv


@dataclass
class Paths:
    input_tsv: Path
    output_dir: Path
    embeddings_dir: Path
    plots_dir: Path
    results_dir: Path
    summaries_dir: Path
    cache_dir: Path


@dataclass
class ModelConfig:
    openai_small: str = "text-embedding-3-small"
    openai_large: str = "text-embedding-3-large"
    anthropic_model: str = "claude-sonnet-4-5-20250929"


@dataclass
class DataConfig:
    text_column: str = "text_normalised"
    id_column: Optional[str] = "uid"


@dataclass
class PipelineConfig:
    seed: int = 42
    paths: Paths = None  # type: ignore
    models: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)

    # Defaults/toggles
    use_models: tuple[str, ...] = ("small", "large")  # which embeddings to compute
    # Use Euclidean by default; for L2-normalized embeddings, Euclidean distance is a monotonic transform of cosine distance
    metric: str = "euclidean"

    # TriMAP defaults
    trimap_n_inliers: int = 3
    trimap_n_outliers: int = 10
    trimap_n_random: int = 5

    # HDBSCAN defaults
    hdbscan_min_cluster_size: int = 3
    hdbscan_min_samples: Optional[int] = 2
    # Optional PCA before clustering (None means skip)
    hdbscan_pca_dim: Optional[int] = 2
    hdbscan_metric: str = "manhattan"
    hdbscan_selection_method: str = "leaf"  # 'eom' or 'leaf'
    hdbscan_cluster_selection_epsilon: float = 0.0

    # HDBSCAN tuning grids (used by grid_search_hdbscan)
    hdbscan_grid_min_cluster_size: Tuple[int, ...] = (5, 10, 15)
    hdbscan_grid_min_samples: Tuple[Optional[int], ...] = (1, 2)
    hdbscan_grid_selection_method: Tuple[str, ...] = ("leaf", "eom")
    hdbscan_grid_selection_epsilon: Tuple[float, ...] = (0.0, 0.01, 0.1)
    hdbscan_grid_allow_single_cluster: Tuple[bool, ...] = (False,)
    # Try a small set of metrics; on unit vectors, euclidean ~ cosine, but manhattan can behave differently
    hdbscan_grid_metric: Tuple[str, ...] = ("euclidean", "manhattan")
    # PCA dimensionalities to try before clustering (None = no PCA)
    hdbscan_grid_pca_dim: Tuple[Optional[int], ...] = (None, 2, 3, 4, 5)

    # Summarization
    top_k_per_cluster: int = 3

    def to_dict(self):
        d = asdict(self)
        d["paths"] = {k: str(v) for k, v in self.paths.__dict__.items()}
        return d


def load_config(env_path: Optional[str] = None) -> PipelineConfig:
    if env_path is None:
        load_dotenv(override=False)
    else:
        load_dotenv(env_path, override=False)

    # Paths
    input_tsv = Path(os.getenv("INPUT_TSV", "data/input.tsv"))
    output_dir = Path(os.getenv("OUTPUT_DIR", "outputs"))
    embeddings_dir = Path(os.getenv("EMBEDDINGS_DIR", output_dir / "embeddings"))
    plots_dir = Path(os.getenv("PLOTS_DIR", output_dir / "plots"))
    results_dir = Path(os.getenv("RESULTS_DIR", output_dir / "results"))
    summaries_dir = Path(os.getenv("SUMMARIES_DIR", output_dir / "summaries"))
    cache_dir = Path(os.getenv("CACHE_DIR", output_dir / "cache"))

    paths = Paths(
        input_tsv=input_tsv,
        output_dir=output_dir,
        embeddings_dir=Path(embeddings_dir),
        plots_dir=Path(plots_dir),
        results_dir=Path(results_dir),
        summaries_dir=Path(summaries_dir),
        cache_dir=Path(cache_dir),
    )

    models = ModelConfig(
        openai_small=os.getenv("OPENAI_EMBEDDING_MODEL_SMALL", "text-embedding-3-small"),
        openai_large=os.getenv("OPENAI_EMBEDDING_MODEL_LARGE", "text-embedding-3-large"),
        anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    )

    data = DataConfig(
        text_column=os.getenv("TEXT_COLUMN", "text_normalised"),
        id_column=os.getenv("ID_COLUMN", "uid") or None,
    )

    cfg = PipelineConfig(
        seed=int(os.getenv("SEED", 42)),
        paths=paths,
        models=models,
        data=data,
    )

    return cfg