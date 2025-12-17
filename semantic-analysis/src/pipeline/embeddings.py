from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

from .config import PipelineConfig
from .utils import ensure_dirs, normalize_rows, text_hash


@dataclass
class EmbeddingRecord:
    id: str
    text: str
    text_hash: str
    model: str
    vector: list[float]


def _batched(iterable: Iterable, n: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch


def _openai_embed(client: OpenAI, model: str, texts: list[str]) -> list[list[float]]:
    # OpenAI embeddings API expects a list of inputs
    resp = client.embeddings.create(model=model, input=texts)
    # Ensure order
    vectors = [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]
    return vectors


def embed_texts_openai(cfg: PipelineConfig, texts: list[str], model: str, batch_size: int = 96, pause: float = 0.5) -> list[list[float]]:
    client = OpenAI()
    vectors: list[list[float]] = []
    for batch in _batched(texts, batch_size):
        backoff = 1.0
        while True:
            try:
                vecs = _openai_embed(client, model, batch)
                vectors.extend(vecs)
                time.sleep(pause)
                break
            except Exception as e:
                # Simple exponential backoff
                print(f"Embedding API error: {e}. Retrying in {backoff:.1f}s...")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
    return vectors


def compute_and_store_embeddings(
    tag: str,
    cfg: PipelineConfig,
    df: pd.DataFrame,
    model_key: str = "small",
    overwrite: bool = False,
) -> Path:
    assert model_key in ("small", "large"), "model_key must be 'small' or 'large'"
    model = cfg.models.openai_small if model_key == "small" else cfg.models.openai_large

    ensure_dirs(cfg.paths.embeddings_dir)

    # Prepare identifiers
    id_col = cfg.data.id_column
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]

    texts = df[cfg.data.text_column].astype(str).tolist()
    hashes = [text_hash(t) for t in texts]

    out_path = cfg.paths.embeddings_dir / f"{tag}_embeddings_{model_key}.parquet"

    if out_path.exists() and not overwrite:
        print(f"Found existing embeddings at {out_path}. Skipping recompute (use overwrite=True to force).")
        return out_path

    print(f"Computing embeddings with model={model} for n={len(texts)} texts ...")
    vectors = embed_texts_openai(cfg, texts, model=model)

    # Normalize for cosine usage
    X = np.array(vectors, dtype=np.float32)
    X_unit = normalize_rows(X)

    recs = pd.DataFrame(
        {
            "id": ids,
            "text": texts,
            "text_hash": hashes,
            "model": model,
            "vector": vectors,
            "vector_unit": list(X_unit.tolist()),
        }
    )

    recs.to_parquet(out_path, index=False)
    print(f"Saved embeddings to {out_path}")
    return out_path


def load_embeddings(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df
