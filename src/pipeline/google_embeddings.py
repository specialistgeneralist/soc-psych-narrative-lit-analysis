from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path as _Path
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .utils import ensure_dirs, normalize_rows, text_hash


@dataclass(frozen=True)
class VertexEmbeddingSettings:
    """Settings for Google Vertex AI text embeddings."""

    project: str
    location: str = "us-central1"
    model: str = "gemini-embedding-001"
    task_type: str = "CLUSTERING"
    output_dimensionality: int | None = 3072
    auto_truncate: bool = True
    model_key: str = "google_gemini_embedding_001"


def _candidate_gcloud_paths() -> list[str]:
    paths: list[str] = []

    env_gcloud = os.getenv("GCLOUD_BIN")
    if env_gcloud:
        paths.append(env_gcloud)

    which_gcloud = shutil.which("gcloud")
    if which_gcloud:
        paths.append(which_gcloud)

    home = _Path.home()
    common = [
        home / "google-cloud-sdk" / "bin" / "gcloud",
        home / "bin" / "gcloud",
        _Path("/opt/homebrew/bin/gcloud"),
        _Path("/usr/local/bin/gcloud"),
    ]
    paths.extend(str(p) for p in common)

    seen: set[str] = set()
    out: list[str] = []
    for p in paths:
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _find_gcloud() -> str:
    for candidate in _candidate_gcloud_paths():
        if os.path.exists(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "Could not find `gcloud`. Set GCLOUD_BIN explicitly or ensure gcloud is installed "
        "in a standard location such as ~/google-cloud-sdk/bin/gcloud."
    )


def _run_quiet(argv: list[str]) -> str:
    proc = subprocess.run(
        argv,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return proc.stdout.strip()


def infer_google_project(explicit_project: str | None = None) -> str:
    """Resolve the Google Cloud project without reading or printing credentials."""

    if explicit_project:
        return explicit_project

    for env_name in ("GOOGLE_CLOUD_PROJECT", "GOOGLE_PROJECT_ID", "GCLOUD_PROJECT"):
        value = os.getenv(env_name)
        if value:
            return value

    try:
        value = _run_quiet([_find_gcloud(), "config", "get-value", "project"])
        if value and value != "(unset)":
            return value
    except Exception:
        pass

    raise RuntimeError(
        "Google Cloud project not configured. Set GOOGLE_CLOUD_PROJECT in .env/environment "
        "or run `gcloud config set project PROJECT_ID`."
    )


def _get_access_token() -> str:
    """Get an OAuth access token using local ADC, falling back to gcloud user auth."""

    gcloud = _find_gcloud()
    commands = [
        [gcloud, "auth", "application-default", "print-access-token"],
        [gcloud, "auth", "print-access-token"],
    ]
    errors: list[str] = []
    for argv in commands:
        try:
            token = _run_quiet(argv)
            if token:
                return token
        except Exception as exc:
            errors.append(f"{' '.join(argv)}: {exc}")
    raise RuntimeError("Unable to obtain Google access token via gcloud. " + " | ".join(errors))


def _vertex_predict_url(settings: VertexEmbeddingSettings) -> str:
    return (
        f"https://{settings.location}-aiplatform.googleapis.com/v1/"
        f"projects/{settings.project}/locations/{settings.location}/publishers/google/"
        f"models/{settings.model}:predict"
    )


def _request_embedding(settings: VertexEmbeddingSettings, text: str, token: str) -> list[float]:
    instance: dict[str, Any] = {
        "content": text,
        "task_type": settings.task_type,
    }
    parameters: dict[str, Any] = {"autoTruncate": settings.auto_truncate}
    if settings.output_dimensionality is not None:
        parameters["outputDimensionality"] = int(settings.output_dimensionality)

    payload = json.dumps({"instances": [instance], "parameters": parameters}).encode("utf-8")
    request = Request(
        _vertex_predict_url(settings),
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
        },
        method="POST",
    )

    with urlopen(request, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))

    predictions = data.get("predictions") or []
    if not predictions:
        raise RuntimeError(f"Vertex embedding response had no predictions: {data}")
    embeddings = predictions[0].get("embeddings") or {}
    values = embeddings.get("values")
    if not values:
        raise RuntimeError(f"Vertex embedding response had no embedding values: {data}")
    return [float(x) for x in values]


def embed_texts_vertex(
    texts: list[str],
    settings: VertexEmbeddingSettings,
    *,
    pause: float = 0.1,
    max_retries: int = 5,
) -> list[list[float]]:
    """Embed texts through Vertex AI.

    `gemini-embedding-001` currently accepts a single input per predict request, so this
    function intentionally sends one text at a time and retries transient failures.
    """

    vectors: list[list[float]] = []
    token = _get_access_token()
    for i, text in enumerate(texts, start=1):
        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                vectors.append(_request_embedding(settings, text, token))
                time.sleep(pause)
                break
            except HTTPError as exc:
                # Refresh token once if it expired or was rejected.
                if exc.code in (401, 403) and attempt == 1:
                    token = _get_access_token()
                if attempt == max_retries:
                    body = exc.read().decode("utf-8", errors="replace")
                    raise RuntimeError(f"Vertex embedding HTTP {exc.code} for item {i}: {body}") from exc
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
            except URLError as exc:
                if attempt == max_retries:
                    raise RuntimeError(f"Vertex embedding URL error for item {i}: {exc}") from exc
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)
    return vectors


def compute_and_store_vertex_embeddings(
    cfg: PipelineConfig,
    df: pd.DataFrame,
    settings: VertexEmbeddingSettings,
    *,
    overwrite: bool = False,
) -> Path:
    """Compute and cache Google Vertex embeddings using the same parquet schema as OpenAI embeddings."""

    ensure_dirs(cfg.paths.embeddings_dir)
    out_path = cfg.paths.embeddings_dir / f"embeddings_{settings.model_key}.parquet"
    if out_path.exists() and not overwrite:
        print(f"Found existing Vertex embeddings at {out_path}. Skipping recompute (use overwrite=True to force).")
        return out_path

    id_col = cfg.data.id_column
    if id_col and id_col in df.columns:
        ids = df[id_col].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]

    texts = df[cfg.data.text_column].astype(str).tolist()
    hashes = [text_hash(t) for t in texts]

    print(
        "Computing Vertex embeddings "
        f"model={settings.model} task_type={settings.task_type} "
        f"output_dimensionality={settings.output_dimensionality} n={len(texts)} ..."
    )
    vectors = embed_texts_vertex(texts, settings)
    X = np.array(vectors, dtype=np.float32)
    X_unit = normalize_rows(X)

    recs = pd.DataFrame(
        {
            "id": ids,
            "text": texts,
            "text_hash": hashes,
            "model": settings.model,
            "provider": "google_vertex",
            "task_type": settings.task_type,
            "output_dimensionality": settings.output_dimensionality,
            "vector": vectors,
            "vector_unit": list(X_unit.tolist()),
        }
    )
    recs.to_parquet(out_path, index=False)
    print(f"Saved Vertex embeddings to {out_path}")
    return out_path
