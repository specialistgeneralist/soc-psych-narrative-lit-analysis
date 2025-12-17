from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Dict, List

import numpy as np
import pandas as pd
from anthropic import Anthropic

from .config import PipelineConfig
from .utils import nearest_to_centroid


@dataclass
class ClusterSummary:
    cluster_id: int
    indices: List[int]
    representative_indices: List[int]
    summary_text: str


def load_prompt_template(path: Path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(template_md: str, cluster_id: int, items: List[str], k: int) -> str:
    items_block = "\n".join([f"- {s}" for s in items])
    prompt = (
        template_md.replace("{{cluster_id}}", str(cluster_id))
        .replace("{{cluster_size}}", str(len(items)))
        .replace("{{k}}", str(k))
        .replace("{{items_block}}", items_block)
    )
    return prompt


def summarize_clusters(
    cfg: PipelineConfig,
    embeddings_unit: np.ndarray,
    df_embeddings: pd.DataFrame,
    labels: np.ndarray,
    prompt_path: Path,
    out_dir: Path,
    k: int = 3,
) -> List[ClusterSummary]:
    client = Anthropic()

    unique_clusters = sorted([c for c in set(labels) if c != -1])
    prompt_template = load_prompt_template(prompt_path)

    summaries: List[ClusterSummary] = []

    for c in unique_clusters:
        idx = np.where(labels == c)[0]
        rep = nearest_to_centroid(embeddings_unit, idx, k=k, metric="cosine")
        items = df_embeddings.iloc[rep]["text"].tolist()

        prompt = render_prompt(prompt_template, cluster_id=c, items=items, k=k)

        msg = client.messages.create(
            model=cfg.models.anthropic_model,
            max_tokens=600,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        # Anthropics python SDK returns content list
        summary_text = "".join([blk.text for blk in msg.content if hasattr(blk, "text")])

        summaries.append(
            ClusterSummary(
                cluster_id=c,
                indices=idx.tolist(),
                representative_indices=rep.tolist(),
                summary_text=summary_text,
            )
        )

        # Save per-cluster summary
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"cluster_{c:02d}_summary.md"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

    return summaries
