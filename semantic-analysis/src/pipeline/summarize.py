from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from anthropic import Anthropic

from .config import PipelineConfig
from .utils import nearest_to_centroid

# --- Cluster summarization (LLM) ---
# This module turns clusters (label assignments) into short written summaries by:
# 1) selecting representative items per cluster from the embedding space, then
# 2) prompting an LLM with a Markdown template.


@dataclass
class ClusterSummary:
    # Persisted per cluster; indices refer to rows in the original embeddings dataframe.
    cluster_id: int
    indices: List[int]
    representative_indices: List[int]
    summary_text: str


def load_prompt_template(path: Path) -> str:
    # Keep templates as plain Markdown with {{placeholders}}.
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def render_prompt(template_md: str, cluster_id: int, cluster_size: int, items: List[str], k: int) -> str:
    # Format representative snippets as a Markdown bullet list for the LLM to read.
    items_block = "\n".join([f"- {s}" for s in items])
    prompt = (
        template_md.replace("{{cluster_id}}", str(cluster_id))
        .replace("{{cluster_size}}", str(cluster_size))
        .replace("{{k}}", str(k))
        .replace("{{items_block}}", items_block)
    )
    return prompt


def render_meta_prompt(template_md: str, summaries: List[ClusterSummary]) -> str:
    # Concatenate all cluster summaries into one "meta" prompt that asks for higher-level synthesis.
    summary_blocks: List[str] = []
    for i, summary in enumerate(summaries, start=1):
        summary_blocks.append(
            "\n".join(
                [
                    f"## Definition summary {i}",
                    f"Cluster ID: {summary.cluster_id}",
                    f"Cluster size: {len(summary.indices)}",
                    f"Representative excerpts summarized upstream: {len(summary.representative_indices)}",
                    "",
                    summary.summary_text.strip(),
                ]
            )
        )

    summaries_block = "\n\n".join(summary_blocks)
    return template_md.rstrip() + "\n\nDefinition summaries:\n\n" + summaries_block + "\n"


def extract_text_content(message) -> str:
    # Anthropic messages return a list of content blocks; we keep only text blocks.
    return "".join([blk.text for blk in message.content if hasattr(blk, "text")]).strip()


def write_meta_summary(
    client: Anthropic,
    cfg: PipelineConfig,
    summaries: List[ClusterSummary],
    meta_prompt_path: Path,
    out_dir: Path,
    tag: str,
) -> Path:
    # Produce a single top-level document summarizing all non-noise clusters.
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tag}_meta_summary.md"

    if not summaries:
        out_path.write_text(
            "No non-noise cluster summaries were generated, so no meta-summary could be produced.\n",
            encoding="utf-8",
        )
        return out_path

    meta_template = load_prompt_template(meta_prompt_path)
    prompt = render_meta_prompt(meta_template, summaries)

    msg = client.messages.create(
        model=cfg.models.anthropic_model,
        max_tokens=3000,
        temperature=0.1,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    meta_summary_text = extract_text_content(msg)
    out_path.write_text(meta_summary_text, encoding="utf-8")
    return out_path


def summarize_clusters(
    cfg: PipelineConfig,
    embeddings_unit: np.ndarray,
    df_embeddings: pd.DataFrame,
    labels: np.ndarray,
    prompt_path: Path,
    out_dir: Path,
    tag: str,
    meta_prompt_path: Optional[Path] = None,
) -> List[ClusterSummary]:
    # Main entry point: generate per-cluster summaries, then a meta-summary across clusters.
    client = Anthropic()

    # Ignore HDBSCAN noise points (-1); only summarize "real" clusters.
    unique_clusters = sorted([c for c in set(labels) if c != -1])
    prompt_template = load_prompt_template(prompt_path)
    resolved_meta_prompt_path = meta_prompt_path or (prompt_path.parent / "meta_synthesis_prompt.md")

    summaries: List[ClusterSummary] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for c in unique_clusters:
        idx = np.where(labels == c)[0]

        # Pick representative items by proximity to the cluster centroid in embedding space.
        rep = nearest_to_centroid(embeddings_unit, idx, k=cfg.top_k_per_cluster, metric="cosine")
        items = df_embeddings.iloc[rep]["text"].tolist()

        prompt = render_prompt(prompt_template, cluster_id=c, cluster_size=len(idx), items=items, k=cfg.top_k_per_cluster)

        msg = client.messages.create(
            model=cfg.models.anthropic_model,
            max_tokens=3000,
            temperature=0.1,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        summary_text = extract_text_content(msg)

        summary = ClusterSummary(
            cluster_id=c,
            indices=idx.tolist(),
            representative_indices=rep.tolist(),
            summary_text=summary_text,
        )
        summaries.append(summary)

        out_path = out_dir / f"{tag}_cluster_{c:02d}_summary.md"
        out_path.write_text(summary_text, encoding="utf-8")

    # Meta-synthesis is intentionally strict: the pipeline expects the prompt file to exist.
    if resolved_meta_prompt_path.exists():
        write_meta_summary(client, cfg, summaries, resolved_meta_prompt_path, out_dir, tag)
    else:
        raise FileNotFoundError(f"Meta synthesis prompt not found: {resolved_meta_prompt_path}")

    return summaries
