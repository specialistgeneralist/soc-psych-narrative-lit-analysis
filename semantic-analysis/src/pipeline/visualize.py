from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.patches import Polygon
from matplotlib import patheffects as path_effects
from matplotlib import colors as mcolors

try:
    # Optional dependency for convex hulls
    from scipy.spatial import ConvexHull  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - safe fallback
    _HAS_SCIPY = False

from .config import PipelineConfig


def _lighten_color(color: tuple[float, float, float], amount: float = 0.4) -> tuple[float, float, float]:
    """
    Lighten a color by mixing it with white.

    amount=0 keeps color unchanged; amount=1 makes it white.
    """
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def plot_embedding(
    cfg: PipelineConfig,
    Y: np.ndarray,
    labels: np.ndarray,
    ids: list[str],
    out_path: Path,
    title: str = "TriMAP 2D embedding with HDBSCAN clusters",
    *,
    draw_hulls: bool = True,
    hull_alpha: float = 0.15,
    hull_edge_alpha: float = 0.7,
    label_clusters: bool = True,
    label_fontsize: int = 9,
    label_with_size: bool = False,
) -> None:
    """
    Plot a 2D embedding with cluster coloring.

    Enhancements:
    - Optionally draw a semi-transparent convex hull for each non-noise cluster.
    - Optionally place a cluster label at the centroid (skip noise label -1).

    Parameters
    ----------
    cfg: PipelineConfig
        Pipeline configuration (unused but kept for interface consistency).
    Y: np.ndarray, shape (n_samples, 2)
        2D embedding coordinates (TriMAP output).
    labels: np.ndarray, shape (n_samples,)
        Cluster labels. Noise should be labelled as -1.
    ids: list[str]
        Identifiers for points; exported to CSV alongside coordinates.
    out_path: Path
        Path to save the figure (png/pdf) and a CSV with coordinates.
    title: str
        Plot title.
    draw_hulls: bool
        If True and SciPy is available, draw a convex hull per cluster (label != -1).
    hull_alpha: float
        Face alpha for hull polygons.
    hull_edge_alpha: float
        Edge alpha for hull polygons.
    label_clusters: bool
        If True, annotate cluster centroids (label != -1).
    label_fontsize: int
        Font size for cluster labels.
    label_with_size: bool
        If True, show cluster size next to label id, e.g. "5 (n=123)".
    """

    df = pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1], "label": labels, "id": ids})

    plt.figure(figsize=(8, 6), dpi=160)

    # Color palette: -1 as grey
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    palette = sns.color_palette("tab20", n_colors=max(n_clusters, 1))

    color_map: dict[int, tuple[float, float, float]] = {}
    color_iter = iter(palette)
    for lab in unique_labels:
        if lab == -1:
            color_map[lab] = (0.6, 0.6, 0.6)
        else:
            color_map[lab] = next(color_iter)

    colors = [color_map[int(l)] for l in labels]
    plt.scatter(df["x"], df["y"], c=colors, s=20, alpha=0.9, edgecolor="none", zorder=10)

    ax = plt.gca()

    # Draw convex hulls per cluster (excluding noise) if requested and SciPy available
    if draw_hulls and _HAS_SCIPY:
        for lab in unique_labels:
            if lab == -1:
                continue
            pts = df.loc[df.label == lab, ["x", "y"]].to_numpy()
            if pts.shape[0] < 3:
                # Too few points for a hull; skip
                continue
            try:
                hull = ConvexHull(pts)
                hull_pts = pts[hull.vertices]
            except Exception:
                # If hull fails for any numeric reason, skip drawing
                continue

            base_color = color_map[int(lab)]
            face_color = _lighten_color(base_color, amount=0.55)
            poly = Polygon(
                hull_pts,
                closed=True,
                facecolor=mcolors.to_rgba(face_color, alpha=hull_alpha),
                edgecolor=mcolors.to_rgba(base_color, alpha=hull_edge_alpha),
                linewidth=1.2,
                zorder=1,
            )
            ax.add_patch(poly)

    # Label cluster centroids (excluding noise)
    if label_clusters:
        for lab in unique_labels:
            if lab == -1:
                continue
            pts = df.loc[df.label == lab, ["x", "y"]].to_numpy()
            if pts.shape[0] == 0:
                continue
            centroid = pts.mean(axis=0)
            label_text = f"{lab}"
            if label_with_size:
                label_text = f"{lab} (n={pts.shape[0]})"

            txt = ax.text(
                centroid[0],
                centroid[1],
                label_text,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color="black",
                weight="bold",
                zorder=20,
            )
            # White outline for legibility
            txt.set_path_effects([
                path_effects.Stroke(linewidth=3.0, foreground="white", alpha=0.9),
                path_effects.Normal(),
            ])

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()

    # Also save CSV of coordinates
    csv_path = out_path.with_suffix(".csv")
    df.to_csv(csv_path, index=False)
