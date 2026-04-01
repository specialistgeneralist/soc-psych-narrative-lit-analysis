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
    # Optional SciPy dependency for convex hulls and KDE contours
    from scipy.spatial import ConvexHull  # type: ignore
    from scipy.stats import gaussian_kde  # type: ignore
    _HAS_SCIPY = True
except Exception:  # pragma: no cover - safe fallback
    _HAS_SCIPY = False

from .config import PipelineConfig


# --- Visualization stage ---
# Render the 2D embedding with cluster coloring and optional cluster boundary overlays (hulls or KDE contours).


def _lighten_color(color: tuple[float, float, float], amount: float = 0.4) -> tuple[float, float, float]:
    """
    Lighten a color by mixing it with white.

    amount=0 keeps color unchanged; amount=1 makes it white.
    """
    r, g, b = mcolors.to_rgb(color)
    return (r + (1 - r) * amount, g + (1 - g) * amount, b + (1 - b) * amount)


def _normalize_percentile_mass(value: float | None) -> float | None:
    # Accept either a fraction (0-1) or a percentile (0-100) and normalize to (0, 1).
    if value is None:
        return None
    v = float(value)
    if v <= 0:
        raise ValueError("Visualization percentile must be > 0.")
    if v > 1.0:
        v = v / 100.0
    if not (0.0 < v < 1.0):
        raise ValueError("Visualization percentile must be between 0 and 1, or between 0 and 100.")
    return v


def _draw_convex_hull(
    ax,
    pts: np.ndarray,
    base_color: tuple[float, float, float],
    hull_alpha: float,
    hull_edge_alpha: float,
) -> None:
    # Convex hull overlays are best-effort: they require SciPy and at least 3 non-collinear points.
    if not _HAS_SCIPY or pts.shape[0] < 3:
        return

    try:
        hull = ConvexHull(pts)
        hull_pts = pts[hull.vertices]
    except Exception:
        return

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


def _draw_kde_percentile_contour(
    ax,
    pts: np.ndarray,
    base_color: tuple[float, float, float],
    mass: float,
    contour_linewidth: float = 1.6,
    contour_alpha: float = 0.95,
    gridsize: int = 200,
) -> None:
    """Draw a KDE contour enclosing approximately `mass` probability mass.

    The contour level is chosen so that the integral of the estimated density over
    the super-level set is approximately equal to `mass`.
    """
    if not _HAS_SCIPY or pts.shape[0] < 5:
        return

    x = pts[:, 0]
    y = pts[:, 1]

    xmin, xmax = float(np.min(x)), float(np.max(x))
    ymin, ymax = float(np.min(y)), float(np.max(y))

    xr = xmax - xmin
    yr = ymax - ymin
    pad_x = max(0.1 * xr, 1e-6)
    pad_y = max(0.1 * yr, 1e-6)

    # Guard against degenerate point clouds
    if xr <= 0:
        xmin -= 0.5
        xmax += 0.5
    else:
        xmin -= pad_x
        xmax += pad_x

    if yr <= 0:
        ymin -= 0.5
        ymax += 0.5
    else:
        ymin -= pad_y
        ymax += pad_y

    try:
        kde = gaussian_kde(pts.T)
    except Exception:
        # Singular covariance or other numerical issue; skip gracefully.
        return

    xx, yy = np.meshgrid(
        np.linspace(xmin, xmax, gridsize),
        np.linspace(ymin, ymax, gridsize),
    )
    grid = np.vstack([xx.ravel(), yy.ravel()])

    try:
        zz = kde(grid).reshape(xx.shape)
    except Exception:
        return

    dx = float((xmax - xmin) / max(gridsize - 1, 1))
    dy = float((ymax - ymin) / max(gridsize - 1, 1))
    cell_area = dx * dy

    z_flat = zz.ravel()
    order = np.argsort(z_flat)[::-1]
    z_sorted = z_flat[order]
    cum_mass = np.cumsum(z_sorted * cell_area)

    total_mass = float(cum_mass[-1]) if cum_mass.size else 0.0
    if total_mass <= 0:
        return

    target_mass = mass * total_mass
    idx = int(np.searchsorted(cum_mass, target_mass, side="left"))
    idx = min(max(idx, 0), len(z_sorted) - 1)
    level = float(z_sorted[idx])

    if not np.isfinite(level):
        return

    try:
        ax.contour(
            xx,
            yy,
            zz,
            levels=[level],
            colors=[mcolors.to_rgba(base_color, alpha=contour_alpha)],
            linewidths=contour_linewidth,
            zorder=2,
        )
    except Exception:
        return


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
    contour_mass_percentile: float | None = None,
) -> None:
    """
    Plot a 2D embedding with cluster coloring.

    Enhancements:
    - Optionally draw a semi-transparent convex hull for each non-noise cluster.
    - Optionally draw a KDE contour boundary enclosing a target probability mass
      for each non-noise cluster instead of a convex hull.
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
        If True and SciPy is available, draw a convex hull per cluster (label != -1)
        when `contour_mass_percentile` is not provided.
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
    contour_mass_percentile: float | None
        If provided, draw a KDE contour enclosing this fraction of probability mass
        (e.g. 0.9 or 90 for a 90% contour) instead of convex hulls.
    """

    contour_mass = _normalize_percentile_mass(contour_mass_percentile)

    df = pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1], "label": labels, "id": ids})

    plt.figure(figsize=(8, 6), dpi=160)

    # Color palette: -1 as grey
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    palette = sns.color_palette("tab20", n_colors=max(n_clusters, 1))

    # Build a deterministic label->color mapping so plots remain comparable across runs.
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

    # Draw cluster boundaries (excluding noise)
    for lab in unique_labels:
        if lab == -1:
            continue
        pts = df.loc[df.label == lab, ["x", "y"]].to_numpy()
        base_color = color_map[int(lab)]

        if contour_mass is not None:
            _draw_kde_percentile_contour(ax, pts, base_color, mass=contour_mass)
        elif draw_hulls:
            _draw_convex_hull(ax, pts, base_color, hull_alpha=hull_alpha, hull_edge_alpha=hull_edge_alpha)

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
