from __future__ import annotations

from pathlib import Path
from typing import Literal

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


# For the typical use-case here (roughly <= 5 substantive clusters), an explicit
# high-contrast palette works better than sampling from a larger categorical map
# like tab20, especially once fills are lightened and drawn transparently.
# The colors below are chosen to stay distinct on a white background and remain
# reasonably robust for color-vision deficiencies.
_DEFAULT_CLUSTER_PALETTE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#B79F00",  # mustard / dark yellow
]

# Step through marker shapes alongside cluster colours so clusters remain
# distinguishable in greyscale, for colour-vision deficiencies, and in dense
# regions where adjacent colours can be hard to separate. These are all common
# filled Matplotlib markers chosen for visibly different silhouettes.
_DEFAULT_CLUSTER_MARKERS = [
    "o",  # circle
    "s",  # square
    "D",  # diamond
    "^",  # triangle up
    ">",  # triangle right
    "v",  # triangle down
    "<",  # triangle left
    "h",  # hexagon
    "P",  # plus filled
    "p",  # pentagon
    "*",  # star
    "X",  # x filled
]

ClusterBoundaryStyle = Literal["density", "hull", "none"]


# --- Visualization stage ---
# Render the 2D embedding with cluster coloring and optional cluster density / legacy hull overlays.


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


def _draw_kde_density_region(
    ax,
    pts: np.ndarray,
    base_color: tuple[float, float, float],
    mass: float,
    fill_alpha: float,
    gridsize: int = 200,
) -> None:
    """Shade the KDE super-level region enclosing approximately `mass` probability mass.

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

    zmax = float(np.max(zz))
    if not np.isfinite(zmax) or level >= zmax:
        return

    face_color = _lighten_color(base_color, amount=0.55)

    try:
        ax.contourf(
            xx,
            yy,
            zz,
            levels=[level, zmax],
            colors=[mcolors.to_rgba(face_color, alpha=fill_alpha)],
            antialiased=True,
            zorder=1,
        )
    except Exception:
        return


def _resolve_marker_size(cfg: PipelineConfig, marker_size: float | None) -> float:
    if marker_size is not None:
        resolved = float(marker_size)
    else:
        visualization_cfg = getattr(cfg, "visualization", None)
        resolved = float(getattr(visualization_cfg, "marker_size", 20.0))
    if resolved <= 0:
        raise ValueError("Visualization marker size must be > 0.")
    return resolved


def plot_embedding(
    cfg: PipelineConfig,
    Y: np.ndarray,
    labels: np.ndarray,
    ids: list[str],
    out_path: Path,
    title: str = "TriMAP 2D embedding with HDBSCAN clusters",
    *,
    draw_hulls: bool = True,
    boundary_style: ClusterBoundaryStyle = "density",
    hull_alpha: float = 0.15,
    hull_edge_alpha: float = 0.7,
    density_alpha: float = 0.15,
    density_percentile: float = 75.0,
    label_clusters: bool = True,
    label_fontsize: int = 12,
    label_with_size: bool = False,
    contour_mass_percentile: float | None = None,
    marker_size: float | None = None,
) -> None:
    """
    Plot a 2D embedding with cluster coloring.

    Enhancements:
    - Optionally draw a semi-transparent convex hull for each non-noise cluster.
    - By default, shade the 2D KDE density region enclosing a target probability
      mass for each non-noise cluster.
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
        Legacy compatibility flag. Hulls are only drawn when `boundary_style="hull"`
        and this is True.
    boundary_style: {"density", "hull", "none"}
        Cluster overlay style. "density" shades a KDE super-level region,
        "hull" uses the legacy convex hull polygon, and "none" disables overlays.
    hull_alpha: float
        Face alpha for hull polygons.
    hull_edge_alpha: float
        Edge alpha for hull polygons.
    density_alpha: float
        Face alpha for density-region shading.
    density_percentile: float
        Density contour mass percentile to shade for each cluster. Accepts either
        0-1 (e.g. 0.75) or 0-100 (e.g. 75). Higher values shade more area.
    label_clusters: bool
        If True, annotate cluster centroids (label != -1).
    label_fontsize: int
        Font size for cluster labels.
    label_with_size: bool
        If True, show cluster size next to label id, e.g. "5 (n=123)".
    contour_mass_percentile: float | None
        Backwards-compatible alias for `density_percentile`. If provided, selects
        `boundary_style="density"` and shades the corresponding KDE mass region.
    marker_size: float | None
        Matplotlib scatter marker size. If None, use `cfg.visualization.marker_size`
        when available, falling back to 20.0.
    """

    if boundary_style not in ("density", "hull", "none"):
        raise ValueError("boundary_style must be one of: 'density', 'hull', 'none'.")

    if contour_mass_percentile is not None:
        boundary_style = "density"
        density_percentile = contour_mass_percentile

    density_mass = _normalize_percentile_mass(density_percentile) if boundary_style == "density" else None

    df = pd.DataFrame({"x": Y[:, 0], "y": Y[:, 1], "label": labels, "id": ids})

    plt.figure(figsize=(8, 6), dpi=160)

    # Color palette: -1 as grey
    unique_labels = sorted(set(labels))
    n_clusters = len([l for l in unique_labels if l != -1])
    if n_clusters <= len(_DEFAULT_CLUSTER_PALETTE):
        palette = sns.color_palette(_DEFAULT_CLUSTER_PALETTE[: max(n_clusters, 1)])
    else:
        # Fall back gracefully if a future run produces more than the usual number
        # of clusters. tab20 is still acceptable in that less common case.
        palette = sns.color_palette("tab20", n_colors=max(n_clusters, 1))

    # Build a deterministic label->color mapping so plots remain comparable across runs.
    color_map: dict[int, tuple[float, float, float]] = {}
    marker_map: dict[int, str] = {}
    non_noise_labels = [lab for lab in unique_labels if lab != -1]
    for idx, lab in enumerate(non_noise_labels):
        color_map[int(lab)] = palette[idx]
        marker_map[int(lab)] = _DEFAULT_CLUSTER_MARKERS[idx % len(_DEFAULT_CLUSTER_MARKERS)]

    for lab in unique_labels:
        if lab == -1:
            color_map[lab] = (0.6, 0.6, 0.6)
            marker_map[lab] = "x"

    resolved_marker_size = _resolve_marker_size(cfg, marker_size)

    ax = plt.gca()

    # Draw one scatter layer per cluster so each cluster receives a stable
    # colour-marker pair. Noise remains grey and uses an x marker.
    for lab in unique_labels:
        cluster_df = df.loc[df.label == lab]
        if cluster_df.empty:
            continue
        if lab == -1:
            ax.scatter(
                cluster_df["x"],
                cluster_df["y"],
                c=[color_map[int(lab)]],
                s=resolved_marker_size,
                marker='x',
                alpha=0.65,
                linewidths=0.8,
                zorder=10,
            )
        else:
            this_marker = marker_map[int(lab)]
            # check if marker is not one of the basic three o, s, D, and scale if so.
            if this_marker not in ("o", "s", "D"):
                this_marker_size = resolved_marker_size * 2 
            else:
                this_marker_size = resolved_marker_size
            ax.scatter(
                cluster_df["x"],
                cluster_df["y"],
                c=[color_map[int(lab)]],
                s=this_marker_size,
                marker=this_marker,
                alpha=0.9,
                edgecolors="white",
                linewidths=0.35,
                zorder=10,
            )

    # Draw cluster boundaries (excluding noise)
    for lab in unique_labels:
        if lab == -1:
            continue
        pts = df.loc[df.label == lab, ["x", "y"]].to_numpy()
        base_color = color_map[int(lab)]

        if density_mass is not None:
            _draw_kde_density_region(ax, pts, base_color, mass=density_mass, fill_alpha=density_alpha)
        elif boundary_style == "hull" and draw_hulls:
            _draw_convex_hull(ax, pts, base_color, hull_alpha=hull_alpha, hull_edge_alpha=hull_edge_alpha)

    # Label cluster centroids (excluding noise)
    if label_clusters:
        for lab in unique_labels:
            if lab == -1:
                continue
            pts = df.loc[df.label == lab, ["x", "y"]].to_numpy()
            base_color = color_map[int(lab)]
            if pts.shape[0] == 0:
                continue
            centroid = pts.mean(axis=0)
            # adjust label text to 1-index, rather than 0-index, for human readability
            label_text = f"{lab + 1}"
            if label_with_size:
                label_text = f"{lab + 1} (n={pts.shape[0]})"

            txt = ax.text(
                centroid[0],
                centroid[1],
                label_text,
                ha="center",
                va="center",
                fontsize=label_fontsize,
                color=base_color,
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
