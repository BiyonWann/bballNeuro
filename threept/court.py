"""Half-court drawing and court plots. All lengths are in feet with the basket at (0, 0)."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, LogNorm
from matplotlib.patches import Arc, Circle, Rectangle

BASELINE_Y = -5.25
HALF_COURT_Y = 41.75
SIDELINE_X = 25
CORNER_THREE_X = 22
ARC_RADIUS = 23.75
CORNER_BREAK_Y = np.sqrt(ARC_RADIUS**2 - CORNER_THREE_X**2)
PAINT_HALF_WIDTH = 8
FREE_THROW_Y = 13.75
RESTRICTED_RADIUS = 4
RIM_RADIUS = 0.75
FREE_THROW_RADIUS = 6
DRAWN_TOP_Y = 35
LINE_COLOR = "black"

# Cell i is always tab20 color i, so a cell keeps its color across plots, seeds and label sets.
CELL_COLORMAP = matplotlib.colormaps["tab20"]
MAX_CELLS = 20
# One bin per cell index: cell i falls between i - 0.5 and i + 0.5.
CELL_NORM = BoundaryNorm(np.arange(MAX_CELLS + 1) - 0.5, MAX_CELLS)
SEASON_MARKERS = ["o", "s", "^", "D", "v", "P"]
LEGEND_MARKER_COLOR = "gray"


def three_point_line(n_points: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """Return (xs, ys) for the 3-point line, from the left corner over the arc to the right.

    The arc gets `n_points` points and each corner adds one baseline point.
    """
    break_angle = np.arcsin(CORNER_THREE_X / ARC_RADIUS)
    angles = np.linspace(-break_angle, break_angle, n_points)
    arc_xs = ARC_RADIUS * np.sin(angles)
    arc_ys = ARC_RADIUS * np.cos(angles)
    xs = np.concatenate([[-CORNER_THREE_X], arc_xs, [CORNER_THREE_X]])
    ys = np.concatenate([[BASELINE_Y], arc_ys, [BASELINE_Y]])
    return xs, ys


def draw_half_court(ax=None):
    """Draw the half-court lines with equal aspect and return the Axes."""
    if ax is None:
        figure, ax = plt.subplots()
    paint_width = 2 * PAINT_HALF_WIDTH
    paint_height = FREE_THROW_Y - BASELINE_Y
    paint = Rectangle((-PAINT_HALF_WIDTH, BASELINE_Y), paint_width, paint_height, fill=False)
    free_throw_circle = Circle((0, FREE_THROW_Y), FREE_THROW_RADIUS, fill=False)
    restricted_diameter = 2 * RESTRICTED_RADIUS
    restricted_area = Arc((0, 0), restricted_diameter, restricted_diameter, theta2=180)
    rim = Circle((0, 0), RIM_RADIUS, fill=False)
    for shape in [paint, free_throw_circle, restricted_area, rim]:
        shape.set_edgecolor(LINE_COLOR)
        ax.add_patch(shape)
    line_xs, line_ys = three_point_line()
    ax.plot(line_xs, line_ys, color=LINE_COLOR)
    ax.set_xlim(-SIDELINE_X, SIDELINE_X)
    ax.set_ylim(BASELINE_Y, DRAWN_TOP_Y)
    ax.set_aspect("equal")
    return ax


def _axis_points(start: float, stop: float, step: float) -> np.ndarray:
    """Return points from `start` to `stop`, `step` apart, always ending exactly at `stop`."""
    points = np.arange(start, stop, step)
    return np.append(points, stop)


def court_grid(step_ft: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Return (grid_x, grid_y) meshgrids covering the drawn court area."""
    xs = _axis_points(-SIDELINE_X, SIDELINE_X, step_ft)
    ys = _axis_points(BASELINE_Y, DRAWN_TOP_Y, step_ft)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x, grid_y


def plot_zone_map(grid_x, grid_y, grid_labels, ax=None):
    """Color each grid point by its cell, draw the court on top, and return the Axes."""
    grid_labels = np.asarray(grid_labels)
    if grid_labels.shape != np.shape(grid_x):
        raise ValueError(f"grid_labels shape {grid_labels.shape} != grid shape {np.shape(grid_x)}")
    if ax is None:
        figure, ax = plt.subplots()
    ax.pcolormesh(
        grid_x, grid_y, grid_labels, cmap=CELL_COLORMAP, norm=CELL_NORM, shading="nearest"
    )
    draw_half_court(ax)
    return ax


def plot_shot_heatmap(x_ft, y_ft, ax=None, bins=60, max_count=None):
    """Draw a log-scaled 2D histogram of shots on the court and return the Axes.

    The bins always span the drawn court, so heatmaps from different seasons share bins.
    Shots outside the drawn court are left out. Pass the same `max_count` to several
    heatmaps to run their color scales from 1 to `max_count`. Without it, each heatmap
    scales to its own counts.
    """
    if max_count is None:
        norm = LogNorm()
    elif max_count < 1:
        raise ValueError(f"max_count must be at least 1, got {max_count}")
    else:
        norm = LogNorm(vmin=1, vmax=max_count)
    if ax is None:
        figure, ax = plt.subplots()
    court_range = [[-SIDELINE_X, SIDELINE_X], [BASELINE_Y, DRAWN_TOP_Y]]
    ax.hist2d(x_ft, y_ft, bins=bins, range=court_range, norm=norm, cmap="viridis")
    draw_half_court(ax)
    return ax


def _cell_colors(n_cells: int) -> np.ndarray:
    """Return the RGBA color of cells 0 .. n_cells - 1, shape (n_cells, 4)."""
    return CELL_COLORMAP(np.arange(n_cells))


def plot_prototypes(points_ft, ax=None, numbered=True):
    """Draw one marker per prototype in its cell color and return the Axes.

    Draws no court lines, so it can be layered on a court or on other seeds.
    """
    points_ft = np.asarray(points_ft)
    if ax is None:
        figure, ax = plt.subplots()
    colors = _cell_colors(len(points_ft))
    ax.scatter(points_ft[:, 0], points_ft[:, 1], c=colors, s=150, edgecolors=LINE_COLOR, zorder=3)
    if numbered:
        for cell in range(len(points_ft)):
            x, y = points_ft[cell]
            ax.text(x, y, str(cell), ha="center", va="center", fontsize=8, zorder=4)
    return ax


def plot_prototype_paths(path_points, season_names, ax=None):
    """Draw each cell's path through the seasons and return the Axes.

    Each cell gets one line in its cell color. Each season gets its own marker shape,
    named in the legend. Draws no court lines.
    """
    if len(path_points) != len(season_names):
        raise ValueError(f"{len(path_points)} seasons of points but {len(season_names)} names")
    if len(season_names) > len(SEASON_MARKERS):
        raise ValueError(f"at most {len(SEASON_MARKERS)} seasons, got {len(season_names)}")
    if ax is None:
        figure, ax = plt.subplots()
    points_by_season = np.stack(path_points)  # (n_seasons, n_cells, 2)
    n_cells = points_by_season.shape[1]
    colors = _cell_colors(n_cells)
    for cell in range(n_cells):
        cell_path = points_by_season[:, cell]
        ax.plot(cell_path[:, 0], cell_path[:, 1], color=colors[cell], zorder=2)
    for season in range(len(season_names)):
        marker = SEASON_MARKERS[season]
        xs = points_by_season[season, :, 0]
        ys = points_by_season[season, :, 1]
        label = season_names[season]
        ax.scatter(xs, ys, c=colors, marker=marker, edgecolors=LINE_COLOR, label=label, zorder=3)
    legend = ax.legend()
    # Season markers in the legend stay gray so they don't read as cell 0.
    for handle in legend.legend_handles:
        handle.set_facecolor(LEGEND_MARKER_COLOR)
    return ax
