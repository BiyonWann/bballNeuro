"""Half-court drawing and court plots. All lengths are in feet with the basket at (0, 0)."""

import matplotlib.pyplot as plt
import numpy as np
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
    ax.pcolormesh(grid_x, grid_y, grid_labels, shading="nearest")
    draw_half_court(ax)
    return ax
