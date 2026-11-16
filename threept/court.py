"""Half-court drawing and court plots. All lengths are in feet with the basket at (0, 0)."""

import numpy as np

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
