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
