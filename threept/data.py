"""Load, clean and reshape NBA shot data into model inputs."""

import numpy as np

SEASONS = {"2000-01": 2000, "2008-09": 2008, "2015-16": 2015, "2024-25": 2024}
DATA_URL = "https://github.com/shufinskiy/nba_data/raw/main/datasets/shotdetail_{year}.tar.xz"
ZONES = [
    "Restricted Area",
    "In The Paint (Non-RA)",
    "Mid-Range",
    "Left Corner 3",
    "Right Corner 3",
    "Above the Break 3",
]
ARC_RADIUS_FT = 23.75
# A little past the baseline, so shots right behind the rim do not wrap to +-pi.
MAX_ANGLE = np.pi / 2 + 0.25
