import numpy as np

from threept import data


def test_constants():
    assert data.SEASONS == {"2000-01": 2000, "2008-09": 2008, "2015-16": 2015, "2024-25": 2024}
    assert data.DATA_URL == (
        "https://github.com/shufinskiy/nba_data/raw/main/datasets/shotdetail_{year}.tar.xz"
    )
    assert data.ZONES == [
        "Restricted Area",
        "In The Paint (Non-RA)",
        "Mid-Range",
        "Left Corner 3",
        "Right Corner 3",
        "Above the Break 3",
    ]
    assert data.ARC_RADIUS_FT == 23.75
    assert data.MAX_ANGLE == np.pi / 2 + 0.25
