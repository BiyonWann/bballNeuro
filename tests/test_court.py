import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from threept import court  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def test_constants_match_interface():
    assert court.BASELINE_Y == -5.25
    assert court.HALF_COURT_Y == 41.75
    assert court.SIDELINE_X == 25
    assert court.CORNER_THREE_X == 22
    assert court.ARC_RADIUS == 23.75
    assert court.PAINT_HALF_WIDTH == 8
    assert court.FREE_THROW_Y == 13.75
    assert court.RESTRICTED_RADIUS == 4
    assert court.RIM_RADIUS == 0.75


def test_corner_break_y_value():
    assert court.CORNER_BREAK_Y == pytest.approx(8.948, abs=1e-3)
    assert 22**2 + court.CORNER_BREAK_Y**2 == pytest.approx(23.75**2)
