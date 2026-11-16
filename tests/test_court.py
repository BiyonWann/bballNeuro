import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from threept import court  # noqa: E402


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


def count_three_point_lines(ax):
    """Return how many lines on `ax` trace exactly the 3-point line."""
    line_xs, line_ys = court.three_point_line()
    count = 0
    for line in ax.lines:
        xs = np.asarray(line.get_xdata(), dtype=float)
        ys = np.asarray(line.get_ydata(), dtype=float)
        if xs.shape == line_xs.shape and np.allclose(xs, line_xs) and np.allclose(ys, line_ys):
            count = count + 1
    return count


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


def test_three_point_line_returns_matching_arrays():
    xs, ys = court.three_point_line()

    assert xs.ndim == 1
    assert xs.shape == ys.shape


def test_arc_points_are_23_75_ft_from_basket():
    xs, ys = court.three_point_line()

    arc_distances = np.hypot(xs[1:-1], ys[1:-1])

    assert arc_distances == pytest.approx(np.full(len(arc_distances), 23.75))


def test_corner_segments_are_vertical_at_22_ft():
    xs, ys = court.three_point_line()

    assert (xs[0], ys[0]) == pytest.approx((-22, -5.25))
    assert (xs[1], ys[1]) == pytest.approx((-22, court.CORNER_BREAK_Y))
    assert (xs[-2], ys[-2]) == pytest.approx((22, court.CORNER_BREAK_Y))
    assert (xs[-1], ys[-1]) == pytest.approx((22, -5.25))


def test_line_starts_and_ends_on_baseline():
    xs, ys = court.three_point_line()

    assert (xs[0], ys[0]) == pytest.approx((-22, court.BASELINE_Y))
    assert (xs[-1], ys[-1]) == pytest.approx((22, court.BASELINE_Y))


def test_corners_meet_arc_at_corner_break():
    xs, ys = court.three_point_line()

    arc_xs = xs[1:-1]
    arc_ys = ys[1:-1]

    assert (arc_xs[0], arc_ys[0]) == pytest.approx((-22, court.CORNER_BREAK_Y))
    assert (arc_xs[-1], arc_ys[-1]) == pytest.approx((22, court.CORNER_BREAK_Y))


def test_arc_has_no_jumps():
    xs, ys = court.three_point_line(n_points=200)

    steps = np.hypot(np.diff(xs[1:-1]), np.diff(ys[1:-1]))

    assert steps.max() < 1


def test_arc_peak_is_at_top_of_key():
    xs, ys = court.three_point_line(n_points=201)

    peak = np.argmax(ys)

    assert ys[peak] == pytest.approx(23.75)
    assert xs[peak] == pytest.approx(0, abs=1e-9)


@pytest.mark.parametrize(("n_points", "expected_length"), [(50, 52), (200, 202)])
def test_n_points_sets_arc_resolution(n_points, expected_length):
    xs, ys = court.three_point_line(n_points=n_points)

    assert len(xs) == expected_length
    assert len(ys) == expected_length


def test_draw_half_court_creates_axes():
    ax = court.draw_half_court()

    assert isinstance(ax, plt.Axes)


def test_draw_half_court_uses_given_axes():
    figure, ax = plt.subplots()

    result = court.draw_half_court(ax)

    assert result is ax
    assert count_three_point_lines(ax) == 1


def test_draw_half_court_limits_and_aspect():
    ax = court.draw_half_court()

    assert ax.get_xlim() == pytest.approx((-25, 25))
    assert ax.get_ylim() == pytest.approx((-5.25, 35))
    assert ax.get_aspect() == 1.0


def test_court_grid_covers_court():
    grid_x, grid_y = court.court_grid(step_ft=0.5)

    assert grid_x.min() == pytest.approx(-25)
    assert grid_x.max() == pytest.approx(25)
    assert grid_y.min() == pytest.approx(-5.25)
    assert grid_y.max() == pytest.approx(35)


def test_court_grid_shape_and_step():
    grid_x, grid_y = court.court_grid(step_ft=0.5)

    assert grid_x.shape == grid_y.shape
    assert np.diff(grid_x[0]) == pytest.approx(np.full(grid_x.shape[1] - 1, 0.5))
    assert np.all(grid_x[:, 0] == grid_x[0, 0])


def test_court_grid_coarse_step_reaches_edges():
    grid_x, grid_y = court.court_grid(step_ft=7)

    assert grid_x.max() == pytest.approx(25)
    assert grid_y.max() == pytest.approx(35)


def small_grid():
    """Return a 2 x 2 grid of court points."""
    return np.meshgrid([-10.0, 10.0], [0.0, 20.0])


def test_zone_map_accepts_grid_shaped_labels():
    grid_x, grid_y = court.court_grid(step_ft=5)
    grid_labels = (grid_x >= 0).astype(int)

    ax = court.plot_zone_map(grid_x, grid_y, grid_labels)

    assert isinstance(ax, plt.Axes)


def test_zone_map_plots_label_array():
    grid_x, grid_y = small_grid()
    grid_labels = np.array([[0, 1], [2, 3]])

    ax = court.plot_zone_map(grid_x, grid_y, grid_labels)

    plotted = np.asarray(ax.collections[0].get_array()).ravel()
    assert plotted.tolist() == [0, 1, 2, 3]


def test_zone_map_shape_mismatch_raises():
    grid_x, grid_y = small_grid()

    with pytest.raises(ValueError):
        court.plot_zone_map(grid_x, grid_y, np.array([0, 1, 2]))


def test_zone_map_draws_court_lines():
    grid_x, grid_y = small_grid()
    figure, ax = plt.subplots()

    result = court.plot_zone_map(grid_x, grid_y, np.array([[0, 1], [2, 3]]), ax=ax)

    assert result is ax
    assert count_three_point_lines(ax) == 1
