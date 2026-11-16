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


def drawn_zone_colors(ax):
    """Render `ax` and return the RGBA face color of each zone-map grid point, row by row."""
    ax.figure.canvas.draw()
    return np.asarray(ax.collections[0].get_facecolor())


def test_cell_color_is_tab20_index():
    grid_x, grid_y = small_grid()
    ax = court.plot_zone_map(grid_x, grid_y, np.array([[0, 1], [2, 3]]))

    colors = drawn_zone_colors(ax)

    tab20 = matplotlib.colormaps["tab20"]
    for cell in [0, 1, 2, 3]:
        assert colors[cell] == pytest.approx(np.array(tab20(cell)))


def test_cell_color_same_across_label_sets():
    grid_x, grid_y = small_grid()
    ax_few = court.plot_zone_map(grid_x, grid_y, np.array([[3, 4], [4, 3]]))
    grid_x6, grid_y6 = np.meshgrid([-10.0, 0.0, 10.0], [0.0, 20.0])
    ax_all = court.plot_zone_map(grid_x6, grid_y6, np.array([[0, 1, 2], [3, 4, 5]]))

    few_colors = drawn_zone_colors(ax_few)
    all_colors = drawn_zone_colors(ax_all)

    assert few_colors[0] == pytest.approx(all_colors[3])


def test_twenty_labels_give_twenty_colors():
    grid_x, grid_y = np.meshgrid([0.0, 1.0, 2.0, 3.0, 4.0], [0.0, 1.0, 2.0, 3.0])
    grid_labels = np.arange(20).reshape(4, 5)
    ax = court.plot_zone_map(grid_x, grid_y, grid_labels)

    colors = drawn_zone_colors(ax)

    tab20 = matplotlib.colormaps["tab20"]
    assert len(np.unique(colors, axis=0)) == 20
    assert colors == pytest.approx(tab20(np.arange(20)))


def four_shots():
    """Return x and y in feet for three shots at (0, 1) and one at (10, 20)."""
    return np.array([0.0, 0.0, 0.0, 10.0]), np.array([1.0, 1.0, 1.0, 20.0])


def heatmap_counts(ax):
    """Return the per-bin shot counts drawn by the heatmap on `ax`."""
    return np.ma.asarray(ax.collections[0].get_array())


def test_heatmap_returns_axes():
    x_ft, y_ft = four_shots()

    ax = court.plot_shot_heatmap(x_ft, y_ft)

    assert isinstance(ax, plt.Axes)


def test_heatmap_uses_log_norm():
    x_ft, y_ft = four_shots()

    ax = court.plot_shot_heatmap(x_ft, y_ft)

    assert isinstance(ax.collections[0].norm, matplotlib.colors.LogNorm)


def test_heatmap_draws_court_lines():
    x_ft, y_ft = four_shots()
    figure, ax = plt.subplots()

    result = court.plot_shot_heatmap(x_ft, y_ft, ax=ax)

    assert result is ax
    assert count_three_point_lines(ax) == 1


def test_heatmap_counts_shots():
    x_ft, y_ft = four_shots()

    ax = court.plot_shot_heatmap(x_ft, y_ft, bins=60)

    assert heatmap_counts(ax).sum() == 4


@pytest.mark.parametrize(
    ("x_ft", "y_ft"),
    [
        (np.array([0.0, 0.0, 0.0]), np.array([1.0, 1.0, 1.0])),
        (np.array([-3.0, 0.0, 3.0]), np.array([2.0, 5.0, 9.0])),
    ],
)
def test_heatmap_bins_span_court(x_ft, y_ft):
    ax = court.plot_shot_heatmap(x_ft, y_ft)

    corners = ax.collections[0].get_coordinates()

    assert corners[..., 0].min() == pytest.approx(-25)
    assert corners[..., 0].max() == pytest.approx(25)
    assert corners[..., 1].min() == pytest.approx(-5.25)
    assert corners[..., 1].max() == pytest.approx(35)


def test_heatmap_drops_shot_past_court():
    x_ft = np.array([0.0, 0.0, 0.0])
    y_ft = np.array([1.0, 1.0, 40.0])

    ax = court.plot_shot_heatmap(x_ft, y_ft)

    assert heatmap_counts(ax).sum() == 2


def test_heatmap_single_bin():
    x_ft = np.zeros(5)
    y_ft = np.ones(5)

    ax = court.plot_shot_heatmap(x_ft, y_ft)
    ax.figure.canvas.draw()

    assert heatmap_counts(ax).max() == 5


def three_prototypes():
    """Return three prototypes in court feet: near the rim, left corner and top of the arc."""
    return np.array([[0.0, 3.0], [-22.0, 0.0], [0.0, 24.0]])


def marker_positions(ax):
    """Return every scatter marker position on `ax`, stacked into one (n, 2) array."""
    positions = [np.asarray(collection.get_offsets()) for collection in ax.collections]
    return np.concatenate(positions)


def test_prototypes_one_marker_each():
    ax = court.plot_prototypes(three_prototypes())

    assert marker_positions(ax) == pytest.approx(three_prototypes())


def test_prototypes_numbered_by_index():
    ax = court.plot_prototypes(three_prototypes())

    texts = [text.get_text() for text in ax.texts]
    positions = np.array([text.get_position() for text in ax.texts])
    assert texts == ["0", "1", "2"]
    assert positions == pytest.approx(three_prototypes())


def test_prototypes_unnumbered_has_no_text():
    ax = court.plot_prototypes(three_prototypes(), numbered=False)

    assert len(ax.texts) == 0


def test_prototypes_marker_colors_match_cells():
    ax = court.plot_prototypes(three_prototypes())

    colors = np.asarray(ax.collections[0].get_facecolor())

    tab20 = matplotlib.colormaps["tab20"]
    assert colors == pytest.approx(tab20(np.arange(3)))


def test_prototypes_draw_no_court_lines():
    ax = court.plot_prototypes(three_prototypes())

    assert count_three_point_lines(ax) == 0


def test_prototypes_two_calls_layer_on_same_axes():
    ax = court.draw_half_court()

    first = court.plot_prototypes(three_prototypes(), ax=ax, numbered=False)
    second = court.plot_prototypes(three_prototypes(), ax=ax, numbered=False)

    assert first is ax
    assert second is ax
    assert len(marker_positions(ax)) == 6
    assert count_three_point_lines(ax) == 1


SEASON_NAMES = ["2000-01", "2015-16", "2024-25"]


def two_cell_paths():
    """Return 3 seasons of 2 cells: cell 0 moves out from the rim, cell 1 into the corner."""
    return [
        np.array([[0.0, 5.0], [-20.0, 0.0]]),
        np.array([[0.0, 10.0], [-21.0, 0.0]]),
        np.array([[0.0, 20.0], [-22.0, 0.0]]),
    ]


def test_paths_one_line_per_cell():
    ax = court.plot_prototype_paths(two_cell_paths(), SEASON_NAMES)

    line_0 = np.column_stack([ax.lines[0].get_xdata(), ax.lines[0].get_ydata()])
    line_1 = np.column_stack([ax.lines[1].get_xdata(), ax.lines[1].get_ydata()])
    assert len(ax.lines) == 2
    assert line_0 == pytest.approx(np.array([[0.0, 5.0], [0.0, 10.0], [0.0, 20.0]]))
    assert line_1 == pytest.approx(np.array([[-20.0, 0.0], [-21.0, 0.0], [-22.0, 0.0]]))


def test_paths_label_seasons():
    ax = court.plot_prototype_paths(two_cell_paths(), SEASON_NAMES)

    legend_texts = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend_texts == SEASON_NAMES


def test_paths_line_colors_match_cells():
    ax = court.plot_prototype_paths(two_cell_paths(), SEASON_NAMES)

    tab20 = matplotlib.colormaps["tab20"]
    for cell in [0, 1]:
        line_color = matplotlib.colors.to_rgba(ax.lines[cell].get_color())
        assert line_color == pytest.approx(tab20(cell))


def test_paths_draw_no_court_lines():
    ax = court.plot_prototype_paths(two_cell_paths(), SEASON_NAMES)

    assert count_three_point_lines(ax) == 0


def test_paths_single_season():
    ax = court.plot_prototype_paths(two_cell_paths()[:1], SEASON_NAMES[:1])

    assert len(ax.lines) == 2
    assert len(ax.lines[0].get_xdata()) == 1
    assert len(ax.lines[1].get_xdata()) == 1


def test_paths_name_count_mismatch_raises():
    with pytest.raises(ValueError):
        court.plot_prototype_paths(two_cell_paths(), SEASON_NAMES[:2])


def test_plot_functions_do_not_mutate_inputs():
    x_ft, y_ft = four_shots()
    grid_x, grid_y = small_grid()
    grid_labels = np.array([[0, 1], [2, 3]])
    prototypes = three_prototypes()
    paths = two_cell_paths()
    originals = [x_ft.copy(), y_ft.copy(), grid_x.copy(), grid_labels.copy(), prototypes.copy()]
    original_paths = [season.copy() for season in paths]

    court.plot_shot_heatmap(x_ft, y_ft)
    court.plot_zone_map(grid_x, grid_y, grid_labels)
    court.plot_prototypes(prototypes)
    court.plot_prototype_paths(paths, SEASON_NAMES)

    after = [x_ft, y_ft, grid_x, grid_labels, prototypes] + paths
    before = originals + original_paths
    for index in range(len(before)):
        assert np.array_equal(before[index], after[index])
