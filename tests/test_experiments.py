import numpy as np
import pandas as pd
import pytest

from threept import experiments

RIM_ZONE = "Restricted Area"
ARC_ZONE = "Above the Break 3"


def make_frame(y_ft, is_three, zones, games):
    return pd.DataFrame(
        {
            "GAME_ID": games,
            "x_ft": [0.0] * len(y_ft),
            "y_ft": y_ft,
            "is_three": is_three,
            "SHOT_ZONE_BASIC": zones,
        }
    )


@pytest.fixture
def small_train():
    return make_frame(
        y_ft=[1, 1, 1, 25, 25, 25],
        is_three=[False, False, False, True, True, False],
        zones=[RIM_ZONE] * 3 + [ARC_ZONE] * 3,
        games=[1, 1, 2, 2, 3, 3],
    )


@pytest.fixture
def small_test():
    return make_frame(
        y_ft=[1, 1, 25, 25, 25],
        is_three=[False, False, True, True, False],
        zones=[RIM_ZONE] * 2 + [ARC_ZONE] * 3,
        games=[4, 4, 5, 5, 5],
    )


@pytest.fixture
def rim_arc_weights():
    return np.array([[0.0, 1.0], [0.0, 25.0]])


def test_evaluate_accuracy(small_train, small_test, rim_arc_weights):
    result = experiments.evaluate(rim_arc_weights, small_train, small_test, "cartesian")

    assert result["accuracy"] == pytest.approx(0.8)


def test_evaluate_cell_types_come_from_train(small_train, rim_arc_weights):
    test = make_frame(
        y_ft=[1, 25, 25],
        is_three=[False, False, False],
        zones=[RIM_ZONE, ARC_ZONE, ARC_ZONE],
        games=[4, 5, 5],
    )

    result = experiments.evaluate(rim_arc_weights, small_train, test, "cartesian")

    assert result["accuracy"] == pytest.approx(1 / 3)


def test_evaluate_baseline_twos_win(small_train, small_test, rim_arc_weights):
    result = experiments.evaluate(rim_arc_weights, small_train, small_test, "cartesian")

    assert result["baseline"] == pytest.approx(0.6)


def test_evaluate_baseline_threes_win(small_test, rim_arc_weights):
    train = make_frame(
        y_ft=[25, 25, 25, 25, 1, 1],
        is_three=[True, True, True, True, False, False],
        zones=[ARC_ZONE] * 4 + [RIM_ZONE] * 2,
        games=[1, 1, 2, 2, 3, 3],
    )

    result = experiments.evaluate(rim_arc_weights, train, small_test, "cartesian")

    assert result["baseline"] == pytest.approx(0.4)


def test_evaluate_qe_uses_test(small_train, rim_arc_weights):
    test = make_frame(
        y_ft=[2, 27], is_three=[False, True], zones=[RIM_ZONE, ARC_ZONE], games=[4, 5]
    )

    result = experiments.evaluate(rim_arc_weights, small_train, test, "cartesian")

    assert result["qe"] == pytest.approx(1.5)


def test_evaluate_ari_perfect(small_train, small_test, rim_arc_weights):
    result = experiments.evaluate(rim_arc_weights, small_train, small_test, "cartesian")

    assert result["ari"] == pytest.approx(1.0)


def test_evaluate_group_shares(small_train, small_test, rim_arc_weights):
    result = experiments.evaluate(rim_arc_weights, small_train, small_test, "cartesian")

    assert result["rim"] == pytest.approx(0.4)
    assert result["mid_range"] == pytest.approx(0.0)
    assert result["beyond_20"] == pytest.approx(0.6)


def test_evaluate_dead_cell(small_train, small_test):
    weights = np.array([[0.0, 1.0], [0.0, 25.0], [0.0, 100.0]])

    result = experiments.evaluate(weights, small_train, small_test, "cartesian")

    assert result["dead_cells"] == 1


def test_evaluate_dead_cells_use_train(small_train):
    weights = np.array([[0.0, 1.0], [0.0, 25.0], [0.0, 10.0]])
    test = make_frame(
        y_ft=[10, 10], is_three=[False, False], zones=[RIM_ZONE, RIM_ZONE], games=[4, 4]
    )

    result = experiments.evaluate(weights, small_train, test, "cartesian")

    assert result["dead_cells"] == 1


def test_evaluate_keys(small_train, small_test, rim_arc_weights):
    result = experiments.evaluate(rim_arc_weights, small_train, small_test, "cartesian")

    assert set(result) == {
        "accuracy",
        "baseline",
        "qe",
        "ari",
        "rim",
        "mid_range",
        "beyond_20",
        "dead_cells",
        "mean_arc_purity",
    }


def test_evaluate_mean_arc_purity(small_train, small_test, rim_arc_weights):
    result = experiments.evaluate(rim_arc_weights, small_train, small_test, "cartesian")

    assert result["mean_arc_purity"] == pytest.approx(2 / 3)


def test_evaluate_arc_purity_uses_test(small_train, rim_arc_weights):
    test = make_frame(
        y_ft=[1, 25, 25],
        is_three=[False, True, True],
        zones=[RIM_ZONE, ARC_ZONE, ARC_ZONE],
        games=[4, 5, 5],
    )

    result = experiments.evaluate(rim_arc_weights, small_train, test, "cartesian")

    assert result["mean_arc_purity"] == pytest.approx(1.0)


def test_evaluate_arc_purity_skips_empty_cell(small_train, small_test):
    weights = np.array([[0.0, 1.0], [0.0, 25.0], [0.0, 26.0]])

    result = experiments.evaluate(weights, small_train, small_test, "cartesian")

    assert result["mean_arc_purity"] == pytest.approx(2 / 3)


def test_evaluate_arc_purity_no_arc_cells(small_train, small_test):
    weights = np.array([[0.0, 1.0], [0.0, 5.0]])

    result = experiments.evaluate(weights, small_train, small_test, "cartesian")

    assert np.isnan(result["mean_arc_purity"])


def test_evaluate_polar(small_train, small_test):
    weights = np.array([[1.0, 0.0], [25.0, 0.0]])

    result = experiments.evaluate(weights, small_train, small_test, "polar")

    assert result["accuracy"] == pytest.approx(0.8)
    assert result["qe"] == pytest.approx(0.0)
    assert result["beyond_20"] == pytest.approx(0.6)


def test_evaluate_bad_kind(small_train, small_test, rim_arc_weights):
    with pytest.raises(ValueError):
        experiments.evaluate(rim_arc_weights, small_train, small_test, "radial")


def repeat_shots(shots, times):
    repeated = shots.loc[shots.index.repeat(times)]
    return repeated.reset_index(drop=True)


@pytest.fixture
def big_train(small_train):
    return repeat_shots(small_train, 30)


@pytest.fixture
def big_test(small_test):
    return repeat_shots(small_test, 30)


def test_run_season_columns_and_rows(big_train, big_test):
    frame, _ = experiments.run_season(big_train, big_test, n_cells=2, seeds=(0, 1, 2))

    assert list(frame["seed"]) == [0, 1, 2]
    assert set(frame.columns) == {
        "seed",
        "accuracy",
        "baseline",
        "qe",
        "ari",
        "rim",
        "mid_range",
        "beyond_20",
        "dead_cells",
        "mean_arc_purity",
        "kmeans_gap",
    }


def test_run_season_weights(big_train, big_test):
    _, all_weights = experiments.run_season(big_train, big_test, n_cells=2, seeds=(0, 1, 2))

    assert len(all_weights) == 3
    for weights in all_weights:
        assert weights.shape == (2, 2)


def test_run_season_rows_match_evaluate(big_train, big_test):
    frame, all_weights = experiments.run_season(big_train, big_test, n_cells=2, seeds=(0, 1, 2))

    for row, weights in enumerate(all_weights):
        expected = experiments.evaluate(weights, big_train, big_test, "cartesian")
        for key, value in expected.items():
            assert frame.loc[row, key] == pytest.approx(value, nan_ok=True)


def test_run_season_separable_values(big_train, big_test):
    frame, _ = experiments.run_season(big_train, big_test, n_cells=2, seeds=(0, 1, 2))

    np.testing.assert_allclose(frame["accuracy"], 0.8)
    np.testing.assert_allclose(frame["ari"], 1.0)
    np.testing.assert_allclose(frame["rim"], 0.4)
    np.testing.assert_allclose(frame["beyond_20"], 0.6)
    np.testing.assert_array_equal(frame["dead_cells"], 0)


def test_run_season_kmeans_gap_small(big_train, big_test):
    frame, _ = experiments.run_season(big_train, big_test, n_cells=2, seeds=(0, 1, 2))

    assert (frame["kmeans_gap"] < 0.5).all()


def test_run_season_deterministic(big_train, big_test):
    first, first_weights = experiments.run_season(big_train, big_test, n_cells=2, seeds=(0, 1))
    second, second_weights = experiments.run_season(big_train, big_test, n_cells=2, seeds=(0, 1))

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(first_weights, second_weights)


def test_run_season_ignores_labels(big_train, big_test):
    scrambled = big_train.copy()
    scrambled["is_three"] = ~scrambled["is_three"]
    # reversed zones put arc labels on rim shots and the reverse
    scrambled["SHOT_ZONE_BASIC"] = np.flip(scrambled["SHOT_ZONE_BASIC"].to_numpy())

    _, original_weights = experiments.run_season(big_train, big_test, n_cells=2, seeds=(0, 1))
    _, scrambled_weights = experiments.run_season(scrambled, big_test, n_cells=2, seeds=(0, 1))

    np.testing.assert_array_equal(original_weights, scrambled_weights)


def test_run_season_polar(big_train, big_test):
    frame, all_weights = experiments.run_season(
        big_train, big_test, kind="polar", n_cells=2, seeds=(0, 1, 2)
    )

    assert len(frame) == 3
    assert all_weights[0].shape == (2, 2)
    np.testing.assert_allclose(frame["accuracy"], 0.8)


def test_run_season_bad_kind(big_train, big_test):
    with pytest.raises(ValueError):
        experiments.run_season(big_train, big_test, kind="radial", n_cells=2)


def test_run_baselines_rows(big_train, big_test):
    frame = experiments.run_baselines(big_train, big_test, n_cells=2, seeds=(0, 1))

    assert len(frame) == 4
    assert list(frame["method"].value_counts().sort_index()) == [2, 2]
    assert set(frame["method"]) == {"kmeans", "random"}
    assert set(frame.columns) == {
        "method",
        "seed",
        "accuracy",
        "baseline",
        "qe",
        "ari",
        "rim",
        "mid_range",
        "beyond_20",
        "dead_cells",
        "mean_arc_purity",
    }


def test_run_baselines_kmeans_values(big_train, big_test):
    frame = experiments.run_baselines(big_train, big_test, n_cells=2, seeds=(0, 1))
    kmeans_rows = frame[frame["method"] == "kmeans"]

    np.testing.assert_allclose(kmeans_rows["accuracy"], 0.8)
    np.testing.assert_allclose(kmeans_rows["ari"], 1.0)
    np.testing.assert_allclose(kmeans_rows["qe"], 0.0, atol=1e-9)
    np.testing.assert_array_equal(kmeans_rows["dead_cells"], 0)


def test_run_baselines_random_is_untrained():
    train = make_frame(
        y_ft=[1, 5, 10, 25],
        is_three=[False, False, False, True],
        zones=[RIM_ZONE, RIM_ZONE, RIM_ZONE, ARC_ZONE],
        games=[1, 2, 3, 4],
    )
    test = train.copy()
    test["GAME_ID"] = [5, 6, 7, 8]

    frame = experiments.run_baselines(train, test, n_cells=4, seeds=(0,))
    random_row = frame[frame["method"] == "random"]

    np.testing.assert_allclose(random_row["qe"], 0.0)


def test_run_baselines_deterministic(big_train, big_test):
    first = experiments.run_baselines(big_train, big_test, n_cells=2, seeds=(0, 1))
    second = experiments.run_baselines(big_train, big_test, n_cells=2, seeds=(0, 1))

    pd.testing.assert_frame_equal(first, second)


def test_run_baselines_bad_kind(big_train, big_test):
    with pytest.raises(ValueError):
        experiments.run_baselines(big_train, big_test, kind="radial", n_cells=2)


@pytest.fixture
def rim_only_test():
    return make_frame(
        y_ft=[1] * 20,
        is_three=[False] * 20,
        zones=[RIM_ZONE] * 20,
        games=[9] * 20,
    )


@pytest.fixture
def shifted_train():
    return make_frame(
        y_ft=[10] * 60,
        is_three=[False] * 60,
        zones=[RIM_ZONE] * 60,
        games=[10] * 60,
    )


def test_run_sequential_shapes(big_train, big_test):
    seasons = [("a", big_train, big_test), ("b", big_train, big_test)]

    all_weights, frame = experiments.run_sequential(seasons, n_cells=2)

    assert len(all_weights) == 2
    for weights in all_weights:
        assert weights.shape == (2, 2)
    assert list(frame["season"]) == ["a", "b"]
    assert list(frame.columns) == [
        "season",
        "mean_movement",
        "rim",
        "mid_range",
        "beyond_20",
        "accuracy",
    ]


def test_run_sequential_scores_each_season(big_train, big_test, rim_only_test):
    seasons = [("a", big_train, big_test), ("b", big_train, rim_only_test)]

    _, frame = experiments.run_sequential(seasons, n_cells=2)

    assert frame.loc[0, "rim"] == pytest.approx(0.4)
    assert frame.loc[0, "beyond_20"] == pytest.approx(0.6)
    assert frame.loc[0, "accuracy"] == pytest.approx(0.8)
    assert frame.loc[1, "rim"] == pytest.approx(1.0)
    assert frame.loc[1, "beyond_20"] == pytest.approx(0.0)
    assert frame.loc[1, "accuracy"] == pytest.approx(1.0)


def test_run_sequential_starts_from_previous(big_train, big_test, shifted_train, rim_only_test):
    seasons = [("a", big_train, big_test), ("b", shifted_train, rim_only_test)]

    all_weights, _ = experiments.run_sequential(seasons, n_cells=2)

    unchanged_rows = (all_weights[0] == all_weights[1]).all(axis=1)
    assert unchanged_rows.sum() == 1


def test_run_sequential_winner_moves_to_new_data(big_train, big_test, shifted_train, rim_only_test):
    seasons = [("a", big_train, big_test), ("b", shifted_train, rim_only_test)]

    all_weights, frame = experiments.run_sequential(seasons, n_cells=2)

    changed_rows = (all_weights[0] != all_weights[1]).any(axis=1)
    moved_weights = all_weights[1][changed_rows]
    np.testing.assert_allclose(moved_weights[0], [0.0, 10.0], atol=0.5)
    distances = np.linalg.norm(all_weights[1] - all_weights[0], axis=1)
    assert frame.loc[1, "mean_movement"] == pytest.approx(distances.mean())


def test_run_sequential_first_movement_nan(big_train, big_test):
    seasons = [("a", big_train, big_test), ("b", big_train, big_test)]

    _, frame = experiments.run_sequential(seasons, n_cells=2)

    assert np.isnan(frame.loc[0, "mean_movement"])
    assert not np.isnan(frame.loc[1, "mean_movement"])


def test_run_sequential_deterministic(big_train, big_test):
    seasons = [("a", big_train, big_test), ("b", big_train, big_test)]

    first_weights, first = experiments.run_sequential(seasons, n_cells=2)
    second_weights, second = experiments.run_sequential(seasons, n_cells=2)

    pd.testing.assert_frame_equal(first, second)
    np.testing.assert_array_equal(first_weights, second_weights)


def test_run_sequential_polar(big_train, big_test):
    seasons = [("a", big_train, big_test), ("b", big_train, big_test)]

    all_weights, frame = experiments.run_sequential(seasons, kind="polar", n_cells=2)

    assert all_weights[1].shape == (2, 2)
    np.testing.assert_allclose(frame["accuracy"], 0.8)


def test_run_sequential_bad_kind(big_train, big_test):
    seasons = [("a", big_train, big_test)]

    with pytest.raises(ValueError):
        experiments.run_sequential(seasons, kind="radial", n_cells=2)


CORNER_AND_TOP = [[22.0, 0.0], [-22.0, 0.0], [0.0, 25.0], [0.0, 24.0]]


def test_shape_test_keys():
    result = experiments.shape_test(np.array(CORNER_AND_TOP), "cartesian")

    assert set(result) == {"corner_mean_ft", "top_mean_ft", "n_corner", "n_top"}


def test_shape_test_classifies_cells():
    result = experiments.shape_test(np.array(CORNER_AND_TOP), "cartesian")

    assert result["corner_mean_ft"] == pytest.approx(22.0)
    assert result["top_mean_ft"] == pytest.approx(24.5)
    assert result["n_corner"] == 2
    assert result["n_top"] == 2


def test_shape_test_ignores_inner_cells():
    weights = np.array(CORNER_AND_TOP + [[0.0, 1.0], [0.0, 19.0]])

    result = experiments.shape_test(weights, "cartesian")

    assert result["n_corner"] == 2
    assert result["n_top"] == 2
    assert result["top_mean_ft"] == pytest.approx(24.5)


def test_shape_test_ignores_wing_cells():
    weights = np.array(CORNER_AND_TOP + [[17.0, 17.0]])

    result = experiments.shape_test(weights, "cartesian")

    assert result["n_corner"] == 2
    assert result["n_top"] == 2
    assert result["corner_mean_ft"] == pytest.approx(22.0)


def test_shape_test_angle_boundaries():
    weights = np.array([[24.0, 23.75 * 1.2], [24.0, 23.75 * 0.4]])

    result = experiments.shape_test(weights, "polar")

    assert result["n_corner"] == 1
    assert result["n_top"] == 1


def test_shape_test_just_inside_boundaries():
    weights = np.array([[24.0, 23.75 * 1.19], [24.0, 23.75 * 0.41]])

    result = experiments.shape_test(weights, "polar")

    assert result["n_corner"] == 0
    assert result["n_top"] == 0


def test_shape_test_no_corner_cells():
    result = experiments.shape_test(np.array([[0.0, 25.0], [0.0, 24.0]]), "cartesian")

    assert np.isnan(result["corner_mean_ft"])
    assert result["n_corner"] == 0
    assert result["top_mean_ft"] == pytest.approx(24.5)


def test_shape_test_no_top_cells():
    result = experiments.shape_test(np.array([[22.0, 0.0], [-22.0, 0.0]]), "cartesian")

    assert np.isnan(result["top_mean_ft"])
    assert result["n_top"] == 0
    assert result["corner_mean_ft"] == pytest.approx(22.0)


def test_shape_test_no_arc_cells():
    result = experiments.shape_test(np.array([[0.0, 1.0], [0.0, 5.0]]), "cartesian")

    assert np.isnan(result["corner_mean_ft"])
    assert np.isnan(result["top_mean_ft"])
    assert result["n_corner"] == 0
    assert result["n_top"] == 0


def test_shape_test_polar():
    arc_ft = 23.75 * np.pi / 2
    weights = np.array([[22.0, arc_ft], [22.0, -arc_ft], [25.0, 0.0], [24.0, 0.0]])

    result = experiments.shape_test(weights, "polar")

    assert result["corner_mean_ft"] == pytest.approx(22.0)
    assert result["top_mean_ft"] == pytest.approx(24.5)
    assert result["n_corner"] == 2
    assert result["n_top"] == 2


def test_shape_test_bad_kind():
    with pytest.raises(ValueError):
        experiments.shape_test(np.array(CORNER_AND_TOP), "radial")


def test_summarize_mean_and_std():
    results = pd.DataFrame({"accuracy": [0.8, 0.9, 1.0]})

    summary = experiments.summarize(results)

    assert summary.loc["accuracy", "mean"] == pytest.approx(0.9)
    assert summary.loc["accuracy", "std"] == pytest.approx(0.1)


def test_summarize_layout():
    results = pd.DataFrame({"accuracy": [0.8, 0.9], "qe": [1.0, 3.0]})

    summary = experiments.summarize(results)

    assert list(summary.index) == ["accuracy", "qe"]
    assert list(summary.columns) == ["mean", "std"]


def test_summarize_excludes_seed():
    results = pd.DataFrame({"seed": [0, 1, 2], "accuracy": [0.8, 0.9, 1.0]})

    summary = experiments.summarize(results)

    assert "seed" not in summary.index


def test_summarize_skips_text_columns():
    results = pd.DataFrame({"method": ["kmeans", "random"], "accuracy": [0.8, 0.9]})

    summary = experiments.summarize(results)

    assert list(summary.index) == ["accuracy"]


def test_summarize_skips_nan():
    results = pd.DataFrame({"mean_arc_purity": [np.nan, 0.5, 0.7]})

    summary = experiments.summarize(results)

    assert summary.loc["mean_arc_purity", "mean"] == pytest.approx(0.6)


def test_summarize_does_not_mutate():
    results = pd.DataFrame({"seed": [0, 1], "accuracy": [0.8, 0.9]})
    before = results.copy()

    experiments.summarize(results)

    pd.testing.assert_frame_equal(results, before)


def test_summarize_run_season_output(big_train, big_test):
    frame, _ = experiments.run_season(big_train, big_test, n_cells=2, seeds=(0, 1, 2))

    summary = experiments.summarize(frame)

    assert summary.loc["accuracy", "mean"] == pytest.approx(0.8)
    assert summary.loc["accuracy", "std"] == pytest.approx(0.0)
    assert "seed" not in summary.index
