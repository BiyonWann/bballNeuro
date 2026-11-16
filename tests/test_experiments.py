import numpy as np
import pandas as pd
import pytest

from threept import experiments

RIM_ZONE = "Restricted Area"
ARC_ZONE = "Above the Break 3"


def make_frame(y_ft, is_three, zones, games):
    """Return shots on the x = 0 line, with one row per entry."""
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
    """Return every shot repeated `times` times, keeping each game's shots together."""
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
    assert all_weights[0].shape == (2, 2)


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
    scrambled["SHOT_ZONE_BASIC"] = scrambled["SHOT_ZONE_BASIC"].to_numpy()[::-1]

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
