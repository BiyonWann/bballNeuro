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
