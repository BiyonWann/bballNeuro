import numpy as np
import pandas as pd
import pytest

from threept import metrics


def test_group_thresholds():
    assert metrics.RIM_MAX_FT == 8
    assert metrics.MID_MAX_FT == 20


def test_prototype_distances_cartesian():
    weights = np.array([[3.0, 4.0], [0.0, -5.0], [0.0, 0.0]])

    distances = metrics.prototype_distances(weights, "cartesian")

    np.testing.assert_allclose(distances, [5.0, 5.0, 0.0])


def test_prototype_distances_polar():
    weights = np.array([[22.0, 30.0], [8.0, -5.0]])

    distances = metrics.prototype_distances(weights, "polar")

    np.testing.assert_allclose(distances, [22.0, 8.0])


def test_prototype_distances_bad_kind():
    weights = np.array([[3.0, 4.0]])

    with pytest.raises(ValueError):
        metrics.prototype_distances(weights, "radial")


@pytest.mark.parametrize(
    "point, expected",
    [
        ([0.0, 10.0], 0.0),
        ([10.0, 0.0], np.pi / 2),
        ([-10.0, 0.0], -np.pi / 2),
        ([10.0, 10.0], np.pi / 4),
    ],
)
def test_prototype_angles_cartesian(point, expected):
    weights = np.array([point])

    angles = metrics.prototype_angles(weights, "cartesian")

    np.testing.assert_allclose(angles, [expected])


def test_prototype_angles_polar():
    weights = np.array([[23.75, 23.75], [22.0, 0.0], [22.0, -23.75 * np.pi / 2]])

    angles = metrics.prototype_angles(weights, "polar")

    np.testing.assert_allclose(angles, [1.0, 0.0, -np.pi / 2])


def test_prototype_angles_bad_kind():
    weights = np.array([[3.0, 4.0]])

    with pytest.raises(ValueError):
        metrics.prototype_angles(weights, "xy")


def test_cell_allocation_edges():
    distances = np.array([0.0, 7.99, 8.0, 20.0, 20.01, 30.0])

    allocation = metrics.cell_allocation(distances)

    assert allocation == {"rim": 2, "mid_range": 2, "beyond_20": 2}


def test_cell_allocation_all_rim():
    distances = np.array([1.0, 2.0, 3.0])

    allocation = metrics.cell_allocation(distances)

    assert allocation == {"rim": 3, "mid_range": 0, "beyond_20": 0}


def test_purity_hand_case():
    labels = np.array([0, 0, 0, 1, 1, 2])
    is_three = np.array([True, True, False, False, False, True])

    purity = metrics.three_point_purity(labels, is_three, 4)

    np.testing.assert_allclose(purity, [2 / 3, 0.0, 1.0, np.nan])


def test_purity_empty_cell_is_nan():
    labels = np.array([0, 0, 0, 1, 1, 2])
    is_three = np.array([True, True, False, False, False, True])

    purity = metrics.three_point_purity(labels, is_three, 4)

    assert np.isnan(purity[3])


def test_purity_accepts_series():
    labels = np.array([0, 0, 0, 1, 1, 2])
    is_three = pd.Series([True, True, False, False, False, True], index=[10, 11, 12, 13, 14, 15])

    purity = metrics.three_point_purity(labels, is_three, 4)

    np.testing.assert_allclose(purity, [2 / 3, 0.0, 1.0, np.nan])


def test_purity_label_out_of_range():
    labels = np.array([0, 3])
    is_three = np.array([True, False])

    with pytest.raises(ValueError):
        metrics.three_point_purity(labels, is_three, 2)


def test_cell_shot_type_majority():
    labels = np.array([0, 0, 0, 1, 1, 2, 2])
    is_three = np.array([True, True, False, False, False, True, False])

    cell_is_three = metrics.cell_shot_type(labels, is_three, 4)

    np.testing.assert_array_equal(cell_is_three, [True, False, False, False])


def test_line_recovery_hand_case():
    cell_is_three = np.array([False, True])
    test_labels = np.array([0, 1, 1, 0, 1])
    test_is_three = np.array([False, True, False, False, True])

    accuracy = metrics.line_recovery_accuracy(cell_is_three, test_labels, test_is_three)

    assert isinstance(accuracy, float)
    assert accuracy == pytest.approx(0.8)


def test_line_recovery_perfect():
    cell_is_three = np.array([False, True])
    test_labels = np.array([0, 1, 1, 0, 1])
    test_is_three = np.array([False, True, True, False, True])

    accuracy = metrics.line_recovery_accuracy(cell_is_three, test_labels, test_is_three)

    assert accuracy == pytest.approx(1.0)


def test_majority_baseline_twos_win():
    train_is_three = np.array([False, False, False, True])
    test_is_three = np.array([False, False, True, False, True])

    accuracy = metrics.majority_baseline_accuracy(train_is_three, test_is_three)

    assert isinstance(accuracy, float)
    assert accuracy == pytest.approx(0.6)


def test_majority_baseline_threes_win():
    train_is_three = np.array([True, True, True, False])
    test_is_three = np.array([False, False, True, False, True])

    accuracy = metrics.majority_baseline_accuracy(train_is_three, test_is_three)

    assert accuracy == pytest.approx(0.4)


def test_majority_baseline_tie_predicts_two():
    train_is_three = np.array([True, False])
    test_is_three = np.array([True, False, False])

    accuracy = metrics.majority_baseline_accuracy(train_is_three, test_is_three)

    assert accuracy == pytest.approx(2 / 3)
