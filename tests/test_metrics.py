import numpy as np
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
