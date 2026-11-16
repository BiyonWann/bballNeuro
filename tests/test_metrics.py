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
