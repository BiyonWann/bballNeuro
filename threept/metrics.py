"""Evaluation metrics and baselines for the competitive learning network."""

import numpy as np

RIM_MAX_FT = 8
MID_MAX_FT = 20
KINDS = ["cartesian", "polar"]
ARC_RADIUS_FT = 23.75


def _check_kind(kind: str) -> None:
    """Raise ValueError unless kind is "cartesian" or "polar"."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")


def prototype_distances(weights: np.ndarray, kind: str) -> np.ndarray:
    """Return each prototype's distance from the basket in feet."""
    _check_kind(kind)
    if kind == "polar":
        return weights[:, 0].copy()
    return np.hypot(weights[:, 0], weights[:, 1])


def prototype_angles(weights: np.ndarray, kind: str) -> np.ndarray:
    """Return each prototype's angle in radians, 0 toward the top of the key."""
    _check_kind(kind)
    if kind == "polar":
        return weights[:, 1] / ARC_RADIUS_FT
    return np.arctan2(weights[:, 0], weights[:, 1])


def cell_allocation(distances: np.ndarray) -> dict:
    """Return how many cells sit at the rim, in the mid-range and beyond 20 ft."""
    is_rim = distances < RIM_MAX_FT
    is_beyond = distances > MID_MAX_FT
    is_mid = ~is_rim & ~is_beyond
    return {
        "rim": int(is_rim.sum()),
        "mid_range": int(is_mid.sum()),
        "beyond_20": int(is_beyond.sum()),
    }


def _shot_counts(labels: np.ndarray, is_three: np.ndarray, n_cells: int) -> tuple:
    """Return (shots, threes) won by each cell, checking labels are valid cells."""
    labels = np.asarray(labels)
    if labels.min() < 0 or labels.max() >= n_cells:
        raise ValueError(f"labels must be between 0 and {n_cells - 1}")
    three_weights = np.asarray(is_three, dtype=float)
    shots = np.bincount(labels, minlength=n_cells)
    threes = np.bincount(labels, weights=three_weights, minlength=n_cells)
    return shots, threes


def three_point_purity(labels: np.ndarray, is_three: np.ndarray, n_cells: int) -> np.ndarray:
    """Return the share of each cell's shots that are threes, NaN for an empty cell."""
    shots, threes = _shot_counts(labels, is_three, n_cells)
    purity = np.full(n_cells, np.nan)
    has_shots = shots > 0
    purity[has_shots] = threes[has_shots] / shots[has_shots]
    return purity


def cell_shot_type(labels: np.ndarray, is_three: np.ndarray, n_cells: int) -> np.ndarray:
    """Return True for each cell where more than half of its shots are threes."""
    shots, threes = _shot_counts(labels, is_three, n_cells)
    return threes > shots / 2


def line_recovery_accuracy(
    cell_is_three: np.ndarray, test_labels: np.ndarray, test_is_three: np.ndarray
) -> float:
    """Return the share of test shots whose cell's shot type matches the real one."""
    predicted = np.asarray(cell_is_three)[np.asarray(test_labels)]
    return float(np.mean(predicted == np.asarray(test_is_three)))


def majority_baseline_accuracy(train_is_three: np.ndarray, test_is_three: np.ndarray) -> float:
    """Return the accuracy of always guessing the training set's more common shot type."""
    guess_three = np.mean(train_is_three) > 0.5
    return float(np.mean(np.asarray(test_is_three) == guess_three))


def _pairwise_distances(points: np.ndarray, others: np.ndarray) -> np.ndarray:
    """Return an (n_points, n_others) array of Euclidean distances."""
    # (n_points, 1, 2) - (1, n_others, 2)
    differences = points[:, np.newaxis, :] - others[np.newaxis, :, :]
    return np.linalg.norm(differences, axis=2)


def quantization_error(inputs: np.ndarray, weights: np.ndarray) -> float:
    """Return the mean distance from each input to its nearest prototype."""
    distances = _pairwise_distances(inputs, weights)
    return float(np.mean(distances.min(axis=1)))
