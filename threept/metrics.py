"""Evaluation metrics and baselines for the competitive learning network."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

RIM_MAX_FT = 8
MID_MAX_FT = 20
KINDS = ["cartesian", "polar"]
# Same value as data.ARC_RADIUS_FT, repeated so metrics never imports data.
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


def _group_masks(distances: np.ndarray) -> dict:
    """Return a boolean mask for each distance group, keyed like cell_allocation."""
    is_rim = distances < RIM_MAX_FT
    is_beyond = distances > MID_MAX_FT
    is_mid = ~is_rim & ~is_beyond
    return {"rim": is_rim, "mid_range": is_mid, "beyond_20": is_beyond}


def cell_allocation(distances: np.ndarray) -> dict:
    """Return how many cells sit at the rim, in the mid-range and beyond 20 ft."""
    allocation = {}
    for group, mask in _group_masks(distances).items():
        allocation[group] = int(mask.sum())
    return allocation


def shot_share_by_group(labels: np.ndarray, distances: np.ndarray) -> dict:
    """Return the share of shots won by rim, mid-range and beyond-20 cells."""
    winner_distances = distances[np.asarray(labels)]
    shares = {}
    for group, mask in _group_masks(winner_distances).items():
        shares[group] = float(mask.mean())
    return shares


def _shot_counts(labels: np.ndarray, is_three: np.ndarray, n_cells: int) -> tuple:
    """Return (shots, threes) won by each cell, checking labels are valid cells."""
    labels = np.asarray(labels)
    is_three = np.asarray(is_three)
    if labels.min() < 0 or labels.max() >= n_cells:
        raise ValueError(f"labels must be between 0 and {n_cells - 1}")
    shots = np.zeros(n_cells)
    threes = np.zeros(n_cells)
    for cell in range(n_cells):
        in_cell = labels == cell
        shots[cell] = in_cell.sum()
        threes[cell] = is_three[in_cell].sum()
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


def adjusted_rand(labels: np.ndarray, zones: np.ndarray) -> float:
    """Return the adjusted Rand index between learned cells and official zones."""
    return float(adjusted_rand_score(zones, labels))


def zone_confusion(
    labels: np.ndarray, zones: np.ndarray, zone_names: list, n_cells: int
) -> pd.DataFrame:
    """Return shot counts with one row per cell and one column per zone name."""
    # Plain arrays, so a Series index from the test split cannot misalign the rows.
    labels = np.asarray(labels)
    zones = np.asarray(zones)
    if not np.isin(zones, zone_names).all():
        raise ValueError("zones contains a value that is not in zone_names")
    counts = pd.crosstab(labels, zones, rownames=["cell"], colnames=["zone"])
    return counts.reindex(index=range(n_cells), columns=zone_names, fill_value=0)


def kmeans_weights(inputs: np.ndarray, n_cells: int, seed: int = 0) -> np.ndarray:
    """Return k-means cluster centers, shape (n_cells, 2), as a baseline map."""
    kmeans = KMeans(n_clusters=n_cells, n_init=10, random_state=seed)
    kmeans.fit(inputs)
    return kmeans.cluster_centers_


def prototype_gap(weights: np.ndarray, reference: np.ndarray) -> float:
    """Return the mean distance between matched prototypes, closest pair first."""
    if len(weights) != len(reference):
        raise ValueError("weights and reference must have the same number of rows")
    distances = _pairwise_distances(reference, weights)
    matched_distances = []
    for _ in range(len(reference)):
        # argmin searches the flattened array, so turn its index back into (row, column).
        row, column = np.unravel_index(np.argmin(distances), distances.shape)
        matched_distances.append(distances[row, column])
        distances[row, :] = np.inf
        distances[:, column] = np.inf
    return float(np.mean(matched_distances))


def prototype_movement(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Return how far each cell's prototype moved, matching cells by index."""
    if before.shape != after.shape:
        raise ValueError("before and after must have the same shape")
    return np.linalg.norm(after - before, axis=1)
