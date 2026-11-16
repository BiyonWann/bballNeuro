"""Experiment runners that tie the data, network and metrics modules together."""

import numpy as np
import pandas as pd

from threept import data, metrics, network

CORNER_MIN_ANGLE = 1.2
TOP_MAX_ANGLE = 0.4


def _line_recovery(weights: np.ndarray, train: pd.DataFrame, test: pd.DataFrame, kind: str) -> dict:
    """Return accuracy and the majority baseline, with cell types learned from train shots."""
    train_labels = network.assign(data.get_inputs(train, kind), weights)
    test_labels = network.assign(data.get_inputs(test, kind), weights)
    cell_is_three = metrics.cell_shot_type(train_labels, train["is_three"], len(weights))
    return {
        "accuracy": metrics.line_recovery_accuracy(cell_is_three, test_labels, test["is_three"]),
        "baseline": metrics.majority_baseline_accuracy(train["is_three"], test["is_three"]),
    }


def _test_fit(weights: np.ndarray, test: pd.DataFrame, kind: str) -> dict:
    """Return quantization error, ARI and group shares measured on test shots."""
    test_inputs = data.get_inputs(test, kind)
    test_labels = network.assign(test_inputs, weights)
    distances = metrics.prototype_distances(weights, kind)
    fit = {
        "qe": metrics.quantization_error(test_inputs, weights),
        "ari": metrics.adjusted_rand(test_labels, test["SHOT_ZONE_BASIC"]),
    }
    fit.update(metrics.shot_share_by_group(test_labels, distances))
    return fit


def _mean_arc_purity(weights: np.ndarray, test: pd.DataFrame, kind: str) -> float:
    """Return the mean test purity of beyond-20 cells that win shots, NaN if there are none."""
    test_labels = network.assign(data.get_inputs(test, kind), weights)
    purity = metrics.three_point_purity(test_labels, test["is_three"], len(weights))
    is_arc = metrics.prototype_distances(weights, kind) > metrics.MID_MAX_FT
    arc_purity = purity[is_arc & ~np.isnan(purity)]
    if len(arc_purity) == 0:
        return float("nan")
    return float(arc_purity.mean())


def evaluate(weights: np.ndarray, train: pd.DataFrame, test: pd.DataFrame, kind: str) -> dict:
    """Return line-recovery, fit and allocation results for a set of prototypes."""
    result = _line_recovery(weights, train, test, kind)
    result.update(_test_fit(weights, test, kind))
    result["mean_arc_purity"] = _mean_arc_purity(weights, test, kind)
    dead_cells = network.find_dead_cells(data.get_inputs(train, kind), weights)
    result["dead_cells"] = len(dead_cells)
    return result


def run_season(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kind: str = "cartesian",
    n_cells: int = 6,
    seeds: tuple = (0, 1, 2, 3, 4),
) -> tuple:
    """Return (results with one row per seed, the trained weights for each seed)."""
    train_inputs = data.get_inputs(train, kind)
    rows = []
    all_weights = []
    for seed in seeds:
        weights = network.train(train_inputs, n_cells=n_cells, seed=seed)
        row = {"seed": seed}
        row.update(evaluate(weights, train, test, kind))
        centers = metrics.kmeans_weights(train_inputs, n_cells, seed)
        row["kmeans_gap"] = metrics.prototype_gap(weights, centers)
        rows.append(row)
        all_weights.append(weights)
    return pd.DataFrame(rows), all_weights


def run_baselines(
    train: pd.DataFrame,
    test: pd.DataFrame,
    kind: str = "cartesian",
    n_cells: int = 6,
    seeds: tuple = (0, 1, 2, 3, 4),
) -> pd.DataFrame:
    """Return evaluate rows for k-means centers and untrained random prototypes, per seed."""
    train_inputs = data.get_inputs(train, kind)
    rows = []
    for seed in seeds:
        kmeans = metrics.kmeans_weights(train_inputs, n_cells, seed)
        random_start = network.init_weights(train_inputs, n_cells, seed)
        for method, weights in [("kmeans", kmeans), ("random", random_start)]:
            row = {"method": method, "seed": seed}
            row.update(evaluate(weights, train, test, kind))
            rows.append(row)
    return pd.DataFrame(rows)


def run_sequential(
    seasons: list, kind: str = "cartesian", n_cells: int = 6, seed: int = 0
) -> tuple:
    """Return (weights after each season, one summary row per season) from one network.

    Each season starts from the previous season's weights, so the first season has no movement.
    """
    all_weights = []
    rows = []
    previous = None
    for name, train, test in seasons:
        train_inputs = data.get_inputs(train, kind)
        weights = network.train(
            train_inputs, n_cells=n_cells, seed=seed, initial_weights=previous
        )
        result = evaluate(weights, train, test, kind)
        movement = float("nan")
        if previous is not None:
            movement = float(metrics.prototype_movement(previous, weights).mean())
        row = {"season": name, "mean_movement": movement}
        for key in ["rim", "mid_range", "beyond_20", "accuracy"]:
            row[key] = result[key]
        rows.append(row)
        all_weights.append(weights)
        previous = weights
    return all_weights, pd.DataFrame(rows)


def _mean_or_nan(values: np.ndarray) -> float:
    """Return the mean of values, or NaN when there are none."""
    if len(values) == 0:
        return float("nan")
    return float(values.mean())


def shape_test(weights: np.ndarray, kind: str) -> dict:
    """Return the mean distance and count of corner cells and top-of-the-arc cells.

    Only cells beyond 20 ft count. A mean is NaN when its group is empty.
    """
    distances = metrics.prototype_distances(weights, kind)
    angles = np.abs(metrics.prototype_angles(weights, kind))
    is_arc = distances > metrics.MID_MAX_FT
    is_corner = is_arc & (angles >= CORNER_MIN_ANGLE)
    is_top = is_arc & (angles <= TOP_MAX_ANGLE)
    return {
        "corner_mean_ft": _mean_or_nan(distances[is_corner]),
        "top_mean_ft": _mean_or_nan(distances[is_top]),
        "n_corner": int(is_corner.sum()),
        "n_top": int(is_top.sum()),
    }
