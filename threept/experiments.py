"""Experiment runners that tie the data, network and metrics modules together."""

import numpy as np
import pandas as pd

from threept import data, metrics, network


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
