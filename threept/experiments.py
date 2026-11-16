"""Experiment runners that tie the data, network and metrics modules together."""

import numpy as np

from threept import data, metrics, network


def evaluate(weights: np.ndarray, train, test, kind: str) -> dict:
    """Return line-recovery results for trained or baseline weights."""
    n_cells = len(weights)
    train_labels = network.assign(data.get_inputs(train, kind), weights)
    test_labels = network.assign(data.get_inputs(test, kind), weights)
    cell_is_three = metrics.cell_shot_type(train_labels, train["is_three"], n_cells)
    accuracy = metrics.line_recovery_accuracy(cell_is_three, test_labels, test["is_three"])
    baseline = metrics.majority_baseline_accuracy(train["is_three"], test["is_three"])
    return {"accuracy": accuracy, "baseline": baseline}
