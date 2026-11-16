"""Competitive learning network: Euclidean winner, winner-only update."""

import numpy as np


def find_winner(shot: np.ndarray, weights: np.ndarray) -> int:
    """Return the index of the weight vector closest to the shot.

    Ties go to the lowest index.
    """
    # (n_cells, 2) - (2,)
    distances = np.linalg.norm(weights - shot, axis=1)
    return int(np.argmin(distances))


def init_weights(inputs: np.ndarray, n_cells: int, seed: int = 0) -> np.ndarray:
    """Return copies of n_cells distinct, randomly chosen rows of inputs."""
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(inputs), n_cells, replace=False)
    return inputs[rows].copy()


def train(
    inputs: np.ndarray,
    n_cells: int = 6,
    epochs: int = 8,
    learning_rate: float = 0.05,
    seed: int = 0,
    initial_weights: np.ndarray | None = None,
    leak_rate: float = 0.0,
) -> np.ndarray:
    """Return the trained weights, one prototype row per cell."""
    if initial_weights is None:
        weights = init_weights(inputs, n_cells, seed)
    else:
        weights = initial_weights
    for epoch in range(epochs):
        k = learning_rate / (1 + epoch)
        for x in inputs:
            winner = find_winner(x, weights)
            w = weights[winner]
            weights[winner] = w + k * (x - w)
    return weights
