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
    elif len(initial_weights) != n_cells:
        raise ValueError(f"initial_weights has {len(initial_weights)} rows, expected {n_cells}")
    else:
        weights = np.array(initial_weights, dtype=float)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        k = learning_rate / (1 + epoch)
        order = rng.permutation(len(inputs))
        for x in inputs[order]:
            winner = find_winner(x, weights)
            leak_losers(x, weights, winner, leak_rate * k)
            w = weights[winner]
            weights[winner] = w + k * (x - w)
    return weights


def leak_losers(x: np.ndarray, weights: np.ndarray, winner: int, leak_k: float) -> None:
    """Move every cell except the winner toward the shot, in place, by leak_k * (x - w).

    Does nothing when leak_k is 0, which is plain winner-take-all learning.
    """
    if leak_k == 0:
        return
    losers = np.arange(len(weights)) != winner
    w = weights[losers]
    # (n_losers, 2) - (2,)
    weights[losers] = w + leak_k * (x - w)
