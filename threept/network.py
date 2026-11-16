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
