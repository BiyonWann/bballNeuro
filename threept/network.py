"""Competitive learning network: Euclidean winner, winner-only update."""

import numpy as np


def find_winner(shot: np.ndarray, weights: np.ndarray) -> int:
    """Return the index of the weight vector closest to the shot.

    Ties go to the lowest index.
    """
    # (n_cells, 2) - (2,)
    distances = np.linalg.norm(weights - shot, axis=1)
    return int(np.argmin(distances))
