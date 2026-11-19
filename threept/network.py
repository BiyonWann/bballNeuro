import numpy as np


def find_winner(shot: np.ndarray, weights: np.ndarray) -> int:
    # (n_cells, 2) - (2,)
    distances = np.linalg.norm(weights - shot, axis=1)
    return int(np.argmin(distances))


def init_weights(inputs: np.ndarray, n_cells: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rows = rng.choice(len(inputs), n_cells, replace=False)
    return inputs[rows].astype(float)


def train(
    inputs: np.ndarray,
    n_cells: int = 6,
    epochs: int = 8,
    learning_rate: float = 0.05,
    seed: int = 0,
    initial_weights: np.ndarray | None = None,
    leak_rate: float = 0.0,
) -> np.ndarray:
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
            _leak_losers(x, weights, winner, leak_rate * k)
            w = weights[winner]
            weights[winner] = w + k * (x - w)
    return weights


def _leak_losers(x: np.ndarray, weights: np.ndarray, winner: int, leak_k: float) -> None:
    if leak_k == 0:
        return
    losers = np.arange(len(weights)) != winner
    w = weights[losers]
    # (n_losers, 2) - (2,)
    weights[losers] = w + leak_k * (x - w)


def assign(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    # (n_shots, 1, 2) - (1, n_cells, 2)
    differences = inputs[:, np.newaxis, :] - weights[np.newaxis, :, :]
    distances = np.linalg.norm(differences, axis=2)
    return np.argmin(distances, axis=1)


def count_wins(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    labels = assign(inputs, weights)
    return np.bincount(labels, minlength=len(weights))


def find_dead_cells(inputs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    wins = count_wins(inputs, weights)
    return np.flatnonzero(wins == 0)
