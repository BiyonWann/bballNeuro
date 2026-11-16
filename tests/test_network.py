import numpy as np
from threept.network import find_winner


def test_find_winner_picks_closest_cell():
    shot = np.array([3.0, 4.0])
    weights = np.array([[0.0, 0.0], [3.0, 5.0], [10.0, 10.0]])

    winner = find_winner(shot, weights)

    assert winner == 1


def test_find_winner_uses_distance_not_dot_product():
    shot = np.array([1.0, 1.0])
    weights = np.array([[10.0, 10.0], [1.0, 1.0]])

    winner = find_winner(shot, weights)

    assert winner == 1


def test_find_winner_returns_int():
    shot = np.array([3.0, 4.0])
    weights = np.array([[0.0, 0.0], [3.0, 5.0], [10.0, 10.0]])

    winner = find_winner(shot, weights)

    assert type(winner) is int


def test_find_winner_tie_goes_to_lowest_index():
    shot = np.array([0.0, 0.0])
    weights = np.array([[5.0, 5.0], [1.0, 0.0], [0.0, 1.0]])

    winner = find_winner(shot, weights)

    assert winner == 1
