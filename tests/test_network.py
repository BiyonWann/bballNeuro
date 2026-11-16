import numpy as np
import pytest

from threept.network import find_winner, init_weights, train


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


def make_line_inputs(n_rows):
    return np.array([[float(i), 10.0 * i] for i in range(n_rows)])


def test_init_weights_rows_come_from_inputs():
    inputs = make_line_inputs(10)

    weights = init_weights(inputs, 4, seed=0)

    assert weights.shape == (4, 2)
    for row in weights:
        assert (inputs == row).all(axis=1).any()


def test_init_weights_rows_are_distinct():
    inputs = make_line_inputs(10)

    weights = init_weights(inputs, 10, seed=0)

    assert len(np.unique(weights, axis=0)) == 10


def test_init_weights_returns_copy():
    inputs = make_line_inputs(10)
    original = inputs.copy()

    weights = init_weights(inputs, 4, seed=0)
    weights[0, 0] = -1.0

    assert np.array_equal(inputs, original)


def test_init_weights_same_seed_same_rows():
    inputs = make_line_inputs(10)

    first = init_weights(inputs, 4, seed=0)
    second = init_weights(inputs, 4, seed=0)

    assert np.array_equal(first, second)


def test_init_weights_different_seed_different_rows():
    inputs = make_line_inputs(10)

    first = init_weights(inputs, 4, seed=0)
    second = init_weights(inputs, 4, seed=1)

    assert not np.array_equal(first, second)


def test_init_weights_too_many_cells_raises():
    inputs = make_line_inputs(3)

    with pytest.raises(ValueError):
        init_weights(inputs, 4, seed=0)


def test_train_single_update_moves_only_winner():
    inputs = np.array([[2.0, 4.0]])
    initial = np.array([[0.0, 0.0], [10.0, 10.0]])

    weights = train(inputs, n_cells=2, epochs=1, learning_rate=0.5, initial_weights=initial)

    assert np.array_equal(weights, np.array([[1.0, 2.0], [10.0, 10.0]]))


def test_train_one_epoch_visits_every_shot():
    inputs = np.array([[0.0, 0.0], [10.0, 0.0]])
    initial = np.array([[1.0, 0.0], [9.0, 0.0]])

    weights = train(inputs, n_cells=2, epochs=1, learning_rate=0.5, initial_weights=initial)

    assert np.array_equal(weights, np.array([[0.5, 0.0], [9.5, 0.0]]))


def test_train_learning_rate_decays_each_epoch():
    inputs = np.array([[4.0, 0.0]])
    initial = np.array([[0.0, 0.0], [20.0, 0.0]])

    weights = train(inputs, n_cells=2, epochs=2, learning_rate=0.5, initial_weights=initial)

    assert np.array_equal(weights, np.array([[2.5, 0.0], [20.0, 0.0]]))


def test_train_resets_learning_rate_each_call():
    inputs = np.array([[4.0, 0.0]])
    initial = np.array([[0.0, 0.0], [20.0, 0.0]])
    first = train(inputs, n_cells=2, epochs=1, learning_rate=0.5, initial_weights=initial)

    second = train(inputs, n_cells=2, epochs=1, learning_rate=0.5, initial_weights=first)

    assert np.array_equal(second, np.array([[3.0, 0.0], [20.0, 0.0]]))
