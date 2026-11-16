import numpy as np
import pytest

from threept.network import (
    assign,
    count_wins,
    find_dead_cells,
    find_winner,
    init_weights,
    train,
)


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


def test_train_does_not_mutate_initial_weights_or_inputs():
    inputs = np.array([[2.0, 4.0]])
    initial = np.array([[0.0, 0.0], [10.0, 10.0]])
    inputs_before = inputs.copy()
    initial_before = initial.copy()

    train(inputs, n_cells=2, epochs=1, learning_rate=0.5, initial_weights=initial)

    assert np.array_equal(inputs, inputs_before)
    assert np.array_equal(initial, initial_before)


def test_train_zero_epochs_returns_copy():
    inputs = np.array([[2.0, 4.0]])
    initial = np.array([[0.0, 0.0], [10.0, 10.0]])

    weights = train(inputs, n_cells=2, epochs=0, learning_rate=0.5, initial_weights=initial)

    assert np.array_equal(weights, initial)
    assert weights is not initial


def test_train_initial_weights_wrong_length_raises():
    inputs = np.array([[2.0, 4.0]])
    initial = np.array([[0.0, 0.0], [10.0, 10.0]])

    with pytest.raises(ValueError):
        train(inputs, n_cells=3, epochs=1, initial_weights=initial)


def test_train_int_initial_weights_give_float_result():
    inputs = np.array([[2, 4]])
    initial = np.array([[0, 0], [10, 10]])

    weights = train(inputs, n_cells=2, epochs=1, learning_rate=0.5, initial_weights=initial)

    assert weights.dtype == float
    assert np.array_equal(weights, np.array([[1.0, 2.0], [10.0, 10.0]]))


def make_cluster_near_origin():
    rng = np.random.default_rng(0)
    return rng.normal(0.0, 1.0, size=(50, 2))


def test_train_leak_moves_loser_by_exact_amount():
    inputs = np.array([[4.0, 0.0]])
    initial = np.array([[1.0, 0.0], [10.0, 0.0]])

    weights = train(
        inputs, n_cells=2, epochs=1, learning_rate=0.5, initial_weights=initial, leak_rate=0.5
    )

    assert np.array_equal(weights, np.array([[2.5, 0.0], [8.5, 0.0]]))


def test_train_leak_moves_dead_cell_toward_data():
    inputs = make_cluster_near_origin()
    initial = np.array([[0.0, 0.0], [100.0, 100.0]])
    data_mean = inputs.mean(axis=0)
    start_distance = np.linalg.norm(initial[1] - data_mean)

    weights = train(inputs, n_cells=2, initial_weights=initial, leak_rate=0.1)

    assert np.linalg.norm(weights[1] - data_mean) < start_distance


def test_train_without_leak_leaves_dead_cell_fixed():
    inputs = make_cluster_near_origin()
    initial = np.array([[0.0, 0.0], [100.0, 100.0]])

    weights = train(inputs, n_cells=2, initial_weights=initial, leak_rate=0.0)

    assert np.array_equal(weights[1], np.array([100.0, 100.0]))


def test_train_shuffle_depends_on_seed():
    inputs = np.array([[float(i), 0.0] for i in range(10)])
    initial = np.array([[5.0, 0.0]])

    first = train(inputs, n_cells=1, epochs=1, learning_rate=0.5, seed=0, initial_weights=initial)
    second = train(inputs, n_cells=1, epochs=1, learning_rate=0.5, seed=1, initial_weights=initial)

    assert not np.array_equal(first, second)


def test_assign_matches_find_winner():
    rng = np.random.default_rng(0)
    inputs = rng.uniform(-25.0, 25.0, size=(100, 2))
    weights = rng.uniform(-25.0, 25.0, size=(6, 2))
    expected = np.array([find_winner(shot, weights) for shot in inputs])

    labels = assign(inputs, weights)

    assert np.array_equal(labels, expected)


def test_assign_tie_goes_to_lowest_index():
    inputs = np.array([[0.0, 0.0], [2.0, 0.0]])
    weights = np.array([[1.0, 0.0], [-1.0, 0.0], [3.0, 0.0]])

    labels = assign(inputs, weights)

    assert np.array_equal(labels, np.array([0, 0]))


def test_assign_returns_int_labels():
    inputs = np.array([[0.0, 0.0], [2.0, 0.0]])
    weights = np.array([[1.0, 0.0], [-1.0, 0.0], [3.0, 0.0]])

    labels = assign(inputs, weights)

    assert labels.shape == (2,)
    assert np.issubdtype(labels.dtype, np.integer)


def test_count_wins_counts_each_cell():
    inputs = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
    weights = np.array([[0.0, 0.0], [10.0, 0.0], [50.0, 50.0]])

    wins = count_wins(inputs, weights)

    assert np.array_equal(wins, np.array([2, 1, 0]))
    assert np.issubdtype(wins.dtype, np.integer)


def test_find_dead_cells_reports_unused_cell():
    inputs = np.array([[0.0, 0.0], [0.1, 0.0], [10.0, 0.0]])
    weights = np.array([[0.0, 0.0], [10.0, 0.0], [50.0, 50.0]])

    dead = find_dead_cells(inputs, weights)

    assert np.array_equal(dead, np.array([2]))


def test_find_dead_cells_empty_when_all_win():
    inputs = np.array([[0.0, 0.0], [10.0, 0.0]])
    weights = np.array([[0.0, 0.0], [10.0, 0.0]])

    dead = find_dead_cells(inputs, weights)

    assert len(dead) == 0
