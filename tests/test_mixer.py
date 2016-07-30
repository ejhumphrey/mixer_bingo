import pytest

import numpy as np

import mixer


def test_import():
    assert mixer is not None


def test_validate(sample_data):
    assert mixer.validate(sample_data)


def test_tokenize():
    data = ['ab', 'a', 'abc']
    tokens = mixer.tokenize(data)
    assert len(tokens) == 3
    for n, k in enumerate('abc'):
        assert tokens[k] == n


def test_items_to_bitmap():
    data = ['ab', 'a', 'abc']
    bmap, enums = mixer.items_to_bitmap(data)
    exp_bmap = np.array([[True, True, False],
                         [True, False, False],
                         [True, True, True]])
    np.testing.assert_array_equal(exp_bmap, bmap)


def test_build_graph_basic(sample_data):
    graph = mixer.build_graph(
        sample_data, forced_edges=None, null_edges=None, interest_func='l0',
        seniority_func='l0', combination_func=np.sum)
    assert graph is not None


def test_build_graph_forced_edge(sample_data):
    graph = mixer.build_graph(
        sample_data, forced_edges=[(0, 1)], null_edges=None,
        interest_func='l0', seniority_func='l0', combination_func=np.sum)
    assert graph is not None


def test_build_graph_null_edge(sample_data):
    graph = mixer.build_graph(
        sample_data, forced_edges=None, null_edges=[(0, 1)],
        interest_func='l0', seniority_func='l0', combination_func=np.sum)
    assert graph is not None
