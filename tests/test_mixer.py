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
