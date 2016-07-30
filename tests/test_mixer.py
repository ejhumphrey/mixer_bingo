import pytest

import mixer


def test_import():
    assert mixer is not None


def test_validate(sample_data):
    assert mixer.validate(sample_data)
