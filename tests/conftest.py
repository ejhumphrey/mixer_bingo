import pytest

import json
import os


@pytest.fixture(scope='module')
def sample_data():
    data_file = os.path.join(os.path.dirname(__file__),
                             os.path.pardir, 'sample_data.json')
    return json.load(open(data_file))
