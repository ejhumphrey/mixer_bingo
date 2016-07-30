import pytest

import os
import pandas as pd


@pytest.fixture(scope='module')
def sample_csv(tmpdir):
    fout = os.path.join(str(tmpdir), 'user_data.csv')
    rows = []
    dframe = pd.DataFrame.from_records(data=rows)
    dframe.to_csv(fout)
    return fout
