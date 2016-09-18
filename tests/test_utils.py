## imports
import pytest
import numpy as np
import pandas as pd
import pymltk.utils as utils


## tests
def test_mode():
    ## setup data
    s = pd.Series(['A', 'B', 'B', 'B'], dtype='category')
    s2 = pd.Series(['A', 'A', 'A'], dtype='object')
    s3 = pd.Series(['C', 'A', 'B'], dtype='object')
    s4 = pd.Series(['C', 'A', 'B', np.nan, 'A'], dtype='object')
    s5 = pd.Series([np.nan, np.nan], dtype='object')

    ## wrong input
    with pytest.raises(ValueError):
            utils.mode()
    with pytest.raises(ValueError):
            utils.mode(s5)
          
    ## check functionality
    assert(utils.mode(s) == 'B')
    assert(utils.mode(s2) == 'A')
    assert(utils.mode(s3) == 'A')
    assert(utils.mode(s4) == 'A')
