## imports
import warnings
import pytest
import numpy as np
import pandas as pd
import dask.dataframe as dd
import pymltk.preprocessing as pp



## tests
def test_parse_columns():
    ## setup data
    cats = pd.Series(['A', 'B', 'C', 'D', 'A', 'B', 'B'], dtype='category')
    objs = pd.Series(['A', 'B', 'C', 'D', 'A', 'B', 'B'], dtype='object')
    ints = pd.Series(np.random.rand(7), dtype='float')
    floats = pd.Series(np.random.rand(7), dtype='float')
    pandas_data = pd.concat([cats, objs, ints, floats], axis=1)
    pandas_data.columns = ['A', 'B', 'C', 'D']
    dask_data = dd.from_pandas(pandas_data, npartitions=2)

    ## run tests for all data sets
    for data in [pandas_data, dask_data]:

        ## wrong input
        with pytest.raises(ValueError):
            pp.parse_columns()
        with pytest.raises(ValueError):
            pp.parse_columns(data=data, dtypes=['a'])
        with pytest.raises(ValueError):
            pp.parse_columns(data=data, dtypes={'str', 'a', 'b'})
        with pytest.raises(ValueError):
            pp.parse_columns(data=data, dtypes={'hello': 'foo'})
        with pytest.raises(NotImplementedError):
            pp.parse_columns(data=data, dtypes={'A': 'object'}, copy=False)
        with pytest.warns(UserWarning):
            pp.parse_columns(data=data, dtypes='foo')

        ## correct input / testing of functionality
        c1 = pp.parse_columns(data=data, dtypes='object')
        assert(len(np.unique(c1.dtypes)) == 1)
        c2 = pp.parse_columns(data=data, dtypes=['str', 'str', 'float', 'float'], verbose=False)
        assert(c2.dtypes['A'].name == 'object')
        assert(c2.dtypes['B'].name == 'object')
        assert(c2.dtypes['C'].name == 'float64')
        assert(c2.dtypes['D'].name == 'float64')
        c3 = pp.parse_columns(data=data, dtypes={'C': 'object'})
        assert(c3.dtypes['A'].name == 'category')
        assert(c3.dtypes['B'].name == 'object')
        assert(c3.dtypes['C'].name == 'object')
        assert(c3.dtypes['D'].name == 'float64')


def test_parse_missing():
    ## setup data
    c1 = pd.Series(['A', 'B', 'C', 'D', 'A', 'B', 'B'], dtype='object')
    c2 = pd.Series(['A', 'B', 'C', 'D', 'A', 'B', 'B'], dtype='object')
    c3 = pd.Series(['1', '2', '3', '4', '5', '6', '7'], dtype='object')
    c4 = pd.Series(np.random.rand(7), dtype='object')
    pandas_data = pd.concat([c1, c2, c3, c4], axis=1)
    pandas_data.columns = ['A', 'B', 'C', 'D']
    dask_data = dd.from_pandas(pandas_data, npartitions=2)

    ## wrong input
    with pytest.raises(ValueError):
        pp.parse_missings()
    with pytest.raises(ValueError):
        pp.parse_missings(pandas_data, missings={1, 2, 3})
    with pytest.raises(ValueError):
        pp.parse_missings(pandas_data, missings={'Z': [1, 2, 3]})

    ## run tests for pandas data // correct input / testing of functionality
    d1 = pp.parse_missings(pandas_data, missings=3)
    assert(np.sum(d1.isnull().values) == 0)
    d2 = pp.parse_missings(pandas_data, missings=['A', '1'])
    assert(np.sum(d2.A.isnull()) == 2)
    assert(np.sum(d2.B.isnull()) == 2)
    assert(np.sum(d2.C.isnull()) == 1)
    assert(np.sum(d2.D.isnull()) == 0)
    d3 = pp.parse_missings(pandas_data, missings={'A': ['A']})
    assert(np.sum(d3.A.isnull()) == 2)
    assert(np.sum(d3.B.isnull()) == 0)
    assert(np.sum(d3.C.isnull()) == 0)
    assert(np.sum(d3.D.isnull()) == 0)

    ## run tests for dask data // correct input / testing of functionality
    d1 = pp.parse_missings(dask_data, missings=[3])
    assert(sum(d1.isnull().sum().compute()) == 0)
    d2 = pp.parse_missings(dask_data, missings=['A', '1'])
    assert(d2.A.isnull().sum().compute())
    assert(d2.B.isnull().sum().compute())
    assert(d2.C.notnull().sum().compute())
    assert(sum(d2.isnull().sum().compute()) == 5)
    d3 = pp.parse_missings(dask_data, missings={'A': ['A']}, verbose=False)
    assert(d3.A.isnull().sum().compute() == 2)
    assert(d3.B.isnull().sum().compute() == 0)
    assert(d3.C.isnull().sum().compute() == 0)
    assert(d3.D.isnull().sum().compute() == 0)


def test_merge_levels():
    ## setup data
    s = pd.Series(['A', 'B', np.nan, 'A', 'A', 'B', 'A', 'A', 'C', 'C', 'A',
                   'B', 'D', 'E', 'D', 'D', 'E', 'F', np.nan], dtype='object')
    s2 = pd.Series(['A', 'B', np.nan, 'A', 'A', 'B', 'A', 'A', 'C', 'C', 'A', 
                    'B', 'D', 'E', 'D', 'D', 'E', 'F', np.nan], dtype='object')
    pandas_data = pd.concat([s, s2], axis=1)
    pandas_data.columns = ['A', 'B']
    dask_data = dd.from_pandas(pandas_data, npartitions=2)

    ## wrong input
    with pytest.raises(ValueError):
        pp.merge_levels(data=None)
    with pytest.raises(ValueError):
        pp.merge_levels(data=pandas_data, merging=None, max_levels=None, min_percent=None)
    with pytest.raises(ValueError):
        pp.merge_levels(data=pandas_data, merging={'A': 'Foo'})
    with pytest.raises(ValueError):
        pp.merge_levels(data=pandas_data, merging={'A': ['E', 'F']}, new_label=42)
    with pytest.raises(ValueError):
        pp.merge_levels(data=pandas_data, max_levels=4.2)
    with pytest.raises(ValueError):
        pp.merge_levels(data=pandas_data, merging='Hello')
    with pytest.raises(ValueError):
        pp.merge_levels(data=pandas_data, min_percent=42)
    with pytest.raises(ValueError):
        pp.merge_levels(data=pandas_data, merging={'Foo': [1, 2, 3]})

    ## tests for pandas data // correct input / testing of functionality
    ## usage of merging
    d, merge_dict = pp.merge_levels(data=pandas_data, merging={'A': ['D', 'E', 'F']},
                                    merge_dict=True, new_label="Others", copy=True)
    assert(set(merge_dict['A']) == set(['D', 'E', 'F']))
    assert(np.sum(d.A.isnull()) == 2)
    assert(np.size(np.unique(d.A)) == 5)
    assert(np.size(np.unique(d.A)) == 5)

    ## usage of max_levels
    d = pp.merge_levels(data=pandas_data, max_levels = 3, merge_dict=False, new_label="Others", copy=True)
    assert(np.sum(d.A.isnull()) == 0)
    assert(np.size(np.unique(d.A)) == 3)
    d = pp.merge_levels(data=pandas_data, max_levels = 25, merge_dict=False, new_label="Others", copy=True)
    assert(np.sum(d.A.isnull()) == 2)
    assert(np.size(np.unique(d.A)) == 7)

    ## usage of min_percent
    d = pp.merge_levels(data=pandas_data, min_percent = 0.31, merge_dict=False, new_label="Others", copy=True)
    assert(np.sum(d.A.isnull()) == 0)
    assert(np.size(np.unique(d.A)) == 5)

    ## combination of min_percent and merging
    d = pp.merge_levels(data=pandas_data, merging = {'A': ['E', 'F']}, min_percent = 0.5,
                        merge_dict=False, new_label="Others", copy=True)
    assert(np.sum(d.A.isnull()) == 0)
    assert(np.size(np.unique(d.A)) == 3)

    ## combination of merging, min_percent and max_levels
    d = pp.merge_levels(data=pandas_data, merging = {'A': ['E', 'F']}, min_percent = 0.5, max_levels = 2, 
                        merge_dict=False, new_label="Others", copy=True)
    assert(np.sum(d.A.isnull()) == 0)
    assert(np.size(np.unique(d.A)) == 1)

    ## tests for dask data // correct input / testing of functionality
    ## usage of merging
    d, merge_dict = pp.merge_levels(data=dask_data, merging={'A': ['D', 'E', 'F']},
                                    merge_dict=True, new_label="Others", copy=True)
    assert(set(merge_dict['A']) == set(['D', 'E', 'F']))
    assert(d.A.isnull().sum().compute() == 2)
    assert(np.size(np.unique(d.A) == 5))
    assert(np.size(np.unique(d.A)) == 5)

    ## usage of max_levels
    d = pp.merge_levels(data=dask_data, max_levels = 3, merge_dict=False, new_label="Others", copy=True)
    assert(d.A.isnull().sum().compute() == 0)
    assert(np.size(np.unique(d.A)) == 3)
    d = pp.merge_levels(data=dask_data, max_levels = 25, merge_dict=False, new_label="Others", copy=True)
    assert(d.A.isnull().sum().compute() == 2)
    assert(np.size(np.unique(d.A)) == 7)

    ## usage of min_percent
    d = pp.merge_levels(data=dask_data, min_percent = 0.31, merge_dict=False, new_label="Others", copy=True)
    assert(d.A.isnull().sum().compute() == 0)
    assert(np.size(np.unique(d.A)) == 5)

    ## combination of min_percent and merging
    d = pp.merge_levels(data=dask_data, merging = {'A': ['E', 'F']}, min_percent = 0.5,
                        merge_dict=False, new_label="Others", copy=True)
    assert(d.A.isnull().sum().compute() == 0)
    assert(np.size(np.unique(d.A)) == 3)

    ## combination of merging, min_percent and max_levels
    d = pp.merge_levels(data=dask_data, merging = {'A': ['E', 'F']}, min_percent = 0.5, max_levels = 2, 
                        merge_dict=False, new_label="Others", copy=True, verbose=False)
    assert(d.A.isnull().sum().compute() == 0)
    assert(np.size(np.unique(d.A)) == 1)



def test_impute_missings():
    ## setup data
    s = pd.Series(['A', 'B', 'B', np.nan], dtype='str')
    f = pd.Series([1.23, 1.2, 1.3, 1.5, np.nan, np.nan, 1.8], dtype='float')
    pandas_data = pd.concat([s, f], axis = 1)
    pandas_data.columns = ['A', 'B']
    dask_data = dd.from_pandas(pandas_data, npartitions=2)

    ## wrong input
    with pytest.raises(ValueError):
        pp.impute_missings(data=None)
    with pytest.raises(ValueError):
        pp.impute_missings(data=pandas_data, imputing=None)
    with pytest.raises(ValueError):
        pp.impute_missings(data=pandas_data, imputing='foo')
    with pytest.raises(ValueError):
        pp.impute_missings(data=pandas_data, imputing={'foo': 'mode'})
    with pytest.raises(ValueError):
        pp.impute_missings(data=pandas_data, imputing={'A': 'mode', 'B': 'foo'})
    with pytest.raises(ValueError):
        pp.impute_missings(data=pandas_data, imputing={'A': 'foo', 'B': 'mean'})

    ## pandas_data // check functionality
    d = pp.impute_missings(pandas_data, imputing={'A':'mode'}, impute_dict=False)
    assert(d.A[3] == 'B')
    assert(np.sum(d.A.isnull()) == 0)
    assert(np.sum(d.B.isnull()) == 2)
    d = pp.impute_missings(pandas_data, imputing={'A': 'level'}, impute_dict=False)
    assert(d.A[3] == 'Missing')
    assert(np.sum(d.A.isnull()) == 0)
    assert(np.sum(d.B.isnull()) == 2)
    d = pp.impute_missings(pandas_data, imputing={'B': 'mean'}, impute_dict=False)
    assert(d.B[4] == 1.406)
    assert(d.B[5] == 1.406)
    assert(np.sum(d.A.isnull()) == 4)
    assert(np.sum(d.B.isnull()) == 0)
    d, dct = pp.impute_missings(pandas_data, imputing={'A': 'mode', 'B': 'median'}, impute_dict=True)
    assert(d.A[3] == 'B')
    assert(d.B[4] == 1.3)
    assert(d.B[5] == 1.3)
    assert(np.sum(d.A.isnull()) == 0)
    assert(np.sum(d.B.isnull()) == 0)
    assert(dct['A'] == 'B')
    assert(dct['B'] == 1.3)

    ## pandas_data // check functionality
    d = pp.impute_missings(dask_data, imputing={'A':'mode'}, impute_dict=False)
    assert(d.compute().A[3] == 'B')
    assert(d.A.isnull().sum().compute() == 0)
    assert(d.B.isnull().sum().compute() == 2)
    d = pp.impute_missings(dask_data, imputing={'A': 'level'}, impute_dict=False)
    assert(d.compute().A[3] == 'Missing')
    assert(d.A.isnull().sum().compute() == 0)
    assert(d.B.isnull().sum().compute() == 2)
    d = pp.impute_missings(dask_data, imputing={'B': 'median'}, impute_dict=False, verbose=False)
    assert(d.compute().B[4] == 1.3)
    assert(d.compute().B[5] == 1.3)
    assert(d.A.isnull().sum().compute() == 4)
    assert(d.B.isnull().sum().compute() == 0)
    d, dct = pp.impute_missings(dask_data, imputing={'A': 'mode', 'B': 'median'}, impute_dict=True)
    assert(d.compute().A[3] == 'B')
    assert(d.compute().B[4] == 1.3)
    assert(d.compute().B[5] == 1.3)
    assert(d.A.isnull().sum().compute() == 0)
    assert(d.B.isnull().sum().compute() == 0)
    assert(dct['A'] == 'B')
    assert(dct['B'] == 1.3)


    
def test_remove_constants():
    ## setup data
    dint_pandas = pd.DataFrame(np.transpose(np.vstack((np.repeat(10, 100),
                                                       np.concatenate((np.repeat(10, 90), np.random.randn(10))),
                                                       np.concatenate((np.repeat(10, 80), np.random.randn(20))),
                                                       np.concatenate((np.repeat(10, 70), np.random.randn(30))),
                                                       np.concatenate((np.repeat(10, 60), np.random.randn(40))),
                                                       np.concatenate((np.repeat(10, 50), np.random.randn(50))),
                                                       np.concatenate((np.repeat(10, 40), np.random.randn(60))),
                                                       np.concatenate((np.repeat(10, 30), np.random.randn(70))),
                                                       np.concatenate((np.repeat(10, 20), np.random.randn(80))),
                                                       np.concatenate((np.repeat(10, 10), np.random.randn(90))),
                                                       np.random.randn(100)))), columns = list('ABCDEFGHIJK'))
    dint_dask = dd.from_pandas(dint_pandas, npartitions=2)
    dstr_pandas = pd.DataFrame(np.transpose(np.vstack((np.repeat('A', 100),
                                                       np.concatenate((np.repeat('A', 90), np.random.choice(['B', 'C', 'D', 'E', 'F', 'G', 'H'], 10))),
                                                       np.concatenate((np.repeat('A', 80), np.random.choice(['B', 'C', 'D', 'E', 'F', 'G', 'H'], 20))),
                                                       np.concatenate((np.repeat('A', 70), np.random.choice(['B', 'C', 'D', 'E', 'F', 'G', 'H'], 30))),
                                                       np.concatenate((np.repeat('A', 60), np.random.choice(['B', 'C', 'D', 'E', 'F', 'G', 'H'], 40))),
                                                       np.concatenate((np.repeat('A', 50), np.random.choice(['B', 'C', 'D', 'E', 'F', 'G', 'H'], 50))),
                                                       np.concatenate((np.repeat('A', 40), np.random.choice(['B', 'C', 'D', 'E', 'F', 'G', 'H'], 60))),
                                                       np.concatenate((np.repeat('A', 30), np.random.choice(['B', 'C', 'D', 'E', 'F', 'G', 'H'], 70))),
                                                       np.concatenate((np.repeat('A', 20), np.random.choice(['B', 'C', 'D', 'E', 'F', 'G', 'H'], 80))),
                                                       np.concatenate((np.repeat('A', 10), np.random.choice(['B', 'C', 'D', 'E', 'F', 'G', 'H'], 90))),
                                                       np.random.choice(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 100)))), columns = list('ABCDEFGHIJK'))
    dstr_dask = dd.from_pandas(dstr_pandas, npartitions=2)
    dmiss = dint_pandas.copy()
    dmiss2 = dint_pandas.copy()
    dmiss.loc[:, 'A'] = np.nan
    dmiss2.loc[np.random.randint(0, dmiss.shape[0], 10),'A'] = np.repeat(np.nan, 10)


    ## wrong input
    with pytest.raises(ValueError):
        pp.remove_constants(data=None)
    with pytest.raises(ValueError):
        pp.remove_constants(data=dint_pandas, percent='foo')
    with pytest.raises(ValueError):
        pp.remove_constants(data=dint_pandas, percent=0.0, keep=42)
    with pytest.raises(ValueError):
        pp.remove_constants(data=dint_pandas, percent=0.0, keep='A', ignore_missing='foo')

    ## check functionality for string and integer data set
    for d in [dint_pandas, dint_dask, dstr_pandas, dstr_dask]:
        ## general functioning
        dm = pp.remove_constants(data=d, removal_dict=False)
        assert(len(dm.columns) == 10)
        assert('A' not in dm.columns.values)

        ## percent
        dm = pp.remove_constants(data=d, percent = 0.05, removal_dict=False)
        assert(len(dm.columns) == 10)        
        assert('A' not in dm.columns.values)

        dm = pp.remove_constants(data=d, percent = 0.11, removal_dict=False)
        assert(len(dm.columns) == 9)
        assert('A' not in dm.columns.values)
        assert('B' not in dm.columns.values)

        dm = pp.remove_constants(data=d, percent = 0.21, removal_dict=False)
        assert(len(dm.columns) == 8)
        assert('A' not in dm.columns.values)
        assert('B' not in dm.columns.values)
        assert('C' not in dm.columns.values)

        ## columns to keep
        dm = pp.remove_constants(data=d, percent = 0.21, keep=['A', 'B', 'C'], removal_dict=False)
        assert(len(dm.columns) == 11)
        assert('A' in dm.columns.values)
        assert('B' in dm.columns.values)
        assert('C' in dm.columns.values)

        ## verbosity
        dm = pp.remove_constants(data=d, percent = 0.21, keep=['A', 'B', 'C'], verbose=False, removal_dict=False)
        assert(len(dm.columns) == 11)
        assert('A' in dm.columns.values)
        assert('B' in dm.columns.values)
        assert('C' in dm.columns.values)

    ## missing values
    dm = pp.remove_constants(data=dmiss, percent = 0.0, ignore_missing = True, removal_dict=False)
    assert(len(dm.columns) == 10)
    assert('A' not in dm.columns.values)

    dm = pp.remove_constants(data=dmiss, percent = 0.0, ignore_missing = False, removal_dict=False)
    assert(len(dm.columns) == 10)
    assert('A' not in dm.columns.values)

    dm = pp.remove_constants(data=dmiss2, percent = 0.0, ignore_missing = True, removal_dict=False)
    assert(len(dm.columns) == 10)
    assert('A' not in dm.columns.values)
    
    dm = pp.remove_constants(data=dmiss2, percent = 0.0, ignore_missing = False, removal_dict=False)
    assert(len(dm.columns) == 11)
    assert('A' in dm.columns.values)

    dm, dct = pp.remove_constants(data=dmiss2, percent = 0.0, ignore_missing = False, removal_dict=True)
    assert(len(dm.columns) == 11)
    assert('A' in dm.columns.values)
    assert(not dct['A'])
