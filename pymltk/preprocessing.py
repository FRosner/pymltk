## imports
import warnings
import math
import numpy as np
import pandas as pd
import dask.dataframe as dd
import pymltk.utils as ut



## statics
IMPUTE_DICT = {str: {'mode': ut.mode,
                     'level': lambda x: 'Missing'}, # category is captured here
               float: {'median': np.median, # int is captured here
                       'mean': np.mean}}



## plain functions
def parse_columns(data=None, dtypes=None, copy=True, verbose=True):
    """
    Parse columns of a given pandas/dask dataframe.

    Given either a pandas or a dask dataframe, parse_columns will try to parse the
    columns with the specified dtypes. If the parsing fails for a certain column,
    a warning will be raised and the next column will be parsed.

    Args:
        data: A pandas or dask dataframe object.
        dtypes: Dtypes to be enforced upon data. Either specified as a single string
            (the same dtype is assumed for all columns), as a list of strings (a dtype
            for every column has to be specified) or as a dictionary with the column
            names of data as keys.
        copy: Boolean, whether to copy the data before modifying it.
            Setting this to False currently only works with pandas.
        verbose: Boolean, whether to print status information during the parsing.

    Returns:
        The parsed dataframe with dtypes as requested.

    Raises:
        ValueError: If no data is specified. If dtypes are not specified as string,
            list of len(data.columns) or dictionary or if the dtypes dictionary contains
            keys which are not valid column names.
        NotImplementedError: Setting copy to False when data is a dask dataframe is not supported.
        UserWarning: If a column cannot be parsed with the specified dtype.
    """
    ## check input
    if data is None or (not isinstance(data, pd.core.frame.DataFrame)
                        and not isinstance(data, dd.core.DataFrame)):
        raise ValueError('Data must be specified as \
        pandas.core.frame.DataFrame or dask.core.DataFrame.')
    column_list = data.columns.tolist()
    if dtypes is None or (not isinstance(dtypes, str)
                          and not isinstance(dtypes, list)
                          and not isinstance(dtypes, dict)):
        raise ValueError('Dtypes must be specified as string, list or dict.')
    if isinstance(dtypes, list) and (len(dtypes) != len(column_list)):
        raise ValueError('If dtypes is specified as list, a dtype for \
        every column has to be specified.')
    if isinstance(dtypes, dict) and len(set(dtypes.keys()) - set(column_list)):
        raise ValueError('Not all entries of given dtypes dictionary map to a valid column name.')

    ## setup dtype dict for all columns
    if isinstance(dtypes, str):
        dtypes_dict = {}
        for col in column_list:
            dtypes_dict[col] = dtypes
    elif isinstance(dtypes, list):
        dtypes_dict = {}
        for i, col in enumerate(column_list):
            dtypes_dict[col] = dtypes[i]
    else:
        dtypes_dict = dtypes

    ## copy object if requesed
    if copy:
        if isinstance(data, pd.core.frame.DataFrame):
            data = data.copy()
    else:
        raise NotImplementedError('Dask does currently not support assignments without copying.')

    ## convert data as requested
    for col in dtypes_dict.keys():
        if verbose:
            print 'Setting dtype of column {} ...'.format(col)
        try:
            dct = {str(col): data[col].astype(dtypes_dict[col])}
            data = data.drop(col, axis=1).assign(**dct)
        except:
            warnings.warn('Column {} could not be converted to {}. \
            Keeping {} dtype.'.format(col, dtypes_dict[col], str(data.dtypes[col])))

    return data


def parse_missings(data=None, missings=None, copy=True, verbose=True):
    """
    Parse missings of a given pandas/dask dataframe.

    Given either a pandas or a dask dataframe, parse_missings will parse the
    specified values as missing values (represented with the numpy.nan type).

    Args:
        data: A pandas or dask dataframe object.
        missings: Specification of values to be parsed as missings. Either specified
            as a single string, a list of strings or as a dictionary with the column
            names of data as keys.
        copy: Boolean, whether to copy the data before modifying it.
            Setting this to False currently only works with pandas.
        verbose: Boolean, whether to print status information during the parsing.

    Returns:
        The parsed dataframe with missing values parsed as requested.

    Raises:
        ValueError: If no data is specified. If missings are not specified as string,
            list or dictionary or if the missing dictionary contains keys which are
            not valid column names.
        NotImplementedError: Setting copy to False when data is a dask dataframe is not supported.
    """
    ## check input
    if data is None or (not isinstance(data, pd.core.frame.DataFrame)
                        and not isinstance(data, dd.core.DataFrame)):
        raise ValueError('Data must be specified as \
        pandas.core.frame.DataFrame or dask.core.DataFrame.')
    column_list = data.columns.tolist()
    if missings is None or (not isinstance(missings, str)
                            and not isinstance(missings, int)
                            and not isinstance(missings, float)
                            and not isinstance(missings, list)
                            and not isinstance(missings, dict)):
        raise ValueError('Missings must be specified as int, float, string, list or dict.')
    if isinstance(missings, dict) and len(set(missings.keys()) - set(column_list)):
        raise ValueError('Not all entries of given missings dictionary map to a valid column name.')

    ## setup dtype dict for all columns
    if not isinstance(missings, dict):
        if not isinstance(missings, list):
            missings = [missings]
        missings_dict = {}
        for col in column_list:
            missings_dict[col] = missings
    else:
        missings_dict = missings

    ## copy object if requesed
    if copy:
        if isinstance(data, pd.core.frame.DataFrame):
            data = data.copy()
    else:
        raise NotImplementedError('Dask does currently not support assignments without copying.')

    ## convert data as requested
    for col in missings_dict.keys():
        if verbose:
            print 'Parsing missings of column {} ...'.format(col)
        if isinstance(data, pd.core.frame.DataFrame):
            dct = {str(col): data[col].where(~data[col].isin(missings_dict[col]), np.nan)}
        else:
            dct = {str(col): data[col].where(~data[col].isin(missings_dict[col]), np.nan).compute()}
        data = data.drop(col, axis=1).assign(**dct)

    return data


def merge_levels(data=None, merging=None, min_percent=None, max_levels=None,
                 new_label='Others', merge_dict=True, copy=True, verbose=True):
    """
    Merge levels of columns in a given pandas/dask dataframe.

    Given either a pandas or a dask dataframe, merge_levels will combine
    levels of columns into a common level labeled according to 'new_label'.
    There are several merging strategies implemented: Via a dictionary specific
    levels to be merged per column can be specified. The argument 'min_percent'
    allows to specify a minimum relative frequency for each level. All levels with
    a frequency less then 'min_percent' are merged together. 'max_levels'
    allows to specify a maximum number of levels per column. The least frequent
    levels are merged together until only 'max_level' levels remain. If multiple
    strategies are requested they are processed sequantielly in the following order:
    First, the explicit merging specified in 'merging' is processed. Second, all levels
    with a frequency less then 'min_percent' are merged together. In the last step,
    the least frequent levels are merged together until only 'max_levels' levels remain.

    Args:
        data: A pandas or dask dataframe object.
        merging: Dictionary, specification of levels to be merged together as dictionary
            with the keys being the column names of the given dataframe.
        min_percent: Float, the threshold relative frequency of each level. Levels with a
            frequency less or equal than 'min_percent' are merged together.
        max_levels: Integer, the maximum number of levels per column. The least frequent
            levels are merged together until only 'max_levels' levels remain.
        new_label: The new label of the merged levels.
        merge_dict: Boolean, whether to return a dictionary with
            the merging applied to each column.
        copy: Boolean, whether to copy the data before modifying it.
            Setting this to False currently only works with pandas.
        verbose: Boolean, whether to print status information during the parsing.

    Returns:
        The processed dataframe with levels merged as requested.

    Raises:
        ValueError: If no data is specified. If not at least one of 'merging', 'max_levels',
            or 'min_percent' is specified. If the to be merged levels are not specified
            as dictionary or if this dictionary contains keys which are not valid column names.
            If 'max_levels' is not a integer (if specified) or 'min_percent' is not a float < 0
            (if specified). If 'new_label' is None or not specified as string.
        NotImplementedError: Setting copy to False when data is a dask dataframe is not supported.
    """
    ## check input
    if data is None or (not isinstance(data, pd.core.frame.DataFrame)
                        and not isinstance(data, dd.core.DataFrame)):
        raise ValueError('Data must be specified as \
        pandas.core.frame.DataFrame or dask.core.DataFrame.')
    column_list = data.columns.tolist()
    if merging is None and min_percent is None and max_levels is None:
        raise ValueError('One of merging, min_percent and max_levels has to be specified.')
    if merging is not None and not isinstance(merging, dict):
        raise ValueError('Merging must be specified as dict.')
    if merging is not None and not all([isinstance(merging[col], list) for col in merging]):
        raise ValueError('All merging entries must be specified as list.')
    if min_percent is not None and not isinstance(min_percent, float):
        raise ValueError('min_percent must be specified as float.')
    if max_levels is not None and not isinstance(max_levels, int):
        raise ValueError('max_levels must be specified as int.')
    if new_label is None or not isinstance(new_label, str):
        raise ValueError('The label for merged levels must be specified as string.')
    if isinstance(merging, dict) and len(set(merging.keys()) - set(column_list)):
        raise ValueError('Not all entries of given merging dictionary map to a valid column name.')

    ## copy object if requested
    if copy:
        if isinstance(data, pd.core.frame.DataFrame):
            data = data.copy()
    else:
        raise NotImplementedError('Dask does currently not support assignments without copying.')

    ## merge dict if requested
    if merge_dict:
        merging_dict = {col: [] for col in column_list}

    ## merge data as requested
    for col in column_list:
        if verbose:
            print 'Merging levels of column {} ...'.format(col)

        ## unique values before
        uniqval = set(np.unique(data[col], return_counts=False))

        ## merging as given in dict
        if merging is not None and col in merging:
            if isinstance(data, pd.core.frame.DataFrame):
                dct = {str(col): pd.Series(np.where(~data[col].isin(merging[col]),
                                                    data[col], new_label))}
                ## does not work for some reason:
                ## data[col].where(~data[col].isin(merging[col]), new_label)}
            else:
                keep = ~data[col].isin(merging[col]).compute()
                dct = {str(col): data[col].where(keep, new_label).compute()}
            data = data.drop(col, axis=1).assign(**dct)

        ## additionally min percent if requested
        if min_percent is not None:
            val, freq = np.unique(data[col], return_counts=True)
            rfreq = freq/float(np.size(data[col]))
            rfreq = rfreq[np.argsort(freq)]
            rfreq = np.cumsum(rfreq)
            merge = val[np.argsort(freq)][rfreq <= min_percent]

            if isinstance(data, pd.core.frame.DataFrame):
                dct = {str(col): pd.Series(np.where(~data[col].isin(merge), data[col], new_label))}
                ##does not work for some reason: data[col].where(~data[col].isin(merge), new_label)}
            else:
                keep = ~data[col].isin(merge).compute()
                dct = {str(col): data[col].where(keep, new_label).compute()}
            data = data.drop(col, axis=1).assign(**dct)

        ## additionally max_level merging if requested
        if max_levels is not None:
            val, freq = np.unique(data[col], return_counts=True)
            nobs = np.size(val) - (max_levels - 1)
            if nobs > 0:
                merge = val[np.argsort(freq)][0:nobs]

                if isinstance(data, pd.core.frame.DataFrame):
                    dct = {str(col): pd.Series(np.where(~data[col].isin(merge),
                                                        data[col], new_label))}
                    ## does not work for some reason:
                    ## data[col].where(~data[col].isin(merge), new_label)}
                else:
                    keep = ~data[col].isin(merge).compute()
                    dct = {str(col): data[col].where(keep, new_label).compute()}
                data = data.drop(col, axis=1).assign(**dct)

        ## merging_dict
        if merge_dict:
            merging_dict[col] = list(uniqval - set(np.unique(data[col])))

    if merge_dict:
        return data, merging_dict
    else:
        return data


def impute_missings(data=None, imputing=None, impute_dict=True, copy=True, verbose=True):
    """
    Imputing missing values in column of a given pandas/dask dataframe.

    Given either a pandas or a dask dataframe, impute_missings will replace
    missing values with the values specified in 'imputing'.

    Args:
        data: A pandas or dask dataframe object.
        imputing: Dictionary, specification of the imputation strategy for each column.
            Currently, the following imputation strategies are supported: 'level' (missing
            values are encoded as new level) and 'mode' for string columns and 'mean'/'median'
            for numerical columns.
        impute_dict: Boolean, whether to return a dictionary with
            the imputed value of each column.
        copy: Boolean, whether to copy the data before modifying it.
            Setting this to False currently only works with pandas.
        verbose: Boolean, whether to print status information during the parsing.

    Returns:
        The processed dataframe with missing values imputed as requested.

    Raises:
        ValueError: If no data is specified. If 'imputing' was not specified
            as dictionary or if this dictionary contains keys which are not valid column names.
            If an invalid imputing strategy was selected, i.e., mean for columns with object dtype.
        NotImplementedError: Setting copy to False when data is a dask dataframe is not supported.
    """
    ## check input
    if data is None or (not isinstance(data, pd.core.frame.DataFrame)
                        and not isinstance(data, dd.core.DataFrame)):
        raise ValueError('Data must be specified as \
        pandas.core.frame.DataFrame or dask.core.DataFrame.')
    column_list = data.columns.tolist()
    if imputing is None:
        raise ValueError('imputing has to be specified.')
    if imputing is not None and not isinstance(imputing, dict):
        raise ValueError('imputing must be specified as dict.')
    if isinstance(imputing, dict) and (len(set(imputing.keys()) - set(column_list))):
        raise ValueError('Not all entries of given \
        imputing dictionary map to a valid column name.')

    ## copy object if requested
    if copy:
        if isinstance(data, pd.core.frame.DataFrame):
            data = data.copy()
    else:
        raise NotImplementedError('Dask does currently not support assignments without copying.')

    ## merge dict if requested
    if impute_dict:
        imputing_dict = {col: [] for col in column_list}

    ## merge data as requested
    for col in column_list:
        if verbose:
            print 'Imputing column {} ...'.format(col)

        ## built index of missings
        if isinstance(data, pd.core.frame.DataFrame):
            null_values = data[col].isnull()
        else:
            null_values = data[col].isnull().compute()

        ## if there are missings, replace them
        if sum(null_values) and col in imputing:
            val = data[col].dropna().head(1)[0]
            if isinstance(val, float):
                try:
                    impfun = IMPUTE_DICT[float][imputing[col]]
                except:
                    raise ValueError("Invalid imputing function for column {}. \
                    Please use one of {}.".
                                     format(col, IMPUTE_DICT[float].keys()))
            else:
                try:
                    impfun = IMPUTE_DICT[str][imputing[col]]
                except:
                    raise ValueError("Invalid imputing function for column {}. \
                    Please use one of {}.".
                                     format(col, IMPUTE_DICT[str].keys()))
            impval = impfun(data[col].dropna())

            if impute_dict:
                imputing_dict[col] = impval

            if isinstance(data, pd.core.frame.DataFrame):
                dct = {str(col): data[col].fillna(impval)}
            else:
                dct = {str(col): data[col].fillna(impval).compute()}
            data = data.drop(col, axis=1).assign(**dct)

    if impute_dict:
        return data, imputing_dict
    else:
        return data


def remove_constants(data=None, percent=0.0, keep=None, ignore_missing=False,
                     removal_dict=True, copy=True, verbose=True):
    """
    Remove columns with no/low variability in a given pandas/dask dataframe.

    Given either a pandas or a dask dataframe, remove_constants will remove
    columns where less or equal 'percent' of the values vary from the mode/mean
    (except they are explicitly specified to be kept), i.e., removing columns
    with no/low variability.

    Args:
        data: A pandas or dask dataframe object.
        percent: Float, the threshold variability. Columns where less or equal
            than 'percent' of the values vary from the mode/mean are removed.
        keep: A single column or a list of column names to keep in any case.
        ignore_missings: Boolean, if 'True' missing values are ignored when
           computing the variability for a given column.
        removal_dict: Boolean, whether to return a list of the columns removed.
        copy: Boolean, whether to copy the data before modifying it.
            Setting this to False currently only works with pandas.
        verbose: Boolean, whether to print status information during the parsing.

    Returns:
        The processed dataframe with columns removed as requested.

    Raises:
        ValueError: If no data is specified. If 'percent' is None or not specified as float.
            If keep is not specified as string or list of strings. If ignore_missings is not
            specified as Boolean.
        NotImplementedError: Setting copy to False when data is a dask dataframe is not supported.
    """
    ## check input
    if data is None or (not isinstance(data, pd.core.frame.DataFrame)
                        and not isinstance(data, dd.core.DataFrame)):
        raise ValueError('Data must be specified as \
        pandas.core.frame.DataFrame or dask.core.DataFrame.')
    if percent is None or not isinstance(percent, float):
        raise ValueError('percent must be specified as float.')
    if keep is not None and not isinstance(keep, list) and not isinstance(keep, str):
        raise ValueError('keep must be specified as str or list.')
    if not isinstance(ignore_missing, bool):
        raise ValueError('ignore_missing must be specified as bool.')

    ## copy object if requested
    if copy:
        if isinstance(data, pd.core.frame.DataFrame):
            data = data.copy()
    else:
        raise NotImplementedError('Dask does currently not support assignments without copying.')

    ## setup
    all_cols = set(data.columns)
    cols_to_check = all_cols - set(keep) if keep is not None else all_cols
    digits = int(math.ceil(np.log10(1/np.finfo(np.float32).eps)))
    cols_dropped = []

    ## check columns (code stolen from mlr/removeConstantFeatures)
    for column in cols_to_check:
        if verbose:
            print 'INFO: Checking column {} ...'.format(column)

        if np.all(data[column].isnull()):
            ratio = 0.0
        else:
            if data.dtypes[0] == 'object':
                col = data[column]
            else:
                col = pd.Series(np.round(data[column], decimals=digits).tolist(), dtype='float')
            mode_val = ut.mode(col)

            if ignore_missing:
                ratio = col.dropna() != mode_val
                ratio = ratio.mean()
            else:
                ratio = col != mode_val
                ratio = ratio.mean()

            if isinstance(ratio, dd.core.Scalar):
                ratio = ratio.compute()

        if ratio <= percent:
            cols_dropped.append(column)
            data = data.drop(column, axis=1)

    if verbose and cols_dropped:
        print 'INFO: Number of columns removed: \
        {} ({}).'.format(len(cols_dropped), ', '.join(cols_dropped))

    if removal_dict:
        dct = {col: True if col in cols_dropped else False for col in all_cols}
        return data, dct
    else:
        return data
