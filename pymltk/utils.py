## imports
from scipy import stats



## helper functions
def mode(data=None):
    """
    Compute mode of given numpy array or pandas series.

    Mode is just a wrapper around scipy.stats.mode which returns
    the mode of a given numpy array or pandas series. Missing values
    are omitted before the mode is computed.

    Args:
        data: A numpy array or pandas series.

    Returns:
        The mode of x as scalar value.

    Raises:
        ValueError: If no data is specified or if all values are missing.
    """
    if data is None:
        raise ValueError('No data specified.')
    if not len(data.dropna()):
        raise ValueError('No valid data specified.')
    mode_val = stats.mode(data, nan_policy='omit')[0]
    return mode_val[0]
