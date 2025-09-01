"""
Monkey-patch functionality to add groupby_fast methods to pandas Series and DataFrame.

This module provides runtime patching to add our optimized GroupBy implementations
as methods on pandas objects, allowing seamless usage like df.groupby_fast('column').sum().
"""

import pandas as pd
from typing import Optional, Union, List, Any
from .api import SeriesGroupBy, DataFrameGroupBy


def groupby_fast_series(
    self: pd.Series, by=None, level=None, **kwargs
) -> SeriesGroupBy:
    """
    Group Series using optimized pandas-plus GroupBy implementation.

    This method provides the same interface as pandas Series.groupby() but uses
    the pandas-plus optimized GroupBy engine for better performance.

    Parameters
    ----------
    by : array-like, optional
        Grouping key(s). Can be any type acceptable to pandas-plus GroupBy constructor.
        If None, must specify level.
    level : int, str, or sequence, optional
        If the Series has a MultiIndex, group by specific level(s) of the index.
        Can be level number(s) or name(s). If None, must specify by.
    **kwargs
        Additional keyword arguments (for pandas compatibility, mostly ignored)

    Returns
    -------
    SeriesGroupBy
        pandas-plus SeriesGroupBy object with optimized performance

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas_plus.groupby import install_groupby_fast
    >>> install_groupby_fast()  # Add groupby_fast methods
    >>>
    >>> s = pd.Series([1, 2, 3, 4, 5, 6])
    >>> groups = pd.Series(['A', 'B', 'A', 'B', 'A', 'B'])
    >>> result = s.groupby_fast(groups).sum()
    >>> print(result)
    A    9
    B   12
    dtype: int64
    """
    return SeriesGroupBy(self, by=by, level=level)


def groupby_fast_dataframe(
    self: pd.DataFrame, by=None, level=None, **kwargs
) -> DataFrameGroupBy:
    """
    Group DataFrame using optimized pandas-plus GroupBy implementation.

    This method provides the same interface as pandas DataFrame.groupby() but uses
    the pandas-plus optimized GroupBy engine for better performance.

    Parameters
    ----------
    by : array-like, optional
        Grouping key(s). Can be any type acceptable to pandas-plus GroupBy constructor.
        If None, must specify level.
    level : int, str, or sequence, optional
        If the DataFrame has a MultiIndex, group by specific level(s) of the index.
        Can be level number(s) or name(s). If None, must specify by.
    **kwargs
        Additional keyword arguments (for pandas compatibility, mostly ignored)

    Returns
    -------
    DataFrameGroupBy
        pandas-plus DataFrameGroupBy object with optimized performance

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas_plus.groupby import install_groupby_fast
    >>> install_groupby_fast()  # Add groupby_fast methods
    >>>
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
    >>> groups = ['X', 'Y', 'X', 'Y']
    >>> result = df.groupby_fast(groups).sum()
    >>> print(result)
       A   B
    X  4  40
    Y  6  60
    """
    return DataFrameGroupBy(self, by=by, level=level)


def install_groupby_fast():
    """
    Install groupby_fast methods on pandas Series and DataFrame classes.

    This function monkey-patches pandas to add optimized groupby_fast methods
    that use the pandas-plus GroupBy implementation. After calling this function,
    all Series and DataFrame objects will have a .groupby_fast() method available.

    Examples
    --------
    >>> import pandas as pd
    >>> from pandas_plus.groupby import install_groupby_fast
    >>>
    >>> # Install the fast groupby methods
    >>> install_groupby_fast()
    >>>
    >>> # Now all pandas objects have groupby_fast
    >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
    >>> gb = df.groupby_fast(['X', 'Y', 'X'])  # Uses pandas-plus implementation
    >>> result = gb.sum()
    """
    # Add methods to the classes
    pd.Series.groupby_fast = groupby_fast_series
    pd.DataFrame.groupby_fast = groupby_fast_dataframe

    print("✅ pandas-plus groupby_fast methods installed!")
    print(
        "   Use df.groupby_fast() and series.groupby_fast() for optimized performance"
    )


def uninstall_groupby_fast():
    """
    Remove groupby_fast methods from pandas Series and DataFrame classes.

    This function removes the monkey-patched methods, restoring pandas to its
    original state. Useful for cleanup or testing.
    """
    if hasattr(pd.Series, "groupby_fast"):
        delattr(pd.Series, "groupby_fast")
    if hasattr(pd.DataFrame, "groupby_fast"):
        delattr(pd.DataFrame, "groupby_fast")

    print("✅ pandas-plus groupby_fast methods removed")


def is_groupby_fast_installed() -> bool:
    """
    Check if groupby_fast methods are currently installed.

    Returns
    -------
    bool
        True if groupby_fast methods are available on pandas objects
    """
    return hasattr(pd.Series, "groupby_fast") and hasattr(pd.DataFrame, "groupby_fast")


# Optional: Auto-install when module is imported
# Uncomment the next line if you want automatic installation
# install_groupby_fast()
