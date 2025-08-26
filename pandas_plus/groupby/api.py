"""
Pandas-compatible API classes for pandas-plus GroupBy operations.

This module provides familiar pandas-like interfaces that utilize the optimized
pandas-plus GroupBy engine for better performance while maintaining full compatibility.
"""

from typing import Optional, Union, List
import pandas as pd
from functools import wraps

from .core import GroupBy, ArrayCollection, ArrayType1D
from abc import ABC, abstractmethod


def groupby_aggregation(description: str, extra_params: str = "", include_numeric_only: bool = True, include_margins: bool = True, **docstring_params):
    """
    Decorator for SeriesGroupBy/DataFrameGroupBy aggregation methods.
    
    This decorator:
    1. Eliminates boilerplate return value processing 
    2. Auto-generates consistent docstrings
    3. Handles mask parameter consistently
    
    Parameters
    ----------
    description : str
        Brief description of what the method does (e.g., "Compute sum of group values")
    extra_params : str, optional
        Additional parameter documentation to include
    **docstring_params : dict
        Additional parameters for docstring template
    """
    def decorator(func):
        method_name = func.__name__
        
        # Generate docstring
        param_docs = f"""        mask : ArrayType1D, optional
            Boolean mask to apply before aggregation"""
        
        if include_margins:
            param_docs += f"""
        margins : bool, default False
            Add margins (subtotals) to result"""
        
        if include_numeric_only:
            param_docs = f"""        numeric_only : bool, default True
            Include only numeric columns
""" + param_docs
        
        if extra_params:
            param_docs = extra_params + "\n" + param_docs
            
        func.__doc__ = f"""
        {description}.
        
        Parameters
        ----------{param_docs}
            
        Returns
        -------
        pd.Series
            Series with group {method_name}s
        """
        
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Call the core grouper method directly - it already returns proper pandas objects
            return func(self, *args, **kwargs)
            
        return wrapper
    return decorator


def groupby_cumulative(description: str):
    """Decorator for cumulative operations."""
    def decorator(func):
        method_name = func.__name__
        
        func.__doc__ = f"""
        {description} for each group.
        
        Returns
        -------
        pd.Series
            Series with {method_name} values
        """
        
        @wraps(func)
        def wrapper(self):
            # Call the core grouper method directly
            return func(self)
            
        return wrapper
    return decorator


class BaseGroupBy(ABC):
    """
    Abstract base class for pandas-plus GroupBy API classes.
    
    This class contains common functionality shared between SeriesGroupBy
    and DataFrameGroupBy classes.
    """

    def __init__(self, obj: Union[pd.Series, pd.DataFrame], by=None, level=None, grouper: Optional[GroupBy] = None):
        if by is None and level is None and grouper is None :
            raise ValueError("Must provide either 'by', 'level' or `grouper` for grouping")

        self._obj = obj
        self._by = by
        self._level = level

        if grouper is not None:
            self._grouper = grouper
            return

        # Build the grouping keys (by first to match pandas order)
        grouping_keys = []

        # Handle by-based grouping first
        if by is not None:
            if isinstance(by, (list, tuple)):
                grouping_keys.extend(by)
            else:
                grouping_keys.append(by)

        # Handle level-based grouping
        if level is not None:
            if isinstance(level, (list, tuple)):
                # Multiple levels
                for lvl in level:
                    if isinstance(lvl, str):
                        # Level name
                        level_idx = obj.index.names.index(lvl)
                    else:
                        # Level number
                        level_idx = lvl
                    grouping_keys.append(obj.index.get_level_values(level_idx))
            else:
                # Single level
                if isinstance(level, str):
                    # Level name
                    level_idx = obj.index.names.index(level)
                else:
                    # Level number
                    level_idx = level
                grouping_keys.append(obj.index.get_level_values(level_idx))

        self._grouper = GroupBy(grouping_keys)

    @property
    def grouper(self) -> GroupBy:
        """Access to the underlying GroupBy engine."""
        return self._grouper

    @property
    def groups(self):
        """Dict mapping group names to row labels."""
        return self._grouper.groups

    @property 
    def ngroups(self) -> int:
        """Number of groups."""
        return self._grouper.ngroups

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(ngroups={self.ngroups})"

    @groupby_aggregation("Compute sum of group values")
    def sum(self, mask: Optional[ArrayType1D] = None, margins: bool = False) -> pd.Series:
        return self._grouper.sum(self._obj, mask=mask, margins=margins)

    @groupby_aggregation("Compute mean of group values")
    def mean(self, mask: Optional[ArrayType1D] = None, margins: bool = False) -> pd.Series:
        return self._grouper.mean(self._obj, mask=mask, margins=margins)

    @groupby_aggregation(
        "Compute standard deviation of group values",
        extra_params="        ddof : int, default 1\n            Degrees of freedom",
    )
    def std(self, ddof: int = 1, mask: Optional[ArrayType1D] = None, margins: bool = False) -> pd.Series:
        return self._grouper.std(self._obj, ddof=ddof, mask=mask, margins=margins)

    @groupby_aggregation(
        "Compute variance of group values",
        extra_params="        ddof : int, default 1\n            Degrees of freedom",
    )
    def var(self, ddof: int = 1, mask: Optional[ArrayType1D] = None, margins: bool = False) -> pd.Series:
        return self._grouper.var(self._obj, ddof=ddof, mask=mask, margins=margins)

    @groupby_aggregation("Compute minimum of group values")
    def min(self, mask: Optional[ArrayType1D] = None, margins: bool = False) -> pd.Series:
        return self._grouper.min(self._obj, mask=mask, margins=margins)

    @groupby_aggregation("Compute maximum of group values")
    def max(self, mask: Optional[ArrayType1D] = None, margins: bool = False) -> pd.Series:
        return self._grouper.max(self._obj, mask=mask, margins=margins)

    @groupby_aggregation(
        "Compute count of non-null group values", include_numeric_only=False, include_margins=False
    )
    def count(self, mask: Optional[ArrayType1D] = None) -> pd.Series:
        return self._grouper.count(self._obj, mask=mask)

    @groupby_aggregation(
        "Compute group sizes (including null values)", include_numeric_only=False, include_margins=False
    )
    def size(self, mask: Optional[ArrayType1D] = None) -> pd.Series:
        return self._grouper.size(self._obj, mask=mask)

    @groupby_aggregation(
        "Get first non-null value in each group",
        extra_params="        numeric_only : bool, default False\n            Include only numeric columns",
        include_margins=False
    )
    def first(self, numeric_only: bool = False, mask: Optional[ArrayType1D] = None) -> pd.Series:
        return self._grouper.first(self._obj, mask=mask)

    @groupby_aggregation(
        "Get last non-null value in each group",
        extra_params="        numeric_only : bool, default False\n            Include only numeric columns",
        include_margins=False
    )
    def last(self, numeric_only: bool = False, mask: Optional[ArrayType1D] = None) -> pd.Series:
        return self._grouper.last(self._obj, mask=mask)

    def nth(self, n: int) -> pd.Series:
        """
        Take nth value from each group.

        Parameters
        ----------
        n : int
            Position to take (0-indexed)

        Returns
        -------
        pd.Series
            Series with nth values
        """
        result = self._grouper.nth(self._obj, n)
        return (
            result
            if isinstance(result, pd.Series)
            else pd.Series(result, name=self._obj.name)
        )

    def head(self, n: int = 5) -> pd.Series:
        """
        Return first n rows of each group.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return

        Returns
        -------
        pd.Series
            Series with first n values from each group
        """
        result = self._grouper.head(self._obj, n)
        return (
            result
            if isinstance(result, pd.Series)
            else pd.Series(result, name=self._obj.name)
        )

    def tail(self, n: int = 5) -> pd.Series:
        """
        Return last n rows of each group.

        Parameters
        ----------
        n : int, default 5
            Number of rows to return

        Returns
        -------
        pd.Series
            Series with last n values from each group
        """
        result = self._grouper.tail(self._obj, n)
        return (
            result
            if isinstance(result, pd.Series)
            else pd.Series(result, name=self._obj.name)
        )

    def agg(self, func, mask: Optional[ArrayType1D] = None) -> pd.Series:
        """
        Apply aggregation function to each group.

        Parameters
        ----------
        func : str or callable
            Aggregation function name or callable

        Returns
        -------
        pd.Series
            Series with aggregated values
        """
        if isinstance(func, str):
            if hasattr(self, func):
                return getattr(self, func)()
            else:
                result = self._grouper.agg(self._obj, func)
        else:
            result = self._grouper.apply(self._obj, func)

        return result

    aggregate = agg  # Alias

    def apply(self, func) -> pd.Series:
        """
        Apply function to each group and combine results.

        Parameters
        ----------
        func : callable
            Function to apply to each group

        Returns
        -------
        pd.Series
            Series with function results
        """
        result = self._grouper.apply(self._obj, func)
        return (
            result
            if isinstance(result, pd.Series)
            else pd.Series(result, name=self._obj.name)
        )

    @groupby_cumulative("Cumulative sum")
    def cumsum(self) -> pd.Series:
        return self._grouper.cumsum(self._obj)

    @groupby_cumulative("Cumulative maximum")
    def cummax(self) -> pd.Series:
        return self._grouper.cummax(self._obj)

    @groupby_cumulative("Cumulative minimum")
    def cummin(self) -> pd.Series:
        return self._grouper.cummin(self._obj)

    @groupby_cumulative(
        "Number each item in each group from 0 to the length of that group - 1"
    )
    def cumcount(self) -> pd.Series:
        return self._grouper.cumcount(self._obj)


class SeriesGroupBy(BaseGroupBy):
    """
    A pandas-like SeriesGroupBy class that uses pandas-plus GroupBy as the engine.
    
    This class provides a familiar pandas interface while leveraging the optimized
    GroupBy implementation for better performance.
    
    Parameters
    ----------
    obj : pd.Series
        The pandas Series to group
    by : array-like, optional
        Grouping key(s), can be any type acceptable to core.GroupBy constructor.
        If None, must specify level.
    level : int, str, or sequence, optional
        If the Series has a MultiIndex, group by specific level(s) of the index.
        Can be level number(s) or name(s). If None, must specify by.
    
    Examples
    --------
    Basic grouping:
    >>> import pandas as pd
    >>> from pandas_plus.groupby import SeriesGroupBy
    >>> s = pd.Series([1, 2, 3, 4, 5, 6])
    >>> groups = pd.Series(['A', 'B', 'A', 'B', 'A', 'B'])
    >>> gb = SeriesGroupBy(s, by=groups)
    >>> gb.sum()
    A    9
    B   12
    dtype: int64
    
    Level-based grouping:
    >>> idx = pd.MultiIndex.from_tuples([('A', 1), ('A', 2), ('B', 1)], names=['letter', 'num'])
    >>> s = pd.Series([10, 20, 30], index=idx)
    >>> gb = SeriesGroupBy(s, level='letter')
    >>> gb.sum()
    A    30
    B    30
    dtype: int64
    """
    
    def __init__(self, obj: pd.Series, by=None, level=None, grouper: Optional[GroupBy] = None):
        if not isinstance(obj, pd.Series):
            raise TypeError("obj must be a pandas Series")
        super().__init__(obj, by=by, level=level, grouper=grouper)
        
    def rolling(self, window: int, min_periods: Optional[int] = None):
        """
        Provide rolling window calculations within groups.
        
        Parameters
        ----------
        window : int
            Size of the moving window
        min_periods : int, optional
            Minimum number of observations required to have a value
            
        Returns
        -------
        SeriesGroupByRolling
            Rolling window object
        """
        return SeriesGroupByRolling(self, window, min_periods)


class SeriesGroupByRolling:
    """
    Rolling window operations for SeriesGroupBy objects.
    
    This class provides rolling window calculations within each group,
    similar to pandas SeriesGroupBy.rolling().
    """
    
    def __init__(self, groupby_obj: SeriesGroupBy, window: int, min_periods: Optional[int] = None):
        self._groupby_obj = groupby_obj
        self._window = window
        self._min_periods = min_periods if min_periods is not None else window
        
    def sum(self) -> pd.Series:
        """Rolling sum within each group."""
        result = self._groupby_obj._grouper.rolling_sum(
            self._groupby_obj._obj, 
            window=self._window
        )
        return result if isinstance(result, pd.Series) else pd.Series(result, name=self._groupby_obj._obj.name)
        
    def mean(self) -> pd.Series:
        """Rolling mean within each group."""
        result = self._groupby_obj._grouper.rolling_mean(
            self._groupby_obj._obj, 
            window=self._window
        )
        return result if isinstance(result, pd.Series) else pd.Series(result, name=self._groupby_obj._obj.name)
        
    def min(self) -> pd.Series:
        """Rolling minimum within each group."""
        result = self._groupby_obj._grouper.rolling_min(
            self._groupby_obj._obj, 
            window=self._window
        )
        return result if isinstance(result, pd.Series) else pd.Series(result, name=self._groupby_obj._obj.name)
        
    def max(self) -> pd.Series:
        """Rolling maximum within each group."""
        result = self._groupby_obj._grouper.rolling_max(
            self._groupby_obj._obj, 
            window=self._window
        )
        return result if isinstance(result, pd.Series) else pd.Series(result, name=self._groupby_obj._obj.name)


class DataFrameGroupBy(BaseGroupBy):
    """
    A pandas-like DataFrameGroupBy class that uses pandas-plus GroupBy as the engine.
    
    This class provides a familiar pandas interface for DataFrame grouping operations
    while leveraging the optimized GroupBy implementation for better performance.
    
    Parameters
    ----------
    obj : pd.DataFrame
        The pandas DataFrame to group
    by : array-like, optional
        Grouping key(s), can be any type acceptable to core.GroupBy constructor.
        If None, must specify level.
    level : int, str, or sequence, optional
        If the DataFrame has a MultiIndex, group by specific level(s) of the index.
        Can be level number(s) or name(s). If None, must specify by.
    
    Examples
    --------
    Basic grouping:
    >>> import pandas as pd
    >>> from pandas_plus.groupby import DataFrameGroupBy
    >>> df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [10, 20, 30, 40]})
    >>> groups = pd.Series(['X', 'Y', 'X', 'Y'])
    >>> gb = DataFrameGroupBy(df, by=groups)
    >>> gb.sum()
        A   B
    X   4  40
    Y   6  60
    """

    def __init__(self, obj: pd.DataFrame, by=None, level=None, grouper: Optional[GroupBy] = None):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("obj must be a pandas DataFrame")
        super().__init__(obj, by=by, level=level, grouper=grouper)

    def __getitem__(self, key):
        """
        Select column(s) from the grouped DataFrame.
        
        Parameters
        ----------
        key : str or list
            Column name(s) to select
            
        Returns
        -------
        SeriesGroupBy or DataFrameGroupBy
            SeriesGroupBy if single column, DataFrameGroupBy if multiple columns
        """
        subset = self._obj[key]
        if isinstance(subset, pd.Series):
            # Single column - return SeriesGroupBy
            return SeriesGroupBy(subset, grouper=self._grouper)
        else:
            # Multiple columns - return DataFrameGroupBy with subset
            return DataFrameGroupBy(subset, grouper=self._grouper)
    
    def rolling(self, window: int, min_periods: Optional[int] = None):
        """
        Provide rolling window calculations within groups.
        
        Parameters
        ----------
        window : int
            Size of the moving window
        min_periods : int, optional
            Minimum number of observations required to have a value
            
        Returns
        -------
        DataFrameGroupByRolling
            Rolling window object
        """
        return DataFrameGroupByRolling(self, window, min_periods)


class DataFrameGroupByRolling:
    """
    Rolling window operations for DataFrameGroupBy objects.
    
    This class provides rolling window calculations within each group,
    similar to pandas DataFrameGroupBy.rolling().
    """
    
    def __init__(self, groupby_obj: DataFrameGroupBy, window: int, min_periods: Optional[int] = None):
        self._groupby_obj = groupby_obj
        self._window = window
        self._min_periods = min_periods if min_periods is not None else window
        
    def sum(self) -> pd.DataFrame:
        """Rolling sum within each group."""
        result = self._groupby_obj._grouper.rolling_sum(
            self._groupby_obj._obj, 
            window=self._window
        )
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        
    def mean(self) -> pd.DataFrame:
        """Rolling mean within each group."""
        result = self._groupby_obj._grouper.rolling_mean(
            self._groupby_obj._obj, 
            window=self._window
        )
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        
    def min(self) -> pd.DataFrame:
        """Rolling minimum within each group."""
        result = self._groupby_obj._grouper.rolling_min(
            self._groupby_obj._obj, 
            window=self._window
        )
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
        
    def max(self) -> pd.DataFrame:
        """Rolling maximum within each group."""
        result = self._groupby_obj._grouper.rolling_max(
            self._groupby_obj._obj, 
            window=self._window
        )
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
