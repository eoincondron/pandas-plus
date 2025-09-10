"""
Pandas-compatible API classes for pandas-plus GroupBy operations.

This module provides familiar pandas-like interfaces that utilize the optimized
pandas-plus GroupBy engine for better performance while maintaining full compatibility.
"""

from typing import Optional, Union, Hashable, Tuple
import pandas as pd
import numpy as np
from functools import wraps

from .core import GroupBy, ArrayType1D
from abc import ABC, abstractmethod


def groupby_aggregation(
    description: str,
    extra_params: str = "",
    include_numeric_only: bool = True,
    include_margins: bool = True,
    **docstring_params,
):
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
        param_docs = """        mask : ArrayType1D, optional
            Boolean mask to apply before aggregation"""

        if include_margins:
            param_docs += """
        margins : bool or list of int, default False
            Add margins (subtotals) to result. If list of integers,
            include margin rows for the specified levels only."""

        if include_numeric_only:
            param_docs = (
                """        numeric_only : bool, default True
            Include only numeric columns
"""
                + param_docs
            )

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
            # Call the core grouper method directly - it already returns
            # proper pandas objects
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

    def __init__(
        self,
        obj: Union[pd.Series, pd.DataFrame],
        by=None,
        level=None,
        grouper: Optional[GroupBy] = None,
    ):
        if by is None and level is None and grouper is None:
            raise ValueError(
                "Must provide either 'by', 'level' or `grouper` for grouping"
            )

        self._obj = obj
        self._by = by
        self._level = level

        if grouper is not None:
            self._grouper = grouper
            return

        # Use specialized method to process by/level arguments
        grouping_keys = self._process_by_argument(by, level)
        self._grouper = GroupBy(grouping_keys)

    @abstractmethod
    def _process_by_argument(self, by, level):
        """
        Process by and level arguments into a list of grouping keys.

        This method should be implemented by subclasses to handle their specific
        column/index naming conventions and data structures.

        Parameters
        ----------
        by : various types
            Grouping key(s), can be column names, arrays, callables, etc.
        level : various types
            Index level name(s) or number(s) for MultiIndex grouping

        Returns
        -------
        list
            List of grouping arrays/keys for GroupBy constructor
        """
        pass

    def _resolve_index_levels(self, level):
        """
        Resolve index level references to actual level values.

        Parameters
        ----------
        level : various types
            Level specification (name, number, list of names/numbers)

        Returns
        -------
        list
            List of index level value arrays
        """
        if not isinstance(level, (list, tuple)):
            levels = [level]
        else:
            levels = level

        return [self._obj.index.get_level_values(level) for level in levels]

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

    def __iter__(self) -> Tuple[Hashable, Union[pd.Series, pd.DataFrame]]:
        for key, indexer in self.groups.items():
            yield key, self._obj.loc[indexer]

    @groupby_aggregation("Compute sum of group values")
    def sum(
        self, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.sum(self._obj, mask=mask, margins=margins)

    @groupby_aggregation("Compute mean of group values")
    def mean(
        self, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.mean(self._obj, mask=mask, margins=margins)

    @groupby_aggregation(
        "Compute standard deviation of group values",
        extra_params="        ddof : int, default 1\n            Degrees of freedom",
    )
    def std(
        self, ddof: int = 1, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.std(self._obj, ddof=ddof, mask=mask, margins=margins)

    @groupby_aggregation(
        "Compute variance of group values",
        extra_params="        ddof : int, default 1\n            Degrees of freedom",
    )
    def var(
        self, ddof: int = 1, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.var(self._obj, ddof=ddof, mask=mask, margins=margins)

    @groupby_aggregation("Compute minimum of group values")
    def min(
        self, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.min(self._obj, mask=mask, margins=margins)

    @groupby_aggregation("Compute maximum of group values")
    def max(
        self, mask: Optional[ArrayType1D] = None, margins: bool = False
    ) -> pd.Series:
        return self._grouper.max(self._obj, mask=mask, margins=margins)

    @groupby_aggregation(
        "Compute count of non-null group values",
        include_numeric_only=False,
        include_margins=False,
    )
    def count(self, mask: Optional[ArrayType1D] = None) -> pd.Series:
        return self._grouper.count(self._obj, mask=mask)

    @groupby_aggregation(
        "Compute group sizes (including null values)",
        include_numeric_only=False,
        include_margins=False,
    )
    def size(self, mask: Optional[ArrayType1D] = None) -> pd.Series:
        return self._grouper.size(mask=mask)

    @groupby_aggregation(
        "Get first non-null value in each group",
        extra_params=(
            "        numeric_only : bool, default False\n"
            "            Include only numeric columns"
        ),
        include_margins=False,
    )
    def first(
        self, numeric_only: bool = False, mask: Optional[ArrayType1D] = None
    ) -> pd.Series:
        return self._grouper.first(self._obj, mask=mask)

    @groupby_aggregation(
        "Get last non-null value in each group",
        extra_params=(
            "        numeric_only : bool, default False\n"
            "            Include only numeric columns"
        ),
        include_margins=False,
    )
    def last(
        self, numeric_only: bool = False, mask: Optional[ArrayType1D] = None
    ) -> pd.Series:
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
    >>> idx = pd.MultiIndex.from_tuples(\n    ...     [('A', 1), ('A', 2), ('B', 1)],\n    ...     names=['letter', 'num'])
    >>> s = pd.Series([10, 20, 30], index=idx)
    >>> gb = SeriesGroupBy(s, level='letter')
    >>> gb.sum()
    A    30
    B    30
    dtype: int64
    """

    def __init__(
        self, obj: pd.Series, by=None, level=None, grouper: Optional[GroupBy] = None
    ):
        if not isinstance(obj, pd.Series):
            raise TypeError("obj must be a pandas Series")
        super().__init__(obj, by=by, level=level, grouper=grouper)

    def _process_by_argument(self, by, level):
        """
        Process by and level arguments for SeriesGroupBy.

        For Series, we handle:
        - by: arrays, lists, callables, or other Series
        - level: index level names or numbers for MultiIndex

        Parameters
        ----------
        by : various types
            Grouping key(s), can be arrays, Series, callables, etc.
        level : various types
            Index level name(s) or number(s) for MultiIndex grouping

        Returns
        -------
        list
            List of grouping arrays/keys for GroupBy constructor
        """
        grouping_keys = []

        # Process by argument first (to match pandas order)
        if by is not None:
            if isinstance(by, (list, tuple)):
                grouping_keys.extend(by)
            else:
                grouping_keys.append(by)

        # Process level argument
        if level is not None:
            level_keys = self._resolve_index_levels(level)
            grouping_keys.extend(level_keys)

        return grouping_keys

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


class BaseGroupByRolling:
    """
    Base class for rolling window operations on GroupBy objects.

    This class provides shared functionality for rolling window calculations
    within each group, reducing code duplication between Series and DataFrame
    rolling operations.

    Parameters
    ----------
    groupby_obj : SeriesGroupBy or DataFrameGroupBy
        The groupby object to apply rolling operations to
    window : int
        Size of the rolling window
    min_periods : int, optional
        Minimum number of observations required to have a value.
        Defaults to window size.
    """

    def __init__(
        self,
        groupby_obj: Union["SeriesGroupBy", "DataFrameGroupBy"],
        window: int,
        min_periods: Optional[int] = None,
    ):
        self._groupby_obj = groupby_obj
        self._window = window
        self._min_periods = min_periods if min_periods is not None else window

    def agg(self, method_name: str, mask: Optional[ArrayType1D] = None):
        """
        Apply a rolling method and return the result.

        Parameters
        ----------
        method_name : str
            Name of the rolling method (e.g., 'sum', 'mean', 'min', 'max')
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation

        Returns
        -------
        pd.Series or pd.DataFrame
            Result of the rolling operation
        """
        method = getattr(self._groupby_obj._grouper, f"rolling_{method_name}")
        return self._format_result(
            method(self._groupby_obj._obj, window=self._window, mask=mask)
        )

    def sum(self, mask: Optional[ArrayType1D] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate rolling sum within each group.

        Parameters
        ----------
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling sum values with same shape as input
        """
        return self.agg("sum", mask=mask)

    def mean(
        self, mask: Optional[ArrayType1D] = None
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate rolling mean within each group.

        Parameters
        ----------
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling mean values with same shape as input
        """
        return self.agg("mean", mask=mask)

    def min(self, mask: Optional[ArrayType1D] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate rolling minimum within each group.

        Parameters
        ----------
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling minimum values with same shape as input
        """
        return self.agg("min", mask=mask)

    def max(self, mask: Optional[ArrayType1D] = None) -> Union[pd.Series, pd.DataFrame]:
        """
        Calculate rolling maximum within each group.

        Parameters
        ----------
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling maximum values with same shape as input
        """
        return self.agg("max", mask=mask)

    def _format_result(self, result) -> Union[pd.Series, pd.DataFrame]:
        """Format the result according to the specific GroupBy type."""
        raise NotImplementedError("Subclasses must implement _format_result")


class SeriesGroupByRolling(BaseGroupByRolling):
    """
    Rolling window operations for SeriesGroupBy objects.

    This class provides rolling window calculations within each group,
    similar to pandas SeriesGroupBy.rolling().
    """

    def _format_result(self, result) -> pd.Series:
        """Format result as a pandas Series."""
        return (
            result
            if isinstance(result, pd.Series)
            else pd.Series(result, name=self._groupby_obj._obj.name)
        )


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

    def __init__(
        self, obj: pd.DataFrame, by=None, level=None, grouper: Optional[GroupBy] = None
    ):
        if not isinstance(obj, pd.DataFrame):
            raise TypeError("obj must be a pandas DataFrame")
        super().__init__(obj, by=by, level=level, grouper=grouper)

    def _process_by_argument(self, by, level):
        """
        Process by and level arguments for DataFrameGroupBy.

        For DataFrame, we handle:
        - by: column names (any hashable type), arrays, Series, callables
        - level: index level names or numbers for MultiIndex
        - Proper resolution of column names including non-string types

        Parameters
        ----------
        by : various types
            Grouping key(s), can be column names, arrays, Series, callables, etc.
        level : various types
            Index level name(s) or number(s) for MultiIndex grouping

        Returns
        -------
        list
            List of grouping arrays/keys for GroupBy constructor
        """
        grouping_keys = []

        # Process by argument first (to match pandas order)
        if by is not None:
            by_keys = self._resolve_by_keys(by)
            grouping_keys.extend(by_keys)

        # Process level argument
        if level is not None:
            level_keys = self._resolve_index_levels(level)
            grouping_keys.extend(level_keys)

        return grouping_keys

    def _resolve_by_keys(self, by):
        """
        Resolve by keys for DataFrame, handling column name resolution.

        Parameters
        ----------
        by : various types
            Single key or list of keys to group by

        Returns
        -------
        list
            List of resolved grouping arrays
        """
        # Special case: if by is a tuple and it's a valid column name,
        # treat as single key
        if isinstance(by, tuple) and by in self._obj.columns:
            by = [by]
        elif not isinstance(by, (list, tuple)):
            by = [by]

        resolved_keys = []

        for key in by:
            # Check for array-like objects first (before checking columns,
            # since arrays aren't hashable)
            if hasattr(key, "__iter__") and not isinstance(key, (str, bytes, tuple)):
                # Array-like object (not string or tuple) - use directly
                if hasattr(key, "__len__") and len(key) != len(self._obj):
                    raise ValueError(
                        f"Length of grouper ({len(key)}) != "
                        f"length of DataFrame ({len(self._obj)})"
                    )
                resolved_keys.append(key)

            elif callable(key):
                # Callable - apply to index
                resolved_keys.append(self._obj.index.map(key))

            else:
                # Try to use as column name (including tuple column names)
                try:
                    if key in self._obj.columns:
                        resolved_keys.append(self._obj[key])
                    elif (
                        hasattr(self._obj.index, "names")
                        and key in self._obj.index.names
                    ):
                        # It's an index level name
                        if isinstance(self._obj.index, pd.MultiIndex):
                            level_idx = self._obj.index.names.index(key)
                            resolved_keys.append(
                                self._obj.index.get_level_values(level_idx)
                            )
                        else:
                            # Single level index
                            resolved_keys.append(self._obj.index)
                    else:
                        raise KeyError(f"Column or index level '{key}' not found")
                except TypeError:
                    # Unhashable type - treat as array-like if it has proper length
                    if hasattr(key, "__len__") and len(key) == len(self._obj):
                        resolved_keys.append(key)
                    else:
                        raise KeyError(f"Invalid grouping key: {key}")

        return resolved_keys

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            return super().__getattribute__(name)

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


class DataFrameGroupByRolling(BaseGroupByRolling):
    """
    Rolling window operations for DataFrameGroupBy objects.

    This class provides rolling window calculations within each group,
    similar to pandas DataFrameGroupBy.rolling(). It inherits from
    BaseGroupByRolling to reduce code duplication.
    """

    def _format_result(self, result) -> pd.DataFrame:
        """Format result as a pandas DataFrame."""
        return result if isinstance(result, pd.DataFrame) else pd.DataFrame(result)
