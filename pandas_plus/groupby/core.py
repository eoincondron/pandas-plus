from collections.abc import Mapping, Sequence
from functools import cached_property, wraps
from typing import Callable, List, Optional
from inspect import signature

import numpy as np
import pandas as pd
import polars as pl

from . import numba as numba_funcs
from ..util import (ArrayType1D, ArrayType2D, TempName, factorize_1d, factorize_2d,
                   convert_data_to_arr_list_and_keys, get_array_name)

ArrayCollection = (
    ArrayType1D | ArrayType2D | Sequence[ArrayType1D] | Mapping[str, ArrayType1D]
)


def array_to_series(arr: ArrayType1D):
    """
    Convert various array types to pandas Series.
    
    Parameters
    ----------
    arr : ArrayType1D
        Input array to convert (numpy array, pandas Series, polars Series, etc.)
        
    Returns
    -------
    pd.Series
        Pandas Series representation of the input array
    """
    if isinstance(arr, pl.Series):
        return arr.to_pandas()
    else:
        return pd.Series(arr)


def val_to_numpy(val: ArrayType1D):
    """
    Convert various array types to numpy array.
    
    Parameters
    ----------
    val : ArrayType1D
        Input array to convert (numpy array, pandas Series, polars Series, etc.)
        
    Returns
    -------
    np.ndarray
        NumPy array representation of the input
    """
    try:
        return val.to_numpy() # type: ignore
    except AttributeError:
        return np.asarray(val)
    

def _get_indexes_from_values(values: ArrayCollection) -> List[pd.Index]:
    """
    Extract pandas Index objects from the provided values.
    
    Parameters
    ----------
    values : ArrayCollection
        Collection of arrays or Series to extract indexes from
        
    Returns
    -------
    List[pd.Index]
        List of pandas Index objects corresponding to each value in the collection
    """
    if isinstance(values, (pd.Series, pd.DataFrame)):
        out = [values.index]
    elif isinstance(values, pl.Series):
        out = []
    elif isinstance(values, (list, tuple)):
        out = [getattr(v, 'index', None) for v in values]
    elif isinstance(values, Mapping):
        return _get_indexes_from_values(list(values.values()))
    elif isinstance(values, np.ndarray):
        out = []
    else:
        return []
    return [index for index in out if index is not None]


def _validate_input_indexes(indexes: List[pd.Index]) -> Optional[pd.Index]:
    """
    Validate that all provided pandas indexes are compatible.
    
    Parameters
    ----------
    indexes : list
        List of pandas Index objects to validate
        
    Returns
    -------
    pd.Index or None
        Returns the first non-trivial index if any exists, otherwise None
        
    Raises
    ------
    ValueError
        If indexes have different lengths or non-trivial indexes don't match
    """
    lengths = set(map(len, indexes))
    if len(lengths) > 1:
        raise ValueError(f"found more than one unique length: {lengths}")
    non_trivial = [
        index
        for index in indexes
        if not (
            isinstance(index, pd.RangeIndex) and index.start == 0 and index.step == 1
        )
    ]
    if len(non_trivial) == 0:
        return

    for left, right in zip(non_trivial, non_trivial[1:]):
        if not left.equals(right):
            raise ValueError

    return non_trivial[0]


def groupby_method(method):

    @wraps(method)
    def wrapper(
        *args, **kwargs
    ):
        bound_args = signature(method).bind(*args, **kwargs)
        group_key = bound_args.arguments['self']
        if not isinstance(group_key, GroupBy):
            bound_args.arguments['self'] = GroupBy(group_key)
        return method(**bound_args.arguments)

    if method.__doc__ is None:
        __doc__ = f"""
        Calculate the group-wise {method.__name__} of the given values over the groups defined by `key`
        
        Parameters
        ----------
        key: An array/Series or a container of same, such as dict, list or DataFrame
            Defines the groups. May be a single dimension like an array or Series, 
            or multi-dimensional like a list/dict of 1-D arrays or 2-D array/DataFrame. 
        values: An array/Series or a container of same, such as dict, list or DataFrame
            The values to be aggregated. May be a single dimension like an array or Series, 
            or multi-dimensional like a list/dict of 1-D arrays or 2-D array/DataFrame. 
        mask: array/Series
            Optional Boolean array which filters elements out of the calculations
            
        Returns
        -------
        pd.Series / pd.DataFrame
        
        The result of the group-by calculation. 
        A Series is returned when `values` is a single array/Series, otherwise a DataFrame. 
        The index of the result has one level per array/column in the group key. 
            
        """
        wrapper.__doc__ = __doc__

    return wrapper


class GroupBy:
    """
    Class for performing group-by operations on arrays.
    
    This class provides methods for aggregating values by group using various
    functions like sum, mean, min, max, etc. It supports multiple group keys
    and various input formats including NumPy arrays, pandas Series/DataFrames,
    and polars Series/DataFrames.
    
    Parameters
    ----------
    group_keys : ArrayCollection
        The keys to group by. Can be a single array-like object or a collection of them.
    """

    def __init__(self, group_keys: ArrayCollection):
        group_key_list, group_key_names = convert_data_to_arr_list_and_keys(group_keys)
        indexes = _get_indexes_from_values(group_key_list)
        common_index = _validate_input_indexes(indexes)
        
        if len(group_key_list) == 1:
            self._group_ikey, self._result_index = factorize_1d(group_key_list[0])
        else:
            self._group_ikey, self._result_index = factorize_2d(*group_key_list)

        self._result_index.names = [
            None if isinstance(key, TempName) else key for key in group_key_names
        ]

    @property
    def ngroups(self):
        """
        Number of groups.
        
        Returns
        -------
        int
            Number of distinct groups
        """
        return len(self.result_index)

    @property
    def result_index(self):
        """
        Index for the result of group-by operations.
        
        Returns
        -------
        pd.Index
            Index with one level per group key
        """
        return self._result_index

    @property
    def group_ikey(self):
        """
        Integer key for each original row identifying its group.
        
        Returns
        -------
        ndarray
            Array of group indices for each original row
        """
        return self._group_ikey
    
    @cached_property
    def has_null_keys(self) -> bool:
        """
        Check if the group keys contain any null values.
        
        Returns
        -------
        bool
            True if any group key contains null values, False otherwise
        """
        return self.group_ikey.min() < 0

    def _apply_gb_func(
        self,
        func_name: str,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
    ):
        """
        Apply a group-by function to values.
        
        Parameters
        ----------
        func : Callable
            Function to apply to each group
        values : ArrayCollection
            Values to aggregate
        mask : ArrayType1D, optional
            Boolean mask to filter values
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group
        margins : bool, default False
            If True, include a total row in the result
            
        Returns
        -------
        pd.Series or pd.DataFrame
            Results of the groupby operation
        """
        value_list, value_names = convert_data_to_arr_list_and_keys(values)
        if len(set(value_names)) != len(value_names):
            raise ValueError(
                "Values must have unique names. "
                f"Found duplicates: {set(value_names)}"
            )

        np_values = list(map(val_to_numpy, value_list))

        indexes = _get_indexes_from_values(value_list)
        common_index = _validate_input_indexes(indexes)

        func = getattr(numba_funcs, f"group_{func_name}")

        results = map(
            lambda v: func(
                group_key=self.group_ikey, values=v, mask=mask, ngroups=self.ngroups + 1
            ),
            np_values,
        )
        out_dict = {}
        for key, result in zip(value_names, results):
            if transform:
                result = out_dict[key] = pd.Series(
                    result[self.group_ikey], common_index
                )
            else:
                result = out_dict[key] = pd.Series(result[:-1], self.result_index)

            return_1d = len(value_list) == 1 and isinstance(values, ArrayType1D)
            if return_1d:
                out = result
                if get_array_name(values) is None:
                    out.name = None
            else:
                out = pd.DataFrame(out_dict)

        if not transform and mask is not None:
            observed = self.sum(values=mask)
            out = out.loc[observed > 0]

        if margins:
            out = add_row_margin(out, agg_func="sum" if func in ("size", "count") else func_name)
        return out

    @groupby_method
    def size(
        self, mask: Optional[ArrayType1D] = None, transform: bool = False, margins: bool = False
    ):
        out = numba_funcs.group_size(group_key=self.group_ikey, mask=mask, ngroups=self.ngroups + 1)
        if transform:
            return out[self.group_ikey]
        
        out = pd.Series(out[:-1], index=self.result_index, name="size")

        if mask is not None:
            observed = self.sum(values=mask)
            out = out.loc[observed > 0]

        if margins:
            out = add_row_margin(out, agg_func="sum")

        return out

    @groupby_method
    def count(
        self, values: ArrayCollection, mask: Optional[ArrayType1D] = None, transform: bool = False, margins: bool = False
    ):
        return self._apply_gb_func(
            "count", values=values, mask=mask, transform=transform, margins=margins
        )
    
    @groupby_method
    def sum(
        self, values: ArrayCollection, mask: Optional[ArrayType1D] = None, transform: bool = False, margins: bool = False
    ):
        return self._apply_gb_func(
            "sum", values=values, mask=mask, transform=transform, margins=margins
        )

    @groupby_method
    def mean(
        self, values: ArrayCollection, mask: Optional[ArrayType1D] = None, transform: bool = False, margins: bool = False
    ):
        kwargs = dict(values=values, mask=mask, transform=transform, margins=margins)
        sum_, count = self.sum(**kwargs), self.count(**kwargs)
        if sum_.ndim == 2:
            timestamp_cols = [col for col, d in sum_.dtypes.items() if d.kind in "mM"]
            tmp_types = {col: "int64" for col in timestamp_cols}
            return sum_.astype(tmp_types, copy=False).div(count).astype(sum_.dtypes[timestamp_cols], copy=False)
        elif sum_.dtype.kind in "mM":
            return (sum_.astype("int64") // count).astype(sum_.dtype)
        else:
            return sum_ / count
    
    @groupby_method
    def min(
        self, values: ArrayCollection, mask: Optional[ArrayType1D] = None, transform: bool = False, margins: bool = False
    ):
        return self._apply_gb_func(
            "min", values=values, mask=mask, transform=transform, margins=margins
        )

    @groupby_method
    def max(
        self, values: ArrayCollection, mask: Optional[ArrayType1D] = None, transform: bool = False, margins: bool = False
    ):
        return self._apply_gb_func(
            "max", values=values, mask=mask, transform=transform, margins=margins
        )

    @groupby_method
    def agg(
        self,
        values: ArrayCollection,
        agg_func: Callable | str | List[str],
        mask: Optional[ArrayType1D] = None,
        transform: bool = False, 
        margins: bool = False,
    ):
        if np.ndim(agg_func) == 0:
            if isinstance(agg_func, Callable):
                agg_func = agg_func.__name__
            func = getattr(self, agg_func)
            return func(values, mask=mask, transform=transform, margins=margins)
        elif np.ndim(agg_func) == 1:
            if isinstance(values, ArrayType1D):
                values = dict.fromkeys(agg_func, values)
            value_list, value_names = convert_data_to_arr_list_and_keys(values)
            if len(agg_func) != len(value_list):
                raise ValueError(
                    f"Mismatch between number of agg funcs ({len(agg_func)}) "
                    f"and number of values ({len(values)})"
                )
            return pd.DataFrame(
                {k:
                    self.agg(v, agg_func=f, mask=mask, transform=transform, margins=margins)
                    for f, (k, v) in zip(agg_func, zip(value_names, value_list))
                 },
            )
        else:
            raise TypeError("agg_func must by a single function name or an iterable of same")

    @groupby_method
    def ratio(
        self,
        values1: ArrayCollection,
        values2: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        agg_func="sum",
    ):
        # check for nullity
        self.agg(values1, agg_func, mask) / self.agg(values2, agg_func, mask)

    @groupby_method
    def subset_ratio(
            self,
            values: ArrayCollection,
            subset_mask: ArrayType1D,
            global_mask: Optional[ArrayType1D] = None,
            agg_func="sum",
    ):
        # check for nullity
        self.agg(values, agg_func, subset_mask & global_mask) / self.agg(values, agg_func, global_mask)

    @groupby_method
    def group_nearby_members(self, values: ArrayType1D, max_diff: int | float):
        """
        Generate subgroups of the groups defined by the GroupBy where the differences between consecutive members of a group are below a threshold.
        For example, group events which are close in time and which belong to the same group defined by the group key.

        self: GroupBy | ArrayType1D
            Vector defining the initial groups
        values:
            Array of numerical values used to determine closeness of the group members, e.g. an array of timestamps.
            Assumed to be monotonic non-decreasing.
        max_diff: float | int
            The threshold distance for forming a new sub-group
        """
        return group_nearby_members(
            group_key=self.group_ikey,
            values=values,
            max_diff=max_diff,
            n_groups=self.ngroups
        )
    

def pivot(
    index: ArrayCollection,
    columns: ArrayCollection,
    values: ArrayCollection,
    agg_func: str = "sum",
    mask: Optional[ArrayType1D] = None,
    margins: bool = False,
):
    """
    Perform a cross-tabulation of the group keys and values.
    
    Parameters
    ----------
    index : ArrayCollection
        Group keys to use as index in the resulting DataFrame
    columns : ArrayCollection
        Group keys to use as columns in the resulting DataFrame
    values : ArrayCollection
        Values to cross-tabulate against the group keys
    agg_func : str, default "sum"
        Aggregation function to apply to the values. Can be a string like "sum", "mean", "min", "max", etc.
    mask : Optional[ArrayType1D], default None
        Boolean mask to filter values before cross-tabulation
    margin : bool, default False
        If True, adds a total row and column to the resulting DataFrame
    Returns
    -------
    pd.DataFrame
        Cross-tabulated DataFrame with group keys as index and values as columns
    """
    if agg_func == 'mean':
        kwargs = locals().copy()
        del kwargs['agg_func']
        return pivot(**kwargs, agg_func="sum") / pivot(**kwargs, agg_func='size')
    
    index, index_names = convert_data_to_arr_list_and_keys(index)
    columns, index_columns = convert_data_to_arr_list_and_keys(columns)
    
    grouper = GroupBy(index + columns)
    if agg_func == "size":
        out = grouper.size(mask=mask)
    else:
        out = grouper.agg(
            values=values,
            agg_func=agg_func,
            mask=mask,
        )

    out = out.unstack(level=list(range(len(index), len(index) + len(columns))), fill_value=0)
    if margins:
        margin_func = "sum" if agg_func in ("count", "size") else agg_func
        out = out.assign(Total=out.agg(margin_func, axis=1))
        out.loc['Total'] = out.agg(margin_func, axis=0)
    return out  


def crosstab(    
    index: ArrayCollection,
    columns: ArrayCollection,
    mask: Optional[ArrayType1D] = None,
    margins: bool = False,
):
    """
    Alias for pivot function.
    
    Parameters and returns are the same as for pivot.
    """
    return pivot(
        index=index, columns=columns, values=None, agg_func="size", mask=mask, margins=margins
    )


def add_row_margin(data: pd.Series | pd.DataFrame, agg_func="sum"):
    """
    Add a total rows to a DataFrame with multi-level index.  
    If the DataFrame has a single level index, it adds a 'All' row with the aggregated values.
    If the DataFrame has a multi-level index, it adds a 'All' row for each level of the index.
    Parameters
    ----------
    df : pd.Series | pd.DataFrame
        The Series or DataFrame to which the total row will be added
    agg_func : str or callable, default "sum    
        Aggregation function to use for calculating the total row.
    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'All' row containing the aggregated values.
    """
    from pandas.core.reshape.util import cartesian_product
    data = data.sort_index()
    index = data.index
    if index.nlevels == 1:
        data.loc['All'] = data.agg(agg_func)  
        return data

    new_levels = [lvl.tolist() + ['All'] for lvl in index.levels]
    new_codes = cartesian_product([np.arange(len(lvl)) for lvl in new_levels])
    new_index = pd.MultiIndex(codes=new_codes, levels=new_levels, names=index.names)
    out = data.reindex(new_index)
    keep = pd.Series(False, index=out.index)
    keep.loc[data.index] = True

    summaries = []
    levels = list(range(data.index.nlevels))
    for level in levels:
        other_levels = [lvl for lvl in levels if lvl != level]
        summary = data.groupby(level=other_levels, observed=True).agg(agg_func)
        summary = add_row_margin(summary, agg_func)
        summary = pd.concat({'All': summary}, names=[data.index.names[lvl] for lvl in [level, *other_levels]])
        summary.index = summary.index.reorder_levels(np.argsort([level, *other_levels]))
        summaries.append(summary)
        
    for summary in summaries:
        out.loc[summary.index] = summary
        keep.loc[summary.index] = True

    return out[keep]