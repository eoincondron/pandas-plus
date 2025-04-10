from collections.abc import Mapping, Sequence
from functools import cached_property, wraps
from typing import Callable, List
from inspect import signature

import numpy as np
import pandas as pd
import polars as pl

from .numba import group_count, group_max, group_mean, group_min, group_sum
from ..util import (ArrayType1D, ArrayType2D, TempName,
                   convert_array_inputs_to_dict, get_array_name)

ArrayCollection = (
    ArrayType1D | ArrayType2D | Sequence[ArrayType1D] | Mapping[ArrayType1D]
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
        return val.to_numpy()
    except AttributeError:
        return np.asarray(val)


def _validate_input_indexes(indexes):
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
        group_key_dict = convert_array_inputs_to_dict(group_keys)
        group_key_dict = {
            key: array_to_series(val) for key, val in group_key_dict.items()
        }
        indexes = [s.index for s in group_key_dict.values()]
        common_index = _validate_input_indexes(indexes)
        if common_index is not None:
            group_key_dict = {
                key: s.set_axis(common_index, axis=0, copy=False)
                for key, s in group_key_dict.items()
            }

        self._group_df = pd.DataFrame(group_key_dict, copy=False)
        self._grouper = self._group_df.groupby(
            list(group_key_dict), observed=True
        )._grouper
        self.key_names = [
            None if isinstance(key, TempName) else key for key in group_key_dict
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
        return self._grouper.ngroups

    @cached_property
    def result_index(self):
        """
        Index for the result of group-by operations.
        
        Returns
        -------
        pd.Index
            Index with one level per group key
        """
        index = self._grouper.result_index
        index.names = self.key_names
        return index

    @cached_property
    def group_ikey(self):
        """
        Integer key for each original row identifying its group.
        
        Returns
        -------
        ndarray
            Array of group indices for each original row
        """
        return self._grouper.group_info[0]

    def _apply_gb_func(
        self,
        func: Callable,
        values: ArrayCollection,
        mask: ArrayType1D = None,
        transform: bool = False,
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
            
        Returns
        -------
        pd.Series or pd.DataFrame
            Results of the groupby operation
        """
        value_dict = convert_array_inputs_to_dict(values)
        np_values = list(map(val_to_numpy, value_dict.values()))
        results = map(
            lambda v: func(
                group_key=self.group_ikey, values=v, mask=mask, ngroups=self.ngroups
            ),
            np_values,
        )
        out_dict = {}
        for key, result in zip(value_dict, results):
            if transform:
                result = out_dict[key] = pd.Series(
                    result[self.group_ikey], self._group_df.index
                )
            else:
                result = out_dict[key] = pd.Series(result, self.result_index)

            return_1d = len(value_dict) == 1 and isinstance(values, ArrayType1D)
            if return_1d:
                out = result
                if get_array_name(values) is None:
                    out.name = None
            else:
                out = pd.DataFrame(out_dict)

        if not transform and mask is not None:
            count = self.sum(values=mask)
            out = out.loc[count > 0]

        return out

    @groupby_method
    def count(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_count, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def sum(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_sum, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def mean(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_mean, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def min(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_min, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def max(
        self, values: ArrayCollection, mask: ArrayType1D = None, transform: bool = False
    ):
        return self._apply_gb_func(
            group_max, values=values, mask=mask, transform=transform
        )

    @groupby_method
    def agg(
        self,
        values: ArrayCollection,
        agg_func: str | List[str],
        mask: ArrayType1D = None,
        transform: bool = False,
    ):
        if np.ndim(agg_func) == 0:
            if isinstance(agg_func, Callable):
                agg_func = agg_func.__name__
            func = getattr(self, agg_func)
            return func(values, mask=mask, transform=transform)
        elif np.ndim(agg_func) == 1:
            values = convert_array_inputs_to_dict(values)
            if len(agg_func) != len(values):
                raise ValueError(
                    f"Mismatch between number of agg funcs ({len(agg_func)}) "
                    f"and number of values ({len(values)})"
                )
            return pd.DataFrame(
                {k:
                    self.agg(v, agg_func=f, mask=mask, transform=transform)
                    for f, (k, v) in zip(agg_func, values.items())
                 },
            )
        else:
            raise TypeError("agg_func must by a single function name or an iterable of same")

    @groupby_method
    def ratio(
        self,
        values1: ArrayCollection,
        values2: ArrayCollection,
        mask: ArrayType1D = None,
        agg_func="sum",
    ):
        # check for nullity
        self.agg(values1, agg_func, mask) / self.agg(values2, agg_func, mask)

    @groupby_method
    def subset_ratio(
            self,
            values: ArrayCollection,
            subset_mask: ArrayType1D,
            global_mask: ArrayType1D = None,
            agg_func="sum",
    ):
        # check for nullity
        self.agg(values, agg_func, subset_mask & global_mask) / self.agg(values, agg_func, global_mask)
