from collections.abc import Mapping, Sequence
from functools import cached_property, wraps
from typing import Callable, List, Optional, Union, Literal, Tuple
from inspect import signature
import multiprocessing

import numpy as np
import pandas as pd
from pandas.core.algorithms import factorize_array
import polars as pl
import pyarrow as pa

from . import numba as numba_funcs
from .factorization import (
    factorize_1d,
    factorize_2d,
    monotonic_factorization,
)
from ..util import (
    ArrayType1D,
    ArrayType2D,
    to_arrow,
    is_categorical,
    array_split_with_chunk_handling,
    convert_data_to_arr_list_and_keys,
    get_array_name,
    series_is_numeric,
    parallel_map,
    mean_from_sum_count,
    _val_to_numpy,
)

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


def _get_indexes_from_values(arr_list: List[ArrayType1D]) -> List[pd.Index]:
    """
    Extract pandas Index objects from the provided values.

    Parameters
    ----------
    arr_list :
        List of arrays or Series to extract indexes from

    Returns
    -------
    List[pd.Index]
        List of pandas Index objects corresponding to each value in the collection
    """
    return [arr.index for arr in arr_list if isinstance(arr, pd.Series)]


def _validate_input_lengths_and_indexes(
    arr_list: List[ArrayType1D],
) -> Optional[pd.Index]:
    """
    Validate that all values have the same length and that any pandas indexes are compatible.

    Parameters
    ----------
    values : ArrayCollection

    Returns
    -------
    pd.Index or None
        Returns the first non-trivial index if any exists, otherwise None

    Raises
    ------
    ValueError
        If indexes have different lengths or non-trivial indexes don't match
    """
    lengths = set(map(len, arr_list))
    if len(lengths) > 1:
        raise ValueError(f"found more than one unique length: {lengths}")
    indexes = _get_indexes_from_values(arr_list)
    if len(indexes) == 0:
        return None

    for left, right in zip(indexes, indexes[1:]):
        if not left.equals(right):
            raise ValueError("Found different indices in the array_inputs")

    return indexes[0]


def groupby_method(method):

    @wraps(method)
    def wrapper(*args, **kwargs):
        bound_args = signature(method).bind(*args, **kwargs)
        group_key = bound_args.arguments["self"]
        if not isinstance(group_key, GroupBy):
            bound_args.arguments["self"] = GroupBy(group_key)
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

    def __init__(
        self,
        group_keys: ArrayCollection,
        sort: bool = True,
        factorize_large_inputs_in_chunks: bool = True,
    ):
        """
        Initialize the GroupBy object with the provided group keys.
        Parameters
        ----------
        group_keys : ArrayCollection
            The keys to group by, which can be a single array-like object or a collection of them.
        """
        if isinstance(group_keys, GroupBy):
            self._group_ikey, self._result_index = (
                group_keys.group_ikey,
                group_keys.result_index,
            )
            return

        group_key_list, group_key_names = convert_data_to_arr_list_and_keys(group_keys)
        self._key_index: pd.Index = _validate_input_lengths_and_indexes(group_key_list)
        self._index_is_sorted = False

        if len(group_key_list) == 1:
            group_key = group_key_list[0]
            is_cat = is_categorical(group_key)
            self._sort = sort and not is_cat

            if is_cat:
                factorize_in_chunks = False
            else:
                try:
                    to_arrow(group_key)
                    chunked = isinstance(to_arrow(group_key), pa.ChunkedArray)
                except TypeError:
                    chunked = False

                # TODO: estimate number of uniques based on initial slice of array
                # and do not factorize in chunk when number of uniques is estimated to be large
                factorize_in_chunks = (
                    factorize_large_inputs_in_chunks and len(group_key) >= 1_000_000
                ) or chunked

            if factorize_in_chunks:
                self._factorize_group_key_in_chunks(group_key)
            else:
                self._group_ikey, self._result_index = factorize_1d(group_key)
        else:
            self._sort = sort
            self._group_ikey, self._result_index = factorize_2d(
                *group_key_list, sort=sort
            )
            if sort:
                self._index_is_sorted = True

    @cached_property
    def _group_key_lengths(self):
        return (
            [len(k) for k in self._group_ikey.chunks]
            if self.key_is_chunked
            else [len(self.group_ikey)]
        )

    @cached_property
    def _chunk_offsets(self):
        return np.cumsum(self._group_key_lengths[:-1])

    @property
    def _n_threads_for_key_factorization(self):
        return 4

    def __len__(self):
        return sum(self._group_key_lengths)

    def _factorize_group_key_in_chunks(self, group_key: ArrayType1D):
        """
        Factorize a large group key array in chunks for better performance.
        This method splits the group key into smaller chunks, factorizes each chunk
        in parallel. The uniques are then combined to form the final result index and
        pointers from the individual code chunks in this index are built.
        These pointers are later used to populate the combined outputs of group-by functions.

        Parameters
        group_key : ArrayType1D
            The group key array to factorize.
        """
        # first try monotonic (increasing) factorization.
        # Optimization for thinks like date/time buckets, cumulative counts etc.
        # Exits as soon as it detects non-monotonicity and uses empty arrays to avoid wasted memory
        cutoff, mono_codes, mono_uniques = monotonic_factorization(group_key)
        mono_codes = mono_codes[:cutoff]
        if cutoff == len(group_key):
            # group_key is fully monotonic
            self._group_ikey, self._result_index = mono_codes, pd.Index(mono_uniques)
            return

        use_monotonic_piece = cutoff > len(group_key) / 4
        if use_monotonic_piece:
            group_key = group_key[cutoff:]

        group_key_list = _val_to_numpy(group_key, as_list=True)
        if len(group_key_list) == 1:
            group_key_chunks = np.array_split(
                group_key_list[0], self._n_threads_for_key_factorization
            )
        else:  # already a ChunkedArray
            group_key_chunks = group_key_list

        chunk_results = parallel_map(factorize_array, list(zip(group_key_chunks)))
        codes_list, unique_list = zip(*chunk_results)

        if use_monotonic_piece:
            codes_list = [mono_codes, *codes_list]
            unique_list = [mono_uniques, *unique_list]

        self._result_index = pd.Index(np.concatenate(unique_list)).unique()

        if self._sort:
            self._result_index = self._result_index.sort_values()
            self._index_is_sorted = True  # not necessary to sort now

        arg_list = [(arr,) for arr in unique_list]
        self._group_key_pointers = parallel_map(
            self.result_index._engine.get_indexer, arg_list
        )
        self._group_ikey = pa.chunked_array(codes_list)

    @property
    def key_is_chunked(self) -> bool:
        """
        Check if the group key is chunked.

        Returns
        -------
        bool
            True if the group key is chunked, False otherwise
        """
        return isinstance(self._group_ikey, pa.ChunkedArray)

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

    @cached_property
    def groups(self):
        """
        Dict mapping group names to row labels.

        Uses optimized numba implementation for better performance with large
        datasets.

        Returns
        -------
        dict
            Dictionary with group names as keys and arrays of row indices as
            values
        """
        return numba_funcs.build_groups_dict_optimized(
            self.group_ikey, self.result_index, self.ngroups
        )

    def _unify_group_key_chunks(self, keep_chunked=False):
        if self.key_is_chunked:
            # Could keep it chunked here and just do the re-pointing
            chunks = [
                p[k] for p, k in zip(self._group_key_pointers, self._group_ikey.chunks)
            ]
            if keep_chunked:
                self._group_ikey = pa.chunked_array(chunks)
            else:
                self._group_ikey = np.concatenate(chunks)

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
        if self.key_is_chunked:
            return self.group_ikey.null_count > 0
        else:
            return self.group_ikey.min() < 0

    @property
    def _max_threads_for_numba(self) -> int:
        """
        Between 1 and 4 threads depending on length of inputs
        """
        return min(4, 1 + len(self) // 1_000_000)

    def _preprocess_arguments(
        self, values: ArrayCollection, mask: Union[ArrayType1D, None]
    ):
        """
        Preprocess and validate input arguments for group-by operations.
        Filters out non-numeric series from DataFrame inputs.
        Checks that all inputs have the same length and compatible indexes.
        Returns the names and list of value arrays along with a common index.
        """
        value_list, value_names = convert_data_to_arr_list_and_keys(values)
        if isinstance(values, (pd.DataFrame, pl.DataFrame)):
            value_list, value_names = map(
                list,
                zip(
                    *[
                        (val, name)
                        for val, name in zip(value_list, value_names)
                        if series_is_numeric(val)
                    ]
                ),
            )

        to_check = value_list
        if mask is not None and pd.api.types.is_bool_dtype(mask):
            to_check = [*to_check, mask]

        common_index = _validate_input_lengths_and_indexes(to_check)
        input_len = len(to_check[0])

        if input_len != len(self):
            raise ValueError(
                f"Length of the input values ({input_len}) does not match length of group keys ({len(self)})"
            )
        if self._key_index is not None and common_index is not None:
            if not self._key_index.equals(common_index):
                raise ValueError(
                    "Pandas index of inputs does not match that of the group keys"
                )

        return value_names, value_list, common_index

    def _add_margins(
        self,
        result: Union[pd.DataFrame, pd.Series],
        margins: Union[bool, List[int]],
        func_name: str,
    ):
        if np.ndim(margins) == 1:
            levels = list(margins)
        else:
            levels = None

        return add_row_margin(
            result,
            agg_func=(
                "sum" if func_name in ("size", "count", "sum_squares") else func_name
            ),
            levels=levels,
        )

    def _build_arg_dict_for_function(self, func, values, mask, **kwargs):
        value_names, value_list, common_index = self._preprocess_arguments(values, mask)

        sig = signature(func)
        shared_kwargs = dict(
            group_key=self.group_ikey,
            mask=mask,
            ngroups=self.ngroups + 1,
            **kwargs,
        )
        if "n_threads" in sig.parameters:
            shared_kwargs["n_threads"] = self._max_threads_for_numba

        bound_args = [
            signature(func).bind(values=x, **shared_kwargs) for x in value_list
        ]
        keys = (name if name else f"_arr_{i}" for i, name in enumerate(value_names))
        arg_dict = {key: args.args for key, args in zip(keys, bound_args)}

        return arg_dict, common_index

    def _find_first_chunk_in_slice(self, mask: slice) -> int:
        """
        When masking with a chunked array with a slice some chunks may be entirely excluded.
        We need to track this to get the correct set of pointers into the combined result downstream.
        """
        if mask.step is not None and self.key_is_chunked:
            raise NotImplementedError(
                "masking with a stepped slicer and chunked group keys is not supported"
            )
        if mask.start is None:
            start = 0
        elif mask.start < 0:
            start = len(self) + mask.start
        else:
            start = mask.start

        # find first chunk within the mask as we need it below to get the right pointers
        cum_length = 0
        for i, x in enumerate(self._group_key_lengths):
            cum_length += x
            if cum_length > start:
                break
        first_chunk_in = i

        return first_chunk_in

    def _resolve_mask_argument_into_chunks(
        self, mask: Union[None, slice, np.ndarray]
    ) -> Tuple[Union[np.ndarray, pa.ChunkedArray], int, List]:
        """ """
        group_key = self.group_ikey
        first_chunk_in = 0
        mask_chunks = [None] * len(self._group_key_lengths)

        if isinstance(mask, slice):
            first_chunk_in = self._find_first_chunk_in_slice(mask)
            group_key = self.group_ikey[mask]
            mask_chunks = mask_chunks[first_chunk_in:]
        else:
            if self.key_is_chunked:
                if not pd.api.types.is_bool_dtype(mask):
                    # Fancy indexing does not work for chunked keys
                    bool_mask = np.full(len(self), False)
                    bool_mask[mask] = True
                    mask = bool_mask
                mask_chunks = array_split_with_chunk_handling(
                    mask, self._group_key_lengths
                )
            else:
                mask_chunks = [mask]

        return group_key, first_chunk_in, mask_chunks

    def _apply_gb_func_across_chunked_group_keys(
        self, func_name: str, value_list, mask=None
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        """
        Apply a group-by function across chunked group keys.
        This method handles cases where the group keys are chunked, applying the
        specified function to each chunk and combining the results.
        If value_list contains multiple arrays, the function is applied to each in parallel.
        This is achieved by splitting the values according to the chunk offsets of the group keys,
        applying the function to each chunk, and then combining the results.
        Thus, the function is applied in parallel across both the chunks of group keys and the multiple value arrays.
        """
        group_key, first_chunk_in, mask_chunks = (
            self._resolve_mask_argument_into_chunks(mask)
        )
        group_keys = group_key.chunks if self.key_is_chunked else [group_key]
        group_key_lengths = [len(k) for k in group_keys]

        func = getattr(numba_funcs, f"group_{func_name}")
        n_values = len(value_list)

        arg_list = []

        if self.key_is_chunked:
            # In this case we are already parallelising across the row axis
            threads_for_one_call = 1
        else:
            n_cpus = multiprocessing.cpu_count()
            max_threads = 2 * n_cpus - 1
            threads_for_one_call = max(1, max_threads // len(value_list))
            threads_for_one_call = min(
                threads_for_one_call, self._max_threads_for_numba
            )

        for values in value_list:
            if isinstance(mask, slice):
                values = values[mask]
            value_chunks = array_split_with_chunk_handling(
                values, chunk_lengths=group_key_lengths
            )
            for i, group_key in enumerate(group_keys):
                pointer = (
                    self._group_key_pointers[first_chunk_in + i]
                    if self.key_is_chunked
                    else self.result_index
                )
                bound_args = signature(func).bind(
                    values=value_chunks[i],
                    group_key=group_key,
                    mask=mask_chunks[i],
                    ngroups=(
                        len(pointer) + 1
                        if self.key_is_chunked
                        else self.ngroups + 1  # +1 for null group
                    ),
                    n_threads=threads_for_one_call,
                    return_count=True,
                )
                arg_list.append(bound_args.args)

        # one result per group-key chunk per value in value_list
        results, counts = zip(*parallel_map(func, arg_list))

        if not self.key_is_chunked:
            # single group key chunk, so we can return the results directly
            return list(zip(list(results), counts))

        # Now combine the results for each value in value_list to get one result per value
        individual_results = []
        # Some functions like 'first' and 'last' don't have nan versions
        if func_name in ("size", "count", "sum_squares"):
            reducer = numba_funcs.ScalarFuncs.nansum
        elif hasattr(numba_funcs.ScalarFuncs, f"nan{func_name}"):
            reducer = getattr(numba_funcs.ScalarFuncs, f"nan{func_name}")
        else:
            reducer = getattr(numba_funcs.ScalarFuncs, func_name)

        for i in range(n_values):
            slice_ = slice(i * len(group_keys), (i + 1) * len(group_keys))
            results_one_value = results[slice_]
            combined = numba_funcs._build_target_for_groupby(
                results_one_value[0].dtype,
                func_name,
                len(self._result_index) + 1,
            )
            counts_one_value = counts[slice_]
            count = np.zeros(len(self._result_index), dtype=np.int64)

            for j, result in enumerate(results_one_value):
                result = result[:-1]  # ignore null group
                pointer = self._group_key_pointers[first_chunk_in + j]
                combined[pointer] = numba_funcs.reduce_array_pair(
                    combined[pointer], result, reducer=reducer, counts=count[pointer]
                )
                count[pointer] += counts_one_value[j][:-1]  # ignore null group
            individual_results.append((combined, count))

        return individual_results

    def _apply_gb_func(
        self,
        func_name: str,
        values: Optional[ArrayCollection] = None,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Apply a group-by function to values.
        If values is a collection or DataFrame/2-D array, applies the function to each element in parallel.

        Parameters
        ----------
        func_name : str
            Name of the group-by function to apply (e.g., 'sum', 'mean', 'min', 'max', 'count', 'size')
        values : ArrayCollection
            Values to aggregate. Can be None if func_name == 'size'
        mask : ArrayType1D, optional
            Boolean mask to filter values
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only.
        observed_only : bool, default True
            If True, only include groups that are observed in the data

        Returns
        -------
        pd.Series or pd.DataFrame
            Results of the groupby operation
        """
        if transform and margins:
            raise ValueError("Cannot use transform and margins together")

        effective_func_name = func_name
        func_is_mean = func_name == "mean"
        if func_is_mean:
            effective_func_name = "sum"  # mean is calculated as sum/count

        value_names, value_list, common_index = self._preprocess_arguments(values, mask)
        return_1d = (
            (len(value_list) == 1)
            and isinstance(values, ArrayType1D)
            or isinstance(values, list)
            and np.ndim(values[0]) == 0
        )

        result_col_names = [
            name if name else f"_arr_{i}" for i, name in enumerate(value_names)
        ]

        results = self._apply_gb_func_across_chunked_group_keys(
            effective_func_name,
            value_list=value_list,
            mask=mask,
        )

        result_len = len(self.result_index)
        result_df = pd.DataFrame(
            {
                key: result[:result_len]
                for key, (result, count) in zip(result_col_names, results)
            },
            index=self.result_index,
            copy=False,
        )

        count_df = pd.DataFrame(
            {
                key: count[:result_len]
                for key, (result, count) in zip(result_col_names, results)
            },
            index=self.result_index,
            copy=False,
        )

        if func_name in ("size", "count"):
            result_df = count_df

        if transform:
            self._unify_group_key_chunks()
            result_df = result_df.iloc[self.group_ikey]
            if common_index is not None:
                result_df.index = common_index
            if return_1d:
                return result_df.squeeze(axis=1)
            else:
                return result_df

        if observed_only:
            observed = count_df.iloc[:, 0] > 0
            if func_name != "size" and not observed.all():
                # necessary but not sufficient condition for a group to be completely masked.
                # count == 0 can mean a group contains only null values so here we calculate the key counts.
                # Could optimize further by adding key count to numba functions outputs.
                # For size, we know there are no nulls and so observed is related to key counts.
                if mask is not None:
                    observed = self.size(mask=mask, observed_only=False) > 0
                else:
                    observed = self.key_count > 0

                result_df = result_df.loc[observed]
                count_df = count_df.loc[observed]

        if (
            self._sort and not self._index_is_sorted
        ):  # combined result for chunked keys are already sorted
            result_df.sort_index(inplace=True)

        if margins:
            result_df = self._add_margins(
                result_df, margins=margins, func_name=effective_func_name
            )
            if func_is_mean:
                count_df = self._add_margins(count_df, margins=margins, func_name="sum")

        if func_is_mean:
            with np.errstate(invalid="ignore", divide="ignore"):
                result_df = pd.DataFrame(
                    {
                        k: mean_from_sum_count(
                            result_df[k], count_df[k].reindex(result_df.index)
                        )
                        for k in result_df
                    }
                )

        if return_1d:
            result_df = result_df.squeeze(axis=1)
            if get_array_name(values) is None:
                result_df.name = None

        return result_df

    @groupby_method
    def size(
        self,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ):
        return self._apply_gb_func(
            "count",
            np.empty(len(self), dtype="int8"),
            mask=mask,
            transform=transform,
            margins=margins,
            observed_only=observed_only,
        )

    @cached_property
    def key_count(self):
        """
        Count of observations for each group, including empty groups.

        Returns
        -------
        pd.Series
            Series with group counts, including zero counts for empty groups
        """
        return self.size(observed_only=False)

    @groupby_method
    def count(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ):
        """
        Count non-null values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to count, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before counting.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..

        Returns
        -------
        pd.Series or pd.DataFrame
            Count of non-null values for each group.
        """
        return self._apply_gb_func(
            "count", values=values, mask=mask, transform=transform, margins=margins
        )

    @groupby_method
    def sum(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ):
        """
        Calculate sum of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to sum, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before summing.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..
        observed_only : bool, default True
            If True, only include groups that are observed in the data.
        Returns
        -------
        pd.Series or pd.DataFrame
            Sum of values for each group.
        """
        return self._apply_gb_func(
            "sum",
            values=values,
            mask=mask,
            transform=transform,
            margins=margins,
            observed_only=observed_only,
        )

    @groupby_method
    def mean(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ):
        """
        Calculate mean of values in each group.
        Parameters
        ----------
        values : ArrayCollection
            Values to calculate mean for, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculating mean.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..
        observed_only : bool, default True
            If True, only include groups that are observed in the data.
        Returns
        -------
        pd.Series or pd.DataFrame
            Mean of values for each group.
        """
        return self._apply_gb_func(
            "mean",
            values=values,
            mask=mask,
            transform=transform,
            margins=margins,
            observed_only=observed_only,
        )

    @groupby_method
    def min(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ):
        """
        Calculate minimum value in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to find minimum for, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before finding minimum.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..
        observed_only : bool, default True
            If True, only include groups that are observed in the data.

        Returns
        -------
        pd.Series or pd.DataFrame
            Minimum value for each group.
        """
        return self._apply_gb_func(
            "min",
            values=values,
            mask=mask,
            transform=transform,
            margins=margins,
            observed_only=observed_only,
        )

    @groupby_method
    def max(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ):
        """
        Calculate maximum value in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to find maximum for, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before finding maximum.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..
        observed_only : bool, default True
            If True, only include groups that are observed in the data.

        Returns
        -------
        pd.Series or pd.DataFrame
            Maximum value for each group.
        """
        return self._apply_gb_func(
            "max",
            values=values,
            mask=mask,
            transform=transform,
            margins=margins,
            observed_only=observed_only,
        )

    @groupby_method
    def median(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        observed_only: bool = True,
    ):
        """Calculate the median of the provided values for each group.
        Parameters
        ----------
        values : ArrayCollection
            Values to calculate the median for, can be a single array/Series or a collection of them.
        mask : Optional[ArrayType1D], default None
            Boolean mask to filter values before calculating the median.
        transform : bool, default False
            If True, return values with the same shape as input rather than one value per group.
        observed_only : bool, default True
            If True, only include groups that are observed in the data.
        Returns
        -------
        pd.Series or pd.DataFrame
            The median of the values for each group.
            If `transform` is True, returns a Series/DataFrame with the same shape as input.
        """
        value_names, value_list, common_index = self._preprocess_arguments(values, mask)

        if mask is None:
            mask = True
        if self.has_null_keys:
            mask = mask & (self.group_ikey >= 0)

        if mask is True:
            mask = slice(None)

        tmp_df = pd.DataFrame(
            {k: v[mask] for k, v in zip(value_names, value_list)}, copy=False
        )
        self._unify_group_key_chunks()
        result = tmp_df.groupby(self.group_ikey[mask], observed=observed_only).median()
        if transform:
            result = result.reindex(self.group_ikey, copy=False)
        else:
            result.index = self.result_index[result.index]
            if len(value_list) == 1 and isinstance(values, ArrayType1D):
                result = result.iloc[:, 0]
        if self._sort and not self._index_is_sorted:
            result.sort_index(inplace=True)
        return result

    @groupby_method
    def var(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        ddof: int = 1,
        observed_only: bool = True,
    ):
        """
        Calculate variance of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate variance for, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculating variance.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..
        ddof : int, default 0
            Delta degrees of freedom.
        observed_only : bool, default True
            If True, only include groups that are observed in the data.

        Returns
        -------
        pd.Series or pd.DataFrame
            Variance of values for each group.
        """
        kwargs = dict(
            mask=mask, margins=margins, transform=transform, observed_only=observed_only
        )
        sq_sum = self._apply_gb_func("sum_squares", values=values, **kwargs)
        sum_sq = self.sum(values=values, **kwargs).astype(float) ** 2
        count = self.count(values=values, **kwargs)
        return (sq_sum - sum_sq / count) / (count - ddof)

    @groupby_method
    def std(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        ddof: int = 1,
        observed_only: bool = True,
    ):
        """
        Calculate standard deviation of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate standard deviation for, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculating standard deviation.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..
        ddof : int, default 0
            Delta degrees of freedom.
        observed_only : bool, default True
            If True, only include groups that are observed in the data.

        Returns
        -------
        pd.Series or pd.DataFrame
            Standard deviation of values for each group.
        """
        return GroupBy.var(**locals()) ** 0.5

    @groupby_method
    def first(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ):
        """
        Get the first non-null value in each group. Use nth(0) for the first value including nulls.

        Parameters
        ----------
        values : ArrayCollection
            Values to get the first value from, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before getting the first value.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..
        observed_only : bool, default True
            If True, only include groups that are observed in the data.

        Returns
        -------
        pd.Series or pd.DataFrame
            First value for each group.
        """
        return self._apply_gb_func(
            "first",
            values=values,
            mask=mask,
            transform=transform,
            margins=margins,
            observed_only=observed_only,
        )

    @groupby_method
    def last(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ):
        """
        Get the last non-null value in each group. Use nth(-1) for the last value including nulls.

        Parameters
        ----------
        values : ArrayCollection
            Values to get the last value from, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before getting the last value.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..
        observed_only : bool, default True
            If True, only include groups that are observed in the data.

        Returns
        -------
        pd.Series or pd.DataFrame
            Last value for each group.
        """
        return self._apply_gb_func(
            "last",
            values=values,
            mask=mask,
            transform=transform,
            margins=margins,
            observed_only=observed_only,
        )

    @groupby_method
    def agg(
        self,
        values: ArrayCollection,
        agg_func: Callable | str | List[str],
        mask: Optional[ArrayType1D] = None,
        transform: bool = False,
        margins: bool = False,
        observed_only: bool = True,
    ):
        """
        Apply aggregation function(s) to values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to aggregate, can be a single array/Series or a collection of them.
        agg_func : callable, str, or list of str
            Aggregation function(s) to apply. Can be a single function name or list of function names.
        mask : ArrayType1D, optional
            Boolean mask to filter values before aggregation.
        transform : bool, default False
            If True, return values with same shape as input rather than one value per group.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..
        observed_only : bool, default True
            If True, only include groups that are observed in the data.

        Returns
        -------
        pd.Series or pd.DataFrame
            Aggregated values for each group.
        """
        if np.ndim(agg_func) == 0:
            if isinstance(agg_func, Callable):
                agg_func = agg_func.__name__
            func = getattr(self, agg_func)
            return func(values, mask=mask, transform=transform, margins=margins)

        elif np.ndim(agg_func) == 1:
            if isinstance(values, ArrayType1D):
                value_list, value_names = [values] * len(agg_func), agg_func
            else:
                value_list, value_names = convert_data_to_arr_list_and_keys(values)

            if len(agg_func) != len(value_list):
                raise ValueError(
                    f"Mismatch between number of agg funcs ({len(agg_func)}) "
                    f"and number of values ({len(values)})"
                )

            args_list = [
                signature(self.agg)
                .bind(
                    v,
                    agg_func=f,
                    mask=mask,
                    transform=transform,
                    margins=margins,
                    observed_only=observed_only,
                )
                .args
                for f, v in zip(agg_func, value_list)
            ]
            results = parallel_map(self.agg, args_list)
            return pd.DataFrame(dict(zip(value_names, results)), copy=False)
        else:
            raise TypeError(
                "agg_func must by a single function name or an iterable of same"
            )

    @groupby_method
    def ratio(
        self,
        values1: ArrayCollection,
        values2: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        agg_func="sum",
        margins: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate ratio of two aggregated values in each group.

        Parameters
        ----------
        values1 : ArrayCollection
            Numerator values for ratio calculation.
        values2 : ArrayCollection
            Denominator values for ratio calculation.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculating ratio.
        agg_func : str, default "sum"
            Aggregation function to apply before ratio calculation.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..

        Returns
        -------
        pd.Series or pd.DataFrame
            Ratio of aggregated values1 to values2 for each group.
        """
        # check for nullity
        value_list_1, _ = convert_data_to_arr_list_and_keys(values1)
        value_list_2, _ = convert_data_to_arr_list_and_keys(values2)
        if len(value_list_1) != len(value_list_2):
            raise ValueError(
                f"Number of columns in values1 and values2 must be equal. \n"
                f"Found {len(value_list_1), len(value_list_2)}"
            )
        for left, right in zip(value_list_1, value_list_2):
            if (pd.isna(left) != pd.isna(right)).any():
                raise ValueError(
                    "Values must have the same nullity as otherwise the ratio is undefined. "
                    f"Found {left} and {right} with different null values."
                )
        kwargs = dict(mask=mask, agg_func=agg_func, margins=margins)
        return self.agg(values1, **kwargs) / self.agg(values2, **kwargs)

    def subset_ratio(
        self,
        values: ArrayCollection,
        subset_mask: ArrayType1D,
        global_mask: Optional[ArrayType1D] = None,
        agg_func="sum",
        margins: bool = False,
    ) -> pd.Series | pd.DataFrame:
        """
        Calculate ratio of subset to total values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate ratio for.
        subset_mask : ArrayType1D
            Boolean mask defining the subset of interest.
        global_mask : ArrayType1D, optional
            Optional global boolean mask to apply to all calculations.
        agg_func : str, default "sum"
            Aggregation function to apply before ratio calculation.
        margins : bool or list of int, default False
            If True, include a total row in the result. If list of integers,
            include margin rows for the specified levels only..

        Returns
        -------
        pd.Series or pd.DataFrame
            Ratio of subset aggregated values to total aggregated values for each group.
        """
        # check for nullity
        kwargs = dict(agg_func=agg_func, margins=margins, values=values)
        return self.agg(**kwargs, mask=subset_mask & global_mask) / self.agg(
            **kwargs, mask=global_mask
        )

    @groupby_method
    def density(
        self,
        values: Optional[ArrayCollection] = None,
        mask: Optional[ArrayType1D] = None,
        margins: bool = False,
    ):
        """
        Calculate density (percentage) of values in each group relative to total.

        Parameters
        ----------
        values : ArrayCollection, optional
            Values to calculate density for. If None, uses group sizes.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculating density.
        margins : bool, default False
            If True, include total values in the result.

        Returns
        -------
        pd.Series or pd.DataFrame
            Density values as percentages for each group.
        """
        if values is None:
            totals = self.size(mask, margins=True)
        else:
            totals = self.sum(values, mask, margins=True)
        if self.result_index.nlevels == 1:
            density = 100 * totals / totals.loc["All"]
            if margins:
                density.loc["All"] = totals.loc["All"]
            else:
                density = density.drop("All")

        elif self.result_index.nlevels == 2:
            density = (
                100
                * totals
                / totals.index.get_level_values(0).map(totals.xs("All", 0, 1))
            )
            all_rows = totals.index.get_level_values(1) == "All"
            if margins:
                density.loc[all_rows] = totals[all_rows]
            else:
                density = density[~all_rows]
        else:
            raise ValueError()

        return density

    def _get_row_selection(
        self,
        values: ArrayCollection,
        ilocs: np.ndarray,
        keep_input_index: bool = False,
        n: Optional[int] = None,
    ):
        value_list, value_names = convert_data_to_arr_list_and_keys(values)
        common_index = _validate_input_lengths_and_indexes(value_list)
        keep = ilocs > -1
        ilocs = ilocs[keep]

        if keep_input_index:
            if common_index is None:
                common_index = pd.RangeIndex(len(value_list[0]))
            out_index = common_index[ilocs]
        else:
            if n is None:
                # For cases where we don't have n, use a simple range index
                n_selected = len(ilocs)
                out_index = pd.RangeIndex(n_selected)
            else:
                new_codes = [np.repeat(c, n)[keep] for c in self.result_index.codes]
                new_codes.append(np.tile(np.arange(n), self.ngroups)[keep])
                new_levels = [*self.result_index.levels, np.arange(n)]
                out_index = pd.MultiIndex(
                    codes=new_codes,
                    levels=new_levels,
                    names=[*self.result_index.names, None],
                )[keep]

        return_1d = isinstance(values, ArrayType1D)
        result = (
            pd.DataFrame(dict(zip(value_names, value_list)), copy=False)
            .iloc[ilocs]
            .set_index(out_index)
        )
        if return_1d:
            result = result.squeeze(axis=1)

        if self._sort:
            result.sort_index(inplace=True)

        return result

    def head(self, values: ArrayCollection, n: int, keep_input_index: bool = False):
        """
        Return first n rows of each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to select from.
        n : int
            Number of rows to select from the beginning of each group.
        keep_input_index : bool, default False
            If True, preserve the original index of the input values, otherwise use the group keys.

        Returns
        -------
        pd.Series or pd.DataFrame
            First n rows from each group.
        """
        # Convert group_ikey to numpy array for numba compatibility
        if self.key_is_chunked:
            print("Unifying chunked group-key before finding head")
            self._unify_group_key_chunks()

        ilocs = numba_funcs._find_first_or_last_n(
            group_key=self.group_ikey,
            ngroups=self.ngroups,
            n=n,
            forward=True,
        )
        return self._get_row_selection(
            values=values, ilocs=ilocs, keep_input_index=keep_input_index, n=n
        )

    def tail(self, values: ArrayCollection, n: int, keep_input_index: bool = False):
        """
        Return last n rows of each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to select from.
        n : int
            Number of rows to select from the end of each group.
        keep_input_index : bool, default False
            If True, preserve the original index of the input values, otherwise use the group keys.

        Returns
        -------
        pd.Series or pd.DataFrame
            Last n rows from each group.
        """
        if self.key_is_chunked:
            print("Unifying chunked group-key before finding tail")
            self._unify_group_key_chunks()

        ilocs = numba_funcs._find_first_or_last_n(
            group_key=self.group_ikey,
            ngroups=self.ngroups,
            n=n,
            forward=False,
        )
        return self._get_row_selection(
            values=values, ilocs=ilocs, keep_input_index=keep_input_index, n=n
        )

    def nth(self, values: ArrayCollection, n: int, keep_input_index: bool = False):
        """
        Return nth row of each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to select from.
        n : int
            The position to select from each group (0-indexed). Can also be negative to select from the end.
        keep_input_index : bool, default False
            If True, preserve the original index of the input values, otherwise use the group keys.

        Returns
        -------
        pd.Series or pd.DataFrame
            The nth row from each group.
        """
        if self.key_is_chunked:
            print("Unifying chunked group-key before finding nth")
            self._unify_group_key_chunks()

        ilocs = numba_funcs._find_nth(
            group_key=self.group_ikey, ngroups=self.ngroups, n=n
        )
        return self._get_row_selection(values, ilocs, keep_input_index, n=n)

    def _apply_rolling_or_cumulative_func(
        self,
        func_name: str,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        **kwargs,
    ):
        """
        Shared implementation for rolling/cumulative aggregation methods.

        Parameters
        ----------
        func_name : str
            Name of the rolling function to call ('rolling_sum', 'rolling_mean', etc.)
        values : ArrayCollection
            Values to aggregate, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation.

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling aggregation results with same shape as input.
        """
        # Get the appropriate numba function
        func = getattr(numba_funcs, func_name)

        if self.key_is_chunked:
            print("Unifying chunked group-key before cumulative group-by")
            self._unify_group_key_chunks()

        arg_dict, common_index = self._build_arg_dict_for_function(
            func,
            values=values,
            mask=mask,
            **kwargs,
        )
        results = parallel_map(func, arg_dict.values())

        out_dict = {}
        for key, result in zip(arg_dict, results):
            out_dict[key] = pd.Series(result, common_index)

        return_1d = len(arg_dict) == 1 and isinstance(values, ArrayType1D)
        out = pd.DataFrame(out_dict)
        if return_1d:
            out = out.squeeze(axis=1)
            if get_array_name(values) is None:
                out.name = None
        return out

    @groupby_method
    def rolling_sum(
        self,
        values: ArrayCollection,
        window: int,
        mask: Optional[ArrayType1D] = None,
    ):
        """
        Calculate rolling sum of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate rolling sum for, can be a single array/Series or a collection of them.
        window : int
            Size of the rolling window.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation.

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling sum of values for each group, same shape as input.
        """
        return self._apply_rolling_or_cumulative_func(
            "rolling_sum", values, window=window, mask=mask
        )

    @groupby_method
    def rolling_mean(
        self,
        values: ArrayCollection,
        window: int,
        mask: Optional[ArrayType1D] = None,
    ):
        """
        Calculate rolling mean of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate rolling mean for, can be a single array/Series or a collection of them.
        window : int
            Size of the rolling window.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation.

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling mean of values for each group, same shape as input.
        """
        return self._apply_rolling_or_cumulative_func(
            "rolling_mean", values, window=window, mask=mask
        )

    @groupby_method
    def rolling_min(
        self,
        values: ArrayCollection,
        window: int,
        mask: Optional[ArrayType1D] = None,
    ):
        """
        Calculate rolling minimum of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate rolling minimum for, can be a single array/Series or a collection of them.
        window : int
            Size of the rolling window.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation.

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling minimum of values for each group, same shape as input.
        """
        return self._apply_rolling_or_cumulative_func(
            "rolling_min", values, window=window, mask=mask
        )

    @groupby_method
    def rolling_max(
        self,
        values: ArrayCollection,
        window: int,
        mask: Optional[ArrayType1D] = None,
    ):
        """
        Calculate rolling maximum of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate rolling maximum for, can be a single array/Series or a collection of them.
        window : int
            Size of the rolling window.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation.

        Returns
        -------
        pd.Series or pd.DataFrame
            Rolling maximum of values for each group, same shape as input.
        """
        return self._apply_rolling_or_cumulative_func(
            "rolling_max", values, window=window, mask=mask
        )

    @groupby_method
    def cumsum(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        skip_na: bool = True,
    ):
        """
        Calculate cumulative sum of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate cumulative sum for, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation.
        skip_na : bool, default True
            Whether to skip NA/null values in the calculation.

        Returns
        -------
        pd.Series or pd.DataFrame
            Cumulative sum of values for each group, same shape as input.
        """
        return self._apply_rolling_or_cumulative_func(
            "cumsum", values, mask, skip_na=skip_na
        )

    @groupby_method
    def cumcount(
        self,
        mask: Optional[ArrayType1D] = None,
    ):
        """
        Calculate cumulative count in each group.
        Note this is the base-0 count of each group regardless of nullity,
        which is consistent with Pandas but inconsistent with the .count method (Pandas has this inconsistency)

        Parameters
        ----------
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation.

        Returns
        -------
        pd.Series
            Cumulative count for each group, same shape as input.
        """
        return self._apply_rolling_or_cumulative_func("cumcount", self.group_ikey, mask)

    @groupby_method
    def cummin(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        skip_na: bool = True,
    ):
        """
        Calculate cumulative minimum of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate cumulative minimum for, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation.
        skip_na : bool, default True
            Whether to skip NA/null values in the calculation.

        Returns
        -------
        pd.Series or pd.DataFrame
            Cumulative minimum of values for each group, same shape as input.
        """
        return self._apply_rolling_or_cumulative_func(
            "cummin", values, mask, skip_na=skip_na
        )

    @groupby_method
    def cummax(
        self,
        values: ArrayCollection,
        mask: Optional[ArrayType1D] = None,
        skip_na: bool = True,
    ):
        """
        Calculate cumulative maximum of values in each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate cumulative maximum for, can be a single array/Series or a collection of them.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculation.
        skip_na : bool, default True
            Whether to skip NA/null values in the calculation.

        Returns
        -------
        pd.Series or pd.DataFrame
            Cumulative maximum of values for each group, same shape as input.
        """
        return self._apply_rolling_or_cumulative_func(
            "cummax", values, mask, skip_na=skip_na
        )

    @groupby_method
    def shift(
        self,
        values: ArrayCollection,
        window: int = 1,
        mask: Optional[ArrayType1D] = None,
    ):
        """
        Shift values within each group by a specified number of periods.

        Parameters
        ----------
        values : ArrayCollection
            Values to shift, can be a single array/Series or a collection of them.
        periods : int, default 1
            Number of periods to shift. Currently only supports periods=1.
        mask : ArrayType1D, optional
            Boolean mask to filter values before shifting.

        Returns
        -------
        pd.Series or pd.DataFrame
            Shifted values for each group, same shape as input.

        Notes
        -----
        Currently only supports periods=1. Multi-period shifting will be
        added in a future version.

        Examples
        --------
        >>> import pandas as pd
        >>> from pandas_plus.groupby import GroupBy
        >>> data = pd.DataFrame({
        ...     'group': ['A', 'A', 'B', 'B'],
        ...     'values': [1, 2, 3, 4]
        ... })
        >>> groupby = GroupBy(data['group'])
        >>> groupby.shift(data['values'])
        0    NaN
        1    1.0
        2    NaN
        3    3.0
        Name: values, dtype: float64
        """
        return self._apply_rolling_or_cumulative_func(
            "rolling_shift", values=values, window=window, mask=mask
        )

    rolling_shift = shift

    @groupby_method
    def diff(
        self,
        values: ArrayCollection,
        window: int = 1,
        mask: Optional[ArrayType1D] = None,
    ):
        """
        Calculate the difference between consecutive elements within each group.

        Parameters
        ----------
        values : ArrayCollection
            Values to calculate differences for, can be a single array/Series or a collection of them.
        periods : int, default 1
            Number of periods to use for calculating difference. Currently only supports periods=1.
        mask : ArrayType1D, optional
            Boolean mask to filter values before calculating differences.

        Returns
        -------
        pd.Series or pd.DataFrame
            First differences for each group, same shape as input.

        Notes
        -----
        Currently only supports periods=1. Multi-period differences will be
        added in a future version.

        Examples
        --------
        >>> import pandas as pd
        >>> from pandas_plus.groupby import GroupBy
        >>> data = pd.DataFrame({
        ...     'group': ['A', 'A', 'B', 'B'],
        ...     'values': [1, 3, 2, 6]
        ... })
        >>> groupby = GroupBy(data['group'])
        >>> groupby.diff(data['values'])
        0    NaN
        1    2.0
        2    NaN
        3    4.0
        Name: values, dtype: float64
        """
        return self._apply_rolling_or_cumulative_func(
            "rolling_diff", values=values, window=window, mask=mask
        )

    rolling_diff = diff

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
        return numba_funcs.group_nearby_members(
            group_key=self.group_ikey,
            values=values,
            max_diff=max_diff,
            n_groups=self.ngroups,
        )


def crosstab(
    index: ArrayCollection,
    columns: ArrayCollection,
    values: Optional[ArrayCollection] = None,
    aggfunc: str = "sum",
    mask: Optional[ArrayType1D] = None,
    margins: Literal[True, False, "row", "column"] = False,
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
    aggfunc : str, default "sum"
        Aggregation function to apply to the values. Can be a string like "sum", "mean", "min", "max", etc.
    mask : Optional[ArrayType1D], default None
        Boolean mask to filter values before cross-tabulation
    margin : Literal[True, False, "row", "column"]
        If True, adds a total row and column to the resulting DataFrame
        if "row", or "column", add margins to that axis only
    Returns
    -------
    pd.DataFrame
        Cross-tabulated DataFrame with group keys as index and values as columns
    """
    index, index_names = convert_data_to_arr_list_and_keys(index)
    columns, index_columns = convert_data_to_arr_list_and_keys(columns)

    n0, n1 = len(index), len(columns)
    levels = list(range(n0 + n1))

    n0, n1 = len(index), len(columns)
    levels = list(range(n0 + n1))
    row_levels = levels[:n0]
    column_levels = levels[n0:]

    do_column_margin = margins in (True, "column")
    do_row_margin = margins in (True, "row")

    margin_levels = []
    if do_row_margin:
        margin_levels += row_levels
    if do_column_margin:
        margin_levels += column_levels

    grouper = GroupBy(index + columns, sort=False)
    if values is None:
        aggregation = grouper.size(mask=mask, margins=margin_levels)
    elif aggfunc == "size":
        raise ValueError(
            "aggfunc == 'size' only valid when values is None. Try count instead (for count of non-null values)"
        )
    else:
        aggregation = grouper.agg(
            values=values,
            agg_func=aggfunc,
            mask=mask,
            margins=margin_levels,
        )

    table = aggregation.unstack(level=column_levels)

    if not do_column_margin:
        all_levels = grouper.result_index.levels
        if len(column_levels) == 1:
            columns = all_levels[-1]
        else:
            columns = pd.MultiIndex.from_product(
                [all_levels[lvl] for lvl in column_levels]
            )

        table = table[[c for c in columns if c in table]]

    return table


def add_row_margin(
    data: pd.Series | pd.DataFrame, agg_func="sum", levels: Optional[List[int]] = None
):
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
        data.loc["All"] = data.agg(agg_func)
        return data

    all_levels = list(range(data.index.nlevels))
    if levels is None:
        levels = all_levels

    new_levels = [index.levels[lvl].tolist() + ["All"] for lvl in all_levels]
    new_codes = cartesian_product([np.arange(len(lvl)) for lvl in new_levels])
    new_index = pd.MultiIndex(codes=new_codes, levels=new_levels, names=index.names)
    out = data.reindex(new_index, fill_value=0)
    keep = pd.Series(False, index=out.index)
    keep.loc[data.index] = True

    summaries = []

    for level in levels:
        other_levels = [lvl for lvl in all_levels if lvl != level]
        summary = data.groupby(level=other_levels, observed=True).agg(agg_func)
        summary = add_row_margin(summary, agg_func)
        summary = pd.concat(
            {"All": summary},
            names=[data.index.names[lvl] for lvl in [level, *other_levels]],
        )
        summary.index = summary.index.reorder_levels(np.argsort([level, *other_levels]))
        summaries.append(summary)

    for summary in summaries:
        out.loc[summary.index] = summary
        keep.loc[summary.index] = True

    for lvl in set(all_levels) - set(levels):
        out.drop("All", level=lvl, inplace=True)

    return out[keep]


def value_counts(x, normalize: bool = False, mask: Optional[ArrayType1D] = None):
    """ """
    vc = GroupBy.size(x)
    if normalize:
        vc = vc / vc.sum()
    return vc
