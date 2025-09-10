import inspect
from inspect import signature
from typing import Callable, Optional, List, Tuple
from functools import wraps, reduce

import numba as nb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from numba.typed import List as NumbaList

from ..util import (
    ArrayType1D,
    check_data_inputs_aligned,
    is_null,
    _null_value_for_numpy_type,
    _maybe_cast_timestamp_arr,
    parallel_map,
    NumbaReductionOps,
    _scalar_func_decorator,
    _val_to_numpy,
)
from .. import nanops


# ===== Array Preparation Methods =====


def _build_target_for_groupby(np_type, operation: str, shape):
    if operation in ("count", "nancount"):
        # for counts, the target is redundant as we collect the counts in a separate array
        target = np.zeros(shape, dtype=bool)
        return target

    dtype = np_type
    if "sum" in operation:
        if np_type.kind in "iub":
            dtype = "uint64" if np_type.kind == "u" else "int64"
        initial_value = 0
    else:
        initial_value = _null_value_for_numpy_type(np.dtype(dtype))

    target = np.full(shape, initial_value, dtype=dtype)

    return target


def _chunk_groupby_args(
    n_chunks: int,
    reduce_func_name: str,
    group_key: np.ndarray,
    values: List[np.ndarray] | np.ndarray | None,
    ngroups: int,
    mask: Optional[np.ndarray] = None,
):
    """
    Splits groupby arguments into chunks for parallel or chunked processing.

    This function partitions the input arrays (`group_key`, `values`, and `mask`) into `n_chunks`
    and prepares argument sets for chunked groupby-reduce operations. It supports both single
    array and list-of-arrays for `values`, ensuring alignment with `group_key`. The resulting
    arguments are suitable for use with the `_group_by_reduce` function.

    Parameters
    ----------
    n_chunks : int
        The number of chunks to split the data into.
    group_key : np.ndarray
        Array of group labels, used to group the data.
    values : list of np.ndarray, np.ndarray, or None
        The data values to be grouped and reduced. Can be a single array, a list of arrays,
        or None if not required.
    ngroups: int
        The number of distinct groups
    reduce_func_name : str
        The name of reduction function to apply to each group.
    mask : np.ndarray, optional
        Optional boolean mask to filter the data.

    Returns
    -------
    chunked_args : list
        A list of `BoundArguments` objects, each containing the arguments for a chunked
        groupby-reduce operation.

    Raises
    ------
    ValueError
        If the total length of `values` does not match the length of `group_key` when
        `values` is a list.

    Notes
    -----
    This function is intended for internal use to facilitate chunked or parallel groupby
    operations, especially when using Numba or similar parallelization tools.
    """
    kwargs = locals().copy()
    del kwargs["n_chunks"]

    if isinstance(values, NumbaList):
        if mask is not None and mask.dtype.kind in "ui":
            assert isinstance(
                values, np.ndarray
            ), "Fancy indexing with chunked args is not allowed"
        chunked_args = [
            kwargs | chunk
            for chunk in _chunk_args_for_chunked_values(group_key, values, mask)
        ]

    elif mask is not None:
        if mask.dtype.kind == "b":
            mask = mask.nonzero()[0]
        chunked_args = (
            kwargs | dict(mask=chunk) for chunk in np.array_split(mask, n_chunks)
        )

    else:
        chunked_args = [
            kwargs | chunk
            for chunk in _chunk_args_for_unchunked_values(group_key, values, n_chunks)
        ]

    return [
        signature(_apply_group_method_single_chunk).bind_partial(**chunk)
        for chunk in chunked_args
    ]


def _chunk_args_for_chunked_values(
    group_key: np.ndarray,
    values: List[np.ndarray],
    mask: Optional[np.ndarray] = None,
):
    lengths = [len(chunk) for chunk in values]
    if sum(lengths) != len(group_key):
        raise ValueError(
            "Length of group_key must match total length of all arrays in values"
        )
    splits = np.cumsum(lengths[:-1])
    key_list = np.array_split(group_key, splits)
    mask_list = [None] * len(key_list) if mask is None else np.array_split(mask, splits)
    return [
        dict(
            group_key=k,
            values=v,
            mask=m,
        )
        for k, v, m in zip(key_list, values, mask_list)
    ]


def _chunk_args_for_unchunked_values(
    group_key: np.ndarray,
    values: List[np.ndarray],
    n_chunks: int,
):
    key_list = np.array_split(group_key, n_chunks)
    value_list = np.array_split(values, n_chunks)
    return [dict(group_key=k, values=v) for k, v in zip(key_list, value_list)]


# ===== Row Selection Methods =====


@nb.njit
def _find_nth(
    group_key: np.ndarray,
    ngroups: np.ndarray,
    n: int,
    mask: Optional[np.ndarray] = None,
):
    out = np.full(ngroups, -1, dtype=np.int64)
    seen = np.zeros(ngroups, dtype=np.int16)
    masked = mask is not None
    if n >= 0:
        rng = range(len(group_key))
    else:
        rng = range(len(group_key) - 1, -1, -1)
        n = -n - 1

    for i in rng:
        k = group_key[i]
        if k < 0:
            continue
        if masked and not mask[i]:
            continue
        if seen[k] == n:
            assert out[k] == -1
            out[k] = i
        seen[k] += 1

    return out


@nb.njit
def _find_first_or_last_n(
    group_key: np.ndarray,
    ngroups: np.ndarray,
    n: int,
    mask: Optional[np.ndarray] = None,
    forward: bool = True,
):
    out = np.full((ngroups, n), -1, dtype=np.int64)
    seen = np.zeros(ngroups, dtype=np.int16)
    masked = mask is not None
    if forward:
        rng = range(len(group_key))
    else:
        rng = range(len(group_key) - 1, -1, -1)

    for i in rng:
        k = group_key[i]
        if k < 0:
            continue
        if masked and not mask[i]:
            continue
        j = seen[k]
        if j < n:
            out[k, j] = i
            seen[k] += 1

    if not forward:
        out = out[:, ::-1]

    return out


def find_first_n(
    group_key: ArrayType1D,
    ngroups: int,
    n: int,
    mask: Optional[ArrayType1D] = None,
):
    """
    Find the first n indices for each group in group_key.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups.
    ngroups : int
        The number of unique groups in group_key.
    n : int
        The number of indices to find for each group.
    mask : Optional[ArrayType1D]
        A boolean mask to filter the elements before finding indices.

    Returns
    -------
    np.ndarray
        An array of shape (ngroups, n) with the first n indices for each group.
    """
    return _find_first_or_last_n(**locals(), forward=True)


def find_last_n(
    group_key: ArrayType1D,
    ngroups: int,
    n: int,
    mask: Optional[ArrayType1D] = None,
):
    """
    Find the last n indices for each group in group_key.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups.
    ngroups : int
        The number of unique groups in group_key.
    n : int
        The number of indices to find for each group.
    mask : Optional[ArrayType1D]
        A boolean mask to filter the elements before finding indices.

    Returns
    -------
    np.ndarray
        An array of shape (ngroups, n) with the last n indices for each group.
    """
    return _find_first_or_last_n(**locals(), forward=False)


# ===== Group Aggregation Methods =====


class ScalarFuncs:

    @_scalar_func_decorator
    def sum(cur_sum, next_val, count):
        if count:
            return cur_sum + next_val, count + 1
        else:
            return next_val, count + 1

    @_scalar_func_decorator
    def nansum(cur_sum, next_val, count):
        if is_null(next_val):
            return cur_sum, count
        elif count:
            return cur_sum + next_val, count + 1
        else:
            return next_val, count + 1

    @_scalar_func_decorator
    def nansum_squares(cur_sum, next_val, count):
        if is_null(next_val):
            return cur_sum, count
        elif count:
            return cur_sum + next_val**2, count + 1
        else:
            return next_val**2, count + 1

    @_scalar_func_decorator
    def max(cur_max, next_val, count):
        if is_null(next_val):
            return next_val, count
        elif count:
            if next_val > cur_max:
                cur_max = next_val
            return cur_max, count + 1
        else:
            return next_val, count + 1

    @_scalar_func_decorator
    def nanmax(cur_max, next_val, count):
        if is_null(next_val):
            return cur_max, count
        elif count:
            if next_val > cur_max:
                cur_max = next_val
            return cur_max, count + 1
        else:
            return next_val, count + 1

    @_scalar_func_decorator
    def min(cur_max, next_val, count):
        if is_null(next_val):
            return next_val, count
        elif count:
            if next_val < cur_max:
                cur_max = next_val
            return cur_max, count + 1
        else:
            return next_val, count + 1

    @_scalar_func_decorator
    def nanmin(cur_min, next_val, count):
        if is_null(next_val):
            return cur_min, count
        elif count:
            if next_val < cur_min:
                cur_min = next_val
            return cur_min, count + 1
        else:
            return next_val, count + 1

    @_scalar_func_decorator
    def nancount(cur_count, next_val, count):
        if is_null(next_val):
            return count, count
        else:
            new_count = count + 1
            return new_count, new_count

    @_scalar_func_decorator
    def count(cur_size, next_val, count):
        new_count = count + 1
        return new_count, new_count

    @_scalar_func_decorator
    def first(cur_first, next_val, count):
        if is_null(next_val):
            return cur_first, count
        elif count:
            return cur_first, count + 1
        else:
            return next_val, count + 1

    @_scalar_func_decorator
    def last(cur_last, next_val, count):
        if is_null(next_val):
            return cur_last, count + 1
        else:
            return next_val, count + 1


@nb.njit(nogil=True)
def _group_by_reduce(
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    reduce_func: Callable,
    indexer: Optional[np.ndarray] = None,
    check_in_bounds: bool = True,
):
    """
    Core numba-compiled groupby reduction function with optional indexing.

    This is the low-level function that performs the actual groupby reduction operation.
    It iterates through the data and applies a reduction function to accumulate values
    within each group. Supports optional indexing for masked operations.

    Parameters
    ----------
    group_key : np.ndarray
        1D integer array where each element indicates the group index for that row.
        Negative values indicate null/missing groups and are skipped.
    values : np.ndarray
        1D array of values to be aggregated. Must be same length as group_key.
    target : np.ndarray
        Pre-allocated array to store the reduction results for each group.
        Length must equal the number of unique groups.
    reduce_func : Callable
        Numba-compiled reduction function from ScalarFuncs (e.g., ScalarFuncs.nansum).
        Must accept (current_value, new_value, count) and return (updated_value, updated_count).
    indexer : np.ndarray, optional
        Array of indices to process. If None, processes all elements sequentially.
        Used for boolean mask operations converted to integer indices.
    check_in_bounds : bool, default True
        Whether to validate that indexer values are within bounds of the arrays.
        Set to False when indexer is guaranteed to be valid (e.g., from nonzero()).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - target: Array of aggregated values for each group
        - count: Array of counts for each group indicating how many values were processed

    Notes
    -----
    - This function is JIT-compiled with numba for optimal performance
    - Negative group keys are treated as null/missing and skipped
    - The target array is modified in-place
    - Thread-safe when used with different target arrays
    """
    count = np.full(len(target), 0, dtype="int64")
    if indexer is None:
        for i in range(len(group_key)):
            key = group_key[i]
            if key < 0:
                continue
            target[key], count[key] = reduce_func(target[key], values[i], count[key])
    else:
        n_rows = len(group_key)
        for i in indexer:
            if check_in_bounds and i >= n_rows:
                raise ValueError(
                    f"Indexer {i} is out of bounds for array of length {n_rows}"
                )
            key = group_key[i]
            if key < 0:
                continue
            target[key], count[key] = reduce_func(target[key], values[i], count[key])

    return target, count


@check_data_inputs_aligned("group_key", "values")
def _apply_group_method_single_chunk(
    reduce_func_name: str,
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
):
    """
    Apply a single reduction function to a chunk of grouped data.

    This function serves as a high-level wrapper around _group_by_reduce that handles
    input conversion, mask processing, and target array preparation. It's designed to
    work with a single chunk of data and is commonly used in parallel processing scenarios.

    Parameters
    ----------
    reduce_func_name : str
        Name of the reduction function from ScalarFuncs to apply (e.g., 'nansum', 'nanmean').
    group_key : ArrayType1D
        Array of group labels for each value. Can be numpy array, pandas Series, etc.
    values : ArrayType1D
        Array of values to be aggregated. Must be same length as group_key.
    ngroups : int
        Total number of unique groups expected in the complete dataset.
        Used for pre-allocating the target array.
    mask : Optional[ArrayType1D], optional
        Boolean mask or integer indices to filter which elements to process.
        If boolean, True values indicate elements to include.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - target: Array of aggregated values for each group (length=ngroups)
        - count: Array of counts for each group indicating number of values processed

    Notes
    -----
    - Converts input arrays to numpy format automatically
    - Boolean masks are converted to integer indices using nonzero()
    - Target array is pre-allocated with appropriate dtype and initial values
    - Function signature is checked to ensure group_key and values are aligned

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import _apply_group_method_single_chunk
    >>>
    >>> group_key = np.array([0, 1, 0, 1, 2])
    >>> values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> target, count = _apply_group_method_single_chunk(
    ...     'nansum', group_key, values, ngroups=3
    ... )
    >>> print(target)  # [4.0, 6.0, 5.0]
    >>> print(count)   # [2, 2, 1]
    """
    group_key = _val_to_numpy(group_key)
    if mask is not None and mask.dtype.kind == "b":
        if len(mask) != len(group_key):
            raise ValueError("Mask must have the same length as group_key")
        indexer = mask.nonzero()[0]
        check_in_bounds = False
    else:
        indexer = mask
        check_in_bounds = True
    target = _build_target_for_groupby(values.dtype, reduce_func_name, ngroups)
    return _group_by_reduce(
        group_key=group_key,
        values=values,
        target=target,
        indexer=indexer,
        reduce_func=getattr(ScalarFuncs, reduce_func_name),
        check_in_bounds=check_in_bounds,
    )


@nb.njit(parallel=True)
def reduce_array_pair(
    x: np.ndarray, y: np.ndarray, reducer: Callable, counts: Optional[np.ndarray] = None
):
    """
    Apply a reduction function element-wise to pairs of arrays using parallel processing.

    This function takes two arrays of equal length and applies a reduction function
    to each pair of corresponding elements. It's optimized for parallel execution
    using numba's prange for better performance on multi-core systems.

    Parameters
    ----------
    x : np.ndarray
        First input array. Must have the same length as y.
    y : np.ndarray
        Second input array. Must have the same length as x.
    reducer : Callable
        Reduction function from ScalarFuncs (e.g., ScalarFuncs.nansum, ScalarFuncs.nanmax).
        Must accept (value1, value2, count) and return (result, updated_count).
        The count parameter is fixed at 1 for this function.

    Returns
    -------
    np.ndarray
        Array of the same length as inputs containing the element-wise reduction results.
        Only the reduced values are returned (not the counts).

    Notes
    -----
    - This function is JIT-compiled with numba and runs in parallel using prange
    - Both input arrays must have the same length
    - The reducer function is called with count=1 for each pair
    - Used primarily for combining results from chunked parallel operations
    - The function assumes x contains accumulated values that will be combined with y

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import reduce_array_pair, ScalarFuncs
    >>>
    >>> x = np.array([1.0, 3.0, 5.0])
    >>> y = np.array([2.0, 4.0, 6.0])
    >>> result = reduce_array_pair(x, y, ScalarFuncs.nansum)
    >>> print(result)  # [3.0, 7.0, 11.0]
    """
    out = x.copy()
    for i in nb.prange(len(x)):
        if counts is None:
            count = 1
        else:
            count = counts[i]
        out[i] = reducer(x[i], y[i], count=count)[0]
    return out


def combine_chunk_results_for_unfactorized_key(
    reduce_func_name: str,
    chunks: List[np.ndarray],
    labels: List[np.ndarray],
    counts: Optional[List[np.ndarray]] = None,
):
    """
    Combine chunk results for unfactorized (non-continuous integer) group keys.

    This function handles the case where group keys are not factorized (e.g., string
    labels, non-continuous integers). It creates a unified index from all chunk labels
    and combines the corresponding values using the specified reduction function.

    Parameters
    ----------
    reduce_func_name : str
        Name of reduction function from ScalarFuncs (e.g., "nansum", "nanmax").
        Used to combine values across chunks for each group.
    chunks : List[np.ndarray]
        List of result arrays from each chunk, one per chunk processed.
        Each array contains the reduced values for groups present in that chunk.
    labels : List[np.ndarray]
        List of group label arrays corresponding to each chunk.
        These identify which groups are present in each chunk's results.
    counts : Optional[List[np.ndarray]], default None
        List of count arrays for each chunk, used for operations that need to
        track the number of observations per group. If None, counts are not tracked.

    Returns
    -------
    tuple[pd.Series, pd.Series | None, pd.Index]
        combined : pd.Series
            Series with the combined results for all groups, indexed by group labels.
        combined_count : pd.Series | None
            Series with combined counts per group if counts were provided, None otherwise.
        all_labels : pd.Index
            Union of all group labels from all chunks.

    Notes
    -----
    - Used when group keys are not factorized (e.g., string group names)
    - Creates a pandas Index union to handle non-overlapping groups across chunks
    - Uses reduce_array_pair to combine values for groups that appear in multiple chunks
    - More memory intensive than factorized key combination due to pandas operations

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import combine_chunk_results_for_unfactorized_key, ScalarFuncs
    >>>
    >>> chunks = [np.array([10.0, 20.0]), np.array([30.0, 5.0])]
    >>> labels = [np.array(['A', 'B']), np.array(['A', 'C'])]
    >>> combined, combined_count, all_labels = combine_chunk_results_for_unfactorized_key(
    ...     ScalarFuncs.nansum, chunks, labels, None
    ... )
    >>> print(combined)  # A: 40.0, B: 20.0, C: 5.0
    """
    all_labels = reduce(pd.Index.union, map(pd.Index, labels))
    target = _build_target_for_groupby(
        chunks[0].dtype, reduce_func_name, len(all_labels)
    )
    combined = pd.Series(target, index=all_labels)
    if counts is None:
        counts = [None] * len(chunks)
        combined_count = None
    else:
        combined_count = pd.Series(0, index=all_labels)
    for key, chunk, count in zip(labels, chunks, counts):
        combined.loc[key] = reduce_array_pair(
            combined.loc[key].values, chunk, getattr(ScalarFuncs, reduce_func_name)
        )
        if combined_count is not None:
            combined_count.loc[key] = combined_count.loc[key] + count

    return combined, combined_count, all_labels


def combine_chunk_results_for_factorized_key(
    reduce_func_name: str,
    chunks: List[np.ndarray],
    counts: Optional[List[np.ndarray]] = None,
):
    """
    Combine chunk results for factorized (continuous integer) group keys.

    This function handles the case where group keys are factorized (0, 1, 2, ..., n-1).
    Since factorized keys have a known, continuous structure, this function can efficiently
    combine results by directly applying the reduction function element-wise across chunks.

    Parameters
    ----------
    reduce_func_name : str
        Name of reduction function from ScalarFuncs (e.g., "nansum", "nanmax").
        Used to combine values across chunks for each group position.
    chunks : List[np.ndarray]
        List of result arrays from each chunk, one per chunk processed.
        All arrays must have the same length (ngroups), with each position
        corresponding to the same group across all chunks.
    counts : Optional[List[np.ndarray]], default None
        List of count arrays for each chunk, used for operations that need to
        track the number of observations per group. If None, counts are not tracked.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | int]
        combined : np.ndarray
            Array with the combined results for all groups, with the same length as input chunks.
        combined_count : np.ndarray | int
            Combined counts per group if counts were provided, otherwise 0.

    Notes
    -----
    - Used when group keys are factorized (continuous integers 0, 1, 2, ...)
    - More memory efficient than unfactorized key combination due to direct array operations
    - All chunks must have the same length since they represent the same groups
    - Uses reduce_array_pair sequentially to combine chunks pairwise

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import combine_chunk_results_for_factorized_key, ScalarFuncs
    >>>
    >>> chunks = [np.array([10.0, 20.0, 30.0]), np.array([5.0, 15.0, 25.0])]
    >>> combined, combined_count = combine_chunk_results_for_factorized_key(
    ...     ScalarFuncs.nansum, chunks, None
    ... )
    >>> print(combined)  # [15.0, 35.0, 55.0]
    """
    combined = chunks[0]

    if counts is None:
        counts = np.zeros(len(chunks))
        combined_count = 0
    else:
        combined_count = counts[0]

    for chunk, count in zip(chunks[1:], counts[1:]):
        combined = reduce_array_pair(
            combined, chunk, getattr(ScalarFuncs, reduce_func_name)
        )
        combined_count = combined_count + count

    return combined, combined_count


def _group_func_wrap(
    reduce_func_name: str,
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    return_count: bool = False,
):
    """
    Applies a reduction function to grouped data, supporting chunked arrays, optional masking, and multi-threading.

    Parameters
    ----------
    reduce_func_name : str
        Name of the reduction function to apply (e.g., 'sum', 'mean', 'count').
    group_key : ArrayType1D
        Array of group labels for each value.
    values : ArrayType1D
        Array of values to be reduced, possibly chunked.
    ngroups : int
        Number of unique groups.
    mask : Optional[ArrayType1D], optional
        Boolean, fancy index or slice mask to filter values and group keys before reduction.
    n_threads : int, default 1
        Number of threads to use for parallel processing. If 1, runs single-threaded.
    return_count : bool, default False
        If True, also returns the count of values per group.

    Returns
    -------
    out : np.ndarray
        Array of reduced values per group.
    count : np.ndarray, optional
        Array of counts per group, returned if `return_count` is True.

    Notes
    -----
    - Handles chunked arrays and fancy indexing.
    - Supports parallel processing for chunked data.
    - Preserves original dtype for datetime and timedelta types.
    """
    if isinstance(mask, slice):
        # slicing creates views at no cost
        values = values[mask]
        group_key = group_key[mask]
        mask = None

    group_key = _val_to_numpy(group_key)
    values = _val_to_numpy(values, as_list=True)
    values_are_chunked = len(values) > 1

    fancy_indexing = False
    if mask is not None:
        mask = _val_to_numpy(mask)
        if mask.dtype.kind in "ui":
            fancy_indexing = True

    values, orig_types = zip(*list(map(_maybe_cast_timestamp_arr, values)))
    orig_type = orig_types[0]

    if reduce_func_name == "sum_squares":
        values = [v.astype(float) for v in values]

    if values_are_chunked:
        if fancy_indexing:
            # fancy indexer doesn't play nicely with chunking values
            # Unchunk the values and follow this path
            values = np.concatenate(values)
            values_are_chunked = False
    else:
        values = values[0]

    kwargs = dict(
        group_key=group_key,
        values=values,
        ngroups=ngroups,
        mask=mask,
        reduce_func_name=reduce_func_name,
    )
    counting = "count" in reduce_func_name

    if n_threads == 1 and not values_are_chunked:
        result, count = _apply_group_method_single_chunk(**kwargs)
        if counting:
            result = count
    else:
        chunked_args = _chunk_groupby_args(**kwargs, n_chunks=n_threads)
        chunks = parallel_map(
            _apply_group_method_single_chunk, [args.args for args in chunked_args]
        )
        chunks, counts = zip(*chunks)

        if counting:
            chunks = counts

        result, count = combine_chunk_results_for_factorized_key(
            "sum" if counting or "sum" in reduce_func_name else reduce_func_name,
            chunks,
            counts,
        )

    if orig_type.kind in "mM":
        result = result.astype(orig_type)

    if return_count:
        return result, count
    else:
        return result


def group_size(
    group_key: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
):
    """
    Count the number of elements in each group defined by group_key.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups.
    ngroups : int
        The number of unique groups in group_key.
    mask : Optional[ArrayType1D]
        A boolean mask to filter the elements before counting.
    n_threads : int
        Number of threads to use for parallel processing.

    Returns
    -------
    ArrayType1D
        An array with the count of elements in each group.
    """
    return _group_func_wrap("count", values=group_key, **locals())


def group_count(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    return_count: bool = False,
):
    """Count the number of non-null values in each group.
    Parameters
    ----------
    group_key : ArrayType1D
        The array defining the groups.
    values : ArrayType1D
        The array of values to count.
    ngroups : int
        The number of unique groups in `group_key`.
    mask : Optional[ArrayType1D], default None
        A mask array to filter the values. If provided, only non-null values where the mask
        is True will be counted.
    n_threads : int, default 1
        The number of threads to use for parallel processing. If set to 1, the function will run in a single thread.
    Returns
    -------
    ArrayType1D
        An array of counts for each group, where the index corresponds to the group key.
    Notes
    -----
    This function counts the number of non-null values in each group defined by `group_key`.
    If a mask is provided, it will only count the values where the mask is True.
    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import group_count
    >>> group_key = np.array([0, 1, 0, 1, 2, 2])
    >>> values = np.array([1, 2, np.nan, 3, 4, np.nan, 5])
    >>> ngroups = 3
    >>> counts = group_count(group_key, values, ngroups)
    >>> print(counts)
    [2 2 1]
    """
    return _group_func_wrap("nancount", **locals())


def group_sum(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    return_count: bool = False,
):
    if isinstance(values, np.ndarray) and values.dtype.kind in "ui":
        reduce_func_name = "sum"
    else:
        reduce_func_name = "nansum"
    return _group_func_wrap(**locals())


def group_sum_squares(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    return_count: bool = False,
):
    return _group_func_wrap(reduce_func_name="nansum_squares", **locals())


def group_mean(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    return_count: bool = False,
):
    kwargs = locals().copy()
    kwargs["return_count"] = True
    sum_, count = _group_func_wrap("nansum", **kwargs)
    sum_, orig_type = _maybe_cast_timestamp_arr(sum_)
    mean = sum_ / count
    if orig_type.kind in "mM":
        mean = mean.astype(orig_type)
    return (mean, count) if return_count else mean


def group_min(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    return_count: bool = False,
):
    return _group_func_wrap("nanmin", **locals())


def group_max(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    return_count: bool = False,
):
    return _group_func_wrap("nanmax", **locals())


def group_first(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    return_count: bool = False,
):
    return _group_func_wrap("first", **locals())


def group_last(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    return_count: bool = False,
):
    return _group_func_wrap("last", **locals())


def _wrap_numba(nb_func):

    @wraps(nb_func.py_func)
    def wrapper(*args, **kwargs):
        bound_args = inspect.signature(nb_func.py_func).bind(*args, **kwargs)
        args = [np.asarray(x) if np.ndim(x) > 0 else x for x in bound_args.args]
        return nb_func(*args)

    wrapper.__nb_func__ = nb_func

    return wrapper


@check_data_inputs_aligned("group_key", "values")
@_wrap_numba
@nb.njit
def group_nearby_members(
    group_key: np.ndarray, values: np.ndarray, max_diff: float | int, n_groups: int
):
    """
    Given a vector of integers defining groups and an aligned numerical vector, values,
    generate subgroups where the differences between consecutive members of a group are below a threshold.
    For example, group events which are close in time and which belong to the same group defined by the group key.

    group_key: np.ndarray
        Vector defining the initial groups
    values:
        Array of numerical values used to determine closeness of the group members, e.g. an array of timestamps.
        Assumed to be monotonic non-decreasing.
    max_diff: float | int
        The threshold distance for forming a new sub-group
    n_groups: int
        The number of unique groups in group_key
    """
    group_counter = -1
    seen = np.full(n_groups, False)
    last_seen = np.empty(n_groups, dtype=values.dtype)
    group_tracker = np.full(n_groups, -1)
    out = np.full(len(group_key), -1)
    for i in range(len(group_key)):
        key = group_key[i]
        current_value = values[i]
        if not seen[key]:
            seen[key] = True
            make_new_group = True
        else:
            make_new_group = abs(current_value - last_seen[key]) > max_diff

        if make_new_group:
            group_counter += 1
            group_tracker[key] = group_counter

        last_seen[key] = current_value
        out[i] = group_tracker[key]

    return out


# ===== Rolling Aggregation Methods =====


def _apply_rolling(
    operation: str,
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
    allow_downcasting: bool = True,
    **kwargs,
):
    """
    General dispatcher for rolling operations that handles 1D vs 2D cases.

    This function dispatches to the appropriate 1D function or uses the 2D wrapper
    based on the dimensionality of the input values.

    Parameters
    ----------
    operation : str
        Name of the rolling operation ('sum', 'mean', 'min', 'max')
    group_key : ArrayType1D
        1D array defining the groups
    values : ArrayType1D
        Values to aggregate.
    ngroups : int
        Number of unique groups in group_key
    window : int
        Size of the rolling window (constant across all groups)
    min_periods : Optional[int]
        Minimum number of non-null observations in window required to have a value
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements
    n_threads : int, default 1
        Number of threads to use for parallel column processing (2D values only)

    Returns
    -------
    np.ndarray
        Rolling aggregation results with same shape as values

    Raises
    ------
    ValueError
        If operation is not supported or values are not 1D/2D
    """
    # Map operation names to 1D functions
    rolling_1d_funcs = {
        "sum": _rolling_sum_or_mean_1d,
        "mean": _rolling_sum_or_mean_1d,
        "min": _rolling_max_or_min_1d,
        "max": _rolling_max_or_min_1d,
        "shift": _rolling_shift_or_diff_1d,
        "diff": _rolling_shift_or_diff_1d,
    }

    if operation not in rolling_1d_funcs:
        raise ValueError(f"Unsupported rolling operation: {operation}")

    # Convert inputs to appropriate numpy arrays
    group_key = _val_to_numpy(group_key)

    if mask is not None:
        mask = _val_to_numpy(mask)

    rolling_1d_func = rolling_1d_funcs[operation]
    values = _val_to_numpy(values, as_list=True)
    values, orig_dtypes = zip(*list(map(_maybe_cast_timestamp_arr, values)))
    orig_dtype = orig_dtypes[0]
    values_are_times = orig_dtype.kind in "mM"

    null_value = _null_value_for_numpy_type(values[0].dtype)
    if allow_downcasting and not values_are_times:
        null_value = np.nan

    kwargs = kwargs | locals()
    kwargs = {k: kwargs[k] for k in signature(rolling_1d_func).parameters}
    result = rolling_1d_func(**kwargs)

    if values_are_times:
        if operation == "diff":
            result = result.view("m8[ns]")
        else:
            result = result.view(orig_dtype)

    return result


@nb.njit(nogil=True, fastmath=False)
def _rolling_sum_or_mean_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    null_value=np.nan,
    want_mean: bool = False,
):
    """
    Core numba function for rolling sum on 1D values.

    Parameters
    ----------
    group_key : np.ndarray
        1D array defining the groups
    values : np.ndarray
        1D array of values to aggregate
    ngroups : int
        Number of unique groups
    window : int
        Rolling window size (constant across all groups)
    mask : Optional[np.ndarray]
        Boolean mask to filter elements

    Returns
    -------
    np.ndarray
        Rolling sums for each position
    """
    if min_periods is None:
        min_periods = window

    out = np.full(len(group_key), null_value)
    masked = mask is not None

    # Track rolling sums and circular buffers for each group
    group_sums = np.zeros(ngroups)
    group_buffers = np.full((ngroups, window), null_value)
    group_positions = np.zeros(ngroups, dtype=np.int16)
    group_non_null = np.zeros(ngroups, dtype=np.int16)
    group_n_seen = np.zeros(ngroups, dtype=np.int16)
    i = -1

    for arr in values:
        for val in arr:
            i += 1
            key = group_key[i]

            if key < 0:  # Skip null keys
                continue

            if masked and not mask[i]:
                continue

            val_is_null = is_null(val)

            # Get current position in circular buffer for this group
            pos = group_positions[key]

            # If buffer is full, subtract the value that will be replaced
            group_full = group_n_seen[key] >= window
            if group_full:
                old_val = group_buffers[key, pos]
                if not is_null(old_val):
                    group_sums[key] -= old_val
                    group_non_null[key] -= 1

            # Add new value
            if not val_is_null:
                group_non_null[key] += 1
                group_sums[key] += val

            group_buffers[key, pos] = val

            # Update position and count
            group_positions[key] = (pos + 1) % window
            if not group_full:
                group_n_seen[key] += 1

            if group_non_null[key] >= min_periods:
                if want_mean:
                    out[i] = group_sums[key] / group_non_null[key]
                else:
                    out[i] = group_sums[key]

    return out


def rolling_sum(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
    allow_downcasting: bool = True,
    n_threads: int = 1,
):
    """
    Calculate rolling sum within each group using optimized circular buffer approach.

    This function uses an optimized algorithm that maintains running sums and circular
    buffers for O(1) add/remove operations, making it much faster than naive
    implementations that recalculate sums for each window.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups
    values : ArrayType1D or np.ndarray
        Values to aggregate. Can be 1D or 2D (for multiple columns)
    ngroups : int
        Number of unique groups in group_key
    window : int
        Size of the rolling window (constant across all groups)
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements
    n_threads : int, default 1
        Number of threads to use for parallel column processing (2D values only)

    Returns
    -------
    np.ndarray
        Rolling sums with same shape as values

    Notes
    -----
    - Window size is constant across all groups
    - Uses circular buffer with O(1) operations for optimal performance
    - Handles NaN values by skipping them in calculations
    - Supports both 1D and 2D (multi-column) input values

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import rolling_sum
    >>>
    >>> # 1D example
    >>> group_key = np.array([0, 0, 0, 1, 1, 1])
    >>> values = np.array([1, 2, 3, 4, 5, 6])
    >>> result = rolling_sum(group_key, values, ngroups=2, window=2)
    >>> print(result)
    [1. 3. 5. 4. 9. 11.]
    >>>
    >>> # 2D example (multiple columns)
    >>> values_2d = np.array([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60]])
    >>> result_2d = rolling_sum(group_key, values_2d, ngroups=2, window=2)
    >>> print(result_2d)
    [[  1.  10.]
     [  3.  30.]
     [  5.  50.]
     [  4.  40.]
     [  9.  90.]
     [ 11. 110.]]
    """
    return _apply_rolling("sum", want_mean=False, **locals())


def rolling_mean(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
    allow_downcasting: bool = True,
    n_threads: int = 1,
):
    """
    Calculate rolling mean within each group.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups
    values : ArrayType1D or np.ndarray
        Values to aggregate. Can be 1D or 2D (for multiple columns)
    ngroups : int
        Number of unique groups in group_key
    window : int
        Size of the rolling window
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements
    n_threads : int, default 1
        Number of threads to use for parallel column processing (2D values only)

    Returns
    -------
    np.ndarray
        Rolling means with same shape as values
    """
    return _apply_rolling("mean", want_mean=True, **locals())


@nb.njit(nogil=True)
def min_or_max_and_position(arr, want_max: bool = True):
    i = 0
    while is_null(arr[i]) and i < len(arr) - 1:
        i += 1
    best = arr[i]
    best_pos = i
    for j, v in enumerate(arr[i + 1 :], i):
        if want_max and v >= best or (not want_max and v <= best):
            best = v
            best_pos = j

    return best, best_pos


@nb.njit(nogil=True, fastmath=False)
def _rolling_max_or_min_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    null_value=np.nan,
    want_max: bool = True,
):
    """
    Optimized core numba function for rolling max/min on 1D values.

    Uses position tracking to avoid scanning entire window on each update.
    Only recomputes min when the current minimum falls out of the window.
    """
    if min_periods is None:
        min_periods = window

    out = np.full(len(group_key), null_value)
    masked = mask is not None
    want_min = not want_max

    # Track rolling max/min and its position in circular buffers for each group
    current_best = np.full(ngroups, -np.inf if want_max else np.inf)
    pos_of_current_best = np.zeros(ngroups, dtype=np.int16)
    group_buffers = np.full((ngroups, window), null_value)
    group_buffer_pos = np.zeros(ngroups, dtype=np.int16)
    group_non_null = np.zeros(ngroups, dtype=np.int16)
    group_n_seen = np.zeros(ngroups, dtype=np.int16)

    i = -1
    for arr in values:
        for val in arr:
            i += 1
            key = group_key[i]

            if key < 0:  # Skip null keys
                continue

            if masked and not mask[i]:
                continue

            val_is_null = is_null(val)

            # Get current position in circular buffer for this group
            pos = group_buffer_pos[key]
            cur_best = current_best[key]

            need_recalc = pos == pos_of_current_best[key]
            need_recalc = True

            n_seen = group_n_seen[key]
            group_full = n_seen >= window
            if group_full:
                to_remove = group_buffers[key, pos]
                if not is_null(to_remove):
                    group_non_null[key] -= 1

            group_buffers[key, pos] = val
            # Add new value
            if not val_is_null:
                if (
                    group_non_null[key] == 0
                    or (want_max and val >= cur_best)
                    or (want_min and val <= cur_best)
                ):
                    current_best[key] = val
                    pos_of_current_best[key] = pos
                    need_recalc = False
                group_non_null[key] += 1

            if group_full and need_recalc:
                # Recompute max from remaining window
                window_vals = group_buffers[key]
                window_best, pos_of_best = min_or_max_and_position(
                    window_vals, want_max
                )
                current_best[key] = window_best
                pos_of_current_best[key] = (pos_of_best - pos) % window

            # Update position and count
            new_position = (pos + 1) % window
            group_buffer_pos[key] = new_position

            if not group_full:
                group_n_seen[key] += 1

            if group_non_null[key] >= min_periods:
                out[i] = current_best[key]

    return out


def rolling_min(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
    allow_downcasting: bool = True,
    n_threads: int = 1,
):
    """
    Calculate rolling minimum within each group.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups
    values : ArrayType1D or np.ndarray
        Values to aggregate. Can be 1D or 2D (for multiple columns)
    ngroups : int
        Number of unique groups in group_key
    window : int
        Size of the rolling window
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements

    Returns
    -------
    np.ndarray
        Rolling minimums with same shape as values
    """
    return _apply_rolling("min", want_max=False, **locals())


def rolling_max(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
    allow_downcasting: bool = True,
    n_threads: int = 1,
):
    """
    Calculate rolling maximum within each group.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups
    values : ArrayType1D or np.ndarray
        Values to aggregate. Can be 1D or 2D (for multiple columns)
    ngroups : int
        Number of unique groups in group_key
    window : int
        Size of the rolling window
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements

    Returns
    -------
    np.ndarray
        Rolling maximums with same shape as values
    """
    return _apply_rolling("max", want_max=True, **locals())


@nb.njit(nogil=True, fastmath=False)
def _rolling_shift_or_diff_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    mask: Optional[np.ndarray] = None,
    null_value: float | int = np.nan,
    want_shift: bool = True,
):
    """
    Optimized core numba function for rolling max/min on 1D values.

    Uses position tracking to avoid scanning entire window on each update.
    Only recomputes min when the current minimum falls out of the window.
    """
    out = np.full(len(group_key), null_value)
    masked = mask is not None

    # Track rolling sums and circular buffers for each group
    group_buffers = np.full((ngroups, window), null_value)
    group_buffer_pos = np.zeros(ngroups, dtype=np.int16)
    group_counts = np.zeros(ngroups, dtype=np.int16)

    i = -1
    for arr in values:
        for val in arr:
            i += 1
            key = group_key[i]

            if key < 0:  # Skip null keys
                continue

            if masked and not mask[i]:
                continue

            # Get current position in circular buffer for this group and add new val
            pos = group_buffer_pos[key]
            if group_counts[key] >= window:
                if want_shift:
                    out[i] = group_buffers[key, pos]
                else:
                    out[i] = val - group_buffers[key, pos]
            else:
                group_counts[key] += 1

            group_buffers[key, pos] = val
            # Update position
            group_buffer_pos[key] = (pos + 1) % window

    return out


def rolling_shift(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    mask: Optional[np.ndarray] = None,
    allow_downcasting: bool = True,
):
    return _apply_rolling("shift", want_shift=True, **locals())


def rolling_diff(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    mask: Optional[np.ndarray] = None,
    allow_downcasting: bool = True,
):
    return _apply_rolling("diff", want_shift=False, **locals())


# ================================
# Cumulative Aggregation Functions
# ================================


@nb.njit(nogil=True, fastmath=False)
def _cumulative_reduce(
    group_key: np.ndarray,
    values: np.ndarray,
    reduce_func: Callable,
    ngroups: int,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
):
    """
    Core numba function for cumulative aggregations within groups.

    This function iterates through data and maintains running aggregated values
    for each group, outputting the cumulative result at each position.

    Parameters
    ----------
    group_key : np.ndarray
        1D array defining the groups
    values : np.ndarray
        1D array of values to aggregate
    reduce_func : callable
        Numba-compiled reduction function (e.g., NumbaReductionOps.sum)
    ngroups : int
        Number of unique groups in group_key
    mask : Optional[np.ndarray]
        Boolean mask to filter elements

    Returns
    -------
    np.ndarray
        Cumulative aggregated values with same shape as input values
    """
    masked = mask is not None
    # Track current state for each group
    group_last_seen = np.full(ngroups, -1)
    group_count = np.zeros(ngroups, dtype="uint32")
    i = -1
    has_null_key = False

    for arr in values:
        for val in arr:
            i += 1
            key = group_key[i]

            if key < 0:
                has_null_key = True
                continue

            last_seen = group_last_seen[key]
            if masked and not mask[i]:
                # For masked values, pass through the current accumulator without updating
                if last_seen >= 0:
                    target[i] = target[last_seen]
                continue

            target[i], group_count[key] = reduce_func(
                target[last_seen], val, group_count[key]
            )
            group_last_seen[key] = i

    return target, has_null_key


def _apply_cumulative(
    operation: str,
    group_key: ArrayType1D,
    values: ArrayType1D | None,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    skip_na: bool = True,
    use_py_func: bool = False,
):
    """
    General dispatcher for cumulative operations.

    Parameters
    ----------
    operation : str
        Name of the cumulative operation ('sum', 'count', 'min', 'max')
    group_key : ArrayType1D
        1D array defining the groups
    values : ArrayType1D or None
        Values to aggregate. Can be None for count operations.
    ngroups : int
        Number of unique groups in group_key
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements
    skip_na : bool, default True
        Whether to skip NaN values in aggregation
    use_py_func: bool
        Do not use the JIT-compiled function. For debugging purposes.

    Returns
    -------
    np.ndarray
        Cumulative aggregation results with appropriate dtype

    Raises
    ------
    ValueError
        If operation is not supported
    """
    # Convert inputs to appropriate numpy arrays
    group_key = _val_to_numpy(group_key)

    if mask is not None:
        mask = _val_to_numpy(mask)

    # Map operation names to reduction functions
    try:
        name = "nan" + operation if skip_na else operation
        reduce_func = getattr(ScalarFuncs, name)
    except AttributeError:
        raise ValueError(f"Unsupported cumulative operation: {name}")

    if values is None:
        raise ValueError(f"values cannot be None for operation '{operation}'")

    counting = "count" in operation

    values = _val_to_numpy(values, as_list=True)
    values, orig_dtypes = zip(*list(map(_maybe_cast_timestamp_arr, values)))
    orig_dtype = orig_dtypes[0]

    target = _build_target_for_groupby(
        values[0].dtype, "sum" if counting else operation, len(group_key)
    )
    func = _cumulative_reduce.py_func if use_py_func else _cumulative_reduce
    result, has_null_keys = func(
        group_key=group_key,
        values=values,
        reduce_func=reduce_func,
        ngroups=ngroups,
        mask=mask,
        target=target,
    )
    if has_null_keys:
        if "count" in operation:
            na_rep = 0
        else:
            na_rep = _null_value_for_numpy_type(result.dtype)
        result[np.asarray(group_key) < 0] = na_rep

    elif orig_dtype.kind in "mM":
        result = result.astype(orig_dtype)

    return result


def cumsum(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    skip_na: bool = True,
):
    """
    Calculate cumulative sum within each group.

    For each group defined by group_key, this function returns the running sum
    of values up to each position. The cumulative sum resets at the beginning
    of each new group.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups
    values : ArrayType1D
        Values to calculate cumulative sum for
    ngroups : int
        Number of unique groups in group_key
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements before aggregation
    skip_na : bool, default True
        Whether to skip NaN values in the sum calculation

    Returns
    -------
    np.ndarray
        Cumulative sums with same shape as input values

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import cumsum
    >>>
    >>> # Basic usage
    >>> group_key = np.array([0, 0, 0, 1, 1, 1])
    >>> values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    >>> result = cumsum(group_key, values, ngroups=2)
    >>> print(result)
    [1. 3. 6. 4. 9. 15.]

    >>> # With NaN values (skip_na=True)
    >>> values_with_nan = np.array([1.0, np.nan, 3.0, 4.0, 5.0, np.nan])
    >>> result = cumsum(group_key, values_with_nan, ngroups=2)
    >>> print(result)
    [1. 1. 4. 4. 9. 9.]
    """
    return _apply_cumulative("sum", **locals())


def cumcount(
    group_key: ArrayType1D,
    values: Optional[ArrayType1D],
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
):
    """
    Calculate cumulative count within each group.

    For each group defined by group_key, this function returns the running count
    of observations up to each position. The count resets at the beginning of
    each new group and starts from 0 (like pandas cumcount).

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups
    ngroups : int
        Number of unique groups in group_key
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements before counting

    Returns
    -------
    np.ndarray
        Cumulative counts with same shape as input, dtype int64

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import cumcount
    >>>
    >>> # Basic usage
    >>> group_key = np.array([0, 0, 0, 1, 1, 1])
    >>> result = cumcount(group_key, ngroups=2)
    >>> print(result)
    [0 1 2 0 1 2]

    >>> # With mask
    >>> mask = np.array([True, False, True, True, True, False])
    >>> result = cumcount(group_key, ngroups=2, mask=mask)
    >>> print(result)
    [1 0 2 1 2 0]
    """
    if values is None:
        values = group_key
    return (
        _apply_cumulative(
            "count",
            **locals(),
        )
        - 1
    )  # Adjust to start from 0 like pandas


def cummin(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    skip_na: bool = True,
):
    """
    Calculate cumulative minimum within each group.

    For each group defined by group_key, this function returns the running minimum
    of values up to each position. The cumulative minimum resets at the beginning
    of each new group.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups
    values : ArrayType1D
        Values to calculate cumulative minimum for
    ngroups : int
        Number of unique groups in group_key
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements before aggregation
    skip_na : bool, default True
        Whether to skip NaN values in the minimum calculation

    Returns
    -------
    np.ndarray
        Cumulative minimums with same shape as input values

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import cummin
    >>>
    >>> # Basic usage
    >>> group_key = np.array([0, 0, 0, 1, 1, 1])
    >>> values = np.array([3.0, 1.0, 4.0, 2.0, 5.0, 1.0])
    >>> result = cummin(group_key, values, ngroups=2)
    >>> print(result)
    [3. 1. 1. 2. 2. 1.]
    """
    return _apply_cumulative("min", group_key, values, ngroups, mask, skip_na)


def cummax(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    skip_na: bool = True,
):
    """
    Calculate cumulative maximum within each group.

    For each group defined by group_key, this function returns the running maximum
    of values up to each position. The cumulative maximum resets at the beginning
    of each new group.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups
    values : ArrayType1D
        Values to calculate cumulative maximum for
    ngroups : int
        Number of unique groups in group_key
    mask : Optional[ArrayType1D]
        Boolean mask to filter elements before aggregation
    skip_na : bool, default True
        Whether to skip NaN values in the maximum calculation

    Returns
    -------
    np.ndarray
        Cumulative maximums with same shape as input values

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import cummax
    >>>
    >>> # Basic usage
    >>> group_key = np.array([0, 0, 0, 1, 1, 1])
    >>> values = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 5.0])
    >>> result = cummax(group_key, values, ngroups=2)
    >>> print(result)
    [1. 3. 3. 4. 4. 5.]
    """
    return _apply_cumulative("max", group_key, values, ngroups, mask, skip_na)


@nb.njit(nogil=True, fastmath=False)
def _build_groups_mapping(group_ikey: np.ndarray, ngroups: int):
    """
    Build groups mapping efficiently in a single pass.

    This function creates a mapping from group indices to arrays of row
    positions where each group occurs. It's optimized for performance with
    large datasets.

    Parameters
    ----------
    group_ikey : np.ndarray
        Integer array where each element indicates group index for that row
    ngroups : int
        Number of unique groups

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        - group_starts: Array of starting positions for each group in indices
        - indices: Flattened array of all row indices sorted by group
    """
    # Count how many items are in each group
    group_counts = np.zeros(ngroups, dtype=np.int64)
    for i in range(len(group_ikey)):
        if group_ikey[i] >= 0:  # Skip null groups (negative indices)
            group_counts[group_ikey[i]] += 1

    # Calculate starting positions for each group in the output array
    group_starts = np.zeros(ngroups + 1, dtype=np.int64)
    for i in range(ngroups):
        group_starts[i + 1] = group_starts[i] + group_counts[i]

    # Create output array to hold all indices
    total_valid = group_starts[ngroups]
    indices = np.empty(total_valid, dtype=np.int64)

    # Track current position for each group while filling
    current_pos = group_starts[:-1].copy()

    # Fill the indices array
    for row_idx in range(len(group_ikey)):
        group_idx = group_ikey[row_idx]
        if group_idx >= 0:  # Skip null groups
            pos = current_pos[group_idx]
            indices[pos] = row_idx
            current_pos[group_idx] += 1

    return group_starts, indices


def build_groups_dict_optimized(group_ikey: np.ndarray, result_index, ngroups: int):
    """
    Build groups dictionary using optimized numba implementation.

    This function creates a dictionary mapping group labels to arrays of row
    indices where each group occurs. It uses an optimized single-pass algorithm
    that's much faster than the original implementation for large datasets.

    Parameters
    ----------
    group_ikey : np.ndarray
        Integer array where each element indicates group index for that row
    result_index : pd.Index
        Index containing the unique group labels
    ngroups : int
        Number of unique groups

    Returns
    -------
    dict
        Dictionary with group labels as keys and numpy arrays of row indices
        as values
    """
    # Use numba-optimized function to get the mapping
    group_starts, indices = _build_groups_mapping(group_ikey, ngroups)
    group_indices = np.array_split(indices, group_starts[1:-1])

    # Build the final dictionary
    groups_dict = {
        key: idx for key, idx in zip(result_index, group_indices) if len(idx) > 0
    }
    return groups_dict
