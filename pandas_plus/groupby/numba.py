import inspect
from inspect import signature
from typing import Callable, Optional, List, Tuple
from functools import wraps

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
)
from .. import nanops


# ===== Array Preparation Methods =====


def _val_to_numpy(
    val: ArrayType1D, as_list: bool = False
) -> Tuple[np.ndarray | NumbaList[np.ndarray], np.dtype]:
    """
    Convert various array types to numpy array.

    Parameters
    ----------
    val : ArrayType1D
        Input array to convert (numpy array, pandas Series, polars Series, etc.)

    Returns
    -------
    Tuple[np.ndarray | NumbaList[np.ndarray], np.dtype]
        NumPy array representation of the input, as a list of arrays or a single array,
        along with the original type if casting timestamps to ints
    """

    if isinstance(val, pl.Series):
        arrow: pa.Array = val.to_arrow()
    elif isinstance(val, pd.Series) and "pyarrow" in str(val.dtype):
        arrow: pa.Array = pa.Array.from_pandas(val)  # type: ignore
    else:
        arrow = None

    chunked = isinstance(
        arrow,
        pa.ChunkedArray,
    )
    if chunked:
        val_list = [chunk.to_numpy() for chunk in arrow.chunks]
    elif hasattr(val, "to_numpy"):
        val_list = [val.to_numpy()]  # type: ignore
    else:
        val_list = [np.asarray(val)]

    val_list, orig_types = zip(*list(map(_maybe_cast_timestamp_arr, val_list)))
    orig_type = orig_types[0]

    if as_list:
        return NumbaList(val_list), orig_type
    else:
        if len(val_list) > 1:
            val = np.concatenate(val_list)
        else:
            val = val_list[0]
        return val, orig_type


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


@check_data_inputs_aligned("group_key", "mask")
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
    indexer: np.ndarray = None,
    check_in_bounds: bool = True,
):
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
            if check_in_bounds and i > n_rows:
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
    group_key = _val_to_numpy(group_key)[0]
    if mask is not None and mask.dtype.kind == "b":
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
    )


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
        Boolean or slice mask to filter values and group keys before reduction.
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

    group_key = _val_to_numpy(group_key)[0]
    values, orig_type = _val_to_numpy(values, as_list=True)
    values_are_chunked = len(values) > 1

    fancy_indexing = False
    if mask is not None:
        mask = _val_to_numpy(mask)[0]
        if mask.dtype.kind in "ui":
            fancy_indexing = True

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
        out, count = _apply_group_method_single_chunk(**kwargs)
        if counting:
            out = count
    else:
        chunked_args = _chunk_groupby_args(**kwargs, n_chunks=n_threads)
        chunks = parallel_map(
            _apply_group_method_single_chunk, [args.args for args in chunked_args]
        )
        chunks, counts = zip(*chunks)
        if counting:
            chunks = counts
        arr = np.vstack(chunks)
        chunk_reduce = "sum" if counting else reduce_func_name.replace("nan", "")
        out = nanops.reduce_2d(chunk_reduce, arr)
        count = sum(counts)

    if orig_type.kind in "mM":
        out = out.astype(orig_type)

    if return_count:
        return out, count
    else:
        return out


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
    group_key = _val_to_numpy(group_key)[0]

    if mask is not None:
        mask = _val_to_numpy(mask)[0]

    rolling_1d_func = rolling_1d_funcs[operation]
    values, orig_dtype = _val_to_numpy(values, as_list=True)
    values_are_times = orig_dtype.kind in "mM"
    null_value = _null_value_for_numpy_type(values[0].dtype)
    if allow_downcasting and not values_are_times:
        null_value = np.nan

    kwargs = kwargs | locals()
    kwargs = {k: kwargs[k] for k in signature(rolling_1d_func).parameters}
    result = rolling_1d_func(**kwargs)

    if orig_dtype.kind in "mM":
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
    group_key = _val_to_numpy(group_key)[0]

    if mask is not None:
        mask = _val_to_numpy(mask)[0]

    # Map operation names to reduction functions
    try:
        name = "nan" + operation if skip_na else operation
        reduce_func = getattr(ScalarFuncs, name)
    except AttributeError:
        raise ValueError(f"Unsupported cumulative operation: {name}")

    if values is None:
        raise ValueError(f"values cannot be None for operation '{operation}'")

    counting = "count" in operation

    values, orig_type = _val_to_numpy(values, as_list=True)
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

    elif orig_type.kind in "mM":
        result = result.astype(orig_type)

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
