import inspect
from inspect import signature
from typing import Callable, Optional, List, Literal
from functools import wraps
from copy import deepcopy

import numba as nb
import numpy as np
import pandas as pd
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


def _array_split_for_lists(arr_list, n_chunks):
    """
    Split a list of arrays or a single array into approximately equal chunks.

    This function is optimized for handling chunked PyArrow arrays backed by
    pandas Series. It distributes the total elements across chunks while
    maintaining array boundaries when possible.

    Parameters
    ----------
    arr_list : list of array-like or np.ndarray
        List of arrays to split, or a single numpy array. Each array can be
        numpy arrays, pandas Series chunks, or other array-like objects.
    n_chunks : int
        Number of chunks to split the data into. Must be positive.

    Returns
    -------
    list of list or list of np.ndarray
        If input is np.ndarray: list of numpy array chunks
        If input is list: list of lists, where each sub-list contains
        array segments that together form one chunk

    Raises
    ------
    ValueError
        If n_chunks is not a positive integer

    Notes
    -----
    - For single numpy arrays, delegates to np.array_split for optimal performance
    - For lists of arrays, splits by total element count, not by number of arrays
    - Attempts to create chunks of approximately equal total length
    - May split individual arrays across chunk boundaries if needed
    - Designed specifically for PyArrow chunked array optimization

    Examples
    --------
    >>> import numpy as np
    >>> from pandas_plus.groupby.numba import array_split_for_lists

    Single array (delegates to numpy):
    >>> arr = np.array([1, 2, 3, 4, 5, 6])
    >>> chunks = array_split_for_lists(arr, 3)
    >>> [chunk.tolist() for chunk in chunks]
    [[1, 2], [3, 4], [5, 6]]

    List of arrays:
    >>> arrays = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7, 8, 9])]
    >>> chunks = array_split_for_lists(arrays, 2)  # Total length 9, ~4-5 per chunk
    >>> # Result: chunks with approximately equal total lengths

    PyArrow chunked array use case:
    >>> # When pandas Series backed by chunked PyArrow arrays are converted
    >>> chunked_arrays = [chunk1, chunk2, chunk3]  # From PyArrow ChunkedArray
    >>> parallel_chunks = array_split_for_lists(chunked_arrays, 4)
    """
    if not isinstance(n_chunks, int) or n_chunks <= 0:
        raise ValueError(f"n_chunks must be a positive integer, got {n_chunks}")

    if isinstance(arr_list, np.ndarray):
        return np.array_split(arr_list, n_chunks)

    lengths = [len(a) for a in arr_list]
    total = sum(lengths)

    # Handle edge case of empty arrays
    if total == 0:
        return [[] for _ in range(n_chunks)]

    chunk_len = int(np.ceil(total / n_chunks))
    chunks = [[]]

    for arr in arr_list:
        remainder = arr
        while len(remainder):
            space = chunk_len - sum(map(len, chunks[-1]))
            if not space:
                chunks.append([])
                continue
            chunks[-1].append(remainder[:space])
            remainder = remainder[space:]

    return chunks


def _val_to_numpy(
    val: ArrayType1D, as_list: bool = False
) -> np.ndarray | NumbaList[np.ndarray]:
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
    if isinstance(val, pd.Series) and "pyarrow" in str(val.dtype):
        val = pa.Array.from_pandas(val)  # type: ignore
        chunked = isinstance(
            val,
            pa.ChunkedArray,
        )
        if chunked and as_list:
            return NumbaList([chunk.to_numpy() for chunk in val.chunks])

    try:
        val = val.to_numpy()  # type: ignore
    except AttributeError:
        val = np.asarray(val)

    if as_list:
        return NumbaList([val])
    else:
        return val


def _build_target_for_groupby(np_type, operation: str, shape):
    if operation == "count":
        target = np.zeros(shape, dtype=np.int64)
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
    group_key: np.ndarray,
    values: List[np.ndarray] | np.ndarray | None,
    target: np.ndarray,
    reduce_func: Optional[Callable] = None,
    mask: Optional[np.ndarray] = None,
):
    if values is not None:
        if sum(len(arr) for arr in values) != len(group_key):
            raise ValueError(
                "Length of group_key must match total length of all arrays in values"
            )
    kwargs = locals().copy()
    del kwargs["n_chunks"]
    shared_kwargs = {"target": target}
    if reduce_func is None:
        iterator = _group_by_counter
    else:
        iterator = _group_by_reduce
        shared_kwargs["reduce_func"] = reduce_func

    chunked_kwargs = [deepcopy(shared_kwargs) for i in range(n_chunks)]
    for name in ["group_key", "values", "mask"]:
        if kwargs[name] is None:
            chunks = [None] * n_chunks
        else:
            chunks = _array_split_for_lists(kwargs[name], n_chunks)
        for chunk_no, arr in enumerate(chunks):
            chunked_kwargs[chunk_no][name] = arr

    chunked_args = [signature(iterator).bind(**kwargs) for kwargs in chunked_kwargs]

    return chunked_args


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
    return _find_first_or_last_n(group_key, ngroups, n, mask, forward=True)


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
    return _find_first_or_last_n(group_key, ngroups, n, mask, forward=False)



# ===== Group Aggregation Methods =====


class ScalarFuncs:

    @_scalar_func_decorator
    def nansum(cur_sum, next_val, seen):
        if is_null(next_val):
            return cur_sum, seen
        elif seen:
            return cur_sum + next_val, True
        else:
            return next_val, True

    @_scalar_func_decorator
    def nanmax(cur_max, next_val, seen):
        if is_null(next_val):
            return cur_max, seen
        elif seen:
            if next_val > cur_max:
                cur_max = next_val
            return cur_max, True
        else:
            return next_val, True

    @_scalar_func_decorator
    def nanmin(cur_min, next_val, seen):
        if is_null(next_val):
            return cur_min, seen
        elif seen:
            if next_val < cur_min:
                cur_min = next_val
            return cur_min, True
        else:
            return next_val, True

    @_scalar_func_decorator
    def count(cur_count, next_val, seen):
        if is_null(next_val):
            return cur_count, seen
        elif seen:
            return cur_count + 1, True
        else:
            return 1, True

    @_scalar_func_decorator
    def size(cur_size, next_val, seen):
        if seen:
            return cur_size + 1, True
        else:
            return 1, True

    @_scalar_func_decorator
    def first(cur_first, next_val, seen):
        if is_null(next_val):
            return cur_first, seen
        elif seen:
            return cur_first, True
        else:
            return next_val, True

    @_scalar_func_decorator
    def last(cur_last, next_val, seen):
        if is_null(next_val):
            return cur_last, seen
        else:
            return next_val, True


@check_data_inputs_aligned("group_key", "values", "mask")
def _group_func_wrap(
    reduce_func_name: str | None,
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    initial_value: Optional[int | float] = None,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
):
    group_key = np.asarray(group_key)
    if mask is not None:
        mask = np.asarray(mask)

    if reduce_func_name is None:
        initial_value = 0

    if values is not None:
        values = np.asarray(values)
        values, orig_type = _maybe_cast_timestamp_arr(values)
        if initial_value is None:
            initial_value = _default_initial_value_for_type(values)

    target = np.full(ngroups, initial_value)

    if reduce_func_name is None:
        iterator = _group_by_counter
    else:
        iterator = _group_by_reduce

    kwargs = dict(
        group_key=group_key,
        values=values,
        target=target,
        mask=mask,
    )
    if reduce_func_name is not None:
        kwargs["reduce_func"] = getattr(NumbaReductionOps, reduce_func_name)

    if n_threads == 1:
        out = iterator(**kwargs)
    else:
        chunked_args = _chunk_groupby_args(**kwargs, n_chunks=n_threads)
        chunks = parallel_map(iterator, [args.args for args in chunked_args])
        arr = np.vstack(chunks)
        chunk_reduce = "sum" if reduce_func_name is None else reduce_func_name
        out = nanops.reduce_2d(chunk_reduce, arr)

    if reduce_func_name is None:
        out = out.astype(np.int64)
    elif orig_type.kind in "mM":
        out = out.astype(orig_type)

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
    return _group_func_wrap(reduce_func_name=None, values=None, **locals())


def group_count(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
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
    return _group_func_wrap(reduce_func_name=None, **locals())


def group_sum(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
):
    if values.dtype.kind == "f":
        initial_value = 0.0
    else:
        initial_value = 0
    return _group_func_wrap("sum", **locals())


def group_mean(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
):
    kwargs = locals().copy()
    sum = group_sum(**kwargs)
    int_sum, orig_type = _maybe_cast_timestamp_arr(sum)
    count = group_count(**kwargs)
    mean = int_sum / count
    if values.dtype.kind in "mM":
        mean = mean.astype(orig_type)
    return mean


def group_min(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
):
    return _group_func_wrap("min", **locals())


def group_max(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
):
    return _group_func_wrap("max", **locals())


class NumbaGroupByMethods:

    @staticmethod
    @check_data_inputs_aligned("group_key", "values", "mask")
    def first(
        group_key: ArrayType1D,
        values: ArrayType1D,
        ngroups: int,
        mask: Optional[ArrayType1D] = None,
        skip_na=True,
    ):
        if skip_na:
            reduce_func_name = "first_skipna"
        else:
            reduce_func_name = "first"
        del skip_na
        return _group_func_wrap(**locals())

    @staticmethod
    @check_data_inputs_aligned("group_key", "values", "mask")
    def last(
        group_key: ArrayType1D,
        values: ArrayType1D,
        ngroups: int,
        mask: Optional[ArrayType1D] = None,
        skip_na=True,
    ):
        if skip_na:
            reduce_func_name = "last_skipna"
        else:
            reduce_func_name = "last"
        del skip_na
        return _group_func_wrap(**locals())


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


@check_data_inputs_aligned("group_key", "values")
def group_diff(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
):
    """
    Calculate the difference between consecutive elements in each group defined by group_key.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups.
    values : ArrayType1D
        1D array of values to calculate differences for.
    ngroups : int
        The number of unique groups in group_key.
    null_value : float | int, default np.nan
        The value to use for nulls in the output.
    mask : Optional[ArrayType1D]
        A boolean mask to filter the elements before calculating differences.

    Returns
    -------
    np.ndarray
        An array with the differences between consecutive elements in each group.
    """
    null_value = _null_value_for_array_type(values)
    if values.dtype.kind in "iu":
        null_value = np.nan
    elif values.dtype.kind == "M":
        null_value = np.timedelta64("NaT", "ns")
    return _group_diff_or_shift(**locals(), shift=False)


@check_data_inputs_aligned("group_key", "values")
def group_shift(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
):
    """
    Shift the values in each group defined by group_key.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups.
    values : ArrayType1D
        1D array of values to shift.
    ngroups : int
        The number of unique groups in group_key.
    mask : Optional[ArrayType1D]
        A boolean mask to filter the elements before shifting.

    Returns
    -------
    np.ndarray
        An array with the shifted values in each group.
    """
    if values.dtype.kind in "iu":
        null_value = np.nan
    null_value = _null_value_for_array_type(values)
    return _group_diff_or_shift(**locals(), shift=True)


@_wrap_numba
@nb.njit
def _group_diff_or_shift(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    null_value: float | int,
    mask: Optional[ArrayType1D] = None,
    shift: bool = False,
):
    """
    Calculate the difference between consecutive elements in each group.

    Parameters
    ----------
    group_key : ArrayType1D
        1D array defining the groups.
    values : ArrayType1D
        1D array of values to calculate differences for.
    ngroups : int
        The number of unique groups in group_key.
    mask : Optional[ArrayType1D]
        A boolean mask to filter the elements before calculating differences.
    n_threads : int
        Number of threads to use for parallel processing.

    Returns
    -------
    np.ndarray
        An array with the differences between consecutive elements in each group.
    """
    previous = np.full(ngroups, null_value)
    seen = np.full(ngroups, False)
    out = np.full(len(group_key), null_value)
    for i in range(len(group_key)):
        key = group_key[i]
        if mask is not None and not mask[i]:
            continue
        if seen[key]:
            seen[key] = True
        elif shift:
            out[i] = previous[key]
        else:
            out[i] = values[i] - previous[key]

        previous[key] = values[i]
    return out


# ===== Rolling Aggregation Methods =====


def _apply_rolling_1d_to_2d(
    rolling_1d_func: Callable,
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    n_threads: int = 1,
):
    """
    General wrapper for applying a 1D rolling function to 2D values.

    This function takes any 1D rolling function and applies it column-wise
    to 2D input values, with optional parallel processing.

    Parameters
    ----------
    rolling_1d_func : callable
        The 1D rolling function to apply to each column
    group_key : np.ndarray
        1D array defining the groups
    values : np.ndarray
        2D array of values to aggregate (rows x columns)
    ngroups : int
        Number of unique groups
    window : int
        Rolling window size (constant across all groups)
    min_periods : Optional[int]
        Minimum number of non-null observations in window required to have a value
    mask : Optional[np.ndarray]
        Boolean mask to filter elements
    n_threads : int
        Number of threads to use for parallel column processing

    Returns
    -------
    np.ndarray
        Results for each position and column
    """
    n_rows, n_cols = values.shape
    kwargs = dict(
        group_key=group_key,
        ngroups=ngroups,
        window=window,
        min_periods=min_periods,
        mask=mask,
    )

    if n_threads == 1 or n_cols == 1:
        # Single-threaded: process columns sequentially
        result = np.empty((n_rows, n_cols))
        for col in range(n_cols):
            result[:, col] = rolling_1d_func(**kwargs, values=values[:, col])
        return result
    else:
        # Multi-threaded: process columns in parallel
        def process_column(col_idx):
            return rolling_1d_func(**kwargs, values=values[:, col_idx])

        # Use parallel_map to process columns in parallel
        column_results = parallel_map(
            process_column, [(i,) for i in range(n_cols)], max_workers=n_threads
        )

        # Combine results into 2D array
        result = np.column_stack(column_results)
        return result


def _apply_rolling(
    operation: str,
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
    n_threads: int = 1,
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
    values : ArrayType1D or np.ndarray
        Values to aggregate. Can be 1D or 2D (for multiple columns)
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
    kwargs = locals().copy()
    del kwargs["operation"]
    # Map operation names to 1D functions
    rolling_1d_funcs = {
        "sum": _rolling_sum_1d,
        "mean": _rolling_mean_1d,
        "min": _rolling_min_1d,
        "max": _rolling_max_1d,
    }

    if operation not in rolling_1d_funcs:
        raise ValueError(f"Unsupported rolling operation: {operation}")

    # Convert inputs to appropriate numpy arrays
    group_key = np.asarray(group_key, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    rolling_1d_func = rolling_1d_funcs[operation]

    if values.ndim == 1:
        del kwargs["n_threads"]
        return rolling_1d_func(**kwargs)
    elif values.ndim == 2:
        return _apply_rolling_1d_to_2d(rolling_1d_func, **kwargs)
    else:
        raise ValueError(f"values must be 1D or 2D, got {values.ndim}D")


@nb.njit(nogil=True, fastmath=False)
def _rolling_sum_or_mean_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    mean: bool = False,
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

    out = np.full(len(values), np.nan)
    masked = mask is not None

    # Track rolling sums and circular buffers for each group
    group_sums = np.zeros(ngroups)
    group_buffers = np.full((ngroups, window), np.nan)
    group_positions = np.zeros(ngroups, dtype=np.int64)
    group_non_null = np.zeros(ngroups, dtype=np.int64)
    group_n_seen = np.zeros(ngroups, dtype=np.int64)

    for i in range(len(group_key)):
        key = group_key[i]

        if key < 0:  # Skip null keys
            continue

        if masked and not mask[i]:
            continue

        val = values[i]
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
            if mean:
                out[i] = group_sums[key] / group_non_null[key]
            else:
                out[i] = group_sums[key]

    return out


def _rolling_sum_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
):
    return _rolling_sum_or_mean_1d(
        **locals(),
        mean=False,
    )


def _rolling_mean_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
):
    return _rolling_sum_or_mean_1d(**locals(), mean=True)


def rolling_sum(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
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
    return _apply_rolling("sum", **locals())


def rolling_mean(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
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
    return _apply_rolling("mean", **locals())


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
def _rolling_max_min_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
    want_max: bool = True,
):
    """
    Optimized core numba function for rolling max/min on 1D values.

    Uses position tracking to avoid scanning entire window on each update.
    Only recomputes min when the current minimum falls out of the window.
    """
    if min_periods is None:
        min_periods = window

    out = np.full(len(values), np.nan)
    masked = mask is not None

    # Track rolling sums and circular buffers for each group
    group_current = np.full(ngroups, 0.0)
    group_cur_positions = np.zeros(ngroups, dtype=np.int64)
    group_buffers = np.full((ngroups, window), np.nan)
    group_buffer_pos = np.zeros(ngroups, dtype=np.int64)
    group_non_null = np.zeros(ngroups, dtype=np.int64)
    group_n_seen = np.zeros(ngroups, dtype=np.int64)

    for i in range(len(group_key)):
        key = group_key[i]

        if key < 0:  # Skip null keys
            continue

        if masked and not mask[i]:
            continue

        val = values[i]
        val_is_null = is_null(val)

        # Get current position in circular buffer for this group
        pos = group_buffer_pos[key]
        cur_pos = group_cur_positions[key]
        cur_best = group_current[key]

        need_recalc = pos == cur_pos

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
                or (not want_max and val <= cur_best)
            ):
                group_current[key] = val
                group_cur_positions[key] = pos
                need_recalc = False
            group_non_null[key] += 1

        if group_full and need_recalc:
            # Recompute max from remaining window
            window_vals = group_buffers[key]
            window_best, cur_pos = min_or_max_and_position(window_vals, want_max)
            group_current[key] = window_best
            group_cur_positions[key] = cur_pos

        # Update position and count
        group_buffer_pos[key] = (pos + 1) % window
        if not group_full:
            group_n_seen[key] += 1

        if group_non_null[key] >= min_periods:
            out[i] = group_current[key]

    return out


def _rolling_min_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
):
    """
    Optimized core numba function for rolling min on 1D values.

    Uses position tracking to avoid scanning entire window on each update.
    Only recomputes min when the current minimum falls out of the window.
    """
    return _rolling_max_min_1d(**locals(), want_max=False)


def _rolling_max_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[np.ndarray] = None,
):
    """
    Optimized core numba function for rolling max on 1D values.

    Uses position tracking to avoid scanning entire window on each update.
    Only recomputes max when the current maximum falls out of the window.
    """
    return _rolling_max_min_1d(**locals(), want_max=True)


def rolling_min(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
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
    return _apply_rolling("min", **locals())


def rolling_max(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
    min_periods: Optional[int] = None,
    mask: Optional[ArrayType1D] = None,
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
    return _apply_rolling("max", **locals())


# ================================
# Cumulative Aggregation Functions
# ================================


@nb.njit(nogil=True, fastmath=False)
def _cumulative_reduce(
    group_key: np.ndarray,
    values: np.ndarray,
    reduce_func: Callable,
    ngroups: int,
    mask: Optional[np.ndarray] = None,
    skip_na: bool = True,
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
    skip_na : bool, default True
        Whether to skip NaN values in aggregation

    Returns
    -------
    np.ndarray
        Cumulative aggregated values with same shape as input values
    """
    out = np.full(len(values), np.nan)
    masked = mask is not None

    # Track current state for each group
    group_accumulators = np.full(ngroups, np.nan)
    group_seen = np.full(ngroups, False)

    for i in range(len(group_key)):
        key = group_key[i]

        if key < 0:
            continue

        if masked and not mask[i]:
            continue

        val = values[i]

        if skip_na and is_null(val):
            # For skipna=True, pass through the current accumulator without updating
            if group_seen[key]:
                out[i] = group_accumulators[key]
            continue

        if group_seen[key]:
            # Update accumulator with new value
            group_accumulators[key] = reduce_func(group_accumulators[key], val)
        else:
            # First non-null value for this group
            group_accumulators[key] = val
            group_seen[key] = True

        out[i] = group_accumulators[key]

    return out


def _apply_cumulative(
    operation: str,
    group_key: ArrayType1D,
    values: ArrayType1D | None,
    ngroups: int,
    mask: Optional[ArrayType1D] = None,
    skip_na: bool = True,
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
    group_key = np.asarray(group_key, dtype=np.int64)

    if mask is not None:
        mask = np.asarray(mask, dtype=bool)

    # Map operation names to reduction functions
    cumulative_funcs = {
        "sum": NumbaReductionOps.sum,
        "count": NumbaReductionOps.count,
        "min": NumbaReductionOps.min,
        "max": NumbaReductionOps.max,
    }

    if operation not in cumulative_funcs:
        raise ValueError(f"Unsupported cumulative operation: {operation}")

    # For count, we create a dummy values array since count ignores the y argument
    if operation == "count":
        if values is None:
            values = np.ones(len(group_key), dtype=np.float64)  # Dummy values
            original_dtype = None
        else:
            values = np.asarray(values)
            original_dtype = values.dtype
            values = values.astype(np.float64)
    else:
        if values is None:
            raise ValueError(f"values cannot be None for operation '{operation}'")
        values = np.asarray(values)
        original_dtype = values.dtype

        # Convert to float for computation, preserving original dtype info
        values = values.astype(np.float64)

    reduce_func = cumulative_funcs[operation]
    result = _cumulative_reduce(group_key, values, reduce_func, ngroups, mask, skip_na)

    # Apply type-specific result handling
    if operation == "count":
        # Replace NaN with 0 for count operations, then subtract 1 to make 0-based like pandas
        result = np.where(np.isnan(result), 0, result - 1).astype(np.int64)
    elif operation == "sum":
        # Type promotion rules for cumsum
        result = _apply_cumsum_type_promotion(result, original_dtype)
    elif operation in ("min", "max"):
        # Preserve exact input type for min/max
        result = _preserve_exact_type(result, original_dtype)

    return result


def _apply_cumsum_type_promotion(
    result: np.ndarray, original_dtype: np.dtype
) -> np.ndarray:
    """
    Apply type promotion rules for cumsum.

    Rules:
    - Signed integers and booleans → int64
    - Unsigned integers → uint64 (though less concerned about this)
    - Floating point types → preserve original precision
    """
    if original_dtype == np.bool_:
        # Handle NaN values before converting to int64
        return np.where(np.isnan(result), 0, result).astype(np.int64)
    elif np.issubdtype(original_dtype, np.signedinteger):
        # Handle NaN values before converting to int64
        return np.where(np.isnan(result), 0, result).astype(np.int64)
    elif np.issubdtype(original_dtype, np.unsignedinteger):
        # Handle NaN values before converting to uint64
        return np.where(np.isnan(result), 0, result).astype(np.uint64)
    elif original_dtype == np.float32:
        return result.astype(np.float32)
    else:
        # float64 and other types
        return result


def _preserve_exact_type(result: np.ndarray, original_dtype: np.dtype) -> np.ndarray:
    """
    Preserve the exact input type for min/max operations.
    """
    if original_dtype == np.bool_:
        # Convert back to boolean, handling NaN as False
        return np.where(np.isnan(result), False, result.astype(np.bool_))
    elif np.issubdtype(original_dtype, np.integer):
        # Handle NaN values for integer types by converting to 0 (or appropriate default)
        if np.issubdtype(original_dtype, np.signedinteger):
            return np.where(np.isnan(result), 0, result).astype(original_dtype)
        else:  # unsigned
            return np.where(np.isnan(result), 0, result).astype(original_dtype)
    else:
        # Float types can handle NaN naturally
        return result.astype(original_dtype)


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
    return _apply_cumulative("sum", group_key, values, ngroups, mask, skip_na)


def cumcount(
    group_key: ArrayType1D,
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
    return _apply_cumulative("count", group_key, None, ngroups, mask)


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


def build_groups_dict_optimized(
    group_ikey: np.ndarray, result_index, ngroups: int
):
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
    groups_dict = {key: idx for key, idx in zip(result_index, group_indices) if len(idx) > 0}   
    return groups_dict
