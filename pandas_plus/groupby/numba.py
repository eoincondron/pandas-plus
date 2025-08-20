import inspect
from inspect import signature
from typing import Callable, Optional
from functools import wraps
from copy import deepcopy

import numba as nb
import numpy as np

from ..util import (
    ArrayType1D,
    check_data_inputs_aligned,
    is_null,
    _null_value_for_array_type,
    _maybe_cast_timestamp_arr,
    parallel_map,
    NumbaReductionOps,
)
from .. import nanops


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


@nb.njit(nogil=True, fastmath=False)
def _group_by_counter(
    group_key: np.ndarray,
    values: np.ndarray | None,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
):
    masked = mask is not None
    skip_na = values is None
    for i in range(len(group_key)):
        key = group_key[i]
        if masked and not mask[i]:
            continue
        if not skip_na:
            if is_null(values[i]):
                continue
        target[key] += 1
    return target


@nb.njit(nogil=True, fastmath=False)
def _group_by_reduce(
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    reduce_func: Callable,
    mask: Optional[np.ndarray] = None,
    skip_na: bool = True,
):
    masked = mask is not None
    seen = np.full(len(target), False)
    for i in range(len(group_key)):
        key = group_key[i]
        val = values[i]
        if (skip_na and is_null(val)) or (masked and not mask[i]):
            continue

        if seen[key]:
            target[key] = reduce_func(target[key], val)
        else:
            target[key] = val
            seen[key] = True

    return target


def _prepare_mask_for_numba(mask):
    if mask is None:
        mask = np.array([], dtype=bool)
    else:
        mask = np.asarray(mask)
        if mask.dtype.kind != "b":
            raise TypeError(f"mask must of Boolean type. Got {mask.dtype}")
    return mask


def _default_initial_value_for_type(arr):
    if arr.dtype.kind == "b":
        return False
    else:
        return _null_value_for_array_type(arr)


@check_data_inputs_aligned("group_key", "values", "mask")
def _chunk_groupby_args(
    n_chunks: int,
    group_key: np.ndarray,
    values: np.ndarray | None,
    target: np.ndarray,
    reduce_func: Optional[Callable] = None,
    mask: Optional[np.ndarray] = None,
):
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
            chunks = np.array_split(kwargs[name], n_chunks)
        for chunk_no, arr in enumerate(chunks):
            chunked_kwargs[chunk_no][name] = arr

    chunked_args = [signature(iterator).bind(**kwargs) for kwargs in chunked_kwargs]

    return chunked_args


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


@nb.njit(nogil=True, fastmath=False)
def _rolling_sum_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    mask: Optional[np.ndarray] = None,
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
    out = np.full(len(values), np.nan)
    masked = mask is not None
    
    # Track rolling sums and circular buffers for each group
    group_sums = np.zeros(ngroups)
    group_buffers = np.full((ngroups, window), np.nan)
    group_positions = np.zeros(ngroups, dtype=nb.int64)
    group_counts = np.zeros(ngroups, dtype=nb.int64)
    
    for i in range(len(group_key)):
        key = group_key[i]
        
        if key < 0:  # Skip null keys
            continue
            
        if masked and not mask[i]:
            continue
            
        val = values[i]
        
        if np.isnan(val):  # Skip NaN values
            continue
            
        # Get current position in circular buffer for this group
        pos = group_positions[key]
        count = group_counts[key]
        
        # If buffer is full, subtract the value that will be replaced
        if count >= window:
            old_val = group_buffers[key, pos]
            if not np.isnan(old_val):
                group_sums[key] -= old_val
        
        # Add new value
        group_buffers[key, pos] = val
        group_sums[key] += val
        
        # Update position and count
        group_positions[key] = (pos + 1) % window
        if count < window:
            group_counts[key] = count + 1
            
        out[i] = group_sums[key]
    
    return out


def _apply_rolling_1d_to_2d(
    rolling_1d_func: Callable,
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
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
    
    if n_threads == 1 or n_cols == 1:
        # Single-threaded: process columns sequentially
        result = np.empty((n_rows, n_cols))
        for col in range(n_cols):
            result[:, col] = rolling_1d_func(group_key, values[:, col], ngroups, window, mask)
        return result
    else:
        # Multi-threaded: process columns in parallel
        def process_column(col_idx):
            return rolling_1d_func(group_key, values[:, col_idx], ngroups, window, mask)
        
        # Use parallel_map to process columns in parallel
        column_results = parallel_map(process_column, [(i,) for i in range(n_cols)], max_workers=n_threads)
        
        # Combine results into 2D array
        result = np.column_stack(column_results)
        return result


def _apply_rolling(
    operation: str,
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
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
        'sum': _rolling_sum_1d,
        'mean': _rolling_mean_1d,
        'min': _rolling_min_1d,
        'max': _rolling_max_1d,
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
        return rolling_1d_func(group_key, values, ngroups, window, mask)
    elif values.ndim == 2:
        return _apply_rolling_1d_to_2d(rolling_1d_func, group_key, values, ngroups, window, mask, n_threads)
    else:
        raise ValueError(f"values must be 1D or 2D, got {values.ndim}D")



@nb.njit(nogil=True, fastmath=False)
def _rolling_mean_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    mask: Optional[np.ndarray] = None,
):
    """
    Core numba function for rolling mean on 1D values.
    """
    out = np.full(len(values), np.nan)
    masked = mask is not None
    
    # Track windows for each group
    group_windows = nb.typed.List()
    for _ in range(ngroups):
        group_windows.append(nb.typed.List.empty_list(nb.float64))
    
    for i in range(len(group_key)):
        key = group_key[i]
        
        if key < 0:
            continue
            
        if masked and not mask[i]:
            continue
        
        val = values[i]
        
        if is_null(val):
            continue
            
        window_vals = group_windows[key]
        window_vals.append(val)
        
        if len(window_vals) > window:
            window_vals.pop(0)
            
        # Calculate mean of current window
        window_sum = 0.0
        for v in window_vals:
            window_sum += v
            
        out[i] = window_sum / len(window_vals)
    
    return out




def rolling_sum(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
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
    return _apply_rolling("sum", group_key, values, ngroups, window, mask, n_threads)


def rolling_mean(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
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
    return _apply_rolling("mean", group_key, values, ngroups, window, mask, n_threads)


@nb.njit(nogil=True, fastmath=False)
def _rolling_min_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    mask: Optional[np.ndarray] = None,
):
    """
    Core numba function for rolling min on 1D values.
    """
    out = np.full(len(values), np.nan)
    masked = mask is not None
    
    # Track windows for each group
    group_windows = nb.typed.List()
    for _ in range(ngroups):
        group_windows.append(nb.typed.List.empty_list(nb.float64))
    
    for i in range(len(group_key)):
        key = group_key[i]
        
        if key < 0:
            continue
            
        if masked and not mask[i]:
            continue
            
        val = values[i]
        
        if np.isnan(val):
            continue
            
        window_vals = group_windows[key]
        window_vals.append(val)
        
        if len(window_vals) > window:
            window_vals.pop(0)
            
        # Calculate min of current window
        window_min = np.inf
        for v in window_vals:
            if v < window_min:
                window_min = v
                
        out[i] = window_min
    
    return out




@nb.njit(nogil=True, fastmath=False)
def _rolling_max_1d(
    group_key: np.ndarray,
    values: np.ndarray,
    ngroups: int,
    window: int,
    mask: Optional[np.ndarray] = None,
):
    """
    Core numba function for rolling max on 1D values.
    """
    out = np.full(len(values), np.nan)
    masked = mask is not None
    
    # Track windows for each group
    group_windows = nb.typed.List()
    for _ in range(ngroups):
        group_windows.append(nb.typed.List.empty_list(nb.float64))
    
    for i in range(len(group_key)):
        key = group_key[i]
        
        if key < 0:
            continue
            
        if masked and not mask[i]:
            continue
            
        val = values[i]
        
        if np.isnan(val):
            continue
            
        window_vals = group_windows[key]
        window_vals.append(val)
        
        if len(window_vals) > window:
            window_vals.pop(0)
            
        # Calculate max of current window
        window_max = -np.inf
        for v in window_vals:
            if v > window_max:
                window_max = v
                
        out[i] = window_max
    
    return out




def rolling_min(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
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
    return _apply_rolling("min", group_key, values, ngroups, window, mask, n_threads)


def rolling_max(
    group_key: ArrayType1D,
    values: ArrayType1D | np.ndarray,
    ngroups: int,
    window: int,
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
    return _apply_rolling("max", group_key, values, ngroups, window, mask, n_threads)
