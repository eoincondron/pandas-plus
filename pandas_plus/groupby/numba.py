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

    chunked_args = [
        signature(iterator).bind(**kwargs) for kwargs in chunked_kwargs
    ]

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
