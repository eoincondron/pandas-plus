from inspect import signature
from typing import Callable
from functools import wraps
from copy import deepcopy

import numba as nb
import numpy as np

from ..util import (ArrayType1D, check_data_inputs_aligned, is_null, _null_value_for_array_type,
                    _maybe_cast_timestamp_arr, parallel_map, NumbaReductionOps)
from .. import nanops


@nb.njit(nogil=True, fastmath=False)
def _group_by_iterator(
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    reduce_func: Callable,
    must_see: bool = True,
):
    seen = np.full(len(target), not must_see)
    for i in range(len(group_key)):
        key = group_key[i]
        val = values[i]
        if is_null(val) or (len(mask) and not mask[i]):
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
    if arr.dtype.kind == 'b':
        return False
    else:
        return _null_value_for_array_type(arr)


@check_data_inputs_aligned("group_key", "values", "mask")
def _chunk_groupby_args(
    n_chunks: int,
    group_key: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    reduce_func: Callable,
    must_see: bool,
):
    mask = _prepare_mask_for_numba(mask)
    values = np.asarray(values)

    kwargs = locals().copy()
    del kwargs['n_chunks']
    shared_kwargs = {k: kwargs[k] for k in ['target', 'reduce_func', 'must_see']}

    chunked_kwargs = [deepcopy(shared_kwargs) for i in range(n_chunks)]
    for name in ['group_key', 'values', 'mask']:
        for chunk_no, arr in enumerate(np.array_split(kwargs[name], n_chunks)):
            chunked_kwargs[chunk_no][name] = arr

    chunked_args = [signature(_group_by_iterator).bind(**kwargs) for kwargs in chunked_kwargs]

    return chunked_args


@check_data_inputs_aligned("group_key", "values", "mask")
def _group_func_wrap(
    reduce_func_name: str,
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    initial_value: int | float = None,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    values = np.asarray(values)
    values, orig_type = _maybe_cast_timestamp_arr(values)
    mask = _prepare_mask_for_numba(mask)
    if initial_value is None:
        initial_value = _default_initial_value_for_type(values)
    target = np.full(ngroups, initial_value)
    if reduce_func_name == 'count':
        out_type = 'int64'
    elif reduce_func_name == 'sum' and orig_type.kind == 'b':
        out_type = 'int64'
    else:
        out_type = orig_type

    kwargs = dict(
        group_key=group_key,
        values=values,
        target=target,
        mask=mask,
        reduce_func=getattr(NumbaReductionOps, reduce_func_name),
        must_see=reduce_func_name != "count",
    )

    if n_threads == 1:
        return _group_by_iterator(**kwargs).astype(out_type)
    else:
        chunked_args = _chunk_groupby_args(**kwargs, n_chunks=n_threads)
        chunks = parallel_map(
            _group_by_iterator, [args.args for args in chunked_args]
        )
        arr = np.vstack(chunks)
        chunk_reduce = "sum" if reduce_func_name == "count" else reduce_func_name
        return nanops.reduce_2d(chunk_reduce, arr).astype(out_type)


@check_data_inputs_aligned("group_key", "values", "mask")
def group_count(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    initial_value = 0
    return _group_func_wrap("count", **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_sum(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    if values.dtype.kind == 'f':
        initial_value = 0.
    else:
        initial_value = 0
    return _group_func_wrap("sum", **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_mean(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    kwargs = locals().copy()
    sum = group_sum(**kwargs)
    int_sum, orig_type = _maybe_cast_timestamp_arr(sum)
    count = group_count(**kwargs)
    mean = (int_sum / count)
    if values.dtype.kind in 'mM':
        mean = mean.astype(orig_type)
    return mean


@check_data_inputs_aligned("group_key", "values", "mask")
def group_min(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
    n_threads: int = 1,
):
    return _group_func_wrap("min", **locals())


@check_data_inputs_aligned("group_key", "values", "mask")
def group_max(
    group_key: ArrayType1D,
    values: ArrayType1D,
    ngroups: int,
    mask: ArrayType1D = None,
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
        mask: ArrayType1D = None,
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
        mask: ArrayType1D = None,
        skip_na=True,
    ):
        if skip_na:
            reduce_func_name = "last_skipna"
        else:
            reduce_func_name = "last"
        del skip_na
        return _group_func_wrap(**locals())
