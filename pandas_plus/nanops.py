import numba as nb
import numpy as np
import pandas as pd
from pandas.core import nanops

from .util import (
    is_null,
    n_threads_from_array_length,
    parallel_map,
    _null_value_for_numpy_type,
    _get_first_non_null,
    NumbaReductionOps,
)


@nb.njit(nogil=True)
def _nb_reduce(reduce_func, arr, skipna: bool = True, initial_value=None):
    """
    Apply a reduction function to an array, with NA/null handling.

    Parameters
    ----------
    reduce_func : callable
        Function that combines two values (e.g., min, max, sum)
    arr : array-like
        Array to reduce
    skipna : bool, default True
        Whether to skip NA/null values
    initial_value : scalar, optional
        Starting value for the reduction

    Returns
    -------
    scalar
        Result of the reduction operation

    Notes
    -----
    This function is JIT-compiled with Numba for performance.
    """
    if initial_value is None:
        if skipna:
            loc, out = _get_first_non_null(arr)
            start = loc + 1
            if loc == -1:  # all null
                return arr[0]
        else:
            start, out = 1, arr[0]
            if is_null(out):
                return out
    else:
        start, out = 0, initial_value

    if skipna:
        for j in range(start, len(arr)):
            x = arr[j]
            if is_null(x):
                continue
            out = reduce_func(out, x)
    else:
        for j in range(start, len(arr)):
            x = arr[j]
            out = reduce_func(out, x)

    return out


def reduce_1d(reduce_func_name: str, arr, skipna: bool = True, n_threads: int = None):
    """
    Apply a reduction function to a 1D array, with optional parallelization.

    Parameters
    ----------
    reduce_func_name : str
        Name of the reduction function ('sum', 'min', 'max', etc.)
    arr : array-like
        1D array to reduce
    skipna : bool, default True
        Whether to skip NA/null values
    n_threads : int, optional
        Number of threads to use for parallel processing. If None,
        determines automatically based on array length.

    Returns
    -------
    scalar
        Result of the reduction operation
    """
    reduce_func = getattr(NumbaReductionOps, reduce_func_name)

    # Check for datetime64 or timedelta64 dtypes
    is_datetime = np.issubdtype(arr.dtype, np.datetime64)
    is_timedelta = np.issubdtype(arr.dtype, np.timedelta64)
    is_count = reduce_func_name == "count"
    if is_datetime and not is_count:
        output_converter = pd.to_datetime
    elif is_timedelta and not is_count:
        output_converter = pd.to_timedelta
    else:
        output_converter = np.asarray

    if is_count:
        kwargs = dict(
            skipna=True,
            initial_value=int(0),
        )
        chunk_reduction = "sum"
    elif "sum" in reduce_func_name:
        kwargs = dict(
            skipna=skipna,
            initial_value=0,
        )
        chunk_reduction = "sum"
    else:
        kwargs = dict(
            skipna=skipna,
            initial_value=None,
        )
        chunk_reduction = reduce_func_name

    # Convert datetime64/timedelta64 to int64 view for numba operations
    if is_datetime or is_timedelta:
        arr = arr.view("int64")

    if n_threads is None:
        n_threads = n_threads_from_array_length(len(arr))

    if n_threads == 1:
        result = output_converter(
            _nb_reduce(reduce_func=reduce_func, arr=arr, **kwargs)
        )
    else:
        chunks = parallel_map(
            lambda a: _nb_reduce(reduce_func=reduce_func, arr=a, **kwargs),
            list(zip(np.array_split(arr, n_threads))),
        )
        chunks = output_converter(chunks)
        result = reduce_1d(chunk_reduction, chunks, skipna=skipna, n_threads=1)

    if is_count:
        result = np.int64(result)

    return result


def reduce_2d(
    reduce_func_name: str, arr, skipna: bool = True, n_threads: int = 1, axis=0
):
    """
    Apply a reduction function to a 2D array along a specified axis.

    Parameters
    ----------
    reduce_func_name : str
        Name of the reduction function ('sum', 'min', 'max', etc.)
    arr : array-like
        2D array to reduce
    skipna : bool, default True
        Whether to skip NA/null values
    n_threads : int, default 1
        Number of threads to use for parallel processing
    axis : int, default 0
        Axis along which to perform the reduction (0=rows, 1=columns)

    Returns
    -------
    ndarray
        1D array of results from the reduction operation
    """
    if axis == 0:
        arr = arr.T
    mapper = lambda x: reduce_1d(
        reduce_func_name, x, skipna=skipna, n_threads=n_threads
    )
    if n_threads == 1:
        results = list(map(mapper, arr))
    else:
        results = parallel_map(mapper, list(zip(arr)))
    return np.array(results)


def reduce(
    arr,
    reduce_func_name: str,
    skipna=True,
    min_count=0,
    axis=None,
    n_threads: int = None,
):
    """
    Apply a reduction function to an array with NA handling and optional parallelization.

    Parameters
    ----------
    arr : array-like
        Array to reduce
    reduce_func_name : str
        Name of the reduction function ('sum', 'min', 'max', etc.)
    skipna : bool, default True
        Whether to skip NA/null values
    min_count : int, default 0
        Minimum number of valid values required to perform the operation
    axis : int, optional
        Axis along which to perform the reduction for 2D arrays
    n_threads : int, optional
        Number of threads to use for parallel processing

    Returns
    -------
    scalar or ndarray
        Result of the reduction operation
    """
    arr = np.asarray(arr)
    if min_count != 0:
        return getattr(nanops, f"nan{reduce_func_name}")(**locals())

    if arr.ndim == 1:
        return reduce_1d(reduce_func_name, arr, skipna=skipna, n_threads=n_threads)
    else:
        if axis is None:
            # warn
            axis = 0
        return reduce_2d(reduce_func_name, arr, axis=axis, n_threads=n_threads)


def nansum(
    arr,
    skipna: bool = True,
    min_count: int = 0,
    axis: int = None,
    n_threads: int = None,
):
    """
    Sum of array elements, ignoring NaNs by default.

    Parameters
    ----------
    arr : array-like
        Array to sum
    skipna : bool, default True
        Whether to skip NA/null values
    min_count : int, default 0
        Minimum number of valid values required to perform the operation
    axis : int, optional
        Axis along which to perform the sum for 2D arrays
    n_threads : int, optional
        Number of threads to use for parallel processing

    Returns
    -------
    scalar or ndarray
        Sum of values
    """
    return reduce(reduce_func_name="sum", **locals())


def count(arr, axis: int = None):
    """
    Count non-NA/non-null values in an array.

    Parameters
    ----------
    arr : array-like
        Array to count values in
    axis : int, optional
        Axis along which to count for 2D arrays

    Returns
    -------
    scalar or ndarray
        Count of non-NA values
    """
    return reduce(reduce_func_name="count", **locals())


def nanmean(
    arr,
    skipna: bool = True,
    min_count: int = 0,
    axis: int = None,
    n_threads: int = None,
):
    """
    Mean of array elements, ignoring NaNs by default.

    Parameters
    ----------
    arr : array-like
        Array to calculate mean of
    skipna : bool, default True
        Whether to skip NA/null values
    min_count : int, default 0
        Minimum number of valid values required to perform the operation
    axis : int, optional
        Axis along which to calculate mean for 2D arrays
    n_threads : int, optional
        Number of threads to use for parallel processing

    Returns
    -------
    scalar or ndarray
        Mean of values
    """
    sum = nansum(**locals())
    n = count(arr, axis=axis)
    if n == 0:
        return _null_value_for_numpy_type(arr.dtype)
    return sum / n


def nanmax(
    arr,
    skipna: bool = True,
    min_count: int = 0,
    axis: int = None,
    n_threads: int = None,
):
    """
    Maximum of array elements, ignoring NaNs by default.

    Parameters
    ----------
    arr : array-like
        Array to find maximum of
    skipna : bool, default True
        Whether to skip NA/null values
    min_count : int, default 0
        Minimum number of valid values required to perform the operation
    axis : int, optional
        Axis along which to find maximum for 2D arrays
    n_threads : int, optional
        Number of threads to use for parallel processing

    Returns
    -------
    scalar or ndarray
        Maximum value(s)
    """
    return reduce(reduce_func_name="max", **locals())


def nanmin(
    arr,
    skipna: bool = True,
    min_count: int = 0,
    axis: int = None,
    n_threads: int = None,
):
    """
    Minimum of array elements, ignoring NaNs by default.

    Parameters
    ----------
    arr : array-like
        Array to find minimum of
    skipna : bool, default True
        Whether to skip NA/null values
    min_count : int, default 0
        Minimum number of valid values required to perform the operation
    axis : int, optional
        Axis along which to find minimum for 2D arrays
    n_threads : int, optional
        Number of threads to use for parallel processing

    Returns
    -------
    scalar or ndarray
        Minimum value(s)
    """
    return reduce(reduce_func_name="min", **locals())


def nanvar(
    arr,
    skipna: bool = True,
    min_count: int = 0,
    axis: int = None,
    n_threads: int = None,
    ddof: int = 1,
):
    """
    Variance of array elements, ignoring NaNs by default.

    Parameters
    ----------
    arr : array-like
        Array to calculate variance of
    skipna : bool, default True
        Whether to skip NA/null values
    min_count : int, default 0
        Minimum number of valid values required to perform the operation
    axis : int, optional
        Axis along which to calculate variance for 2D arrays
    n_threads : int, optional
        Number of threads to use for parallel processing
    ddof : int, default 1
        Delta degrees of freedom for calculating variance

    Returns
    -------
    scalar or ndarray
        Variance of values
    """
    kwargs = locals().copy()
    del kwargs["ddof"]
    n = count(arr, axis=axis)
    sum_sq = reduce(reduce_func_name="sum_square", **kwargs)
    sum = reduce(reduce_func_name="sum", **kwargs)
    d = n - ddof
    if d == 0 or n == 0:
        return _null_value_for_numpy_type(arr.dtype)
    return (sum_sq - sum**2 / n) / d


def nanstd(
    arr,
    skipna: bool = True,
    min_count: int = 0,
    axis: int = None,
    n_threads: int = None,
    ddof: int = 1,
):
    """
    Standard deviation of array elements, ignoring NaNs by default.

    Parameters
    ----------
    arr : array-like
        Array to calculate standard deviation of
    skipna : bool, default True
        Whether to skip NA/null values
    min_count : int, default 0
        Minimum number of valid values required to perform the operation
    axis : int, optional
        Axis along which to calculate standard deviation for 2D arrays
    n_threads : int, optional
        Number of threads to use for parallel processing
    ddof : int, default 1
        Delta degrees of freedom for calculating standard deviation

    Returns
    -------
    scalar or ndarray
        Standard deviation of values
    """
    return nanvar(**locals()) ** 0.5
