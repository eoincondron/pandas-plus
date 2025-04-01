from functools import reduce, wraps
import operator
import os
from inspect import signature
from typing import Mapping, Union, Any, Callable, TypeVar, cast, List, Optional
import concurrent.futures

import numba as nb
import numpy as np
import pandas as pd
import polars as pl
from numba.core.extending import overload

T = TypeVar("T")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])

MIN_INT = np.iinfo(np.int64).min
MAX_INT = np.iinfo(np.int64).max

ArrayType1D = Union[np.ndarray, pl.Series, pd.Series, pd.Index, pd.Categorical]
ArrayType2D = Union[np.ndarray, pl.DataFrame, pd.DataFrame, pd.MultiIndex]


def is_null(x):
    """
    Check if a value is considered null/NA.
    
    Parameters
    ----------
    x : scalar
        Value to check
        
    Returns
    -------
    bool
        True if value is null, False otherwise
        
    Notes
    -----
    This function is overloaded with specialized implementations for 
    various numeric types via Numba's overload mechanism.
    """
    dtype = np.asarray(x).dtype
    if np.issubdtype(dtype, np.float64):
        return np.isnan(x)

    elif np.issubdtype(dtype, np.int64):
        return x == MIN_INT

    else:
        return False


@overload(is_null)
def jit_is_null(x):
    if isinstance(x, nb.types.Float) or isinstance(x, float):

        def is_null(x):

            return np.isnan(x)

        return is_null
    if isinstance(x, nb.types.Integer):

        def is_null(x):
            return x == MIN_INT

        return is_null
    elif isinstance(x, nb.types.Boolean):

        def is_null(x):
            return False

        return is_null

@nb.njit(parallel=True)
def arr_is_null(arr):
    out = np.zeros(len(arr), dtype=nb.bool_)
    for i in nb.prange(len(arr)):
        out[i] = is_null(arr[i])
    return out


def _null_value_for_array_type(arr: np.ndarray):
    """
    Get the appropriate null/NA value for the given array's dtype.
    
    Parameters
    ----------
    arr : np.ndarray
        Array whose dtype determines the null value
        
    Returns
    -------
    scalar
        Appropriate null value (min value for integers, NaN for floats, max for unsigned)
        
    Raises
    ------
    TypeError
        If the array's dtype doesn't have a defined null representation
    """
    error = TypeError(f"No null value for {arr.dtype}")
    match arr.dtype.kind:
        case 'i':
            if arr.dtype.itemsize >= 4:
                return np.iinfo(arr.dtype).min
            else:
                raise error
        case 'f':
            return np.array(np.nan, dtype=arr.dtype)
        case 'u':
            if arr.dtype.itemsize >= 4:
                return np.iinfo(arr.dtype).max
            else:
                raise error
        case 'm':
            return np.array('NaT', dtype='m8[ns]')
        case 'M':
            return np.array('NaT', dtype='m8[ns]')
        case _:
            raise error


def _maybe_cast_timestamp_arr(arr):
    if arr.dtype.kind in 'mM':
        return arr.view('int64'), arr.dtype
    else:
        return arr, arr.dtype


def check_data_inputs_aligned(
    *args_to_check, check_index: bool = True
) -> Callable[[F], F]:
    """
    Factory function that returns a decorator which ensures all arguments passed to the
    decorated function have equal length and, if pandas objects and check_index is True,
    share a common index.

    Args:
        check_index: If True, also checks that pandas objects share the same index

    Returns:
        A decorator function
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if not args:
                return func(*args, **kwargs)

            arguments = signature(func).bind(*args, **kwargs).arguments
            lengths = {}
            # Extract args that have a length
            for k, x in arguments.items():
                if not args_to_check or k in args_to_check:
                    if x is not None:
                        lengths[k] = len(x)
            if len(set(lengths.values())) > 1:
                raise ValueError(
                    f"All arguments must have equal length. " f"Got lengths: {lengths}"
                )

            # Check pandas objects share the same index
            if check_index:
                pandas_args = [
                    arg for arg in args if isinstance(arg, (pd.Series, pd.DataFrame))
                ]
                if pandas_args:
                    first_index = pandas_args[0].index
                    for arg in pandas_args[1:]:
                        if not first_index.equals(arg.index):
                            raise ValueError(
                                "All pandas objects must share the same index"
                            )

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


def parallel_map(
    func: Callable[[T], R], arg_list: List[T], max_workers: Optional[int] = None
) -> List[R]:
    """
    Apply a function to each item in a list in parallel using concurrent.futures.

    Args:
        func: The function to apply to each item
        arg_list: List of items to process
        max_workers: Maximum number of worker threads or processes (None = auto)

    Returns:
        List of results in the same order as the input items

    Example:
        >>> def square(x):
        ...     return x * x
        >>> parallel_map(square, [1, 2, 3, 4, 5])
        [1, 4, 9, 16, 25]
    """
    # Choose between ProcessPoolExecutor and ThreadPoolExecutor based on your needs
    # ProcessPoolExecutor is better for CPU-bound tasks
    # ThreadPoolExecutor is better for I/O-bound tasks
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and store the future objects
        future_to_index = {
            executor.submit(func, *args): i for i, args in enumerate(arg_list)
        }

        # Collect results in the original order
        results = [None] * len(arg_list)

        # Process completed futures as they finish
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                print(f"Item at index {index} generated an exception: {exc}")
                raise

    return results


def n_threads_from_array_length(arr_len: int):
    """
    Calculate a reasonable number of threads based on array length.
    
    Parameters
    ----------
    arr_len : int
        Length of the array to be processed
        
    Returns
    -------
    int
        Number of threads to use (at least 1, at most 2*cpu_count-2)
    """
    return min(max(1, arr_len // int(2e6)), os.cpu_count() * 2 - 2)


def parallel_reduce(reducer, reduce_func_name: str, chunked_args):
    """
    Apply reduction function in parallel and combine results.
    
    Parameters
    ----------
    reducer : callable
        Function to apply to each chunk of data
    reduce_func_name : str
        Name of the reduction function ('count', 'sum', 'max', etc.)
    chunked_args : list
        Arguments for the reducer function split into chunks
        
    Returns
    -------
    array-like
        Combined result after applying the reduction function to all chunks
        
    Raises
    ------
    ValueError
        If the reduction function is not supported for parallel execution
    """
    try:
        reduce_func_vec = dict(
            count=operator.add,
            sum=operator.add,
            sum_square=operator.add,
            max=np.maximum,
            min=np.minimum,
        )[reduce_func_name]
    except:
        raise ValueError(f"Multi-threading not supported for {reduce_func_name}")
    results = parallel_map(reducer, chunked_args)
    return reduce(reduce_func_vec, results)


def _get_first_non_null(arr) -> (int, T):
    """
    Find the first non-null value in an array. Return its location and value

    Parameters
    ----------
    arr : array-like
        Array to search for non-null values

    Returns
    -------
    tuple
        (index, value) of first non-null value, or (-1, np.nan) if all values are null

    Notes
    -----
    This function is JIT-compiled with Numba for performance.
    """
    for i, x in enumerate(arr):
        if not is_null(x):
            return i, x
    return -1, np.nan


@overload(_get_first_non_null, nogil=True)
def jit_get_first_non_null(arr):
    if isinstance(arr.dtype, nb.types.Float):

        return _get_first_non_null

    elif isinstance(arr.dtype, nb.types.Integer):

        def f(arr):
            for i, x in enumerate(arr):
                if not is_null(x):
                    return i, x
            return -1, MIN_INT

        return f

    elif arr.dtype.kind == 'b':
        def f(x):
            return 0, arr[0]

        return f

def _scalar_func_decorator(func):
    return staticmethod(nb.njit(nogil=True)(func))


class NumbaReductionOps:

    @_scalar_func_decorator
    def count(x, y):
        return x + 1

    @_scalar_func_decorator
    def min(x, y):
        return x if x <= y else y

    @_scalar_func_decorator
    def max(x, y):
        return x if x >= y else y

    @_scalar_func_decorator
    def sum(x, y):
        return x + y

    @_scalar_func_decorator
    def first(x, y):
        return x

    @_scalar_func_decorator
    def first_skipna(x, y):
        return y if is_null(x) else x

    @_scalar_func_decorator
    def last(x, y):
        return y

    @_scalar_func_decorator
    def last_skipna(x, y):
        return x if is_null(y) else y

    @_scalar_func_decorator
    def sum_square(x, y):
        return x + y**2
