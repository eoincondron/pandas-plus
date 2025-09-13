from functools import reduce, wraps
import operator
import os
from inspect import signature
from typing import Mapping, Union, Any, Callable, TypeVar, cast, List, Optional, Tuple
import concurrent.futures

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import numba as nb
from numba.typed import List as NumbaList
from numba.core.extending import overload
from pandas.core.sorting import get_group_index

T = TypeVar("T")
R = TypeVar("R")
F = TypeVar("F", bound=Callable[..., Any])

MIN_INT = np.iinfo(np.int64).min
MAX_INT = np.iinfo(np.int64).max

ArrayType1D = Union[
    np.ndarray,
    pl.Series,
    pd.Series,
    pd.Index,
    pd.Categorical,
    pa.ChunkedArray,
    pa.Array,
]
ArrayType2D = Union[np.ndarray, pl.DataFrame, pl.LazyFrame, pd.DataFrame, pd.MultiIndex]


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


def _null_value_for_numpy_type(np_type: np.dtype):
    """
    Get the appropriate null/NA value for the given array's dtype.

    Parameters
    ----------
    np_type : np.dtype
        Numpy dtype of the array
    Returns
    -------
    scalar
        Appropriate null value (min value for integers, NaN for floats, max for unsigned)

    Raises
    ------
    TypeError
        If the array's dtype doesn't have a defined null representation
    """
    error = TypeError(f"No null value for {np_type}")
    match np_type.kind:
        case "i":
            return np.iinfo(np_type).min
        case "f":
            return np.array([np.nan], dtype=np_type)[0]
        case "u":
            return np.iinfo(np_type).max
        case "m":
            return np.timedelta64("NaT", "ns")
        case "M":
            return np.datetime64("NaT", "ns")
        case "b":
            return False
        case _:
            raise error


def _maybe_cast_timestamp_arr(arr) -> Tuple[np.ndarray, np.dtype]:
    if arr.dtype.kind in "mM":
        return arr.view("int64"), arr.dtype
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
    func: Callable[[T], R],
    arg_list: List[T],
    max_workers: Optional[int] = None,
    use_threads: bool = True,
) -> List[R]:
    """
    Apply a function to each item in a list in parallel using concurrent.futures.

    Args:
        func: The function to apply to each item
        arg_list: List of items to process
        max_workers: Maximum number of worker threads or processes (None = auto)
        use_threads: If True, use threads; if False, use processes

    Returns:
        List of results in the same order as the input items

    Example:
        >>> def square(x):
        ...     return x * x
        >>> parallel_map(square, [1, 2, 3, 4, 5])
        [1, 4, 9, 16, 25]
    """
    arg_list = list(arg_list)
    if len(arg_list) == 1:
        return [func(*arg_list[0])]

    if use_threads:
        Executor = concurrent.futures.ThreadPoolExecutor
    else:
        Executor = concurrent.futures.ProcessPoolExecutor

    with Executor(max_workers=max_workers) as executor:
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

    elif isinstance(arr.dtype, nb.types.Boolean):

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


def get_array_name(
    array: Union[np.ndarray, pd.Series, pl.Series, pa.ChunkedArray, pa.Array],
):
    """
    Get the name attribute of an array if it exists and is not empty.

    Parameters
    ----------
    array : Union[np.ndarray, pd.Series, pl.Series]
        Array-like object to get name from

    Returns
    -------
    str or None
        The name of the array if it exists and is not empty, otherwise None
    """
    name = getattr(array, "name", None)
    if name is None or name == "":
        return None
    return name


def to_arrow(a: ArrayType1D, zero_copy_only: bool = True) -> pa.Array | pa.ChunkedArray:
    """
    Convert various array types to PyArrow Array or ChunkedArray with minimal copying.

    This function provides a unified interface for converting different array-like objects
    (NumPy arrays, pandas Series/Index/Categorical, polars Series, and PyArrow structures)
    to PyArrow format. It aims to minimize memory copying by leveraging zero-copy
    conversions where possible.

    Parameters
    ----------
    a : ArrayType1D
        Input array to convert. Can be one of:
        - numpy.ndarray
        - pandas.Series, pandas.Index, pandas.Categorical
        - polars.Series
        - pyarrow.Array, pyarrow.ChunkedArray

    Returns
    -------
    pa.Array or pa.ChunkedArray
        PyArrow representation of the input array. Returns pa.ChunkedArray for
        inputs that are already chunked, pa.Array otherwise.

    Raises
    ------
    TypeError
        If the input type is not supported for conversion to PyArrow format.

    Notes
    -----
    Zero-copy conversions are attempted where possible:
    - PyArrow Array/ChunkedArray: Returns input directly (no copy)
    - Polars Series: Uses polars' built-in to_arrow() method (zero-copy)
    - Pandas with ArrowDtype: Uses PyArrow's from_pandas() (minimal copy)
    - NumPy arrays: Uses PyArrow's array() constructor with zero_copy_only=False

    For pandas Categorical data, the function converts to a PyArrow DictionaryArray
    which preserves the categorical structure while enabling efficient operations.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import pyarrow as pa
    >>> from pandas_plus.util import to_arrow

    NumPy array conversion:
    >>> arr = np.array([1, 2, 3, 4, 5])
    >>> arrow_arr = to_arrow(arr)
    >>> type(arrow_arr)
    <class 'pyarrow.lib.Int64Array'>

    Pandas Series conversion:
    >>> series = pd.Series([1.1, 2.2, 3.3])
    >>> arrow_arr = to_arrow(series)
    >>> type(arrow_arr)
    <class 'pyarrow.lib.DoubleArray'>

    Categorical conversion:
    >>> cat = pd.Categorical(['a', 'b', 'a', 'c'])
    >>> arrow_arr = to_arrow(cat)
    >>> type(arrow_arr)
    <class 'pyarrow.lib.DictionaryArray'>

    PyArrow pass-through (no copy):
    >>> pa_arr = pa.array([1, 2, 3])
    >>> result = to_arrow(pa_arr)
    >>> result is pa_arr  # Same object, no copy
    True
    """
    if isinstance(a, pl.Series):
        return a.to_arrow()
    elif isinstance(a, pd.core.base.PandasObject):
        if isinstance(a.dtype, pd.ArrowDtype):
            return pa.Array.from_pandas(a)  # type: ignore
        elif isinstance(a.dtype, pd.CategoricalDtype):
            a = pd.Series(a)
            return pa.DictionaryArray.from_arrays(a.cat.codes.values, a.cat.categories)
        else:
            if zero_copy_only and pd.api.types.is_bool_dtype(a):
                raise TypeError("Zero copy conversions not possible with boolean types")
            return pa.array(np.asarray(a))
    elif isinstance(a, np.ndarray):
        if zero_copy_only and a.dtype == bool:
            raise TypeError("Zero copy conversions not possible with boolean types")
        return pa.array(a)
    elif isinstance(a, (pa.Array, pa.ChunkedArray)):
        return a  # ChunkedArray is already a PyArrow structure
    else:
        raise TypeError(f"Cannot convert type {type(a)} to arrow")


def series_is_numeric(series: pl.Series | pd.Series):
    dtype = series.dtype
    if isinstance(series, pl.Series):
        return dtype.is_numeric() or dtype.is_temporal() or dtype == pl.Boolean
    else:
        return not (
            pd.api.types.is_object_dtype(series)
            or isinstance(dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(dtype)
            or "dictionary" in str(dtype)
        )


def is_categorical(a):
    if isinstance(a, pd.core.base.PandasObject):
        return isinstance(a.dtype, pd.CategoricalDtype) or "dictionary" in str(a.dtype)
    elif isinstance(a, pl.Series):
        return a.dtype == pl.Categorical
    elif isinstance(a, pa.ChunkedArray):
        return isinstance(a.chunks[0], pa.DictionaryArray)
    else:
        return isinstance(a, pa.DictionaryArray)


def array_split_with_chunk_handling(
    a: ArrayType1D, chunk_lengths: List[int]
) -> List[np.ndarray]:
    """
    Split an array into chunks with optimized handling for PyArrow ChunkedArrays.

    This function efficiently splits arrays, with special optimizations for PyArrow
    ChunkedArrays where the existing chunks align with the desired split boundaries.
    When chunk boundaries align, it avoids expensive concatenation and re-splitting
    operations by directly converting existing chunks to numpy arrays.

    Parameters
    ----------
    arr : ArrayType1D
        The array to split. Can be numpy array, pandas Series, PyArrow Array or
        ChunkedArray, or any type supported by `to_arrow()`.
    chunk_lengths : list of int
        List of desired chunk lengths. Must sum to the total length of the array.

    Returns
    -------
    list of numpy.ndarray
        List of numpy arrays corresponding to the requested chunks.

    Raises
    ------
    ValueError
        If the sum of chunk_lengths does not equal the length of the input array.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5, 6])
    >>> chunk_lengths = [2, 3, 1]
    >>> chunks = array_split_with_chunk_handling(arr, chunk_lengths)
    >>> [chunk.tolist() for chunk in chunks]
    [[1, 2], [3, 4, 5], [6]]

    >>> # Optimized case with PyArrow ChunkedArray
    >>> chunked_arr = pa.chunked_array([pa.array([1, 2]), pa.array([3, 4, 5]), pa.array([6])])
    >>> chunks = array_split_with_chunk_handling(chunked_arr, [2, 3, 1])
    >>> [chunk.tolist() for chunk in chunks]
    [[1, 2], [3, 4, 5], [6]]
    """
    if sum(chunk_lengths) != len(a):
        raise ValueError(
            f"Sum of chunk_lengths ({sum(chunk_lengths)}) must equal array length ({len(a)}). "
            f"Got chunk_lengths: {chunk_lengths}"
        )

    offsets = np.cumsum(chunk_lengths)[:-1]
    arr_list = _val_to_numpy(a, as_list=True)
    if len(arr_list) > 1:
        if len(arr_list) == len(chunk_lengths) and all(
            len(c) == k for c, k in zip(arr_list, chunk_lengths)
        ):
            return arr_list
        else:
            arr = np.concatenate(arr_list)
    else:
        arr = arr_list[0]
    return np.array_split(arr, offsets)


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
    np.ndarray | NumbaList[np.ndarray]
        NumPy array representation of the input, as a list of arrays or a single array,
    """
    try:
        arrow: pa.Array = to_arrow(val)
        is_chunked = isinstance(
            arrow,
            pa.ChunkedArray,
        )
    except TypeError:
        is_chunked = False

    if is_chunked:
        val_list = [chunk.to_numpy() for chunk in arrow.chunks]
    elif hasattr(val, "to_numpy"):
        val_list = [val.to_numpy()]  # type: ignore
    else:
        val_list = [np.asarray(val)]

    if as_list:
        return NumbaList(val_list)
    else:
        if len(val_list) > 1:
            val = np.concatenate(val_list)
        else:
            val = val_list[0]
        return val


def convert_data_to_arr_list_and_keys(
    data,
) -> Tuple[List[ArrayType1D], List[str]]:
    """
    Convert various array-like inputs to a dictionary of named arrays.

    Parameters
    ----------
    data : Various types
        Input arrays in various formats (Mapping, list/tuple of arrays, 2D array,
        pandas/polars Series or DataFrame)

    Returns
    -------
    dict
        Dictionary mapping array names to arrays

    Raises
    ------
    TypeError
        If the input type is not supported
    """
    if isinstance(data, Mapping):
        array = dict(data)
        return list(array.values()), list(array.keys())
    elif isinstance(data, (tuple, list)):
        if np.ndim(data[0]) == 0:
            try:
                data = np.array(data)
            except ValueError:
                raise ValueError(
                    "Could not convert list input containing scalars to an array"
                )
            return convert_data_to_arr_list_and_keys(data)
        names = map(get_array_name, data)
        return list(data), list(names)
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        return convert_data_to_arr_list_and_keys(list(data.T))
    elif isinstance(
        data,
        (
            pd.Series,
            pl.Series,
            np.ndarray,
            pd.Index,
            pd.Categorical,
            pa.ChunkedArray,
            pa.Array,
        ),
    ):
        name = get_array_name(data)
        return [data], [name]
    elif isinstance(data, (pl.DataFrame, pl.LazyFrame, pd.DataFrame)):
        if isinstance(data, pl.LazyFrame):
            # Collect LazyFrame to DataFrame first
            data = data.collect()
        names = list(data.columns)
        return [data[key] for key in names], names
    else:
        raise TypeError(f"Input type {type(data)} not supported")


def pretty_cut(x: ArrayType1D, bins: ArrayType1D | List, precision: int = None):
    """
    Create a categorical with pretty labels by cutting data into bins.

    Parameters
    ----------
    x : ArrayType1D
        1-D array-like data to be binned. Can be np.ndarray, pd.Series,
        pl.Series, pd.Index, or pd.Categorical.
    bins : ArrayType1D or list
        Monotonically increasing array of bin edges, defining the intervals.
        Values will be sorted internally.

    Returns
    -------
    pd.Categorical or pd.Series
        Categorical with human-readable interval labels. If input `x` is a
        pd.Series, returns pd.Series with same index and name; otherwise
        returns pd.Categorical.

    Notes
    -----
    The function creates interval labels with the following format:
    - First bin: " <= {first_bin_edge}"
    - Middle bins: "{left_edge + 1} - {right_edge}" for integer data,
                   "{left_edge} - {right_edge}" for float data
    - Last bin: " > {last_bin_edge}"

    For integer data, the left edge is incremented by 1 to create
    non-overlapping intervals. NaN values in float arrays are assigned
    code -1 (missing category).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> x = pd.Series([1, 5, 10, 15, 20])
    >>> bins = [5, 10, 15]
    >>> result = pretty_cut(x, bins)
    >>> result.categories
    Index([' <= 5', '6 - 10', '11 - 15', ' > 15'], dtype='object')
    """
    bins = np.sort(bins)
    np_type = np.asarray(x).dtype
    is_integer = np_type.kind in "ui" and bins.dtype.kind in "ui"

    if precision is None and not is_integer:

        def get_decimals(x):
            x = str(x)
            int, *decimals = str(x).split(".")
            return len(decimals)

        precision = max(map(get_decimals, bins))

    labels = [f" <= {bins[0]}"]
    for left, right in zip(bins, bins[1:]):
        if is_integer:
            left = str(left + is_integer)
            right = str(right)
        else:
            left, right = (f"{x:.{precision}f}" for x in [left, right])
        if left == right:
            labels.append(str(left))
        else:
            labels.append(f"{left} - {right}")
    labels.append(f" > {bins[-1]}")
    codes = bins.searchsorted(x)
    if np_type.kind == "f":
        codes[np.isnan(x)] = -1
    out = pd.Categorical.from_codes(codes, labels)
    if isinstance(x, pd.Series):
        out = pd.Series(out, index=x.index, name=x.name)

    return out


@nb.njit(parallel=True)
def _nb_dot(a: List[np.ndarray], b: np.ndarray, out: np.ndarray) -> np.ndarray:
    for row in nb.prange(len(a[0])):
        for col in nb.prange(len(b)):
            out[row] += a[col][row] * b[col]
    return out


def nb_dot(a: Union[np.ndarray, pd.DataFrame, pl.DataFrame], b: ArrayType1D):
    if isinstance(a, np.ndarray) and a.ndim != 2:
        raise ValueError("a must be a 2-dimensional array or DataFrame")
    if a.shape[1] != len(b):
        raise ValueError(f"shapes {a.shape} and {b.shape} are not aligned. ")
    if isinstance(a, np.ndarray):
        arr_list = a.T
    else:
        arr_list = NumbaList([np.asarray(a[col]) for col in a.columns])

    kinds = [a.dtype.kind for a in arr_list]
    return_type = np.float64 if "f" in kinds else np.int64

    if not len(a):
        out = np.zeros(0, dtype=return_type)
    else:
        out = _nb_dot(arr_list, np.asarray(b), out=np.zeros(len(a), dtype=return_type))
    if isinstance(a, pd.DataFrame):
        out = pd.Series(out, a.index)
    return out


def bools_to_categorical(
    df: pd.DataFrame, sep: str = " & ", na_rep="None", allow_duplicates=True
):
    """
    Convert a boolean DataFrame to a categorical Series with combined labels.

    This function creates a categorical representation where each unique row
    pattern in the boolean DataFrame becomes a category. Column names where
    True values occur are joined with a separator to form the category labels.

    Parameters
    ----------
    df : pd.DataFrame
        Boolean DataFrame where each column represents a feature/condition
        and each row represents an observation.
    sep : str, default " & "
        Separator string used to join column names when multiple columns
        are True in the same row.
    na_rep : str, default "None"
        String representation for rows where all values are False.
        Must not match any column name in the DataFrame.
    allow_duplicates : bool, default True
        If True, allows multiple True values per row (joined with separator).
        If False, raises ValueError when any row has more than one True value.

    Returns
    -------
    pd.Series
        Series with categorical dtype containing the combined labels.
        Index matches the input DataFrame's index.

    Raises
    ------
    ValueError
        If `na_rep` matches any column name in the DataFrame, or if
        `allow_duplicates` is False and any row contains multiple True values.

    Notes
    -----
    The function uses numpy.unique to identify distinct row patterns,
    making it memory efficient for DataFrames with many repeated patterns.

    Categories are created in the order they appear in the unique row patterns,
    not necessarily in alphabetical order.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({
    ...     'A': [True, False, True, False],
    ...     'B': [False, True, False, True],
    ...     'C': [False, False, True, True]
    ... })
    >>> result = bools_to_categorical(df)
    >>> result.cat.categories.tolist()
    ['A', 'B', 'A & C', 'B & C']

    >>> # Custom separator
    >>> result = bools_to_categorical(df, sep=' | ')
    >>> result[2]  # Row with A=True, C=True
    'A | C'

    >>> # All False row handling
    >>> df_with_empty = pd.DataFrame({
    ...     'X': [True, False],
    ...     'Y': [False, False]
    ... })
    >>> result = bools_to_categorical(df_with_empty, na_rep='Empty')
    >>> result[1]
    'Empty'
    """
    if na_rep in df:
        raise ValueError(f"na_rep={na_rep} clashes with one of the column names")
    min_bits = min([x for x in [8, 16, 32, 64] if x > df.shape[1]])
    bit_mask = nb_dot(df, 2 ** np.arange(df.shape[1], dtype=f"int{min_bits}"))
    uniques, codes = np.unique(bit_mask, return_inverse=True)

    cats = []
    for bit_mask in uniques:
        labels = []
        for i, col in enumerate(df.columns):
            if bit_mask & 2**i:
                labels.append(col)
        if labels:
            if not allow_duplicates and len(labels) > 1:
                raise ValueError(
                    "Some rows have more than one True value and allow_duplicates is False"
                )
            cat = sep.join(labels)
        else:
            cat = na_rep
        cats.append(cat)

    out = pd.Categorical.from_codes(codes, cats)
    out = pd.Series(out, index=df.index)

    return out


def mean_from_sum_count(sum_: pd.Series, count: pd.Series):
    """
    Compute mean from sum and count, handling datetime and timedelta types.
    Parameters
    ----------
    sum_ : pd.Series
        Series containing the sum values.
    count : pd.Series
        Series containing the count values.

    Returns
    -------
    pd.Series
    Series containing the computed mean values.

    """
    if sum_.dtype.kind in "mM":
        return (sum_.astype("int64") // count).astype(sum_.dtype)
    else:
        return sum_ / count
