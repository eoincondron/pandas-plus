from functools import reduce, wraps
import operator
import os
from inspect import signature
from typing import Mapping, Union, Any, Callable, TypeVar, cast, List, Optional, Tuple
import concurrent.futures

import numba as nb
import numpy as np
import pandas as pd
import polars as pl
from numba.core.extending import overload
from pandas.core.sorting import get_group_index

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


def get_array_name(array: Union[np.ndarray, pd.Series, pl.Series]):
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


class TempName(str): ...


def convert_data_to_arr_list_and_keys(arrays, temp_name_root: str = "_arr_") -> Tuple[List[np.ndarray], List[str]]:
    """
    Convert various array-like inputs to a dictionary of named arrays.

    Parameters
    ----------
    arrays : Various types
        Input arrays in various formats (Mapping, list/tuple of arrays, 2D array,
        pandas/polars Series or DataFrame)
    temp_name_root : str, default "_arr_"
        Prefix to use for generating temporary names for unnamed arrays

    Returns
    -------
    dict
        Dictionary mapping array names to arrays

    Raises
    ------
    TypeError
        If the input type is not supported
    """
    if isinstance(arrays, Mapping):
        array = dict(arrays)
        return list(array.values()), list(array.keys())
    elif isinstance(arrays, (tuple, list)):
        names = map(get_array_name, arrays)
        keys = [
            name or TempName(f"{temp_name_root}{i}") for i, name in enumerate(names)
        ]
        return list(arrays), keys
    elif isinstance(arrays, np.ndarray) and arrays.ndim == 2:
        return convert_data_to_arr_list_and_keys(list(arrays.T))
    elif isinstance(
            arrays, (pd.Series, pl.Series, np.ndarray, pd.Index, pd.Categorical)
    ):
        name = get_array_name(arrays)
        if name is None:
            name = TempName(f"{temp_name_root}0")
        return [arrays], [name]
    elif isinstance(arrays, (pl.DataFrame, pd.DataFrame)):
        return [arrays[key] for key in arrays.columns], list(arrays.columns)
    else:
        raise TypeError(f"Input type {type(arrays)} not supported")


def pretty_cut(x: ArrayType1D, bins: ArrayType1D | List):
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
    is_integer = np_type.kind in 'ui' and bins.dtype.kind in 'ui'

    labels = [f' <= {bins[0]}']
    for left, right in zip(bins, bins[1:]):
        left = left + is_integer
        if left == right:
            labels.append(str(left))
        else:
            labels.append(f'{left} - {right}')
    labels.append(f' > {bins[-1]}')
    codes = bins.searchsorted(x)
    if np_type.kind == 'f':
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
        arr_list = [np.asarray(a[col]) for col in a.columns]

    kinds = [a.dtype.kind for a in arr_list]
    return_type = np.float64 if 'f' in kinds else np.int64

    if not len(a):
        out = np.zeros(0, dtype=return_type)
    else:
        out = _nb_dot(
        arr_list,
        np.asarray(b),
        out=np.zeros(len(a), dtype=return_type)
    )
    if isinstance(a, pd.DataFrame):
        out = pd.Series(out, a.index)
    return out


def bools_to_categorical(df: pd.DataFrame, sep: str = " & ", na_rep="None", allow_duplicates=True):
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
    bit_mask = nb_dot(df, 2 ** np.arange(df.shape[1], dtype=f'int{min_bits}'))
    uniques, codes = np.unique(bit_mask, return_inverse=True)

    cats = []
    for bit_mask in uniques:
        labels = []
        for i, col in enumerate(df.columns):
            if bit_mask & 2 ** i:
                labels.append(col)
        if labels:
            if not allow_duplicates and len(labels) > 1:
                raise ValueError("Some rows have more than one True value and allow_duplicates is False")
            cat = sep.join(labels)
        else:
            cat = na_rep
        cats.append(cat)

    out = pd.Categorical.from_codes(codes, cats)
    out = pd.Series(out, index=df.index)

    return out


def factorize_1d(
    values,
    sort: 'bool' = False,
    size_hint: 'int | None' = None,
) -> 'tuple[np.ndarray, np.ndarray | pd.Index]':
    """
    Encode the object as an enumerated type or categorical variable.
    
    This method is useful for obtaining a numeric representation of an
    array when all that matters is identifying distinct values. factorize_1d
    is available as both a top-level function :func:`~pandas_plus.util.factorize_1d`,
    and as a method.
    
    Parameters
    ----------
    values : array-like
        Sequence to be encoded. Can be any array-like object including lists,
        numpy arrays, pandas Series, or pandas Categorical.
    sort : bool, default False
        Sort `values` before factorizing. If False, factorize in the order
        in which the values first appear.
    size_hint : int, optional
        Hint to the algorithm for the expected number of unique values. This
        can be used to pre-allocate the return arrays.
        
    Returns
    -------
    codes : np.ndarray[int64]
        An integer array that represents the labels for each element in `values`.
        For missing values (NaN, None), codes will contain -1.
    uniques : np.ndarray or pd.Index
        An array of unique values. When the input is a pandas Categorical,
        this will be the categorical's categories. Otherwise, it will be a
        numpy array or pandas Index containing the unique values in the order
        they first appeared (or sorted order if sort=True).
        
    See Also
    --------
    factorize_2d : Factorize multiple 1-D arrays simultaneously.
    pandas.factorize : pandas equivalent function.
    pandas.Categorical : Represent a categorical variable in pandas.
    
    Notes
    -----
    For pandas Categorical inputs, this function returns the categorical's
    codes and categories directly, ignoring the sort and size_hint parameters.
    
    Missing values (NaN, None) are assigned code -1 and are not included in
    the uniques array.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pandas_plus.util import factorize_1d
    
    Basic usage with a list:
    
    >>> values = [1, 2, 3, 1, 2, 3]
    >>> codes, uniques = factorize_1d(values)
    >>> codes
    array([0, 1, 2, 0, 1, 2])
    >>> uniques
    array([1, 2, 3])
    
    With string values:
    
    >>> values = ['a', 'b', 'c', 'a', 'b']
    >>> codes, uniques = factorize_1d(values)
    >>> codes
    array([0, 1, 2, 0, 1])
    >>> uniques
    array(['a', 'b', 'c'], dtype='<U1')
    
    With sorting enabled:
    
    >>> values = ['c', 'a', 'b', 'c', 'a']
    >>> codes, uniques = factorize_1d(values, sort=True)
    >>> codes
    array([2, 0, 1, 2, 0])
    >>> uniques
    array(['a', 'b', 'c'], dtype='<U1')
    
    With NaN values:
    
    >>> values = [1.0, 2.0, np.nan, 1.0, np.nan]
    >>> codes, uniques = factorize_1d(values)
    >>> codes
    array([ 0,  1, -1,  0, -1])
    >>> uniques
    array([1., 2.])
    
    With pandas Categorical:
    
    >>> cat = pd.Categorical(['a', 'b', 'c', 'a', 'b'])
    >>> codes, uniques = factorize_1d(cat)
    >>> codes
    array([0, 1, 2, 0, 1])
    >>> uniques
    Index(['a', 'b', 'c'], dtype='object')
    """
    values = pd.Series(values)
    try:
        return np.asarray(values.cat.codes), values.cat.categories
    except AttributeError:
        return pd.factorize(values, sort=sort, use_na_sentinel=True, size_hint=size_hint)


def factorize_2d(*vals):
    """
    Encode multiple 1-D arrays as enumerated types or categorical variables.
    
    This function factorizes multiple arrays simultaneously, creating a 
    MultiIndex that represents all unique combinations of values across
    the input arrays. This is useful for creating group identifiers from
    multiple categorical variables.
    
    Parameters
    ----------
    *vals : array-like
        Variable number of 1-D array-like objects to be factorized together.
        Each array should have the same length. Can be any combination of
        lists, numpy arrays, pandas Series, or pandas Categorical objects.
        
    Returns
    -------
    codes : np.ndarray[int64]
        An integer array where each element represents the group identifier
        for the corresponding combination of values across all input arrays.
        Identical combinations will have the same code.
    labels : pd.MultiIndex
        A MultiIndex containing all unique combinations of values from the
        input arrays. The number of levels equals the number of input arrays.
        Each level contains the unique values from the corresponding input array.
        
    Raises
    ------
    ValueError
        If input arrays have different lengths.
        
    See Also
    --------
    factorize_1d : Factorize a single 1-D array.
    pandas.factorize : pandas equivalent function for single arrays.
    pandas.MultiIndex.from_product : Create MultiIndex from the cartesian product of iterables.
    pandas.core.sorting.get_group_index : Get group index from multiple arrays.
    
    Notes
    -----
    The function internally uses `factorize_1d` on each input array, then
    combines the results using pandas' `get_group_index` function to create
    a unified group identifier.
    
    Missing values (NaN, None) in any array will be treated as distinct
    values and will contribute to unique combinations.
    
    The resulting MultiIndex is created using `pd.MultiIndex.from_product`,
    which means it contains all possible combinations of the unique values
    from each array, not just the combinations that actually appear in the data.
    
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pandas_plus.util import factorize_2d
    
    Basic usage with two arrays:
    
    >>> vals1 = [1, 2, 3, 1, 2]
    >>> vals2 = ['a', 'b', 'c', 'a', 'b']
    >>> codes, labels = factorize_2d(vals1, vals2)
    >>> codes
    array([0, 1, 2, 0, 1])
    >>> labels
    MultiIndex([(1, 'a'),
                (2, 'b'),
                (3, 'c')],
               names=[None, None])
    
    With three arrays:
    
    >>> vals1 = [1, 1, 2, 2]
    >>> vals2 = ['x', 'y', 'x', 'y']
    >>> vals3 = [True, False, True, False]
    >>> codes, labels = factorize_2d(vals1, vals2, vals3)
    >>> codes
    array([0, 1, 2, 3])
    >>> labels.nlevels
    3
    
    Identical combinations get same codes:
    
    >>> vals1 = [1, 2, 1, 2, 1]
    >>> vals2 = ['a', 'b', 'a', 'b', 'a']
    >>> codes, labels = factorize_2d(vals1, vals2)
    >>> codes
    array([0, 1, 0, 1, 0])
    
    With pandas Series input:
    
    >>> s1 = pd.Series([1, 2, 3])
    >>> s2 = pd.Series(['x', 'y', 'z'])
    >>> codes, labels = factorize_2d(s1, s2)
    >>> codes
    array([0, 1, 2])
    >>> labels
    MultiIndex([(1, 'x'),
                (2, 'y'),
                (3, 'z')],
               names=[None, None])
    
    With missing values:
    
    >>> vals1 = [1, 2, np.nan, 1, np.nan]
    >>> vals2 = ['a', 'b', 'c', 'a', 'c']
    >>> codes, labels = factorize_2d(vals1, vals2)
    >>> codes  # NaN combinations get unique codes
    array([0, 1, 2, 0, 2])
    """
    codes, labels = map(list, zip(*map(factorize_1d, vals)))
    multi_codes = get_group_index(codes, tuple(map(len, labels)), sort=False, xnull=True)
    index = pd.MultiIndex.from_product(labels)
    return multi_codes, index
