from typing import Union, Tuple

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import numba as nb
from pandas.core.sorting import get_group_index

from pandas_plus.util import (
    get_array_name,
    parallel_map,
    ArrayType1D,
    _val_to_numpy,
    _maybe_cast_timestamp_arr,
)


def factorize_arrow_arr(
    arr: Union[pa.Array, pa.ChunkedArray, pl.Series, pd.Series],
) -> "tuple[np.ndarray, np.ndarray | pd.Index]":
    """
    Method for factorizing the arrow arrays, including polars Series and Pandas Series backed by pyarrow
    """
    name = get_array_name(arr)
    if isinstance(arr, pl.Series):
        arr = arr.to_arrow()
    elif isinstance(arr, pd.Series):
        arr = pa.Array.from_pandas(arr)
    elif isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    arr = arr.dictionary_encode()
    if isinstance(arr, pa.ChunkedArray):
        arr = arr.combine_chunks()

    codes = arr.indices.to_numpy(zero_copy_only=False)
    labels = pd.Index(arr.dictionary.to_numpy(zero_copy_only=False), name=name)

    return codes, labels


@nb.njit
def _monotonic_factorization(arr_list, total_len):
    codes = np.empty(total_len, dtype=np.uint32)
    labels = np.empty(total_len, dtype=arr_list[0].dtype)

    arr_num = 0
    arr = arr_list[arr_num]

    labels[0] = arr[0]
    n_labels = 1
    codes[0] = 0
    prev = arr[0]

    cur_arr_pos = 0
    for i in range(1, total_len):
        cur_arr_pos += 1
        if cur_arr_pos == len(arr):
            arr_num += 1
            arr = arr_list[arr_num]
            cur_arr_pos = 0

        x = arr[cur_arr_pos]
        if x < prev:
            return i, codes, labels[:n_labels]
        elif x > prev:
            labels[n_labels] = x
            n_labels += 1
        codes[i] = n_labels - 1
        prev = x

    return i + 1, codes, labels[:n_labels]


def monotonic_factorization(arr: ArrayType1D) -> Tuple[int, np.ndarray, np.ndarray]:
    """
    Factorize an array using optimized monotonic factorization.

    This function attempts to factorize an array by assuming it is monotonically
    increasing. It provides a significant performance optimization for arrays that
    are sorted or nearly sorted (such as date/time buckets, cumulative counts, etc.).
    The function exits early as soon as it detects non-monotonicity to avoid wasted
    computation and memory allocation.

    Parameters
    ----------
    arr : ArrayType1D
        Input array to factorize. Can be numpy array, pandas Series/Index/Categorical,
        polars Series, or PyArrow Array/ChunkedArray.

    Returns
    -------
    cutoff : int
        Index position where monotonicity was broken, or len(arr) if the entire
        array is monotonic. This indicates how many elements were successfully
        processed using the optimized monotonic approach.
    codes : np.ndarray
        Integer codes representing the factorized values. Only elements up to
        `cutoff` contain valid codes; remaining elements are uninitialized.
        Shape: (len(arr),), dtype: np.uint32
    labels : np.ndarray
        Unique values found during monotonic factorization. Only elements up to
        the number of unique values found are valid; remaining elements are
        uninitialized. The dtype matches the original array's dtype.

    Notes
    -----
    This is an optimization function that should be called before falling back
    to general factorization methods. It's particularly effective for:

    - Time series data with increasing timestamps
    - Cumulative counts or IDs
    - Pre-sorted categorical data
    - Sequential data with natural ordering

    If the function returns cutoff < len(arr), the caller should fall back to
    general factorization methods for the complete array.

    Examples
    --------
    >>> arr = np.array([1, 2, 3, 4, 5])  # Fully monotonic
    >>> cutoff, codes, labels = monotonic_factorization(arr)
    >>> cutoff
    5
    >>> codes
    array([0, 1, 2, 3, 4], dtype=uint32)
    >>> labels
    array([1, 2, 3, 4, 5])

    >>> arr = np.array([1, 2, 3, 1, 5])  # Breaks monotonicity at index 3
    >>> cutoff, codes, labels = monotonic_factorization(arr)
    >>> cutoff
    3
    """
    arr_list = _val_to_numpy(arr, as_list=True)
    arr_list, orig_types = zip(*list(map(_maybe_cast_timestamp_arr, arr_list)))
    orig_type = orig_types[0]

    total_len = len(arr)
    cutoff, codes, labels = _monotonic_factorization(arr_list, total_len)
    labels = labels.astype(orig_type)
    return cutoff, codes, labels


def factorize_1d(
    values,
    sort: "bool" = False,
    size_hint: "int | None" = None,
) -> "tuple[np.ndarray, np.ndarray | pd.Index]":
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
    if isinstance(values, (pl.Series, pa.Array, pa.ChunkedArray)) or (
        hasattr(values, "dtype") and isinstance(values.dtype, pd.ArrowDtype)
    ):
        return factorize_arrow_arr(values)

    if not isinstance(values, pd.Series):
        values = pd.Series(values)

    if isinstance(values.dtype, pd.CategoricalDtype):
        cat = values.cat
        codes = np.asarray(cat.codes)
        labels = pd.Categorical(cat.categories, ordered=cat.ordered)
        labels = pd.Index(labels, name=values.name)
        return codes, labels
    elif pd.api.types.is_bool_dtype(values):
        codes = np.asarray(values).view("int8")
        labels = pd.Index([False, True], name=values.name)
        return codes, labels
    else:
        codes, uniques = pd.factorize(values.values, use_na_sentinel=True)

        # Handle sorting manually if needed
        if sort and len(uniques) > 0:
            try:
                sort_idx = np.argsort(uniques)
                uniques = uniques[sort_idx]
                # Remap codes
                null = codes == -1
                codes[:] = np.argsort(sort_idx)[codes]
                codes[null] = -1
            except (TypeError, ValueError):
                # If sorting fails, just return unsorted
                pass

        return codes, pd.Index(uniques, name=values.name)


def factorize_2d(*vals, sort: bool = False):
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
    sort : bool, default False
        If True, the unique combinations of values will be sorted before
        factorization. If False, the order of combinations will be based on
        the order in which they first appear in the input arrays.

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
    factored = parallel_map(lambda x: factorize_1d(x, sort=sort), list(zip(vals)))
    codes, labels = zip(*factored)
    multi_codes = get_group_index(
        codes, tuple(map(len, labels)), sort=False, xnull=True
    )
    from pandas.core.reshape.util import cartesian_product

    index = pd.MultiIndex(
        codes=cartesian_product([np.arange(len(lvl)) for lvl in labels]),
        levels=labels,
        names=[get_array_name(v) for v in vals],
    )
    return multi_codes, index
