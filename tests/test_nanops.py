import pytest

parametrize = pytest.mark.parametrize
import numpy as np
import pandas as pd

from pandas_plus import nanops


@pytest.fixture(scope="session")
def arr():
    return np.random.rand(100)


@pytest.fixture(scope="session")
def series(arr):
    return pd.Series(arr)


@parametrize("op_name", ["min", "max", "sum", "count"])
@parametrize("with_nans", [False, True])
@parametrize("skipna", [True, False])
@parametrize("n_threads", [1, 2])
@parametrize("integers", [True, False])
def test_reduce_1d(series, op_name, with_nans, skipna, n_threads, integers):
    if integers:
        series = (series * 100).round().astype(int)
    if with_nans:
        series = series.where(series.index > 1)
    result = nanops.reduce_1d(
        op_name, series.values, skipna=skipna, n_threads=n_threads
    )
    expected = (
        series.count() if op_name == "count" else series.agg(op_name, skipna=skipna)
    )
    np.testing.assert_array_almost_equal(result, expected)


@parametrize("op_name", ["min", "max", "count"])
@parametrize("with_nans", [False, True])
@parametrize("skipna", [True, False])
@parametrize("n_threads", [1, 2])
@parametrize("dtype", ['datetime64[ns]', 'timedelta64[ns]'])
def test_reduce_1d_timestamps(op_name, with_nans, skipna, n_threads, dtype):
    series = pd.Series(np.arange(100, 200), dtype=dtype)
    if with_nans:
        series = series.where(series.index > 1)
    result = nanops.reduce_1d(
        op_name, series.values, skipna=skipna, n_threads=n_threads
    )
    if op_name == "count":
        assert result == series.count()
    else:
        expected = series.agg(op_name, skipna=skipna)
        # testing scalars here but using Series as a hack to equate pd.NaT
        pd.testing.assert_series_equal(pd.Series(expected), pd.Series(result))


@parametrize("op_name", ["min", "max", "sum", "mean", "std", "var"])
@parametrize("with_nans", [False, True])
@parametrize("skipna", [True, False])
@parametrize("n_threads", [1, 2])
def test_nanops_1d(series, op_name, with_nans, skipna, n_threads):
    if with_nans:
        series = series.where(series.index > 1)
    func = getattr(nanops, f"nan{op_name}")
    result = func(series.values, skipna=skipna)
    expected = series.agg(op_name, skipna=skipna)
    np.testing.assert_array_almost_equal(result, expected)


@parametrize("op_name", ["min", "max", "sum", "mean", "std", "var"])
@parametrize("n_threads", [1, 2])
def test_nanops_multi_thread_nans(series, op_name, n_threads):
    series = series.where(series.index == 0)  # only one non-null value to ensure one thread is all null
    func = getattr(nanops, f"nan{op_name}")
    result = func(series.values, skipna=True, n_threads=n_threads)
    expected = series.agg(op_name, skipna=True)
    np.testing.assert_array_almost_equal(result, expected)


@parametrize("op_name", ["min", "max", "sum", "mean", "std", "var"])
def test_nanops_all_nans(op_name):
    series = pd.Series([np.nan] * 3)
    func = getattr(nanops, f"nan{op_name}")
    result = func(series.values, skipna=True)
    expected = series.agg(op_name, skipna=True)
    np.testing.assert_array_almost_equal(result, expected)


@parametrize("op_name", ["min", "max", "sum", "count"])
@parametrize("with_nans", [False, True])
@parametrize("skipna", [True, False])
@parametrize("n_threads", [1, 2])
@parametrize("integers", [True, False])
def test_reduce_2d(series, op_name, with_nans, skipna, n_threads, integers):
    if integers:
        series = (series * 100).round().astype(int)
    if with_nans:
        series = series.where(series.index > 1)

    arr = series.values.reshape((len(series) // 2, 2))
    result = nanops.reduce_2d(
        op_name, arr, skipna=skipna, n_threads=n_threads
    )
    df = pd.DataFrame(arr)
    expected = (
        df.count() if op_name == "count" else df.agg(op_name, skipna=skipna)
    )
    np.testing.assert_array_almost_equal(result, expected.values)
