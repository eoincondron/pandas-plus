import numpy as np
import pandas as pd
import pytest

from pandas_plus.groupby import GroupBy
from pandas_plus.util import MIN_INT, _null_value_for_array_type


class TestGroupBy:

    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
    @pytest.mark.parametrize("key_dtype", [int, str, float, "float32"])
    @pytest.mark.parametrize("key_type", [np.array, pd.Series])
    @pytest.mark.parametrize("value_dtype", [int, float, "float32", bool])
    @pytest.mark.parametrize("value_type", [np.array, pd.Series])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_basic(
        self, method, key_dtype, key_type, value_dtype, value_type, use_mask
    ):
        key = key_type([1, 1, 2, 1, 3, 3, 6, 1, 6], dtype=key_dtype)
        value = value_type([-1, 0.3, 4, 3.5, 8, 6, 3, 1, 12.6]).astype(value_dtype)

        if use_mask:
            mask = pd_mask = key.astype(int) != 1
        else:
            mask = None
            pd_mask = slice(None)
        result = getattr(GroupBy, method)(key, value, mask=mask)

        expected = getattr(pd.Series(value)[pd_mask].groupby(key[pd_mask]), method)()

        pd.testing.assert_series_equal(result, expected, check_dtype=method != "mean")

        gb = GroupBy(key)
        result = getattr(gb, method)(value, mask=mask)
        pd.testing.assert_series_equal(result, expected, check_dtype=method != "mean")

    @pytest.mark.parametrize("use_mask", [True, False])
    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
    def test_floats_with_nulls(self, method, use_mask):
        key = pd.Series([1, 1, 2, 1, 3, 3, 6, 1, 6])
        series = pd.Series([.1, 0, 3.5, 3, 8, 6, 7, 1, 1.2],)
        null_mask = key.isin([2, 6])
        series = series.where(~null_mask)
        if use_mask:
            mask = key != 3
            pd_mask = mask
        else:
            mask = None
            pd_mask = slice(None)
        result = getattr(GroupBy, method)(key, series, mask=mask)
        expected = series[pd_mask].groupby(key[pd_mask]).agg(method).astype(result.dtype)
        pd.testing.assert_series_equal(result, expected)


    @pytest.mark.parametrize("use_mask", [True, False])
    @pytest.mark.parametrize("method", ["mean", "min", "max"])
    @pytest.mark.parametrize("dtype", ["datetime64[ns]", "timedelta64[ns]"])
    def test_timestamps_with_nulls(self, method, use_mask, dtype):
        key = pd.Series([1, 1, 2, 1, 3, 3, 6, 1, 6])
        series = pd.Series(np.arange(len(key)), dtype=dtype)
        null_mask = key.isin([2, 6])
        series = series.where(~null_mask)
        if use_mask:
            mask = key != 3
            pd_mask = mask
        else:
            mask = None
            pd_mask = slice(None)
        result = getattr(GroupBy, method)(key, series, mask=mask)
        expected = series[pd_mask].groupby(key[pd_mask]).agg(method).astype(result.dtype)
        pd.testing.assert_series_equal(result, expected)
