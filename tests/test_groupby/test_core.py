import numpy as np
import pandas as pd
import pytest

from pandas_plus.groupby import GroupBy


def assert_pd_equal(left, right, **kwargs):
    if isinstance(left, pd.Series):
        pd.testing.assert_series_equal(left, right, **kwargs)
    else:
        pd.testing.assert_frame_equal(left, right, **kwargs)


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
        index = pd.RangeIndex(2, 11)
        key = key_type(pd.Series([1, 1, 2, 1, 3, 3, 6, 1, 6], index=index, dtype=key_dtype))
        values = value_type(pd.Series([-1, 0.3, 4, 3.5, 8, 6, 3, 1, 12.6], index=index)).astype(value_dtype)

        if use_mask:
            mask = pd_mask = key.astype(int) != 1
        else:
            mask = None
            pd_mask = slice(None)

        result = getattr(GroupBy, method)(key, values, mask=mask)

        expected = getattr(pd.Series(values, index=index)[pd_mask].groupby(key[pd_mask]), method)()
        pd.testing.assert_series_equal(result, expected, check_dtype=method != "mean")

        gb = GroupBy(key)
        result = getattr(gb, method)(values, mask=mask)
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

    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
    @pytest.mark.parametrize("value_type", [np.array, list])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_2d_variants(
            self, method, value_type, use_mask
    ):
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        values = value_type([
            np.random.rand(len(key)), np.random.randint(0, 9, len(key))
        ])

        if use_mask:
            mask = pd_mask = key != 1
        else:
            mask = None
            pd_mask = slice(None)

        result = getattr(GroupBy, method)(key, values.T if value_type is np.array else values, mask=mask)

        compare_df = pd.DataFrame(dict(zip(['_arr_0', '_arr_1'], values)))
        expected = getattr(compare_df[pd_mask].groupby(key[pd_mask]), method)()
        pd.testing.assert_frame_equal(result, expected, check_dtype=method != "mean")

    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
    @pytest.mark.parametrize("value_type", [pd.DataFrame, dict])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_mapping_variants(
            self, method, value_type, use_mask
    ):
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        values = value_type(dict(
            a=np.random.rand(len(key)),
            b=np.random.randint(0, 9, len(key)),
        ))
        if use_mask:
            mask = pd_mask = key != 1
        else:
            mask = None
            pd_mask = slice(None)

        result = getattr(GroupBy, method)(key, values, mask=mask)

        expected = getattr(pd.DataFrame(values)[pd_mask].groupby(key[pd_mask]), method)()
        pd.testing.assert_frame_equal(result, expected, check_dtype=method != "mean")

    @pytest.mark.parametrize("agg_func", ["sum", "mean", "min", "max"])
    @pytest.mark.parametrize("value_type",[pd.Series, pd.DataFrame])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_agg_single_func_mode(
            self, agg_func, value_type, use_mask
    ):
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        values = pd.Series(np.random.rand(len(key)))
        if value_type is pd.DataFrame:
            values = pd.DataFrame(dict(a=values, b=values * 2))

        if use_mask:
            mask = pd_mask = key != 1
        else:
            mask = None
            pd_mask = slice(None)

        result = GroupBy.agg(key, values, agg_func=agg_func, mask=mask)

        expected = values[pd_mask].groupby(key[pd_mask]).agg(agg_func)
        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("value_type", [pd.DataFrame, dict])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_agg_multi_func_mode(
            self, value_type, use_mask
    ):
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        values = value_type(dict(
            b=np.random.rand(len(key)),
            a=np.random.randint(0, 9, len(key)),
        ))
        if use_mask:
            mask = pd_mask = key != 1
        else:
            mask = None
            pd_mask = slice(None)

        result = GroupBy.agg(key, values, agg_func=['mean', 'sum'], mask=mask)
        expected = pd.DataFrame(values)[pd_mask].groupby(key[pd_mask]).agg({'b': 'mean', 'a': 'sum'})
        pd.testing.assert_frame_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("categorical", [False, True])
    def test_null_keys(self, categorical):
        key = pd.Series([1, 1, 2, 1, 3, 3, 6, 1, 6])
        if categorical:
            key = key.astype("category")
        values = pd.Series(np.random.rand(len(key)))
        key.iloc[1] = np.nan
        result = GroupBy.sum(key, values)
        expected = values.groupby(key, observed=True).sum()
        assert(result == expected).all()

        # Test with mask
        mask = key != 1
        result_masked = GroupBy.sum(key, values, mask=mask)
        expected_masked = values[mask].groupby(key[mask], observed=True).sum()
        breakpoint()
        assert(result_masked == expected_masked).all()
