import numpy as np
import pandas as pd
import pyarrow as pa
import polars as pl
import pytest

from pandas_plus.groupby.core import GroupBy, add_row_margin, crosstab

from .conftest import assert_pd_equal


class TestGroupBy:

    @pytest.mark.parametrize(
        "method", ["sum", "mean", "min", "max", "var", "std", "first", "last"]
    )
    @pytest.mark.parametrize("key_dtype", [int, str, float, "float32", "category"])
    @pytest.mark.parametrize("key_type", [np.array, pd.Series, pl.Series])
    @pytest.mark.parametrize(
        "value_dtype", [int, float, "float32", bool, "double[pyarrow]"]
    )
    @pytest.mark.parametrize("value_type", [np.array, pd.Series, pl.Series])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_basic(
        self, method, key_dtype, key_type, value_dtype, value_type, use_mask
    ):
        if value_dtype == bool and method in ("var", "std"):
            return
        index = pd.RangeIndex(2, 11)
        key = pd.Series(
            [1, 1, 2, 1, 3, 3, 6, 1, 6],
            index=index,
            dtype=key_dtype,
        )
        values = pd.Series([-1, 0.3, 4, 3.5, 8, 6, 3, 1, 12.6], index=index).astype(
            value_dtype
        )

        if use_mask:
            mask = key != 1
            expected = values[mask].groupby(key[mask], observed=True).agg(method)
        else:
            mask = None
            expected = values.groupby(key, observed=True).agg(method)

        if key_dtype == "category" and key_type is not pd.Series:
            expected.index = np.array(expected.index)

        key = key_type(key)
        values = value_type(values)

        result = getattr(GroupBy, method)(key, values, mask=mask)

        assert_pd_equal(result, expected, check_dtype=False)
        assert result.dtype.kind == expected.dtype.kind

        gb = GroupBy(key)
        result = getattr(gb, method)(values, mask=mask)
        assert_pd_equal(result, expected, check_dtype=False)
        assert result.dtype.kind == expected.dtype.kind

    def test_pyarrow_dictionary_key(self):
        key = pl.Series("bar", ["a", "b"] * 3, dtype=pl.Categorical)
        values = pl.Series(
            "foo",
            np.arange(6),
        )
        result = GroupBy.sum(key, values)
        expected = values.to_pandas().groupby(key.to_pandas().astype(str)).sum()
        assert_pd_equal(result, expected)

        key = key.to_pandas(types_mapper=pd.ArrowDtype)
        result = GroupBy.sum(key, values)
        assert_pd_equal(result, expected)

    @pytest.mark.parametrize("use_mask", [True, False])
    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
    def test_floats_with_nulls(self, method, use_mask):
        key = pd.Series([1, 1, 2, 1, 3, 3, 6, 1, 6])
        series = pd.Series(
            [0.1, 0, 3.5, 3, 8, 6, 7, 1, 1.2],
        )
        null_mask = key.isin([2, 6])
        series = series.where(~null_mask)
        if use_mask:
            mask = key != 3
            pd_mask = mask
        else:
            mask = None
            pd_mask = slice(None)
        result = getattr(GroupBy, method)(key, series, mask=mask)
        expected = (
            series[pd_mask].groupby(key[pd_mask]).agg(method).astype(result.dtype)
        )
        assert_pd_equal(result, expected)

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
        expected = (
            series[pd_mask].groupby(key[pd_mask]).agg(method).astype(result.dtype)
        )
        assert_pd_equal(result, expected)

    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
    @pytest.mark.parametrize("value_type", [np.array, list])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_2d_variants(self, method, value_type, use_mask):
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        values = value_type(
            [
                np.random.rand(len(key)),
                np.random.randint(0, 9, len(key)),
                np.arange(len(key)) % 2 == 0,
            ]
        )

        if use_mask:
            mask = pd_mask = key != 1
        else:
            mask = None
            pd_mask = slice(None)

        result = getattr(GroupBy, method)(
            key, values.T if value_type is np.array else values, mask=mask
        )

        compare_df = pd.DataFrame(dict(zip(["_arr_0", "_arr_1", "_arr_2"], values)))
        expected = getattr(compare_df[pd_mask].groupby(key[pd_mask]), method)()
        assert_pd_equal(result, expected.astype(result.dtypes), dtype_kind_only=True)

    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
    @pytest.mark.parametrize("value_type", [pd.DataFrame, dict, pl.DataFrame])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_mapping_variants(self, method, value_type, use_mask):
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        value_dict = dict(
            a=np.random.rand(len(key)),
            b=np.random.randint(0, 9, len(key)),
        )
        values = value_type(value_dict)
        if use_mask:
            mask = pd_mask = key != 1
        else:
            mask = None
            pd_mask = slice(None)

        result = getattr(GroupBy, method)(key, values, mask=mask)

        expected = getattr(
            pd.DataFrame(value_dict)[pd_mask].groupby(key[pd_mask]), method
        )()
        assert_pd_equal(result, expected, dtype_kind_only=True)

    @pytest.mark.parametrize("agg_func", ["sum", "mean", "min", "max"])
    @pytest.mark.parametrize("value_type", [pd.Series, pd.DataFrame])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_agg_single_func_mode(self, agg_func, value_type, use_mask):
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
    def test_agg_multi_func_mode(self, value_type, use_mask):
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        values = value_type(
            dict(
                b=np.random.rand(len(key)),
                a=np.random.randint(0, 9, len(key)),
            )
        )
        if use_mask:
            mask = pd_mask = key != 1
        else:
            mask = None
            pd_mask = slice(None)

        result = GroupBy.agg(key, values, agg_func=["mean", "sum"], mask=mask)
        expected = (
            pd.DataFrame(values)[pd_mask]
            .groupby(key[pd_mask])
            .agg({"b": "mean", "a": "sum"})
        )
        assert_pd_equal(result, expected, check_dtype=False)

    def test_agg_multi_threading_small_data(self):
        """Test agg method with small data (single-threaded) and multiple agg functions."""
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        values = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])

        gb = GroupBy(key)
        assert gb._max_threads_for_numba == 1  # Small data should use single thread

        # Test with single value, multiple agg functions
        result = gb.agg(values, agg_func=["sum", "mean", "min", "max"])

        # Expected column names should match the agg function names
        expected_columns = ["sum", "mean", "min", "max"]
        assert list(result.columns) == expected_columns

        # Compare with pandas
        expected = values.groupby(key).agg(["sum", "mean", "min", "max"])
        expected.columns = expected_columns
        assert_pd_equal(result, expected, check_dtype=False)

    def test_agg_multi_threading_large_data(self):
        """Test agg method with large data (multi-threaded) and multiple agg functions."""
        np.random.seed(42)  # For reproducibility
        size = 2_000_000  # Large enough to trigger multi-threading
        key = pd.Series(np.random.randint(0, 100, size=size))
        values = pd.Series(np.random.rand(size))

        gb = GroupBy(key)
        assert gb._max_threads_for_numba > 1  # Large data should use multiple threads

        # Test with single value, multiple agg functions
        result = gb.agg(values, agg_func=["sum", "mean"])

        # Expected column names should match the agg function names
        expected_columns = ["sum", "mean"]
        assert list(result.columns) == expected_columns

        # Verify that results make sense (sums should be larger than means for most groups)
        assert (result["sum"] >= result["mean"]).all()

        # Test internal consistency: mean should roughly equal sum/count for each group
        result_count = gb.count(values)
        computed_mean = result["sum"] / result_count
        np.testing.assert_allclose(result["mean"], computed_mean, rtol=1e-10)

    @pytest.mark.parametrize("n_agg_funcs", [2, 3, 4])
    @pytest.mark.parametrize("n_input_values", [1, 2, 3])
    def test_agg_output_column_naming(self, n_agg_funcs, n_input_values):
        """Test that agg method produces correct column names for different combinations."""
        key = np.array([1, 1, 2, 1, 3, 3, 6, 1, 6])
        agg_funcs = ["sum", "mean", "min", "max"][:n_agg_funcs]

        if n_input_values == 1:
            # Single input value - column names should be agg function names
            values = pd.Series([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5])
            result = GroupBy.agg(key, values, agg_func=agg_funcs)

            expected_columns = agg_funcs
            assert list(result.columns) == expected_columns

        else:
            # Multiple input values - each gets paired with corresponding agg function
            value_dict = {
                f"col_{i}": pd.Series(
                    [
                        float(i + 1) * j
                        for j in [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5]
                    ]
                )
                for i in range(n_input_values)
            }
            values = pd.DataFrame(value_dict)

            # Should match number of agg funcs to number of input columns
            if n_agg_funcs != n_input_values:
                with pytest.raises(
                    ValueError, match="Mismatch between number of agg funcs"
                ):
                    GroupBy.agg(key, values, agg_func=agg_funcs)
            else:
                result = GroupBy.agg(key, values, agg_func=agg_funcs)

                expected_columns = list(value_dict.keys())
                assert list(result.columns) == expected_columns

                # Each column should be aggregated with its corresponding function
                for i, (col_name, agg_func) in enumerate(
                    zip(expected_columns, agg_funcs)
                ):
                    expected_col = values[col_name].groupby(key).agg(agg_func)
                    assert_pd_equal(result[col_name], expected_col, check_dtype=False)

    @pytest.mark.parametrize("categorical", [False, True])
    def test_null_keys(self, categorical):
        key = pd.Series([1, 1, 2, 1, 3, 3, 6, 1, 6])
        if categorical:
            key = key.astype("category")
        values = pd.Series(np.random.rand(len(key)))
        key.iloc[1] = np.nan
        result = GroupBy.sum(key, values)
        expected = values.groupby(key, observed=True).sum()
        assert np.isclose(result, expected).all()

        # Test with mask
        mask = key != 1
        result_masked = GroupBy.sum(key, values, mask=mask)
        expected_masked = values[mask].groupby(key[mask], observed=True).sum()
        assert np.isclose(result_masked, expected_masked).all()

    @pytest.mark.parametrize("factorize_in_chunks", [False, True])
    @pytest.mark.parametrize("use_mask", [False, True])
    def test_large_data(self, use_mask, factorize_in_chunks):
        key = pd.Series(np.random.randint(0, 1000, size=10_000_000))
        values = pd.Series(np.random.rand(10_000_000))

        gb = GroupBy(key, factorize_large_inputs_in_chunks=factorize_in_chunks)
        assert gb.ngroups == 1000  # Check number of groups
        if factorize_in_chunks:
            assert gb.key_is_chunked
            assert len(gb._group_key_pointers) == 4
        else:
            assert gb.group_ikey.shape[0] == 10_000_000  # Check group indices length
        assert gb._max_threads_for_numba > 1

        if use_mask:
            mask = key != 1
            pd_mask = mask
        else:
            mask = None
            pd_mask = slice(None)

        # Test with sum
        result_sum = gb.sum(values, mask=mask)
        expected_sum = values[pd_mask].groupby(key[pd_mask]).sum()
        assert_pd_equal(result_sum, expected_sum)

        # Test with bools
        bools = values > values.median()
        result_percentage = gb.mean(bools, mask=mask)
        expected_percentage = bools[pd_mask].groupby(key[pd_mask]).mean()
        assert_pd_equal(result_percentage, expected_percentage)

        # Test with multiple columns
        result = gb.mean([values, bools], mask=mask)
        expected_mean = values[pd_mask].groupby(key[pd_mask]).mean()
        expected = pd.DataFrame(
            {"_arr_0": expected_mean, "_arr_1": expected_percentage}
        )
        assert_pd_equal(result, expected)

    def test_categorical_order_preserved(self):
        key = pd.Categorical.from_codes(
            [0, 1, 2, 3, 1, 2, 3],
            categories=["first", "second", "third", "fourth"],
            ordered=True,
        )
        values = pd.Series(np.random.rand(len(key)))

        gb = GroupBy(key)
        result = gb.sum(values)
        expected = values.groupby(key, observed=True).sum().loc[result.index]
        np.testing.assert_array_equal(result, expected)
        assert result.index.tolist() == ["first", "second", "third", "fourth"]

    @pytest.mark.parametrize("agg_func", ["sum", "median"])
    @pytest.mark.parametrize("arg_name_to_be_wrong", ["self", "mask", "values"])
    def test_length_mismatch_fail(self, agg_func, arg_name_to_be_wrong):
        s = np.arange(10)
        kwargs = dict(self=s % 2, values=s, mask=s < 8)
        kwargs[arg_name_to_be_wrong] = kwargs[arg_name_to_be_wrong][:-1]
        with pytest.raises(ValueError):
            getattr(GroupBy, agg_func)(**kwargs)

    @pytest.mark.parametrize("agg_func", ["sum", "median"])
    @pytest.mark.parametrize("arg_name_to_be_wrong", ["self", "mask", "values"])
    def test_index_mismatch_fail(self, agg_func, arg_name_to_be_wrong):
        s = pd.Series(np.arange(10))
        kwargs = dict(self=s % 2, values=s, mask=s < 8)
        kwargs[arg_name_to_be_wrong].index += 1
        with pytest.raises(ValueError):
            getattr(GroupBy, agg_func)(**kwargs)

    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max", "count"])
    def test_lazyframe_support(self, method):
        """Test that LazyFrame inputs work with GroupBy operations."""
        # Create test data with consistent types
        key_data = [1, 1, 2, 1, 3, 3, 6, 1, 6]
        value_data = [1.0, 2.0, 4.0, 3.5, 8.0, 6.0, 3.0, 1.0, 12.6]  # All floats

        # Create LazyFrame
        lazy_df = pl.DataFrame({"values": value_data}).lazy()
        key = pd.Series(key_data)

        # Test with LazyFrame as values - results in DataFrame, so compare with DataFrame
        result = getattr(GroupBy, method)(key, lazy_df)
        if method == "count":
            expected = pd.DataFrame(
                {
                    "values": pd.Series(value_data, dtype="float64")
                    .groupby(key, observed=True)
                    .count()
                }
            )
        else:
            expected = pd.DataFrame(
                {
                    "values": pd.Series(value_data, dtype="float64")
                    .groupby(key, observed=True)
                    .agg(method)
                }
            )

        assert_pd_equal(result, expected, check_dtype=False)

        # Test with LazyFrame as group key - single column LazyFrame becomes Series
        key_lazy_df = pl.DataFrame({"key": key_data}).lazy()
        values = pd.Series(value_data, dtype="float64")

        result = getattr(GroupBy, method)(key_lazy_df, values)
        if method == "count":
            expected = values.groupby(pd.Series(key_data, name="key")).count()
        else:
            expected = values.groupby(pd.Series(key_data, name="key")).agg(method)

        assert_pd_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("nlevels", [1, 2, 3])
@pytest.mark.parametrize("aggfunc", ["sum", "min", "max"])
def test_add_row_margin(aggfunc, nlevels):
    df = pd.DataFrame(
        {
            "Bools": [True, False] * 15,
            "Strings": ["A", "B", "C"] * 10,
            "Ints": np.repeat(np.arange(10), 3),
            "X": np.random.rand(30),
        }
    )
    summary = df.groupby(["Bools", "Strings", "Ints"][:nlevels]).X.agg(aggfunc)
    with_margin = add_row_margin(summary, agg_func=aggfunc)
    assert (with_margin.reindex(summary.index) == summary).all().all()

    if nlevels == 1:
        assert with_margin.loc["All"] == summary.agg(aggfunc)
    else:
        total_key = ["All"] * nlevels
        assert np.isclose(with_margin.loc[tuple(total_key)], summary.agg(aggfunc)).all()
        for i, level in enumerate(summary.index.levels):
            key = level[0]
            ix = total_key.copy()
            ix[i] = key
            assert np.isclose(
                with_margin.loc[tuple(ix)], summary.xs(key, 0, i).agg(aggfunc)
            ).all()


@pytest.mark.parametrize(
    "aggfunc", ["mean", "count", "sum", "min", "max", "var", "std"]
)
@pytest.mark.parametrize("margins", [False, True, "row", "column"])
@pytest.mark.parametrize("use_mask", [False, True])
def test_pivot_table_basic(margins, use_mask, aggfunc):
    index = pd.Series([1, 1, 2, 1, 3, 3, 6, 1, 6] * 10)
    columns = pd.Series(["A", "B", "C", "A", "B", "C", "A", "B", "C"] * 10)
    values = pd.Series(np.random.rand(len(index)))

    if use_mask:
        mask = index != 1
    else:
        mask = slice(None)

    result = crosstab(
        index,
        columns,
        values,
        margins=margins,
        aggfunc=aggfunc,
        mask=mask if use_mask else None,
    )
    expected = pd.crosstab(
        index[mask],
        columns[mask],
        values=values[mask],
        aggfunc=aggfunc,
        margins=bool(margins),
    )

    if margins == "row":
        del expected["All"]
    elif margins == "column":
        expected = expected.drop("All")
    assert_pd_equal(result, expected, check_dtype=False, check_names=False)


@pytest.mark.parametrize("multi_values", [True, False])
@pytest.mark.parametrize("aggfunc", ["mean", "sum", "std"])
def test_pivot_table_multi_levels(aggfunc, multi_values):
    index = [np.random.randint(0, 5, 100) for _ in range(2)]
    columns = [np.random.randint(0, 5, 100) for _ in range(2)]
    values = pd.Series(np.random.rand(100))

    result = crosstab(
        index,
        columns,
        values={"a": values, "b": values * 2} if multi_values else values,
        margins=True,
        aggfunc=aggfunc,
    )
    expected = pd.crosstab(
        index,
        columns,
        values=values,
        aggfunc=aggfunc,
        margins=True,
    )

    if multi_values:
        pd.testing.assert_frame_equal(2 * result.a, result.b)
        result = result.a

    common_index = expected.index.intersection(result.index)
    common_cols = expected.columns.intersection(result.columns)
    pd.testing.assert_frame_equal(
        expected.reindex(index=common_index, columns=common_cols),
        result.reindex(index=common_index, columns=common_cols),
    )

    expected_margin = pd.crosstab(
        index[1], columns[0], values, aggfunc=aggfunc, margins=True
    )

    pd.testing.assert_frame_equal(
        result.loc["All"].xs("All", 1, 1).reindex_like(expected_margin), expected_margin
    )

    expected_margin = pd.crosstab(
        index[1], columns[1], values, aggfunc=aggfunc, margins=True
    )

    pd.testing.assert_frame_equal(
        result.loc["All"]["All"].reindex_like(expected_margin), expected_margin
    )


def test_pivot_some_keys_missing():
    k1 = pd.Categorical(["a", "b", "c"])[np.arange(10) % 2]
    k2 = pd.Categorical(["x", "y", "z"])[np.arange(10) % 2]
    result = crosstab(k1, k2)
    expected = pd.DataFrame([
       [5, np.nan], 
       [np.nan, 5],
    ], k1[[0, 1]], k2[[0, 1]])
    assert_pd_equal(result, expected)


class TestGroupByRowSelection:
    """Test class for GroupBy head, tail, and nth methods."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing row selection methods."""
        # Group key: [A, A, A, B, B, C, C, C, C]
        # Values:    [1, 2, 3, 4, 5, 6, 7, 8, 9]
        return {
            "key": pd.Series(["A", "A", "A", "B", "B", "C", "C", "C", "C"]),
            "values": pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9]),
            "df_values": pd.DataFrame(
                {
                    "col1": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                    "col2": [10, 20, 30, 40, 50, 60, 70, 80, 90],
                }
            ),
        }

    # Tests for head method with keep_input_index=True (simpler case that works)
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_head_with_keep_input_index(self, sample_data, n):
        """Test head method with keep_input_index=True."""
        key = sample_data["key"]
        values = sample_data["values"]

        gb = GroupBy(key)
        result = gb.head(values, n=n, keep_input_index=True)

        # Compare with pandas groupby (which keeps original index by default)
        expected = values.groupby(key, observed=True).head(n)
        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_head_dataframe_with_keep_input_index(self, sample_data, n):
        """Test head method with DataFrame and keep_input_index=True."""
        key = sample_data["key"]
        values = sample_data["df_values"]

        gb = GroupBy(key)
        result = gb.head(values, n=n, keep_input_index=True)

        expected = values.groupby(key, observed=True).head(n)
        assert_pd_equal(result, expected, check_dtype=False)

    # Tests for tail method with keep_input_index=True
    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_tail_with_keep_input_index(self, sample_data, n):
        """Test tail method with keep_input_index=True."""
        key = sample_data["key"]
        values = sample_data["values"]

        gb = GroupBy(key)
        result = gb.tail(values, n=n, keep_input_index=True)

        expected = values.groupby(key, observed=True).tail(n)
        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("n", [1, 2, 3])
    def test_tail_dataframe_with_keep_input_index(self, sample_data, n):
        """Test tail method with DataFrame and keep_input_index=True."""
        key = sample_data["key"]
        values = sample_data["df_values"]

        gb = GroupBy(key)
        result = gb.tail(values, n=n, keep_input_index=True)

        expected = values.groupby(key, observed=True).tail(n)
        assert_pd_equal(result, expected, check_dtype=False)

    # Tests for nth method with keep_input_index=True
    @pytest.mark.parametrize("n", [0, 1, 2, -1, -2])
    def test_nth_with_keep_input_index(self, sample_data, n):
        """Test nth method with keep_input_index=True."""
        key = sample_data["key"]
        values = sample_data["values"]

        gb = GroupBy(key)
        result = gb.nth(values, n=n, keep_input_index=True)

        expected = values.groupby(key, observed=True).nth(n)
        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("n", [0, 1, -1])
    def test_nth_dataframe_with_keep_input_index(self, sample_data, n):
        """Test nth method with DataFrame and keep_input_index=True."""
        key = sample_data["key"]
        values = sample_data["df_values"]

        gb = GroupBy(key)
        result = gb.nth(values, n=n, keep_input_index=True)

        expected = values.groupby(key, observed=True).nth(n)
        assert_pd_equal(result, expected, check_dtype=False)

    # Edge case tests
    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_head_edge_cases(self, sample_data, n):
        """Test edge cases for head method."""
        key = sample_data["key"]
        values = sample_data["values"]

        gb = GroupBy(key)
        result = gb.head(values, n=n, keep_input_index=True)

        expected = values.groupby(key, observed=True).head(n)
        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("n", [0, 1, 2])
    def test_tail_edge_cases(self, sample_data, n):
        """Test edge cases for tail method."""
        key = sample_data["key"]
        values = sample_data["values"]

        gb = GroupBy(key)
        result = gb.tail(values, n=n, keep_input_index=True)

        expected = values.groupby(key, observed=True).tail(n)
        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("n", [10, -10, 100])
    def test_nth_out_of_bounds(self, sample_data, n):
        """Test nth method with out-of-bounds indices."""
        key = sample_data["key"]
        values = sample_data["values"]

        gb = GroupBy(key)
        result = gb.nth(values, n=n, keep_input_index=True)

        # Should return empty or NaN values for out of bounds
        expected = values.groupby(key, observed=True).nth(n)
        assert_pd_equal(result, expected, check_dtype=False)

    # Test with different input types
    def test_head_input_types(self, sample_data):
        """Test head method with numpy array input."""
        key = sample_data["key"]
        values_orig = sample_data["values"]

        # Test with numpy array (this works)
        values = values_orig.values

        gb = GroupBy(key)
        result = gb.head(values, n=2, keep_input_index=True)

        # Expected should always match pandas behavior
        expected = values_orig.groupby(key, observed=True).head(2)
        assert_pd_equal(result, expected, check_dtype=False)

    def test_different_key_types(self, sample_data):
        """Test with numpy array key types."""
        key_orig = sample_data["key"]
        values = sample_data["values"]

        # Test with numpy array (this works)
        key = key_orig.values

        gb = GroupBy(key)
        result = gb.head(values, n=2, keep_input_index=True)

        expected = values.groupby(key_orig).head(2)
        assert_pd_equal(result, expected, check_dtype=False)

    def test_empty_groups(self):
        """Test behavior with empty groups or no data."""
        # Empty data
        key = pd.Series([], dtype=str)
        values = pd.Series([], dtype=int)

        gb = GroupBy(key)

        # Test head
        result = gb.head(values, n=2, keep_input_index=True)
        expected = pd.Series([], dtype=values.dtype)
        assert_pd_equal(result, expected, check_dtype=False)

        # Test tail
        result = gb.tail(values, n=2, keep_input_index=True)
        expected = pd.Series([], dtype=values.dtype)
        assert_pd_equal(result, expected, check_dtype=False)

        # Test nth
        result = gb.nth(values, n=0, keep_input_index=True)
        expected = pd.Series([], dtype=values.dtype)
        assert_pd_equal(result, expected, check_dtype=False)

    def test_single_group(self):
        """Test with data that has only one group."""
        key = pd.Series(["A"] * 5)
        values = pd.Series([1, 2, 3, 4, 5])

        gb = GroupBy(key)

        # Test head
        result = gb.head(values, n=3, keep_input_index=True)
        expected = values.groupby(key, observed=True).head(3)
        assert_pd_equal(result, expected, check_dtype=False)

        # Test tail
        result = gb.tail(values, n=3, keep_input_index=True)
        expected = values.groupby(key, observed=True).tail(3)
        assert_pd_equal(result, expected, check_dtype=False)

        # Test nth
        result = gb.nth(values, n=1, keep_input_index=True)
        expected = values.groupby(key, observed=True).nth(1)
        assert_pd_equal(result, expected, check_dtype=False)

    def test_large_n_values(self):
        """Test with n larger than group sizes."""
        key = pd.Series(["A", "A", "B", "C"])
        values = pd.Series([1, 2, 3, 4])

        gb = GroupBy(key)

        # Test head with large n
        result = gb.head(values, n=10, keep_input_index=True)
        expected = values.groupby(key, observed=True).head(10)
        assert_pd_equal(result, expected, check_dtype=False)

        # Test tail with large n
        result = gb.tail(values, n=10, keep_input_index=True)
        expected = values.groupby(key, observed=True).tail(10)
        assert_pd_equal(result, expected, check_dtype=False)


@pytest.fixture(scope="module")
def parquet_files(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("test")
    N = 10_000
    a = np.arange(N)
    df = pd.DataFrame(
        dict(
            ints=a,
            cat=pd.Categorical.from_codes(a % 6, list("qwerty")),
            floats=np.random.rand(N),
            bools=a % 3 == 0,
            # times=pd.Timestamp("20250101") + a.astype("m8[ns]"),
            timedeltas=a.astype("m8[ns]"),
        )
    )
    # df["floats"] = df["floats"].where(df.floats > df.floats.median())  # add NaNs
    files = [tmpdir.join("df1.parquet"), tmpdir.join("df2.parquet")]
    for file in files:
        df.to_parquet(file)

    return files


@pytest.fixture(scope="module")
def df_chunked(parquet_files):
    df_chunked = pd.read_parquet(parquet_files, dtype_backend="pyarrow")
    assert isinstance(pa.Array.from_pandas(df_chunked.ints), pa.ChunkedArray)
    return df_chunked


@pytest.fixture(scope="module")
def df_np_backed(parquet_files):
    return pd.read_parquet(parquet_files)


@pytest.mark.parametrize(
    "method",
    [
        "sum",
        "mean",
        "min",
        "max",
        "var",
        "first",
        "last",
        "cumsum",
        "cummin",
        "cummax",
        "shift",
        "diff",
    ],
)
def test_group_by_methods_vs_pandas_with_chunked_arrays(df_chunked, method):
    cols = ["ints", "floats", "timedeltas"]
    gb = df_chunked.groupby("cat", sort=False, observed=True)
    for col in cols:
        try:
            expected = getattr(gb[col], method)()
        except TypeError:
            continue
        result = getattr(GroupBy, method)(
            df_chunked.cat,
            df_chunked[col],
        )
        if len(expected) < len(df_chunked):
            expected.index = expected.index.astype(str)

        assert_pd_equal(result, expected, check_dtype=False), col


@pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
def test_group_by_rolling_methods_vs_pandas_with_chunked_arrays(df_chunked, method):
    cols = ["ints", "floats"]
    window = 5
    gb = df_chunked.groupby("cat", sort=False, observed=True).rolling(window)
    expected = getattr(gb[cols], method)()
    result = getattr(GroupBy, f"rolling_{method}")(
        df_chunked.cat, df_chunked[cols], window=window
    )
    expected = expected.reset_index(level=0, drop=True).sort_index()

    assert_pd_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
def test_group_by_rolling_methods_vs_pandas_with_np_arrays(df_np_backed, method):
    cols = ["ints", "floats"]
    window = 5
    gb = df_np_backed.groupby("cat", sort=False, observed=True).rolling(window)
    expected = getattr(gb[cols], method)()
    result = getattr(GroupBy, f"rolling_{method}")(
        df_np_backed.cat, df_np_backed[cols], window=window
    )
    expected = expected.reset_index(level=0, drop=True).sort_index()
    assert_pd_equal(result, expected, check_dtype=False)


@pytest.mark.parametrize("method", ["sum", "mean", "min", "max"])
def test_group_by_rolling_methods_vs_pandas_with_timedeltas(df_np_backed, method):
    window = 5
    result = getattr(GroupBy, f"rolling_{method}")(
        df_np_backed.cat, df_np_backed.timedeltas, window=window
    )
    df_np_backed["time_int"] = df_np_backed.timedeltas.astype(int)
    gb = df_np_backed.groupby("cat", sort=False, observed=True).rolling(window)
    expected = getattr(gb["time_int"], method)()
    expected = expected.reset_index(level=0, drop=True).sort_index().astype("m8[ns]")

    assert_pd_equal(result, expected, check_dtype=False, check_names=False)


@pytest.mark.parametrize("use_mask", [False, True])
@pytest.mark.parametrize("partial", [False, True])
def test_monotonic_group_key(partial, use_mask):
    labels = np.arange(2, 1002)
    mono_key = np.repeat(labels, 2000)
    if partial:
        mono_key = np.concatenate([mono_key, mono_key])

    gb = GroupBy(mono_key)
    assert gb.key_is_chunked is partial
    assert gb.result_index.equals(pd.Index(labels))

    arr = np.random.rand(len(mono_key))
    if use_mask:
        mask = arr > 0.5
    else:
        mask = slice(None)

    result = gb.mean(arr, mask=mask)
    expected = pd.Series(arr)[mask].groupby(mono_key[mask]).mean()
    assert_pd_equal(result, expected)
