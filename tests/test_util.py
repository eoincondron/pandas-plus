import numba as nb
import numpy as np
import pandas as pd
import polars as pl
import polars.testing
import pytest
import pyarrow as pa

from pandas_plus.util import (
    MAX_INT,
    MIN_INT,
    _get_first_non_null,
    _null_value_for_numpy_type,
    convert_data_to_arr_list_and_keys,
    get_array_name,
    is_null,
    pretty_cut,
    bools_to_categorical,
    nb_dot,
    monotonic_factorization,
    factorize_1d,
    factorize_2d,
    array_split_with_chunk_handling,
    to_arrow,
)


class TestArrayFunctions:
    def test_get_array_name_with_numpy(self):
        # NumPy arrays don't have names
        arr = np.array([1, 2, 3])
        assert get_array_name(arr) is None

    def test_get_array_name_with_pandas(self):
        # Pandas Series with name
        named_series = pd.Series([1, 2, 3], name="test_series")
        assert get_array_name(named_series) == "test_series"

        named_series = pd.Series([1, 2, 3], name=0)
        assert get_array_name(named_series) == 0

        # Pandas Series without name
        unnamed_series = pd.Series([1, 2, 3])
        assert get_array_name(unnamed_series) is None

        # Pandas Series with empty name
        empty_name_series = pd.Series([1, 2, 3], name="")
        assert get_array_name(empty_name_series) is None

    def test_get_array_name_with_polars(self):
        # Polars Series with name
        named_series = pl.Series("test_series", [1, 2, 3])
        assert get_array_name(named_series) == "test_series"

        # Polars Series with empty name
        empty_name_series = pl.Series("", [1, 2, 3])
        assert get_array_name(empty_name_series) is None

    def test_convert_mapping_to_dict(self):
        # Test with dictionary
        input_dict = {"a": np.array([1, 2]), "b": np.array([3, 4])}
        result_arrs, result_names = convert_data_to_arr_list_and_keys(input_dict)
        assert dict(zip(result_names, result_arrs)) == input_dict

        # Test with other mapping types
        from collections import OrderedDict

        ordered_dict = OrderedDict([("x", np.array([1, 2])), ("y", np.array([3, 4]))])
        result_arrs, result_names = convert_data_to_arr_list_and_keys(ordered_dict)
        assert dict(zip(result_names, result_arrs)) == ordered_dict

    def test_convert_list_to_dict(self):
        # Test with list of named arrays
        arrays = [
            pd.Series([1, 2, 3], name="first"),
            pd.Series([4, 5, 6], name="second"),
        ]
        result_arrs, result_names = convert_data_to_arr_list_and_keys(arrays)
        assert result_names == ["first", "second"]
        for left, right in zip(result_arrs, arrays):
            pd.testing.assert_series_equal(left, right)

        # Test with list of unnamed arrays
        arrays = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        ]

        result_arrs, result_names = convert_data_to_arr_list_and_keys(arrays)
        assert result_names == [None, None]

        for left, right in zip(result_arrs, arrays):
            np.testing.assert_array_equal(left, right)

        # Test with mixed named and unnamed arrays
        arrays = [
            pd.Series([1, 2, 3], name="named"),
            np.array([4, 5, 6]),
            pd.Series([7, 8, 9]),
        ]
        result_arrs, result_names = convert_data_to_arr_list_and_keys(arrays)
        for left, right in zip(result_arrs, arrays):
            if isinstance(right, pd.Series):
                pd.testing.assert_series_equal(left, right)
            else:
                np.testing.assert_array_equal(left, right)

    def test_convert_numpy_array_to_dict(self):
        # Test with 1D numpy array
        arr = np.array([1, 2, 3])
        result_arrs, result_names = convert_data_to_arr_list_and_keys(arr)

        assert [None] == result_names
        np.testing.assert_array_equal(result_arrs[0], arr)

        # Test with 2D numpy array (should return empty dict as per function logic)
        arr_2d = np.array([[1, 2], [3, 4]])
        result_arrs, result_names = convert_data_to_arr_list_and_keys(arr_2d)
        assert result_names == [None, None]

        for i in range(2):
            np.testing.assert_array_equal(result_arrs[i], arr_2d[:, i])

    def test_convert_series_to_dict(self):
        # Test with named pandas Series
        series = pd.Series([1, 2, 3], name="test_series")
        result_arrs, result_names = convert_data_to_arr_list_and_keys(series)

        assert result_names == ["test_series"]
        pd.testing.assert_series_equal(result_arrs[0], series)

        # Test with unnamed pandas Series
        series = pd.Series([1, 2, 3])
        result_arrs, result_names = convert_data_to_arr_list_and_keys(series)
        assert [None] == result_names
        pd.testing.assert_series_equal(result_arrs[0], series)

        # Test with polars Series
        series = pl.Series("polars_series", [1, 2, 3])
        result_arrs, result_names = convert_data_to_arr_list_and_keys(series)
        assert ["polars_series"] == result_names
        pl.testing.assert_series_equal(result_arrs[0], series)

    def test_convert_dataframe_to_dict(self):
        # Test with pandas DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result_arrs, result_names = convert_data_to_arr_list_and_keys(df)

        assert len(result_arrs) == 2
        assert result_names == ["a", "b"]
        for i, col in enumerate(df.columns):
            pd.testing.assert_series_equal(result_arrs[i], df[col])

        # Test with polars DataFrame
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        result_arrs, result_names = convert_data_to_arr_list_and_keys(df)
        assert result_names == ["x", "y"]
        for i, col in enumerate(df.columns):
            pl.testing.assert_series_equal(result_arrs[i], df[col])

        # Test with polars LazyFrame
        lazy_df = pl.DataFrame({"x": [1, 2], "y": [3, 4]}).lazy()
        result_arrs, result_names = convert_data_to_arr_list_and_keys(lazy_df)
        assert result_names == ["x", "y"]
        expected_df = lazy_df.collect()
        for i, col in enumerate(expected_df.columns):
            pl.testing.assert_series_equal(result_arrs[i], expected_df[col])

    def test_unsupported_type(self):
        # Test with unsupported type
        with pytest.raises(TypeError, match="Input type <class 'int'> not supported"):
            convert_data_to_arr_list_and_keys(123)


class TestIsNullFunction:
    def test_is_null_python_floats(self):
        """Test is_null with Python float values"""
        assert is_null(float("nan"))
        assert not is_null(0.0)
        assert not is_null(-1.5)
        assert not is_null(1e10)

    def test_is_null_numpy_floats(self):
        """Test is_null with NumPy float values"""
        assert is_null(np.nan)
        assert is_null(np.float64("nan"))
        assert not is_null(np.float64(0.0))
        assert not is_null(np.float32(-1.5))

    def test_is_null_integers(self):
        """Test is_null with integer values"""
        assert not is_null(0)
        assert not is_null(-1)
        assert not is_null(100)
        # MIN_INT should be considered null for integers
        assert is_null(MIN_INT)

    def test_is_null_booleans(self):
        """Test is_null with boolean values"""
        assert not is_null(True)
        assert not is_null(False)


# Numba-compiled test functions to test the JIT implementation
@nb.njit
def _test_jit_float_null():
    return is_null(np.nan), is_null(0.0)


@nb.njit
def _test_jit_int_null():
    return is_null(MIN_INT), is_null(0)


@nb.njit
def _test_jit_bool_null():
    return is_null(True), is_null(False)


class TestJitIsNull:
    def test_jit_is_null_floats(self):
        """Test JIT-compiled is_null with float values"""
        is_nan_null, is_zero_null = _test_jit_float_null()
        assert is_nan_null
        assert not is_zero_null

    def test_jit_is_null_integers(self):
        """Test JIT-compiled is_null with integer values"""
        is_min_int_null, is_zero_null = _test_jit_int_null()
        assert is_min_int_null
        assert not is_zero_null

    def test_jit_is_null_booleans(self):
        """Test JIT-compiled is_null with boolean values"""
        is_true_null, is_false_null = _test_jit_bool_null()
        assert not is_true_null
        assert not is_false_null


# Test functions for _get_first_non_null
@nb.njit
def _test_jit_get_first_non_null_with_nans():
    arr = np.array([np.nan, np.nan, 3.0, 4.0, np.nan])
    return _get_first_non_null(arr)


@nb.njit
def _test_jit_get_first_non_null_all_nans():
    arr = np.array([np.nan, np.nan, np.nan])
    return _get_first_non_null(arr)


@nb.njit
def _test_jit_get_first_non_null_no_nans():
    arr = np.array([1.0, 2.0, 3.0])
    return _get_first_non_null(arr)


@nb.njit
def _test_jit_get_first_non_null_with_integers():
    arr = np.array([MIN_INT, 1, 2, MIN_INT])
    return _get_first_non_null(arr)


class TestGetFirstNonNull:
    def test_get_first_non_null_with_nans(self):
        """Test _get_first_non_null with array containing NaN values"""
        idx, val = _test_jit_get_first_non_null_with_nans()
        assert idx == 2
        assert val == 3.0

    def test_get_first_non_null_all_nans(self):
        """Test _get_first_non_null with array of all NaN values"""
        idx, val = _test_jit_get_first_non_null_all_nans()
        assert idx == -1
        assert np.isnan(val)

    def test_get_first_non_null_no_nans(self):
        """Test _get_first_non_null with array containing no NaN values"""
        idx, val = _test_jit_get_first_non_null_no_nans()
        assert idx == 0
        assert val == 1.0

    def test_get_first_non_null_with_integers(self):
        """Test _get_first_non_null with integer array containing MIN_INT values"""
        idx, val = _test_jit_get_first_non_null_with_integers()
        assert idx == 1
        assert val == 1


class TestNullValueForArrayType:
    def test_null_value_for_int64(self):
        """Test null value for int64 array"""
        arr = np.array([1, 2, 3], dtype=np.int64)
        null_value = _null_value_for_numpy_type(arr.dtype)
        assert null_value == MIN_INT
        assert null_value == np.iinfo(np.int64).min

    def test_null_value_for_int32(self):
        """Test null value for int32 array"""
        arr = np.array([1, 2, 3], dtype=np.int32)
        null_value = _null_value_for_numpy_type(arr.dtype)
        assert null_value == np.iinfo(np.int32).min

    def test_null_value_for_float64(self):
        """Test null value for float64 array"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        null_value = _null_value_for_numpy_type(arr.dtype)
        assert np.isnan(null_value)

    def test_null_value_for_float32(self):
        """Test null value for float32 array"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        null_value = _null_value_for_numpy_type(arr.dtype)
        assert np.isnan(null_value)

    def test_null_value_for_uint64(self):
        """Test null value for uint64 array"""
        arr = np.array([1, 2, 3], dtype=np.uint64)
        null_value = _null_value_for_numpy_type(arr.dtype)
        assert null_value == np.iinfo(np.uint64).max

    def test_null_value_for_uint32(self):
        """Test null value for uint32 array"""
        arr = np.array([1, 2, 3], dtype=np.uint32)
        null_value = _null_value_for_numpy_type(arr.dtype)
        assert null_value == np.iinfo(np.uint32).max

    def test_null_value_for_complex(self):
        """Test null value for complex array - should raise TypeError"""
        arr = np.array([1 + 2j, 3 + 4j], dtype=complex)
        with pytest.raises(TypeError):
            _null_value_for_numpy_type(arr.dtype)


class TestPrettyCut:
    """Test cases for the pretty_cut function."""

    def test_basic_integer_cut(self):
        """Test basic integer cutting with integer bins."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 3", "4 - 6", "7 - 9", " > 9"]
        assert result.categories.tolist() == expected_labels

        # Check specific values
        assert result[0] == " <= 3"  # 1
        assert result[3] == "4 - 6"  # 4
        assert result[6] == "7 - 9"  # 7
        assert result[9] == " > 9"  # 10

    def test_basic_float_cut(self):
        """Test basic float cutting with float bins."""
        x = np.array([1.0, 2.5, 3.7, 4.2, 5.8, 6.1, 7.9, 8.3, 9.4, 10.7])
        bins = [3.0, 6.0, 9.0]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 3.0", "3.0 - 6.0", "6.0 - 9.0", " > 9.0"]
        assert result.categories.tolist() == expected_labels

    def test_mixed_int_float_bins(self):
        """Test with integer data and float bins."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [3.5, 6.5, 9.5]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 3.5", "3.5 - 6.5", "6.5 - 9.5", " > 9.5"]
        assert result.categories.tolist() == expected_labels

    def test_pandas_series_input(self):
        """Test with pandas Series input."""
        x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="test_series")
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)

        # Should return a pandas Series
        assert isinstance(result, pd.Series)
        assert result.name == "test_series"
        assert result.index.equals(x.index)

        expected_labels = [" <= 3", "4 - 6", "7 - 9", " > 9"]
        assert result.cat.categories.tolist() == expected_labels

    def test_polars_series_input(self):
        """Test with polars Series input."""
        x = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)

        # Should return a Categorical
        assert isinstance(result, pd.Categorical)
        expected_labels = [" <= 3", "4 - 6", "7 - 9", " > 9"]
        assert result.categories.tolist() == expected_labels

    def test_list_bins(self):
        """Test with bins provided as a list."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [3, 6, 9]  # List instead of array
        result = pretty_cut(x, bins)

        expected_labels = [" <= 3", "4 - 6", "7 - 9", " > 9"]
        assert result.categories.tolist() == expected_labels

    def test_single_bin(self):
        """Test with a single bin value."""
        x = np.array([1, 2, 3, 4, 5])
        bins = [3]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 3", " > 3"]
        assert result.categories.tolist() == expected_labels

        # Check values
        assert result[0] == " <= 3"  # 1
        assert result[4] == " > 3"  # 5

    def test_empty_array(self):
        """Test with empty input array."""
        x = np.array([])
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)

        assert len(result) == 0
        expected_labels = [" <= 3", "3 - 6", "6 - 9", " > 9"]
        assert result.categories.tolist() == expected_labels

    def test_all_values_below_first_bin(self):
        """Test when all values are below the first bin."""
        x = np.array([1, 2])
        bins = [5, 10]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 5", "6 - 10", " > 10"]
        assert result.categories.tolist() == expected_labels
        assert all(result == " <= 5")

    def test_all_values_above_last_bin(self):
        """Test when all values are above the last bin."""
        x = np.array([15, 20])
        bins = [5, 10]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 5", "6 - 10", " > 10"]
        assert result.categories.tolist() == expected_labels
        assert all(result == " > 10")

    def test_float_with_nan_values(self):
        """Test float array with NaN values."""
        x = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        bins = [2.0, 5.0]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 2.0", "2.0 - 5.0", " > 5.0"]
        assert result.categories.tolist() == expected_labels

        # NaN values should result in NaN categories
        assert pd.isna(result[1])
        assert pd.isna(result[4])

        # Non-NaN values should be categorized correctly
        assert result[0] == " <= 2.0"  # 1.0
        assert result[2] == "2.0 - 5.0"  # 3.0
        assert result[5] == " > 5.0"  # 6.0

    def test_integer_boundary_values(self):
        """Test integer boundary handling."""
        x = np.array([3, 4, 6, 7, 9, 10])
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 3", "4 - 6", "7 - 9", " > 9"]
        assert result.categories.tolist() == expected_labels

        # Test boundary values
        assert result[0] == " <= 3"  # 3 (at boundary)
        assert result[1] == "4 - 6"  # 4 (start of next bin)
        assert result[2] == "4 - 6"  # 6 (at boundary)
        assert result[3] == "7 - 9"  # 7 (start of next bin)
        assert result[4] == "7 - 9"  # 9 (at boundary)
        assert result[5] == " > 9"  # 10 (above last bin)

    def test_float_boundary_values(self):
        """Test float boundary handling."""
        x = np.array([3.0, 3.1, 6.0, 6.1, 9.0, 9.1])
        bins = [3.0, 6.0, 9.0]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 3.0", "3.0 - 6.0", "6.0 - 9.0", " > 9.0"]
        assert result.categories.tolist() == expected_labels

        # Test boundary values (floats don't get +1 adjustment)
        assert result[0] == " <= 3.0"  # 3.0 (at boundary)
        assert result[1] == "3.0 - 6.0"  # 3.1 (just above boundary)
        assert result[2] == "3.0 - 6.0"  # 6.0 (at boundary)
        assert result[3] == "6.0 - 9.0"  # 6.1 (just above boundary)
        assert result[4] == "6.0 - 9.0"  # 9.0 (at boundary)
        assert result[5] == " > 9.0"  # 9.1 (above last bin)

    def test_unsorted_bins(self):
        """Test behavior with unsorted bins."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [6, 3, 9]  # Unsorted
        result = pretty_cut(x, bins)

        # The function should still work because bins get converted to array
        # Labels will reflect the order given
        expected_labels = [" <= 3", "4 - 6", "7 - 9", " > 9"]
        assert result.categories.tolist() == expected_labels

    @pytest.mark.parametrize("array_type", [np.array, pd.Series, pl.Series])
    def test_different_array_types(self, array_type):
        """Test with different array types."""
        if array_type == pd.Series:
            x = array_type([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name="test")
        else:
            x = array_type([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        bins = [3, 6, 9]
        result = pretty_cut(x, bins)

        expected_labels = [" <= 3", "4 - 6", "7 - 9", " > 9"]

        if isinstance(x, pd.Series):
            assert isinstance(result, pd.Series)
            assert result.cat.categories.tolist() == expected_labels
        else:
            assert isinstance(result, pd.Categorical)
            assert result.categories.tolist() == expected_labels

    def test_preserve_series_attributes(self):
        """Test that pandas Series attributes are preserved."""
        index = pd.Index(["a", "b", "c", "d", "e"], name="test_index")
        x = pd.Series([1, 2, 3, 4, 5], index=index, name="test_series")
        bins = [2, 4]
        result = pretty_cut(x, bins)

        assert isinstance(result, pd.Series)
        assert result.name == "test_series"
        assert result.index.equals(x.index)
        assert result.index.name == "test_index"


class TestBoolsToCategorical:
    """Test cases for the bools_to_categorical function."""

    def test_basic_functionality(self):
        """Test basic conversion of boolean DataFrame to categorical."""
        df = pd.DataFrame(
            {
                "A": [True, False, True, False],
                "B": [False, True, False, True],
                "C": [False, False, True, True],
            }
        )
        result = bools_to_categorical(df)

        assert isinstance(result, pd.Series)
        assert result.index.equals(df.index)

        # Check categories
        expected_categories = ["A", "B", "A & C", "B & C"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)

    def test_all_false_row(self):
        """Test handling of rows with all False values."""
        df = pd.DataFrame(
            {
                "A": [True, False, False],
                "B": [False, True, False],
                "C": [False, False, False],
            }
        )
        result = bools_to_categorical(df)

        expected_categories = ["A", "B", "None"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)
        assert result[2] == "None"

    def test_custom_separator(self):
        """Test custom separator for joining labels."""
        df = pd.DataFrame({"X": [True, False], "Y": [True, False], "Z": [False, True]})
        result = bools_to_categorical(df, sep=" | ")

        expected_categories = ["X | Y", "Z"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)
        assert result[0] == "X | Y"

    def test_custom_na_representation(self):
        """Test custom NA representation for all-False rows."""
        df = pd.DataFrame({"A": [True, False], "B": [False, False]})
        result = bools_to_categorical(df, na_rep="Missing")

        expected_categories = ["A", "Missing"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)
        assert result[1] == "Missing"

    def test_allow_duplicates_true(self):
        """Test with allow_duplicates=True (default behavior)."""
        df = pd.DataFrame({"A": [True, False], "B": [True, False], "C": [False, True]})
        result = bools_to_categorical(df, allow_duplicates=True)

        expected_categories = ["A & B", "C"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)
        assert result[0] == "A & B"

    def test_allow_duplicates_false_success(self):
        """Test with allow_duplicates=False when no duplicates exist."""
        df = pd.DataFrame(
            {
                "A": [True, False, False],
                "B": [False, True, False],
                "C": [False, False, True],
            }
        )
        result = bools_to_categorical(df, allow_duplicates=False)

        expected_categories = ["A", "B", "C"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)

    def test_allow_duplicates_false_failure(self):
        """Test with allow_duplicates=False when duplicates exist - should raise ValueError."""
        df = pd.DataFrame(
            {
                "A": [True, False],
                "B": [True, False],  # This creates a duplicate True in the first row
            }
        )

        with pytest.raises(ValueError) as excinfo:
            bools_to_categorical(df, allow_duplicates=False)

        assert "allow_duplicates is False" in str(excinfo.value)

    def test_single_column(self):
        """Test with single column DataFrame."""
        df = pd.DataFrame({"OnlyCol": [True, False, True]})
        result = bools_to_categorical(df)

        expected_categories = ["OnlyCol", "None"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)
        assert result[0] == "OnlyCol"
        assert result[1] == "None"

    def test_all_true_columns(self):
        """Test with all columns True in some rows."""
        df = pd.DataFrame(
            {
                "A": [True, True, False],
                "B": [True, False, True],
                "C": [True, False, False],
            }
        )
        result = bools_to_categorical(df)

        expected_categories = ["A & B & C", "A", "B"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)
        assert result[0] == "A & B & C"

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["A", "B"]).astype(bool)
        result = bools_to_categorical(df)

        assert len(result) == 0
        assert isinstance(result, pd.Series)

    def test_preserve_index(self):
        """Test that original DataFrame index is preserved."""
        custom_index = pd.Index(["row1", "row2", "row3"], name="custom_idx")
        df = pd.DataFrame(
            {"A": [True, False, True], "B": [False, True, False]}, index=custom_index
        )

        result = bools_to_categorical(df)

        assert result.index.equals(df.index)
        assert result.index.name == "custom_idx"

    def test_identical_rows(self):
        """Test with identical boolean patterns."""
        df = pd.DataFrame(
            {
                "A": [True, True, False, False],
                "B": [False, False, True, True],
                "C": [True, True, False, False],
            }
        )
        result = bools_to_categorical(df)

        # Should have only 2 unique categories since rows 0&1 and 2&3 are identical
        expected_categories = ["A & C", "B"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)

        # Check that identical rows get same category
        assert result[0] == result[1]  # Both should be 'A & C'
        assert result[2] == result[3]  # Both should be 'B'

    def test_complex_pattern(self):
        """Test with a more complex boolean pattern."""
        df = pd.DataFrame(
            {
                "Feature1": [True, False, True, False, False],
                "Feature2": [True, True, False, False, True],
                "Feature3": [False, True, True, True, False],
                "Feature4": [False, False, False, True, True],
            }
        )
        result = bools_to_categorical(df)

        assert len(result) == 5
        assert isinstance(result, pd.Series)

        # Check that all values are valid categories
        for val in result:
            assert val in result.cat.categories

    @pytest.mark.parametrize("sep", [" & ", " | ", " + ", "-"])
    def test_different_separators(self, sep):
        """Test with different separator characters."""
        df = pd.DataFrame({"A": [True, False], "B": [True, False]})
        result = bools_to_categorical(df, sep=sep)

        expected_category = f"A{sep}B"
        assert expected_category in result.cat.categories
        assert result[0] == expected_category

    @pytest.mark.parametrize("na_rep", ["None", "Missing", "Empty", "N/A"])
    def test_different_na_representations(self, na_rep):
        """Test with different NA representation strings."""
        df = pd.DataFrame({"A": [True, False], "B": [False, False]})
        result = bools_to_categorical(df, na_rep=na_rep)

        assert na_rep in result.cat.categories
        assert result[1] == na_rep

    def test_column_names_with_special_characters(self):
        """Test with column names containing special characters."""
        df = pd.DataFrame(
            {
                "Feature-1": [True, False],
                "Feature_2": [False, True],
                "Feature 3": [True, True],
            }
        )
        result = bools_to_categorical(df)

        expected_categories = ["Feature-1 & Feature 3", "Feature_2 & Feature 3"]
        assert sorted(result.cat.categories.tolist()) == sorted(expected_categories)


class TestNbDot:
    """Test cases for the nb_dot function."""

    def test_numpy_array_basic(self):
        """Test nb_dot with basic numpy arrays."""
        a = np.array([[1, 2, 3], [4, 5, 6]]).T
        b = np.array([1, 2])
        result = nb_dot(a, b)

        expected = np.array([9, 12, 15])  # [1*1 + 4*2, 2*1 + 5*2, 3*1 + 6*2]
        np.testing.assert_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_numpy_array_float(self):
        """Test nb_dot with float numpy arrays."""
        a = np.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])
        b = np.array([2.0, 3.0])
        result = nb_dot(a, b)

        expected = np.dot(a, b)
        np.testing.assert_array_almost_equal(result, expected)

    def test_pandas_dataframe_basic(self):
        """Test nb_dot with pandas DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        b = np.array([1, 2])
        result = nb_dot(df, b)

        expected = pd.Series(
            [9, 12, 15], index=df.index
        )  # [1*1 + 4*2, 2*1 + 5*2, 3*1 + 6*2]
        pd.testing.assert_series_equal(result, expected)
        assert isinstance(result, pd.Series)

    def test_pandas_dataframe_with_index(self):
        """Test nb_dot with pandas DataFrame with custom index."""
        custom_index = pd.Index(["a", "b", "c"], name="test_idx")
        df = pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}, index=custom_index)
        b = np.array([2, 3])
        result = nb_dot(df, b)

        expected = pd.Series(
            [14, 19, 24], index=custom_index
        )  # [1*2 + 4*3, 2*2 + 5*3, 3*2 + 6*3]
        pd.testing.assert_series_equal(result, expected)
        assert result.index.equals(df.index)

    def test_polars_dataframe_basic(self):
        """Test nb_dot with polars DataFrame."""
        df = pl.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        b = np.array([1, 2])
        result = nb_dot(df, b)

        expected = np.array([9, 12, 15])  # [1*1 + 4*2, 2*1 + 5*2, 3*1 + 6*2]
        np.testing.assert_array_equal(result, expected)
        assert isinstance(result, np.ndarray)

    def test_single_column(self):
        """Test nb_dot with single column input."""
        a = np.array([[1], [2], [3]])
        b = np.array([5])
        result = nb_dot(a, b)

        expected = np.array([5, 10, 15])  # [1*5, 2*5, 3*5]
        np.testing.assert_array_equal(result, expected)

    def test_single_row(self):
        """Test nb_dot with single row input."""
        a = np.array([[1, 2, 3, 4]])
        b = np.array([1, 2, 3, 4])
        result = nb_dot(a, b)

        expected = np.array([30])  # [1*1 + 2*2 + 3*3 + 4*4]
        np.testing.assert_array_equal(result, expected)

    def test_zeros(self):
        """Test nb_dot with zero arrays."""
        a = np.zeros((3, 2))
        b = np.array([1, 2])
        result = nb_dot(a, b)

        expected = np.zeros(3)
        np.testing.assert_array_equal(result, expected)

    def test_negative_values(self):
        """Test nb_dot with negative values."""
        a = np.array([[-1, 2], [3, -4], [-5, -6]])
        b = np.array([2, -1])
        result = nb_dot(a, b)

        expected = np.array(
            [-4, 10, -4]
        )  # [-1*2 + 3*(-1), 2*2 + (-4)*(-1), -5*2 + (-6)*(-1)]
        np.testing.assert_array_equal(result, expected)

    def test_large_arrays(self):
        """Test nb_dot with larger arrays."""
        np.random.seed(42)
        a = np.random.randint(0, 10, size=(100, 50))
        b = np.random.randint(0, 10, size=50)
        result = nb_dot(a, b)

        # Compare with numpy's built-in dot product
        expected = np.dot(a, b)
        np.testing.assert_array_equal(result, expected)
        assert len(result) == 100

    def test_error_1d_array(self):
        """Test nb_dot raises ValueError for 1D array."""
        a = np.array([1, 2, 3])  # 1D array
        b = np.array([1, 2, 3])

        with pytest.raises(
            ValueError, match="a must be a 2-dimensional array or DataFrame"
        ):
            nb_dot(a, b)

    def test_error_3d_array(self):
        """Test nb_dot raises ValueError for 3D array."""
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 3D array
        b = np.array([1, 2])

        with pytest.raises(
            ValueError, match="a must be a 2-dimensional array or DataFrame"
        ):
            nb_dot(a, b)

    def test_error_shape_mismatch(self):
        """Test nb_dot raises ValueError for mismatched shapes."""
        a = np.array([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
        b = np.array([1, 2])  # shape (2,) - mismatch with a.shape[1]=3

        with pytest.raises(ValueError, match="shapes .* are not aligned"):
            nb_dot(a, b)

    def test_pandas_series_input_b(self):
        """Test nb_dot with pandas Series as b parameter."""
        a = np.array([[1, 2], [3, 4]])
        b = pd.Series([2, 3])
        result = nb_dot(a, b)

        expected = np.array([8, 18])  # [1*2 + 2*3, 3*2 + 4*3]
        np.testing.assert_array_equal(result, expected)

    def test_polars_series_input_b(self):
        """Test nb_dot with polars Series as b parameter."""
        a = np.array([[1, 2], [3, 4]])
        b = pl.Series([2, 3])
        result = nb_dot(a, b)

        expected = np.array([8, 18])  # [1*2 + 2*3, 3*2 + 4*3]
        np.testing.assert_array_equal(result, expected)

    def test_mixed_dtypes(self):
        """Test nb_dot with mixed integer and float dtypes."""
        a = np.array([[1, 2.5], [3.5, 4]], dtype=float)
        b = np.array([2, 3], dtype=int)
        result = nb_dot(a, b)

        expected = np.array([9.5, 19.0])  # [1*2 + 2.5*3, 3.5*2 + 4*3]
        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize("array_type", [np.array, pd.DataFrame, pl.DataFrame])
    def test_different_input_types(self, array_type):
        """Test nb_dot with different input array types."""
        if array_type == pd.DataFrame:
            a = array_type({"col1": [1, 2], "col2": [3, 4]})
        elif array_type == pl.DataFrame:
            a = array_type({"col1": [1, 2], "col2": [3, 4]})
        else:  # numpy
            a = array_type([[1, 3], [2, 4]])

        b = np.array([2, 1])
        result = nb_dot(a, b)

        expected_values = np.dot(a, b)  # [1*2 + 3*1, 2*2 + 4*1]

        if isinstance(a, pd.DataFrame):
            assert isinstance(result, pd.Series)
            np.testing.assert_array_equal(result.values, expected_values)
        else:
            assert isinstance(result, np.ndarray)
            np.testing.assert_array_equal(result, expected_values)

    @pytest.mark.parametrize("length", [10**3, 10**6, 10**7])
    def test_consistency_with_numpy_dot(self, length):
        """Test that nb_dot produces same results as numpy.dot."""
        np.random.seed(123)
        a = np.random.randn(length, 5)
        b = np.random.randn(5)

        result = nb_dot(a, b)
        expected = np.dot(a, b)

        np.testing.assert_array_almost_equal(result, expected)


class TestArraySplitWithChunkHandling:
    """Test the array_split_with_chunk_handling function."""

    def test_numpy_array_basic(self):
        """Test splitting a numpy array into basic chunks."""
        arr = np.array([1, 2, 3, 4, 5, 6])
        chunk_lengths = [2, 3, 1]

        result = array_split_with_chunk_handling(arr, chunk_lengths)

        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(result[1], [3, 4, 5])
        np.testing.assert_array_equal(result[2], [6])

    def test_pandas_series(self):
        """Test splitting a pandas Series."""
        series = pd.Series([10, 20, 30, 40, 50])
        chunk_lengths = [2, 2, 1]

        result = array_split_with_chunk_handling(series, chunk_lengths)

        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [10, 20])
        np.testing.assert_array_equal(result[1], [30, 40])
        np.testing.assert_array_equal(result[2], [50])

    def test_pyarrow_array(self):
        """Test splitting a PyArrow Array."""
        arr = pa.array([1, 2, 3, 4, 5, 6, 7])
        chunk_lengths = [3, 2, 2]

        result = array_split_with_chunk_handling(arr, chunk_lengths)

        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [1, 2, 3])
        np.testing.assert_array_equal(result[1], [4, 5])
        np.testing.assert_array_equal(result[2], [6, 7])

    def test_pyarrow_chunked_array_optimized_path(self):
        """Test the optimized path for PyArrow ChunkedArray when chunks align."""
        # Create chunked array with chunks that align with desired split
        chunk1 = pa.array([1, 2])
        chunk2 = pa.array([3, 4, 5])
        chunk3 = pa.array([6])
        chunked_arr = pa.chunked_array([chunk1, chunk2, chunk3])

        # Request splits that align with existing chunks
        chunk_lengths = [2, 3, 1]

        result = array_split_with_chunk_handling(chunked_arr, chunk_lengths)

        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(result[1], [3, 4, 5])
        np.testing.assert_array_equal(result[2], [6])

        # Verify result arrays are numpy arrays
        assert all(isinstance(chunk, np.ndarray) for chunk in result)

    def test_pyarrow_chunked_array_fallback_path(self):
        """Test fallback path when PyArrow ChunkedArray chunks don't align."""
        # Create chunked array with chunks that don't align with desired split
        chunk1 = pa.array([1, 2, 3])
        chunk2 = pa.array([4, 5, 6])
        chunked_arr = pa.chunked_array([chunk1, chunk2])

        # Request splits that don't align with existing chunks
        chunk_lengths = [2, 2, 2]

        result = array_split_with_chunk_handling(chunked_arr, chunk_lengths)

        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(result[1], [3, 4])
        np.testing.assert_array_equal(result[2], [5, 6])

    def test_single_chunk(self):
        """Test splitting into a single chunk."""
        arr = np.array([1, 2, 3, 4])
        chunk_lengths = [4]

        result = array_split_with_chunk_handling(arr, chunk_lengths)

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], [1, 2, 3, 4])

    def test_empty_array(self):
        """Test splitting an empty array."""
        arr = np.array([])
        chunk_lengths = []

        result = array_split_with_chunk_handling(arr, chunk_lengths)

        # np.array_split with empty offsets returns [array([], dtype=...)]
        # so we expect 1 empty array, not 0 arrays
        assert len(result) == 1
        assert len(result[0]) == 0

    def test_single_element_chunks(self):
        """Test splitting into single element chunks."""
        arr = np.array([10, 20, 30])
        chunk_lengths = [1, 1, 1]

        result = array_split_with_chunk_handling(arr, chunk_lengths)

        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [10])
        np.testing.assert_array_equal(result[1], [20])
        np.testing.assert_array_equal(result[2], [30])

    def test_different_dtypes(self):
        """Test with different data types."""
        # Float array
        float_arr = np.array([1.1, 2.2, 3.3, 4.4])
        result = array_split_with_chunk_handling(float_arr, [2, 2])
        np.testing.assert_array_equal(result[0], [1.1, 2.2])
        np.testing.assert_array_equal(result[1], [3.3, 4.4])

        # String array
        str_arr = np.array(["a", "b", "c", "d"])
        result = array_split_with_chunk_handling(str_arr, [1, 3])
        np.testing.assert_array_equal(result[0], ["a"])
        np.testing.assert_array_equal(result[1], ["b", "c", "d"])

    def test_error_chunk_lengths_sum_mismatch(self):
        """Test error when chunk_lengths sum doesn't match array length."""
        arr = np.array([1, 2, 3, 4, 5])
        chunk_lengths = [2, 2]  # Sum is 4, but array length is 5

        with pytest.raises(
            ValueError,
            match=r"Sum of chunk_lengths \(4\) must equal array length \(5\)",
        ):
            array_split_with_chunk_handling(arr, chunk_lengths)

    def test_error_chunk_lengths_sum_too_large(self):
        """Test error when chunk_lengths sum is larger than array length."""
        arr = np.array([1, 2, 3])
        chunk_lengths = [2, 3]  # Sum is 5, but array length is 3

        with pytest.raises(
            ValueError,
            match=r"Sum of chunk_lengths \(5\) must equal array length \(3\)",
        ):
            array_split_with_chunk_handling(arr, chunk_lengths)

    def test_error_message_includes_chunk_lengths(self):
        """Test that error message includes the actual chunk_lengths for debugging."""
        arr = np.array([1, 2, 3, 4])
        chunk_lengths = [1, 1, 1]  # Sum is 3, but array length is 4

        with pytest.raises(ValueError, match=r"Got chunk_lengths: \[1, 1, 1\]"):
            array_split_with_chunk_handling(arr, chunk_lengths)

    def test_zero_length_chunks(self):
        """Test behavior with zero-length chunks (edge case)."""
        arr = np.array([1, 2, 3])
        chunk_lengths = [0, 2, 0, 1, 0]

        result = array_split_with_chunk_handling(arr, chunk_lengths)

        assert len(result) == 5
        assert len(result[0]) == 0  # Empty chunk
        np.testing.assert_array_equal(result[1], [1, 2])
        assert len(result[2]) == 0  # Empty chunk
        np.testing.assert_array_equal(result[3], [3])
        assert len(result[4]) == 0  # Empty chunk

    def test_chunked_array_different_chunk_counts(self):
        """Test ChunkedArray with different number of chunks than requested splits."""
        # ChunkedArray has 2 chunks, but we want 3 splits
        chunk1 = pa.array([1, 2, 3, 4])
        chunk2 = pa.array([5, 6])
        chunked_arr = pa.chunked_array([chunk1, chunk2])

        chunk_lengths = [2, 2, 2]  # 3 chunks requested

        result = array_split_with_chunk_handling(chunked_arr, chunk_lengths)

        assert len(result) == 3
        np.testing.assert_array_equal(result[0], [1, 2])
        np.testing.assert_array_equal(result[1], [3, 4])
        np.testing.assert_array_equal(result[2], [5, 6])

    @pytest.mark.parametrize(
        "arr_type",
        [
            lambda x: np.array(x),
            lambda x: pd.Series(x),
            lambda x: pa.array(x),
        ],
    )
    def test_consistent_results_across_types(self, arr_type):
        """Test that results are consistent across different array types."""
        data = [10, 20, 30, 40, 50, 60]
        arr = arr_type(data)
        chunk_lengths = [2, 3, 1]

        result = array_split_with_chunk_handling(arr, chunk_lengths)

        # Results should be the same regardless of input type
        expected_chunks = [[10, 20], [30, 40, 50], [60]]

        assert len(result) == len(expected_chunks)
        for actual, expected in zip(result, expected_chunks):
            np.testing.assert_array_equal(actual, expected)


class TestToArrow:
    """Test the to_arrow function with focus on zero-copy conversions."""

    def test_numpy_array_conversion(self):
        """Test conversion of NumPy arrays to PyArrow."""
        # Test different dtypes (non-boolean)
        test_arrays = [
            np.array([1, 2, 3, 4, 5], dtype=np.int64),
            np.array([1.1, 2.2, 3.3], dtype=np.float64),
            np.array(["a", "b", "c"], dtype="<U1"),
        ]

        for arr in test_arrays:
            result = to_arrow(arr)
            assert isinstance(result, pa.Array)

            # Some PyArrow arrays may require copy for to_numpy(), so allow that
            if result.type == pa.string():
                assert result.to_pylist() == arr.tolist()
            else:
                assert result.to_numpy(zero_copy_only=False).tolist() == arr.tolist()

            # Verify type preservation where possible
            if arr.dtype == np.int64:
                assert result.type == pa.int64()
            elif arr.dtype == np.float64:
                assert result.type == pa.float64()

    def test_numpy_boolean_array_zero_copy_only_true(self):
        """Test that boolean arrays raise TypeError with zero_copy_only=True."""
        bool_arr = np.array([True, False, True], dtype=bool)

        # Should raise TypeError with default zero_copy_only=True
        with pytest.raises(
            TypeError, match="Zero copy conversions not possible with boolean types"
        ):
            to_arrow(bool_arr)

    def test_numpy_boolean_array_zero_copy_only_false(self):
        """Test boolean array conversion with zero_copy_only=False."""
        bool_arr = np.array([True, False, True], dtype=bool)

        # Should work with zero_copy_only=False
        result = to_arrow(bool_arr, zero_copy_only=False)
        assert isinstance(result, pa.Array)
        assert result.type == pa.bool_()
        # For boolean arrays, use to_pylist() instead of to_numpy()
        assert result.to_pylist() == bool_arr.tolist()

    def test_pandas_series_conversion(self):
        """Test conversion of pandas Series to PyArrow."""
        # Basic series
        series = pd.Series([1, 2, 3, 4, 5])
        result = to_arrow(series)
        assert isinstance(result, pa.Array)
        assert result.to_numpy(zero_copy_only=False).tolist() == series.tolist()

        # Series with different dtypes
        float_series = pd.Series([1.1, 2.2, 3.3, 4.4])
        result = to_arrow(float_series)
        assert isinstance(result, pa.Array)
        assert result.type == pa.float64()
        np.testing.assert_array_equal(
            result.to_numpy(zero_copy_only=False), float_series.values
        )

    def test_pandas_boolean_series_zero_copy_only_true(self):
        """Test that pandas boolean Series raise TypeError with zero_copy_only=True."""
        bool_series = pd.Series([True, False, True, False])

        # Should raise TypeError with default zero_copy_only=True
        with pytest.raises(
            TypeError, match="Zero copy conversions not possible with boolean types"
        ):
            to_arrow(bool_series)

    def test_pandas_boolean_series_zero_copy_only_false(self):
        """Test pandas boolean Series conversion with zero_copy_only=False."""
        bool_series = pd.Series([True, False, True, False])

        # Should work with zero_copy_only=False
        result = to_arrow(bool_series, zero_copy_only=False)
        assert isinstance(result, pa.Array)
        assert result.type == pa.bool_()
        # For boolean arrays, use to_pylist() instead of to_numpy()
        assert result.to_pylist() == bool_series.tolist()

    def test_pandas_index_conversion(self):
        """Test conversion of pandas Index to PyArrow."""
        index = pd.Index([10, 20, 30, 40])
        result = to_arrow(index)
        assert isinstance(result, pa.Array)
        assert result.to_numpy(zero_copy_only=False).tolist() == index.tolist()

    def test_pandas_categorical_conversion(self):
        """Test conversion of pandas Categorical to PyArrow DictionaryArray."""
        categories = ["A", "B", "C"]
        cat = pd.Categorical(["A", "B", "A", "C", "B"], categories=categories)

        result = to_arrow(cat)
        assert isinstance(result, pa.DictionaryArray)

        # Check that dictionary values are preserved
        dictionary_values = result.dictionary.to_pylist()
        assert dictionary_values == categories

        # Check that indices are correct
        expected_indices = [0, 1, 0, 2, 1]  # A=0, B=1, C=2
        assert result.indices.to_pylist() == expected_indices

    def test_pandas_arrow_dtype_conversion(self):
        """Test conversion of pandas Series with ArrowDtype."""
        # Create pandas Series with PyArrow backend
        pa_array = pa.array([1, 2, 3, 4])
        series = pd.Series(pa_array, dtype=pd.ArrowDtype(pa.int64()))

        result = to_arrow(series)
        assert isinstance(result, pa.Array)
        assert result.type == pa.int64()
        assert result.to_numpy().tolist() == [1, 2, 3, 4]

    def test_polars_series_conversion(self):
        """Test conversion of polars Series to PyArrow."""
        pl_series = pl.Series([1, 2, 3, 4, 5])
        result = to_arrow(pl_series)

        assert isinstance(result, pa.Array)
        assert result.to_numpy(zero_copy_only=False).tolist() == pl_series.to_list()

    def test_pyarrow_array_passthrough(self):
        """Test that PyArrow Arrays are returned as-is (zero copy)."""
        original = pa.array([1, 2, 3, 4, 5])
        result = to_arrow(original)

        # Should be the exact same object (no copy)
        assert result is original
        assert id(result) == id(original)

    def test_pyarrow_chunked_array_passthrough(self):
        """Test that PyArrow ChunkedArrays are returned as-is (zero copy)."""
        chunk1 = pa.array([1, 2, 3])
        chunk2 = pa.array([4, 5, 6])
        original = pa.chunked_array([chunk1, chunk2])

        result = to_arrow(original)

        # Should be the exact same object (no copy)
        assert result is original
        assert id(result) == id(original)
        assert isinstance(result, pa.ChunkedArray)
        assert len(result.chunks) == 2

    def test_zero_copy_verification_numpy(self):
        """Verify that NumPy array conversion minimizes copying."""
        # Create a large array to make copying more expensive
        large_arr = np.arange(1000, dtype=np.int64)
        original_data_ptr = large_arr.__array_interface__["data"][0]

        result = to_arrow(large_arr)

        # PyArrow should attempt to use the same memory where possible
        # Note: PyArrow may still need to copy for compatibility reasons,
        # but we verify the conversion succeeds efficiently
        assert isinstance(result, pa.Array)
        assert result.type == pa.int64()
        np.testing.assert_array_equal(result.to_numpy(zero_copy_only=False), large_arr)

    def test_zero_copy_verification_polars(self):
        """Verify that Polars Series conversion is efficient."""
        # Polars to_arrow() should be zero-copy
        large_series = pl.Series(list(range(1000)))

        result = to_arrow(large_series)

        assert isinstance(result, pa.Array)
        assert result.to_numpy(zero_copy_only=False).tolist() == large_series.to_list()

    def test_unsupported_type_error(self):
        """Test that unsupported types raise TypeError."""
        with pytest.raises(TypeError, match="Cannot convert type.*to arrow"):
            to_arrow("not_an_array_type")

        with pytest.raises(TypeError, match="Cannot convert type.*to arrow"):
            to_arrow({"dict": "not_supported"})

    def test_empty_arrays(self):
        """Test conversion of empty arrays."""
        # Empty NumPy array
        empty_np = np.array([], dtype=np.int64)
        result = to_arrow(empty_np)
        assert isinstance(result, pa.Array)
        assert len(result) == 0
        assert result.type == pa.int64()

        # Empty pandas Series
        empty_series = pd.Series([], dtype="int64")
        result = to_arrow(empty_series)
        assert isinstance(result, pa.Array)
        assert len(result) == 0

        # Empty PyArrow array (passthrough)
        empty_arrow = pa.array([], type=pa.int64())
        result = to_arrow(empty_arrow)
        assert result is empty_arrow

    def test_null_value_handling(self):
        """Test conversion of arrays with null/NaN values."""
        # Test with pandas Series containing explicit None values for object dtype
        series_with_none = pd.Series([1, None, 3, None], dtype="object")
        result = to_arrow(series_with_none)
        assert isinstance(result, pa.Array)
        assert result.null_count == 2

        # Test with string arrays containing None
        str_series_with_none = pd.Series(["a", None, "c", None], dtype="object")
        result = to_arrow(str_series_with_none)
        assert isinstance(result, pa.Array)
        assert result.null_count == 2

    def test_datetime_conversion(self):
        """Test conversion of datetime arrays."""
        dates = pd.date_range("2023-01-01", periods=4)
        result = to_arrow(dates)
        assert isinstance(result, pa.Array)
        assert pa.types.is_timestamp(result.type)

    def test_string_arrays(self):
        """Test conversion of string arrays."""
        # NumPy string array
        str_array = np.array(["hello", "world", "test"])
        result = to_arrow(str_array)
        assert isinstance(result, pa.Array)
        assert result.to_pylist() == ["hello", "world", "test"]

        # Pandas string series
        str_series = pd.Series(["pandas", "arrow", "conversion"])
        result = to_arrow(str_series)
        assert isinstance(result, pa.Array)
        assert result.to_pylist() == ["pandas", "arrow", "conversion"]

    def test_memory_efficiency_large_data(self):
        """Test that conversion is memory efficient for large datasets."""
        # Create a moderately large dataset
        large_data = np.random.randint(0, 1000, size=10_000)

        # Convert to various formats and then to arrow
        formats_to_test = [
            large_data,  # NumPy
            pd.Series(large_data),  # Pandas
            pa.array(large_data),  # PyArrow (should be passthrough)
        ]

        for data in formats_to_test:
            result = to_arrow(data)
            assert isinstance(result, pa.Array)
            np.testing.assert_array_equal(
                result.to_numpy(zero_copy_only=False), large_data
            )

            # For PyArrow input, should be same object
            if isinstance(data, pa.Array):
                assert result is data

    @pytest.mark.parametrize(
        "dtype",
        [
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
            np.float32,
            np.float64,
        ],
    )
    def test_dtype_preservation(self, dtype):
        """Test that dtypes are preserved correctly during conversion."""
        arr = np.array([1, 2, 3], dtype=dtype)

        result = to_arrow(arr)
        assert isinstance(result, pa.Array)

        # Convert back and check dtype is preserved
        roundtrip = result.to_numpy(zero_copy_only=False)
        if dtype in (np.int8, np.int16, np.int32):
            # PyArrow may upcast small integers to int64
            assert roundtrip.dtype in (dtype, np.int64)
        elif dtype in (np.uint8, np.uint16, np.uint32):
            # PyArrow may upcast small unsigned integers
            assert roundtrip.dtype in (dtype, np.uint64)
        else:
            assert roundtrip.dtype == dtype or np.issubdtype(roundtrip.dtype, dtype)

    def test_zero_copy_only_parameter_comprehensive(self):
        """Test the zero_copy_only parameter with various input types."""
        # Test with non-boolean types (should work with both True and False)
        int_arr = np.array([1, 2, 3, 4, 5])
        float_arr = np.array([1.1, 2.2, 3.3])
        str_arr = np.array(["a", "b", "c"])

        for arr in [int_arr, float_arr, str_arr]:
            # Should work with zero_copy_only=True (default)
            result_true = to_arrow(arr, zero_copy_only=True)
            assert isinstance(result_true, pa.Array)

            # Should work with zero_copy_only=False
            result_false = to_arrow(arr, zero_copy_only=False)
            assert isinstance(result_false, pa.Array)

            # Results should be equal
            if result_true.type == pa.string():
                assert result_true.to_pylist() == result_false.to_pylist()
            else:
                np.testing.assert_array_equal(
                    result_true.to_numpy(zero_copy_only=False),
                    result_false.to_numpy(zero_copy_only=False),
                )

        # Test with boolean types (should raise with True, work with False)
        bool_arr = np.array([True, False, True])
        bool_series = pd.Series([True, False, True, False])

        for bool_input in [bool_arr, bool_series]:
            # Should raise TypeError with zero_copy_only=True
            with pytest.raises(
                TypeError, match="Zero copy conversions not possible with boolean types"
            ):
                to_arrow(bool_input, zero_copy_only=True)

            # Should work with zero_copy_only=False
            result = to_arrow(bool_input, zero_copy_only=False)
            assert isinstance(result, pa.Array)
            assert result.type == pa.bool_()
            assert result.to_pylist() == bool_input.tolist()
