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
    factorize_1d,
    factorize_2d,
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
def test_jit_float_null():
    return is_null(np.nan), is_null(0.0)


@nb.njit
def test_jit_int_null():
    return is_null(MIN_INT), is_null(0)


@nb.njit
def test_jit_bool_null():
    return is_null(True), is_null(False)


class TestJitIsNull:
    def test_jit_is_null_floats(self):
        """Test JIT-compiled is_null with float values"""
        is_nan_null, is_zero_null = test_jit_float_null()
        assert is_nan_null
        assert not is_zero_null

    def test_jit_is_null_integers(self):
        """Test JIT-compiled is_null with integer values"""
        is_min_int_null, is_zero_null = test_jit_int_null()
        assert is_min_int_null
        assert not is_zero_null

    def test_jit_is_null_booleans(self):
        """Test JIT-compiled is_null with boolean values"""
        is_true_null, is_false_null = test_jit_bool_null()
        assert not is_true_null
        assert not is_false_null


# Test functions for _get_first_non_null
@nb.njit
def test_jit_get_first_non_null_with_nans():
    arr = np.array([np.nan, np.nan, 3.0, 4.0, np.nan])
    return _get_first_non_null(arr)


@nb.njit
def test_jit_get_first_non_null_all_nans():
    arr = np.array([np.nan, np.nan, np.nan])
    return _get_first_non_null(arr)


@nb.njit
def test_jit_get_first_non_null_no_nans():
    arr = np.array([1.0, 2.0, 3.0])
    return _get_first_non_null(arr)


@nb.njit
def test_jit_get_first_non_null_with_integers():
    arr = np.array([MIN_INT, 1, 2, MIN_INT])
    return _get_first_non_null(arr)


class TestGetFirstNonNull:
    def test_get_first_non_null_with_nans(self):
        """Test _get_first_non_null with array containing NaN values"""
        idx, val = test_jit_get_first_non_null_with_nans()
        assert idx == 2
        assert val == 3.0

    def test_get_first_non_null_all_nans(self):
        """Test _get_first_non_null with array of all NaN values"""
        idx, val = test_jit_get_first_non_null_all_nans()
        assert idx == -1
        assert np.isnan(val)

    def test_get_first_non_null_no_nans(self):
        """Test _get_first_non_null with array containing no NaN values"""
        idx, val = test_jit_get_first_non_null_no_nans()
        assert idx == 0
        assert val == 1.0

    def test_get_first_non_null_with_integers(self):
        """Test _get_first_non_null with integer array containing MIN_INT values"""
        idx, val = test_jit_get_first_non_null_with_integers()
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


class TestFactorize1D:
    """Test cases for the factorize_1d function."""

    def test_basic_integer_array(self):
        """Test factorize_1d with basic integer array."""
        values = [1, 2, 3, 1, 2, 3, 1]
        codes, labels = factorize_1d(values)

        assert isinstance(codes, np.ndarray)
        assert isinstance(labels, (np.ndarray, pd.Index))

        # Check that codes are correct
        expected_codes = np.array([0, 1, 2, 0, 1, 2, 0])
        np.testing.assert_array_equal(codes, expected_codes)

        # Check that labels are correct
        expected_labels = [1, 2, 3]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_string_array(self):
        """Test factorize_1d with string array."""
        values = ["a", "b", "c", "a", "b", "c"]
        codes, labels = factorize_1d(values)

        expected_codes = np.array([0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(codes, expected_codes)

        expected_labels = ["a", "b", "c"]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_float_array(self):
        """Test factorize_1d with float array."""
        values = [1.5, 2.5, 3.5, 1.5, 2.5]
        codes, labels = factorize_1d(values)

        expected_codes = np.array([0, 1, 2, 0, 1])
        np.testing.assert_array_equal(codes, expected_codes)

        expected_labels = [1.5, 2.5, 3.5]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_pandas_series_input(self):
        """Test factorize_1d with pandas Series input."""
        values = pd.Series([1, 2, 3, 1, 2, 3])
        codes, labels = factorize_1d(values)

        expected_codes = np.array([0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(codes, expected_codes)

        expected_labels = [1, 2, 3]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_categorical_series_input(self):
        """Test factorize_1d with categorical Series input."""
        values = pd.Categorical(["a", "b", "c", "a", "b"])
        codes, labels = factorize_1d(values)

        expected_codes = np.array([0, 1, 2, 0, 1])
        np.testing.assert_array_equal(codes, expected_codes)

        # For categorical, labels should be the categories
        expected_labels = ["a", "b", "c"]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_with_nan_values(self):
        """Test factorize_1d with NaN values."""
        values = [1.0, 2.0, np.nan, 1.0, np.nan, 3.0]
        codes, labels = factorize_1d(values)

        # NaN values should get code -1
        expected_codes = np.array([0, 1, -1, 0, -1, 2])
        np.testing.assert_array_equal(codes, expected_codes)

        expected_labels = [1.0, 2.0, 3.0]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_with_none_values(self):
        """Test factorize_1d with None values."""
        values = [1, 2, None, 1, None, 3]
        codes, labels = factorize_1d(values)

        # None values should get code -1
        expected_codes = np.array([0, 1, -1, 0, -1, 2])
        np.testing.assert_array_equal(codes, expected_codes)

        expected_labels = [1, 2, 3]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_empty_array(self):
        """Test factorize_1d with empty array."""
        values = []
        codes, labels = factorize_1d(values)

        assert len(codes) == 0
        assert len(labels) == 0

    def test_single_value(self):
        """Test factorize_1d with single value."""
        values = [42]
        codes, labels = factorize_1d(values)

        expected_codes = np.array([0])
        np.testing.assert_array_equal(codes, expected_codes)

        expected_labels = [42]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_sorted_option(self):
        """Test factorize_1d with sort=True."""
        values = ["c", "a", "b", "c", "a"]
        codes, labels = factorize_1d(values, sort=True)

        # With sort=True, labels should be sorted
        expected_labels = ["a", "b", "c"]
        np.testing.assert_array_equal(labels, expected_labels)

        # Codes should correspond to sorted labels
        expected_codes = np.array([2, 0, 1, 2, 0])
        np.testing.assert_array_equal(codes, expected_codes)

    def test_size_hint_option(self):
        """Test factorize_1d with size_hint parameter."""
        values = [1, 2, 3, 1, 2, 3]
        codes, labels = factorize_1d(values, size_hint=10)

        # Should still work correctly with size_hint
        expected_codes = np.array([0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(codes, expected_codes)

        expected_labels = [1, 2, 3]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_boolean_array(self):
        """Test factorize_1d with boolean array."""
        values = [True, False, True, False, True]
        codes, labels = factorize_1d(values)

        expected_codes = np.array([1, 0, 1, 0, 1])
        np.testing.assert_array_equal(codes, expected_codes)

        expected_labels = [False, True]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_duplicates_preserved(self):
        """Test that factorize_1d preserves duplicate patterns."""
        values = [1, 1, 1, 2, 2, 3]
        codes, labels = factorize_1d(values)

        expected_codes = np.array([0, 0, 0, 1, 1, 2])
        np.testing.assert_array_equal(codes, expected_codes)

        expected_labels = [1, 2, 3]
        np.testing.assert_array_equal(labels, expected_labels)

    def test_large_array(self):
        """Test factorize_1d with larger array."""
        np.random.seed(42)
        values = np.random.choice(["A", "B", "C", "D"], size=1000)
        codes, labels = factorize_1d(values)

        # Check that all codes are valid
        assert codes.max() < len(labels)
        assert codes.min() >= 0

        # Check that we can reconstruct the original values
        reconstructed = labels[codes]
        np.testing.assert_array_equal(reconstructed, values)

    def test_return_types(self):
        """Test that factorize_1d returns correct types."""
        values = [1, 2, 3, 1, 2]
        codes, labels = factorize_1d(values)

        assert isinstance(codes, np.ndarray)
        assert codes.dtype == np.int64
        assert isinstance(labels, (np.ndarray, pd.Index))


class TestFactorize2D:
    """Test cases for the factorize_2d function."""

    def test_basic_two_arrays(self):
        """Test factorize_2d with two basic arrays."""
        vals1 = [1, 2, 3, 1, 2]
        vals2 = ["a", "b", "c", "a", "b"]
        codes, labels = factorize_2d(vals1, vals2)

        assert isinstance(codes, np.ndarray)
        assert isinstance(labels, pd.MultiIndex)

        expected_codes = np.array([0, 4, 8, 0, 4])
        np.testing.assert_array_equal(codes, expected_codes)
        assert [labels[c] for c in codes] == list(zip(vals1, vals2))

        # Check that labels are MultiIndex
        assert labels.nlevels == 2
        expected_level0 = [1, 2, 3]
        expected_level1 = ["a", "b", "c"]
        np.testing.assert_array_equal(labels.levels[0], expected_level0)
        np.testing.assert_array_equal(labels.levels[1], expected_level1)

    def test_three_arrays(self):
        """Test factorize_2d with three arrays."""
        vals1 = [1, 2, 1, 2]
        vals2 = ["a", "b", "a", "b"]
        vals3 = [True, False, True, False]
        codes, labels = factorize_2d(vals1, vals2, vals3)

        assert isinstance(codes, np.ndarray)
        assert isinstance(labels, pd.MultiIndex)
        assert labels.nlevels == 3

        # Check that duplicate combinations get same codes
        assert codes[0] == codes[2]  # (1, 'a', True)
        assert codes[1] == codes[3]  # (2, 'b', False)

    def test_single_array(self):
        """Test factorize_2d with single array."""
        vals = [1, 2, 3, 1, 2, 3]
        codes, labels = factorize_2d(vals)

        assert isinstance(codes, np.ndarray)
        assert isinstance(labels, pd.MultiIndex)
        assert labels.nlevels == 1

        expected_codes = np.array([0, 1, 2, 0, 1, 2])
        np.testing.assert_array_equal(codes, expected_codes)

    def test_pandas_series_input(self):
        """Test factorize_2d with pandas Series input."""
        vals1 = pd.Series([1, 2, 3, 1, 2])
        vals2 = pd.Series(["x", "y", "z", "x", "y"])
        codes, labels = factorize_2d(vals1, vals2)

        expected_codes = np.array([0, 4, 8, 0, 4])
        np.testing.assert_array_equal(codes, expected_codes)
        assert [labels[c] for c in codes] == list(zip(vals1, vals2))

        assert labels.nlevels == 2

    def test_with_nan_values(self):
        """Test factorize_2d with NaN values."""
        vals1 = [1.0, 2.0, np.nan, 1.0, np.nan]
        vals2 = ["a", "b", "c", "a", "c"]
        codes, labels = factorize_2d(vals1, vals2)

        # NaN combinations should get unique codes
        assert codes[0] == codes[3]  # (1.0, 'a') should be same
        assert codes[2] == codes[4] == -1  # (NaN, 'c') should be same

        assert isinstance(labels, pd.MultiIndex)
        assert labels.nlevels == 2
        assert labels.levels[0].tolist() == [1.0, 2.0]

    def test_with_nan_values_in_two_levels(self):
        """Test factorize_2d with NaN values."""
        vals1 = [1.0, 2.0, np.nan, 1.0, np.nan]
        vals2 = ["a", "b", "c", "a", np.nan]
        codes, labels = factorize_2d(vals1, vals2)

        # NaN combinations should get unique codes
        assert codes[0] == codes[3]  # (1.0, 'a') should be same
        assert codes[2] == codes[4] == -1  # (NaN, 'c') should be same

        assert isinstance(labels, pd.MultiIndex)
        assert labels.nlevels == 2
        assert labels.levels[0].tolist() == [1.0, 2.0]
        assert labels.levels[1].tolist() == ["a", "b", "c"]

    def test_different_lengths_error(self):
        """Test factorize_2d with arrays of different lengths."""
        vals1 = [1, 2, 3]
        vals2 = ["a", "b"]  # Different length

        # This should raise an error due to different lengths
        with pytest.raises(ValueError):
            factorize_2d(vals1, vals2)

    def test_empty_arrays(self):
        """Test factorize_2d with empty arrays."""
        vals1 = []
        vals2 = []
        codes, labels = factorize_2d(vals1, vals2)

        assert len(codes) == 0
        assert isinstance(labels, pd.MultiIndex)
        assert labels.nlevels == 2

    def test_single_value_arrays(self):
        """Test factorize_2d with single value arrays."""
        vals1 = [42]
        vals2 = ["single"]
        codes, labels = factorize_2d(vals1, vals2)

        expected_codes = np.array([0])
        np.testing.assert_array_equal(codes, expected_codes)

        assert labels.nlevels == 2

    def test_boolean_arrays(self):
        """Test factorize_2d with boolean arrays."""
        vals1 = [True, False, True, False]
        vals2 = [False, True, False, True]
        codes, labels = factorize_2d(vals1, vals2)

        # Check that combinations are correctly identified
        assert codes[0] == codes[2]  # (True, False) should be same
        assert codes[1] == codes[3]  # (False, True) should be same

        assert labels.nlevels == 2

    def test_mixed_types(self):
        """Test factorize_2d with mixed data types."""
        vals1 = [1, 2.5, 3, 1, 2.5]
        vals2 = ["a", "b", "c", "a", "b"]
        codes, labels = factorize_2d(vals1, vals2)

        expected_codes = np.array([0, 4, 8, 0, 4])
        np.testing.assert_array_equal(codes, expected_codes)
        assert [labels[c] for c in codes] == list(zip(vals1, vals2))

        assert labels.nlevels == 2

    def test_categorical_input(self):
        """Test factorize_2d with categorical input."""
        vals1 = pd.Categorical(["x", "y", "z", "x", "y"])
        vals2 = [1, 2, 3, 1, 2]
        codes, labels = factorize_2d(vals1, vals2)

        expected_codes = np.array([0, 4, 8, 0, 4])
        np.testing.assert_array_equal(codes, expected_codes)
        assert [labels[c] for c in codes] == list(zip(vals1, vals2))

        assert labels.nlevels == 2

    def test_large_arrays(self):
        """Test factorize_2d with larger arrays."""
        np.random.seed(42)
        vals1 = np.random.choice(["A", "B", "C"], size=1000)
        vals2 = np.random.choice([1, 2, 3, 4], size=1000)
        codes, labels = factorize_2d(vals1, vals2)

        # Check that all codes are valid
        assert codes.max() < len(labels)
        assert codes.min() >= 0

        # Check that labels is MultiIndex with 2 levels
        assert labels.nlevels == 2
        assert isinstance(labels, pd.MultiIndex)

    def test_return_types(self):
        """Test that factorize_2d returns correct types."""
        vals1 = [1, 2, 3]
        vals2 = ["a", "b", "c"]
        codes, labels = factorize_2d(vals1, vals2)

        assert isinstance(codes, np.ndarray)
        assert codes.dtype == np.int64
        assert isinstance(labels, pd.MultiIndex)

    def test_unique_combinations(self):
        """Test that factorize_2d correctly identifies unique combinations."""
        vals1 = [1, 1, 2, 2, 1, 2]
        vals2 = ["a", "b", "a", "b", "a", "b"]
        codes, labels = factorize_2d(vals1, vals2)

        # Should have 4 unique combinations: (1,'a'), (1,'b'), (2,'a'), (2,'b')
        unique_codes = np.unique(codes)
        assert len(unique_codes) == 4

        # Check that identical combinations get same codes
        assert codes[0] == codes[4]  # (1, 'a')
        assert codes[1] != codes[2]  # (1, 'b') != (2, 'a')
        assert codes[3] == codes[5]  # (2, 'b')

    def test_codes_consistency(self):
        """Test that codes are consistently assigned."""
        vals1 = [3, 1, 2, 3, 1, 2]
        vals2 = ["z", "x", "y", "z", "x", "y"]
        codes, labels = factorize_2d(vals1, vals2)

        # Same combinations should have same codes
        assert codes[0] == codes[3]  # (3, 'z')
        assert codes[1] == codes[4]  # (1, 'x')
        assert codes[2] == codes[5]  # (2, 'y')


class TestFactorize1DComprehensive:
    """Comprehensive test suite for factorize_1d function using parametrized tests."""

    @pytest.mark.parametrize(
        "values,expected_codes,expected_uniques",
        [
            ([1, 2, 3, 1, 2, 3], [0, 1, 2, 0, 1, 2], [1, 2, 3]),
            (["a", "b", "c", "a", "b"], [0, 1, 2, 0, 1], ["a", "b", "c"]),
            ([True, False, True, True, False], [1, 0, 1, 1, 0], [False, True]),
            ([42], [0], [42]),
            ([42, 42, 42, 42], [0, 0, 0, 0], [42]),
            (
                [1 + 2j, 3 + 4j, 1 + 2j, 3 + 4j, 5 + 6j],
                [0, 1, 0, 1, 2],
                [1 + 2j, 3 + 4j, 5 + 6j],
            ),
            (["α", "β", "γ", "α", "β"], [0, 1, 2, 0, 1], ["α", "β", "γ"]),
        ],
        ids=[
            "integer_list",
            "string_values",
            "boolean_values",
            "single_value",
            "single_unique_repeated",
            "complex_numbers",
            "unicode_strings",
        ],
    )
    def test_basic_factorization(self, values, expected_codes, expected_uniques):
        """Test basic factorization with various input types."""
        codes, uniques = factorize_1d(values)

        np.testing.assert_array_equal(codes, np.array(expected_codes))
        np.testing.assert_array_equal(uniques, np.array(expected_uniques))

    @pytest.mark.parametrize(
        "values,expected_codes",
        [
            ([1.0, 2.0, np.nan, 1.0, np.nan], [0, 1, -1, 0, -1]),
            ([1, 2, None, 1, None], [0, 1, -1, 0, -1]),
            ([np.nan, np.nan, np.nan], [-1, -1, -1]),
            ([1, np.nan, 2, None, 1, np.nan, None], [0, -1, 1, -1, 0, -1, -1]),
        ],
        ids=["float_nan", "none_values", "all_nan", "mixed_nan_none"],
    )
    def test_null_value_handling(self, values, expected_codes):
        """Test handling of null values (NaN, None)."""
        codes, uniques = factorize_1d(values)

        np.testing.assert_array_equal(codes, np.array(expected_codes))
        # Check that NaN/None are not in uniques
        assert all(pd.notna(uniques))

    @pytest.mark.parametrize(
        "array_constructor",
        [
            lambda x: x,
            np.array,
            pd.Series,
        ],
        ids=["plain_list", "numpy_array", "pandas_series"],
    )
    def test_different_input_types(self, array_constructor):
        """Test factorize_1d with different array-like input types."""
        values = array_constructor([1, 2, 3, 1, 2])
        codes, uniques = factorize_1d(values)

        expected_codes = np.array([0, 1, 2, 0, 1])
        expected_uniques = np.array([1, 2, 3])

        np.testing.assert_array_equal(codes, expected_codes)
        np.testing.assert_array_equal(uniques, expected_uniques)

    @pytest.mark.parametrize(
        "sort_val,expected_uniques,expected_codes",
        [
            (False, ["c", "a", "y", "b"], [0, 1, 2, 3, 0, 1]),
            (True, ["a", "b", "c", "y"], [2, 0, 3, 1, 2, 0]),
        ],
        ids=["unsorted_first_appearance", "sorted_alphabetical"],
    )
    def test_sort_parameter(self, sort_val, expected_uniques, expected_codes):
        """Test sort parameter behavior."""
        values = ["c", "a", "y", "b", "c", "a"]
        codes, uniques = factorize_1d(values, sort=sort_val)

        np.testing.assert_array_equal(codes, np.array(expected_codes))
        np.testing.assert_array_equal(uniques, np.array(expected_uniques))

        # Verify reconstruction works
        reconstructed = uniques[codes]
        np.testing.assert_array_equal(reconstructed, values)

    def test_pandas_categorical_input(self):
        """Test factorize_1d with pandas Categorical input."""
        cat = pd.Categorical(["a", "b", "c", "a", "b"])
        codes, uniques = factorize_1d(cat)

        expected_codes = np.array([0, 1, 2, 0, 1])
        expected_uniques = pd.Index(["a", "b", "c"], dtype="object")

        np.testing.assert_array_equal(codes, expected_codes)
        pd.testing.assert_index_equal(uniques, expected_uniques)

    def test_pandas_categorical_with_unused_categories(self):
        """Test factorize_1d with pandas Categorical having unused categories."""
        cat = pd.Categorical(["a", "b", "a"], categories=["a", "b", "c", "d"])
        codes, uniques = factorize_1d(cat)

        expected_codes = np.array([0, 1, 0])
        expected_uniques = pd.Index(["a", "b", "c", "d"], dtype="object")

        np.testing.assert_array_equal(codes, expected_codes)
        pd.testing.assert_index_equal(uniques, expected_uniques)

    def test_ordered_categorical(self):
        """Test factorize_1d preserves category order for ordered categoricals."""
        cat = pd.Categorical(
            ["medium", "low", "high", "medium"],
            categories=["low", "medium", "high"],
            ordered=True,
        )
        codes, uniques = factorize_1d(cat)

        expected_codes = np.array([1, 0, 2, 1])
        expected_uniques = pd.Index(["low", "medium", "high"], dtype="object")

        np.testing.assert_array_equal(codes, expected_codes)
        pd.testing.assert_index_equal(uniques, expected_uniques)

    @pytest.mark.skipif(not hasattr(pa, "array"), reason="PyArrow not available")
    def test_pyarrow_array_input(self):
        """Test factorize_1d with PyArrow Array input."""
        values = pa.array([1, 2, 3, 1, 2])
        codes, uniques = factorize_1d(values)

        expected_codes = np.array([0, 1, 2, 0, 1])

        np.testing.assert_array_equal(codes, expected_codes)
        assert isinstance(uniques, pd.Index)
        assert len(uniques) == 3

    @pytest.mark.skipif(not hasattr(pl, "Series"), reason="Polars not available")
    def test_polars_series_input(self):
        """Test factorize_1d with Polars Series input."""
        values = pl.Series([1, 2, 3, 1, 2])
        codes, uniques = factorize_1d(values)

        expected_codes = np.array([0, 1, 2, 0, 1])

        np.testing.assert_array_equal(codes, expected_codes)
        assert isinstance(uniques, pd.Index)
        assert len(uniques) == 3

    def test_pandas_series_with_arrow_dtype(self):
        """Test factorize_1d with pandas Series backed by PyArrow."""
        try:
            values = pd.Series([1, 2, 3, 1, 2], dtype="int64[pyarrow]")
            codes, uniques = factorize_1d(values)

            expected_codes = np.array([0, 1, 2, 0, 1])

            np.testing.assert_array_equal(codes, expected_codes)
            assert isinstance(uniques, pd.Index)
            assert len(uniques) == 3
        except ImportError:
            pytest.skip("PyArrow backend not available for pandas")

    @pytest.mark.parametrize(
        "values",
        [
            [],
            np.array([]),
            pd.Series([]),
        ],
        ids=["empty_list", "empty_numpy", "empty_pandas"],
    )
    def test_empty_input(self, values):
        """Test factorize_1d with empty input."""
        codes, uniques = factorize_1d(values)

        assert len(codes) == 0
        assert len(uniques) == 0
        assert codes.dtype == np.int64

    def test_size_hint_parameter(self):
        """Test factorize_1d with size_hint parameter."""
        values = [1, 2, 3, 1, 2]
        codes, uniques = factorize_1d(values, size_hint=10)

        expected_codes = np.array([0, 1, 2, 0, 1])
        expected_uniques = np.array([1, 2, 3])

        np.testing.assert_array_equal(codes, expected_codes)
        np.testing.assert_array_equal(uniques, expected_uniques)

    def test_return_types(self):
        """Test that factorize_1d returns correct types."""
        values = [1, 2, 3]
        codes, uniques = factorize_1d(values)

        assert isinstance(codes, np.ndarray)
        assert codes.dtype == np.int64
        assert isinstance(uniques, (np.ndarray, pd.Index))

    def test_datetime_values(self):
        """Test factorize_1d with datetime values."""
        dates = pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-01"])
        codes, uniques = factorize_1d(dates)

        expected_codes = np.array([0, 1, 0])

        np.testing.assert_array_equal(codes, expected_codes)
        assert len(uniques) == 2

    def test_large_input_array(self):
        """Test factorize_1d with a larger input array."""
        np.random.seed(42)
        n = 10000
        values = np.random.choice(["A", "B", "C", "D"], size=n)
        codes, uniques = factorize_1d(values)

        assert len(codes) == n
        assert codes.dtype == np.int64
        assert len(uniques) <= 4
        assert all(code >= -1 and code < 4 for code in codes)

    def test_integer_overflow_edge_case(self):
        """Test factorize_1d with very large integers."""
        large_ints = [2**62, 2**63 - 1, 2**62, 2**63 - 1]
        codes, uniques = factorize_1d(large_ints)

        expected_codes = np.array([0, 1, 0, 1])
        expected_uniques = np.array([2**62, 2**63 - 1])

        np.testing.assert_array_equal(codes, expected_codes)
        np.testing.assert_array_equal(uniques, expected_uniques)
