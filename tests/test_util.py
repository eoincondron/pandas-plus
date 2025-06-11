import unittest

import numba as nb
import numpy as np
import pandas as pd
import polars as pl
import polars.testing
import pytest

from pandas_plus.util import (
    MAX_INT, MIN_INT, _get_first_non_null, _null_value_for_array_type,
    convert_array_inputs_to_dict, get_array_name, is_null, pretty_cut
)


class TestArrayFunctions(unittest.TestCase):
    def test_get_array_name_with_numpy(self):
        # NumPy arrays don't have names
        arr = np.array([1, 2, 3])
        self.assertIsNone(get_array_name(arr))

    def test_get_array_name_with_pandas(self):
        # Pandas Series with name
        named_series = pd.Series([1, 2, 3], name="test_series")
        self.assertEqual(get_array_name(named_series), "test_series")

        named_series = pd.Series([1, 2, 3], name=0)
        self.assertEqual(get_array_name(named_series), 0)

        # Pandas Series without name
        unnamed_series = pd.Series([1, 2, 3])
        self.assertIsNone(get_array_name(unnamed_series))

        # Pandas Series with empty name
        empty_name_series = pd.Series([1, 2, 3], name="")
        self.assertIsNone(get_array_name(empty_name_series))

    def test_get_array_name_with_polars(self):
        # Polars Series with name
        named_series = pl.Series("test_series", [1, 2, 3])
        self.assertEqual(get_array_name(named_series), "test_series")

        # Polars Series with empty name
        empty_name_series = pl.Series("", [1, 2, 3])
        self.assertIsNone(get_array_name(empty_name_series))

    def test_convert_mapping_to_dict(self):
        # Test with dictionary
        input_dict = {"a": np.array([1, 2]), "b": np.array([3, 4])}
        result = convert_array_inputs_to_dict(input_dict)
        self.assertEqual(result, input_dict)

        # Test with other mapping types
        from collections import OrderedDict

        ordered_dict = OrderedDict([("x", np.array([1, 2])), ("y", np.array([3, 4]))])
        result = convert_array_inputs_to_dict(ordered_dict)
        self.assertEqual(result, dict(ordered_dict))

    def test_convert_list_to_dict(self):
        # Test with list of named arrays
        arrays = [
            pd.Series([1, 2, 3], name="first"),
            pd.Series([4, 5, 6], name="second"),
        ]
        expected = {"first": arrays[0], "second": arrays[1]}
        result = convert_array_inputs_to_dict(arrays)
        self.assertEqual(result.keys(), expected.keys())
        for k in expected:
            pd.testing.assert_series_equal(result[k], expected[k])

        # Test with list of unnamed arrays
        arrays = [
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
        ]
        expected = {"_arr_0": arrays[0], "_arr_1": arrays[1]}
        result = convert_array_inputs_to_dict(arrays)
        self.assertEqual(result.keys(), expected.keys())
        for k in expected:
            np.testing.assert_array_equal(result[k], expected[k])

        # Test with mixed named and unnamed arrays
        arrays = [
            pd.Series([1, 2, 3], name="named"),
            np.array([4, 5, 6]),
            pd.Series([7, 8, 9]),
        ]
        expected = {"named": arrays[0], "_arr_1": arrays[1], "_arr_2": arrays[2]}
        result = convert_array_inputs_to_dict(arrays)
        self.assertEqual(result.keys(), expected.keys())

    def test_convert_numpy_array_to_dict(self):
        # Test with 1D numpy array
        arr = np.array([1, 2, 3])
        result = convert_array_inputs_to_dict(arr)
        self.assertEqual(len(result), 1)
        self.assertTrue("_arr_0" in result)
        np.testing.assert_array_equal(result["_arr_0"], arr)

        # Test with 2D numpy array (should return empty dict as per function logic)
        arr_2d = np.array([[1, 2], [3, 4]])
        result = convert_array_inputs_to_dict(arr_2d)
        self.assertEqual(list(result), ["_arr_0", "_arr_1"])

        np.testing.assert_array_equal(result["_arr_0"], arr_2d[:, 0])

    def test_convert_series_to_dict(self):
        # Test with named pandas Series
        series = pd.Series([1, 2, 3], name="test_series")
        result = convert_array_inputs_to_dict(series)
        self.assertEqual(len(result), 1)
        self.assertTrue("test_series" in result)
        pd.testing.assert_series_equal(result["test_series"], series)

        # Test with unnamed pandas Series
        series = pd.Series([1, 2, 3])
        result = convert_array_inputs_to_dict(series)
        self.assertEqual(len(result), 1)
        self.assertTrue("_arr_0" in result)
        pd.testing.assert_series_equal(result["_arr_0"], series)

        # Test with polars Series
        series = pl.Series("polars_series", [1, 2, 3])
        result = convert_array_inputs_to_dict(series)
        self.assertEqual(len(result), 1)
        self.assertTrue("polars_series" in result)
        pl.testing.assert_series_equal(result["polars_series"], series)

    def test_convert_dataframe_to_dict(self):
        # Test with pandas DataFrame
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = convert_array_inputs_to_dict(df)
        self.assertEqual(len(result), 2)
        self.assertTrue("a" in result and "b" in result)
        pd.testing.assert_series_equal(result["a"], df["a"])
        pd.testing.assert_series_equal(result["b"], df["b"])

        # Test with polars DataFrame
        df = pl.DataFrame({"x": [1, 2], "y": [3, 4]})
        result = convert_array_inputs_to_dict(df)
        self.assertEqual(len(result), 2)
        self.assertTrue("x" in result and "y" in result)
        pl.testing.assert_series_equal(result["x"], df["x"])
        pl.testing.assert_series_equal(result["y"], df["y"])

    def test_unsupported_type(self):
        # Test with unsupported type
        with pytest.raises(TypeError, match="Input type <class 'int'> not supported"):
            convert_array_inputs_to_dict(123)


class TestIsNullFunction(unittest.TestCase):
    def test_is_null_python_floats(self):
        """Test is_null with Python float values"""
        self.assertTrue(is_null(float('nan')))
        self.assertFalse(is_null(0.0))
        self.assertFalse(is_null(-1.5))
        self.assertFalse(is_null(1e10))

    def test_is_null_numpy_floats(self):
        """Test is_null with NumPy float values"""
        self.assertTrue(is_null(np.nan))
        self.assertTrue(is_null(np.float64('nan')))
        self.assertFalse(is_null(np.float64(0.0)))
        self.assertFalse(is_null(np.float32(-1.5)))

    def test_is_null_integers(self):
        """Test is_null with integer values"""
        self.assertFalse(is_null(0))
        self.assertFalse(is_null(-1))
        self.assertFalse(is_null(100))
        # MIN_INT should be considered null for integers
        self.assertTrue(is_null(MIN_INT))

    def test_is_null_booleans(self):
        """Test is_null with boolean values"""
        self.assertFalse(is_null(True))
        self.assertFalse(is_null(False))


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


class TestJitIsNull(unittest.TestCase):
    def test_jit_is_null_floats(self):
        """Test JIT-compiled is_null with float values"""
        is_nan_null, is_zero_null = test_jit_float_null()
        self.assertTrue(is_nan_null)
        self.assertFalse(is_zero_null)

    def test_jit_is_null_integers(self):
        """Test JIT-compiled is_null with integer values"""
        is_min_int_null, is_zero_null = test_jit_int_null()
        self.assertTrue(is_min_int_null)
        self.assertFalse(is_zero_null)

    def test_jit_is_null_booleans(self):
        """Test JIT-compiled is_null with boolean values"""
        is_true_null, is_false_null = test_jit_bool_null()
        self.assertFalse(is_true_null)
        self.assertFalse(is_false_null)


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


class TestGetFirstNonNull(unittest.TestCase):
    def test_get_first_non_null_with_nans(self):
        """Test _get_first_non_null with array containing NaN values"""
        idx, val = test_jit_get_first_non_null_with_nans()
        self.assertEqual(idx, 2)
        self.assertEqual(val, 3.0)

    def test_get_first_non_null_all_nans(self):
        """Test _get_first_non_null with array of all NaN values"""
        idx, val = test_jit_get_first_non_null_all_nans()
        self.assertEqual(idx, -1)
        self.assertTrue(np.isnan(val))

    def test_get_first_non_null_no_nans(self):
        """Test _get_first_non_null with array containing no NaN values"""
        idx, val = test_jit_get_first_non_null_no_nans()
        self.assertEqual(idx, 0)
        self.assertEqual(val, 1.0)
        
    def test_get_first_non_null_with_integers(self):
        """Test _get_first_non_null with integer array containing MIN_INT values"""
        idx, val = test_jit_get_first_non_null_with_integers()
        self.assertEqual(idx, 1)
        self.assertEqual(val, 1)


class TestNullValueForArrayType(unittest.TestCase):
    def test_null_value_for_int64(self):
        """Test null value for int64 array"""
        arr = np.array([1, 2, 3], dtype=np.int64)
        null_value = _null_value_for_array_type(arr)
        self.assertEqual(null_value, MIN_INT)
        self.assertEqual(null_value, np.iinfo(np.int64).min)
    
    def test_null_value_for_int32(self):
        """Test null value for int32 array"""
        arr = np.array([1, 2, 3], dtype=np.int32)
        null_value = _null_value_for_array_type(arr)
        self.assertEqual(null_value, np.iinfo(np.int32).min)
    
    def test_null_value_for_int16(self):
        """Test null value for int16 array - should raise TypeError"""
        arr = np.array([1, 2, 3], dtype=np.int16)
        with self.assertRaises(TypeError):
            _null_value_for_array_type(arr)
    
    def test_null_value_for_float64(self):
        """Test null value for float64 array"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        null_value = _null_value_for_array_type(arr)
        self.assertTrue(np.isnan(null_value))

    def test_null_value_for_float32(self):
        """Test null value for float32 array"""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        null_value = _null_value_for_array_type(arr)
        self.assertTrue(np.isnan(null_value))
        self.assertEqual(null_value.dtype, np.float32)
    
    def test_null_value_for_uint64(self):
        """Test null value for uint64 array"""
        arr = np.array([1, 2, 3], dtype=np.uint64)
        null_value = _null_value_for_array_type(arr)
        self.assertEqual(null_value, np.iinfo(np.uint64).max)

    def test_null_value_for_uint32(self):
        """Test null value for uint32 array"""
        arr = np.array([1, 2, 3], dtype=np.uint32)
        null_value = _null_value_for_array_type(arr)
        self.assertEqual(null_value, np.iinfo(np.uint32).max)
    
    def test_null_value_for_uint16(self):
        """Test null value for uint16 array - should raise TypeError"""
        arr = np.array([1, 2, 3], dtype=np.uint16)
        with self.assertRaises(TypeError):
            _null_value_for_array_type(arr)
    
    def test_null_value_for_bool(self):
        """Test null value for boolean array - should raise TypeError"""
        arr = np.array([True, False], dtype=bool)
        with self.assertRaises(TypeError):
            _null_value_for_array_type(arr)
    
    def test_null_value_for_complex(self):
        """Test null value for complex array - should raise TypeError"""
        arr = np.array([1+2j, 3+4j], dtype=complex)
        with self.assertRaises(TypeError):
            _null_value_for_array_type(arr)


class TestPrettyCut(unittest.TestCase):
    """Test cases for the pretty_cut function."""

    def test_basic_integer_cut(self):
        """Test basic integer cutting with integer bins."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 3', '4 - 6', '7 - 9', ' > 9']
        self.assertEqual(result.categories.tolist(), expected_labels)
        
        # Check specific values
        self.assertEqual(result[0], ' <= 3')  # 1
        self.assertEqual(result[3], '4 - 6')  # 4
        self.assertEqual(result[6], '7 - 9')  # 7
        self.assertEqual(result[9], ' > 9')   # 10

    def test_basic_float_cut(self):
        """Test basic float cutting with float bins."""
        x = np.array([1.0, 2.5, 3.7, 4.2, 5.8, 6.1, 7.9, 8.3, 9.4, 10.7])
        bins = [3.0, 6.0, 9.0]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 3.0', '3.0 - 6.0', '6.0 - 9.0', ' > 9.0']
        self.assertEqual(result.categories.tolist(), expected_labels)

    def test_mixed_int_float_bins(self):
        """Test with integer data and float bins."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [3.5, 6.5, 9.5]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 3.5', '3.5 - 6.5', '6.5 - 9.5', ' > 9.5']
        self.assertEqual(result.categories.tolist(), expected_labels)

    def test_pandas_series_input(self):
        """Test with pandas Series input."""
        x = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='test_series')
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)
        
        # Should return a pandas Series
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, 'test_series')
        self.assertTrue(result.index.equals(x.index))
        
        expected_labels = [' <= 3', '4 - 6', '7 - 9', ' > 9']
        self.assertEqual(result.cat.categories.tolist(), expected_labels)

    def test_polars_series_input(self):
        """Test with polars Series input."""
        x = pl.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)
        
        # Should return a Categorical
        self.assertIsInstance(result, pd.Categorical)
        expected_labels = [' <= 3', '4 - 6', '7 - 9', ' > 9']
        self.assertEqual(result.categories.tolist(), expected_labels)

    def test_list_bins(self):
        """Test with bins provided as a list."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [3, 6, 9]  # List instead of array
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 3', '4 - 6', '7 - 9', ' > 9']
        self.assertEqual(result.categories.tolist(), expected_labels)

    def test_single_bin(self):
        """Test with a single bin value."""
        x = np.array([1, 2, 3, 4, 5])
        bins = [3]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 3', ' > 3']
        self.assertEqual(result.categories.tolist(), expected_labels)
        
        # Check values
        self.assertEqual(result[0], ' <= 3')  # 1
        self.assertEqual(result[4], ' > 3')   # 5

    def test_empty_array(self):
        """Test with empty input array."""
        x = np.array([])
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)
        
        self.assertEqual(len(result), 0)
        expected_labels = [' <= 3', '4 - 6', '7 - 9', ' > 9']
        self.assertEqual(result.categories.tolist(), expected_labels)

    def test_all_values_below_first_bin(self):
        """Test when all values are below the first bin."""
        x = np.array([1, 2])
        bins = [5, 10]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 5', '6 - 10', ' > 10']
        self.assertEqual(result.categories.tolist(), expected_labels)
        self.assertTrue(all(result == ' <= 5'))

    def test_all_values_above_last_bin(self):
        """Test when all values are above the last bin."""
        x = np.array([15, 20])
        bins = [5, 10]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 5', '6 - 10', ' > 10']
        self.assertEqual(result.categories.tolist(), expected_labels)
        self.assertTrue(all(result == ' > 10'))

    def test_float_with_nan_values(self):
        """Test float array with NaN values."""
        x = np.array([1.0, np.nan, 3.0, 4.0, np.nan, 6.0])
        bins = [2.0, 5.0]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 2.0', '2.0 - 5.0', ' > 5.0']
        self.assertEqual(result.categories.tolist(), expected_labels)
        
        # NaN values should result in NaN categories
        self.assertTrue(pd.isna(result[1]))
        self.assertTrue(pd.isna(result[4]))
        
        # Non-NaN values should be categorized correctly
        self.assertEqual(result[0], ' <= 2.0')    # 1.0
        self.assertEqual(result[2], '2.0 - 5.0')  # 3.0
        self.assertEqual(result[5], ' > 5.0')     # 6.0

    def test_integer_boundary_values(self):
        """Test integer boundary handling."""
        x = np.array([3, 4, 6, 7, 9, 10])
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 3', '4 - 6', '7 - 9', ' > 9']
        self.assertEqual(result.categories.tolist(), expected_labels)
        
        # Test boundary values
        self.assertEqual(result[0], ' <= 3')  # 3 (at boundary)
        self.assertEqual(result[1], '4 - 6')  # 4 (start of next bin)
        self.assertEqual(result[2], '4 - 6')  # 6 (at boundary)
        self.assertEqual(result[3], '7 - 9')  # 7 (start of next bin)
        self.assertEqual(result[4], '7 - 9')  # 9 (at boundary)
        self.assertEqual(result[5], ' > 9')   # 10 (above last bin)

    def test_float_boundary_values(self):
        """Test float boundary handling."""
        x = np.array([3.0, 3.1, 6.0, 6.1, 9.0, 9.1])
        bins = [3.0, 6.0, 9.0]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 3.0', '3.0 - 6.0', '6.0 - 9.0', ' > 9.0']
        self.assertEqual(result.categories.tolist(), expected_labels)
        
        # Test boundary values (floats don't get +1 adjustment)
        self.assertEqual(result[0], ' <= 3.0')    # 3.0 (at boundary)
        self.assertEqual(result[1], '3.0 - 6.0')  # 3.1 (just above boundary)
        self.assertEqual(result[2], '3.0 - 6.0')  # 6.0 (at boundary)
        self.assertEqual(result[3], '6.0 - 9.0')  # 6.1 (just above boundary)
        self.assertEqual(result[4], '6.0 - 9.0')  # 9.0 (at boundary)
        self.assertEqual(result[5], ' > 9.0')     # 9.1 (above last bin)

    def test_unsorted_bins(self):
        """Test behavior with unsorted bins."""
        x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        bins = [6, 3, 9]  # Unsorted
        result = pretty_cut(x, bins)
        
        # The function should still work because bins get converted to array
        # Labels will reflect the order given
        expected_labels = [' <= 6', '6 - 3', '3 - 9', ' > 9']
        self.assertEqual(result.categories.tolist(), expected_labels)

    @pytest.mark.parametrize("array_type", [np.array, pd.Series, pl.Series])
    def test_different_array_types(self, array_type):
        """Test with different array types."""
        if array_type == pd.Series:
            x = array_type([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], name='test')
        else:
            x = array_type([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        bins = [3, 6, 9]
        result = pretty_cut(x, bins)
        
        expected_labels = [' <= 3', '4 - 6', '7 - 9', ' > 9']
        
        if isinstance(x, pd.Series):
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(result.cat.categories.tolist(), expected_labels)
        else:
            self.assertIsInstance(result, pd.Categorical)
            self.assertEqual(result.categories.tolist(), expected_labels)

    def test_preserve_series_attributes(self):
        """Test that pandas Series attributes are preserved."""
        index = pd.Index(['a', 'b', 'c', 'd', 'e'], name='test_index')
        x = pd.Series([1, 2, 3, 4, 5], index=index, name='test_series')
        bins = [2, 4]
        result = pretty_cut(x, bins)
        
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.name, 'test_series')
        self.assertTrue(result.index.equals(x.index))
        self.assertEqual(result.index.name, 'test_index')


if __name__ == "__main__":
    unittest.main()
