import pytest
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import polars as pl

from pandas_plus.groupby.factorization import (
    factorize_1d,
    factorize_2d,
    monotonic_factorization,
)


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

        expected_codes = np.array([0, 1, 2, 0, 1])
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

        expected_codes = np.array([0, 1, 2, 0, 1])
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

        expected_codes = np.array([0, 1, 2, 0, 1])
        np.testing.assert_array_equal(codes, expected_codes)
        assert [labels[c] for c in codes] == list(zip(vals1, vals2))

        assert labels.nlevels == 2

    def test_categorical_input(self):
        """Test factorize_2d with categorical input."""
        vals1 = pd.Categorical(["x", "y", "z", "x", "y"])
        vals2 = [1, 2, 3, 1, 2]
        codes, labels = factorize_2d(vals1, vals2)

        expected_codes = np.array([0, 1, 2, 0, 1])
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
        expected_uniques = pd.CategoricalIndex(["a", "b", "c"])

        np.testing.assert_array_equal(codes, expected_codes)
        pd.testing.assert_index_equal(uniques, expected_uniques)

    def test_pandas_categorical_with_unused_categories(self):
        """Test factorize_1d with pandas Categorical having unused categories."""
        cat = pd.Categorical(["a", "b", "a"], categories=["a", "b", "c", "d"])
        codes, uniques = factorize_1d(cat)

        expected_codes = np.array([0, 1, 0])
        expected_uniques = pd.CategoricalIndex(["a", "b", "c", "d"])

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
        expected_uniques = pd.CategoricalIndex(["low", "medium", "high"], ordered=True)

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

    @pytest.mark.parametrize("n_uniques", [1000, 10_000, 1000_000])
    @pytest.mark.parametrize("n_levels", [2, 3])
    def test_performance_large_data(self, n_levels, n_uniques):
        np.random.seed(37)
        N = 10_000_000
        max_int = int(n_uniques ** (1 / n_levels))
        keys = [np.random.randint(0, max_int, N) for _ in range(n_levels)]

        codes, index = factorize_2d(*keys)  # ensure JIT compilation

        t0 = time.perf_counter()
        codes, index = factorize_2d(*keys)
        duration = time.perf_counter() - t0

        assert index.is_unique
        expanded = index[codes]

        for i, key in enumerate(keys):
            assert (expanded.get_level_values(i) == key).all()

        t0 = time.perf_counter()
        pd.Series(keys[0]).groupby(keys).grouper._get_compressed_codes()
        pandas_duration = time.perf_counter() - t0

        assert pandas_duration > 1.5 * duration


@pytest.mark.parametrize("use_chunks", [False, True])
@pytest.mark.parametrize("partial", [False, True])
@pytest.mark.parametrize("arr_type", [int, "m8[ns]", float])
def test_monotonic_factorization_on_montonic(use_chunks, partial, arr_type):
    sorted_arr = np.repeat(2 + np.arange(6), 2).astype(arr_type)
    expected_cutoff = len(sorted_arr)
    if partial:
        sorted_arr = np.concat([sorted_arr, sorted_arr])
    if use_chunks:
        sorted_arr = pa.chunked_array(np.array_split(sorted_arr, 5))

    cutoff, codes, labels = monotonic_factorization(sorted_arr)
    assert cutoff == expected_cutoff

    expected_codes, expected_labels = factorize_1d(sorted_arr)
    if not partial:
        np.testing.assert_array_equal(codes, expected_codes)
        np.testing.assert_array_equal(labels, expected_labels)
