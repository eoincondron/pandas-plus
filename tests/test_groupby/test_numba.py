import numba as nb
import numpy as np
import pytest

from pandas_plus.groupby.numba import (NumbaGroupByMethods, group_count, group_mean, group_min, group_sum)
from pandas_plus.util import is_null as py_isnull, MIN_INT, NumbaReductionOps


@nb.njit
def is_null(x):
    return py_isnull(x)


@pytest.mark.parametrize(
    "values",
    [
        (3, 2),
        (2, 3),
        (-1, -5),
        (-5, 1),
        (1.2, 3.14),
        (-1.1516, 0),
    ],
)
@pytest.mark.parametrize("method", [sum, min, max])
def test_scalar_methods(method, values):
    result = getattr(NumbaReductionOps, method.__name__)(*values)
    expected = method(values)
    assert result == expected


class TestGroupSum:

    def test_basic_functionality(self):
        """Test basic functionality with simple inputs."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

        result = group_sum(group_key, values, ngroups=3, mask=None)
        expected = np.array([4.0, 7.0, 4.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_with_mask(self):
        """Test with a mask that filters some values."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask = np.array([True, True, False, True, False], dtype=np.bool_)

        result = group_sum(group_key, values, ngroups=3, mask=mask)

        # Expect: group 0 = 1.0 (skip 3.0 due to mask), group 1 = 2.0 (skip 5.0), group 2 = 4.0
        expected = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_empty_inputs(self):
        """Test with empty inputs."""
        group_key = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float64)
        mask = np.array([], dtype=np.bool_)

        result = group_sum(group_key, values, ngroups=0, mask=mask)

        expected = np.array([], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_integer_values(self):
        """Test with integer values instead of floats."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        result = group_sum(group_key, values, ngroups=3, mask=None)

        expected = np.array([4, 7, 4], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_all_masked(self):
        """Test with all values masked out."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask = np.array([False, False, False, False, False], dtype=np.bool_)

        result = group_sum(group_key, values, ngroups=3, mask=mask)

        expected = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_input_validation(self):
        """Test that inputs have compatible shapes and types."""
        # Test that group_key and values must have same length
        group_key = np.array([0, 1, 0], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        with pytest.raises(ValueError):
            group_sum(group_key, values, ngroups=3, mask=None)

        # Test that mask must have same length as group_key if not empty
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask = np.array([True, False, True], dtype=np.bool_)  # Wrong length

        with pytest.raises(ValueError):
            group_sum(group_key, values, ngroups=3, mask=mask)


class TestNullChecks:
    def test_is_null_with_nan(self):
        """Test that _is_null identifies NaN values correctly."""
        assert is_null(np.nan) == True
        assert is_null(np.float64("nan")) == True

    def test_is_null_with_numbers(self):
        """Test that _is_null returns False for valid numbers."""
        assert is_null(0.0) == False
        assert is_null(-1.5) == False
        assert is_null(1e10) == False

    def test_is_null_with_min_int(self):
        """Test that is_null identifies MIN_INT correctly."""
        assert is_null(MIN_INT) == True

    def test_is_null_with_normal_ints(self):
        """Test that is_null returns False for regular integers."""
        assert is_null(0) == False
        assert is_null(-1) == False
        assert is_null(100) == False


class TestGroupCount:
    def test_group_count_with_no_nulls(self):
        """Test _group_count with data containing no null values."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int_)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask = np.ones(len(group_key), dtype=bool)
        ngroups = 3

        result = group_count(group_key, values, mask=mask, ngroups=ngroups)

        # Expected: 2 values in group 0, 2 values in group 1, 1 value in group 2
        expected = np.array([2, 2, 1], dtype=np.int_)
        np.testing.assert_array_equal(result, expected)

    def test_group_count_with_nulls(self):
        """Test _group_count with data containing null values."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int_)
        values = np.array([1.0, np.nan, 3.0, np.nan, 5.0], dtype=np.float64)
        mask = np.ones(len(group_key), dtype=bool)
        ngroups = 3

        result = group_count(group_key, values, mask=mask, ngroups=ngroups)
        expected = np.array([2, 1, 0], dtype=np.int_)
        np.testing.assert_array_equal(result, expected)

    def test_group_count_with_mask(self):
        """Test _group_count with a mask that excludes some elements."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int_)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        # Mask out index 1 and 3
        mask = np.array([True, False, True, False, True], dtype=bool)
        ngroups = 3

        result = group_count(group_key, values, mask=mask, ngroups=ngroups)
        # Expected: 2 values in group 0, 1 value in group 1, 0 values in group 2
        expected = np.array([2, 1, 0], dtype=np.int_)
        np.testing.assert_array_equal(result, expected)

    def test_group_count_empty_mask(self):
        """Test _group_count with an empty mask (should process all values)."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int_)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask = None
        ngroups = 3

        result = group_count(group_key, values, mask=mask, ngroups=ngroups)
        # Expected: 2 values in group 0, 2 values in group 1, 1 value in group 2
        expected = np.array([2, 2, 1], dtype=np.int_)
        np.testing.assert_array_equal(result, expected)

    def test_group_count_with_int_nulls(self):
        """Test _group_count with integer data and int null check."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int_)
        values = np.array([1, MIN_INT, 3, MIN_INT, 5])
        mask = np.ones(len(group_key), dtype=bool)
        ngroups = 3

        result = group_count(group_key, values, mask=mask, ngroups=ngroups)
        # Expected: 2 values in group 0, 1 value in group 1, 0 values in group 2
        expected = np.array([2, 1, 0], dtype=np.int_)
        np.testing.assert_array_equal(result, expected)


class TestGroupMean:

    def test_basic_functionality(self):
        """Test basic functionality with simple inputs."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)

        result = group_mean(group_key, values, ngroups=3, mask=None)
        expected = np.array([2.0, 3.5, 4.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_with_mask(self):
        """Test with a mask that filters some values."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask = np.array([True, True, False, True, False], dtype=np.bool_)

        result = group_mean(group_key, values, ngroups=3, mask=mask)

        # Expect: group 0 = 1.0 (skip 3.0 due to mask), group 1 = 2.0 (skip 5.0), group 2 = 4.0
        expected = np.array([1.0, 2.0, 4.0], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_empty_inputs(self):
        """Test with empty inputs."""
        group_key = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float64)
        mask = np.array([], dtype=np.bool_)

        result = group_mean(group_key, values, ngroups=0, mask=mask)

        expected = np.array([], dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_integer_values(self):
        """Test with integer values instead of floats."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1, 2, 3, 4, 5], dtype=np.int64)

        result = group_mean(group_key, values, ngroups=3, mask=None)

        expected = np.array([2.0, 3.5, 4.0], dtype=float)
        np.testing.assert_array_equal(result, expected)

    def test_all_masked(self):
        """Test with all values masked out."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask = np.array([False, False, False, False, False], dtype=np.bool_)

        result = group_mean(group_key, values, ngroups=3, mask=mask)

        expected = np.array([np.nan] * 3, dtype=np.float64)
        np.testing.assert_array_equal(result, expected)

    def test_input_validation(self):
        """Test that inputs have compatible shapes and types."""
        # Test that group_key and values must have same length
        group_key = np.array([0, 1, 0], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        with pytest.raises(ValueError):
            group_mean(group_key, values, ngroups=3, mask=None)

        # Test that mask must have same length as group_key if not empty
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        mask = np.array([True, False, True], dtype=np.bool_)  # Wrong length

        with pytest.raises(ValueError):
            group_mean(group_key, values, ngroups=3, mask=mask)


@pytest.mark.parametrize("dtype", [float, int, bool, np.uint64])
def test_group_min(dtype):
    # Test that mask must have same length as group_key if not empty
    group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
    values = np.arange(5).astype(dtype)
    result = group_min(group_key, values, ngroups=3)
    expected = np.array([0, 1, 3], dtype=dtype)
    np.testing.assert_array_equal(result, expected)

    if dtype in (float, int):
        values[0] = np.nan if dtype == float else MIN_INT
        result = group_min(group_key, values, ngroups=3)
        expected = np.array([2, 1, 3], dtype=dtype)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("first", [2, 1, 0]),
        # ("last", [5, -1, 0]),
    ],
)
@pytest.mark.parametrize("dtype", [float, int, bool, np.uint64])
def test_group_first_last(method, dtype, expected):
    func = getattr(NumbaGroupByMethods, method)
    # Test that mask must have same length as group_key if not empty
    group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
    values = np.array([2, 1, 5, 0, -1]).astype(dtype)
    result = func(group_key, values, ngroups=3)
    expected = np.array(expected, dtype=dtype)
    np.testing.assert_array_equal(result, expected)
