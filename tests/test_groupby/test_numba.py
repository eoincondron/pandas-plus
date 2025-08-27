import numba as nb
import numpy as np
import pandas as pd
import pytest
from inspect import signature
from numba.typed import List as NumbaList

from pandas_plus.groupby.numba import (
    ScalarFuncs,
    _chunk_groupby_args,
    _group_by_reduce,
    group_nearby_members,
    group_count,
    group_mean,
    group_max,
    group_min,
    group_sum,
    rolling_sum,
    rolling_mean,
    rolling_min,
    rolling_max,
    cumsum,
    cumcount,
    cummin,
    cummax,
)
from pandas_plus.util import is_null as py_isnull, MIN_INT


@nb.njit
def is_null(x):
    return py_isnull(x)


@pytest.mark.parametrize("seen", [True, False])
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
def test_scalar_methods(method, values, seen):
    result = getattr(ScalarFuncs, "nan" + method.__name__)(*values, seen)
    expected = method(values) if seen else values[1], True
    assert result == expected


@pytest.mark.parametrize("seen", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        (-1.1516, np.nan),
        (3, np.iinfo(int).min),
    ],
)
@pytest.mark.parametrize("method", [sum, min, max])
def test_scalar_methods_with_nans(method, values, seen):
    result = getattr(ScalarFuncs, "nan" + method.__name__)(*values, seen)
    expected = values[0], seen
    assert result == expected


class TestChunkGroupbyArgs:

    def test_basic_functionality(self):
        """Test basic functionality with simple inputs."""
        # Setup
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = NumbaList([np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)])
        target = np.zeros(3)
        mask = np.ones(5, dtype=bool)
        reduce_func = ScalarFuncs.nansum
        n_chunks = 2

        # Call the function
        chunked_args = _chunk_groupby_args(
            n_chunks=n_chunks,
            group_key=group_key,
            values=values,
            target=target,
            mask=mask,
            reduce_func=reduce_func,
        )

        # Verify results
        assert len(chunked_args) == n_chunks

        # Each chunked argument should be bound arguments for _group_by_iterator
        for args in chunked_args:
            assert args.signature == signature(_group_by_reduce)

        # Check first chunk
        first_chunk = chunked_args[0]
        assert len(first_chunk.args[0]) <= 3  # group_key length should be around half
        assert len(first_chunk.args[1]) <= 3  # values length should be around half
        assert len(first_chunk.args[4]) <= 3  # mask length should be around half
        assert (
            first_chunk.args[2].shape == target.shape
        )  # target shape should be unchanged

        # Test with actual _group_by_iterator
        results = [_group_by_reduce(*args.args) for args in chunked_args]
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.shape == target.shape for r in results)

    def test_with_empty_mask(self):
        """Test with mask=None."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = NumbaList([np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)])
        target = np.zeros(3)
        reduce_func = ScalarFuncs.nansum
        n_chunks = 2

        chunked_args = _chunk_groupby_args(
            n_chunks=n_chunks,
            group_key=group_key,
            values=values,
            target=target,
            mask=np.array([], dtype=bool),
            reduce_func=reduce_func,
        )

        assert len(chunked_args) == n_chunks

        # Check that mask was properly prepared
        for args in chunked_args:
            mask = args.arguments["mask"]
            assert isinstance(mask, np.ndarray)  # mask is prepared
            assert mask.dtype == bool  # mask is boolean
            assert len(mask) == 0  # mask length matches chunk length

    def test_different_chunk_numbers(self):
        """Test with different numbers of chunks."""
        group_key = np.array([0, 1, 0, 2, 1, 3, 4, 5, 6, 7], dtype=np.int64)
        values = NumbaList(
            [
                np.array(
                    [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                    dtype=np.float64,
                )
            ]
        )
        target = np.zeros(8)
        mask = np.ones(10, dtype=bool)
        reduce_func = ScalarFuncs.nansum

        for n_chunks in [1, 2, 3, 5, 10]:
            chunked_args = _chunk_groupby_args(
                n_chunks=n_chunks,
                group_key=group_key,
                values=values,
                target=target,
                mask=mask,
                reduce_func=reduce_func,
            )

            assert len(chunked_args) == n_chunks

            # Check that the total length of all chunks equals the original length
            total_group_key_length = sum(len(args.args[0]) for args in chunked_args)
            assert total_group_key_length == len(group_key)

            total_values_length = sum(
                len(arr) for args in chunked_args for arr in args.args[1]
            )
            assert total_values_length == len(group_key)

    def test_target_is_copied(self):
        """Test that target arrays are copied, not shared between chunks."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = NumbaList([np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)])
        target = np.zeros(3)
        mask = np.ones(5, dtype=bool)
        reduce_func = ScalarFuncs.nansum
        n_chunks = 2

        chunked_args = _chunk_groupby_args(
            n_chunks=n_chunks,
            group_key=group_key,
            values=values,
            target=target,
            mask=mask,
            reduce_func=reduce_func,
        )

        # Modify target in first chunk and verify it doesn't affect second chunk
        chunked_args[0].args[2][0] = 999.0
        assert chunked_args[1].args[2][0] == 0.0

    def test_with_boolean_values(self):
        """Test with boolean values."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = NumbaList([np.array([True, False, True, False, True], dtype=bool)])
        target = np.zeros(3, dtype=bool)
        mask = np.ones(5, dtype=bool)
        reduce_func = ScalarFuncs.nansum
        n_chunks = 2

        chunked_args = _chunk_groupby_args(
            n_chunks=n_chunks,
            group_key=group_key,
            values=values,
            target=target,
            mask=mask,
            reduce_func=reduce_func,
        )

        # Check values were preserved
        all_values = np.concatenate(
            [arr for args in chunked_args for arr in args.args[1]]
        )
        np.testing.assert_array_equal(np.sort(all_values), np.sort(values[0]))

        # Test with actual _group_by_iterator
        results = [_group_by_reduce(*args.args) for args in chunked_args]
        assert all(r.dtype == bool for r in results)


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

    @pytest.mark.parametrize(
        "func", [group_count, group_sum, group_mean, group_min, group_max]
    )
    def test_multi_threaded(self, func):
        N = 2_000_000
        group_key = np.arange(N) % 5
        values = np.random.rand(N)
        result = func(group_key, values, ngroups=5, n_threads=4)
        func_name = func.__name__.split("_")[1]
        expected = pd.Series(values).groupby(group_key).agg(func_name).values
        np.testing.assert_array_almost_equal(result, expected)


class TestGroupNearbyMembers:
    def test_basic_functionality(self):
        """Test basic functionality with simple inputs."""
        # Setup - Group 0 has increasing values, Group 1 has some gaps
        group_key = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        values = np.array(
            [1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 20.0, 21.0], dtype=np.float64
        )
        max_diff = 5.0
        n_groups = 2  # We have group 0 and group 1

        # Call the function
        result = group_nearby_members(group_key, values, max_diff, n_groups)

        # Verify results:
        # All values in group 0 should be in the same subgroup (diff <= 5)
        # Group 1 should be split into two subgroups (10->11 and 20->21)
        expected_subgroups = np.array([0, 0, 0, 0, 1, 1, 2, 2])
        np.testing.assert_array_equal(result, expected_subgroups)

    def test_all_new_groups(self):
        """Test when all values exceed max_diff (each value is its own group)."""
        group_key = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        values = np.array([1.0, 10.0, 20.0, 5.0, 15.0], dtype=np.float64)
        max_diff = 1.0  # Very small difference threshold
        n_groups = 2

        result = group_nearby_members(group_key, values, max_diff, n_groups)

        # Each value should be in its own group
        expected_subgroups = np.array([0, 1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected_subgroups)

    def test_single_group_per_key(self):
        """Test when all values in each key group are within max_diff."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, 1.5, 2.0, 10.0, 10.5, 11.0], dtype=np.float64)
        max_diff = 10.0  # Large difference threshold
        n_groups = 2

        result = group_nearby_members(group_key, values, max_diff, n_groups)

        # Should have one subgroup per original group
        expected_subgroups = np.array([0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(result, expected_subgroups)

    def test_with_integer_values(self):
        """Test with integer values instead of floats."""
        group_key = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        values = np.array([1, 2, 10, 5, 15], dtype=np.int64)
        max_diff = 5
        n_groups = 2

        result = group_nearby_members(group_key, values, max_diff, n_groups)

        # Group 0: [1,2] should be one group, 10 another
        # Group 1: 5 and 15 should be separate groups
        expected_subgroups = np.array([0, 0, 1, 2, 3])
        np.testing.assert_array_equal(result, expected_subgroups)

    def test_interleaved_groups(self):
        """Test with interleaved group keys."""
        group_key = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        values = np.array([1.0, 10.0, 2.0, 20.0, 10.0, 21.0], dtype=np.float64)
        max_diff = 5.0
        n_groups = 2

        result = group_nearby_members(group_key, values, max_diff, n_groups)

        # Group 0: [1,2] should be one group, 10 another
        # Group 1: [10] one group, [20,21] another
        expected_subgroups = np.array([0, 1, 0, 2, 3, 2])
        np.testing.assert_array_equal(result, expected_subgroups)

    def test_empty_inputs(self):
        """Test with empty inputs."""
        group_key = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float64)
        max_diff = 5.0
        n_groups = 0

        result = group_nearby_members(group_key, values, max_diff, n_groups)

        # Should return empty array
        assert len(result) == 0

    def test_with_negative_values(self):
        """Test with negative values."""
        group_key = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        values = np.array([-10.0, -5.0, 0.0, -20.0, -15.0], dtype=np.float64)
        max_diff = 7.0
        n_groups = 2

        result = group_nearby_members(group_key, values, max_diff, n_groups)

        # Group 0: [-10, -5, 0] should split into two groups: [-10, -5] and [0]
        # Group 1: [-20, -15] should be one group (diff = 5)
        expected_subgroups = np.array([0, 0, 0, 1, 1])
        np.testing.assert_array_equal(result, expected_subgroups)

    def test_different_length_fail(self):
        """Test with negative values."""
        group_key = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        values = np.array([0, 2, 4, 4], dtype=np.float64)
        max_diff = 7.0
        n_groups = 2
        with pytest.raises(ValueError):
            group_nearby_members(group_key, values, max_diff, n_groups)


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
        ngroups = 3

        result = group_count(group_key, values, mask=None, ngroups=ngroups)

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


class TestRollingAggregation:
    """Test class for rolling aggregation functions."""

    def test_rolling_sum_1d_basic(self):
        """Test basic rolling sum with 1D values."""
        # Group key: [0, 0, 0, 1, 1, 1]
        # Values:    [1, 2, 3, 4, 5, 6]
        # Window:    2
        # Expected rolling sums:
        # Group 0: [1, 3, 5] (1, 1+2, 2+3)
        # Group 1: [4, 9, 11] (4, 4+5, 5+6)
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        ngroups = 2
        window = 2

        result = rolling_sum(group_key, values, ngroups, window)
        expected = np.array([np.nan, 3.0, 5.0, np.nan, 9.0, 11.0])
        np.testing.assert_array_almost_equal(result, expected)

        result = rolling_sum(group_key, values, ngroups, window, min_periods=1)
        expected = np.array([1, 3.0, 5.0, 4, 9.0, 11.0])

    def test_rolling_sum_1d_window_larger_than_group(self):
        """Test rolling sum with window size larger than group size."""
        group_key = np.array([0, 0, 1, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        ngroups = 2
        window = 5  # Larger than group size

        result = rolling_sum(group_key, values, ngroups, window, min_periods=1)
        expected = np.array([1.0, 3.0, 3.0, 7.0])  # Should sum all available values
        np.testing.assert_array_almost_equal(result, expected)

        result = rolling_sum(group_key, values, ngroups, window)
        expected = np.array([np.nan] * 4)  # Should sum all available values
        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_sum_2d_basic(self):
        """Test basic rolling sum with 2D values (multiple columns)."""
        group_key = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        values = np.array(
            [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0], [5.0, 50.0]],
            dtype=np.float64,
        )
        ngroups = 2
        window = 2

        result = rolling_sum(
            group_key,
            values,
            ngroups,
            window,
        )
        expected = np.array(
            [
                [np.nan, np.nan],  # First value in group 0
                [3.0, 30.0],  # 1+2, 10+20 for group 0
                [5.0, 50.0],  # 2+3, 20+30 for group 0
                [np.nan, np.nan],  # First value in group 1
                [9.0, 90.0],  # 4+5, 40+50 for group 1
            ]
        )

        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_sum_with_mask(self):
        """Test rolling sum with mask filtering."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        mask = np.array([True, False, True, True, True, False], dtype=bool)
        ngroups = 2
        window = 2

        result = rolling_sum(
            group_key, values, ngroups, window, min_periods=1, mask=mask
        )

        # Only positions where mask=True should have results
        # Group 0: positions 0,2 with values 1,3 -> [1, NaN, 4] (1, skip, 1+3)
        # Group 1: positions 3,4 with values 4,5 -> [4, 9, NaN] (4, 4+5, skip)
        expected = np.array([1.0, np.nan, 4.0, 4.0, 9.0, np.nan])

        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_sum_with_nan_values(self):
        """Test rolling sum with NaN values in input."""
        group_key = np.array([0, 0, 0, 1, 1], dtype=np.int64)
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0], dtype=np.float64)
        ngroups = 2
        window = 2

        result = rolling_sum(group_key, values, ngroups, window, min_periods=1)

        # NaN values should be skipped
        # Group 0: [1, NaN, 4] (1, skip NaN, 1+3)
        # Group 1: [4, 9] (4, 4+5)
        expected = np.array([1.0, 1.0, 3.0, 4.0, 9.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_mean_1d_basic(self):
        """Test basic rolling mean with 1D values."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([2.0, 4.0, 6.0, 8.0, 10.0, 12.0], dtype=np.float64)
        ngroups = 2
        window = 2

        result = rolling_mean(group_key, values, ngroups, window, min_periods=1)
        # Group 0: [2, 3, 5] (2/1, (2+4)/2, (4+6)/2)
        # Group 1: [8, 9, 11] (8/1, (8+10)/2, (10+12)/2)
        expected = np.array([2, 3.0, 5.0, 8.0, 9.0, 11.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_min_1d_basic(self):
        """Test basic rolling min with 1D values."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([3.0, 1.0, 4.0, 6.0, 2.0, 5.0], dtype=np.float64)
        ngroups = 2
        window = 2

        result = rolling_min(group_key, values, ngroups, window, min_periods=1)
        # Group 0: [3, 1, 1] (3, min(3,1), min(1,4))
        # Group 1: [6, 2, 2] (6, min(6,2), min(2,5))
        expected = np.array([3.0, 1.0, 1.0, 6.0, 2.0, 2.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_max_1d_basic(self):
        """Test basic rolling max with 1D values."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([3.0, 1.0, 4.0, 6.0, 2.0, 5.0], dtype=np.float64)
        ngroups = 2
        window = 2

        result = rolling_max(group_key, values, ngroups, window, min_periods=1)
        # Group 0: [3, 3, 4] (3, max(3,1), max(1,4))
        # Group 1: [6, 6, 5] (6, max(6,2), max(2,5))
        expected = np.array([3.0, 3.0, 4.0, 6.0, 6.0, 5.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_operations_2d(self):
        """Test that all rolling operations work with 2D values."""
        group_key = np.array([0, 0, 1, 1], dtype=np.int64)
        values = np.array(
            [[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]], dtype=np.float64
        )
        ngroups = 2
        window = 2

        # Test rolling sum
        result_sum = rolling_sum(group_key, values, ngroups, window, min_periods=1)
        expected_sum = np.array(
            [
                [1.0, 4.0],  # Group 0, first value
                [3.0, 7.0],  # Group 0, 1+2, 4+3
                [3.0, 2.0],  # Group 1, first value
                [7.0, 3.0],  # Group 1, 3+4, 2+1
            ]
        )
        np.testing.assert_array_almost_equal(result_sum, expected_sum)

        # Test rolling mean
        result_mean = rolling_mean(group_key, values, ngroups, window, min_periods=1)
        expected_mean = np.array(
            [
                [1.0, 4.0],  # Group 0, first value
                [1.5, 3.5],  # Group 0, (1+2)/2, (4+3)/2
                [3.0, 2.0],  # Group 1, first value
                [3.5, 1.5],  # Group 1, (3+4)/2, (2+1)/2
            ]
        )
        np.testing.assert_array_almost_equal(result_mean, expected_mean)

    def test_rolling_operations_empty_groups(self):
        """Test rolling operations with empty input."""
        group_key = np.array([], dtype=np.int64)
        values = np.array([], dtype=np.float64)
        ngroups = 0
        window = 2

        result = rolling_sum(group_key, values, ngroups, window)
        expected = np.array([])

        np.testing.assert_array_equal(result, expected)

    def test_rolling_operations_window_one(self):
        """Test rolling operations with window size 1."""
        group_key = np.array([0, 0, 1, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        ngroups = 2
        window = 1

        # With window=1, rolling sum should equal original values
        result = rolling_sum(group_key, values, ngroups, window)
        expected = values.copy()

        np.testing.assert_array_almost_equal(result, expected)

    def test_rolling_operations_with_negative_group_keys(self):
        """Test that negative group keys (null keys) are properly skipped."""
        group_key = np.array([0, -1, 0, 1, 1], dtype=np.int64)  # -1 represents null
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        ngroups = 2
        window = 2

        result = rolling_sum(group_key, values, ngroups, window, min_periods=1)

        # Position 1 with key=-1 should be skipped (remain NaN)
        expected = np.array([1.0, np.nan, 4.0, 4.0, 9.0])

        np.testing.assert_array_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "rolling_func", [rolling_sum, rolling_mean, rolling_min, rolling_max]
    )
    def test_rolling_operations_input_validation(self, rolling_func):
        """Test input validation for rolling operations."""
        group_key = np.array([0, 1, 0], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ngroups = 2
        window = 2

        # Test with 3D values (should raise error)
        values_3d = np.array([[[1.0]]], dtype=np.float64)
        with pytest.raises(ValueError, match="values must be 1D or 2D"):
            rolling_func(group_key[:1], values_3d, ngroups, window)

    def test_rolling_multithreading_2d(self):
        """Test multi-threading functionality for 2D rolling operations."""
        group_key = np.array([0, 0, 1, 1, 2, 2])
        values = np.array(
            [
                [1.0, 10.0, 100.0, 1000.0],
                [2.0, 20.0, 200.0, 2000.0],
                [3.0, 30.0, 300.0, 3000.0],
                [4.0, 40.0, 400.0, 4000.0],
                [5.0, 50.0, 500.0, 5000.0],
                [6.0, 60.0, 600.0, 6000.0],
            ]
        )
        ngroups = 3
        window = 2

        # Test single-threaded vs multi-threaded results
        result_single = rolling_sum(group_key, values, ngroups, window, n_threads=1)
        result_multi = rolling_sum(group_key, values, ngroups, window, n_threads=4)

        # Results should be identical
        np.testing.assert_array_almost_equal(result_single, result_multi)

        # Test with different rolling operations
        for rolling_func in [rolling_mean, rolling_min, rolling_max]:
            result_single = rolling_func(
                group_key, values, ngroups, window, n_threads=1
            )
            result_multi = rolling_func(group_key, values, ngroups, window, n_threads=4)
            np.testing.assert_array_almost_equal(result_single, result_multi)


@pytest.mark.parametrize("window", [1, 2, 5, 8])
@pytest.mark.parametrize("method", ["mean", "sum", "min", "max"])
def test_numba_rolling_sum_1d_equivalence_wth_min_periods(window, method):
    """Test equivalence of numba_rolling_sum_1d with pandas rolling sum."""
    arr = np.array([1, np.nan, -2, 3, np.nan, 4, 5, np.nan])
    key = np.zeros(len(arr), dtype=int)
    arr = np.repeat(arr, 2)
    key = np.arange(len(arr)) % 2
    func = dict(mean=rolling_mean, sum=rolling_sum, min=rolling_min, max=rolling_max)[
        method
    ]

    for min_periods in range(1, window + 1):
        x = func(key, arr, ngroups=1, window=window, min_periods=min_periods)
        expected = (
            pd.Series(arr)
            .groupby(key)
            .rolling(window, min_periods=min_periods)
            .agg(method)
            .sort_index(level=1)
            .to_numpy()
        )
        np.testing.assert_array_equal(x, expected)


class TestCumulativeAggregation:
    """Test class for cumulative aggregation functions (cumsum, cumcount, cummin, cummax)."""

    def test_cumsum_basic(self):
        """Test basic cumulative sum functionality."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        ngroups = 2

        result = cumsum(group_key, values, ngroups)
        expected = np.array([1.0, 3.0, 6.0, 4.0, 9.0, 15.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_cumsum_with_nan_skip_na_true(self):
        """Test cumsum with NaN values when skip_na=True."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0, np.nan], dtype=np.float64)
        ngroups = 2

        result = cumsum(group_key, values, ngroups, skip_na=True)
        expected = np.array([1.0, 1.0, 4.0, 4.0, 9.0, 9.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_cumsum_with_nan_skip_na_false(self):
        """Test cumsum with NaN values when skip_na=False."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, np.nan, 3.0, 4.0, 5.0, np.nan], dtype=np.float64)
        ngroups = 2

        result = cumsum(group_key, values, ngroups, skip_na=False)
        # With skip_na=False, NaN should propagate through cumulative sums
        expected = np.array([1.0, np.nan, np.nan, 4.0, 9.0, np.nan])

        np.testing.assert_array_almost_equal(result, expected)

    def test_cumsum_with_mask(self):
        """Test cumsum with boolean mask."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        mask = np.array([True, False, True, True, True, False], dtype=bool)
        ngroups = 2

        result = cumsum(group_key, values, ngroups, mask=mask)
        # Only process elements where mask is True
        expected = np.array([1.0, 1.0, 4.0, 4.0, 9.0, 9.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_cumcount_basic(self):
        """Test basic cumulative count functionality."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        ngroups = 2

        result = cumcount(group_key, values=None, ngroups=ngroups)
        expected = np.array([0, 1, 2, 0, 1, 2], dtype=np.int64)

        np.testing.assert_array_equal(result, expected)

    def test_cumcount_with_mask(self):
        """Test cumcount with boolean mask."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        mask = np.array([True, False, True, True, True, False], dtype=bool)
        ngroups = 2

        result = cumcount(group_key, values=None, ngroups=ngroups, mask=mask)
        # Only count elements where mask is True
        expected = np.array([0, 0, 1, 0, 1, 1], dtype=np.int64)

        np.testing.assert_array_equal(result, expected)

    def test_cumcount_with_values(self):
        """Test cumcount with boolean mask."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1, np.nan, 2, 3, np.nan, 4])
        ngroups = 2

        result = cumcount(group_key, values=values, ngroups=ngroups)
        expected = np.array([0, 0, 1, 0, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

        mask = np.array([True, True, False, True, False, True], dtype=bool)
        # Only count elements where mask is True
        result = cumcount(group_key, values=values, ngroups=ngroups, mask=mask)
        expected = np.array([0, 0, 0, 0, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_cummin_basic(self):
        """Test basic cumulative minimum functionality."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([3.0, 1.0, 4.0, 2.0, 5.0, 1.0], dtype=np.float64)
        ngroups = 2

        result = cummin(group_key, values, ngroups)
        expected = np.array([3.0, 1.0, 1.0, 2.0, 2.0, 1.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_cummax_basic(self):
        """Test basic cumulative maximum functionality."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, 3.0, 2.0, 4.0, 1.0, 5.0], dtype=np.float64)
        ngroups = 2

        result = cummax(group_key, values, ngroups)
        expected = np.array([1.0, 3.0, 3.0, 4.0, 4.0, 5.0])

        np.testing.assert_array_almost_equal(result, expected)

    def test_cumulative_operations_with_negative_group_keys(self):
        """Test cumulative operations handle negative group keys correctly."""
        group_key = np.array([0, 0, -1, 1, 1, -1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        ngroups = 2

        # Negative group keys should be ignored
        result_sum = cumsum(group_key, values, ngroups)
        expected_sum = np.array([1.0, 3.0, np.nan, 4.0, 9.0, np.nan])
        np.testing.assert_array_almost_equal(result_sum, expected_sum)

        result_count = cumcount(group_key, values=None, ngroups=ngroups)
        expected_count = np.array([0, 1, 0, 0, 1, 0], dtype=np.int64)
        np.testing.assert_array_equal(result_count, expected_count)

    def test_cumulative_operations_empty_groups(self):
        """Test cumulative operations with empty groups."""
        group_key = np.array([0, 0, 2, 2], dtype=np.int64)  # Group 1 is empty
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        ngroups = 3

        result_sum = cumsum(group_key, values, ngroups)
        expected_sum = np.array([1.0, 3.0, 3.0, 7.0])
        np.testing.assert_array_almost_equal(result_sum, expected_sum)

        result_count = cumcount(group_key, values=None, ngroups=ngroups)
        expected_count = np.array([0, 1, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(result_count, expected_count)

    def test_cumulative_operations_single_group(self):
        """Test cumulative operations with a single group."""
        group_key = np.array([0, 0, 0, 0], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        ngroups = 1

        result_sum = cumsum(group_key, values, ngroups)
        expected_sum = np.array([1.0, 3.0, 6.0, 10.0])
        np.testing.assert_array_almost_equal(result_sum, expected_sum)

        result_min = cummin(group_key, values, ngroups)
        expected_min = np.array([1.0, 1.0, 1.0, 1.0])
        np.testing.assert_array_almost_equal(result_min, expected_min)

        result_max = cummax(group_key, values, ngroups)
        expected_max = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_almost_equal(result_max, expected_max)

    def test_cumulative_operations_mixed_groups(self):
        """Test cumulative operations with mixed group patterns."""
        group_key = np.array([1, 0, 1, 0, 1], dtype=np.int64)
        values = np.array([5.0, 2.0, 3.0, 1.0, 4.0], dtype=np.float64)
        ngroups = 2

        result_sum = cumsum(group_key, values, ngroups)
        # Group 0: positions 1,3 → values 2,1 → cumsum 2,3
        # Group 1: positions 0,2,4 → values 5,3,4 → cumsum 5,8,12
        expected_sum = np.array([5.0, 2.0, 8.0, 3.0, 12.0])
        np.testing.assert_array_almost_equal(result_sum, expected_sum)

        result_count = cumcount(group_key, values=None, ngroups=ngroups)
        expected_count = np.array([0, 0, 1, 1, 2], dtype=np.int64)
        np.testing.assert_array_equal(result_count, expected_count)

    @pytest.mark.parametrize(
        "cumulative_func,operation", [(cumsum, "sum"), (cummin, "min"), (cummax, "max")]
    )
    def test_cumulative_operations_input_validation(self, cumulative_func, operation):
        """Test input validation for cumulative operations."""
        group_key = np.array([0, 0, 1, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        ngroups = 2

        # Test with invalid operation (should fail on dispatcher level)
        # These functions should work normally since numpy handles broadcasting
        result = cumulative_func(group_key, values, ngroups)
        assert result is not None
        assert len(result) == len(group_key)

    def test_cumcount_input_validation(self):
        """Test input validation for cumcount."""
        group_key = np.array([0, 0, 1, 1], dtype=np.int64)
        ngroups = 2

        # cumcount doesn't need values, should work fine
        result = cumcount(group_key, None, ngroups)
        expected = np.array([0, 1, 0, 1], dtype=np.int64)
        np.testing.assert_array_equal(result, expected)

    def test_cumulative_operations_large_groups(self):
        """Test cumulative operations with larger datasets."""
        np.random.seed(42)
        n = 1000
        n_groups = 5

        group_key = np.random.randint(0, n_groups, n, dtype=np.int64)
        values = np.random.randn(n).astype(np.float64)

        # Test that cumsum works and produces reasonable results
        result_sum = cumsum(group_key, values, n_groups)

        # Verify shape and that we have no unexpected NaNs where data exists
        assert result_sum.shape == (n,)
        valid_mask = group_key >= 0
        assert not np.isnan(result_sum[valid_mask]).all()

        # Test cumcount
        result_count = cumcount(group_key, None, n_groups)
        assert result_count.shape == (n,)
        assert result_count.dtype == np.int64
        assert (result_count >= 0).all()

    def test_cumulative_operations_comparison_with_pandas(self):
        """Compare our cumulative operations with pandas results."""
        group_key = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        ngroups = 2

        # Create pandas DataFrame for comparison
        df = pd.DataFrame({"group": group_key, "values": values})

        # Compare cumsum
        pandas_cumsum = df.groupby("group")["values"].cumsum().values
        our_cumsum = cumsum(group_key, values, ngroups)
        np.testing.assert_array_almost_equal(our_cumsum, pandas_cumsum)

        # Compare cumcount (now both start from 0)
        pandas_cumcount = df.groupby("group").cumcount().values
        our_cumcount = cumcount(group_key, None, ngroups)
        np.testing.assert_array_equal(our_cumcount, pandas_cumcount)

        # Compare cummin
        pandas_cummin = df.groupby("group")["values"].cummin().values
        our_cummin = cummin(group_key, values, ngroups)
        np.testing.assert_array_almost_equal(our_cummin, pandas_cummin)

        # Compare cummax
        pandas_cummax = df.groupby("group")["values"].cummax().values
        our_cummax = cummax(group_key, values, ngroups)
        np.testing.assert_array_almost_equal(our_cummax, pandas_cummax)

    def test_type_preservation_cumsum(self):
        """Test that cumsum follows proper type promotion rules."""
        group_key = np.array([0, 0, 1, 1], dtype=np.int64)
        ngroups = 2

        # Test signed integers → int64
        for int_type in [np.int8, np.int16, np.int32, np.int64]:
            values = np.array([1, 2, 3, 4], dtype=int_type)
            result = cumsum(group_key, values, ngroups)
            assert (
                result.dtype == np.int64
            ), f"Expected int64 for {int_type}, got {result.dtype}"

        # Test boolean → int64
        bool_values = np.array([True, False, True, False], dtype=bool)
        result = cumsum(group_key, bool_values, ngroups)
        assert result.dtype == np.int64, f"Expected int64 for bool, got {result.dtype}"

        # Test unsigned integers → uint64
        for uint_type in [np.uint8, np.uint16, np.uint32, np.uint64]:
            values = np.array([1, 2, 3, 4], dtype=uint_type)
            result = cumsum(group_key, values, ngroups)
            assert (
                result.dtype == np.uint64
            ), f"Expected uint64 for {uint_type}, got {result.dtype}"

        # Test float32 → float32
        float32_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        result = cumsum(group_key, float32_values, ngroups)
        assert result.dtype == np.float32, f"Expected float32, got {result.dtype}"

        # Test float64 → float64
        float64_values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        result = cumsum(group_key, float64_values, ngroups)
        assert result.dtype == np.float64, f"Expected float64, got {result.dtype}"

    @pytest.mark.parametrize(
        "dtype",
        [
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float32",
            "float64",
            "bool",
        ],
    )
    def test_type_preservation_cummin_cummax(self, dtype):
        """Test that cummin and cummax preserve exact input types."""
        group_key = np.array([0, 0, 1, 1], dtype=np.int64)
        ngroups = 2
        values = np.array([3, 1, 4, 2], dtype=dtype)
        # Test cummin
        result_min = cummin(group_key, values, ngroups)

        assert (
            result_min.dtype == dtype
        ), f"cummin: Expected {dtype} for input {values.dtype}, got {result_min.dtype}"

        # Test cummax
        result_max = cummax(group_key, values, ngroups)
        assert (
            result_max.dtype == dtype
        ), f"cummax: Expected {dtype} for input {values.dtype}, got {result_max.dtype}"

    def test_type_preservation_boolean_edge_cases(self):
        """Test boolean type handling edge cases."""
        group_key = np.array([0, 0, -1, 1, 1], dtype=np.int64)  # Include negative key
        bool_values = np.array([True, False, True, False, True], dtype=bool)
        ngroups = 2

        # cumsum: bool → int64
        result_sum = cumsum(group_key, bool_values, ngroups)
        expected_sum = np.array([1, 1, np.iinfo("int64").min, 0, 1])
        np.testing.assert_array_equal(result_sum, expected_sum)

        # cummin: bool → bool
        result_min = cummin(group_key, bool_values, ngroups)
        assert result_min.dtype == bool

        # cummax: bool → bool
        result_max = cummax(group_key, bool_values, ngroups)
        assert result_max.dtype == bool

    def test_type_preservation_with_nan_values(self):
        """Test type preservation when NaN values are present."""
        group_key = np.array([0, 0, 1, 1], dtype=np.int64)
        ngroups = 2

        # Test float32 with NaN
        float32_values = np.array([1.0, np.nan, 3.0, 4.0], dtype=np.float32)

        result_sum = cumsum(group_key, float32_values, ngroups)
        assert result_sum.dtype == np.float32

        result_min = cummin(group_key, float32_values, ngroups)
        assert result_min.dtype == np.float32

        result_max = cummax(group_key, float32_values, ngroups)
        assert result_max.dtype == np.float32

    def test_type_preservation_with_masks(self):
        """Test type preservation when masks are applied."""
        group_key = np.array([0, 0, 1, 1], dtype=np.int64)
        mask = np.array([True, False, True, True], dtype=bool)
        ngroups = 2

        # Test with int32 values and mask
        int32_values = np.array([1, 2, 3, 4], dtype=np.int32)

        result_sum = cumsum(group_key, int32_values, ngroups, mask=mask)
        assert result_sum.dtype == np.int64  # int32 → int64

        result_min = cummin(group_key, int32_values, ngroups, mask=mask)
        assert result_min.dtype == np.int32  # Preserve exact type

        result_max = cummax(group_key, int32_values, ngroups, mask=mask)
        assert result_max.dtype == np.int32  # Preserve exact type
