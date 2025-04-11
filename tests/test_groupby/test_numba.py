import numba as nb
import numpy as np
import pandas as pd
import pytest
from inspect import signature

from pandas_plus.groupby.numba import (NumbaGroupByMethods, _chunk_groupby_args, _group_by_iterator,
                                       group_nearby_members, group_count, group_mean,
                                       group_max, group_min, group_sum)
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


class TestChunkGroupbyArgs:
    
    def test_basic_functionality(self):
        """Test basic functionality with simple inputs."""
        # Setup
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        target = np.zeros(3)
        mask = np.ones(5, dtype=bool)
        reduce_func = NumbaReductionOps.sum
        n_chunks = 2
        
        # Call the function
        chunked_args = _chunk_groupby_args(
            n_chunks=n_chunks,
            group_key=group_key,
            values=values,
            target=target,
            mask=mask,
            reduce_func=reduce_func,
            must_see=True
        )
        
        # Verify results
        assert len(chunked_args) == n_chunks
        
        # Each chunked argument should be bound arguments for _group_by_iterator
        for args in chunked_args:
            assert args.signature == signature(_group_by_iterator)
            
        # Check first chunk
        first_chunk = chunked_args[0]
        assert len(first_chunk.args[0]) <= 3  # group_key length should be around half
        assert len(first_chunk.args[1]) <= 3  # values length should be around half
        assert len(first_chunk.args[3]) <= 3  # mask length should be around half
        assert first_chunk.args[2].shape == target.shape  # target shape should be unchanged
        
        # Test with actual _group_by_iterator
        results = [_group_by_iterator(*args.args) for args in chunked_args]
        assert all(isinstance(r, np.ndarray) for r in results)
        assert all(r.shape == target.shape for r in results)
    
    def test_with_empty_mask(self):
        """Test with mask=None."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        target = np.zeros(3)
        reduce_func = NumbaReductionOps.sum
        n_chunks = 2
        
        chunked_args = _chunk_groupby_args(
            n_chunks=n_chunks,
            group_key=group_key,
            values=values,
            target=target,
            mask=np.array([], dtype=bool),
            reduce_func=reduce_func,
            must_see=True
        )
        
        assert len(chunked_args) == n_chunks
        
        # Check that mask was properly prepared
        for args in chunked_args:
            mask = args.arguments['mask']
            assert isinstance(mask, np.ndarray)  # mask is prepared
            assert mask.dtype == bool  # mask is boolean
            assert len(mask) == 0  # mask length matches chunk length
    
    def test_different_chunk_numbers(self):
        """Test with different numbers of chunks."""
        group_key = np.array([0, 1, 0, 2, 1, 3, 4, 5, 6, 7], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], dtype=np.float64)
        target = np.zeros(8)
        mask = np.ones(10, dtype=bool)
        reduce_func = NumbaReductionOps.sum
        
        for n_chunks in [1, 2, 3, 5, 10]:
            chunked_args = _chunk_groupby_args(
                n_chunks=n_chunks,
                group_key=group_key,
                values=values,
                target=target,
                mask=mask,
                reduce_func=reduce_func,
                must_see=True
            )
            
            assert len(chunked_args) == n_chunks
            
            # Check that the total length of all chunks equals the original length
            total_group_key_length = sum(len(args.args[0]) for args in chunked_args)
            assert total_group_key_length == len(group_key)
            
            total_values_length = sum(len(args.args[1]) for args in chunked_args)
            assert total_values_length == len(values)
    
    def test_target_is_copied(self):
        """Test that target arrays are copied, not shared between chunks."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        target = np.zeros(3)
        mask = np.ones(5, dtype=bool)
        reduce_func = NumbaReductionOps.sum
        n_chunks = 2
        
        chunked_args = _chunk_groupby_args(
            n_chunks=n_chunks,
            group_key=group_key,
            values=values,
            target=target,
            mask=mask,
            reduce_func=reduce_func,
            must_see=True
        )
        
        # Modify target in first chunk and verify it doesn't affect second chunk
        chunked_args[0].args[2][0] = 999.0
        assert chunked_args[1].args[2][0] == 0.0
    
    def test_with_boolean_values(self):
        """Test with boolean values."""
        group_key = np.array([0, 1, 0, 2, 1], dtype=np.int64)
        values = np.array([True, False, True, False, True], dtype=bool)
        target = np.zeros(3, dtype=bool)
        mask = np.ones(5, dtype=bool)
        reduce_func = NumbaReductionOps.sum
        n_chunks = 2
        
        chunked_args = _chunk_groupby_args(
            n_chunks=n_chunks,
            group_key=group_key,
            values=values,
            target=target,
            mask=mask,
            reduce_func=reduce_func,
            must_see=True
        )
        
        # Check values were preserved
        all_values = np.concatenate([args.args[1] for args in chunked_args])
        np.testing.assert_array_equal(np.sort(all_values), np.sort(values))
        
        # Test with actual _group_by_iterator
        results = [_group_by_iterator(*args.args) for args in chunked_args]
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

    @pytest.mark.parametrize("func", [group_count, group_sum, group_mean, group_min, group_max])
    def test_multi_threaded(self, func):
        N = 2_000_000
        group_key = np.arange(N) % 5
        values = np.random.rand(N)
        result = func(group_key, values, ngroups=5, n_threads=4)
        func_name = func.__name__.split('_')[1]
        expected = pd.Series(values).groupby(group_key).agg(func_name).values
        np.testing.assert_array_almost_equal(result, expected)


class TestGroupNearbyMembers:
    def test_basic_functionality(self):
        """Test basic functionality with simple inputs."""
        # Setup - Group 0 has increasing values, Group 1 has some gaps
        group_key = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0, 10.0, 11.0, 20.0, 21.0], dtype=np.float64)
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
