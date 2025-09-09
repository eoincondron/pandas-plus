"""
Comprehensive tests for GroupBy methods with chunked group keys.

This module tests all GroupBy methods when the group keys are chunked,
either because they are large arrays (>=1M elements) or because they are 
already backed by ChunkedArrays (e.g., from parquet files).
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import polars as pl
import pytest

from pandas_plus.groupby.core import GroupBy

from .conftest import assert_pd_equal


class TestChunkedGroupKeys:
    """Test all GroupBy methods with chunked group keys."""

    @pytest.fixture
    def large_data(self):
        """Create large dataset that triggers chunked factorization."""
        np.random.seed(42)
        size = 2_000_000  # Large enough to trigger chunking

        # Create group keys with reasonable number of groups
        n_groups = 1000
        group_keys = np.random.randint(0, n_groups, size=size, dtype=np.int32)

        # Create various value types
        float_values = np.random.randn(size).astype(np.float64)
        int_values = np.random.randint(-100, 100, size=size, dtype=np.int64)
        bool_values = np.random.choice([True, False], size=size)
        gb_chunked = GroupBy(group_keys, factorize_large_inputs_in_chunks=True)

        return {
            "group_keys": group_keys,
            "float_values": float_values,
            "int_values": int_values,
            "bool_values": bool_values,
            "size": size,
            "n_groups": n_groups,
            "gb_chunked": gb_chunked,
        }

    @pytest.fixture
    def pyarrow_chunked_data(self):
        """Create data backed by PyArrow ChunkedArrays."""
        size = 10_000
        n_groups = 50
        np.random.seed(42)

        # Create data and then convert to chunked arrays
        group_keys = np.random.randint(0, n_groups, size=size, dtype=np.int32)
        float_values = np.random.randn(size).astype(np.float64)
        int_values = np.random.randint(-100, 100, size=size, dtype=np.int64)

        # Convert to PyArrow chunked arrays with multiple chunks
        chunk_size = 2000
        group_chunks = [group_keys[i:i+chunk_size] for i in range(0, size, chunk_size)]
        float_chunks = [float_values[i:i+chunk_size] for i in range(0, size, chunk_size)]
        int_chunks = [int_values[i:i+chunk_size] for i in range(0, size, chunk_size)]

        chunked_group_keys = pa.chunked_array([pa.array(chunk) for chunk in group_chunks])
        chunked_float_values = pa.chunked_array([pa.array(chunk) for chunk in float_chunks])
        chunked_int_values = pa.chunked_array([pa.array(chunk) for chunk in int_chunks])

        gb_chunked = GroupBy(group_keys, factorize_large_inputs_in_chunks=True)

        return {
            'group_keys': chunked_group_keys,
            'float_values': chunked_float_values,
            'int_values': chunked_int_values,
            'group_keys_np': group_keys,
            'float_values_np': float_values,
            'int_values_np': int_values,
            'size': size,
            'n_groups': n_groups, 
            'gb_chunked': gb_chunked,
        }

    @pytest.mark.parametrize("method", [
        "sum", "mean", "min", "max", "var", "std", "first", "last", "count"
    ])
    def test_basic_aggregation_methods_large_chunked(self, large_data, method):
        """Test basic aggregation methods with large chunked group keys."""
        data = large_data

        # Test with float values
        gb_chunked = data['gb_chunked']
        result = getattr(gb_chunked, method)(data['float_values'])

        gb_regular = GroupBy(data["group_keys"], factorize_large_inputs_in_chunks=False)
        expected = getattr(gb_regular, method)(data["float_values"])

        assert_pd_equal(result, expected)

        # Verify basic properties
        assert len(result) <= data['n_groups']  # Should have at most n_groups
        assert not result.isna().all()  # Should have some non-null results

    @pytest.mark.parametrize("method", [
        "sum", "mean", "min", "max", "first", "last", "count"
    ])
    def test_basic_aggregation_methods_pyarrow_chunked(self, pyarrow_chunked_data, method):
        """Test basic aggregation methods with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        assert gb_chunked.key_is_chunked  # Verify it's actually chunked
        result = getattr(gb_chunked, method)(data['float_values'])

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        assert not gb_regular.key_is_chunked  # Verify it's not chunked
        expected = getattr(gb_regular, method)(data['float_values_np'])

        assert_pd_equal(result, expected, check_dtype=False)

    def test_size_method_large_chunked(self, large_data):
        """Test size method specifically with large chunked group keys."""
        data = large_data

        gb = GroupBy(data['group_keys'], factorize_large_inputs_in_chunks=True)
        result = gb.size()

        # Verify basic properties
        assert len(result) <= data['n_groups']
        assert result.sum() == data['size']  # Total size should match input
        assert (result > 0).all()  # All groups should have positive size

    def test_size_method_pyarrow_chunked(self, pyarrow_chunked_data):
        """Test size method with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = getattr(gb_chunked, 'size')()

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = getattr(gb_regular, 'size')()

        assert_pd_equal(result, expected)

    @pytest.mark.parametrize("use_mask", [False, True])
    def test_with_mask_large_chunked(self, large_data, use_mask):
        """Test aggregation methods with masks on large chunked data."""
        data = large_data

        gb = GroupBy(data['group_keys'], factorize_large_inputs_in_chunks=True)

        if use_mask:
            # Create a mask that filters out ~half the data
            mask = data['float_values'] > 0
        else:
            mask = None

        result = gb.sum(data['float_values'], mask=mask)

        # Verify basic properties
        assert len(result) <= data['n_groups']
        assert not result.isna().all()

    @pytest.mark.parametrize("use_mask", [False, True])
    def test_with_mask_pyarrow_chunked(self, pyarrow_chunked_data, use_mask):
        """Test aggregation methods with masks on PyArrow chunked data."""
        data = pyarrow_chunked_data

        if use_mask:
            # Create a mask that filters out ~half the data
            mask = data['float_values_np'] > 0
        else:
            mask = None

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = gb_chunked.sum(data['float_values'], mask=mask)

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = gb_regular.sum(data['float_values_np'], mask=mask)

        assert_pd_equal(result, expected, check_dtype=False)

    def test_agg_method_large_chunked(self, large_data):
        """Test agg method with large chunked group keys."""
        data = large_data

        gb = GroupBy(data['group_keys'], factorize_large_inputs_in_chunks=True)

        # Test single function
        result_sum = gb.agg(data['float_values'], agg_func='sum')
        result_direct = gb.sum(data['float_values'])
        assert_pd_equal(result_sum, result_direct, rtol=1e-10)

        # Test multiple functions on single array
        result_multi = gb.agg(data['float_values'], agg_func=['sum', 'mean'])
        assert len(result_multi.columns) == 2
        assert 'sum' in result_multi.columns
        assert 'mean' in result_multi.columns

    def test_agg_method_pyarrow_chunked(self, pyarrow_chunked_data):
        """Test agg method with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = gb_chunked.agg(data['float_values'], agg_func='sum')

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = gb_regular.agg(data['float_values_np'], agg_func='sum')

        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("window", [3, 5, 10])
    def test_rolling_methods_large_chunked(self, large_data, window):
        """Test rolling methods with large chunked group keys."""
        data = large_data

        # Use a smaller sample for rolling methods (they're memory intensive)
        sample_size = 100_000
        sample_idx = np.random.choice(data['size'], sample_size, replace=False)
        sample_keys = data['group_keys'][sample_idx]
        sample_values = data['float_values'][sample_idx]

        gb = GroupBy(sample_keys, factorize_large_inputs_in_chunks=True)

        # Test rolling_sum
        result = gb.rolling_sum(sample_values, window=window)
        assert len(result) == sample_size
        assert not result.isna().all()

    @pytest.mark.parametrize("method", ["rolling_sum", "rolling_mean", "rolling_min", "rolling_max"])
    def test_rolling_methods_pyarrow_chunked(self, pyarrow_chunked_data, method):
        """Test rolling methods with PyArrow chunked arrays."""
        data = pyarrow_chunked_data
        window = 3

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = getattr(gb_chunked, method)(data['float_values'], window=window)

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = getattr(gb_regular, method)(data['float_values_np'], window=window)

        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("method", ["cumsum", "cummin", "cummax"])
    def test_cumulative_methods_large_chunked(self, large_data, method):
        """Test cumulative methods with large chunked group keys."""
        data = large_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data["group_keys"])
        result = getattr(gb_chunked, method)(data["float_values"])

        # Compare with numpy arrays
        gb_regular = GroupBy(data["group_keys"], factorize_large_inputs_in_chunks=False)
        expected = getattr(gb_regular, method)(data["float_values"])
        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("method", ["cumsum", "cummin", "cummax"])
    def test_cumulative_methods_pyarrow_chunked(self, pyarrow_chunked_data, method):
        """Test cumulative methods with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = getattr(gb_chunked, method)(data['float_values'])

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = getattr(gb_regular, method)(data['float_values_np'])

        assert_pd_equal(result, expected, check_dtype=False)

    def test_cumcount_large_chunked(self, large_data):
        """Test cumcount method with large chunked group keys."""
        data = large_data
        gb_chunked = data['gb_chunked']

        # Compare with numpy arrays
        expected = GroupBy(data["group_keys"], factorize_large_inputs_in_chunks=False).cumcount()
        result = gb_chunked.cumcount()
        assert_pd_equal(result, expected, check_dtype=False)
        # Compare with numpy arrays
        expected = GroupBy.cumcount(data["group_keys"])

        assert (result >= 0).all()  # Should be non-negative
        assert result.dtype == np.dtype('int64')  # Should be integer type

    def test_cumcount_pyarrow_chunked(self, pyarrow_chunked_data):
        """Test cumcount method with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = gb_chunked.cumcount()

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = gb_regular.cumcount()

        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_head_tail_methods_large_chunked(self, large_data, n):
        """Test head and tail methods with large chunked group keys."""
        data = large_data

        # Use smaller sample
        sample_size = 50_000
        sample_idx = np.random.choice(data['size'], sample_size, replace=False)
        sample_keys = data['group_keys'][sample_idx]
        sample_values = data['float_values'][sample_idx]

        gb = GroupBy(sample_keys, factorize_large_inputs_in_chunks=True)

        # Test head
        result_head = gb.head(sample_values, n=n, keep_input_index=True)
        assert len(result_head) <= sample_size

        # Test tail
        result_tail = gb.tail(sample_values, n=n, keep_input_index=True)
        assert len(result_tail) <= sample_size

    @pytest.mark.parametrize("method", ["head", "tail"])
    @pytest.mark.parametrize("n", [1, 2, 5])
    def test_selection_methods_pyarrow_chunked(self, pyarrow_chunked_data, method, n):
        """Test head, tail, and nth methods with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = getattr(gb_chunked, method)(data['float_values'], n=n, keep_input_index=True)

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = getattr(gb_regular, method)(data['float_values_np'], n=n, keep_input_index=True)

        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize("n", [0, 1, -1])
    def test_nth_method_pyarrow_chunked(self, pyarrow_chunked_data, n):
        """Test nth method with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = gb_chunked.nth(data['float_values'], n=n, keep_input_index=True)

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = gb_regular.nth(data['float_values_np'], n=n, keep_input_index=True)

        assert_pd_equal(result, expected, check_dtype=False)

    def test_shift_diff_methods_pyarrow_chunked(self, pyarrow_chunked_data):
        """Test shift and diff methods with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test shift
        gb_chunked = GroupBy(data['group_keys'])
        result_shift = gb_chunked.shift(data['float_values'], window=1)

        gb_regular = GroupBy(data['group_keys_np'])
        expected_shift = gb_regular.shift(data['float_values_np'], window=1)

        assert_pd_equal(result_shift, expected_shift, check_dtype=False)

        # Test diff
        result_diff = gb_chunked.diff(data['float_values'], window=1)
        expected_diff = gb_regular.diff(data['float_values_np'], window=1)

        assert_pd_equal(result_diff, expected_diff, check_dtype=False)

    def test_multiple_columns_large_chunked(self, large_data):
        """Test multiple column operations with large chunked group keys."""
        data = large_data

        gb = GroupBy(data['group_keys'], factorize_large_inputs_in_chunks=True)

        # Test with multiple columns as list
        values_list = [data['float_values'], data['int_values']]
        result = gb.sum(values_list)

        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == 2
        assert len(result) <= data['n_groups']

    def test_multiple_columns_pyarrow_chunked(self, pyarrow_chunked_data):
        """Test multiple column operations with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        values_list = [data['float_values'], data['int_values']]
        result = gb_chunked.sum(values_list)

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        values_list_np = [data['float_values_np'], data['int_values_np']]
        expected = gb_regular.sum(values_list_np)

        assert_pd_equal(result, expected, check_dtype=False)

    def test_transform_flag_large_chunked(self, large_data):
        """Test transform=True flag with large chunked group keys."""
        data = large_data

        # Use smaller sample for transform (memory intensive)
        sample_size = 10_000
        sample_idx = np.random.choice(data['size'], sample_size, replace=False)
        sample_keys = data['group_keys'][sample_idx]
        sample_values = data['float_values'][sample_idx]

        gb = GroupBy(sample_keys, factorize_large_inputs_in_chunks=True)
        result = gb.mean(sample_values, transform=True)

        assert len(result) == sample_size
        assert not result.isna().all()

    def test_transform_flag_pyarrow_chunked(self, pyarrow_chunked_data):
        """Test transform=True flag with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = gb_chunked.mean(data['float_values'], transform=True)

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = gb_regular.mean(data['float_values_np'], transform=True)

        assert_pd_equal(result, expected, check_dtype=False)

    def test_median_method_pyarrow_chunked(self, pyarrow_chunked_data):
        """Test median method with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = gb_chunked.median(data['float_values'])

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = gb_regular.median(data['float_values_np'])

        assert_pd_equal(result, expected, check_dtype=False)

    def test_var_std_methods_pyarrow_chunked(self, pyarrow_chunked_data):
        """Test var and std methods with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        for method in ['var', 'std']:
            # Test with chunked arrays
            gb_chunked = GroupBy(data['group_keys'])
            result = getattr(gb_chunked, method)(data['float_values'])

            # Compare with numpy arrays
            gb_regular = GroupBy(data['group_keys_np'])
            expected = getattr(gb_regular, method)(data['float_values_np'])

            assert_pd_equal(result, expected, check_dtype=False)

    def test_chunk_unification_triggered(self, pyarrow_chunked_data):
        """Test that chunk unification is triggered for methods that need it."""
        data = pyarrow_chunked_data

        gb = GroupBy(data['group_keys'])
        assert gb.key_is_chunked

        # Methods like median should trigger unification
        result = gb.median(data['float_values'])

        # After median call, the group key should be unified
        # (median calls _unify_group_key_chunks internally)
        assert isinstance(result, pd.Series)
        assert len(result) <= data['n_groups']

    def test_error_conditions_chunked(self, pyarrow_chunked_data):
        """Test error conditions with chunked group keys."""
        data = pyarrow_chunked_data

        gb = GroupBy(data['group_keys'])

        # Test length mismatch
        short_values = data['float_values_np'][:100]
        with pytest.raises(ValueError, match="Length of the input values"):
            gb.sum(short_values)

    @pytest.mark.parametrize("observed_only", [True, False])
    def test_observed_only_flag_pyarrow_chunked(self, pyarrow_chunked_data, observed_only):
        """Test observed_only flag with PyArrow chunked arrays."""
        data = pyarrow_chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data['group_keys'])
        result = gb_chunked.sum(data['float_values'], observed_only=observed_only)

        # Compare with numpy arrays
        gb_regular = GroupBy(data['group_keys_np'])
        expected = gb_regular.sum(data['float_values_np'], observed_only=observed_only)

        assert_pd_equal(result, expected, check_dtype=False)

    def test_memory_efficiency_large_chunked(self, large_data):
        """Test that chunked operations don't use excessive memory."""
        data = large_data

        gb = GroupBy(data['group_keys'], factorize_large_inputs_in_chunks=True)

        # This should not cause memory issues
        result = gb.sum(data['float_values'])

        # Verify we get a reasonable result
        assert len(result) <= data['n_groups']
        assert not result.isna().all()

    def test_group_key_properties_chunked(self, pyarrow_chunked_data):
        """Test GroupBy properties work correctly with chunked keys."""
        data = pyarrow_chunked_data

        gb = GroupBy(data['group_keys'])

        # Test basic properties
        assert gb.key_is_chunked
        assert gb.ngroups <= data['n_groups']
        assert len(gb) == data['size']

        # Test that result_index is accessible
        assert isinstance(gb.result_index, pd.Index)
        assert len(gb.result_index) == gb.ngroups

    def test_consistency_between_chunked_and_regular(self, pyarrow_chunked_data):
        """Test that chunked and regular GroupBy produce identical results."""
        data = pyarrow_chunked_data

        # Create both chunked and regular GroupBy objects
        gb_chunked = GroupBy(data['group_keys'])  # ChunkedArray input
        gb_regular = GroupBy(data['group_keys_np'])  # numpy array input

        assert gb_chunked.key_is_chunked
        assert not gb_regular.key_is_chunked

        # Test multiple methods for consistency (excluding first/last which need special handling)
        methods_to_test = ['sum', 'mean', 'min', 'max', 'count', 'size']

        for method in methods_to_test:
            if method == 'size':
                result_chunked = getattr(gb_chunked, method)()
                result_regular = getattr(gb_regular, method)()
            else:
                result_chunked = getattr(gb_chunked, method)(data['float_values'])
                result_regular = getattr(gb_regular, method)(data['float_values_np'])

            try:
                assert_pd_equal(result_chunked, result_regular, check_dtype=False)
            except AssertionError as e:
                raise AssertionError(f"Method {method} produced different results: {e}")
