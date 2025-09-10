"""
Comprehensive tests for fancy indexers and slices as masks in GroupBy methods.

This module tests all combinations of:
- Mask types: boolean masks, fancy indexers (integer arrays), slices
- Group key types: chunked vs non-chunked
- Value types: chunked vs non-chunked, single arrays vs DataFrames
- Edge cases: out-of-bounds indexers, empty masks, etc.
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest

from pandas_plus.groupby.core import GroupBy

from .conftest import assert_pd_equal


class TestMaskIndexing:
    """Test GroupBy methods with various mask types."""

    @pytest.fixture
    def basic_data(self):
        """Create basic test data for mask testing."""
        np.random.seed(42)
        size = 1000
        n_groups = 20

        group_keys = np.random.randint(0, n_groups, size=size, dtype=np.int32)
        float_values = np.random.randn(size).astype(np.float64)
        int_values = np.random.randint(-50, 50, size=size, dtype=np.int64)

        return {
            "group_keys": group_keys,
            "float_values": float_values,
            "int_values": int_values,
            "size": size,
            "n_groups": n_groups,
        }

    @pytest.fixture
    def chunked_data(self):
        """Create chunked test data for mask testing."""
        np.random.seed(42)
        size = 2000
        n_groups = 6

        # Create data and convert to chunked arrays
        group_keys = np.random.randint(0, n_groups, size=size, dtype=np.int32)
        float_values = np.random.randn(size).astype(np.float64)
        int_values = np.random.randint(-50, 50, size=size, dtype=np.int64)

        # Convert to PyArrow chunked arrays
        chunk_size = 500
        group_chunks = [
            group_keys[i : i + chunk_size] for i in range(0, size, chunk_size)
        ]
        float_chunks = [
            float_values[i : i + chunk_size] for i in range(0, size, chunk_size)
        ]
        int_chunks = [
            int_values[i : i + chunk_size] for i in range(0, size, chunk_size)
        ]

        chunked_group_keys = pa.chunked_array(
            [pa.array(chunk) for chunk in group_chunks]
        )
        chunked_float_values = pa.chunked_array(
            [pa.array(chunk) for chunk in float_chunks]
        )
        chunked_int_values = pa.chunked_array([pa.array(chunk) for chunk in int_chunks])

        return {
            "group_keys": chunked_group_keys,
            "float_values": chunked_float_values,
            "int_values": chunked_int_values,
            "group_keys_np": group_keys,
            "float_values_np": float_values,
            "int_values_np": int_values,
            "size": size,
            "n_groups": n_groups,
        }

    # Slice mask tests
    @pytest.mark.parametrize(
        "slice_obj",
        [
            slice(None, 500),  # First half
            slice(250, 750),  # Middle section
            slice(500, None),  # Second half
            slice(None, None, 2),  # Every other element
            slice(100, 900, 3),  # Every third element in range
            slice(None, None, -1),  # Reverse order
            slice(800, 200, -2),  # Reverse with step
            slice(None),  # Trivial Slice
            slice(-1000),  # Last 1000
        ],
    )
    @pytest.mark.parametrize("method", ["mean", "max", "count"])
    def test_slice_masks_basic_data(self, basic_data, slice_obj, method):
        """Test slice masks with basic (non-chunked) data."""
        data = basic_data

        gb = GroupBy(data["group_keys"])
        result = getattr(gb, method)(data["float_values"], mask=slice_obj)

        sliced_keys = data["group_keys"][slice_obj]
        sliced_values = data["float_values"][slice_obj]
        gb_sliced = GroupBy(sliced_keys)
        expected = getattr(gb_sliced, method)(sliced_values)

        assert_pd_equal(result, expected, check_dtype=False)

    @pytest.mark.parametrize(
        "slice_obj",
        [
            slice(None),  # Trivial
            slice(None, 1000),  # First half
            slice(500, 1500),  # Middle section
            slice(1000, None),  # Second half
        ],
    )
    @pytest.mark.parametrize("method", ["sum", "mean", "max"])
    def test_slice_masks_chunked_data(self, chunked_data, slice_obj, method):
        """Test slice masks with chunked data."""
        data = chunked_data

        # Test with chunked arrays
        gb_chunked = GroupBy(data["group_keys"])

        # Compare with numpy arrays using same slice
        sliced_keys = data["group_keys_np"][slice_obj]
        sliced_values = data["float_values_np"][slice_obj]
        gb_regular = GroupBy(sliced_keys)
        assert_pd_equal(gb_chunked.size(mask=slice_obj), gb_regular.size())

        result = getattr(gb_chunked, method)(data["float_values"], mask=slice_obj)
        expected = getattr(gb_regular, method)(sliced_values)

        assert_pd_equal(result, expected, check_dtype=False)

    # Fancy indexer tests - currently not implemented for chunked group keys
    @pytest.mark.parametrize(
        "method", ["sum", "mean", "min", "max", "count", "first", "last"]
    )
    def test_fancy_indexer_masks_basic(self, basic_data, method):
        data = basic_data

        # Create fancy indexer
        indexer = np.array([0, 5, 10, 50, 100])

        gb = GroupBy(data["group_keys"])
        result = getattr(gb, method)(data["float_values"], mask=indexer)
        expected = GroupBy.agg(
            data["group_keys"][indexer], data["float_values"][indexer], method
        )
        assert_pd_equal(result, expected)

    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max", "count"])
    def test_fancy_indexer_masks_chunked_data(self, chunked_data, method):
        data = chunked_data

        # Create fancy indexer
        indexer = np.array([0, 10, 100, 500, 1000])
        gb = GroupBy(data["group_keys"])
        result = getattr(gb, method)(data["float_values"], mask=indexer)
        expected = GroupBy.agg(
            pd.Series(data["group_keys"])[indexer],
            pd.Series(data["float_values"])[indexer],
            method,
        )
        assert_pd_equal(result, expected)

    # Boolean mask tests (existing functionality verification)
    @pytest.mark.parametrize("method", ["sum", "mean", "min", "max", "count"])
    def test_boolean_masks_consistency(self, basic_data, method):
        """Test that boolean masks still work correctly."""
        data = basic_data

        # Create boolean masks
        masks = [
            data["float_values"] > 0,  # Positive values
            data["float_values"] < -0.5,  # Negative values
            np.abs(data["float_values"]) < 1.0,  # Values close to zero
            (data["group_keys"] % 2) == 0,  # Even group keys
        ]

        gb = GroupBy(data["group_keys"])

        for mask in masks:
            result = getattr(gb, method)(data["float_values"], mask=mask)

            # Compare with manual boolean indexing
            masked_keys = data["group_keys"][mask]
            masked_values = data["float_values"][mask]
            gb_masked = GroupBy(masked_keys)
            expected = getattr(gb_masked, method)(masked_values)

            assert_pd_equal(result, expected, check_dtype=False)

    # DataFrame input tests
    def test_masks_with_dataframe_input(self, basic_data):
        """Test masks with DataFrame inputs."""
        data = basic_data

        # Create DataFrame input
        df = pd.DataFrame(
            {
                "float_col": data["float_values"],
                "int_col": data["int_values"],
            }
        )

        gb = GroupBy(data["group_keys"])

        # Test with slice
        slice_mask = slice(100, 800, 2)
        result = gb.sum(df, mask=slice_mask)

        # Compare with manual slicing
        sliced_keys = data["group_keys"][slice_mask]
        sliced_df = df.iloc[slice_mask]
        gb_sliced = GroupBy(sliced_keys)
        expected = gb_sliced.sum(sliced_df)

        assert_pd_equal(result, expected, check_dtype=False)

        # Test with boolean mask
        bool_mask = data["float_values"] > 0
        result = gb.mean(df, mask=bool_mask)

        # Compare with manual masking
        masked_keys = data["group_keys"][bool_mask]
        masked_df = df[bool_mask]
        gb_masked = GroupBy(masked_keys)
        expected = gb_masked.mean(masked_df)

        assert_pd_equal(result, expected, check_dtype=False)

        fancy_mask = bool_mask.nonzero()[0]
        result = gb.mean(df, mask=fancy_mask)
        assert_pd_equal(result, expected, check_dtype=False)

    # Edge cases and error conditions
    def test_out_of_bounds_fancy_indexer_raises(self, basic_data):
        data = basic_data
        gb = GroupBy(data["group_keys"])

        # Test various out-of-bounds conditions - all should raise NotImplementedError for now
        out_of_bounds_indexers = [
            np.array([0, 5, data["size"]]),  # One element out of bounds
            np.array([data["size"] + 100]),  # Far out of bounds
            np.array([data["size"] * 2]),  # Way out of bounds
        ]

        for indexer in out_of_bounds_indexers:
            with pytest.raises(ValueError):
                gb.sum(data["float_values"], mask=indexer)

    def test_empty_masks(self, basic_data):
        """Test behavior with empty masks."""
        data = basic_data
        gb = GroupBy(data["group_keys"])

        empty_indexer = np.array([], dtype=np.int64)
        gb.sum(data["float_values"], mask=empty_indexer)

        # Empty slice
        empty_slice = slice(500, 500)  # Empty range
        result = gb.sum(data["float_values"], mask=empty_slice)
        assert len(result) == 0

        # Empty boolean mask
        empty_bool = np.zeros(data["size"], dtype=bool)
        result = gb.sum(data["float_values"], mask=empty_bool)
        assert len(result) == 0

    def test_single_element_masks(self, basic_data):
        """Test masks that select only one element."""
        data = basic_data
        gb = GroupBy(data["group_keys"])

        single_indexer = np.array([500])
        gb.sum(data["float_values"], mask=single_indexer)

        # Single element slice works
        single_slice = slice(500, 501)
        result = gb.sum(data["float_values"], mask=single_slice)
        # Should have at most one group
        assert len(result) <= 1
        if len(result) == 1:
            # Value should match the selected element
            expected_value = data["float_values"][500]
            assert result.iloc[0] == expected_value

    def test_fancy_indexer_not_implemented_scenarios(self, basic_data):
        """Test various fancy indexer scenarios that are not yet implemented."""
        data = basic_data
        gb = GroupBy(data["group_keys"])

        fancy_indexers = [
            np.array([100, 200, 100, 300, 200, 100]),  # Duplicates
            np.array([-1, -10, -100, -500]),  # Negative indices
            np.array([500]),  # Single element
            np.array([0, data["size"] - 1]),  # First and last
        ]

        for indexer in fancy_indexers:
            gb.sum(data["float_values"], mask=indexer)

    # Performance and memory tests
    def test_large_slice_performance(self, chunked_data):
        """Test performance with large slice masks."""
        data = chunked_data
        gb = GroupBy(data["group_keys"])

        # Large slice
        large_slice = slice(None, len(data) // 2)  # Every other element

        # This should complete without memory issues
        result = gb.sum(data["float_values"], mask=large_slice)

        # Basic validation
        assert len(result) <= data["n_groups"]
        assert not result.isna().all()

    def test_large_fancy_indexer_not_implemented(self, chunked_data):
        """Test that large fancy indexers raise NotImplementedError."""
        data = chunked_data
        gb = GroupBy(data["group_keys"])

        # Large random indexer
        large_indexer = np.random.choice(
            data["size"], data["size"] // 10, replace=False
        )

        gb.sum(data["float_values"], mask=large_indexer)

    # Mixed data type tests
    def test_masks_with_mixed_value_types(self, basic_data):
        """Test masks work with different value types."""
        data = basic_data
        gb = GroupBy(data["group_keys"])

        # Create different value types
        bool_values = data["float_values"] > 0
        datetime_values = pd.to_datetime("2023-01-01") + pd.to_timedelta(
            np.arange(data["size"]), unit="D"
        )

        slice_mask = slice(100, 800, 2)

        # Test with boolean values
        result_bool = gb.sum(bool_values.astype(int), mask=slice_mask)
        assert result_bool.dtype in [np.int64, np.float64]

        # Test with datetime values (should work for count/size)
        result_count = gb.count(datetime_values, mask=slice_mask)
        assert result_count.dtype == np.int64

    # Transform operations with masks
    def test_masks_with_transform_operations(self, basic_data):
        """Test that masks work correctly with transform=True."""
        data = basic_data
        gb = GroupBy(data["group_keys"])

        slice_mask = slice(200, 700)
        # Transform operations should return same length as mask
        result = gb.mean(data["float_values"], mask=slice_mask, transform=True)

        bool_mask = np.full(len(gb), False)
        bool_mask[slice_mask] = True
        expected = gb.mean(data["float_values"], mask=bool_mask, transform=True)

        assert_pd_equal(result, expected)

    # Consistency tests between chunked and non-chunked
    def test_mask_consistency_chunked_vs_regular(self, chunked_data):
        """Test that masks produce identical results for chunked vs regular data."""
        data = chunked_data

        # Test various mask types (excluding fancy indexer which is not implemented)
        masks = [
            slice(100, 1500),  # Slice
            data["float_values_np"] > 0,  # Boolean mask
            slice(None, 1000),  # Simple slice
            data["float_values_np"] < -0.5,  # Another boolean mask
        ]

        methods = ["sum", "mean", "min", "max", "count"]

        for mask in masks:
            for method in methods:
                # Chunked version
                gb_chunked = GroupBy(data["group_keys"])
                result_chunked = getattr(gb_chunked, method)(
                    data["float_values"], mask=mask
                )

                # Regular version
                gb_regular = GroupBy(data["group_keys_np"])
                result_regular = getattr(gb_regular, method)(
                    data["float_values_np"], mask=mask
                )

                assert_pd_equal(result_chunked, result_regular, check_dtype=False)

    # Margin operations with masks
    @pytest.mark.parametrize("margins", [True, False])
    def test_masks_with_margins(self, basic_data, margins):
        """Test masks work correctly with margin operations."""
        data = basic_data
        gb = GroupBy(data["group_keys"])

        slice_mask = slice(None, 500, 2)

        # Test with margins
        result = gb.sum(data["float_values"], mask=slice_mask, margins=margins)

        if margins:
            # Should have additional margin rows
            assert "All" in result.index or len(result) > gb.ngroups
        else:
            # Should not have margin rows
            assert len(result) <= gb.ngroups

    # Observed-only flag with masks
    @pytest.mark.parametrize("observed_only", [True, False])
    def test_masks_with_observed_only_flag(self, basic_data, observed_only):
        """Test masks work correctly with observed_only flag."""
        data = basic_data

        # Create categorical group keys to test observed_only behavior
        cat_groups = pd.Categorical(
            data["group_keys"], categories=range(data["n_groups"] + 10)
        )
        gb = GroupBy(cat_groups)

        slice_mask = slice(None, 500)
        result = gb.sum(
            data["float_values"], mask=slice_mask, observed_only=observed_only
        )

        if observed_only:
            # Should only include groups that appear in the data
            assert all(result.index.isin(data["group_keys"][slice_mask]))
        # Note: observed_only=False behavior depends on implementation details
