"""
Unit tests for array_split_for_lists function.

Tests the chunking functionality for both single arrays and lists of arrays,
with focus on PyArrow chunked array optimization scenarios.
"""

import pytest
import numpy as np
import pandas as pd
from pandas_plus.groupby.numba import _array_split_for_lists


class TestArraySplitForLists:
    """Test array_split_for_lists function."""

    def test_single_numpy_array(self):
        """Test that single numpy arrays delegate to np.array_split."""
        arr = np.array([1, 2, 3, 4, 5, 6])
        chunks = _array_split_for_lists(arr, 3)

        # Should return list of numpy arrays
        assert len(chunks) == 3
        assert all(isinstance(chunk, np.ndarray) for chunk in chunks)

        # Compare with direct np.array_split
        expected = np.array_split(arr, 3)
        for chunk, exp in zip(chunks, expected):
            np.testing.assert_array_equal(chunk, exp)

    def test_single_numpy_array_uneven_split(self):
        """Test numpy array split with uneven division."""
        arr = np.array([1, 2, 3, 4, 5])
        chunks = _array_split_for_lists(arr, 3)

        assert len(chunks) == 3
        # First two chunks should have 2 elements, last should have 1
        assert len(chunks[0]) == 2
        assert len(chunks[1]) == 2
        assert len(chunks[2]) == 1

        # Verify content
        np.testing.assert_array_equal(chunks[0], [1, 2])
        np.testing.assert_array_equal(chunks[1], [3, 4])
        np.testing.assert_array_equal(chunks[2], [5])

    def test_list_of_arrays_equal_chunks(self):
        """Test list of arrays with equal chunk sizes."""
        arrays = [np.array([1, 2, 3]), np.array([4, 5]), np.array([6, 7, 8, 9])]
        # Total length: 9, split into 3 chunks of ~3 elements each
        chunks = _array_split_for_lists(arrays, 3)

        assert len(chunks) == 3
        assert all(isinstance(chunk, list) for chunk in chunks)

        # Verify total lengths are approximately equal
        chunk_lengths = [sum(len(arr) for arr in chunk) for chunk in chunks]
        assert all(length == 3 for length in chunk_lengths)

        # Verify content by flattening and reconstructing
        flattened = []
        for chunk in chunks:
            for arr in chunk:
                flattened.extend(arr)

        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert flattened == expected

    def test_list_of_arrays_uneven_split(self):
        """Test list of arrays with uneven total length."""
        arrays = [np.array([1, 2, 3, 4, 5]), np.array([6, 7]), np.array([8, 9, 10])]
        # Total length: 10, split into 3 chunks
        chunks = _array_split_for_lists(arrays, 3)

        assert len(chunks) == 3

        # Calculate chunk sizes (should be ceil(10/3) = 4 max per chunk)
        chunk_lengths = [sum(len(arr) for arr in chunk) for chunk in chunks]

        # With chunk_len = ceil(10/3) = 4, we expect sizes around 4, 4, 2
        assert max(chunk_lengths) <= 4
        assert sum(chunk_lengths) == 10  # All elements preserved

        # Verify no elements lost or duplicated
        flattened = []
        for chunk in chunks:
            for arr in chunk:
                flattened.extend(arr)

        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        assert flattened == expected

    def test_arrays_crossing_chunk_boundaries(self):
        """Test that arrays can be split across chunk boundaries."""
        arrays = [np.array([1, 2, 3, 4, 5, 6, 7, 8])]  # Single large array
        # Split into 3 chunks
        chunks = _array_split_for_lists(arrays, 3)

        assert len(chunks) == 3

        # With chunk_len = ceil(8/3) = 3, first array should be split
        chunk_lengths = [sum(len(arr) for arr in chunk) for chunk in chunks]
        assert chunk_lengths == [3, 3, 2]  # 3, 3, 2 elements per chunk

        # First chunk should contain first 3 elements
        first_chunk_data = []
        for arr in chunks[0]:
            first_chunk_data.extend(arr)
        assert first_chunk_data == [1, 2, 3]

        # Verify all elements preserved
        flattened = []
        for chunk in chunks:
            for arr in chunk:
                flattened.extend(arr)
        assert flattened == [1, 2, 3, 4, 5, 6, 7, 8]

    def test_more_chunks_than_total_elements(self):
        """Test case where n_chunks > total elements."""
        arrays = [np.array([1]), np.array([2])]
        chunks = _array_split_for_lists(arrays, 5)

        # Should create chunks, some potentially empty
        assert len(chunks) >= 2  # At least as many as non-empty

        # Total elements should be preserved
        total_elements = sum(sum(len(arr) for arr in chunk) for chunk in chunks)
        assert total_elements == 2

        # Verify content
        flattened = []
        for chunk in chunks:
            for arr in chunk:
                flattened.extend(arr)
        assert sorted(flattened) == [1, 2]

    def test_single_chunk(self):
        """Test splitting into a single chunk."""
        arrays = [np.array([1, 2, 3]), np.array([4, 5])]
        chunks = _array_split_for_lists(arrays, 1)

        assert len(chunks) == 1

        # Single chunk should contain all arrays
        flattened = []
        for arr in chunks[0]:
            flattened.extend(arr)
        assert flattened == [1, 2, 3, 4, 5]

    def test_empty_arrays_in_list(self):
        """Test handling of empty arrays in the list."""
        arrays = [np.array([1, 2]), np.array([]), np.array([3, 4, 5])]  # Empty array
        chunks = _array_split_for_lists(arrays, 2)

        # Should handle empty arrays gracefully
        flattened = []
        for chunk in chunks:
            for arr in chunk:
                flattened.extend(arr)

        assert flattened == [1, 2, 3, 4, 5]  # Empty array ignored

    def test_different_array_types(self):
        """Test with different array-like types."""
        arrays = [
            np.array([1, 2, 3]),
            [4, 5, 6],  # Python list
            pd.Series([7, 8, 9]).values,  # Pandas array
        ]
        chunks = _array_split_for_lists(arrays, 2)

        # Should work with mixed array types
        flattened = []
        for chunk in chunks:
            for arr in chunk:
                flattened.extend(arr)

        assert flattened == [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def test_pyarrow_like_scenario(self):
        """Test scenario simulating PyArrow chunked arrays."""
        # Simulate what happens when PyArrow ChunkedArray is converted
        # Multiple small chunks that need to be redistributed for parallel processing
        pyarrow_chunks = [
            np.array([1, 2]),  # Small chunk 1
            np.array([3, 4, 5, 6]),  # Medium chunk 2
            np.array([7]),  # Small chunk 3
            np.array([8, 9, 10, 11, 12]),  # Large chunk 4
        ]
        # Total: 12 elements, split for parallel processing

        chunks = _array_split_for_lists(pyarrow_chunks, 3)

        assert len(chunks) == 3

        # Verify balanced distribution for parallel processing
        chunk_lengths = [sum(len(arr) for arr in chunk) for chunk in chunks]
        # With chunk_len = ceil(12/3) = 4, expect roughly equal sizes
        assert all(3 <= length <= 5 for length in chunk_lengths)
        assert sum(chunk_lengths) == 12

        # Verify all data preserved in order
        flattened = []
        for chunk in chunks:
            for arr in chunk:
                flattened.extend(arr)

        expected = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert flattened == expected

    def test_edge_case_zero_chunks(self):
        """Test edge case with zero chunks (should raise ValueError)."""
        arrays = [np.array([1, 2, 3])]

        with pytest.raises(ValueError, match="n_chunks must be a positive integer"):
            chunks = _array_split_for_lists(arrays, 0)

    def test_edge_case_negative_chunks(self):
        """Test edge case with negative chunks (should raise ValueError)."""
        arrays = [np.array([1, 2, 3])]

        # For numpy arrays, should catch before delegation
        numpy_arr = np.array([1, 2, 3])
        with pytest.raises(ValueError, match="n_chunks must be a positive integer"):
            chunks = _array_split_for_lists(numpy_arr, -1)

        # For list of arrays, should also raise ValueError
        with pytest.raises(ValueError, match="n_chunks must be a positive integer"):
            chunks = _array_split_for_lists(arrays, -1)

    def test_edge_case_empty_arrays(self):
        """Test edge case with completely empty arrays."""
        arrays = [np.array([]), np.array([]), np.array([])]
        chunks = _array_split_for_lists(arrays, 2)

        # Should return empty chunks
        assert len(chunks) == 2
        assert all(chunk == [] for chunk in chunks)


class TestArraySplitForListsPerformance:
    """Performance-focused tests for array_split_for_lists."""

    def test_large_array_performance(self):
        """Test performance with large arrays."""
        large_array = np.arange(10000)

        # Should complete quickly and delegate to numpy
        chunks = _array_split_for_lists(large_array, 4)

        assert len(chunks) == 4
        assert sum(len(chunk) for chunk in chunks) == 10000

    def test_many_small_arrays_performance(self):
        """Test performance with many small arrays."""
        many_arrays = [np.array([i, i + 1]) for i in range(0, 1000, 2)]

        # Should handle many small arrays efficiently
        chunks = _array_split_for_lists(many_arrays, 8)

        assert len(chunks) == 8

        # Verify total elements preserved
        total_elements = sum(sum(len(arr) for arr in chunk) for chunk in chunks)
        assert total_elements == 1000


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__])
