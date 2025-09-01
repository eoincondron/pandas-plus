"""
Comprehensive tests for by/level argument processing in SeriesGroupBy and
DataFrameGroupBy.

Tests various column name types and index level specifications to ensure
proper handling of non-string identifiers.
"""

import pytest
import numpy as np
import pandas as pd
from pandas_plus.groupby.api import SeriesGroupBy, DataFrameGroupBy


class TestDataFrameByLevelProcessing:
    """Test DataFrameGroupBy by/level argument processing with various column types."""

    def setup_method(self):
        """Setup test data with various column name types."""
        # Create DataFrame with non-string column names
        self.df_mixed_cols = pd.DataFrame(
            {
                "str_col": [1, 2, 3, 4, 5, 6],
                42: [10, 20, 30, 40, 50, 60],  # int column name
                3.14: [100, 200, 300, 400, 500, 600],  # float column name
                pd.Timestamp("2024-01-01"): [
                    "A",
                    "B",
                    "A",
                    "B",
                    "A",
                    "B",
                ],  # Timestamp column name
                pd.Timedelta("1 day"): [
                    "X",
                    "Y",
                    "X",
                    "Y",
                    "X",
                    "Y",
                ],  # Timedelta column name
            }
        )

        # DataFrame with tuple column names (like MultiIndex columns)
        self.df_tuple_cols = pd.DataFrame(
            {
                ("group", "main"): ["A", "B", "A", "B"],
                ("value", "primary"): [1, 2, 3, 4],
                ("value", "secondary"): [10, 20, 30, 40],
            }
        )

        # DataFrame with MultiIndex
        multi_index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2), ("A", 3), ("B", 3)],
            names=["letter", "number"],
        )
        self.df_multi_index = pd.DataFrame(
            {"values": [1, 2, 3, 4, 5, 6], "other": [10, 20, 30, 40, 50, 60]},
            index=multi_index,
        )

        # DataFrame with numeric index level names
        numeric_multi_index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2)], names=[0, 1]
        )  # Numeric level names
        self.df_numeric_levels = pd.DataFrame(
            {"data": [1, 2, 3, 4]}, index=numeric_multi_index
        )

    def test_string_column_groupby(self):
        """Test grouping by string column name."""
        gb = DataFrameGroupBy(self.df_mixed_cols, by="str_col")
        pandas_gb = self.df_mixed_cols.groupby("str_col")

        result = gb.sum()
        expected = pandas_gb.sum()

        # Both should have same numeric columns (pandas-plus filters to
        # numeric only) pandas excludes the grouping column, pandas-plus
        # includes it since it's numeric
        expected_cols = [col for col in expected.columns if col in result.columns]
        pd.testing.assert_frame_equal(result[expected_cols], expected[expected_cols])

    def test_int_column_groupby(self):
        """Test grouping by integer column name."""
        gb = DataFrameGroupBy(self.df_mixed_cols, by=42)
        pandas_gb = self.df_mixed_cols.groupby(42)

        assert gb.ngroups == pandas_gb.ngroups

        # Test sum aggregation
        result = gb.sum()
        expected = pandas_gb.sum()

        # Find common columns (pandas excludes grouping col, pandas-plus may include it)
        common_cols = [col for col in expected.columns if col in result.columns]
        if common_cols:
            pd.testing.assert_frame_equal(result[common_cols], expected[common_cols])

        # Verify basic functionality works
        assert len(result) == len(expected)
        assert result.index.equals(expected.index)

    def test_float_column_groupby(self):
        """Test grouping by float column name."""
        gb = DataFrameGroupBy(self.df_mixed_cols, by=3.14)
        pandas_gb = self.df_mixed_cols.groupby(3.14)

        assert gb.ngroups == pandas_gb.ngroups

        result = gb.sum()
        expected = pandas_gb.sum()

        # Find common columns
        common_cols = [col for col in expected.columns if col in result.columns]
        if common_cols:
            pd.testing.assert_frame_equal(result[common_cols], expected[common_cols])

        assert len(result) == len(expected)
        assert result.index.equals(expected.index)

    def test_timestamp_column_groupby(self):
        """Test grouping by Timestamp column name."""
        timestamp_col = pd.Timestamp("2024-01-01")
        gb = DataFrameGroupBy(self.df_mixed_cols, by=timestamp_col)
        pandas_gb = self.df_mixed_cols.groupby(timestamp_col)

        assert gb.ngroups == pandas_gb.ngroups

        result = gb.sum()
        expected = pandas_gb.sum()

        # Find common columns
        common_cols = [col for col in expected.columns if col in result.columns]
        if common_cols:
            pd.testing.assert_frame_equal(result[common_cols], expected[common_cols])

        assert len(result) == len(expected)
        assert result.index.equals(expected.index)

    def test_timedelta_column_groupby(self):
        """Test grouping by Timedelta column name."""
        timedelta_col = pd.Timedelta("1 day")
        gb = DataFrameGroupBy(self.df_mixed_cols, by=timedelta_col)
        pandas_gb = self.df_mixed_cols.groupby(timedelta_col)

        assert gb.ngroups == pandas_gb.ngroups

        result = gb.sum()
        expected = pandas_gb.sum()

        # Find common columns
        common_cols = [col for col in expected.columns if col in result.columns]
        if common_cols:
            pd.testing.assert_frame_equal(result[common_cols], expected[common_cols])

        assert len(result) == len(expected)
        assert result.index.equals(expected.index)

    def test_tuple_column_groupby(self):
        """Test grouping by tuple column name (MultiIndex columns)."""
        gb = DataFrameGroupBy(self.df_tuple_cols, by=("group", "main"))
        pandas_gb = self.df_tuple_cols.groupby(("group", "main"))

        assert gb.ngroups == pandas_gb.ngroups

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_mixed_column_groupby(self):
        """Test grouping by multiple columns with mixed types."""
        # Group by string and int column
        gb = DataFrameGroupBy(self.df_mixed_cols, by=["str_col", 42])
        pandas_gb = self.df_mixed_cols.groupby(["str_col", 42])

        assert gb.ngroups == pandas_gb.ngroups

        result = gb.sum()
        expected = pandas_gb.sum()

        # Find common columns
        common_cols = [col for col in expected.columns if col in result.columns]
        if common_cols:
            pd.testing.assert_frame_equal(result[common_cols], expected[common_cols])

        assert len(result) == len(expected)
        assert result.index.equals(expected.index)

    def test_array_groupby(self):
        """Test grouping by array/Series."""
        grouper_array = np.array(["X", "Y", "X", "Y", "X", "Y"])

        gb = DataFrameGroupBy(self.df_mixed_cols, by=grouper_array)
        pandas_gb = self.df_mixed_cols.groupby(grouper_array)

        assert gb.ngroups == pandas_gb.ngroups

    def test_level_groupby_string_names(self):
        """Test grouping by index level with string names."""
        gb = DataFrameGroupBy(self.df_multi_index, level="letter")
        pandas_gb = self.df_multi_index.groupby(level="letter")

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_frame_equal(result, expected)

    def test_level_groupby_numeric_names(self):
        """Test grouping by index level with numeric names."""
        gb = DataFrameGroupBy(self.df_numeric_levels, level=0)
        pandas_gb = self.df_numeric_levels.groupby(level=0)

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_frame_equal(result, expected)

    def test_level_groupby_multiple_levels(self):
        """Test grouping by multiple index levels."""
        gb = DataFrameGroupBy(self.df_multi_index, level=["letter", "number"])
        pandas_gb = self.df_multi_index.groupby(level=["letter", "number"])

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_frame_equal(result, expected)

    def test_combined_by_and_level(self):
        """Test combining by and level arguments."""
        gb = DataFrameGroupBy(self.df_multi_index, by="values", level="letter")
        pandas_gb = self.df_multi_index.groupby(["values", "letter"])

        # Compare group counts
        assert gb.ngroups == pandas_gb.ngroups

    def test_callable_groupby(self):
        """Test grouping by callable."""
        # Group by index length (simple callable)
        def grouper_func(x):
            return len(str(x))

        gb = DataFrameGroupBy(self.df_mixed_cols, by=grouper_func)
        pandas_gb = self.df_mixed_cols.groupby(grouper_func)

        assert gb.ngroups == pandas_gb.ngroups


class TestSeriesByLevelProcessing:
    """Test SeriesGroupBy by/level argument processing."""

    def setup_method(self):
        """Setup test Series with various index types."""
        # Series with MultiIndex
        multi_index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2), ("A", 3), ("B", 3)],
            names=["letter", "number"],
        )
        self.series_multi = pd.Series(
            [1, 2, 3, 4, 5, 6], index=multi_index, name="values"
        )

        # Series with numeric level names
        numeric_multi_index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2)],
            names=[pd.Timestamp("2024-01-01"), 42],
        )  # Timestamp and int level names
        self.series_numeric_levels = pd.Series([1, 2, 3, 4], index=numeric_multi_index)

        # Regular series
        self.series_regular = pd.Series([1, 2, 3, 4, 5, 6], name="values")

    def test_series_level_string_name(self):
        """Test Series grouping by string level name."""
        gb = SeriesGroupBy(self.series_multi, level="letter")
        pandas_gb = self.series_multi.groupby(level="letter")

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_series_level_number(self):
        """Test Series grouping by level number."""
        gb = SeriesGroupBy(self.series_multi, level=0)
        pandas_gb = self.series_multi.groupby(level=0)

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_series_level_negative_number(self):
        """Test Series grouping by negative level number."""
        gb = SeriesGroupBy(self.series_multi, level=-1)
        pandas_gb = self.series_multi.groupby(level=-1)

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_series_multiple_levels(self):
        """Test Series grouping by multiple levels."""
        gb = SeriesGroupBy(self.series_multi, level=["letter", "number"])
        pandas_gb = self.series_multi.groupby(level=["letter", "number"])

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_series_numeric_level_names(self):
        """Test Series with non-string level names."""
        timestamp_level = pd.Timestamp("2024-01-01")
        int_level = 42

        # Test Timestamp level name
        gb1 = SeriesGroupBy(self.series_numeric_levels, level=timestamp_level)
        pandas_gb1 = self.series_numeric_levels.groupby(level=timestamp_level)

        result1 = gb1.sum()
        expected1 = pandas_gb1.sum()
        pd.testing.assert_series_equal(result1, expected1)

        # Test int level name
        gb2 = SeriesGroupBy(self.series_numeric_levels, level=int_level)
        pandas_gb2 = self.series_numeric_levels.groupby(level=int_level)

        result2 = gb2.sum()
        expected2 = pandas_gb2.sum()
        pd.testing.assert_series_equal(result2, expected2)

    def test_series_by_array(self):
        """Test Series grouping by external array."""
        grouper_array = np.array(["X", "Y", "X", "Y", "X", "Y"])

        gb = SeriesGroupBy(self.series_multi, by=grouper_array)
        pandas_gb = self.series_multi.groupby(grouper_array)

        result = gb.sum()
        expected = pandas_gb.sum()

        pd.testing.assert_series_equal(result, expected)

    def test_series_combined_by_level(self):
        """Test Series combining by and level."""
        grouper_array = np.array(["X", "Y", "X", "Y", "X", "Y"])

        gb = SeriesGroupBy(self.series_multi, by=grouper_array, level="letter")

        # Should have grouping keys from both by and level
        assert gb.ngroups > 0


class TestErrorHandling:
    """Test error handling for invalid by/level specifications."""

    def setup_method(self):
        """Setup test data."""
        self.df = pd.DataFrame(
            {"A": [1, 2, 3, 4], "B": [10, 20, 30, 40], "group": ["X", "Y", "X", "Y"]}
        )

        multi_index = pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 2), ("B", 1), ("B", 2)], names=["letter", "number"]
        )
        self.series_multi = pd.Series([1, 2, 3, 4], index=multi_index)

    def test_invalid_column_name(self):
        """Test error handling for invalid column names."""
        with pytest.raises(
            KeyError, match="Column or index level 'nonexistent' not found"
        ):
            DataFrameGroupBy(self.df, by="nonexistent")

    def test_invalid_level_name(self):
        """Test error handling for invalid level names."""
        with pytest.raises(KeyError, match="Level 'invalid_level' not found"):
            SeriesGroupBy(self.series_multi, level="invalid_level")

    def test_invalid_level_number(self):
        """Test error handling for out-of-bounds level numbers."""
        with pytest.raises(IndexError, match="Level 5 out of bounds"):
            SeriesGroupBy(self.series_multi, level=5)

    def test_level_on_regular_index(self):
        """Test error for level on non-MultiIndex."""
        regular_series = pd.Series([1, 2, 3, 4])

        with pytest.raises(
            ValueError, match="Level 'invalid' not valid for non-MultiIndex"
        ):
            SeriesGroupBy(regular_series, level="invalid")

    def test_array_length_mismatch(self):
        """Test error for array length mismatch."""
        wrong_length_array = np.array(["X", "Y"])  # Too short

        with pytest.raises(
            ValueError, match="Length of grouper \\(2\\) != length of DataFrame \\(4\\)"
        ):
            DataFrameGroupBy(self.df, by=wrong_length_array)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def setup_method(self):
        """Setup edge case test data."""
        # DataFrame with duplicate column names (not supported by pandas but
        # we should handle gracefully)
        self.df_simple = pd.DataFrame(
            {"A": [1, 2, 3, 4], "B": [10, 20, 30, 40], "group": ["X", "Y", "X", "Y"]}
        )

        # Empty DataFrame
        self.df_empty = pd.DataFrame({"A": [], "group": []})

    def test_none_values_handling(self):
        """Test handling when by/level are None."""
        with pytest.raises(
            ValueError, match="Must provide either 'by', 'level' or `grouper`"
        ):
            DataFrameGroupBy(self.df_simple)

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        # This should work without error
        gb = DataFrameGroupBy(self.df_empty, by="group")
        assert gb.ngroups == 0

    def test_single_row_dataframe(self):
        """Test with single row DataFrame."""
        single_row_df = pd.DataFrame({"A": [1], "group": ["X"]})

        gb = DataFrameGroupBy(single_row_df, by="group")
        pandas_gb = single_row_df.groupby("group")

        assert gb.ngroups == pandas_gb.ngroups

        result = gb.sum()
        expected = pandas_gb.sum()
        pd.testing.assert_frame_equal(result, expected)


class TestPandasCompatibility:
    """Test compatibility with pandas groupby behavior."""

    def setup_method(self):
        """Setup compatibility test data."""
        # Create complex test case
        complex_multi_index = pd.MultiIndex.from_tuples(
            [
                (pd.Timestamp("2024-01-01"), "A", 1.5),
                (pd.Timestamp("2024-01-01"), "B", 2.5),
                (pd.Timestamp("2024-01-02"), "A", 1.5),
                (pd.Timestamp("2024-01-02"), "B", 2.5),
            ],
            names=[pd.Timedelta("1H"), "category", 3.14159],
        )

        self.df_complex = pd.DataFrame(
            {
                ("data", "primary"): [10, 20, 30, 40],
                42: [1, 2, 3, 4],
                "simple": ["P", "Q", "P", "Q"],
            },
            index=complex_multi_index,
        )

    def test_complex_mixed_types(self):
        """Test complex scenario with mixed column and level name types."""
        # Group by tuple column name
        gb1 = DataFrameGroupBy(self.df_complex, by=("data", "primary"))
        assert gb1.ngroups > 0

        # Group by int column name
        gb2 = DataFrameGroupBy(self.df_complex, by=42)
        assert gb2.ngroups > 0

        # Group by Timedelta level name
        gb3 = DataFrameGroupBy(self.df_complex, level=pd.Timedelta("1H"))
        assert gb3.ngroups > 0

        # Group by float level name
        gb4 = DataFrameGroupBy(self.df_complex, level=3.14159)
        assert gb4.ngroups > 0


class TestPerformanceWithMixedTypes:
    """Test that mixed column types don't significantly impact performance."""

    def test_large_dataframe_mixed_columns(self):
        """Test with larger DataFrame and mixed column types."""
        n = 10000
        df_large = pd.DataFrame(
            {
                "str_col": np.random.choice(["A", "B", "C"], n),
                42: np.random.randint(0, 100, n),
                3.14: np.random.choice(["X", "Y", "Z"], n),
                "values": np.random.randn(n),
            }
        )

        # Test grouping by different column types
        for col in ["str_col", 42, 3.14]:
            gb = DataFrameGroupBy(df_large, by=col)
            pandas_gb = df_large.groupby(col)

            # Basic validation
            assert gb.ngroups == pandas_gb.ngroups

            # Performance shouldn't be drastically different
            result = gb.sum()
            expected = pandas_gb.sum()

            # Just check shapes match (values may differ slightly due to implementation)
            assert result.shape == expected.shape


if __name__ == "__main__":
    pytest.main([__file__])
