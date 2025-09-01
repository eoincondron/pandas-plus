"""
Unit tests for rolling methods with mask parameter functionality.
"""

import pytest
import numpy as np
import pandas as pd
from pandas_plus.groupby.api import SeriesGroupBy, DataFrameGroupBy


class TestRollingMask:
    """Test rolling window operations with mask parameter."""

    def setup_method(self):
        """Setup test data."""
        self.data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8], name="values")
        self.groups = pd.Series(["A", "A", "A", "A", "B", "B", "B", "B"], name="groups")

        # DataFrame test data
        self.df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6], "B": [10, 20, 30, 40, 50, 60]})
        self.df_groups = pd.Series(["X", "Y", "X", "Y", "X", "Y"])

    def test_series_rolling_mask_parameter_exists(self):
        """Test that rolling methods accept mask parameter."""
        series_gb = SeriesGroupBy(self.data, by=self.groups)
        rolling_obj = series_gb.rolling(window=2)

        # Test that all methods accept mask parameter without error
        mask = np.array([True, False, True, True, False, True, True, False])

        methods = ["sum", "mean", "min", "max"]
        for method in methods:
            method_func = getattr(rolling_obj, method)
            # Should not raise an error
            result = method_func(mask=mask)
            assert isinstance(result, pd.Series)
            assert len(result) == len(self.data)

    def test_dataframe_rolling_mask_parameter_exists(self):
        """Test that DataFrame rolling methods accept mask parameter."""
        df_gb = DataFrameGroupBy(self.df, by=self.df_groups)
        rolling_obj = df_gb.rolling(window=2)

        mask = np.array([True, False, True, True, False, True])

        methods = ["sum", "mean", "min", "max"]
        for method in methods:
            method_func = getattr(rolling_obj, method)
            # Should not raise an error
            result = method_func(mask=mask)
            assert isinstance(result, pd.DataFrame)
            assert result.shape == self.df.shape

    def test_rolling_mask_none_parameter(self):
        """Test that mask=None works correctly."""
        series_gb = SeriesGroupBy(self.data, by=self.groups)
        rolling_obj = series_gb.rolling(window=2)

        # Results should be identical for mask=None and no mask
        result_none = rolling_obj.sum(mask=None)
        result_no_mask = rolling_obj.sum()

        pd.testing.assert_series_equal(result_none, result_no_mask)

    def test_rolling_mask_functionality_basic(self):
        """Test basic functionality of mask parameter."""
        series_gb = SeriesGroupBy(self.data, by=self.groups)
        rolling_obj = series_gb.rolling(window=2)

        # Create simple mask
        mask = np.array([True, True, True, True, True, True, True, True])
        result_all_true = rolling_obj.sum(mask=mask)
        result_no_mask = rolling_obj.sum()

        # With all True mask, results should match no mask
        pd.testing.assert_series_equal(result_all_true, result_no_mask)

    def test_rolling_mask_preserves_shape(self):
        """Test that mask parameter preserves output shape."""
        series_gb = SeriesGroupBy(self.data, by=self.groups)
        rolling_obj = series_gb.rolling(window=3)

        mask = np.array([True, False, True, False, True, False, True, False])

        methods = ["sum", "mean", "min", "max"]
        for method in methods:
            result = getattr(rolling_obj, method)(mask=mask)
            assert len(result) == len(self.data)
            assert isinstance(result, pd.Series)

    def test_dataframe_rolling_mask_preserves_shape(self):
        """Test that DataFrame rolling mask preserves shape."""
        df_gb = DataFrameGroupBy(self.df, by=self.df_groups)
        rolling_obj = df_gb.rolling(window=2)

        mask = np.array([True, False, True, True, False, True])

        result = rolling_obj.sum(mask=mask)
        assert result.shape == self.df.shape
        assert list(result.columns) == list(self.df.columns)

    def test_rolling_mask_different_window_sizes(self):
        """Test mask parameter with different window sizes."""
        series_gb = SeriesGroupBy(self.data, by=self.groups)
        mask = np.array([True, False, True, True, False, True, True, False])

        for window in [1, 2, 3]:
            rolling_obj = series_gb.rolling(window=window)
            result = rolling_obj.sum(mask=mask)
            assert len(result) == len(self.data)
            assert isinstance(result, pd.Series)

    def test_rolling_agg_method_with_mask(self):
        """Test that the agg method also supports mask parameter."""
        series_gb = SeriesGroupBy(self.data, by=self.groups)
        rolling_obj = series_gb.rolling(window=2)

        mask = np.array([True, False, True, True, False, True, True, False])

        # Test agg method directly
        result = rolling_obj.agg("sum", mask=mask)
        assert isinstance(result, pd.Series)
        assert len(result) == len(self.data)

    def test_rolling_mask_with_small_data(self):
        """Test mask parameter with small datasets."""
        small_data = pd.Series([1, 2])
        small_groups = pd.Series(["A", "A"])
        small_mask = np.array([True, False])

        series_gb = SeriesGroupBy(small_data, by=small_groups)
        rolling_obj = series_gb.rolling(window=2)

        result = rolling_obj.sum(mask=small_mask)
        assert len(result) == 2
        assert isinstance(result, pd.Series)

    def test_series_groupby_rolling_inheritance(self):
        """Test that SeriesGroupByRolling properly inherits from BaseGroupByRolling."""
        series_gb = SeriesGroupBy(self.data, by=self.groups)
        rolling_obj = series_gb.rolling(window=2)

        # Test that it has the mask parameter in its methods
        import inspect

        for method_name in ["sum", "mean", "min", "max"]:
            method = getattr(rolling_obj, method_name)
            sig = inspect.signature(method)
            assert "mask" in sig.parameters
            # Check that mask has Optional[ArrayType1D] annotation or similar
            mask_param = sig.parameters["mask"]
            assert mask_param.default is None

    def test_dataframe_groupby_rolling_inheritance(self):
        """Test that DataFrameGroupByRolling properly inherits from
        BaseGroupByRolling."""
        df_gb = DataFrameGroupBy(self.df, by=self.df_groups)
        rolling_obj = df_gb.rolling(window=2)

        # Test that it has the mask parameter in its methods
        import inspect

        for method_name in ["sum", "mean", "min", "max"]:
            method = getattr(rolling_obj, method_name)
            sig = inspect.signature(method)
            assert "mask" in sig.parameters
            # Check that mask has default None
            mask_param = sig.parameters["mask"]
            assert mask_param.default is None


class TestRollingMaskEdgeCases:
    """Test edge cases for rolling mask functionality."""

    def test_empty_data_with_mask(self):
        """Test mask parameter with empty data."""
        empty_data = pd.Series([], dtype=float)
        empty_groups = pd.Series([], dtype=str)
        empty_mask = np.array([], dtype=bool)

        series_gb = SeriesGroupBy(empty_data, by=empty_groups)
        rolling_obj = series_gb.rolling(window=2)

        result = rolling_obj.sum(mask=empty_mask)
        assert len(result) == 0
        assert isinstance(result, pd.Series)

    def test_single_group_with_mask(self):
        """Test mask parameter with single group."""
        data = pd.Series([1, 2, 3, 4])
        groups = pd.Series(["A", "A", "A", "A"])
        mask = np.array([True, False, True, True])

        series_gb = SeriesGroupBy(data, by=groups)
        rolling_obj = series_gb.rolling(window=2)

        result = rolling_obj.sum(mask=mask)
        assert len(result) == 4
        assert isinstance(result, pd.Series)


if __name__ == "__main__":
    pytest.main([__file__])
