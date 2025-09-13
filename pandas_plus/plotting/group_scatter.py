from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
try:
    from sklearn import linear_model as lm
except ModuleNotFoundError:
    print("scikit-learn is an optional dependency of pandas-plus need for GroupScatter")
    raise 

from pandas_plus.util import ArrayType1D


@dataclass
class GroupScatter:
    """
    Class for creating group scatter plots with regression lines.

    This class groups x values into bins and calculates the mean y value for each bin,
    then fits a regression line to represent the relationship between x and y.

    Parameters
    ----------
    x : ArrayType1D
        X values for the scatter plot
    y : ArrayType1D
        Y values for the scatter plot
    n_groups : int, default 25
        Number of bins to group x values into
    filter : ArrayType1D, default slice(None)
        Boolean mask or slice to filter input data
    sample_weight : ArrayType1D, optional
        Weights to apply during regression fitting
    deg : int, default 1
        Degree of the polynomial for regression
    fit_intercept : bool, default True
        Whether to fit an intercept term in the regression
    """

    x: ArrayType1D
    y: ArrayType1D
    n_groups: int = 25
    filter: ArrayType1D = slice(None)
    sample_weight: ArrayType1D = None
    deg: int = 1
    fit_intercept: bool = True

    def __post_init__(self):
        self._x = np.asarray(self.x[self.filter])
        self._y = np.asarray(self.y[self.filter])
        null_filter = np.isnan(self._x) | np.isnan(self._y)
        if null_filter.any():
            self._x = self._x[~null_filter]
            self._y = self._y[~null_filter]
        self._calculate_bins()
        self._calculate_regression()

    @cached_property
    def _X(self):
        X = self._x[:, None]
        if self.deg > 1:
            X = X ** np.arange(1, self.deg + 1)
        return X

    def _calculate_bins(self):
        """
        Group x values into bins and calculate mean y value for each bin.

        Creates two attributes:
        - bins: pandas.qcut bins for the x values
        - y_means: mean y value for each bin
        """
        self.bins = pd.qcut(self._x, q=self.n_groups, duplicates="drop")
        self.y_means = pd.Series(self._y).groupby(self.bins, observed=True).mean()

    def _calculate_regression(self):
        """
        Fit a regression line to the data.

        Creates several attributes:
        - fit: The fitted sklearn LinearRegression model
        - r_squared: R^2 score of the regression
        - regression_curve: Predicted y values for each x
        - regression_coefs: Coefficients of the regression
        """
        # self.regression_coefs = np.polyfit(self._x, self._y, self.deg)
        # self.regression_poly = np.polynomial.Polynomial(self.regression_coefs[::-1])
        # self.regression_curve = Series(self.regression_poly(self._x).values, self._x.values)
        self.fit = fit = lm.LinearRegression(fit_intercept=self.fit_intercept).fit(
            X=self._X,
            y=self._y,
            sample_weight=self.sample_weight,
        )
        self.r_squared = fit.score(self._X, self._y)
        self.regression_curve = pd.Series(fit.predict(self._X), self._x)
        self.regression_coefs = [*fit.coef_, fit.intercept_]

    def plot(self, **plot_kwargs):
        """
        Create a scatter plot with regression line.

        The plot shows the mean y values for each bin of x values (as points)
        and the regression line fit to the original data.

        Parameters
        ----------
        **plot_kwargs
            Additional keyword arguments passed to pandas.Series.plot

        Returns
        -------
        matplotlib.axes.Axes
            The matplotlib axes containing the plot
        """
        bin_means = (self.bins.categories.right + self.bins.categories.left) / 2
        ax = pd.Series(self.y_means.values, bin_means).plot(**plot_kwargs, style="o")
        self.regression_curve.plot(style="-", ax=ax)

        return ax
