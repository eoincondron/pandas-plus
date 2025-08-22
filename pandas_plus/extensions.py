from functools import partialmethod
from typing import Optional
import numpy as np
import pandas as pd


S = pd.Series
DF = pd.DataFrame


for method_name, alias, kwargs in [
    ("to_frame", "tf", {}),
    ("value_counts", "vc", {}),
    ("sort_values", "sv", {}),
    ("sort_values", "svd", {"ascending": False}),
    ("groupby", "gb", {"observed": True}),
    ("drop_duplicates", "dd", {}),
]:
    for obj in (pd.Series, pd.DataFrame):
        try:
            method = getattr(obj, method_name)
        except AttributeError:
            continue
        if kwargs:
            method = partialmethod(method, **kwargs)
        setattr(obj, alias, method)


def filter_cols(df, regex: Optional[str] = None, like: Optional[str] = None):
    return df.filter(regex=regex, like=like)


DF.filter_cols = filter_cols


def drop_cols(df, columns):
    return df.drop(columns=columns)


DF.drop_cols = drop_cols


def pc(self, prec=1, exclude=tuple()):
    if self.ndim == 2:
        factors = [1 if c in exclude else 100 for c in self]
    else:
        factors = 100
    return self.mul(factors).round(prec)


S.pc = pc
DF.pc = pc


def normalize(self, to=1):
    return self / self.sum() * to


S.normalize = normalize
DF.normalize = normalize


def categorize_objects(self: pd.DataFrame, columns=None, exclude=None, inplace=True):
    if columns is None:
        columns = self.columns[self.dtypes == object]
    if exclude is not None:
        columns = columns.difference(exclude)

    if not inplace:
        self = self.copy()
        return self
    self[columns] = self[columns].astype("category")


DF.categorize_objects = categorize_objects


def select_numeric(self):
    return self.select_dtypes([np.number])


DF.select_numeric = select_numeric


def heat_map(self, precision=1, **kwargs):
    return self.style.background_gradient(**kwargs).format(precision=precision)


DF.heat_map = heat_map
DF.hm = heat_map
DF.hm2 = partialmethod(heat_map, axis=None)
