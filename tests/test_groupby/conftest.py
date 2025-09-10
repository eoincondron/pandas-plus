import pandas as pd


def assert_pd_equal(left, right, dtype_kind_only: bool = False, **kwargs):
    if isinstance(left, pd.Series):
        if dtype_kind_only:
            if kwargs.get("check_dtype", True):
                msg = f"dtypes have different kinds: {left.dtype}, {right.dtype}"
                assert left.dtype.kind == right.dtype.kind, msg
            right = right.astype(left.dtype)
        pd.testing.assert_series_equal(left, right, **kwargs)
    else:
        if dtype_kind_only:
            if kwargs.get("check_dtype", True):
                msg = f"dtypes have different kinds: {left.dtypes}, {right.dtypes}"
                assert all(a.kind == b.kind for a, b in zip(left.dtypes, right.dtypes))
            right = right.astype(left.dtypes)
        pd.testing.assert_frame_equal(left, right, **kwargs)
