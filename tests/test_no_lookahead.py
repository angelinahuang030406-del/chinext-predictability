"""The look-ahead guard.

If any factor uses information from after month t, then recomputing it
with all post-t data deleted must change its value at t. So: truncate the
raw price files at t, rerun the whole feature pipeline, and require the
row at t to be identical to 10 decimal places. Three spreads, three
different cutoff dates.
"""

import numpy as np
import pandas as pd
import pytest

import features


CASES = [
    ("composite", "csi300", "2020-06-30"),
    ("chinext50", "composite", "2019-12-31"),
    ("basic", "composite", "2022-03-31"),
]


@pytest.mark.parametrize("index,bench,cutoff", CASES)
def test_truncated_recompute_matches(tmp_path, monkeypatch, index, bench, cutoff):
    t = pd.Timestamp(cutoff)
    full = features.build_spread(index, bench).drop(columns="y")

    for name in {index, bench}:
        features.load_prices(name).loc[:t].to_csv(tmp_path / f"prices_{name}.csv")
    monkeypatch.setattr(features, "DATA_DIR", str(tmp_path))
    truncated = features.build_spread(index, bench).drop(columns="y")

    row_full = full.loc[t].to_numpy(dtype=float)
    row_trunc = truncated.loc[t].to_numpy(dtype=float)
    assert not np.isnan(row_full).all(), "picked a cutoff before factors exist"
    assert np.allclose(row_full, row_trunc, rtol=1e-10, atol=1e-12, equal_nan=True)
