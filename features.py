"""Build the monthly factor panel for each spread.

Everything here is point-in-time by construction: factors are fixed at
month-end t using only data up to t, the target is the month t+1 spread,
and every historical statistic (percentile, z-score) uses an expanding
window that starts at the series' own first observation. The warmup of 24
months is deliberately conservative -- a percentile over 6 observations is
noise pretending to be a number.
"""

import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WARMUP = 24

# The regressions use the original study's 120-month window. Data outside it
# still matters: the months before feed the warmups (momentum needs 12m of
# history plus 24m of z-score burn-in = 36m), which is why data.py pulls
# everything the source has.
STUDY_START, STUDY_END = "2014-12", "2024-11"

# (spread name, index, benchmark). Layer 1 first, then the slices.
SPREADS = [
    ("composite-csi300", "composite", "csi300"),
    ("chinext-comp", "chinext", "composite"),
    ("chinext50-comp", "chinext50", "composite"),
    ("chinext300-comp", "chinext300", "composite"),
    ("basic-comp", "basic", "composite"),
    ("largecap-comp", "largecap", "composite"),
    ("chinext200-comp", "chinext200", "composite"),
    ("smallcap-comp", "smallcap", "composite"),
]


def load_prices(name):
    path = os.path.join(DATA_DIR, f"prices_{name}.csv")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df


def monthly_series(name):
    """Month-end close, monthly return, within-month realized vol, and
    monthly mean dollar volume for one index."""
    df = load_prices(name)
    daily_ret = df["close"].pct_change()
    out = pd.DataFrame({
        "close": df["close"].resample("ME").last(),
        "ret": df["close"].resample("ME").last().pct_change(),
        "vol": daily_ret.resample("ME").std() * np.sqrt(252),
        "dollar_vol": (df["close"] * df["volume"]).resample("ME").mean(),
    })
    # a month with fewer than 10 trading days is not a real month
    out["vol"] = out["vol"].where(daily_ret.resample("ME").count() >= 10)
    return out


def expanding_pct(s, warmup=WARMUP):
    """Rank of the current value within [start, t]. The first `warmup`
    non-missing months return NaN rather than a meaningless rank."""
    r = s.expanding(min_periods=1).rank(pct=True)
    return r.where(s.notna().cumsum() > warmup)


def expanding_z(s, warmup=WARMUP):
    mu = s.expanding(min_periods=1).mean()
    sd = s.expanding(min_periods=warmup).std()
    return (s - mu) / sd


def momentum_12_1(close):
    """12-1 momentum: skip the most recent month to keep short-term
    reversal out of the trend signal."""
    return close.shift(1) / close.shift(12) - 1.0


def load_pe(name):
    """Month-end PE if we have a cached series for this index, else None."""
    path = os.path.join(DATA_DIR, f"pe_{name}.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    date_col = df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.set_index(date_col).sort_index()
    # legulegu ships several PE variants; the TTM column is the usable one
    pe_col = next((c for c in df.columns if "TTM" in str(c) or "滚动" in str(c)), df.columns[-1])
    pe = pd.to_numeric(df[pe_col], errors="coerce").resample("ME").last()
    return pe.where(pe > 0)


def build_spread(index, bench):
    """Factor panel for one spread. Valuation enters only when both legs
    have a PE series -- with free data that is currently none of them, and
    the panel simply carries three factors instead of four."""
    a, b = monthly_series(index), monthly_series(bench)
    idx = a.index.union(b.index)
    a, b = a.reindex(idx), b.reindex(idx)

    X = pd.DataFrame(index=idx)
    X["mom"] = expanding_z(momentum_12_1(a["close"]) - momentum_12_1(b["close"]))
    X["vol"] = expanding_z(a["vol"])
    X["dvol_pct"] = expanding_pct(a["dollar_vol"])

    pe_a, pe_b = load_pe(index), load_pe(bench)
    if pe_a is not None and pe_b is not None:
        rel_ep = (1 / pe_a).reindex(idx) - (1 / pe_b).reindex(idx)
        X["val_pct"] = expanding_pct(rel_ep)

    X["y"] = (a["ret"] - b["ret"]).shift(-1)
    return X


def study_sample(panel):
    """Rows actually used in regressions: study window, complete cases."""
    return panel.loc[STUDY_START:STUDY_END].dropna()


def all_spreads():
    return {name: build_spread(i, b) for name, i, b in SPREADS}


if __name__ == "__main__":
    for name, panel in all_spreads().items():
        cols = [c for c in panel.columns if c != "y"]
        net = study_sample(panel)
        span = f"{net.index.min():%Y-%m} -> {net.index.max():%Y-%m}" if len(net) else "-"
        print(f"{name:18s} factors={cols} n={len(net):3d}  {span}")
