"""Download daily index data from free public sources and cache it locally.

Prices come from Sina (via akshare). Sina serves each index from its launch
date, so there is no backfilled history in this dataset -- see the README's
data section for why that matters. Valuation (PE) comes from Legulegu where
available; the study runs without the valuation factor for spreads it can't
cover.
"""

import argparse
import os
import signal

import akshare as ak
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# code -> (sina symbol, short name used in filenames and tables)
INDICES = {
    "399102": ("sz399102", "composite"),
    "399006": ("sz399006", "chinext"),
    "399673": ("sz399673", "chinext50"),
    "399012": ("sz399012", "chinext300"),
    "399640": ("sz399640", "basic"),
    "399293": ("sz399293", "largecap"),
    "399019": ("sz399019", "chinext200"),
    "399020": ("sz399020", "smallcap"),
    "000300": ("sh000300", "csi300"),
}

# Legulegu covers only some indices, under display names rather than codes.
PE_SYMBOLS = {
    "000300": "沪深300",
    "399673": "创业板50",
}


class _Timeout(Exception):
    pass


def _with_timeout(fn, seconds=30):
    """Run fn() but give up after `seconds`. Legulegu hangs on unknown
    symbols instead of erroring, so a hard timeout is the only safe guard."""
    def _raise(signum, frame):
        raise _Timeout()

    old = signal.signal(signal.SIGALRM, _raise)
    signal.alarm(seconds)
    try:
        return fn()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def fetch_prices(force=False):
    os.makedirs(DATA_DIR, exist_ok=True)
    for code, (symbol, name) in INDICES.items():
        path = os.path.join(DATA_DIR, f"prices_{name}.csv")
        if os.path.exists(path) and not force:
            print(f"cached  {name}")
            continue
        df = ak.stock_zh_index_daily(symbol=symbol)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()
        df.to_csv(path)
        print(f"fetched {name}: {len(df)} rows, {df.index.min().date()} -> {df.index.max().date()}")


def fetch_pe(force=False):
    for code, symbol in PE_SYMBOLS.items():
        name = INDICES[code][1]
        path = os.path.join(DATA_DIR, f"pe_{name}.csv")
        if os.path.exists(path) and not force:
            print(f"cached  pe_{name}")
            continue
        try:
            df = _with_timeout(lambda: ak.stock_index_pe_lg(symbol=symbol))
        except (_Timeout, Exception) as e:
            print(f"skipped pe_{name}: {type(e).__name__}")
            continue
        df.to_csv(path, index=False)
        print(f"fetched pe_{name}: {len(df)} rows")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--force", action="store_true", help="re-download even if cached")
    parser.add_argument("--skip-pe", action="store_true", help="prices only")
    args = parser.parse_args()

    fetch_prices(force=args.force)
    if not args.skip_pe:
        fetch_pe(force=args.force)
