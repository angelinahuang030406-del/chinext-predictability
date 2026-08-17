"""Risk metrics per index, and the two-factor decomposition.

The punchline of the study lives here: expected returns don't separate
these indices, risk exposures do. Sharpe ratios are deliberately absent --
they need a risk-free series, and the free sources don't provide one that
can be redistributed. Vol, drawdown, beta and VaR don't need it.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import monthly_series

# Risk metrics use the calendar decade, matching the original study. The
# regression window starts one month earlier for sample-size reasons, but
# including Dec-2014 (CSI300 +25% in a single month) would distort every
# beta and vol in this table.
RISK_START, RISK_END = "2015-01", "2024-12"

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

INDEX_NAMES = ["composite", "chinext", "chinext50", "chinext300", "basic",
               "largecap", "chinext200", "smallcap", "csi300"]
MIN_MONTHS = 24


def risk_metrics():
    csi300 = monthly_series("csi300")["ret"].loc[RISK_START:RISK_END]
    rows = []
    for name in INDEX_NAMES:
        r = monthly_series(name)["ret"].loc[RISK_START:RISK_END].dropna()
        if len(r) < MIN_MONTHS:
            rows.append({"index": name, "n": len(r)})
            continue
        nav = (1 + r).cumprod()
        mkt = csi300.reindex(r.index)
        rows.append({
            "index": name, "n": len(r),
            "ann_ret": round((1 + r).prod() ** (12 / len(r)) - 1, 3),
            "ann_vol": round(r.std() * np.sqrt(12), 3),
            "max_dd": round((nav / nav.cummax() - 1).min(), 3),
            "beta": round(r.cov(mkt) / mkt.var(), 2) if name != "csi300" else 1.0,
            "var95_m": round(r.quantile(0.05), 3),
        })
    return pd.DataFrame(rows)


def two_factor_loadings():
    """Contemporaneous R_i = a + b*sector + s*size + e. Sector is the
    composite's return; size is the largecap-composite spread, which the
    public source only serves from 2019 -- the loadings sample is shorter
    than the study window and says so in the n column."""
    comp = monthly_series("composite")["ret"]
    size = (monthly_series("largecap")["ret"] - comp).dropna()
    rows = []
    for name in ["chinext", "chinext50", "chinext300", "basic", "chinext200", "smallcap"]:
        r = monthly_series(name)["ret"].loc[RISK_START:RISK_END]
        df = pd.DataFrame({"y": r, "sector": comp, "size": size}).dropna()
        if len(df) < MIN_MONTHS:
            rows.append({"index": name, "n": len(df)})
            continue
        fit = sm.OLS(df["y"], sm.add_constant(df[["sector", "size"]])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 4})
        rows.append({
            "index": name, "n": len(df),
            "beta_sector": round(fit.params["sector"], 2),
            "t_sector": round(fit.tvalues["sector"], 1),
            "s_size": round(fit.params["size"], 2),
            "t_size": round(fit.tvalues["size"], 1),
            "r2": round(fit.rsquared, 3),
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rm = risk_metrics()
    ld = two_factor_loadings()
    rm.to_csv(os.path.join(RESULTS_DIR, "risk_metrics.csv"), index=False)
    ld.to_csv(os.path.join(RESULTS_DIR, "loadings.csv"), index=False)
    print(rm.to_string(index=False))
    print()
    print(ld.to_string(index=False))
