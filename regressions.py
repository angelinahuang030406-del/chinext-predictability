"""In-sample regressions: OLS with Newey-West errors, plus the standard
diagnostics that make the t-statistics worth reading.

Tiers are mechanical and were fixed before looking at any results:
>= 60 complete months -> main group; 36-60 -> appendix (momentum and vol
only); below 36 -> no regression. See DESIGN_NOTES.md.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import adfuller

from features import all_spreads, study_sample

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
NW_LAGS = 4
MAIN_FLOOR, APPENDIX_FLOOR = 60, 36


def vif(X):
    """1 / (1 - R^2) of each factor regressed on the others."""
    out = {}
    Z = (X - X.mean()) / X.std()
    for col in X.columns:
        others = [c for c in X.columns if c != col]
        A = sm.add_constant(Z[others])
        r2 = sm.OLS(Z[col], A).fit().rsquared
        out[col] = 1.0 / (1.0 - r2)
    return out


def run_spread(name, panel):
    net = study_sample(panel)
    factors = [c for c in net.columns if c != "y"]

    if len(net) < APPENDIX_FLOOR:
        return {"spread": name, "tier": "descriptive", "n": len(net)}, None
    tier = "main" if len(net) >= MAIN_FLOOR else "appendix"
    if tier == "appendix":
        factors = [f for f in ("mom", "vol") if f in factors]

    X = sm.add_constant(net[factors])
    fit = sm.OLS(net["y"], X).fit(cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})

    row = {
        "spread": name, "tier": tier, "n": len(net),
        "r2": round(fit.rsquared, 3), "adj_r2": round(fit.rsquared_adj, 3),
        "adf_p": round(adfuller(net["y"], autolag="AIC")[1], 4),
        "dw": round(durbin_watson(fit.resid), 2),
        "max_vif": round(max(vif(net[factors]).values()), 2),
    }
    coefs = pd.DataFrame({
        "spread": name,
        "term": X.columns,
        "coef": fit.params.round(4),
        "nw_t": fit.tvalues.round(2),
        "p": fit.pvalues.round(3),
    })
    return row, coefs


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary, coef_frames = [], []
    for name, panel in all_spreads().items():
        row, coefs = run_spread(name, panel)
        summary.append(row)
        if coefs is not None:
            coef_frames.append(coefs)
    summary = pd.DataFrame(summary)
    coefs = pd.concat(coef_frames, ignore_index=True)
    summary.to_csv(os.path.join(RESULTS_DIR, "insample_summary.csv"), index=False)
    coefs.to_csv(os.path.join(RESULTS_DIR, "insample_coefs.csv"), index=False)
    return summary, coefs


if __name__ == "__main__":
    summary, coefs = run_all()
    print(summary.to_string(index=False))
    print()
    sig = coefs[(coefs["term"] != "const") & (coefs["nw_t"].abs() >= 1.65)]
    print("|t| >= 1.65:" if len(sig) else "no factor reaches |t| = 1.65")
    if len(sig):
        print(sig.to_string(index=False))
