"""Robustness: rerun the in-sample regressions with the 2015 crash removed.

The window 2015-01 to 2016-02 covers the ChiNext bubble, the crash, and the
circuit-breaker episode. If a coefficient owes its significance to that one
stretch, it is a crash artifact, not a rule.
"""

import os

import pandas as pd
import statsmodels.api as sm

from features import all_spreads, study_sample
from regressions import MAIN_FLOOR, NW_LAGS

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
CRASH_START, CRASH_END = "2015-01-01", "2016-02-29"


def run_excrash():
    rows = []
    for name, panel in all_spreads().items():
        net = study_sample(panel)
        if len(net) < MAIN_FLOOR:
            continue
        factors = [c for c in net.columns if c != "y"]
        sub = net[(net.index < CRASH_START) | (net.index > CRASH_END)]
        for label, sample in (("full", net), ("ex-crash", sub)):
            fit = sm.OLS(sample["y"], sm.add_constant(sample[factors])).fit(
                cov_type="HAC", cov_kwds={"maxlags": NW_LAGS})
            row = {"spread": name, "sample": label, "n": len(sample),
                   "r2": round(fit.rsquared, 3)}
            for f in factors:
                row[f"t_{f}"] = round(fit.tvalues[f], 2)
            rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS_DIR, "robust_excrash.csv"), index=False)
    return out


if __name__ == "__main__":
    print(run_excrash().to_string(index=False))
