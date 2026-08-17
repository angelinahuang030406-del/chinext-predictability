"""Out-of-sample evaluation: 80/20 chronological split.

Train on the first 80% of each spread's sample, freeze the coefficients,
predict the last 20% month by month. The benchmark is the training-period
mean -- beating it is the minimum bar for claiming any predictability.
"""

import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

from features import all_spreads, study_sample
from regressions import APPENDIX_FLOOR, MAIN_FLOOR

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
TRAIN_FRAC = 0.8


def evaluate_spread(name, panel):
    net = study_sample(panel)
    if len(net) < MAIN_FLOOR:
        return None, None
    factors = [c for c in net.columns if c != "y"]

    k = int(len(net) * TRAIN_FRAC)
    train, test = net.iloc[:k], net.iloc[k:]

    fit = sm.OLS(train["y"], sm.add_constant(train[factors])).fit()
    pred = fit.params["const"] + test[factors].mul(fit.params[factors]).sum(axis=1)
    bench = train["y"].mean()

    e_model = test["y"] - pred
    e_bench = test["y"] - bench
    row = {
        "spread": name,
        "train_n": len(train), "test_n": len(test),
        "test_span": f"{test.index.min():%Y-%m} -> {test.index.max():%Y-%m}",
        "oos_r2": round(1 - (e_model ** 2).sum() / (e_bench ** 2).sum(), 4),
        "hit_model": round((np.sign(test["y"]) == np.sign(pred)).mean(), 3),
        "hit_bench": round((np.sign(test["y"]) == np.sign(bench)).mean(), 3),
    }
    series = pd.DataFrame({"spread": name, "y": test["y"], "pred": pred, "bench": bench})
    return row, series


def run_all():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows, frames = [], []
    for name, panel in all_spreads().items():
        row, series = evaluate_spread(name, panel)
        if row is not None:
            rows.append(row)
            frames.append(series)
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "oos_summary.csv"), index=False)
    pd.concat(frames).to_csv(os.path.join(RESULTS_DIR, "oos_predictions.csv"))
    return summary


if __name__ == "__main__":
    print(run_all().to_string(index=False))
