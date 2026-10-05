"""Run the whole study in order. Each step writes its output to results/."""

import os

import data
import evaluate
import oos
import regressions
import risk
import robustness

if __name__ == "__main__":
    if not os.path.exists(os.path.join(data.DATA_DIR, "prices_composite.csv")):
        print("== downloading data ==")
        data.fetch_prices()
        data.fetch_pe()

    print("== in-sample regressions ==")
    summary, _ = regressions.run_all()
    print(summary.to_string(index=False))

    print("\n== robustness: drop the 2015 crash ==")
    print(robustness.run_excrash().to_string(index=False))

    print("\n== out-of-sample (80/20) ==")
    print(oos.run_all().to_string(index=False))

    print("\n== risk metrics ==")
    rm = risk.risk_metrics()
    ld = risk.two_factor_loadings()
    rm.to_csv(os.path.join(risk.RESULTS_DIR, "risk_metrics.csv"), index=False)
    ld.to_csv(os.path.join(risk.RESULTS_DIR, "loadings.csv"), index=False)
    print(rm.to_string(index=False))
    print(ld.to_string(index=False))

    print("\n== figures ==")
    evaluate.fig_oos()
    evaluate.fig_size_tilt()
    print("done. see results/")
