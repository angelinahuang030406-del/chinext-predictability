"""Guards for the two leaks the factor test can't see.

1. Target alignment: y at month t must be the NEXT month's spread, rebuilt
   here independently from the raw price files. Setting y to the current
   month's spread (the classic leak) fails this.
2. Train/test separation: out-of-sample predictions must not depend on the
   test-period targets. Scramble those targets; if any prediction or the
   benchmark moves, test data leaked into the fit.
"""

import numpy as np
import pandas as pd
import pytest

import features
import oos


@pytest.mark.parametrize("index,bench", [("composite", "csi300"), ("basic", "composite")])
def test_target_is_next_month_spread(index, bench):
    panel = features.build_spread(index, bench)

    def month_end_ret(name):
        close = features.load_prices(name)["close"].resample("ME").last()
        return close.pct_change()

    spread = month_end_ret(index) - month_end_ret(bench)
    for t in ["2016-03-31", "2019-08-31", "2023-11-30"]:
        t = pd.Timestamp(t)
        next_month = spread.index[spread.index.get_loc(t) + 1]
        assert np.isclose(panel.loc[t, "y"], spread.loc[next_month], rtol=1e-10)


def test_oos_split_is_chronological():
    """The test months must be the last 20% of the sample. A random split
    trains on the future to predict the past -- the scramble test below
    can't see that, because test-row targets still aren't used in the fit."""
    for name, index, bench in features.SPREADS[:5]:
        panel = features.build_spread(index, bench)
        net = features.study_sample(panel)
        _, preds = oos.evaluate_spread(name, panel)
        assert list(preds.index) == list(net.index[-len(preds):])


def test_oos_predictions_ignore_test_targets():
    name, index, bench = features.SPREADS[0]
    panel = features.build_spread(index, bench)
    _, base = oos.evaluate_spread(name, panel)

    scrambled = panel.copy()
    test_months = base.index
    rng = np.random.default_rng(0)
    scrambled.loc[test_months, "y"] = rng.normal(0, 1, len(test_months))
    _, after = oos.evaluate_spread(name, scrambled)

    assert np.allclose(base["pred"], after["pred"], rtol=1e-12)
    assert np.allclose(base["bench"], after["bench"], rtol=1e-12)
