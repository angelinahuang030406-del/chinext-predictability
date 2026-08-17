# chinext-predictability

Testing whether return differences between eight ChiNext broad-market
indices are predictable — with strict point-in-time discipline and a
verifiable no-lookahead test. **The answer is no: out-of-sample, the model
loses to a plain historical mean. What separates these indices is risk,
not expected return.**

![Out-of-sample: model vs historical-mean benchmark](results/fig_oos.png)

## Quickstart

    pip install -r requirements.txt
    python data.py        # pulls daily index data (Sina via akshare), caches to data/
    python main.py        # runs the full study, writes tables and figures to results/
    pytest                # runs the no-lookahead guard

Raw data isn't committed (the original study used licensed terminal
exports that can't be redistributed), which is why `data.py` exists. The
full run takes a few minutes on a laptop.

This is a public rebuild of a study I first ran on licensed data during
an internship. Same design, same 120-month window (2014-12 to 2024-11),
free data sources. DESIGN_NOTES.md lists exactly what differs and why.

## Research design

The eight indices correlate at 0.95+ monthly, so regressing each one
separately would estimate the same thing eight times. I split the problem
into two layers: ChiNext-vs-CSI300 (is the sector worth holding?) and
each-index-vs-Composite (which slice?). By construction the two layers add
back up to each index's excess return over CSI 300.

Three monthly factors, all computable at month-end t: 12-1 momentum
(relative to the spread's benchmark), realized volatility from daily
returns, and a dollar-volume percentile as an activity proxy. The original
study had a fourth factor — an E/P plus B/P valuation composite — but free
sources don't provide PE history for enough of these indices, so the
public version runs without it. It was insignificant everywhere in the
original study, at every horizon, so nothing rests on it.

Estimation is OLS with Newey-West errors. Out-of-sample: train on the
first 80% of each sample chronologically, freeze the coefficients, predict
the last 20%, and benchmark against predicting the training-period mean.

## Data, and the backfill problem

Several of these indices were launched years after their base date, and
vendor data fills the gap with backfilled simulation. The original study
had to handle that in tiers (fully-real indices, one 43%-backfilled index
kept with a mandatory real-period recheck, two ~85%-backfilled ones sent
to an appendix). The recheck earned its keep: a marginally significant
momentum signal on the LargeCap spread disappeared once the backfilled
period was dropped.

The public data makes this simpler in an honest way: Sina serves each
index from its launch date, so there is no backfilled history here at
all. The cost is sample: LargeCap has only 33 usable months in the study
window (below the pre-registered 36-month floor, so it's descriptive
only), and ChiNext 200 / Small Cap have essentially none. Five spreads
remain for the regressions.

## Validity checks

Three things guard the pipeline. First, look-ahead: every historical
statistic uses an expanding window, and `tests/test_no_lookahead.py`
verifies it — it truncates all raw data after a cutoff month, reruns the
whole feature pipeline, and asserts the values at the cutoff are identical
to 10 decimal places, for three spreads at three different dates. If
anything peeks at the future, `pytest` fails. Second, the regressions
themselves: the spreads are stationary, factor VIFs stay under 1.5, and
Durbin-Watson sits at 1.9-2.1; inference uses Newey-West errors
throughout. Third, the tiering rules above were fixed before any results
were seen.

## Results

In-sample, three coefficients clear the 5% significance bar: momentum on
the sector spread (t = -2.22, a reversal), volatility on the sector spread
(t = +2.20), and momentum on the ChiNext50 spread (t = +2.65, a
continuation). Dollar volume on the Basic spread is borderline
(t = -1.98). R² for these regressions sits between 2% and 7% — normal
territory for monthly return prediction.

Out of sample, the picture flips:

| Spread              | In-sample significant? | OOS R²  | Hit rate (model / mean) |
|---------------------|------------------------|---------|-------------------------|
| Composite − CSI300  | yes (mom, vol)         | -6.7%   | 42% / 54%               |
| ChiNext50 − Comp    | yes (mom)              | -5.5%   | 58% / 58%               |
| Basic − Comp        | borderline             | -1.7%   | 63% / 71%               |
| ChiNext − Comp      | no                     | +2.6%   | 54% / 50%               |
| ChiNext300 − Comp   | no                     | +4.2%   | 59% / 64%               |

Put the two side by side and they don't match: the spreads that looked
significant in-sample fail out-of-sample, and the model's hit rate mostly
sits below just predicting the training-period mean. This is overfitting —
the model learned noise, not signal. The conclusion stands: no exploitable
predictability at these horizons, consistent with Welch & Goyal (2008),
whose paper I should have read before building the model rather than
after.

What actually distinguishes these indices is risk. Over 2015-2024 they all
run at 32-35% annualized volatility (CSI 300: 21%) with maximum drawdowns
of -63% to -71%, and a two-factor decomposition (sector + size) gets R²
above 0.98 for every index I can estimate. Sector betas are all ~1.0; the
only real dimension is size tilt: +0.92 (ChiNext 50) down to +0.21
(Basic), with the two young small-cap indices too short to estimate on
public data — the original study put them near -1.0. Two details I didn't
expect: the index selected by trading volume (ChiNext 50) carries the most
extreme large-cap tilt, and the index built to cover 85% of market cap
behaves almost exactly like the whole market (s = 0.21). Construction
documents tell you the sign of a tilt, not its size.

![Size tilt is the only real difference](results/fig_size_tilt.png)

## Limitations

Price indices only, so dividends are ignored (understates CSI 300 by
1-2%/yr — a level effect, not a timing one). The valuation factor from the
original study is absent here, and the turnover factor is proxied by
dollar volume. Only the one-month horizon is tested in this rebuild; the
original also ran 3- and 6-month horizons with the same conclusion. The
three youngest indices contribute little or nothing — that's the honest
price of refusing backfilled history. And with ~30 coefficients across
five regressions, two or three "significant" ones is roughly what
multiple testing hands you for free, which is exactly how the
out-of-sample results say to read them.

## Repo structure

`data.py` pulls and caches raw index data · `features.py` builds the
factors with expanding windows · `regressions.py` runs OLS + Newey-West
and diagnostics · `oos.py` does the 80/20 evaluation · `risk.py` computes
risk metrics and the two-factor loadings · `evaluate.py` makes the
figures · `main.py` runs everything in order.

## References

Welch, I. and Goyal, A. (2008). A Comprehensive Look at the Empirical
Performance of Equity Premium Prediction. *Review of Financial Studies*.
