# Design notes

Short notes on the choices that aren't obvious from the code. The README
tells the story; this file answers "why did you do it that way".

## Study window

Regressions use 2014-12 to 2024-11 — the same 120-month window as the
original study, so results are comparable. Data outside the window is not
wasted: everything before 2014-12 feeds the warmups (below), and data.py
deliberately pulls the full history the source offers.

## Warmup accounting

Every factor needs history before it produces its first usable value:

| Factor            | Needs                              | Total warmup |
|-------------------|------------------------------------|--------------|
| momentum (12-1)   | 12m lookback, then 24m z-score     | 36 months    |
| realized vol      | 24m z-score                        | 24 months    |
| dollar-volume pct | 24m expanding percentile           | 24 months    |
| valuation pct     | 24m expanding percentile (if PE)   | 24 months    |

Momentum is the binding constraint: an index needs 36 months of history
before it enters a regression at all. The 24-month burn-in is deliberate —
a percentile or z-score over a handful of observations is noise pretending
to be a number.

## Original study vs this public rebuild

The original study ran on licensed terminal data (iFind/CSMAR exports).
Those files can't be redistributed, so this repo rebuilds the study on
free sources. The differences, honestly listed:

| | Original | Public rebuild |
|---|---|---|
| Prices | licensed export, includes vendor backfill | Sina via akshare, starts at each launch date (no backfill) |
| Valuation factor | E/P + B/P composite (all indices) | dropped — free PE covers too few indices |
| Turnover factor | exchange turnover rate | dollar-volume percentile (proxy) |
| LargeCap spread | 109 months (backfilled prices) | ~33 months of real data in-window: below the 36-month floor, so descriptive only |
| ChiNext200 / SmallCap | exploratory appendix | no usable in-window sample |

Two consequences worth stating. First, the no-backfill data is a feature,
not a bug: the original study's robustness checks showed the one marginal
signal on the LargeCap spread was an artifact of backfilled history, and
the public data never contained that history to begin with. Second, the
valuation factor was insignificant everywhere in the original study at
every horizon, so dropping it does not change any conclusion.

If you have licensed data, `data.py` is the only file to swap: cache
prices as `data/prices_<name>.csv` and PE as `data/pe_<name>.csv`, and the
rest of the pipeline (including the valuation factor) activates unchanged.

## Admission and tiers (pre-registered, mechanical)

A spread enters the main regressions only if its complete-case sample in
the study window is at least 60 months; 36–60 months goes to an appendix;
below 36, descriptive statistics only. These thresholds were fixed in the
original study before any results were seen, and this rebuild keeps them.

## Why no walk-forward here

The original study also ran an expanding-window walk-forward; conclusions
matched the 80/20 split (both negative). The public version keeps only the
80/20 split for simplicity — one out-of-sample scheme, fully explainable.
