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

The no-backfill data is a feature, not a bug: backfilled history is
simulated by the index provider after the fact, under rules designed with
hindsight, and any signal found in it can't be checked against reality.
The public data never contains it. The price is sample size for the
youngest indices.

If you have licensed data, `data.py` is the only file to swap: cache
prices as `data/prices_<name>.csv` and PE as `data/pe_<name>.csv`, and the
rest of the pipeline (including the valuation factor) activates unchanged.

## Admission and tiers (fixed in advance, mechanical)

A spread enters the main regressions only if its complete-case sample in
the study window is at least 60 months; 36–60 months goes to an appendix;
below 36, descriptive statistics only. These thresholds were written down
before any results were seen, and this rebuild keeps them.

## Why one out-of-sample scheme

The 80/20 chronological split is the only out-of-sample test here, kept
deliberately simple: train once, freeze, predict. With 19-24 test months
per spread a single OOS R² is noisy, which is why `oos.py` also reports a
bootstrap interval — it shows how little any one number can say.

## Why the crash check

2015-01 to 2016-02 (the ChiNext bubble, crash, and circuit-breaker
episode) is the most extreme stretch in the sample. A coefficient that
owes its t-stat to fourteen months of panic is not a rule. `robustness.py`
reruns every main regression without them.
