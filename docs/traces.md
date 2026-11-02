# The pipeline on real traffic: Wikipedia page views

`.github/workflows/traces.yml` runs the retraining pipeline, unchanged, on
hourly user page views of whole Wikipedia projects over the last 30 days
(Wikimedia REST API, converted by `forecast_training.traces`). The series are
real traffic with a daily and a weekly pattern; the three languages peak at
different UTC hours. Settings: 1-hour step, 24-hour window, 6-hour horizon,
last 20% held out, dry run against a throwaway MLflow, `--baseline best`.
Numbers from the run of 2026-11-01 (run 41364145647, data 2 Oct – 31 Oct);
the weekly run keeps newer ones in its artifacts.

Selection is by the asymmetric cost (a missed unit of demand = 3 idle units,
see [training/README.md](../training/README.md#selection-cost-not-rmse));
RMSE and coverage are shown next to it. Units: views per hour.

| Project | Winner | Cost | Coverage | vs persistence | vs seasonal persistence |
|---|---|---|---|---|---|
| en.wikipedia | `ridge_delta` | 332 242 | 0.75 | −87% | −11% |
| de.wikipedia | `hgb_q75_delta_season` | 71 900 | 0.77 | −90% | −24% |
| ja.wikipedia | `hgb_delta` | 81 002 | 0.67 | −89% | −14% |

Full candidate tables, baselines in italics:

**en.wikipedia**

| candidate | holdout cost | coverage | holdout RMSE | holdout MAE | level tracking |
|---|---|---|---|---|---|
| ridge_delta | 332 242 | 0.75 | 316 117 | 222 979 | 0.88 |
| hgb_q75_delta_season | 402 384 | 0.72 | 400 568 | 285 912 | 1.00 |
| hgb_q75_delta | 424 035 | 0.74 | 431 893 | 317 289 | 1.00 |
| hgb_delta_season | 461 162 | 0.58 | 367 686 | 269 126 | 1.00 |
| rf_delta | 484 360 | 0.59 | 393 997 | 293 930 | 1.00 |
| rf_delta_tod | 485 738 | 0.60 | 405 089 | 299 069 | 1.00 |
| hgb_delta | 494 978 | 0.62 | 417 300 | 310 798 | 1.00 |
| hgb_delta_tod | 510 208 | 0.62 | 419 760 | 313 878 | 1.00 |
| rf_delta_season | 520 386 | 0.54 | 389 447 | 289 353 | 1.00 |
| rf_level | 632 749 | 0.65 | 506 434 | 402 878 | 0.00 (ineligible) |
| *persistence* | 2 464 958 | 0.45 | 1 435 547 | 1 227 931 | — |
| *seasonal persistence* | 373 339 | 0.77 | 368 494 | 256 527 | — |

**de.wikipedia**

| candidate | holdout cost | coverage | holdout RMSE | holdout MAE | level tracking |
|---|---|---|---|---|---|
| rf_level | 64 679 | 0.78 | 62 337 | 46 564 | 0.00 (ineligible) |
| hgb_q75_delta_season | 71 900 | 0.77 | 69 719 | 54 199 | 1.00 |
| rf_delta_season | 77 119 | 0.60 | 64 420 | 45 309 | 1.00 |
| hgb_delta_season | 77 890 | 0.62 | 60 638 | 44 016 | 1.00 |
| hgb_delta | 77 994 | 0.65 | 74 057 | 51 589 | 1.00 |
| hgb_delta_tod | 77 997 | 0.65 | 74 281 | 51 734 | 1.00 |
| hgb_q75_delta | 78 576 | 0.78 | 83 471 | 62 630 | 1.00 |
| rf_delta | 79 576 | 0.66 | 74 108 | 52 060 | 1.00 |
| rf_delta_tod | 80 011 | 0.69 | 78 186 | 54 733 | 1.00 |
| ridge_delta | 81 678 | 0.71 | 79 867 | 57 445 | 0.92 |
| *persistence* | 732 556 | 0.34 | 461 256 | 352 507 | — |
| *seasonal persistence* | 94 418 | 0.67 | 97 510 | 66 768 | — |

**ja.wikipedia**

| candidate | holdout cost | coverage | holdout RMSE | holdout MAE | level tracking |
|---|---|---|---|---|---|
| hgb_delta | 81 002 | 0.67 | 72 127 | 53 279 | 1.00 |
| rf_delta | 81 215 | 0.68 | 71 740 | 52 367 | 1.00 |
| hgb_delta_tod | 81 254 | 0.66 | 72 063 | 52 987 | 1.00 |
| rf_delta_tod | 81 481 | 0.67 | 71 420 | 52 521 | 1.00 |
| hgb_q75_delta | 82 476 | 0.76 | 87 270 | 64 252 | 1.00 |
| rf_delta_season | 82 606 | 0.68 | 72 902 | 52 292 | 1.00 |
| hgb_delta_season | 84 518 | 0.65 | 72 901 | 52 347 | 1.00 |
| hgb_q75_delta_season | 87 183 | 0.73 | 85 418 | 62 339 | 1.00 |
| rf_level | 95 070 | 0.78 | 84 473 | 65 664 | 0.00 (ineligible) |
| ridge_delta | 97 184 | 0.68 | 85 291 | 64 442 | 0.93 |
| *persistence* | 712 798 | 0.41 | 458 940 | 365 714 | — |
| *seasonal persistence* | 94 290 | 0.63 | 89 593 | 62 331 | — |

## What it says

- **The honest baseline is "the same hour yesterday", and the models beat
  it by 11–24%, not 80–90%.** Against plain persistence every delta model
  looks spectacular (−87% to −90% cost) because hourly traffic moves a lot
  in six hours. Against seasonal persistence the lead is real but modest.
  This is the number to quote.
- **The quantile and seasonal candidates earn their place on one of three
  series.** On de.wikipedia `hgb_q75_delta_season` wins outright: 24% below
  the seasonal baseline, coverage 0.77, and the next three places are the
  other seasonal candidates. On en.wikipedia the two `q75` models take second
  and third behind the linear model. On ja.wikipedia all the tree models are
  within 2% of each other and the plain `hgb_delta` wins by a hair; the
  seasonal lags add nothing when the 24-hour window already spans the cycle.
- **The quantile models deliver the headroom they promise.** `hgb_q75_*`
  coverage is 0.72–0.78 on every series (0.75 is the quantile they
  estimate), against 0.58–0.68 for the mean models, at an RMSE 10–20%
  higher. Under RMSE they would never be chosen; under the cost they are
  competitive or winning. That is the point of changing the metric.
- **The level-tracking gate still earns its place.** On de.wikipedia the
  notebook's kind of model, `rf_level`, has the best cost and is still
  rejected: its level tracking is 0.00, so above the training range it would
  forecast less traffic than is arriving. The gate costs 11% here.
- **The best family depends on the series.** Linear `ridge_delta` wins the
  smooth English aggregate by a wide margin, trees elsewhere; this is why
  the pipeline keeps several candidates instead of one model.
- **Time-of-day features add nothing here**, as before: `_tod` and plain
  variants are within ±1% on every project. A 24-sample hourly window
  already spans a whole day. The phase matters when the window is short
  relative to the cycle, the autoscaling setting (ten 5-minute samples
  against a 24-hour day), which the synthetic test and the benchmark cover.

The previous run (2026-10-31, RMSE as the metric, persistence as the only
baseline) had the same three winners for en and ja and `hgb_delta` for de;
its headline "76–85% better than persistence" is the number the first bullet
corrects.

## Reproduce

```bash
pip install -r training/requirements.txt
PYTHONPATH=training python -m forecast_training.traces wikipedia --project de.wikipedia --days 30 --out views.json
MLFLOW_TRACKING_URI=sqlite:///mlflow.db PYTHONPATH=training python -m forecast_training \
  --export views.json --step-seconds 3600 --time-steps 24 --horizon 6 --period-seconds 86400 \
  --baseline best --dry-run
```
