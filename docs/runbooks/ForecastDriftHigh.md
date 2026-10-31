# ForecastDriftHigh

**Severity:** warning · **Source:** forecast-serving (`forecast_rolling_mae`)

## What it means

The forecast service compares every forecast it made with the value that
actually arrived in a later `/predict` window and keeps a rolling mean absolute
error per horizon step (`forecast_rolling_mae{horizon_step}`, over the last
`FORECAST_MAE_WINDOW` = 100 resolved forecasts). This alert fires when any step
has stayed above **twice the holdout MAE the loaded model achieved in training**
(`forecast_model_holdout_mae{horizon_step}`, from the registry version's tags)
for 30 minutes. A model without that tag, such as one registered by the
notebook, is held to a fixed 0.25 (model units: demand divided by the
ScaledObject's `capacity`).

The model is no longer tracking the workload. The scaler keeps working — in
`mode: max` it still never reports less than observed demand — but the lead
time the forecast was supposed to buy is gone, and in `mode: forecast` the
workload may be under-provisioned.

## How to check

1. Open the **Model quality** dashboard: which horizon steps are high, since
   when, and whether the error is growing or stepped up at a reload
   (*Loaded model* panel, `forecast_model_reloads_total`).
2. Compare with the champion's holdout error in MLflow
   (`holdout_rmse_step_<k>` on the registered version). A rolling MAE ~2x the
   holdout RMSE is drift; a rolling MAE close to it means the threshold is
   simply too tight for this workload.
3. Check whether the *workload* changed rather than the model: the
   **Predictive scaling** dashboard shows observed demand; a new daily
   pattern, a new level (deployment, traffic migration) or a changed
   `capacity` in the ScaledObject all show up as error.
4. Confirm the windows are consistent: `stepSeconds` in the ScaledObject must
   match the model's `step_seconds` (`GET /model-info`), otherwise forecasts
   are matched to the wrong instants and the error is meaningless.

## How to fix

- **Workload changed, model stale:** retrain now instead of waiting for the
  nightly CronJob: `kubectl -n predictive create job --from=cronjob/forecast-retrain
  retrain-manual`. The pipeline only moves `@champion` to a model that beats
  the current one on data neither has seen, then calls `/reload`.
- **A reload made it worse:** roll back by pointing the alias at the previous
  version (`mlflow` UI or `MlflowClient().set_registered_model_alias(...)`)
  and `POST /reload`.
- **Threshold wrong for this workload:** the factor 2 in
  `monitoring/prometheus-rules.yaml` (`ForecastDriftHigh`) is the only knob;
  the baseline itself comes from each model's own training run.
- **Not enough signal yet:** the gauge is empty until the scaler's history
  window has filled and at least one horizon has elapsed; if the alert fired
  right after a restart with few resolved forecasts
  (`forecast_abs_error_count`), wait for the window to fill.
