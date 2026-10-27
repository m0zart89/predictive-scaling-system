# ForecastReloadErrorRate

**Severity:** warning · **Source:** forecast-serving (`http_requests_total{handler="/reload"}`)

## What it means

`POST /reload` is returning 5xx: the service cannot fetch a model from
MLflow. If a model was already loaded it keeps serving the old one
(`forecast_model_loaded == 1`); if not, `/ready` is 503 and the scaler falls
back to observed demand.

`/reload` is called by the retraining job after it moves the `@champion`
alias, and by `scripts`/operators after a manual promotion.

## How to check

1. The response body says why: `curl -X POST http://<serving>/reload` —
   `Alias @champion not found`, `No model version found in stage`,
   connection refused, or a download error.
2. MLflow: `kubectl get pods -l app=mlflow`, `curl http://mlflow:5000/health`;
   the registry must contain model `cpu-forecasting` with alias `champion`
   (or a `Production` stage for notebook-registered models).
3. The version's artifacts: `model/`, `scaler/`, `forecast_meta.json` — a
   version registered without the scaler artifact cannot be served.
4. `MLFLOW_TRACKING_URI` in the serving pod matches the MLflow Service.

## How to fix

- No champion yet: run training (`python -m forecast_training --export
  notebooks/data-jan-2022.json ...` or the CronJob), then `POST /reload`.
- MLflow down: restore it (PVC, OOM); the alert clears on the next
  successful reload.
- Broken version: move the alias to a good version and reload.
