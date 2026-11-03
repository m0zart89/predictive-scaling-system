# ForecastReloadErrorRate

**Severity:** warning · **Source:** forecast-serving (`forecast_model_reloads_total{result="failure"}`)

## What it means

A forecast-serving pod's lifecycle worker has been failing to load the model
from MLflow for 15 minutes. The worker runs in every replica: without a
model it retries the registry every `MODEL_RETRY_SECONDS`; with one it polls
the `@champion` alias every `MODEL_REFRESH_SECONDS` and loads the new
version when the alias moved. A failure never unloads a working model: if
the pod had one it keeps serving it (`forecast_model_loaded == 1`), if not it
stays unready (`/readyz` 503) and the scaler falls back to observed demand.

There is no reload endpoint any more; nothing pushes models into the pods.

## How to check

1. The pod's log says why: `kubectl logs deploy/forecast-serving | grep -i "registry\|model"` —
   `registry ... unavailable`, `alias @champion ... not found`, `no version ...
   in stage`, or an artifact download error.
2. MLflow: `kubectl get pods -l app=mlflow`, `curl http://mlflow:5000/health`;
   the registry must contain model `cpu-forecasting` with alias `champion`
   (or a `Production` stage for notebook-registered models).
3. The version's artifacts: `model/`, `scaler/scaler.pkl` (and
   `scaler/channels.pkl` for multivariate models), `forecast_meta.json` — a
   version registered without the scaler artifact cannot be served.
4. `MLFLOW_TRACKING_URI` in the serving pod matches the MLflow Service; with
   `mlflow.auth` on, the pod has `MLFLOW_TRACKING_USERNAME`/`_PASSWORD`.
5. `/readyz` and `/model-info` on the pod: which version it serves and
   whether it came from the cache (`model_stage: cache`).

## How to fix

- No champion yet: run training (`scripts/up.sh` does it on a fresh cluster;
  or `kubectl create job --from=cronjob/forecast-retrain retrain-now`). The
  pods pick the version up within `MODEL_RETRY_SECONDS`.
- MLflow down: restore it (database, object store, PVC, OOM); the alert
  clears on the next successful load.
- Broken version: move the alias to a good version; every replica swaps on
  its next poll.
