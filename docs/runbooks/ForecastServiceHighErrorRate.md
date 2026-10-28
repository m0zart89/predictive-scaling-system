# ForecastServiceHighErrorRate

**Severity:** warning · **Source:** forecast-serving (`http_requests_total`)

## What it means

More than 0.5% of HTTP requests to forecast-serving returned 5xx over the
last 5 minutes. The scaler treats a failed `/predict` as "no forecast" and
scales on observed demand, so the workload keeps scaling reactively — but
every failed call is a lost forecast, and a sustained error rate usually means
the model or MLflow is broken rather than a transient blip.

## How to check

1. Which endpoint: `sum by (handler, status) (rate(http_requests_total{status=~"5.."}[5m]))`
   (`status` is the class, `5xx`).
   - `/predict` 500: inference error — the model raised, or returned a
     non-finite value (`forecast_prediction_failures_total{reason="inference"}`).
   - `/reload` 500: MLflow unreachable or no champion — see
     [ForecastReloadErrorRate](ForecastReloadErrorRate.md).
2. Pod logs: `kubectl logs deploy/forecast-serving` — inference errors are
   logged with the input window.
3. `GET /model-info`: does `time_steps` match what callers send? A mismatch
   gives 422, not 5xx, but a model whose scaler artifact is missing or
   corrupt fails at inference.

## How to fix

- Inference errors after a reload: roll the `@champion` alias back to the
  previous version and `POST /reload`.
- Errors from a bad artifact: re-run training
  (`kubectl -n predictive create job --from=cronjob/forecast-retrain retrain-manual`)
  to register a fresh version.
- Transient (pod restart during a rollout): the alert clears after 2 minutes
  of clean traffic.
