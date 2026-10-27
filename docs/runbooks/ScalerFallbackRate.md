# ScalerFallbackRate

**Severity:** warning · **Source:** predictive-scaler (`predictive_scaler_decisions_total`)

## What it means

More than half of the scaling decisions for a ScaledObject over the last
15 minutes used **observed** demand (`source="observed"`) instead of a
forecast. The workload is being scaled reactively — exactly like a plain HPA —
so nothing is broken for the workload, but the predictive part of the system
is not doing its job.

The scaler falls back to observed demand when:

| Cause | Metric that shows it |
|---|---|
| The history window has fewer than `timeSteps` samples (after a restart, a new ScaledObject, or a Prometheus gap) | `predictive_scaler_decisions_total{source="observed"}` only |
| forecast-serving is unreachable, not ready (503) or answers garbage | `predictive_scaler_forecast_failures_total{reason="unavailable"\|"not_ready"\|"bad_response"}` |

## How to check

1. **Predictive scaling** dashboard → *Forecast failures by reason*. No
   failures means the window is still filling: with the default
   `timeSteps: 10` and `stepSeconds: 300` that takes 50 minutes after the
   workload first appears in Prometheus.
2. `reason="not_ready"`: forecast-serving has no model loaded —
   `kubectl get pods`, `curl /ready`, `curl /model-info`; see
   [ForecastReloadErrorRate](ForecastReloadErrorRate.md).
3. `reason="unavailable"`: DNS, NetworkPolicy or the service is down —
   check `forecastServiceURL` in the ScaledObject trigger and
   [ForecastServiceDown](ForecastServiceDown.md).
4. `reason="bad_response"`: usually a window-length mismatch — the
   ScaledObject's `timeSteps` differs from the model's `time_steps`
   (`GET /model-info`), so `/predict` returns 422. The scaler logs the exact
   error: `kubectl -n predictive logs deploy/predictive-scaler`.
5. Prometheus gaps shorten the window: look for holes in the workload's
   series over the last `timeSteps × stepSeconds`.

## How to fix

- Window filling after a restart: nothing to do; the alert clears on its own.
- Load a model / fix the model: `POST /reload` once MLflow has a champion.
- Mismatched `timeSteps`/`stepSeconds`: edit the ScaledObject trigger
  metadata to match the model; the scaler picks it up on the next call.
- Persistent scrape gaps: the scaler only uses the trailing run of
  consecutive samples, so a hole every few minutes keeps the window short —
  fix the scrape target (pod annotations, Prometheus pod discovery); see
  [ScalerPrometheusErrors](ScalerPrometheusErrors.md).
