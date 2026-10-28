# ForecastPredictLatencyHigh

**Severity:** warning · **Source:** forecast-serving (`http_request_duration_seconds`)

## What it means

p99 latency of `POST /predict` has been above 200 ms for 5 minutes. The
scaler's per-request timeout is `timeoutSeconds` (default 3 s), so this is
not yet causing fallbacks, but inference on a 10-value window should take a
few milliseconds; something is contending for the pod.

## How to check

1. **Forecast serving** dashboard: is latency high on every endpoint (pod
   starved) or only on `/predict` (model)?
2. `kubectl top pod` against the pod's CPU limit — a throttled pod shows
   latency on everything.
3. Did it start at a reload (`forecast_model_reloads_total`)? A much larger
   model (more trees, a longer horizon) is slower.
4. Request rate: many ScaledObjects polling with `cacheSeconds: "0"` can
   overload a single replica.

## How to fix

- Raise the CPU limit or add replicas (`replicaCount` in
  `helm/forecast-serving/values.yaml`); the service is stateless.
- Restore the scaler's `cacheSeconds` (default 15) if it was lowered.
- Roll back the model if the regression coincides with a reload.
