# ForecastNoTrafficToPredictEndpoint

**Severity:** warning · **Source:** forecast-serving (`http_requests_total`)

## What it means

`POST /predict` has received no requests for 15 minutes. The service may be
fine; the problem is usually upstream: the predictive scaler is not calling
it, because KEDA is not calling the scaler, because no ScaledObject is
active, or because the scaler is stuck in its own fallback.

## How to check

1. Is anything supposed to be calling? `kubectl get scaledobject -A` —
   `READY`/`ACTIVE` columns, and `kubectl describe scaledobject ...` events.
2. Scaler: `predictive_scaler_grpc_requests_total` rising means KEDA is
   calling; `predictive_scaler_decisions_total{source="observed"}` rising
   without `/predict` traffic means the history window is not full (see
   [ScalerFallbackRate](ScalerFallbackRate.md)).
3. Scaler logs for `forecast unavailable`: the scaler may be calling a wrong
   `forecastServiceURL`.
4. Prometheus scraping gap on the serving pod makes `absent(...)` fire too —
   check `up` for the serving target.

## How to fix

- No ScaledObject / KEDA down: restore KEDA or the ScaledObject.
- Wrong `forecastServiceURL`: fix the trigger metadata.
- Expected quiet period (no autoscaled workloads deployed): silence the alert.
