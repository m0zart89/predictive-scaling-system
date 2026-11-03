# ForecastQualityBackendDown

**Severity:** warning · **Source:** forecast-serving (`forecast_quality_backend_up`)

## What it means

forecast-serving keeps the state of its forecast scoring (the forecasts
waiting for their actual values, the rolling error windows per model version
and horizon step) in Redis when `QUALITY_BACKEND=redis`
(`global.quality.backend: redis` in the chart). A pod has not been able to
reach that Redis for 10 minutes. Nothing else is affected: `/predict` keeps
answering (the backend is skipped for a few seconds after each failure, so
the socket timeout is not paid on every request), the scaler keeps scaling.
What stops is the quality signal: `forecast_rolling_mae` is not updated,
`ForecastDriftHigh` cannot fire from it, and forecasts made meanwhile are
never scored (they are not kept anywhere else).

The scaler's own scoring (`predictive_scaler_forecast_abs_error`) does not
depend on Redis and keeps working.

## How to check

1. The serving log names the error once per outage:
   `kubectl logs deploy/forecast-serving | grep "quality backend"` —
   `connection refused`, `timeout`, `NOAUTH`, `OOM command not allowed`.
2. The bundled Redis: `kubectl get pods -l app=redis`, `kubectl logs deploy/redis`;
   `kubectl exec deploy/redis -- redis-cli ping` and `info memory`
   (`maxmemory` is 64 MB by default; `volatile-ttl` evicts rather than refuses,
   so `OOM` means a non-expiring key was written by something else).
3. `REDIS_URL` in the serving pod (`kubectl get secret forecast-serving -o yaml`)
   matches the Service (`redis:6379` for the bundled one) and carries the
   password when the Redis requires one.
4. A NetworkPolicy may be in the way: the `redis` policy admits
   `app.kubernetes.io/name=forecast-serving` only.
5. `forecast_quality_backend_errors_total` per pod tells whether it is one
   replica (a node network problem) or all of them (Redis itself).

## How to fix

- Redis down or OOM-killed: let it restart; the state it held had a TTL of a
  day at most, so nothing needs restoring. The alert clears on the first
  call that succeeds.
- Wrong URL or password: fix `global.quality.redisUrl` and roll
  forecast-serving.
- Redis is not wanted: set `global.quality.backend: memory` (per-pod scoring,
  exact with one replica) and roll forecast-serving.
