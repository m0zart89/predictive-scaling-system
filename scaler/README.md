# predictive-scaler

A [KEDA external scaler](https://keda.sh/docs/latest/concepts/external-scalers/)
that scales a workload on **forecast** demand instead of current demand.

For each `ScaledObject` it reads the last `historySteps` samples of a PromQL
query (at least the `timeSteps` window), sends them to `forecast-serving`'s
`POST /predict`, and reports the result to KEDA as an external metric. KEDA's HPA then sets
`replicas = ceil(reported / targetValue)`.

## Trigger metadata

| Key | Required | Default | Meaning |
|---|---|---|---|
| `scalerAddress` | yes | — | `predictive-scaler.<ns>.svc.cluster.local:6000` (read by KEDA) |
| `prometheusServerAddress` | yes | — | Prometheus base URL |
| `query` | yes | — | PromQL returning **one** series of total demand (aggregate with `sum`) |
| `forecastServiceURL` | yes | — | forecast-serving base URL |
| `targetValue` | yes | — | demand one replica should carry, in the query's unit |
| `capacity` | no | `1` | reference capacity in cores: model input = demand / capacity, forecast demand = coefficient × capacity. The chart's CronJob trains on the same series divided by the same number and records it in the model (`capacity_reference`) |
| `capacityQuery` | no | — | read the reference from Prometheus instead (an instant query for the workload's envelope in cores, e.g. `max(kube_pod_container_resource_limits{container="api", resource="cpu"}) * 20`); on any failure the last good value, then `capacity`, is used and logged once (`event=capacity_fallback`). Keep it the fixed envelope, never the live replica total |
| `capacityRefreshSeconds` | no | `300` | how long a value from `capacityQuery` is reused |
| `capacityDriftFactor` | no | `2` | a reference more than this factor away from the `capacity_reference` the served model was trained with is logged (`event=capacity_drift`) and exported as `predictive_scaler_capacity_drift` (`0` = no check) |
| `timeSteps` | no | `10` | the window: consecutive samples a forecast needs; must match `time_steps` in forecast-serving's `GET /model-info` |
| `historySteps` | no | `timeSteps` | samples fetched and sent with each forecast, ≥ `timeSteps`. Models with seasonal features need a cycle more (`input_length` in `/model-info`, e.g. 10 + 288 for a day of 5-minute samples); the umbrella chart sets it from the training period. Samples older than the window may be missing (sent as `null`; serving interpolates) |
| `stepSeconds` | no | `300` | spacing of samples; must match the model's `step_seconds` (`/model-info`), 300 for models trained on 5-minute data |
| `mode` | no | `max` | `max`: report max(forecast, observed). `forecast`: report the forecast alone |
| `forecastAggregation` | no | `next` | `next`: first forecast step. `max`: peak over the model's horizon (more lead time) |
| `activationValue` | no | `0` | `IsActive` is true when reported demand exceeds this |
| `cacheSeconds` | no | `15` | reuse a decision for this long (KEDA polls often) |
| `timeoutSeconds` | no | `3` | per-request timeout for Prometheus |
| `forecastTimeoutSeconds` | no | `0.2` | timeout for the `POST /predict` call; a slow answer is a failure (reactive decision, counts against the service breaker) |
| `forecastBreakerFailures` | no | `3` | consecutive failed forecast calls (timeout, 5xx, network, unusable body) that open the service breaker (`0` = never) |
| `forecastBreakerOpenSeconds` | no | `30` | how long the service is not called once its breaker is open; then one call probes it |
| `smoothingAlpha` | no | `0.3` | EWMA on the forecast demand, S = α·Y + (1−α)·S_prev, before it meets observed demand; `1` = raw forecast |
| `smoothingMaxAgeSeconds` | no | `600` | an EWMA state not updated for this long is seeded again from the next forecast (`0` = never) |
| `spikeMinSteps` | no | `3` | hysteresis gate: a forecast that needs more replicas than observed demand must do so for this many consecutive horizon steps, else it is noise and the decision is observed demand (`0`/`1` = off; inactive on horizons shorter than this) |
| `metricName` | no | `predicted-demand` | external metric name |
| `forecastSurgeFactor` | no | `3` | `mode=max` guard: the forecast may exceed observed demand by at most this factor (`0` = no cap) |
| `breaker` | no | `on` | circuit breaker (below): `off` to always use the forecast |
| `breakerResidualZ` | no | `4` | residual signal: robust z-score of (actual − forecast made for now) that trips on two consecutive polls (twice it trips at once) |
| `breakerLevelShift` | no | `2.5` | level signal: observed / value one season ago above this (or below its inverse) for `breakerCalmPolls` polls |
| `breakerJumpFactor` | no | `2` | jump signal: a one-step rise larger than this × the largest rise in the fetched history |
| `breakerJumpMinSamples` | no | `48` | the jump signal waits until the history holds this many samples (a workload minutes old has no normal to compare with) |
| `breakerMinOpenSeconds` | no | `600` | minimum time the breaker stays open; doubles on a re-trip while half-open (up to 1 h) |
| `breakerCalmPolls` | no | `6` | consecutive calm polls needed to close from half-open |
| `channels` | no | — | multivariate models: JSON object of extra input series, channel name → PromQL, sent next to the demand series (`/model-info` lists the model's channels) |
| `targetChannel` | no | `cpu` | name of the demand series when `channels` is set |

Invalid metadata is rejected with gRPC `INVALID_ARGUMENT`, which KEDA shows
as an event on the `ScaledObject`.

Each `/predict` call also carries `series` (`<namespace>/<scaledobject>`),
`timestamp` (the instant the window was read at) and `stepSeconds`, so forecast-serving can
score the forecast against the actual value a later window brings
(`forecast_rolling_mae` by horizon step).

## Failure policy

| Condition | Reported value |
|---|---|
| forecast-serving unreachable, slow (`forecastTimeoutSeconds`), 5xx, not ready (503) or bad answer | observed demand |
| `forecastBreakerFailures` such failures in a row | observed demand without calling the service, for `forecastBreakerOpenSeconds`; `predictive_scaler_forecast_service_breaker_open` = 1, `predictive_scaler_forecast_skipped_total` counts the polls |
| forecast rise shorter than `spikeMinSteps` consecutive horizon steps | observed demand; `predictive_scaler_spikes_suppressed_total` |
| fewer than `timeSteps` consecutive samples at the end of the history (new workload, scrape gap, NaN) | observed demand |
| gaps further back, inside `historySteps` | forecast as usual; the gaps go to serving as `null` |
| circuit breaker open (the series does not resemble the history: a jump, a level far from a season ago, forecasts wrong by more than the noise) | observed demand; the forecast is still made and scored, `predictive_scaler_breaker_open` = 1 |
| forecast above `forecastSurgeFactor` × observed (`mode=max`) | the capped value |
| forecast used | its EWMA (`smoothingAlpha`), `predictive_scaler_smoothed_demand`; never below observed with `mode=max` |
| Prometheus error, or query returns no samples (any channel) | gRPC `UNAVAILABLE` → KEDA `fallback` |

## Circuit breaker

`predictive_scaler/breaker.py`, one per ScaledObject, in memory. While the
present stops resembling the history the model learned, the forecast is
taken out of the decision until the series has been calm for a while:
closed → open (`breakerMinOpenSeconds`, doubled on a re-trip) → half-open
(`breakerCalmPolls` calm polls) → closed. The open intervals are exported as
`predictive_scaler_breaker_open` so retraining can exclude them
(`forecast_training.trigger` does). Add a plain `cpu` trigger to the
ScaledObject (`scalerDefaults.reactiveFloor` in the chart) so KEDA keeps
scaling on CPU whatever the scaler reports.

## Units: what the forecast numbers are

`/predict` says what its numbers are (`units`, `capacity_reference`;
`docs/relative-delta-alignment.md`). A model trained by the chart's CronJob
answers in `capacity_fraction`: the horizon is a fraction of the capacity its
series was divided by, so the scaler multiplies by its own reference
(`capacity` or `capacityQuery`) to get cores and checks that the two agree
(`predictive_scaler_capacity_drift`). Older models (`scaled`) are multiplied
by the same reference as before. Pod size never reaches the model: a 1-core
and an 8-core variant of a workload differ in `targetValue`, not in the
coefficient.

## Metric conditioning

`predictive_scaler/smoothing.py`, between the forecast and the number
`GetMetrics` returns, all per ScaledObject and in memory:

* **EWMA.** `S_t = α·Y_t + (1−α)·S_{t−1}` on the forecast demand
  (`smoothingAlpha`, 0.3 by default). A sustained rise gets through in a
  few polls (90 % after seven at α = 0.3); a one-poll false peak moves the
  reported value by 30 % of its height instead of all of it, which is what
  keeps the HPA — with its 300 s scale-down stabilisation — from paying five
  minutes of replicas for a 30-second blip. The first forecast seeds the
  state; a state older than `smoothingMaxAgeSeconds` is seeded again.
  `mode=max` still floors the result at observed demand: smoothing delays
  scale-up on a false forecast, never on real load.
* **Hysteresis gate.** Before the forecast is used, the horizon vector is
  checked in replica terms (`ceil(demand / targetValue)`, what KEDA will
  compute): the steps that need more replicas than observed demand must
  form a run of at least `spikeMinSteps` consecutive steps, else the rise is
  noise and the decision rests on observed demand. The forecast is still
  recorded and scored.
* **Forecast-service breaker.** The `POST /predict` call has its own
  `forecastTimeoutSeconds` (200 ms); a timeout, a 5xx, a refused connection
  or an unusable body is a failure and the decision is reactive. After
  `forecastBreakerFailures` failures in a row the breaker opens for
  `forecastBreakerOpenSeconds` and the service is not called at all; one
  call then probes it. KEDA keeps scaling whatever the forecast service does.

Every event is logged as `event=decision|forecast_smoothed|spike_suppressed|
forecast_failed|forecast_service_breaker_open|forecast_service_breaker_closed`
with `namespace=… scaledobject=…` key-value fields.

## Forecast quality, scored here

Every decision fetches a history that holds the actual values earlier
forecasts aimed at, so the scaler scores its own forecasts
(`predictive_scaler/quality.py`): exact matching, per replica, no state
shared between replicas, in model units so it compares with the model's own
`forecast_model_holdout_mae`. `ForecastDriftHigh` reads it.

## Mutual TLS

With `TLS_DIR` set (a mounted cert-manager Secret: `tls.crt`, `tls.key`,
`ca.crt`) the gRPC server requires a client certificate from the same CA
(KEDA's `TriggerAuthentication` with `caCert`/`tlsClientCert`/`tlsClientKey`
and `enableTLS: "true"` on the trigger) and the HTTP client presents the
same certificate to an https `forecastServiceURL`. Kubernetes' grpc probes
carry no certificate, so a plaintext health-only server answers on
`HEALTH_PORT` (6001). Prometheus is always plain HTTP. `global.tls.enabled`
in the umbrella chart sets all of it up.

## Metrics (`:9100/metrics`)

All per-ScaledObject series carry `namespace` and `scaledobject` labels.

- `predictive_scaler_observed_demand`, `predictive_scaler_predicted_demand`,
  `predictive_scaler_reported_demand`, `predictive_scaler_desired_replicas`
- `predictive_scaler_decisions_total{source="forecast|observed"}`
- `predictive_scaler_forecast_failures_total{reason="unavailable|not_ready|bad_response"}`
- `predictive_scaler_prometheus_failures_total`
- `predictive_scaler_forecast_demand{step}` — the forecast per horizon step, query units
- `predictive_scaler_forecast_abs_error{step}` (histogram, model units) and
  `predictive_scaler_forecast_residual` — the scaler's own scoring of its past forecasts
- `predictive_scaler_breaker_open`, `predictive_scaler_breaker_trips_total{reason="residual|level|jump"}`,
  `predictive_scaler_breaker_bypass_total`
- `predictive_scaler_forecast_latency_seconds`
- `predictive_scaler_grpc_requests_total{method,code}`

## Development

```bash
cd scaler
pip install -r requirements-dev.txt
pytest tests/          # unit + gRPC tests, no cluster needed
./gen_proto.sh         # after editing predictive_scaler/externalscaler.proto
python -m predictive_scaler   # gRPC :6000, metrics :9100
```
