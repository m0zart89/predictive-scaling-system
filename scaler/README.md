# predictive-scaler

A [KEDA external scaler](https://keda.sh/docs/latest/concepts/external-scalers/)
that scales a workload on **forecast** demand instead of current demand.

For each `ScaledObject` it reads the last `timeSteps` samples of a PromQL
query, sends them to `forecast-serving`'s `POST /predict`, and reports the
result to KEDA as an external metric. KEDA's HPA then sets
`replicas = ceil(reported / targetValue)`.

## Trigger metadata

| Key | Required | Default | Meaning |
|---|---|---|---|
| `scalerAddress` | yes | — | `predictive-scaler.<ns>.svc.cluster.local:6000` (read by KEDA) |
| `prometheusServerAddress` | yes | — | Prometheus base URL |
| `query` | yes | — | PromQL returning **one** series of total demand (aggregate with `sum`) |
| `forecastServiceURL` | yes | — | forecast-serving base URL |
| `targetValue` | yes | — | demand one replica should carry, in the query's unit |
| `capacity` | no | `1` | model input = demand / capacity; pick it so the workload's range fits the model's training range |
| `timeSteps` | no | `10` | samples per forecast; must match the model's `TIME_STEPS` |
| `stepSeconds` | no | `300` | spacing of samples; the model was trained on 5-minute data |
| `mode` | no | `max` | `max`: report max(forecast, observed). `forecast`: report the forecast alone |
| `forecastAggregation` | no | `next` | `next`: first forecast step. `max`: peak over the model's horizon (more lead time) |
| `activationValue` | no | `0` | `IsActive` is true when reported demand exceeds this |
| `cacheSeconds` | no | `15` | reuse a decision for this long (KEDA polls often) |
| `timeoutSeconds` | no | `3` | per-request timeout for Prometheus and forecast-serving |
| `metricName` | no | `predicted-demand` | external metric name |

Invalid metadata is rejected with gRPC `INVALID_ARGUMENT`, which KEDA shows
as an event on the `ScaledObject`.

Each `/predict` call also carries `series` (`<namespace>/<scaledobject>`),
`timestamp` (the instant the window was read at) and `stepSeconds`, so forecast-serving can
score the forecast against the actual value a later window brings
(`forecast_rolling_mae` by horizon step).

## Failure policy

| Condition | Reported value |
|---|---|
| forecast-serving unreachable, not ready (503) or bad answer | observed demand |
| fewer than `timeSteps` samples | observed demand |
| Prometheus error, or query returns no samples | gRPC `UNAVAILABLE` → KEDA `fallback` |

## Metrics (`:9100/metrics`)

All per-ScaledObject series carry `namespace` and `scaledobject` labels.

- `predictive_scaler_observed_demand`, `predictive_scaler_predicted_demand`,
  `predictive_scaler_reported_demand`, `predictive_scaler_desired_replicas`
- `predictive_scaler_decisions_total{source="forecast|observed"}`
- `predictive_scaler_forecast_failures_total{reason="unavailable|not_ready|bad_response"}`
- `predictive_scaler_prometheus_failures_total`
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
