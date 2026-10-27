# Architecture: Predictive Autoscaling System

**Document type:** Internal design document  
**Scope:** System design, data flow, constraints, trade-offs  
**Status:** v1.0 — functionally complete prototype

---

## 1. System Context

The system turns a stream of workload metrics into **scaling decisions made ahead of load**. It sits between the observability layer (Prometheus, where demand is measured) and the control layer (KEDA / HPA, where replicas are set), and it is split into three loops that run on different time scales:

| Loop | Period | Components |
|---|---|---|
| Scaling | seconds (KEDA polling) | predictive-scaler → forecast-serving → KEDA → HPA |
| Model quality | minutes (one forecast horizon) | forecast-serving scores its own past forecasts; Prometheus alerts |
| Retraining | daily (CronJob) | forecast-training → MLflow registry → `/reload` |

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│ Kubernetes namespace: predictive (helm/predictive-stack)                            │
│                                                                                     │
│   workload pods ──/metrics──► Prometheus ◄── scrapes ── forecast-serving, scaler    │
│   (cpu-burner)                   │  ▲                       │                       │
│        ▲                         │  │ rules: SLOs, drift,   │ forecast_rolling_mae  │
│        │ replicas                │  │ fallbacks → runbooks  │ forecast_model_info   │
│        │                         │  └────────────────────── Grafana (3 dashboards)   │
│       HPA ◄── KEDA ◄─gRPC─ predictive-scaler ──query_range──┘                       │
│     (managed       GetMetrics      │                                                │
│      by KEDA)                      │ POST /predict {values, series, timestamp}      │
│                                    ▼                                                │
│                            forecast-serving ◄──── models:/cpu-forecasting@champion  │
│                              │  ▲ POST /reload              │                       │
│                              │  │                           ▼                       │
│                              │  └──────── forecast-training (CronJob, nightly)      │
│                              │                  │  candidates → gates → compare     │
│                              └── /metrics       └────────────► MLflow registry      │
└─────────────────────────────────────────────────────────────────────────────────────┘
        ▲ Ingress: grafana | prometheus | mlflow | forecast .<domain>
```

The forecast service itself knows nothing about replicas or clusters. It is a stateless function from a window of recent observations to a horizon of future values, plus a bounded memory of its own forecasts so it can report how wrong they turned out to be.

---

## 2. Data Flow

### Scaling path (every KEDA poll, ~15 s)

```
KEDA ──GetMetrics(ScaledObjectRef + trigger metadata)──► predictive-scaler
  1. Prometheus query_range: last timeSteps samples of `query`, stepSeconds apart
  2. model input = demand / capacity          (workload units → model units)
  3. POST /predict {values, series: ns/name, timestamp, step_seconds}
  4. predicted = (max over horizon | next step) × capacity
  5. reported  = max(predicted, observed)     (mode=max) | predicted (mode=forecast)
  6. cache for cacheSeconds
KEDA ◄── metricValue = reported ──► HPA: replicas = ceil(reported / targetValue)
```

The scaler never writes to the Kubernetes API; KEDA and the HPA own the replica count, min/max bounds and scale-down stabilisation.

### Model quality path (inside forecast-serving)

Each `/predict` window carries the truth about earlier calls: when a window arrives whose samples cover the target instant of a forecast made earlier (`timestamp + k × step_seconds`, matched within a quarter step), the forecast for horizon step *k* is scored once. `forecast_abs_error{horizon_step}` and `forecast_rolling_mae{horizon_step}` need no external store; memory is bounded by series × pending forecasts × rolling window. `series` keeps different workloads' windows apart.

### Retraining path (nightly CronJob, or the bootstrap Job on a fresh cluster)

```
Prometheus (TRAIN_DAYS of `query`) ─► regularise onto the step grid, fill gaps by time of day
   ─► fit candidates on the chronological training split
   ─► holdout RMSE + level-tracking score per candidate
   ─► eligible = tracking ≥ 0.5 and beats persistence
   ─► compare the best with @champion on windows after both the split and the champion's data_end_ts
   ─► register version (model/, scaler/, forecast_meta.json), tag data_end_ts, move @champion
   ─► POST /reload on forecast-serving
```

### Deployment path

CI builds the four images on every PR and publishes them to GHCR on `main`. `scripts/up.sh` creates a kind cluster with ingress-nginx and metrics-server, installs `helm/predictive-stack` (KEDA as a chart dependency, forecast-serving and predictive-scaler as file subcharts), trains the first model in-cluster from the saved export and smoke-tests the REST and gRPC APIs. The e2e workflow runs the same script and then ten minutes of load with assertions; the benchmark workflow runs two full compressed days.

---

## 3. Components and Contracts

| Component | Owns | Contract |
|---|---|---|
| `training/forecast_training` | candidates, gates, registry promotion | registered version = `model/` (sklearn, pickled with the pinned scikit-learn) + `scaler/scaler.pkl` (MinMaxScaler) + `forecast_meta.json` (`time_steps`, `horizon`, `step_seconds`, `features`, `target`); tag `data_end_ts` |
| `serving/app.py` | inference, hot reload, quality tracking | `POST /predict {values[time_steps], series?, timestamp?, step_seconds?}` → `{prediction, predictions[horizon], step_seconds, model, stage, model_version}`; `/ready` 503 until a model is loaded; legacy notebook models (no meta, `Production` stage) still load |
| `scaler/predictive_scaler` | KEDA ExternalScaler gRPC, decision policy | trigger metadata validated per call (`INVALID_ARGUMENT` on errors); failure policy: forecast errors → observed demand, Prometheus errors → `UNAVAILABLE` → KEDA `fallback` |
| `monitoring/` | rules, dashboards, runbooks | `render.py` validates dashboards against the metric names the code exports, checks every alert links a runbook, and syncs copies into the umbrella chart |
| `helm/predictive-stack` | the whole system in one release | `demo.scaling=none\|predictive\|hpa`; `training.bootstrap.enabled`; `global.imageTag` |
| `bench/` | predictive vs reactive measurement | `load.py` (compressed day, `bench_*` metrics), `collect.py`, `report.py` → `docs/benchmark.md` |

`tests/test_training_serving.py` is the contract test between training and serving: a model the pipeline registers must serve exactly the forecast the pipeline computed.

### Metrics the system exports

| Metric | From | Meaning |
|---|---|---|
| `predictive_scaler_{observed,predicted,reported}_demand`, `_desired_replicas` | scaler | the decision, per ScaledObject |
| `predictive_scaler_decisions_total{source}` | scaler | forecast vs reactive fallback |
| `predictive_scaler_forecast_failures_total{reason}`, `_prometheus_failures_total` | scaler | why a fallback happened |
| `forecast_rolling_mae{horizon_step}`, `forecast_abs_error{horizon_step}` | serving | forecast quality |
| `forecast_model_info{model_version,...}`, `forecast_model_loaded`, `forecast_model_reloads_total` | serving | which model, and whether reloads work |
| `bench_latency_seconds`, `bench_requests_total`, `bench_clients`, `bench_sim_hour` | load generator | what clients experienced during a benchmark |

---

## 4. Design Decisions

| Decision | Alternatives | Why |
|---|---|---|
| **KEDA external scaler** (gRPC) as the integration point | custom metrics adapter (`custom.metrics.k8s.io`); a controller that patches `replicas`; `/recommend-scale` verdicts | KEDA already owns the HPA lifecycle, min/max, fallback and scale-down policy; the scaler stays a stateless function of trigger metadata, one deployment serves any number of workloads, and KEDA's `fallback` gives a principled answer to "no data". A metrics adapter needs cluster-wide API aggregation for the same result; a replica-patching controller re-implements the HPA. |
| **Report `max(forecast, observed)`** (`mode=max`) by default | trust the forecast alone | A wrong forecast can then only cost money (early capacity), never availability (scaling below current use). `mode=forecast` exists for workloads where scale-down speed matters more. |
| **Delta target with anchored features** (`hgb_delta`, `rf_delta`) | the notebook's level-target Random Forest (`rf_level`) | Tree ensembles average training targets and cannot predict above the highest level they saw. On a ramp past the training range the notebook's model forecast *within* the range (0.56 on a ramp to 3 cores): for an autoscaler that means scaling down into a spike. Predicting the change from the last value, with the window expressed relative to it, follows new levels one-for-one at a ~3% RMSE cost. |
| **Level-tracking gate** before RMSE | select by holdout RMSE only | RMSE on in-range data rewards exactly the pinned behaviour that is unsafe out of range; the gate (`mean(f(X+2m) − f(X+m)) / m ≥ 0.5`) measures the failure mode directly. |
| **Compare with the champion only on data neither model has seen** (`data_end_ts` tag, `min_fresh_windows`) | compare on the current holdout | A model always looks best on data it was fitted to; without the rule the nightly job would replace a good champion with a model that merely saw yesterday. |
| **Registry aliases** (`@champion`) instead of stages | `Production` stage | Stages are deprecated in MLflow; an alias is a single pointer the pipeline moves atomically and serving resolves on `/reload`. The stage is still honoured as a fallback so notebook-registered models keep working. |
| **Forecast quality measured in serving, from the windows it already receives** | a separate feedback service; a Prometheus recording rule joining forecasts with actuals | The information is already in the request stream; no new store, no new deployment, no clock skew between systems. The cost is per-request bookkeeping and the need for callers to pass `series`/`timestamp` for exact matching. |
| **Dashboards and rules as JSON/YAML, validated and rendered by a script** | hand-edited ConfigMaps; Grafana as the source of truth | The validator catches a renamed metric before Grafana shows "No data"; the same files feed the raw manifests and the Helm chart. |
| **One umbrella chart with KEDA as a dependency** | separate installs per component; kustomize | One command, one namespace, namespace-scoped RBAC, pinned versions. The only wrinkle — a `ScaledObject` cannot be in the same install as the CRDs that define it — is handled by `demo.scaling` and a two-phase `up.sh`. |
| **kind + ingress-nginx** for local and CI | Minikube with NodePorts and a hard-coded node IP | The same script runs on a laptop and in GitHub Actions; `*.localhost` hosts need no DNS. |
| **Classical models on lag features** | LSTM/GRU | On a few thousand points of 5-minute data, tree models with a sliding window generalise better and train in seconds; the notebook measured it. |

---

## 5. Constraints

- **Window and spacing are the model's.** Callers must send exactly `time_steps` values at `step_seconds` spacing (both in `GET /model-info`); the scaler's `timeSteps`/`stepSeconds` must match. Time can be compressed (the demo and benchmark use 30-second samples with a model trained on 30-second samples, or on 5-minute samples for a quick look), but then the horizon is compressed too.
- **Lead time = horizon × step.** Six steps of 5 minutes give 30 minutes with `forecastAggregation: max`; the forecast cannot see further than that, and a step-change the window has not started to show is not forecastable.
- **Units.** The model works in its training units; `capacity` in the trigger converts workload demand into them. A wrong `capacity` makes forecasts and the drift threshold meaningless.
- **One replica of everything** in the chart: MLflow on SQLite, Prometheus on `emptyDir`, single serving and scaler pods. Correct for a prototype and CI; production needs external stores and replicas.
- **Latency.** Inference is sub-millisecond; the scaler's per-request timeout (3 s) and cache (15 s) bound the control loop's cost.

---

## 6. Failure Modes

| Failure | Behaviour | Signal | Runbook |
|---|---|---|---|
| forecast-serving down / not ready / bad answer | scaler reports observed demand (reactive scaling continues) | `predictive_scaler_forecast_failures_total{reason}`, `ScalerFallbackRate` | `docs/runbooks/ScalerFallbackRate.md` |
| Prometheus down, query empty or multi-series | scaler returns `UNAVAILABLE`; KEDA applies `fallback.replicas` | `predictive_scaler_prometheus_failures_total`, `ScalerPrometheusErrors` | `ScalerPrometheusErrors.md` |
| Model stopped tracking the workload (drift) | forecasts still served; `mode=max` protects availability | `forecast_rolling_mae`, `ForecastDriftHigh` | `ForecastDriftHigh.md` |
| MLflow down | serving keeps the model in memory; `/reload` fails; retraining job fails | `ForecastReloadErrorRate`, Job status | `ForecastReloadErrorRate.md` |
| No champion yet (fresh cluster) | `/ready` 503; scaler falls back to observed demand | `forecast_model_loaded == 0` | `ForecastReloadErrorRate.md` |
| Retraining finds a worse model | champion kept; decision and reason logged to MLflow and stdout | run tags `decision`, `decision_reason` | — |
| Scrape gap on the workload | short window → observed demand until it refills | `decisions_total{source="observed"}` | `ScalerFallbackRate.md` |

---

## 7. What the Benchmark Measures

Two runs of the same compressed day on the same cluster — a CPU HPA at 70% and the predictive scaler with `mode=max`, `forecastAggregation=max` — compared on p95 latency during the ramps, the share of slow requests, replica-minutes and time to first scale-up. The method and caveats are in [bench/README.md](bench/README.md); the latest numbers in [docs/benchmark.md](docs/benchmark.md).

The honest frame: on a periodic pattern the forecast can only help at the ramps, and only if the window already shows the ramp starting. On flat load the two scalers behave identically; on an unprecedented spike the delta model follows but cannot anticipate.

---

## 8. Future Improvements

- **Alert-driven retraining:** `ForecastDriftHigh` → Alertmanager webhook → `kubectl create job --from=cronjob/forecast-retrain`.
- **Time-of-day features** once more than a few days of real data exist, so the model anticipates the morning ramp rather than extrapolating its start.
- **Additional signals** (memory, queue depth) as features for workloads where CPU is not the bottleneck.
- **Multi-tenant scaler configuration** via a CRD instead of trigger metadata.
- **Production stores:** PostgreSQL and object storage behind MLflow, a long-term metrics backend behind Prometheus.
