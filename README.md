# AI infrastructure prototype for predictive autoscaling and workload-aware resource optimization

**Status:** Prototype / Portfolio  
**Stack:** Python · FastAPI · gRPC · MLflow · scikit-learn · Kubernetes · KEDA · Prometheus · Grafana

---

## Why this matters

As workloads become more compute-constrained and expensive, reactive scaling leads to latency spikes and inefficient resource usage.

This system demonstrates how predictive signals can be used to improve both performance and cost efficiency.

---

## System Overview

Kubernetes autoscalers (HPA, KEDA) are reactive: they see a metric cross a threshold and then add pods, so for workloads with a predictable pattern — business-hours traffic, scheduled batches — scaling always lags the load by at least one scrape-and-react cycle. The result is latency at every ramp-up and idle capacity during every predictable quiet period.

This project closes that gap end to end: a forecasting service turns recent observations into a multi-step forecast, a KEDA external scaler turns the forecast into a replica count, a retraining pipeline keeps the model honest, and the forecast's own error is measured in production. Everything deploys with one Helm chart and one script on a kind cluster, and a benchmark workflow measures what the forecast buys against a plain CPU HPA.

---

## Architecture

```
┌───────────────────────────────────────────────────────────────────────────────┐
│ namespace predictive (helm/predictive-stack)                                  │
│                                                                               │
│  cpu-burner pods ──/metrics──► Prometheus ◄──── scrapes ─── forecast-serving   │
│       ▲                          │   ▲                            │  scaler    │
│       │ replicas                 │   │ rules → alerts → runbooks  │            │
│      HPA ◄── KEDA ◄── gRPC ── predictive-scaler ◄── query_range ──┘            │
│   (managed by KEDA)   GetMetrics      │                                       │
│                                       │ POST /predict {values, series, ts}    │
│                                       ▼                                       │
│                               forecast-serving ◄── models:/cpu-forecasting@champion
│                                 │ ▲ POST /reload                │             │
│                       /metrics  │ └── forecast-training (CronJob) ─► MLflow    │
│                                 ▼                                             │
│                              Grafana: Predictive scaling · Model quality      │
└───────────────────────────────────────────────────────────────────────────────┘
      Ingress: grafana | prometheus | mlflow | forecast .localhost
```

### Data flow

1. **Scaling (every KEDA poll).** The scaler reads the last `timeSteps` samples of the workload's demand from Prometheus, normalises them, calls `POST /predict`, and reports `max(forecast over the horizon, observed)` to KEDA. KEDA's HPA sets `replicas = ceil(reported / targetValue)`.
2. **Model quality (every forecast horizon).** forecast-serving keeps its recent forecasts in memory; when a later window carries the value it forecast, it records the error per horizon step. `ForecastDriftHigh` fires when the rolling MAE stays high.
3. **Retraining (nightly).** The CronJob pulls recent history from Prometheus, trains four candidates, applies the level-tracking gate, compares the winner with the current champion on data neither has seen, moves the `@champion` alias in MLflow and calls `/reload`.
4. **Deployment.** CI publishes four images to GHCR; `scripts/up.sh` installs the chart on kind, trains the first model in-cluster and smoke-tests the REST and gRPC APIs; `e2e.yml` adds ten minutes of load with assertions.

A fuller treatment, with the decisions table: [ARCHITECTURE.md](ARCHITECTURE.md).

---

## Components

### Retraining pipeline (`training/`)

`forecast_training` loads a demand series (live Prometheus or a saved export), regularises it onto the step grid, trains the candidates (`rf_level`, `rf_delta`, `hgb_delta`, `ridge_delta`, and `hgb_delta_tod` / `rf_delta_tod` with the time of day as a feature), scores them on a chronological holdout and registers the winner with its scaler and `forecast_meta.json`. Promotion rules and the model contract: [training/README.md](training/README.md). The notebook in `notebooks/` is the original exploration record.

### Model registry (MLflow)

Versioned models with a `@champion` alias the pipeline moves atomically; artifacts are served through the MLflow server (no shared volume). The serving layer loads `models:/cpu-forecasting@champion` and falls back to the legacy `Production` stage, so notebook-registered models still work.

### Inference service (`serving/`)

A stateless FastAPI service: validates the window, applies the stored scaler and the model's feature/target transform, returns the whole horizon, hot-swaps the model on `/reload` without a restart, and scores its own past forecasts (`serving/quality.py`). `/ready` answers 503 until a model is loaded, which is how Kubernetes keeps traffic away from a pod that has nothing to serve.

### Predictive scaler (`scaler/`)

A KEDA external scaler (gRPC). Each `ScaledObject` carries its own PromQL query, target per replica, `capacity` (units conversion) and forecast settings in trigger metadata, so one deployment serves any number of workloads. Failure policy, metrics and metadata reference: [scaler/README.md](scaler/README.md).

### Observability (`monitoring/`)

Prometheus discovers annotated pods in the namespace; alert rules cover availability, latency, model health, forecast drift and scaler fallbacks, each with a runbook in `docs/runbooks/`; three Grafana dashboards are provisioned from `monitoring/dashboards/` and validated in CI.

### Benchmark (`bench/`)

A compressed-day load generator and a workflow that runs the same day under a CPU HPA and under the predictive scaler, then writes [docs/benchmark.md](docs/benchmark.md). Method and caveats: [bench/README.md](bench/README.md).

---

## API

### `POST /predict`

Takes the last `time_steps` values of the series (raw units, `step_seconds` apart — both in `GET /model-info`) and returns the forecast for every step of the model's horizon.

**Request:**
```json
{
  "values": [0.20, 0.25, 0.30, 0.28, 0.22, 0.35, 0.40, 0.38, 0.33, 0.29],
  "series": "predictive/cpu-burner",
  "timestamp": 1700000000,
  "step_seconds": 300
}
```

`series`, `timestamp` (unix time of the last value) and `step_seconds` (their spacing, when it differs from the model's) are optional. They let the service score this forecast against the actual value a later window brings; the KEDA scaler sends all three so forecasts for different `ScaledObject`s are never compared with each other's data.

**Response:**
```json
{
  "prediction": 0.2966,
  "predictions": [0.2966, 0.3102, 0.3187, 0.3140, 0.3055, 0.2990],
  "step_seconds": 300,
  "model": "cpu-forecasting",
  "stage": "@champion",
  "model_version": "3"
}
```

`prediction` is the first step; `predictions` covers the horizon (one value for notebook-registered models). `stage` and `model_version` say which registry entry served the answer, so a caller can see a model swap between requests.

### `POST /reload`

Re-fetches the champion from MLflow and swaps it in memory, atomically with its scaler and metadata. The retraining job calls it after moving the alias. On failure the previous model keeps serving and the call returns 500.

### `GET /model-info`

Loaded version, run id, `time_steps`, `horizon`, `step_seconds`, feature/target kinds, and the current rolling MAE per horizon step.

### `GET /live`, `GET /ready`

Liveness (process up) and readiness (a model is loaded). Split on purpose: a pod waiting for MLflow is alive but not ready, and must not be restarted for it.

### `GET /metrics`

Prometheus format: HTTP metrics from `prometheus-fastapi-instrumentator` plus the `forecast_*` metrics below.

---

## Observability

The serving layer is instrumented with `prometheus-fastapi-instrumentator`, which automatically tracks:

- `http_requests_total` — request count by endpoint and status code
- `http_request_duration_seconds` — latency histogram

On top of that the service measures its own **forecast quality** without any external store (`serving/quality.py`): every forecast is kept in memory, and when a later `/predict` window carries the value that was forecast, the error is recorded per horizon step:

- `forecast_abs_error{horizon_step}` — histogram of |actual − forecast|
- `forecast_rolling_mae{horizon_step}` — mean absolute error over the last 100 resolved forecasts (also in `GET /model-info`)
- `forecast_model_info{model_version,...}` — which version is serving

Memory is bounded (series × pending forecasts × rolling window, all configurable). The scaler's own metrics (`predictive_scaler_*`, see [`scaler/README.md`](scaler/README.md)) show observed vs predicted vs reported demand per `ScaledObject`.

**Dashboards and rules as code.** `monitoring/dashboards/*.json` and `monitoring/prometheus-rules.yaml` are the sources; `python monitoring/render.py` validates them (panel grid, unique ids, every PromQL metric and label name checked against what the code exports, a runbook for every alert) and syncs them into the Helm chart's ConfigMaps. CI runs it with `--check`.

**Alerts and runbooks.** `monitoring/prometheus-rules.yaml`: `ForecastDriftHigh` (rolling MAE above threshold for 30 min), `ScalerFallbackRate` (over half of decisions on observed demand for 15 min), `ScalerPrometheusErrors` (KEDA is applying the fallback), plus the availability, latency and model-health SLOs. Each alert's `runbook_url` points to `docs/runbooks/<Alert>.md`, and CI fails if the file is missing.

This matters for two reasons beyond standard uptime monitoring:

1. **Prediction latency is a hard constraint.** If this service sits in a scaling control loop, a slow response delays the scaling decision. A latency regression in the inference path should be treated the same as a latency regression in a critical API.

2. **Request rate reflects controller behavior.** Sudden drops or spikes in `/predict` call rate indicate upstream changes (autoscaler reconfiguration, controller failures). The metric acts as a proxy for control plane health.

---

## Infrastructure Integration

The forecast is wired into Kubernetes autoscaling through a **KEDA external scaler** ([`scaler/`](scaler/)): KEDA asks it for a metric, it answers with forecast demand, and the HPA that KEDA manages scales the workload on that number. A runnable walkthrough with a demo workload is in [`demo/`](demo/README.md).

### Reactive scaling (current state of the art)

```
CPU metric exceeds threshold → HPA fires → pod added → load stabilises
```

The lag between threshold breach and new pod readiness is typically 30–90 seconds depending on image pull time, startup probes, and JVM/interpreter warmup.

### Proactive scaling (target pattern)

```
Forecast predicts threshold breach in T+5min → scale event triggered at T → pod ready before load arrives
```

The forecast service enables this by exposing predicted utilization as a queryable value. The integration mechanism depends on the autoscaler in use:

### How the predictive scaler works (implemented)

```
Prometheus ──query_range (last 10 samples)──► predictive-scaler ──POST /predict──► forecast-serving
                                                     │
                                  reported = max(forecast, observed)
                                                     │
            HPA ◄── replicas = ceil(reported / targetValue) ◄── KEDA ◄── gRPC GetMetrics
```

- **Stateless and generic.** Each `ScaledObject` carries its own PromQL query, target per replica and forecast URL in trigger metadata; one scaler deployment serves any number of workloads.
- **Forecast adds capacity, never removes it.** In the default `max` mode the scaler reports `max(forecast, observed)`, so the forecast can scale up ahead of load but cannot scale below what is in use right now. `mode: forecast` trusts the model fully.
- **Degrades to reactive, not to zero.** If forecast-serving is down or the history window is not full yet, the scaler reports observed demand — the workload keeps scaling like a plain HPA. If Prometheus itself fails, it returns a gRPC error and KEDA applies the `ScaledObject`'s `fallback` replicas instead of the scaler inventing a number.
- **Observable.** Per-`ScaledObject` metrics for observed, predicted and reported demand, desired replicas, fallbacks and forecast latency; native gRPC health probes.
- **Why a number, not a verdict.** An earlier prototype endpoint returned scale-up/down/hold against thresholds. KEDA needs a number it can divide by a per-replica target, so the scaler calls `/predict` and does the replica arithmetic itself; the verdict endpoint was removed.

### Other integration options

**Custom metrics adapter:** An HPA with `externalMetric` type reads from an external metrics API. A bridge service could expose the forecast as a custom metric in Kubernetes' metrics pipeline (via `custom.metrics.k8s.io`).

**Scheduler integration:** For batch workloads, the forecast can inform whether to admit or delay a job submission — simpler than autoscaling and often sufficient for capacity management.

### Why not VPA

VPA adjusts resource requests/limits, not replica count. For workloads where the bottleneck is concurrent request capacity rather than per-pod CPU headroom, HPA-style horizontal scaling is the right lever. VPA is complementary but targets a different problem.

---

## Model Selection & Design Decisions

### Why Random Forest

Eight model families were evaluated in the notebook on the same holdout period (the last 20%, about 3 days, of a 15-day dataset: 10–25 January 2022 at 5-minute resolution):

| Model | Test RMSE |
|---|---|
| Baseline (hourly mean) | 0.028 |
| **Random Forest** | **0.084** |
| k-NN | 0.089 |
| SVR | 0.097 |
| Linear Regression | 0.138 |
| Bidirectional RNN/LSTM/GRU | ~0.164 |

The baseline uses in-sample hourly averages as predictions — it is a reference point to sanity-check the others, not a deployable model.

Neural networks (RNN, LSTM, GRU) underperform on this dataset. About 3,500 training windows at 5-minute intervals is not enough for sequence models to learn meaningful temporal structure beyond what the tree-based models capture with a simple sliding window. Random Forest with 10-step lag features achieves better generalisation with no hyperparameter tuning.

The general principle applies beyond this dataset: for short time-series with strong periodicity, classical ML with appropriate feature engineering frequently outperforms deep learning. The right model for the task depends on data volume, not what is currently fashionable.

### Why a sliding window of 10 steps

10 steps covers a 50-minute lookback. For daily-periodic CPU patterns, this is enough to capture the local trend without encoding the full periodic signal. A longer window would require the caller to maintain a larger buffer; a shorter window loses context. This is a design parameter (`TIME_STEPS` env var) — the right value is workload-specific.

### Why FastAPI over a streaming architecture

For this use case, request-response is the right protocol. An autoscaler polling every 30 seconds does not need a Kafka consumer. Keeping the integration point as a simple HTTP API minimises coupling and makes the service easy to call from any controller, regardless of its tech stack.

### Why MLflow for the registry

MLflow provides the model versioning and stage promotion semantics needed to safely update the Production model. The alternative would be manually versioning model files and rebuilding container images on every retrain — that approach loses audit trail and complicates rollbacks. MLflow's `models:/name/stage` URI scheme means the serving layer never needs to change to pick up a new model version.

---

## Limitations

These are known simplifications, not omissions:

- **Single-metric forecasting.** The model predicts CPU only. Real workload characterisation requires memory, network I/O, and queue depth. Each adds a feature dimension and a data collection dependency.
- **Drift is measured, not yet acted on.** `forecast_rolling_mae` and `ForecastDriftHigh` flag a model that stopped tracking the workload; retraining is still on the nightly schedule rather than triggered by the alert. The MAE is only as good as the scaler's windows: it needs `stepSeconds` to match the model and a few horizons of traffic before it says anything.
- **One benchmark, one small node.** The predictive-vs-reactive comparison ([docs/benchmark.md](docs/benchmark.md)) runs on a single CI runner that also hosts the load generator and the platform, on a pattern the model was trained on. It is one sample and shows the mechanism, not a capacity figure.
- **Anticipation depends on the clock.** The `_tod` candidates add the phase of the day to the window, so the model can move before a daily ramp is visible; it still cannot foresee a ramp that is not periodic, and it needs a few days of history to learn the pattern.
- **Short, noisy history.** Fifteen days of 5-minute data with large step-to-step swings: the best models beat persistence by ~60% RMSE but still carry ~0.2 cores of error per step. More history (and the weekly pattern it would expose) is the cheapest improvement available.
- **Single-node scale.** The stack is sized for kind and CI runners: one replica of everything, SQLite behind MLflow, Prometheus on an `emptyDir`. The chart is the production shape (Ingress, GHCR images, namespace RBAC), not production capacity.

---

## Retraining Pipeline

The notebook is the exploration record; production models come from [`training/`](training/README.md), run nightly by the chart's `forecast-retrain` CronJob:

```
Prometheus (14 days) → regularise + fill gaps → train candidates → holdout RMSE
     → level-tracking check → compare with @champion on data it has not seen
     → register + move alias → POST /reload on forecast-serving
```

What it changed, measured on the January data (6-step / 30-minute horizon, last 20% held out):

| Model | Holdout RMSE | Level tracking | Eligible |
|---|---|---|---|
| persistence (last value) | 0.536 | — | baseline |
| `rf_level` (the notebook's model) | **0.193** | 0.00 | no |
| `hgb_delta` | 0.199 | 1.00 | **selected** |
| `rf_delta` | 0.200 | 1.00 | yes |
| `ridge_delta` | 0.246 | 0.38 | no |

The notebook's Random Forest has the best RMSE on this holdout, and it would be the wrong model to autoscale on: given load above anything in training it forecasts *within* the training range — on a ramp to 3 cores it predicted 0.56. *Level tracking* measures exactly that (how far the forecast moves when the input moves beyond the training range), and candidates below 0.5 are not eligible. The delta models predict the change from the last value, so they follow new levels one-for-one at a ~3% RMSE cost.

Other properties:

- **Multi-step forecasts.** `/predict` now returns `predictions` for the whole horizon (`prediction` is still the next step). The scaler can scale for the horizon peak with `forecastAggregation: max`, buying lead time equal to the horizon.
- **Fair champion comparison.** Each registered version is tagged with `data_end_ts`; the next run compares against it only on windows after that point (and keeps the champion if there are too few), because a model always looks best on data it was fitted to.
- **Registry aliases, not stages.** Serving loads `models:/cpu-forecasting@champion` and falls back to the legacy `Production` stage, so notebook-registered models keep working.
- **Pinned transitive dependencies.** MLflow 2.11 fails to import with setuptools ≥ 81 and SQLAlchemy 2.1; both are now pinned in serving and training.

## Benchmark Results

Predictive scaler vs a plain CPU HPA, same kind cluster, same compressed day of load with two ramps ([docs/benchmark.md](docs/benchmark.md), method in [bench/README.md](bench/README.md)):

| | Reactive HPA | Predictive | |
|---|---|---|---|
| Morning ramp: time to the 3 replicas it needed | 165 s | 60 s | −64% |
| Morning ramp: p95 latency | 77 ms | 71 ms | −8% |
| Afternoon ramp, peak, whole day: p95 latency | same | same | |
| Replica-minutes over the day | 96.75 | 109 | +13% |

What that says, honestly:

- **The forecast helps where capacity has to grow from a low base.** On the morning ramp the predictive scaler had the capacity in place about 100 s sooner and latency during the ramp was lower.
- **Part of that is not foresight.** The scaler entered the ramp with two replicas instead of one: `forecastAggregation: max` over a noisy night occasionally asked for a second replica, and each request reset the HPA's 300-second scale-down window. The forecast itself rose only once the ramp was visible in the ten-sample window; with no time-of-day features the model extrapolates a ramp that has started, it does not anticipate one that has not.
- **Where the HPA already holds capacity, there is nothing to win.** Both scalers kept three replicas through the lunch dip, so the afternoon ramp, the peak and the worst minute look the same, and neither run had a request over 500 ms.
- **It costs.** 13% more replica-minutes, mostly the second replica held overnight.

The setup matters as much as the model: the first run of this benchmark trained the model on raw cores while the scaler fed it `demand / capacity`, and the forecast over-provisioned all night. The CronJob now trains on exactly the series the scaler sends (and CI checks the rendered query).

**On real traffic.** The same pipeline, unchanged, on hourly Wikipedia page views (three languages, 30 days, weekly in CI): the delta models beat persistence by 76–85% six hours out, the level-tracking gate rejects the best-RMSE model on one series because it cannot follow new levels, and the time-of-day features add nothing when the window already spans a day. Tables and reasoning: [docs/traces.md](docs/traces.md).

## Future Work

Stage-by-stage plan with ready-to-run briefs: [docs/ROADMAP.md](docs/ROADMAP.md).

**Additional signals**  
Add memory utilisation and request queue depth as input features. CPU alone is a weak proxy for capacity need in memory-bound or I/O-bound workloads.

**GPU workloads**  
The notebook's sequence models already run on CUDA when it is available; the production pipeline is CPU-only scikit-learn because the data is small. For GPU inference workloads the same pattern applies with a different demand metric (GPU utilisation or queue depth); serving and the scaler do not change.

**Alert-triggered retraining**  
`ForecastDriftHigh` already measures the regression; the next step is to let it start the retraining job (Alertmanager webhook → `kubectl create job`) instead of waiting for the nightly run.

---

## Quick Start

Everything runs on a local [kind](https://kind.sigs.k8s.io/) cluster from one script — the same script CI runs ([`.github/workflows/e2e.yml`](.github/workflows/e2e.yml)). Needs `docker`, `kind`, `kubectl`, `helm`, `curl`.

```bash
./scripts/up.sh
```

What it does: creates the cluster (with ingress-nginx and metrics-server), builds the four images and loads them into kind, installs the [`helm/predictive-stack`](helm/predictive-stack/) umbrella chart (MLflow, Prometheus, Grafana, forecast-serving, predictive-scaler, KEDA, the demo workload with its `ScaledObject`, the retraining CronJob), trains the first model from `notebooks/data-jan-2022.json` in-cluster and calls `/reload`, then smoke-tests `/predict` and the scaler's gRPC API. About 10 minutes on a laptop.

Then:

| | |
|---|---|
| Grafana (admin/admin) | http://grafana.localhost |
| Prometheus | http://prometheus.localhost |
| MLflow | http://mlflow.localhost |
| Forecast API | http://forecast.localhost/docs |

`*.localhost` resolves to the cluster's ingress on Linux (systemd-resolved) and in browsers; elsewhere add the four hosts to `/etc/hosts` as `127.0.0.1`, or set `DOMAIN=127.0.0.1.nip.io`.

```bash
# Inference
curl -X POST http://forecast.localhost/predict -H 'Content-Type: application/json' \
  -d '{"values":[0.2,0.25,0.3,0.28,0.22,0.35,0.4,0.38,0.33,0.29]}'

# Watch the demo scale (see demo/README.md for load generation)
kubectl -n predictive get hpa,scaledobject,pods -w
kubectl -n predictive logs deploy/predictive-scaler -f
```

Knobs (environment variables of `up.sh`): `IMAGE_TAG` (`dev` builds locally; a commit sha pulls the images CI published to GHCR), `STEP_SECONDS=30` to compress time so the forecast window fills in 5 minutes instead of 50, `DEMO_SCALING=hpa` for the reactive baseline, `TRAIN=0` / `SMOKE=0` to skip steps. Re-running upgrades the release in place; `kind delete cluster --name predictive` removes everything.

### Pilot on your own cluster

The same chart, without the demo, scaling your services ([`helm/predictive-stack/values-pilot.yaml`](helm/predictive-stack/values-pilot.yaml)):

```bash
helm dependency build helm/predictive-stack
helm upgrade --install predictive helm/predictive-stack -n predictive --create-namespace \
  -f helm/predictive-stack/values-pilot.yaml --set global.imageTag=latest
# KEDA's CRDs exist now; add your services as workloads
cat > my-workloads.yaml <<'YAML'
workloads:
  - name: api
    deployment: api
    query: sum(rate(container_cpu_usage_seconds_total{namespace="predictive", container="api"}[2m]))
    targetValue: "0.7"     # demand one replica should carry, in the query's units
    capacity: "4"          # model input = demand / capacity
    maxReplicas: 20
YAML
helm upgrade predictive helm/predictive-stack -n predictive --reuse-values -f my-workloads.yaml
```

Each entry becomes a `ScaledObject`; the nightly CronJob trains the model on the first entry's series divided by its `capacity`, so scaler and model always agree on units. Until the first training run there is no champion and the scaler scales on observed demand; `kubectl -n predictive create job --from=cronjob/forecast-retrain retrain-now` starts one as soon as there is a day of history. Prometheus keeps 30 days on a PersistentVolume, which is what the time-of-day features need.

### Pieces on their own

- **Images** are published by CI to `ghcr.io/m0zart89/predictive-scaling-system/{forecast-serving,predictive-scaler,cpu-burner,forecast-training}:{<sha>,latest}` on every push to `main`.
- **Helm charts**: `helm/predictive-stack` installs the whole system (`helm dependency build` first — KEDA is a dependency); `helm/forecast-serving` and `helm/predictive-scaler` work standalone.
- **Plain manifests**: `helm template predictive helm/predictive-stack -n predictive --set demo.scaling=predictive > stack.yaml` renders everything for `kubectl apply` (install KEDA's CRDs first).
- **Train a model** from anywhere with access to MLflow:
  ```bash
  pip install -r training/requirements.txt
  MLFLOW_TRACKING_URI=http://mlflow.localhost PYTHONPATH=training \
    python -m forecast_training --export notebooks/data-jan-2022.json \
      --exclude 2022-01-17T00:00:00Z/2022-01-17T08:00:00Z --reload-url http://forecast.localhost
  ```
  The notebook (`notebooks/project.ipynb`) remains as the exploration record; models it promotes to the `Production` stage are still served when no `@champion` alias exists.

---

## Project Structure

```
predictive-scaling-system/
├── .github/workflows/
│   ├── ci.yml                   # lint → tests → dashboards check → images (→ GHCR on main) → helm lint
│   ├── e2e.yml                  # up.sh on kind, 10 min of load, scaling assertions
│   └── benchmark.yml            # predictive vs reactive, manual + weekly, commits docs/benchmark.md
├── serving/                     # FastAPI inference service
│   ├── app.py                   # /predict, /reload, /model-info, probes
│   └── quality.py               # forecast error tracking (forecast_abs_error / forecast_rolling_mae)
├── scaler/                      # KEDA external scaler (gRPC): Prometheus → /predict → replicas
│   ├── predictive_scaler/       # config, clients, decision engine, gRPC server, metrics
│   └── tests/
├── training/                    # retraining pipeline: candidates, level-tracking gate, MLflow aliases
│   ├── forecast_training/       # data, features (shared contract), models, registry, CLI
│   └── tests/
├── helm/
│   ├── predictive-stack/        # umbrella chart: the whole system, KEDA as a dependency
│   │   └── files/               # dashboards and rules synced from monitoring/ by render.py
│   ├── forecast-serving/        # standalone chart for the service
│   └── predictive-scaler/       # standalone chart for the scaler
├── scripts/
│   ├── up.sh                    # kind cluster → helm install → train → smoke tests
│   ├── e2e.sh                   # load + verify scaling (used by e2e.yml)
│   └── kind-config.yaml
├── bench/                       # load generator, run.sh, collect.py, report.py
├── monitoring/
│   ├── dashboards/              # Grafana dashboards (source of truth, JSON)
│   ├── prometheus-rules.yaml    # alerting rules (source of truth)
│   └── render.py                # validates dashboards + rules, syncs them into the chart
├── demo/                        # cpu-burner workload (image) and walkthrough
├── docs/
│   ├── ROADMAP.md               # stages: done and briefs
│   ├── benchmark.md             # generated by benchmark.yml
│   ├── traces.md                # the pipeline on Wikipedia traffic (traces.yml)
│   ├── pitch.md                 # the project in nine sentences
│   └── runbooks/                # one page per alert
├── notebooks/                   # exploration record + the January 2022 export
├── tests/                       # serving, training↔serving contract, monitoring, bench
├── ARCHITECTURE.md
├── CHANGELOG.md
└── README.md
```
