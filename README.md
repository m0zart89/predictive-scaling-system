# AI infrastructure prototype for predictive autoscaling and workload-aware resource optimization

**Status:** Prototype / Portfolio  
**Stack:** Python · FastAPI · gRPC · MLflow · scikit-learn · Kubernetes · KEDA · Prometheus · Grafana

---

## Why this matters

As workloads become more compute-constrained and expensive, reactive scaling leads to latency spikes and inefficient resource usage.

This system demonstrates how predictive signals can be used to improve both performance and cost efficiency.

---

## System Overview

Kubernetes autoscalers (HPA, VPA, KEDA) operate reactively: they observe a metric crossing a threshold and then trigger a scaling event. For workloads with predictable patterns — scheduled batch jobs, business-hours traffic, recurring ETL pipelines — this means the scaling action always lags the load by at least one scrape-and-react cycle. The result is latency spikes at ramp-up and wasted capacity during predictable quiet periods.

This system is a prototype for a different approach: expose a forecasting API that infrastructure controllers can query ahead of time to anticipate load rather than react to it. A scheduler or autoscaler calls `POST /predict` with recent observations, receives a predicted CPU utilization value for the next interval, and uses that to make a proactive scaling decision.

The design goal is not to build a better autoscaler. It is to demonstrate the integration point: how a trained ML model becomes a queryable infrastructure service that fits into an existing Kubernetes control plane.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       Kubernetes Cluster                     │
│                                                              │
│  ┌─────────────┐    ┌─────────────────────┐    ┌──────────┐  │
│  │   MLflow    │◄───│  forecast-serving   │───►│Prometheus│  │
│  │             │    │                     │    │          │  │
│  │  Tracking   │    │  POST /predict      │    │ scrapes  │  │
│  │  Registry   │    │  POST /reload       │    │ /metrics │  │
│  │  Artifacts  │    │  GET  /health       │    └────┬─────┘  │
│  └─────────────┘    └─────────────────────┘         │        │
│        ▲                                      ┌──────▼─────┐ │
│        │                                      │  Grafana   │ │
│  ┌─────┴──────────────┐                       └────────────┘ │
│  │  Training Pipeline  │                                     │
│  │  (Jupyter notebook) │                                     │
│  └─────────────────────┘                                     │
└──────────────────────────────────────────────────────────────┘

External clients (autoscaler, scheduler, capacity tool)
        │
        └──► POST /predict  →  forecast-serving  →  MLflow model
```

### Data flow

1. **Training:** A Jupyter notebook loads raw CPU metrics (Prometheus export format), engineers sliding-window features, trains and evaluates multiple model families, and logs all runs to MLflow — parameters, metrics, and artifacts.
2. **Registration:** The best-performing model is promoted to `Production` in the MLflow Model Registry. The preprocessing scaler is stored as a co-located artifact.
3. **Serving:** The inference service loads the `Production` model from MLflow on startup. It exposes a REST endpoint that any infrastructure component can call.
4. **Observability:** The serving pod exposes a `/metrics` endpoint. Prometheus scrapes it on a 15-second interval. Grafana visualises request rate, prediction latency, and error rate.
5. **Model updates:** Retraining produces a new MLflow run. Promoting the new version to `Production` and calling `POST /reload` on the serving pod hot-swaps the model — no pod restart, no downtime.

---

## Components

### Training Pipeline (`notebooks/`)

A notebook-based pipeline that covers EDA, feature engineering, model training, evaluation, and MLflow logging. Eight model families are trained and compared on the same train/test split. All runs are tracked: hyperparameters, RMSE, artifacts. The notebook's final cell registers the best sklearn model to the MLflow Model Registry.

This is intentionally simple — the goal is demonstrating the MLflow integration contract, not building a sophisticated training system.

### Model Registry (MLflow)

Deployed as a Kubernetes `Deployment` with a `PersistentVolumeClaim` backing the artifact store and SQLite metadata backend. Provides:

- Experiment tracking across all training runs
- Versioned model storage with stage transitions (`None → Staging → Production`)
- Artifact co-location: the model and its preprocessing scaler are stored together under the same run

The serving layer depends on MLflow at startup. The model URI is `models:/cpu-forecasting/Production` — the serving service never references a specific run ID directly.

### Inference Service (`serving/`)

A stateless FastAPI application. Its role in the system is to translate a sequence of recent infrastructure observations into a scalar forecast that a controller can act on. It handles:

- Loading the `Production` model and scaler from MLflow at startup
- Input validation and preprocessing (inverse-scaling on output so consumers work in original units)
- Structured JSON responses with model provenance metadata
- Prometheus metrics via `prometheus-fastapi-instrumentator`
- Hot reload without pod restart (`POST /reload`)

Because the service is stateless and all state lives in MLflow, horizontal scaling is straightforward: add replicas, each pod loads the model independently.

### Observability (`monitoring/`)

Prometheus is configured to scrape the serving pod via static config and pod annotations (`prometheus.io/scrape: "true"`). Collected metrics include HTTP request count by endpoint, request duration histograms, and standard Python process metrics.

Grafana is deployed alongside Prometheus with three dashboards provisioned from `monitoring/dashboards/` (*Predictive scaling*, *Model quality*, *Forecast serving*). Alerting rules in `monitoring/prometheus-rules.yaml` cover availability, latency, model health, forecast drift and scaler fallbacks; every alert links a runbook in `docs/runbooks/`.

---

## Inference Service

### Who calls it

In a real integration, `/predict` would be called by:

- A **KEDA external scaler** — KEDA supports custom scalers that query arbitrary HTTP endpoints; the scaler would send recent observations and use the prediction to compute a target replica count
- A **custom HPA adapter** — Kubernetes' External Metrics API allows HPA to scale on arbitrary metrics; a bridge service could call `/predict` and expose the result as a custom metric
- A **capacity planning tool** — batch queries over historical windows to backtest how much earlier scaling could have triggered
- A **scheduler sidecar** — deciding whether to defer a batch job based on predicted headroom

### Design assumptions

The service assumes:
- Observations arrive at fixed 5-minute intervals
- The caller provides the last 10 values in raw (unscaled) form
- The caller is responsible for buffering and windowing; the service is stateless per request
- The forecast horizon is one step (5 minutes); multi-step forecasting would require a different model

---

## API

### `POST /predict`

Accepts the last N CPU utilization values (where N matches `TIME_STEPS`, default 10). Returns the predicted value for the next interval.

**Request:**
```json
{
  "values": [0.20, 0.25, 0.30, 0.28, 0.22, 0.35, 0.40, 0.38, 0.33, 0.29],
  "series": "demo/cpu-burner",
  "timestamp": 1700000000
}
```

`series` and `timestamp` (unix time of the last value) are optional. They let the service score this forecast against the actual value a later window brings; the KEDA scaler sends both so forecasts for different `ScaledObject`s are never compared with each other's data.

**Response:**
```json
{
  "prediction": 0.2966,
  "model": "cpu-forecasting",
  "stage": "Production"
}
```

The `model` and `stage` fields in the response are intentional: they let the caller verify which model version served the prediction. This matters in environments where multiple model versions may be staged concurrently.

### `POST /reload`

Triggers a hot reload of the `Production` model from MLflow. The serving pod acquires the new model in memory without restarting. Designed for the retraining workflow: promote new version in MLflow → call `/reload` → zero-downtime model swap.

### `GET /health`

Returns model load status. Used by Kubernetes readiness and liveness probes. The pod is only marked ready when a model is loaded — preventing traffic from reaching an uninitialized pod.

```json
{"status": "ok", "model_loaded": true}
```

### `GET /metrics`

Prometheus-format metrics endpoint. Exposes HTTP counters, duration histograms, and process metrics. Scraped by Prometheus every 15 seconds.

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

**Dashboards as code.** `monitoring/dashboards/*.json` is the source; `python monitoring/render.py` validates the JSON (panel grid, unique ids, every PromQL metric name checked against what the code exports) and renders the Grafana ConfigMap into `monitoring/grafana.yaml`. CI runs it with `--check`.

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
- **Why not `/recommend-scale`.** That endpoint returns a scale-up/down/hold verdict against thresholds. KEDA needs a number it can divide by a per-replica target, so the scaler calls `/predict` and does the replica arithmetic itself.

### Other integration options

**Custom metrics adapter:** An HPA with `externalMetric` type reads from an external metrics API. A bridge service could expose the forecast as a custom metric in Kubernetes' metrics pipeline (via `custom.metrics.k8s.io`).

**Scheduler integration:** For batch workloads, the forecast can inform whether to admit or delay a job submission — simpler than autoscaling and often sufficient for capacity management.

### Why not VPA

VPA adjusts resource requests/limits, not replica count. For workloads where the bottleneck is concurrent request capacity rather than per-pod CPU headroom, HPA-style horizontal scaling is the right lever. VPA is complementary but targets a different problem.

---

## Model Selection & Design Decisions

### Why Random Forest

Eight model families were evaluated on the same holdout period (last 4 days of a 24-day dataset):

| Model | Test RMSE |
|---|---|
| Baseline (hourly mean) | 0.028 |
| **Random Forest** | **0.084** |
| k-NN | 0.089 |
| SVR | 0.097 |
| Linear Regression | 0.138 |
| Bidirectional RNN/LSTM/GRU | ~0.164 |

The baseline uses in-sample hourly averages as predictions — it is a reference point to sanity-check the others, not a deployable model.

Neural networks (RNN, LSTM, GRU) underperform on this dataset. 7,200 training points at 5-minute intervals is not enough for sequence models to learn meaningful temporal structure beyond what the tree-based models capture with a simple sliding window. Random Forest with 10-step lag features achieves better generalisation with no hyperparameter tuning.

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
- **Autoscaling not yet benchmarked in a cluster.** The KEDA external scaler is implemented and tested (unit tests plus gRPC tests through the generated client), and checked end-to-end against the real forecast service, but there is no measured comparison with a reactive HPA yet.
- **Short, noisy history.** One month of 5-minute data with large step-to-step swings: the best models beat persistence by ~60% RMSE but still carry ~0.2 cores of error per step. More history (and the weekly pattern it would expose) is the cheapest improvement available.
- **Local Kubernetes only.** Manifests use NodePort and Minikube-specific image loading. This is appropriate for local development; production would use `Ingress`, image registries, and namespace RBAC.

---

## Retraining Pipeline

The notebook is the exploration record; production models come from [`training/`](training/README.md), run nightly by `k8s/retrain-cronjob.yaml`:

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

## Future Work

Stage-by-stage plan with ready-to-run briefs: [docs/ROADMAP.md](docs/ROADMAP.md).

**Predictive vs reactive benchmark**  
Run the demo workload under a daily traffic pattern twice — once with a CPU-based HPA, once with the predictive scaler — and compare p95 latency during ramps and idle replica-minutes. This is what turns the prototype into evidence. (The KEDA external scaler it relies on is implemented in `scaler/`.)

**Additional signals**  
Add memory utilisation and request queue depth as input features. CPU alone is a weak proxy for capacity need in memory-bound or I/O-bound workloads.

**Alert-triggered retraining**  
`ForecastDriftHigh` already measures the regression; the next step is to let it start the retraining job (Alertmanager webhook → `kubectl create job`) instead of waiting for the nightly run.

**GPU workload support**  
The training code is device-agnostic (`device = 'cuda' if available`). For GPU inference workloads, the same forecasting pattern applies with different metric sources. The serving infrastructure does not change.

---

## Quick Start

### Option A — Helm (recommended)

```bash
minikube start
kubectl apply -f mlflow/k8s.yaml

eval $(minikube docker-env)
docker build -t cpu-serving:latest serving/

# Deploy observability stack
kubectl apply -f monitoring/prometheus-rules.yaml
kubectl apply -f monitoring/prometheus-config.yaml
kubectl apply -f monitoring/loki.yaml
kubectl apply -f monitoring/grafana.yaml

# Deploy serving via Helm
helm install forecast-serving helm/forecast-serving/ \
  --set image.tag=latest \
  --set mlflow.trackingUri=http://mlflow:5000

kubectl get pods
```

### Option B — raw manifests (local dev)

```bash
minikube start
kubectl apply -f mlflow/k8s.yaml
eval $(minikube docker-env)
docker build -t cpu-serving:latest serving/
kubectl apply -f k8s/serving.yaml
kubectl apply -f monitoring/prometheus-rules.yaml
kubectl apply -f monitoring/prometheus-config.yaml
kubectl apply -f monitoring/loki.yaml
kubectl apply -f monitoring/grafana.yaml
kubectl get pods
```

**Train a model and register it to MLflow** (sets the `@champion` alias):
```bash
pip install -r training/requirements.txt
MLFLOW_TRACKING_URI=http://192.168.49.2:30500 PYTHONPATH=training \
  python -m forecast_training --export notebooks/data-jan-2022.json \
    --exclude 2022-01-17T00:00:00Z/2022-01-17T08:00:00Z
```
Nightly retraining from live Prometheus: build `training/` into the cluster and apply `k8s/retrain-cronjob.yaml`.
The notebook (`notebooks/project.ipynb`) remains as the exploration record; models it promotes to the `Production` stage are still served when no `@champion` alias exists.

**Load model into the running service:**
```bash
curl -X POST http://192.168.49.2:30800/reload
```

**Test:**
```bash
# Health check
curl http://192.168.49.2:30800/health

# Inference
curl -X POST http://192.168.49.2:30800/predict \
  -H 'Content-Type: application/json' \
  -d '{"values":[0.2,0.25,0.3,0.28,0.22,0.35,0.4,0.38,0.33,0.29]}'

# MLflow UI
open http://192.168.49.2:30500

# Prometheus
open http://192.168.49.2:30900

# Grafana (admin/admin) — Prometheus + Loki datasources pre-provisioned
open http://192.168.49.2:30300
```

---

## Project Structure

```
vuna/
├── .github/
│   └── workflows/
│       └── ci.yml               # Lint → tests → dashboards check → image builds → helm lint
├── training/                    # Retraining pipeline: candidates, level-tracking gate, MLflow aliases
│   ├── forecast_training/       # data, features (shared contract), models, registry, CLI
│   ├── tests/                   # unit + end-to-end against a SQLite MLflow registry
│   └── Dockerfile
├── scaler/                      # KEDA external scaler (gRPC): Prometheus → /predict → replicas
│   ├── predictive_scaler/       # config, clients, decision engine, gRPC server, metrics
│   ├── tests/                   # unit + gRPC tests
│   ├── gen_proto.sh             # regenerate gRPC stubs from externalscaler.proto
│   └── Dockerfile
├── demo/                        # cpu-burner workload + ScaledObject + walkthrough
├── helm/
│   ├── predictive-scaler/       # Helm chart for the scaler
│   └── forecast-serving/        # Helm chart (production deployment path)
│       ├── Chart.yaml
│       ├── values.yaml
│       └── templates/
│           ├── deployment.yaml
│           ├── service.yaml
│           ├── secret.yaml      # MLflow URI stored as K8s Secret
│           ├── hpa.yaml         # HorizontalPodAutoscaler (opt-in)
│           ├── ingress.yaml     # Ingress (opt-in)
│           └── serviceaccount.yaml
├── serving/
│   ├── app.py                   # FastAPI inference service
│   ├── quality.py               # forecast error tracking (forecast_abs_error / forecast_rolling_mae)
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   ├── project.ipynb            # EDA, training, MLflow logging
│   └── data-jan-2022.json       # Raw Prometheus export (input data)
├── mlflow/
│   └── k8s.yaml                 # Deployment + PVC + Service
├── k8s/
│   ├── serving.yaml             # Raw manifests (local dev alternative to Helm)
│   └── retrain-cronjob.yaml     # Nightly retraining job
├── monitoring/
│   ├── prometheus-config.yaml   # Prometheus ConfigMap + Deployment + Service
│   ├── prometheus-rules.yaml    # Alerting rules: SLOs, forecast drift, scaler fallbacks
│   ├── dashboards/              # Grafana dashboards (source of truth, JSON)
│   ├── render.py                # validates dashboards + rules, renders the ConfigMap
│   ├── grafana.yaml             # Grafana + datasources + generated dashboards ConfigMap
│   └── loki.yaml                # Loki log aggregation
├── docs/
│   ├── ROADMAP.md               # Done / next stages with briefs for each
│   └── runbooks/                # one page per alert
├── ARCHITECTURE.md
└── README.md
```
