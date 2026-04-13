# AI infrastructure prototype for predictive autoscaling and workload-aware resource optimization

**Status:** Prototype / Portfolio  
**Stack:** Python · FastAPI · MLflow · scikit-learn · Kubernetes · Prometheus · Grafana

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

Grafana is deployed alongside Prometheus. In a production system, the dashboards would include prediction latency SLOs, error rate alerts, and model version tracking. Here the stack is deployed and functional; dashboard definitions are not committed.

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
  "values": [0.20, 0.25, 0.30, 0.28, 0.22, 0.35, 0.40, 0.38, 0.33, 0.29]
}
```

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

This matters for two reasons beyond standard uptime monitoring:

1. **Prediction latency is a hard constraint.** If this service sits in a scaling control loop, a slow response delays the scaling decision. A latency regression in the inference path should be treated the same as a latency regression in a critical API.

2. **Request rate reflects controller behavior.** Sudden drops or spikes in `/predict` call rate indicate upstream changes (autoscaler reconfiguration, controller failures). The metric acts as a proxy for control plane health.

In a production deployment, the Grafana dashboard would include alerting rules on p99 latency and error rate.

---

## Infrastructure Integration

This section describes how the system would integrate with Kubernetes in a real deployment. **None of this is currently implemented** — the autoscaling integration is conceptual.

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

**KEDA external scaler:** KEDA's `ScaledObject` supports HTTP-based external scalers. A lightweight adapter service would call `POST /predict`, compare the result against a configured threshold, and return a target replica count to KEDA. This requires ~50 lines of Go or Python — the heavy lifting is in the forecasting service.

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
- **Single-step horizon.** The model predicts one interval (5 minutes) ahead. Multi-step forecasting (15, 30, 60 minutes) is needed for capacity planning decisions with longer lead times and requires either a different model architecture or an autoregressive wrapper.
- **No concept drift detection.** The model is trained once on January 2022 data. In production, CPU patterns shift as workloads change. Without drift detection and scheduled retraining, prediction quality degrades silently.
- **No autoscaling integration.** The KEDA/HPA integration described above is not implemented. The service provides the data contract; the controller integration is left as future work.
- **Static training pipeline.** The notebook is not automated. In production, retraining should be triggered by a CronJob or event, not a manual notebook execution.
- **No persistent Grafana dashboards.** Dashboard definitions are not committed to the repository. In a real system, dashboards would be version-controlled and provisioned via ConfigMap.
- **Local Kubernetes only.** Manifests use NodePort and Minikube-specific image loading. This is appropriate for local development; production would use `Ingress`, image registries, and namespace RBAC.

---

## Future Work

**KEDA external scaler integration**  
Implement the HTTP adapter that translates `/predict` output into KEDA's scaler protocol. This closes the loop from forecast to actual scaling action and is the most direct path to validating the end-to-end system.

**Automated retraining pipeline**  
Replace the manual notebook with a Kubernetes `CronJob` that: fetches recent metrics from Prometheus, retrains the model, logs to MLflow, and promotes the new version to `Production` if validation metrics improve. This makes the system self-updating rather than point-in-time.

**Multi-step forecasting**  
Extend the model to output a forecast horizon (e.g., next 6 steps / 30 minutes). This enables capacity planning tools to reason about scaling lead times and pod startup budgets.

**Additional signals**  
Add memory utilisation and request queue depth as input features. CPU alone is a weak proxy for capacity need in memory-bound or I/O-bound workloads.

**Model drift alerting**  
Compare rolling prediction error against a baseline. A Prometheus alert on prediction MAE regression triggers retraining — making degradation observable before it affects scaling decisions.

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

**Train models and register to MLflow:**
```bash
cd notebooks && jupyter notebook project.ipynb
# Run all cells. The final cell promotes the best model to Production.
```

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
│       └── ci.yml               # Lint → build → helm lint pipeline
├── helm/
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
│   ├── Dockerfile
│   └── requirements.txt
├── notebooks/
│   ├── project.ipynb            # EDA, training, MLflow logging
│   └── data-jan-2022.json       # Raw Prometheus export (input data)
├── mlflow/
│   └── k8s.yaml                 # Deployment + PVC + Service
├── k8s/
│   └── serving.yaml             # Raw manifests (local dev alternative to Helm)
├── monitoring/
│   ├── prometheus-config.yaml   # Prometheus ConfigMap + Deployment + Service
│   ├── prometheus-rules.yaml    # Alerting rules (availability + latency SLOs)
│   ├── grafana.yaml             # Grafana + provisioned datasources + dashboard
│   └── loki.yaml                # Loki log aggregation
├── ARCHITECTURE.md
└── README.md
```
