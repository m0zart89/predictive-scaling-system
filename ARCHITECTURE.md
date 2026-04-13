# Architecture: Workload Forecasting Service

**Document type:** Internal design document  
**Scope:** System design, data flow, constraints, and trade-offs  
**Status:** Prototype

---

## 1. System Context

This service is a **decision-support component** in a Kubernetes-based platform. It does not make scaling decisions itself. It provides a queryable forecast that a controller — an autoscaler, a scheduler, a capacity tool — can incorporate into its own decision logic.

```
┌──────────────────────────────────────────────────────────────────┐
│                        Platform Layer                            │
│                                                                  │
│   Prometheus ──────────────────────────────────► Grafana         │
│       │                                                          │
│       │ scrapes /metrics                                         │
│       │                          ┌─────────────────────────┐     │
│       ▼                          │   Kubernetes Control    │     │
│  forecast-serving ◄──────────────┤   Plane                 │     │
│       │                          │                         │     │
│       │ POST /predict            │   HPA / KEDA / custom   │     │
│       ▼                          │   controller            │     │
│  MLflow Registry                 └─────────────────────────┘     │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

Within a platform, this system occupies the space between the **observability layer** (where metrics live) and the **control layer** (where scaling decisions are made). It consumes raw metric observations and produces a scalar forecast. The control layer decides what to do with that forecast.

This separation is intentional. The forecasting service has no knowledge of cluster state, replica counts, or pod scheduling. It is a stateless function: recent observations in, prediction out.

---

## 2. Data Flow

### Training path

```
Prometheus export (JSON)
        │
        ▼
notebooks/project.ipynb
  ├── parse timestamps and values
  ├── fill gaps (linear / mean imputation)
  ├── build sliding window features (lag-10 window → next value)
  ├── train N model families
  ├── evaluate on holdout split
  ├── log all runs to MLflow (params, metrics, artifacts)
  └── register best model → MLflow Model Registry (stage: Production)
                │
                ▼
        MLflow artifact store (PVC)
          ├── model binary (sklearn pipeline)
          └── scaler binary (MinMaxScaler, pickled)
```

The training pipeline is notebook-driven and runs manually. It is not triggered automatically. All artifacts and metrics produced by every run are retained in MLflow — retraining does not overwrite history.

### Inference path

```
Caller (autoscaler, scheduler, tool)
        │
        │  POST /predict
        │  body: {"values": [v1, v2, ..., v10]}
        ▼
forecast-serving (FastAPI)
  ├── validate input length == TIME_STEPS
  ├── scaler.transform(input)        # MinMaxScaler, loaded from MLflow
  ├── model.predict(scaled_input)    # Random Forest, loaded from MLflow
  ├── scaler.inverse_transform(output)
  └── return {"prediction": float, "model": str, "stage": str}
        │
        ▼
Caller receives predicted CPU utilization for next interval
```

The service is stateless per request. It holds the model and scaler in memory (loaded at startup or on `POST /reload`). There is no request-level state, no caching, no session.

### Model update path

```
Retrain notebook (manual)
        │
        ▼
New run logged to MLflow
        │
        ▼
Operator promotes new version → Production stage in MLflow UI or API
        │
        ▼
POST /reload called on forecast-serving
        │
        ▼
Service re-fetches Production model + scaler from MLflow
New model active in memory, no pod restart required
```

The `/reload` endpoint is the synchronisation point between the registry and the running service. Until `/reload` is called, the service continues serving predictions from the previously loaded model version.

---

## 3. Control Flow

### Expected integration pattern

```
[Control loop — runs every N seconds]

1. Collect recent CPU observations (last 10 × 5min intervals)
   Source: Prometheus query or metrics buffer

2. POST /predict {"values": [...10 values...]}
   → response: {"prediction": 0.31, ...}

3. Evaluate: predicted_value vs threshold (e.g. 0.75)
   if predicted_value > threshold:
       trigger scale-up by M replicas
   elif predicted_value < low_threshold:
       schedule scale-down

4. Kubernetes controller applies scaling decision
```

The caller is responsible for:
- Buffering and windowing observations
- Defining the threshold logic
- Translating the prediction into a scaling action
- Handling `/predict` failures gracefully (fall back to reactive scaling)

The forecast service is responsible for:
- Returning a calibrated prediction quickly
- Reporting which model version produced the prediction
- Exposing its own health and metrics

This boundary is deliberate. Embedding threshold logic or scaling decisions inside the forecast service would couple it to a specific controller and make it harder to reuse across different workloads or autoscaler implementations.

### Call timing

The service is designed for periodic polling, not streaming. A typical control loop runs every 30–60 seconds. At that frequency, one `POST /predict` per cycle is the expected load. Burst patterns (e.g., multiple controllers polling simultaneously) are handled by running multiple replicas of the stateless serving pod.

---

## 4. Components

### Training pipeline (`notebooks/`)

**Responsibility:** Produce a versioned, registered model artifact that the inference service can load.

The notebook covers data parsing, gap filling, feature engineering, model training, and MLflow logging. Eight model families are evaluated: Random Forest, k-NN, SVR, Linear Regression, and bidirectional RNN/LSTM/GRU variants.

The key contract this component must satisfy: every registered `Production` model must have a co-located `scaler` artifact in the same MLflow run. The serving layer expects both to be present and will fail to load if the scaler is missing.

The notebook is not idempotent in its current form — re-running it creates new MLflow runs and may produce a new `Production` version. In a production system, this pipeline would be replaced by a triggered job with explicit versioning gates.

### Model Registry (MLflow)

**Responsibility:** Version, store, and serve model artifacts with stage semantics.

MLflow provides two things the serving layer depends on:

1. **Stage-based lookup** — the serving service loads `models:/cpu-forecasting/Production`. It never references a run ID directly. This means a model can be updated without changing any configuration in the serving layer.
2. **Artifact co-location** — the scaler is stored as an artifact of the same run that produced the registered model. This avoids version mismatch between model and scaler.

MLflow runs as a single-replica Deployment with a PVC backing both the SQLite metadata database and the artifact store. This is suitable for a single-cluster prototype. For multi-cluster or high-availability use, the backend store would need to be externalised (PostgreSQL, S3/GCS for artifacts).

### Inference Service (`serving/`)

**Responsibility:** Expose model predictions as a reliable, observable HTTP endpoint.

The service does minimal work beyond the prediction itself: input validation, scaling, inference, inverse scaling, and structured response. It does not contain business logic about what to do with a prediction — that is the caller's responsibility.

**Endpoint surface:**

| Endpoint | Method | Purpose |
|---|---|---|
| `/live` | GET | Liveness probe — process is running |
| `/ready` | GET | Readiness probe — model is loaded and inference is possible |
| `/model-info` | GET | Operational metadata: version, run_id, loaded_at |
| `/predict` | POST | Core inference: observations → predicted utilization |
| `/recommend-scale` | POST | Decision-support prototype: prediction → scaling recommendation |
| `/reload` | POST | Hot-swap Production model from MLflow without pod restart |
| `/metrics` | GET | Prometheus metrics |
| `/health` | GET | Legacy health check (backwards-compatible) |

Key design properties:

- **Split liveness / readiness.** `/live` returns 200 as long as the process is running. `/ready` returns 503 until a model is loaded. This prevents Kubernetes from restarting a healthy pod that is simply waiting for MLflow — a common failure mode when using a single `/health` for both probes.
- **Stateless per request.** Model and scaler are in memory. No per-request state, no session, no caching.
- **Startup tolerance.** If MLflow is unavailable at startup, the service starts in degraded state. Kubernetes readiness probe holds traffic until `/reload` succeeds after MLflow recovers.
- **Hot reload.** `POST /reload` swaps the in-memory model without pod restart. On success, returns the new model metadata so the caller can confirm the version change.
- **Provenance in response.** `/predict` includes `model_version` in every response. Callers can detect a mid-deployment model swap by watching this field change.
- **Input guard.** Validates value count, rejects NaN/inf values before reaching the model. Non-finite predictions from the model are also caught and rejected.
- **`/recommend-scale` is a prototype.** It applies a configurable threshold policy on top of the inference result. It does not call the Kubernetes API. Thresholds are caller-supplied so different workloads can use their own headroom policies.

**Custom Prometheus metrics** (beyond HTTP instrumentation):

| Metric | Type | Labels |
|---|---|---|
| `forecast_prediction_requests_total` | Counter | — |
| `forecast_prediction_failures_total` | Counter | `reason={validation,inference}` |
| `forecast_model_reloads_total` | Counter | `result={success,failure}` |
| `forecast_model_loaded` | Gauge | — |

`forecast_model_loaded` is the key operational signal: a sustained `0` means the service is alive but unable to serve predictions, likely due to MLflow connectivity or an empty Production stage.

### Monitoring stack (`monitoring/`)

**Responsibility:** Expose serving behavior as time-series metrics and logs for operational visibility.

Prometheus scrapes `forecast-serving` at 15-second intervals via pod annotations. Alerting rules are defined in `monitoring/prometheus-rules.yaml` and cover two SLO dimensions:

- **Availability:** 5xx error rate > 0.5% over 5 minutes → `warning`
- **Latency:** p99 `/predict` duration > 200ms over 5 minutes → `warning`
- **Model health:** `forecast_model_loaded == 0`, no `/predict` traffic in 15 minutes, `/reload` returning errors

Loki is deployed for log aggregation. The service emits structured log lines on model load, reload, prediction failures, and startup — making log-based alerting possible without custom instrumentation.

Grafana is pre-provisioned with both Prometheus and Loki datasources via ConfigMap. Dashboard definitions are committed to `monitoring/grafana.yaml` and automatically loaded on pod start.

---

## 5. System Constraints

### Prediction horizon

The model predicts one step ahead (5 minutes). This is the only horizon it was trained for. Using the model to forecast beyond one step requires either:

- Autoregressive iteration (feed prediction back as input), which accumulates error rapidly for this model class
- A separate model trained explicitly on a multi-step target

Five minutes is sufficient for HPA-style horizontal scaling decisions (pod startup time on a warm node is typically 15–30 seconds). It is insufficient for capacity planning decisions that require 30–60 minute lead times.

### Input requirements

The caller must provide exactly 10 consecutive observations at 5-minute intervals. The service does not handle:

- Missing values in the input window
- Variable-length windows
- Observations at irregular intervals

Gap handling is the caller's responsibility. In practice, a monitoring agent buffering Prometheus metrics would need to interpolate or reject windows with gaps before calling `/predict`.

### Latency

Random Forest inference on a 10-feature input is sub-millisecond. The dominant latency factors are:

- Network round-trip to the cluster (cluster-internal: <1ms; external: depends on ingress)
- FastAPI request/response overhead (~1–2ms)
- Scaler transform (~0.1ms)

Total inference latency is expected to be under 5ms cluster-internally. This is not a latency-sensitive path for 30-second polling intervals, but it would matter if the service were embedded in a tight control loop running at sub-second frequency.

### Compute

The model runs on CPU. No GPU is required or configured. Random Forest inference is single-threaded and uses negligible CPU — a 0.1 CPU request in the Kubernetes Deployment spec is appropriate.

Memory footprint: the model and scaler together consume under 50MB in memory. The serving pod's memory limit can be set conservatively.

### Concurrency

FastAPI with a single Uvicorn worker processes one request at a time by default. At the expected call frequency (one request per 30–60 seconds), this is not a bottleneck. Under high-concurrency load (e.g., multiple controllers polling simultaneously), the right scaling mechanism is horizontal pod scaling, not threading.

---

## 6. Failure Modes

### Bad predictions

**Cause:** Model trained on data that no longer reflects current workload patterns (concept drift), or input window contains anomalous values.

**System behavior:** The service returns a prediction without error. The caller cannot distinguish a correct prediction from a stale-model prediction based on the response alone.

**Mitigation path:** Compare rolling prediction error against actuals. A Prometheus recording rule that tracks prediction MAE over a sliding window can trigger an alert when accuracy degrades. This requires the caller to report actual values back — a feedback loop not currently implemented.

### Stale model

**Cause:** Workload patterns changed after the last training run. `/reload` has not been called after a new model was promoted to Production.

**System behavior:** Predictions remain valid for the model version in memory. Accuracy degrades silently relative to a freshly trained model.

**Mitigation path:** Automated retraining on a schedule (weekly or triggered by drift detection) + automated `/reload` call after promotion. Currently both steps are manual.

### MLflow unavailable

**Cause:** MLflow pod crash, PVC mount failure, OOMKill.

**System behavior at startup:** `load_model()` raises an exception. The service starts in a degraded state (`model_loaded: false`). Kubernetes readiness probe blocks traffic. The pod will not receive requests until `/reload` succeeds after MLflow recovers.

**System behavior during operation:** A running pod with a loaded model is unaffected by MLflow downtime — the model is in memory. `/reload` will fail until MLflow recovers. Model updates are blocked during the outage.

### Inference service unavailable

**Cause:** Pod crash, OOMKill, node failure, resource starvation.

**System behavior:** The caller receives connection errors or timeouts from `/predict`.

**Expected caller behavior:** Fall back to reactive scaling (standard HPA threshold behavior). The forecast service should be treated as an enhancement to the control loop, not a dependency. Callers must implement a fallback — this is a design requirement for any integration.

With `replicas: 1` in the current Deployment, there is no redundancy. A pod restart causes a gap in availability during the startup + model load period (typically 10–20 seconds given MLflow response time).

### Prometheus scrape failure

**Cause:** Prometheus pod unavailable, network partition, target pod restart.

**System behavior:** Metric gaps in Prometheus. Grafana shows blank intervals. No impact on inference.

**Operational impact:** Loss of latency and error rate visibility. If alerting rules are defined on `http_requests_total`, a scrape gap can trigger false alerts (rate drops to zero). Alerting rules should use `absent()` checks to distinguish scrape failures from genuine zero-traffic periods.

---

## 7. Trade-offs & Design Decisions

### Simple model over complex

Random Forest was selected over LSTM/GRU not for simplicity's sake, but because it produces better predictions on this dataset. The training dataset is ~7,200 observations (24 days × 288 intervals/day). Sequence models require substantially more data to outperform well-featured tree-based models. On sub-10K point datasets with strong periodicity, lag features + Random Forest is a sound choice.

The decision principle: use the simplest model that achieves acceptable accuracy. A more complex model that trains longer, is harder to debug, and performs worse is not a better engineering choice.

### API-based inference over embedded

The inference service is a separate network endpoint rather than a library embedded in the autoscaler or controller. This adds a network hop but provides:

- **Language independence.** Any controller (Go, Python, Java) can call an HTTP API.
- **Independent deployment.** The model can be updated without redeploying the controller.
- **Observability boundary.** The service exposes its own metrics. Latency and error rate are visible independently of the caller.
- **Reusability.** The same endpoint can serve multiple callers with different threshold logic.

The trade-off is network latency and an additional failure domain. For the expected call frequency (sub-minute intervals), this is acceptable.

### MLflow over manual versioning

The alternative to MLflow is file-based versioning: save model files with timestamps, bake them into container images, redeploy on update. This approach loses audit trail, makes rollbacks manual, and couples model updates to the container release cycle.

MLflow provides stage semantics (`Staging → Production`) that cleanly separate model evaluation from serving. The `models:/name/stage` URI scheme means the serving layer's configuration does not change when a new model is promoted — only the registry state changes.

The trade-off is an additional stateful service to operate. MLflow adds a PVC dependency and a startup dependency for the serving pod.

### Kubernetes deployment over standalone

Running all components as Kubernetes Deployments demonstrates the target operational model: every component has health probes, resource limits, restart policies, and is managed by the control plane. This is the correct deployment model for any service that should participate in a production platform.

The alternative (Docker Compose, standalone processes) would be simpler to run locally but would not demonstrate the integration patterns relevant to platform engineering.

---

## 8. Scaling Considerations

### Inference service

The serving pod is stateless. Scaling horizontally (`replicas: N`) works without coordination — each pod loads the model from MLflow independently at startup. There is no shared in-memory state between replicas.

Load balancing across replicas is handled by the Kubernetes Service (round-robin by default). For a polling-frequency workload, a single replica is sufficient. Multiple replicas add availability during rolling updates.

Resource profile per replica: <0.1 CPU idle, <0.2 CPU under load; <200MB memory. The service is not compute-intensive.

### MLflow

MLflow with SQLite backend is a single-writer system. It does not scale horizontally in the current configuration. For high-concurrency training environments (many parallel training runs logging simultaneously), the backend store should be migrated to PostgreSQL. Artifact storage should be migrated to S3 or GCS to decouple storage scaling from the MLflow pod.

For this use case (infrequent training, single-writer), SQLite is adequate.

### Prometheus and Grafana

Both are single-replica in the current deployment. Prometheus is a stateful component — its TSDB is local to the pod. For production use, Prometheus would be replaced by a managed metrics backend (Thanos, Cortex, Grafana Mimir) or a cloud provider's metrics service, with Prometheus used only as a scrape agent.

### The forecast model itself

The current model does not support parallel inference requests in the sense that a GPU-based model would. Python's GIL means CPU-bound inference is effectively single-threaded per process. For the expected call frequency, this is not a constraint. If the service were used as a high-frequency inference endpoint (sub-second polling from many callers), the right approach is horizontal pod scaling, not threading.

---

## 9. Future Improvements

**KEDA external scaler**  
The highest-value near-term integration. KEDA's `ScaledObject` supports HTTP-based external scalers. A thin adapter service would call `POST /predict`, compare the result against a configured threshold, and return a target replica count. This closes the loop from forecast to scaling action without modifying the forecast service itself.

**Automated retraining**  
Replace manual notebook execution with a Kubernetes `CronJob`. The job fetches recent metric data from Prometheus (or a metrics store), retrains the model, logs to MLflow, runs validation, and promotes to Production if the new model meets accuracy criteria. Adds `/reload` call at the end. This makes the system self-maintaining against slow concept drift.

**Multi-step forecast**  
Train a model with a multi-step output horizon (e.g., 6 steps / 30 minutes). This enables capacity planning tools to reason about scaling lead times rather than just the next interval. Requires either a direct multi-output model or an autoregressive wrapper with error-bounding.

**Streaming input pipeline**  
Replace batch notebook training with a streaming pipeline that consumes metrics from Kafka or a Prometheus remote write endpoint, incrementally updates features, and triggers retraining when data volume thresholds are met. Relevant for workloads where patterns change faster than a daily/weekly retraining cycle can track.

**Prediction feedback loop**  
Instrument the caller to report actual values back to the service (or to a shared store). Use this to compute rolling prediction MAE in Prometheus. Alert when accuracy degrades beyond a threshold. This is the missing feedback loop that would make drift detection operational rather than theoretical.
