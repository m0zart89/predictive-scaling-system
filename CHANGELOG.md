# Changelog

Stages follow [docs/ROADMAP.md](docs/ROADMAP.md). Versions are git tags on
`main`; `v1.0.0` is the first release where every stage is in.

## v1.1.0 — after the audit (PR #5)

- **Time-of-day features.** `hgb_delta_tod` / `rf_delta_tod` add sin/cos of
  the last sample's phase in the seasonal cycle (`period_seconds`); serving
  evaluates it for the request's timestamp. On the synthetic daily pattern
  the timed model beats the same model without the phase at the 30-minute
  horizon.
- **Pilot readiness.** Prometheus on a PersistentVolume with 30-day
  retention; `values-pilot.yaml`; README section on running it on your own
  cluster.
- **Workloads.** `.Values.workloads` renders one ScaledObject per service
  through the template the demo uses; the CronJob trains on
  `training.workload`'s series divided by its capacity. A release with
  nothing to train on installs without the CronJob.
- **Reload auth.** `/reload` requires `X-Reload-Token`; the chart generates
  the token once and shares it with the training jobs. Optional
  NetworkPolicy for forecast-serving.
- **Drift threshold from the model.** Each version is tagged with its
  per-step holdout MAE, serving exports it, `ForecastDriftHigh` fires at
  twice it (fixed 0.25 only without a baseline).
- **Real traces.** `forecast_training.traces` converts Wikimedia page views
  and timestamp/value CSVs (Alibaba and the like); `traces.yml` trains on
  three articles weekly and fails unless a candidate beats persistence.

## v1.0.0 — stages 3–6 (PR #4)

### Stage 6 — polish
- Fresh-eye review, fixes with tests: the availability and reload alerts used
  a label the instrumentator does not export (`status_code`) and could never
  fire — `render.py` now checks metric *and* label names in alert rules;
  serving swaps model, scaler and metadata atomically; a failed `/reload`
  no longer reports the still-serving model as unloaded; latency buckets fine
  enough for the p99 alert; the scaler sends only contiguous windows with the
  last sample's own timestamp.
- Duplicates removed: serving imports the feature/target transforms from
  `training/forecast_training/features.py` (the serving image builds from
  the repository root); the raw-manifest deployment path (`k8s/`, `mlflow/`,
  `monitoring/*` manifests, `demo/*.yaml`) and the unfed Loki are gone — the
  chart is the one deployment path. Dead `/recommend-scale` and `/health`
  endpoints removed.
- README rewritten for the system as it is (dataset length corrected to 15
  days); `ARCHITECTURE.md` around the three loops with a decisions table;
  `CHANGELOG.md`; `docs/pitch.md`; `docs/ROADMAP.md` stages 3–6 in Done.

### Stage 5 — benchmark: predictive vs reactive HPA
- `bench/load.py`: one day of traffic compressed into 40 minutes with sharp
  ramps, exported as `bench_*` metrics; `bench/run.sh` runs warm-up →
  training → CPU-HPA day → predictive day on one kind cluster;
  `bench/collect.py` and `bench/report.py` turn Prometheus series into
  `docs/benchmark.md` and charts.
- `.github/workflows/benchmark.yml`: manual, weekly or via the `benchmark`
  PR label; commits the report.
- Result: on the morning ramp the needed capacity was in place in 60 s vs
  165 s and p95 during the ramp fell 8%; no difference on the afternoon ramp
  and at peak; +13% replica-minutes. README states where the forecast did
  not help and why.
- The first run exposed a units bug: the CronJob trained on raw cores while
  the scaler feeds the model `demand / capacity`. Fixed in the chart, checked
  in CI.

### Stage 4 — portable deployment
- CI publishes `forecast-serving`, `predictive-scaler`, `cpu-burner` and
  `forecast-training` to GHCR (`<sha>`, `latest`) on pushes to `main`.
- `helm/predictive-stack`: the whole system in one release — MLflow,
  Prometheus (namespace-scoped discovery and RBAC), Grafana with the
  dashboards from `monitoring/`, forecast-serving, predictive-scaler, KEDA as
  a dependency, the retraining CronJob, the demo workload with
  `demo.scaling=none|predictive|hpa`, one Ingress per component host.
- `scripts/up.sh`: kind cluster → helm install → first model trained
  in-cluster → smoke tests of `/predict` and the scaler's gRPC API; replaces
  the Minikube-bound `start.sh` and its hard-coded node IP.
- `.github/workflows/e2e.yml`: `up.sh` on kind, ten minutes of ramped load,
  assertions that decisions came from forecasts and that the HPA rescaled;
  logs and metrics as artifacts.

### Stage 3 — observability of forecast quality and drift
- `serving/quality.py`: forecasts are scored against the actual values later
  windows carry — `forecast_abs_error` and `forecast_rolling_mae` by horizon
  step, in bounded memory; `forecast_model_info`. `/predict` accepts
  `series`, `timestamp` and `step_seconds`; the scaler sends them.
- Alerts `ForecastDriftHigh`, `ScalerFallbackRate`, `ScalerPrometheusErrors`;
  every alert links a runbook in `docs/runbooks/`.
- Dashboards as code (`monitoring/dashboards/`), validated and rendered by
  `monitoring/render.py` in CI.

## Stage 2 — retraining pipeline (PR #2)
- `training/forecast_training`: candidates (`rf_level`, `rf_delta`,
  `hgb_delta`, `ridge_delta`), level-tracking gate, persistence baseline,
  fair comparison with the champion on unseen data, MLflow aliases,
  `forecast_meta.json` contract, nightly CronJob, `/reload` after promotion.
- Serving: multi-step `predictions`, alias with stage fallback.

## Stage 1 — KEDA external scaler (PR #1)
- `scaler/predictive_scaler`: gRPC ExternalScaler reading Prometheus,
  calling `/predict`, reporting `max(forecast, observed)`; per-ScaledObject
  metrics; failure policy (observed-demand fallback, KEDA `fallback` on
  Prometheus errors). Demo workload `cpu-burner` and Helm chart.

## Stage 0 — forecasting service prototype
- FastAPI serving with MLflow model loading and hot reload, notebook
  training, Prometheus/Grafana/Loki manifests, Helm chart, CI.
