# Changelog

Stages follow [docs/ROADMAP.md](docs/ROADMAP.md). Versions are git tags on
`main`; `v1.0.0` is the first release where every stage is in.

## v1.0.0 — stages 3–6 (PR #4)

### Stage 6 — polish
- Fresh-eye review of the repository: dead code and README/code mismatches
  fixed, tests added where the review found gaps.
- `ARCHITECTURE.md` rewritten around the three loops (scaling, model quality,
  retraining) with a decisions table; `CHANGELOG.md`; `docs/pitch.md`.
- `docs/ROADMAP.md`: stages 3–6 moved to Done.

### Stage 5 — benchmark: predictive vs reactive HPA
- `bench/load.py`: one day of traffic compressed into 40 minutes with sharp
  ramps, exported as `bench_*` metrics; `bench/run.sh` runs warm-up →
  training → CPU-HPA day → predictive day on one kind cluster;
  `bench/collect.py` and `bench/report.py` turn Prometheus series into
  `docs/benchmark.md` and charts.
- `.github/workflows/benchmark.yml`: manual and weekly; commits the report.
- README: results and the conditions under which the forecast does not help.

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
  the Minikube-bound `start.sh`. Raw manifests lost their NodePorts and the
  hard-coded node IP.
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
