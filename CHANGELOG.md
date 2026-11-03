# Changelog

Stages follow [docs/ROADMAP.md](docs/ROADMAP.md). Versions are git tags on
`main`; `v1.0.0` is the first release where every stage is in.

## v1.2.0 — production hardening (PRs #8–#16)

### Relative delta alignment (PR #16) — docs/relative-delta-alignment.md applied

- **Models speak fractions of capacity.** `--units capacity_fraction
  --capacity-reference C` (`TRAIN_UNITS`, `TRAIN_CAPACITY_REFERENCE`, set by
  the chart's CronJob from the workload's `capacity`): the target channel is
  not re-scaled (`CapacityFractionScaler`), so the delta target is exactly
  `(CPU_{t+k} − CPU_t) / C`; `forecast_meta.json` format 5 and the version
  tags carry `units` and `capacity_reference`; a series above 1.5 × C is
  refused. `scaled` (MinMax, formats ≤ 4) stays the default and keeps
  loading.
- **Serving says what its numbers are:** `units` and `capacity_reference`
  in `/predict` and `/model-info`; a bundle claiming fractions with a
  non-identity target scaler is rejected at load.
- **Scaler: reference capacity at runtime.** `capacityQuery` (PromQL for
  the workload's envelope in cores, cached `capacityRefreshSeconds`, last
  good value then the static `capacity` on failure), the forecast
  coefficient × that reference, the ledger scored against the reference
  each forecast was made with, `predictive_scaler_capacity_reference{source}`
  and `predictive_scaler_capacity_drift` with `capacityDriftFactor`;
  `event=capacity_fallback|capacity_changed|capacity_drift|forecast_scaled`
  logs. Chart keys per workload; CI checks the rendering.

### Model warm-up (PR #13)

- **A model is warmed up before it is swapped in.** After the download
  (registry or cache) and before the reference swap, `warm_up` runs one
  forecast on a synthetic window shaped like a real request: the first
  call's lazy state (prediction buffers, threadpools, the feature
  pipeline's first allocation) is paid off the request path, and a version
  that loads but cannot forecast (wrong horizon, non-finite output, an
  exception) is rejected while the current model keeps serving
  (`forecast_model_reloads_total{result="failure"}`).
  `forecast_model_warmup_seconds` reports the last warm-up.

### Scaler metric conditioning (PR #12)

- **EWMA on the forecast.** `smoothingAlpha` (0.3): `S = α·Y + (1−α)·S_prev`
  on the forecast demand before it meets observed demand, per ScaledObject,
  thread-safe, seeded again after `smoothingMaxAgeSeconds` without a
  forecast; `predictive_scaler_smoothed_demand`. `mode=max` still floors at
  observed demand.
- **Hysteresis gate.** `spikeMinSteps` (3): a forecast that needs more
  replicas than observed demand for fewer than that many consecutive
  horizon steps is noise; the decision is observed demand
  (`predictive_scaler_spikes_suppressed_total`).
- **Forecast-service breaker.** `forecastTimeoutSeconds` (0.2) for the
  `/predict` call; timeouts and 5xx get their own failure reasons; after
  `forecastBreakerFailures` (3) failures in a row the service is not called
  for `forecastBreakerOpenSeconds` (30) and the scaler is reactive
  (`predictive_scaler_forecast_service_breaker_open`, `_trips_total`,
  `predictive_scaler_forecast_skipped_total`).
- Decisions are logged as `event=… key=value` lines. The benchmark numbers in
  `docs/benchmark.md` predate the smoothing; the next `benchmark` run shows
  what α = 0.3 costs in lead time.
### Quality state in Redis (PR #11)

- **`serving/quality.py` has a Redis backend.** `RedisMetricsEngine` keeps
  the forecasts waiting to be scored and the rolling error windows in Redis
  instead of the pod: every forecast is written with a 24 h TTL
  (`forecast:{series}:{timestamp}`, `SET … EX NX`), each (forecast, step) is
  claimed once across replicas and its error pushed into
  `metrics:mae:{model_version}:{horizon_idx}` with `LPUSH` + `LTRIM` on the
  server, bounded by `FORECAST_MAE_WINDOW`; two pipelined round trips per
  `/predict`, off the event loop. A Redis outage is logged once, counted
  (`forecast_quality_backend_up`, `forecast_quality_backend_errors_total`,
  `ForecastQualityBackendDown`) and skipped for a cooldown; `/predict` never
  fails because of it. `QUALITY_BACKEND=memory` (the bounded in-pod tracker)
  stays the default without `REDIS_URL`.
- **Chart:** `global.quality.backend: redis` renders a bundled Redis
  (`redis.enabled`, no persistence, `volatile-ttl`) with a NetworkPolicy, or
  points every replica at `global.quality.redisUrl`; `values-prod.yaml` uses
  it instead of switching the tracker off. `scripts/up.sh REDIS=1`; the kind
  e2e runs with it and checks the backend answered.

### Serving lifecycle (PR #10)

- **No more `/reload`.** The endpoint, its token Secret, the training jobs'
  reload call and the client certificate for it are gone. A lifecycle
  worker in every replica polls the registry alias (`MODEL_REFRESH_SECONDS`)
  and hot-swaps the model under an `asyncio.Lock`; startup fetches the
  champion once before the port opens (`STARTUP_LOAD_TIMEOUT`) and a
  registry outage leaves the pod alive, unready, serving the cache when
  there is one, retrying every `MODEL_RETRY_SECONDS`.
- **Async inference.** `POST /predict` is `async`; scaling, features and
  the scikit-learn predict run in a bounded thread pool
  (`INFERENCE_THREADS`), never on the event loop. `STRICT_INPUT_LENGTH`
  rejects inputs longer than `input_length` (default: the tail is used).
- **Probes:** `/healthz` and `/readyz` (200 with the served version, 503
  without a model); `/live` and `/ready` remain as aliases.
  `ForecastReloadErrorRate` now watches `forecast_model_reloads_total`.

### Production (PR #9) — docs/production-blueprint.md applied

- **HA without shared state.** The scaler scores its own forecasts from the
  history it fetches (`predictive_scaler_forecast_abs_error{step}`,
  `ForecastDriftHigh` reads it); serving polls the registry alias
  (`MODEL_REFRESH_SECONDS`) and keeps a last-known-good model cache
  (`MODEL_CACHE_DIR`); subcharts 0.3.0 with PDBs, rolling-update strategy,
  anti-affinity, spread, startup probes, read-only root; `values-prod.yaml`.
- **MLflow on PostgreSQL + object storage** (`mlflow.backendStore`,
  `mlflow.artifacts`, replicas, PDB, `mlflow db upgrade` hook, basic-auth,
  client credentials); `mlflow/Dockerfile` → `mlflow-server` image.
- **Prometheus modes:** builtin, Prometheus Operator CR with Thanos sidecar,
  external (Thanos Query / VictoriaMetrics); `workload:demand` recording
  rules per workload and `prometheus.recordDemand`; `prometheus.trainingUrl`.
- **Circuit breaker** in the scaler (residual, level, jump signals; open /
  half-open / closed with hold and backoff), `forecastSurgeFactor`,
  `reactiveFloor` CPU trigger, `ForecastBreakerOpenLong` alert and runbook,
  `SPIKE` in the benchmark load generator.
- **Mutual TLS** on both loops from cert-manager (`global.tls`): scaler gRPC
  server with a plaintext health port, scaler and training clients, uvicorn
  with client-certificate verification and a plain metrics port, the
  forecast ingress with backend mTLS, KEDA `TriggerAuthentication`;
  default-deny ingress network policies; cert-manager in `up.sh TLS=1` and
  the kind e2e.
- **Alert-driven retraining:** `forecast_training.trigger` (Alertmanager
  webhook → Job cloned from the CronJob, breaker intervals excluded,
  `TRAIN_REASON` tag), `training.trigger` and `alertmanager` in the chart.
- **Multivariate channels:** `channels` in the model contract (format 4),
  per-channel scalers, `--channel NAME=PromQL`, `/predict` with one list per
  channel, scaler `channels`/`targetChannel`, `workloads[].channels`.

### Models (PR #8)

- **Selection by cost, not RMSE.** Candidates are ranked by an asymmetric
  cost: a unit of demand the forecast missed is worth `--under-weight`
  (default 3) idle units; `coverage` is logged next to RMSE and MAE. The
  champion comparison uses the same cost. `hgb_q75_delta` estimates the
  75th percentile of the change, the forecast that minimises that cost.
- **Seasonal persistence baseline.** "The same time one period ago" is
  scored on every run next to plain persistence; `--baseline
  persistence|seasonal|best` says which the winner must beat. The
  real-traces workflow uses `best` and prints both.
- **Seasonal lags.** `anchored_season` features add the level one cycle ago
  and the change that followed it over the horizon; `hgb_delta_season`,
  `rf_delta_season`, `hgb_q75_delta_season`. The input grows to
  `time_steps + season_steps` samples (`forecast_meta.json`, format 3;
  `/model-info` reports `input_length`). `/predict` takes at least that many
  values, uses the tail and interpolates `null` gaps in the older part
  (`forecast_input_filled_total`). The scaler fetches `historySteps` grid
  instants (holes as `null`) and the umbrella chart sets it to the window
  plus `periodSeconds / stepSeconds` per workload. Existing ScaledObjects
  and models are unaffected: `historySteps` defaults to `timeSteps`,
  `season_steps` to 0.
- **Benchmark.** With selection by cost the morning ramp had its capacity in
  45 s against the HPA's 180 s (v1.1: 105 s), p95 during the ramp −6%,
  replica-minutes +13%; the afternoon ramp and the peak are unchanged.
  `/model-info` now names the promoted `candidate`.
- **Real traces, re-read.** With the seasonal baseline the Wikipedia result
  is 11–24% better than "the same hour yesterday" (not 76–85% better than
  persistence); `hgb_q75_delta_season` wins de.wikipedia, the quantile models
  cover 0.72–0.78 of the actuals against 0.58–0.68 for the mean models
  (`docs/traces.md`).

## v1.1.0 — after the audit (PR #5)

- **Time-of-day features.** `hgb_delta_tod` / `rf_delta_tod` add sin/cos of
  the last sample's phase in the seasonal cycle (`period_seconds`); serving
  evaluates it for the request's timestamp. On the synthetic daily pattern
  the timed model beats the same model without the phase at the 30-minute
  horizon; in the benchmark it won the training but did not widen the lead
  over the HPA (105 s vs 180 s to the ramp's capacity, p95 −2%, replica-
  minutes +11%); on hourly Wikipedia traffic it is within ±1% of the plain
  model.
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
- **Real traces.** `forecast_training.traces` converts hourly Wikimedia page
  views and timestamp/value CSVs (Alibaba and the like); `traces.yml` trains
  on three projects weekly and fails unless a candidate beats persistence.
  First run: delta models 76–85% better than persistence six hours out; the
  gate rejects a pinned model with the best RMSE; `_tod` features add
  nothing with a 24-hour window (`docs/traces.md`).

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
