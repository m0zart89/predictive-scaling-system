# Roadmap

Where the project is and what could come next. Every stage was built from a
ready-to-paste brief for an agentic coding session; the briefs are in the
history of this file.

## Done

| Stage | What | Where |
|---|---|---|
| 0 | Forecasting service prototype: FastAPI + MLflow, notebook training, Prometheus/Grafana manifests, Helm chart, CI | branch `milestone/v0.1.0` |
| 1 | **KEDA external scaler** closes the loop forecast → replicas; demo workload; failure policy (reactive fallback, KEDA `fallback` on Prometheus errors) | `scaler/`, `demo/`, `helm/predictive-scaler/` — PR #1 |
| 2 | **Retraining pipeline** with a level-tracking gate (the notebook's model was pinned to its training range), delta-target candidates, 30-minute horizon, fair champion comparison on unseen data, MLflow aliases, nightly CronJob | `training/` — PR #2 |
| 3 | **Observability of forecast quality and drift**: serving scores its own past forecasts (`forecast_rolling_mae` by horizon step, bounded memory); `ForecastDriftHigh`, `ScalerFallbackRate`, `ScalerPrometheusErrors`; runbook per alert; dashboards and rules as code, validated in CI | `serving/quality.py`, `monitoring/`, `docs/runbooks/` — PR #4 |
| 4 | **Portable deployment**: images on GHCR, `helm/predictive-stack` umbrella chart (KEDA as a dependency, Ingress, namespace RBAC), `scripts/up.sh` on kind, e2e workflow with scaling assertions | `helm/predictive-stack/`, `scripts/`, `.github/workflows/e2e.yml` — PR #4 |
| 5 | **Benchmark** predictive scaler vs CPU HPA on a compressed day with sharp ramps: p95 during ramps, slow requests, replica-minutes, time to first scale-up | `bench/`, `docs/benchmark.md`, `.github/workflows/benchmark.yml` — PR #4 |
| 6 | **Polish**: review fixes, one implementation of the feature transforms, one deployment path, ARCHITECTURE decisions table, CHANGELOG, pitch | `ARCHITECTURE.md`, `CHANGELOG.md`, `docs/pitch.md` — PR #4 |
| 7 | **After the audit (v1.1)**: time-of-day features, Prometheus on a volume + pilot profile, `workloads` in the chart, `/reload` token + NetworkPolicy, drift threshold from the model's holdout error, real traces (Wikipedia, CSV) | `training/`, `helm/predictive-stack/`, `.github/workflows/traces.yml` — PR #5 |

## Next

All planned stages are done. Release: tag `v1.0.0` on `main` after PR #4 is
merged (by the owner, through a GitHub release). The briefs the stages were
built from are in the git history of this file (`git log -p docs/ROADMAP.md`).

Candidates for a next stage, roughly by value:

- **Alert-driven retraining.** `ForecastDriftHigh` → Alertmanager webhook →
  `kubectl create job --from=cronjob/forecast-retrain`.
- **One model per workload.** Today one release trains one model
  (`training.workload`); services with different patterns need separate
  releases or per-workload models and aliases.
- **Day-of-week features** once there are two or more weeks of history
  (the weekly pattern is visible in the Wikipedia traces).

## Deferred ideas

- Additional signals (memory, queue depth) as features — needs a workload
  where CPU is not the bottleneck.
- Weekly seasonality once more than a month of data exists.
- Multi-tenant scaler config via CRD instead of trigger metadata.
