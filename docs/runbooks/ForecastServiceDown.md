# ForecastServiceDown

**Severity:** critical · **Source:** Prometheus (`up`)

## What it means

Prometheus has not been able to scrape forecast-serving for 1 minute, or the
target has disappeared altogether. While the service is down the predictive
scaler falls back to observed demand (`ScalerFallbackRate` will follow), so
workloads keep scaling reactively; nothing is pinned.

## How to check

1. `kubectl get pods -l app.kubernetes.io/name=forecast-serving` (Helm) or
   `-l app=cpu-serving` (raw manifests): CrashLoopBackOff, Pending, OOMKilled?
2. `kubectl describe pod ...` for image pull errors and probe failures. Note
   that `/ready` returning 503 does **not** stop scraping: the pod is scraped
   by its address, not through the Service, so `up == 0` means the process
   is not answering at all.
3. Service / scrape config: the static `cpu-serving:8080` target in
   `monitoring/prometheus-config.yaml` must match the Service name of the
   deployment in use.

## How to fix

- Crash on start: check logs; the service starts even without MLflow
  (degraded, `/ready` 503) so a crash is a bug or a bad image.
- Pending: resource requests vs node capacity.
- Wrong target name: fix the scrape config or the Service, then reload
  Prometheus (`kubectl rollout restart deploy/prometheus`).
