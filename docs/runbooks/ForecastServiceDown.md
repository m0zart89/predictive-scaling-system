# ForecastServiceDown

**Severity:** critical · **Source:** Prometheus (`up`)

## What it means

Prometheus has not been able to scrape forecast-serving for 1 minute, or the
target has disappeared altogether. While the service is down the predictive
scaler falls back to observed demand (`ScalerFallbackRate` will follow), so
workloads keep scaling reactively; nothing is pinned.

## How to check

1. `kubectl -n predictive get pods -l app.kubernetes.io/name=forecast-serving`:
   CrashLoopBackOff, Pending, OOMKilled?
2. `kubectl describe pod ...` for image pull errors and probe failures. Note
   that `/ready` returning 503 does **not** stop scraping: the pod is scraped
   by its address, not through the Service, so `up == 0` means the process
   is not answering at all.
3. Scrape config: the pod must carry the `prometheus.io/scrape` annotations
   and live in the namespace Prometheus discovers (the release namespace;
   see the `kubernetes-pods` job in `helm/predictive-stack/templates/prometheus.yaml`).

## How to fix

- Crash on start: check logs; the service starts even without MLflow
  (degraded, `/ready` 503) so a crash is a bug or a bad image.
- Pending: resource requests vs node capacity.
- Missing annotations or wrong namespace: fix the Deployment, then restart
  Prometheus if its config changed (`kubectl -n predictive rollout restart deploy/prometheus`).
