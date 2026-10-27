# ScalerPrometheusErrors

**Severity:** critical · **Source:** predictive-scaler (`predictive_scaler_prometheus_failures_total`)

## What it means

The scaler could not read demand for a ScaledObject from Prometheus for
5 minutes — the query failed, timed out, returned no samples, or returned
more than one series. Without data the scaler does **not** guess: it answers
KEDA with gRPC `UNAVAILABLE`, and after `fallback.failureThreshold` failures
KEDA pins the workload to `fallback.replicas` (4 in the demo). The workload is
no longer autoscaled in either direction until this clears.

## How to check

1. Scaler logs name the failure: `kubectl -n keda logs deploy/predictive-scaler
   | grep -i prometheus` — `unreachable`, `HTTP 4xx/5xx`, `returned N series`
   or `no samples`.
2. Run the ScaledObject's `query` in the Prometheus UI over the last
   `timeSteps × stepSeconds`:
   - **Error:** PromQL mistake in the trigger metadata.
   - **Empty:** the workload's pods are not being scraped — check the pod
     annotations (`prometheus.io/scrape`, `port`, `path`) and that Prometheus
     discovers the namespace (see the `kubernetes-pods` job in
     `monitoring/prometheus-config.yaml`).
   - **Several series:** wrap the query in `sum(...)`.
3. `prometheusServerAddress` in the trigger must be reachable from the
   scaler's namespace: `kubectl -n keda exec deploy/predictive-scaler --
   python -c "import urllib.request;print(urllib.request.urlopen('http://prometheus.default.svc.cluster.local:9090/-/ready').status)"`.
4. Prometheus itself: `kubectl get pods -l app=prometheus`, restarts, OOM.

## How to fix

- Fix the query or the address in the ScaledObject; KEDA passes the new
  metadata on the next poll.
- Restore scraping of the workload (annotations / discovery / RBAC).
- Restore Prometheus. While it is down the KEDA `fallback` keeps the
  workload at a fixed size; raise `fallback.replicas` if that size is too
  small for the current load.
- The alert clears once queries succeed again; KEDA resumes scaling on the
  scaler's value after the next successful call.
