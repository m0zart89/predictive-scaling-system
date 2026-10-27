# Predictive autoscaling demo

Closes the loop from forecast to replicas:

```
cpu-burner pods ──/metrics──► Prometheus ◄── query_range ──┐
      ▲                                                     │
      │ replicas                                  predictive-scaler ── POST /predict ──► forecast-serving
      │                                                     │
     HPA ◄── external metric ◄── KEDA ◄── gRPC GetMetrics ──┘
```

`cpu-burner` burns CPU in proportion to request load. The predictive scaler
reads its total CPU demand from Prometheus, asks `forecast-serving` for the
next interval and reports `max(forecast, observed)` to KEDA, which sets
`replicas = ceil(demand / 0.35 cores)`.

## Prerequisites

The forecasting stack from the main [Quick Start](../README.md#quick-start)
running in the `default` namespace (MLflow, Prometheus, forecast-serving via
Helm, with a model loaded). Re-apply `monitoring/prometheus-config.yaml` if
it was deployed before this demo existed: it adds the pod-discovery job that
scrapes the `demo` and `keda` namespaces.

## Run

```bash
# 1. KEDA
helm repo add kedacore https://kedacore.github.io/charts
helm install keda kedacore/keda -n keda --create-namespace

# 2. Images (Minikube's Docker daemon)
eval $(minikube docker-env)
docker build -t predictive-scaler:latest scaler/
docker build -t cpu-burner:latest demo/cpu-burner/

# 3. Scaler (same namespace as KEDA; the ScaledObject points at it)
helm install predictive-scaler helm/predictive-scaler/ -n keda

# 4. Workload + ScaledObject
kubectl create namespace demo
kubectl apply -f demo/cpu-burner.yaml
kubectl apply -f demo/scaledobject.yaml

# 5. Load
kubectl -n demo run load --rm -it --image=busybox --restart=Never -- \
  sh -c 'while true; do wget -q -O- "http://cpu-burner:8080/work?ms=100" >/dev/null; done'
```

If you installed the serving chart via raw manifests (Option B), change
`forecastServiceURL` in `scaledobject.yaml` to
`http://cpu-serving.default.svc.cluster.local:8080`.

## Watch it

```bash
kubectl -n demo get hpa,scaledobject,pods -w

# What the scaler decided and why
kubectl -n keda logs deploy/predictive-scaler -f

# Observed vs predicted vs reported demand, as Prometheus sees it
kubectl -n keda port-forward svc/predictive-scaler 9100
curl -s localhost:9100/metrics | grep -E 'predictive_scaler_(observed|predicted|reported)_demand'
```

## Timing

The model was trained on 5-minute samples and expects 10 of them, so with the
default `stepSeconds: "300"` the scaler needs ~50 minutes of history before
the first forecast; until then it scales on observed demand (logged as
`source=observed`). For a quick demo set `stepSeconds: "30"` — the window
fills in five minutes and the forecast looks 30 s ahead instead of 5 min.

## Failure behaviour

| What breaks | What the scaler does | Visible as |
|---|---|---|
| forecast-serving down or not ready | reports observed demand (plain reactive scaling) | `predictive_scaler_forecast_failures_total` |
| fewer than `timeSteps` samples | reports observed demand | `predictive_scaler_decisions_total{source="observed"}` |
| Prometheus down / no samples | returns gRPC `UNAVAILABLE`; KEDA applies `fallback.replicas` (4) after 3 failures | `predictive_scaler_prometheus_failures_total`, KEDA events |
