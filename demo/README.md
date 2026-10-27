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

## Run

The whole stack, demo included, comes up with one command (see the main
[Quick Start](../README.md#quick-start)):

```bash
STEP_SECONDS=30 ./scripts/up.sh      # 30 s steps: the forecast window fills in 5 min
```

Then put load on the workload:

```bash
kubectl -n predictive run load --rm -it --image=busybox:1.36 --restart=Never -- \
  sh -c 'while true; do wget -q -O- "http://cpu-burner:8080/work?ms=100" >/dev/null; done'
```

`scripts/e2e.sh` does the same with a ramp for 10 minutes and checks that the
scaler's decisions came from forecasts and that the HPA changed replicas.

### Raw manifests

`cpu-burner.yaml` and `scaledobject.yaml` are the `kubectl apply` variant for
a cluster where KEDA (`helm install keda kedacore/keda -n keda
--create-namespace`), the scaler (`helm install predictive-scaler
helm/predictive-scaler/ -n keda`) and the serving stack from the raw manifests
already run; they assume the `demo` namespace and the GHCR images.

## Watch it

```bash
kubectl -n predictive get hpa,scaledobject,pods -w

# What the scaler decided and why
kubectl -n predictive logs deploy/predictive-scaler -f

# Observed vs predicted vs reported demand, desired replicas, fallback rate
open http://grafana.localhost/d/predictive-scaling
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
