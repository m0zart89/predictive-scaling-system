#!/usr/bin/env bash
#
# End-to-end check on a running stack (see scripts/up.sh): put load on
# cpu-burner for LOAD_SECONDS with a ramp, then verify that
#
#   * the predictive scaler made decisions from forecasts
#     (predictive_scaler_decisions_total{source="forecast"} grew), and
#   * KEDA's HPA actually changed the replica count.
#
# Logs, events and the relevant Prometheus series are written to OUT_DIR for
# CI artifacts. Exit code is non-zero if either check fails.
#
# Environment:
#   NAMESPACE       release namespace                   (default: predictive)
#   DOMAIN          ingress domain                       (default: localhost)
#   LOAD_SECONDS    how long to load the workload        (default: 600)
#   RAMP_AT         seconds after which the load steps up (default: 180)
#   OUT_DIR         where to write artifacts             (default: e2e-artifacts)
#   REDIS           1 = also require forecast-serving's Redis quality backend
#                   to have answered (forecast_quality_backend_up) (default: 0)
set -euo pipefail

cd "$(dirname "$0")/.."

NAMESPACE=${NAMESPACE:-predictive}
DOMAIN=${DOMAIN:-localhost}
LOAD_SECONDS=${LOAD_SECONDS:-600}
RAMP_AT=${RAMP_AT:-180}
OUT_DIR=${OUT_DIR:-e2e-artifacts}
REDIS=${REDIS:-0}
PROM="http://prometheus.$DOMAIN"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
icurl() { local host=$1; shift; curl -sS --resolve "$host:80:127.0.0.1" "$@"; }
promq() {   # instant query → first sample value (or empty)
  icurl "prometheus.$DOMAIN" -G "$PROM/api/v1/query" --data-urlencode "query=$1" \
    | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(r[0]["value"][1] if r else "")'
}
k() { kubectl -n "$NAMESPACE" "$@"; }

mkdir -p "$OUT_DIR"
START=$(date +%s)

# ── baseline ───────────────────────────────────────────────────────────────────
before_forecast=$(promq 'sum(predictive_scaler_decisions_total{source="forecast"}) or vector(0)')
replicas_before=$(k get deploy cpu-burner -o jsonpath='{.status.replicas}')
log "before: forecast decisions=$before_forecast, cpu-burner replicas=$replicas_before"

# ── load ───────────────────────────────────────────────────────────────────────
# One pod runs batches of concurrent /work calls: 1 client, then 4 after RAMP_AT.
log "Loading cpu-burner for ${LOAD_SECONDS}s (ramp at ${RAMP_AT}s)"
k delete pod e2e-load --ignore-not-found >/dev/null
k run e2e-load --image=busybox:1.36 --restart=Never --command \
  --env="LOAD_SECONDS=$LOAD_SECONDS" --env="RAMP_AT=$RAMP_AT" -- sh -c '
start=$(date +%s)
while [ $(( $(date +%s) - start )) -lt "$LOAD_SECONDS" ]; do
  el=$(( $(date +%s) - start ))
  if [ "$el" -lt "$RAMP_AT" ]; then n=1; ms=100; else n=4; ms=200; fi
  i=0
  while [ $i -lt $n ]; do
    wget -q -O- -T 10 "http://cpu-burner:8080/work?ms=$ms" >/dev/null 2>&1 &
    i=$((i+1))
  done
  wait
done
echo "load finished"' >/dev/null

# Sample the HPA and pods every 30 s while the load runs.
: > "$OUT_DIR/hpa-timeline.txt"
while k get pod e2e-load -o jsonpath='{.status.phase}' 2>/dev/null | grep -qE 'Pending|Running'; do
  {
    echo "--- $(date -u '+%H:%M:%S') (+$(( $(date +%s) - START ))s)"
    k get hpa -o wide 2>/dev/null || true
    k get pods -l app=cpu-burner --no-headers 2>/dev/null | awk '{print $1, $3}' || true
  } >> "$OUT_DIR/hpa-timeline.txt"
  sleep 30
done
k logs pod/e2e-load > "$OUT_DIR/load.log" 2>&1 || true

# ── collect ────────────────────────────────────────────────────────────────────
log "Collecting artifacts into $OUT_DIR"
END=$(date +%s)
k get all -o wide > "$OUT_DIR/resources.txt" 2>&1 || true
k get hpa -o yaml > "$OUT_DIR/hpa.yaml" 2>&1 || true
k get scaledobject -o yaml > "$OUT_DIR/scaledobject.yaml" 2>&1 || true
k get events --sort-by=.lastTimestamp > "$OUT_DIR/events.txt" 2>&1 || true
for d in predictive-scaler forecast-serving cpu-burner mlflow prometheus grafana redis; do
  k logs "deploy/$d" --all-containers=true --tail=2000 > "$OUT_DIR/$d.log" 2>&1 || true
done
k logs deploy/keda-operator --tail=500 > "$OUT_DIR/keda-operator.log" 2>&1 || true
for q in \
  'predictive_scaler_observed_demand' 'predictive_scaler_predicted_demand' \
  'predictive_scaler_reported_demand' 'predictive_scaler_desired_replicas' \
  'predictive_scaler_smoothed_demand' 'predictive_scaler_spikes_suppressed_total' \
  'predictive_scaler_forecast_service_breaker_open' \
  'histogram_quantile(0.95, sum by (le) (rate(predictive_scaler_forecast_latency_seconds_bucket[2m])))' \
  'sum by (source) (predictive_scaler_decisions_total)' \
  'count(up{app="cpu-burner"} == 1)' 'forecast_rolling_mae' 'forecast_quality_backend_up' \
  'histogram_quantile(0.95, sum by (le) (rate(cpu_burner_request_duration_seconds_bucket[1m])))'; do
  name=$(echo "$q" | tr -c 'a-zA-Z0-9' '_' | sed 's/_*$//')
  icurl "prometheus.$DOMAIN" -G "$PROM/api/v1/query_range" --data-urlencode "query=$q" \
    --data-urlencode "start=$START" --data-urlencode "end=$END" --data-urlencode "step=15" \
    > "$OUT_DIR/series-$name.json" || true
done

# ── verify ─────────────────────────────────────────────────────────────────────
after_forecast=$(promq 'sum(predictive_scaler_decisions_total{source="forecast"}) or vector(0)')
observed=$(promq 'sum(predictive_scaler_decisions_total{source="observed"}) or vector(0)')
max_desired=$(promq "max_over_time(predictive_scaler_desired_replicas[$((LOAD_SECONDS + 120))s])")
max_pods=$(promq "max_over_time((count(up{app=\"cpu-burner\"} == 1))[$((LOAD_SECONDS + 120))s:15s])")
replicas_after=$(k get deploy cpu-burner -o jsonpath='{.status.replicas}')
rescales=$(k get events --field-selector reason=SuccessfulRescale -o name 2>/dev/null | wc -l)
mae=$(promq 'max(forecast_rolling_mae)')
quality_up=$(promq 'min(forecast_quality_backend_up)')
quality_errors=$(promq 'sum(forecast_quality_backend_errors_total) or vector(0)')
failures=$(icurl "prometheus.$DOMAIN" -G "$PROM/api/v1/query" --data-urlencode 'query=sum by (reason) (predictive_scaler_forecast_failures_total)' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(", ".join(f"{x[\"metric\"].get(\"reason\",\"?\")}={x[\"value\"][1]}" for x in r) or "none")' 2>/dev/null || echo "n/a")
trips=$(icurl "prometheus.$DOMAIN" -G "$PROM/api/v1/query" --data-urlencode 'query=sum by (reason) (predictive_scaler_breaker_trips_total)' \
  | python3 -c 'import json,sys; r=json.load(sys.stdin)["data"]["result"]; print(", ".join(f"{x[\"metric\"].get(\"reason\",\"?\")}={x[\"value\"][1]}" for x in r) or "none")' 2>/dev/null || echo "n/a")
breaker_open=$(promq 'max(predictive_scaler_breaker_open) or vector(0)')
suppressed=$(promq 'sum(predictive_scaler_spikes_suppressed_total) or vector(0)')
service_trips=$(promq 'sum(predictive_scaler_forecast_service_breaker_trips_total) or vector(0)')
forecast_p95=$(promq "histogram_quantile(0.95, sum by (le) (rate(predictive_scaler_forecast_latency_seconds_bucket[$((LOAD_SECONDS + 120))s])))")
scaler_mae=$(promq 'max(rate(predictive_scaler_forecast_abs_error_sum[10m]) / rate(predictive_scaler_forecast_abs_error_count[10m]))')

{
  echo "forecast decisions: $before_forecast -> $after_forecast (observed-demand decisions: $observed)"
  echo "forecast failures by reason: $failures; breaker trips by reason: $trips; breaker open now: ${breaker_open:-n/a}"
  echo "spikes suppressed by the gate: ${suppressed:-n/a}; forecast-service breaker trips: ${service_trips:-n/a}; forecast call p95: ${forecast_p95:-n/a}s"
  echo "scaler-side forecast MAE, step max over 10m (model units): ${scaler_mae:-n/a}"
  echo "desired replicas (max): ${max_desired:-n/a}; scraped cpu-burner pods (max): ${max_pods:-n/a}"
  echo "cpu-burner replicas: $replicas_before -> $replicas_after; SuccessfulRescale events: $rescales"
  echo "forecast rolling MAE (max over steps): ${mae:-n/a}"
  echo "quality backend (redis) up: ${quality_up:-n/a}; backend errors: ${quality_errors:-n/a}"
} | tee "$OUT_DIR/summary.txt"

fail=0
if ! python3 -c "import sys; sys.exit(0 if float('${after_forecast:-0}') > float('${before_forecast:-0}') else 1)"; then
  echo "FAIL: predictive_scaler_decisions_total{source=\"forecast\"} did not grow"; fail=1
fi
if [ "$rescales" -lt 1 ] && ! python3 -c "import sys; sys.exit(0 if float('${max_pods:-0}') > 1 else 1)"; then
  echo "FAIL: the HPA never changed the replica count"; fail=1
fi
if [ "$REDIS" = "1" ]; then
  # the serving replicas scored their forecasts in Redis: the backend answered
  # and the shared rolling MAE exists
  if [ "${quality_up:-}" != "1" ]; then
    echo "FAIL: forecast_quality_backend_up is '${quality_up:-n/a}', expected 1"; fail=1
  fi
  if [ -z "${mae:-}" ]; then
    echo "FAIL: forecast_rolling_mae never appeared with the redis backend"; fail=1
  fi
fi
[ "$fail" = 0 ] && log "e2e OK" || { log "e2e FAILED"; exit 1; }
