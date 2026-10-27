#!/usr/bin/env bash
#
# Predictive scaler vs reactive HPA on one kind cluster (see bench/README.md).
#
#   warm-up (HPA, WARMUP_DAYS compressed days) → train on it → reset
#   → run A: plain CPU HPA, one day → reset → run B: predictive scaler, one day
#   → collect both from Prometheus → bench/report.py → docs/benchmark.md + PNG
#
# Expects the stack from scripts/up.sh (run with TRAIN=0 so the model comes
# from the warm-up, STEP_SECONDS=30, DEMO_SCALING=none). bench/run.sh is what
# .github/workflows/benchmark.yml runs; it works the same on a laptop.
#
# Environment:
#   NAMESPACE, RELEASE, DOMAIN     as in up.sh                  (predictive / predictive / localhost)
#   IMAGE_TAG, REGISTRY            cpu-burner image for the load generator
#   DAY_SECONDS                    length of one compressed day (default: 2400 = 40 min)
#   WARMUP_DAYS                    days of warm-up load to train on (default: 2)
#   STEP_SECONDS                   scaler / training step        (default: 30)
#   MAX_REPLICAS                   demo.maxReplicas              (default: 6)
#   WORK_MS, THINK_MS              per-request CPU work and think time (50 / 100)
#   OUT_DIR                        results directory             (default: bench-results)
#   REPORT_MD, PNG_DIR             report outputs (docs/benchmark.md, docs/benchmark)
#   SKIP_WARMUP=1                  reuse an existing champion (shorter manual runs)
set -euo pipefail

cd "$(dirname "$0")/.."

NAMESPACE=${NAMESPACE:-predictive}
RELEASE=${RELEASE:-predictive}
DOMAIN=${DOMAIN:-localhost}
IMAGE_TAG=${IMAGE_TAG:-dev}
REGISTRY=${REGISTRY:-ghcr.io/m0zart89/predictive-scaling-system}
DAY_SECONDS=${DAY_SECONDS:-2400}
WARMUP_DAYS=${WARMUP_DAYS:-2}
STEP_SECONDS=${STEP_SECONDS:-30}
MAX_REPLICAS=${MAX_REPLICAS:-6}
WORK_MS=${WORK_MS:-50}
THINK_MS=${THINK_MS:-100}
OUT_DIR=${OUT_DIR:-bench-results}
REPORT_MD=${REPORT_MD:-docs/benchmark.md}
PNG_DIR=${PNG_DIR:-docs/benchmark}
SKIP_WARMUP=${SKIP_WARMUP:-0}
RESET_SECONDS=${RESET_SECONDS:-90}

log() { echo "[$(date '+%H:%M:%S')] $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }
k() { kubectl -n "$NAMESPACE" "$@"; }
helm_set() {   # helm_set key=value ...
  local args=()
  for kv in "$@"; do args+=(--set "$kv"); done
  # no --wait: forecast-serving is unready until the warm-up model is trained
  helm upgrade "$RELEASE" helm/predictive-stack -n "$NAMESPACE" --reuse-values "${args[@]}" \
    --timeout 10m >/dev/null
  k rollout status deploy/cpu-burner --timeout=300s >/dev/null
}

load_phase() {   # load_phase <name> <days>
  local phase=$1 days=$2
  k delete job "bench-load-$phase" --ignore-not-found >/dev/null
  PHASE=$phase NAMESPACE=$NAMESPACE IMAGE="$REGISTRY/cpu-burner:$IMAGE_TAG" DAYS=$days \
    DAY_SECONDS=$DAY_SECONDS WORK_MS=$WORK_MS THINK_MS=$THINK_MS \
    envsubst < bench/load-job.yaml | kubectl apply -f - >/dev/null
  local timeout=$(( days * DAY_SECONDS + 600 ))
  log "load '$phase': $days day(s) x ${DAY_SECONDS}s (timeout ${timeout}s)"
  k wait --for=condition=complete "job/bench-load-$phase" --timeout="${timeout}s" >/dev/null \
    || { k logs "job/bench-load-$phase" | tail -20; die "load job $phase did not complete"; }
  k logs "job/bench-load-$phase" > "$OUT_DIR/load-$phase.log" 2>&1 || true
}

reset_workload() {
  log "reset: scaling=none, 1 replica, ${RESET_SECONDS}s quiet"
  helm_set demo.scaling=none
  k scale deploy/cpu-burner --replicas=1 >/dev/null
  k rollout status deploy/cpu-burner --timeout=120s >/dev/null
  sleep "$RESET_SECONDS"
}

mkdir -p "$OUT_DIR" "$PNG_DIR"
command -v envsubst >/dev/null || die "envsubst (gettext) is required"
k create configmap bench-load --from-file=load.py=bench/load.py --dry-run=client -o yaml \
  | kubectl apply -f - >/dev/null
helm_set demo.maxReplicas="$MAX_REPLICAS" demo.predictive.stepSeconds="$STEP_SECONDS" \
  demo.predictive.mode=max demo.predictive.forecastAggregation=max

# ── warm-up + training ─────────────────────────────────────────────────────────
if [ "$SKIP_WARMUP" != "1" ]; then
  log "warm-up under the plain HPA so the training data has the workload's real dynamics"
  helm_set demo.scaling=hpa
  load_phase warmup "$WARMUP_DAYS"

  log "training on the warm-up data (${STEP_SECONDS}s step)"
  days=$(python3 -c "print($WARMUP_DAYS * $DAY_SECONDS / 86400 + 0.002)")
  helm_set training.cronjob.stepSeconds="$STEP_SECONDS" training.cronjob.days="$days" \
    "training.cronjob.query=sum(rate(process_cpu_seconds_total{app=\"cpu-burner\"}[1m]))" \
    training.cronjob.minImprovement=0
  k delete job bench-train --ignore-not-found >/dev/null
  k create job bench-train --from=cronjob/forecast-retrain >/dev/null
  k wait --for=condition=complete job/bench-train --timeout=20m >/dev/null \
    || { k logs job/bench-train | tail -30; die "training failed"; }
  k logs job/bench-train > "$OUT_DIR/train.log" 2>&1
  grep -q '"decision": "promoted"' "$OUT_DIR/train.log" \
    || { tail -30 "$OUT_DIR/train.log"; die "training did not promote a model"; }
  # wait until forecast-serving reports the new version
  for i in $(seq 1 30); do
    curl -sS --resolve "forecast.$DOMAIN:80:127.0.0.1" "http://forecast.$DOMAIN/model-info" \
      | grep -q '"model_loaded": *true' && break
    sleep 5
  done
  curl -sS --resolve "forecast.$DOMAIN:80:127.0.0.1" "http://forecast.$DOMAIN/model-info" \
    | tee "$OUT_DIR/model-info.json"; echo
  reset_workload
fi

meta=(--meta "commit=${GITHUB_SHA:-$(git rev-parse HEAD 2>/dev/null || echo unknown)}"
      --meta "date=$(date -u +%Y-%m-%d)" --meta "day_seconds=$DAY_SECONDS"
      --meta "step_seconds=$STEP_SECONDS" --meta "max_replicas=$MAX_REPLICAS"
      --meta "work_ms=$WORK_MS" --meta "think_ms=$THINK_MS")

# ── run A: reactive HPA ────────────────────────────────────────────────────────
log "run A: plain CPU HPA (70%)"
helm_set demo.scaling=hpa
sleep 30
A_START=$(date +%s)
load_phase hpa 1
A_END=$(date +%s)
k get hpa -o yaml > "$OUT_DIR/hpa-a.yaml" 2>&1 || true
reset_workload

# ── run B: predictive ──────────────────────────────────────────────────────────
log "run B: predictive scaler (mode=max, forecastAggregation=max)"
helm_set demo.scaling=predictive
sleep 30
B_START=$(date +%s)
load_phase predictive 1
B_END=$(date +%s)
k get hpa,scaledobject -o yaml > "$OUT_DIR/hpa-b.yaml" 2>&1 || true
k logs deploy/predictive-scaler --tail=3000 > "$OUT_DIR/predictive-scaler.log" 2>&1 || true
k logs deploy/forecast-serving --tail=1000 > "$OUT_DIR/forecast-serving.log" 2>&1 || true

# ── collect + report ───────────────────────────────────────────────────────────
log "collecting from Prometheus"
sleep 30   # last scrapes
python3 bench/collect.py --label hpa --start "$A_START" --end "$A_END" --host "prometheus.$DOMAIN" \
  --out "$OUT_DIR/hpa.json" "${meta[@]}"
python3 bench/collect.py --label predictive --start "$B_START" --end "$B_END" --host "prometheus.$DOMAIN" \
  --out "$OUT_DIR/predictive.json" "${meta[@]}"

log "report → $REPORT_MD, $PNG_DIR/"
python3 bench/report.py "$OUT_DIR/hpa.json" "$OUT_DIR/predictive.json" \
  --out "$REPORT_MD" --png-dir "$PNG_DIR" --json "$OUT_DIR/summary.json"
cp "$REPORT_MD" "$OUT_DIR/benchmark.md"
cp "$PNG_DIR"/*.png "$OUT_DIR/" 2>/dev/null || true
log "Done."
