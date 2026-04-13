#!/bin/bash
set -euo pipefail

MINIKUBE_IP="192.168.49.2"
SERVING_URL="http://${MINIKUBE_IP}:30800"
MLFLOW_URL="http://${MINIKUBE_IP}:30500"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_pod() {
  local label=$1
  local timeout=${2:-120}
  log "Waiting for pod: $label ..."
  kubectl wait --for=condition=ready pod -l "app=$label" --timeout="${timeout}s"
}

wait_for_http() {
  local url=$1
  local timeout=${2:-60}
  log "Waiting for HTTP: $url ..."
  for i in $(seq 1 $timeout); do
    if curl -sf "$url" -o /dev/null 2>/dev/null; then
      return 0
    fi
    sleep 1
  done
  echo "ERROR: $url did not respond in ${timeout}s" >&2
  return 1
}

# ── 1. Minikube ────────────────────────────────────────────────────────────────
if ! minikube status | grep -q "Running"; then
  log "Starting minikube..."
  minikube start
else
  log "Minikube already running."
fi

# ── 2. Docker image ────────────────────────────────────────────────────────────
log "Building cpu-serving image inside minikube..."
eval "$(minikube docker-env)"
export DOCKER_API_VERSION=1.43
docker build -t cpu-serving:latest serving/ -q
log "Image built."

# ── 3. Deploy ──────────────────────────────────────────────────────────────────
log "Applying manifests..."
kubectl apply -f mlflow/k8s.yaml
kubectl apply -f monitoring/prometheus-config.yaml
kubectl apply -f monitoring/prometheus-rules.yaml
kubectl apply -f monitoring/grafana.yaml
kubectl apply -f k8s/serving.yaml

# ── 4. Wait for readiness ─────────────────────────────────────────────────────
wait_for_pod mlflow 180
wait_for_pod prometheus 60
wait_for_pod grafana 60
wait_for_pod cpu-serving 60

# ── 5. Reload model in serving ────────────────────────────────────────────────
wait_for_http "${MLFLOW_URL}/health" 30
log "Triggering model reload in cpu-serving..."
RELOAD_BODY=$(curl -s -o /tmp/reload_resp.json -w "%{http_code}" -X POST "${SERVING_URL}/reload")
if [ "$RELOAD_BODY" = "200" ]; then
  python3 -c "import json; d=json.load(open('/tmp/reload_resp.json')); print(f'  model reload: {d}')"
else
  log "WARNING: /reload returned HTTP $RELOAD_BODY — model may already be loaded at startup"
  cat /tmp/reload_resp.json 2>/dev/null || true
fi

# ── 6. Smoke test ─────────────────────────────────────────────────────────────
log "Smoke test /predict..."
RESULT=$(curl -sf -X POST "${SERVING_URL}/predict" \
  -H 'Content-Type: application/json' \
  -d '{"values":[0.2,0.25,0.3,0.28,0.22,0.35,0.4,0.38,0.33,0.29]}')
echo "  prediction: $RESULT"

# ── 7. Summary ────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  MLflow       http://${MINIKUBE_IP}:30500"
echo "  Serving      http://${MINIKUBE_IP}:30800"
echo "  Prometheus   http://${MINIKUBE_IP}:30900"
echo "  Grafana      http://${MINIKUBE_IP}:30300"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "Done."
