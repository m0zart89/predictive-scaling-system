#!/usr/bin/env bash
#
# Bring the whole predictive autoscaling stack up on a local kind cluster:
#
#   kind cluster (+ ingress-nginx, metrics-server) → images → helm install
#   → train the first model from notebooks/data-jan-2022.json → /reload
#   → smoke-test /predict and the scaler's gRPC GetMetrics
#
# Reproducible on a laptop and in CI (.github/workflows/e2e.yml). Needs
# docker, kind, kubectl, helm, curl. Idempotent: re-running upgrades the
# release in place.
#
# Environment:
#   CLUSTER_NAME   kind cluster name                     (default: predictive)
#   NAMESPACE      release namespace                     (default: predictive)
#   RELEASE        helm release name                     (default: predictive)
#   IMAGE_TAG      tag of the four project images        (default: dev)
#   BUILD_IMAGES   1 = docker build + kind load; 0 = pull from GHCR
#                                                         (default: 1 if IMAGE_TAG=dev, else 0)
#   STEP_SECONDS   ScaledObject stepSeconds; 30 compresses time for demos/CI (default: 300)
#   DEMO_SCALING   predictive | hpa | none               (default: predictive)
#   MAX_REPLICAS   demo.maxReplicas                      (default: 10)
#   TRAIN          1 = train the first model from the saved export (default: 1)
#   SMOKE          1 = run the smoke tests at the end     (default: 1)
#   TLS            1 = install cert-manager, create a self-signed CA issuer
#                  (predictive-ca) and enable mutual TLS on both internal loops
#                  (global.tls.enabled)                   (default: 0)
#   HELM_EXTRA     extra arguments for helm upgrade (e.g. "--set grafana.anonymous=false")
#   DOMAIN         ingress domain                         (default: localhost)
set -euo pipefail

cd "$(dirname "$0")/.."

CLUSTER_NAME=${CLUSTER_NAME:-predictive}
NAMESPACE=${NAMESPACE:-predictive}
RELEASE=${RELEASE:-predictive}
IMAGE_TAG=${IMAGE_TAG:-dev}
if [ "${BUILD_IMAGES:-}" = "" ]; then
  [ "$IMAGE_TAG" = "dev" ] && BUILD_IMAGES=1 || BUILD_IMAGES=0
fi
STEP_SECONDS=${STEP_SECONDS:-300}
DEMO_SCALING=${DEMO_SCALING:-predictive}
MAX_REPLICAS=${MAX_REPLICAS:-10}
TRAIN=${TRAIN:-1}
TLS=${TLS:-0}
CERT_MANAGER_VERSION=${CERT_MANAGER_VERSION:-v1.16.3}
SMOKE=${SMOKE:-1}
HELM_EXTRA=${HELM_EXTRA:-}
DOMAIN=${DOMAIN:-localhost}
REGISTRY=${REGISTRY:-ghcr.io/m0zart89/predictive-scaling-system}

INGRESS_NGINX_VERSION=${INGRESS_NGINX_VERSION:-controller-v1.15.1}
METRICS_SERVER_VERSION=${METRICS_SERVER_VERSION:-v0.9.0}

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
die()  { echo "ERROR: $*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "$1 is required"; }

# Reach a host-named Ingress without DNS: resolve <name>.$DOMAIN to 127.0.0.1.
icurl() {
  local host=$1; shift
  curl -sS --resolve "$host:80:127.0.0.1" "$@"
}

for tool in docker kind kubectl helm curl; do need "$tool"; done

# ── 1. cluster ─────────────────────────────────────────────────────────────────
if kind get clusters 2>/dev/null | grep -qx "$CLUSTER_NAME"; then
  log "kind cluster '$CLUSTER_NAME' already exists"
else
  log "Creating kind cluster '$CLUSTER_NAME'"
  kind create cluster --name "$CLUSTER_NAME" --config scripts/kind-config.yaml --wait 120s
fi
kubectl config use-context "kind-$CLUSTER_NAME" >/dev/null

log "Installing ingress-nginx ($INGRESS_NGINX_VERSION)"
kubectl apply -f "https://raw.githubusercontent.com/kubernetes/ingress-nginx/$INGRESS_NGINX_VERSION/deploy/static/provider/kind/deploy.yaml" >/dev/null

log "Installing metrics-server ($METRICS_SERVER_VERSION) — needed by the plain CPU HPA"
kubectl apply -f "https://github.com/kubernetes-sigs/metrics-server/releases/download/$METRICS_SERVER_VERSION/components.yaml" >/dev/null
# kind's kubelets use self-signed certificates
if ! kubectl -n kube-system get deployment metrics-server -o jsonpath='{.spec.template.spec.containers[0].args}' \
    | grep -q kubelet-insecure-tls; then
  kubectl -n kube-system patch deployment metrics-server --type=json -p='[
    {"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]' >/dev/null
fi

# ── 2. images ──────────────────────────────────────────────────────────────────
# name:context:dockerfile — serving builds from the root (shares training's features.py)
IMAGES="forecast-serving:.:serving/Dockerfile predictive-scaler:scaler:scaler/Dockerfile
        cpu-burner:demo/cpu-burner:demo/cpu-burner/Dockerfile forecast-training:training:training/Dockerfile"
if [ "$BUILD_IMAGES" = "1" ]; then
  for entry in $IMAGES; do
    IFS=: read -r name context dockerfile <<< "$entry"
    log "Building $REGISTRY/$name:$IMAGE_TAG from $context/ ($dockerfile)"
    docker build -q -f "$dockerfile" -t "$REGISTRY/$name:$IMAGE_TAG" "$context" >/dev/null
    kind load docker-image --name "$CLUSTER_NAME" "$REGISTRY/$name:$IMAGE_TAG"
  done
else
  log "Using images $REGISTRY/*:$IMAGE_TAG from the registry"
fi

# ── 3. wait for the add-ons the stack depends on ───────────────────────────────
log "Waiting for ingress-nginx"
kubectl -n ingress-nginx rollout status deployment/ingress-nginx-controller --timeout=300s >/dev/null
kubectl -n ingress-nginx wait --for=condition=ready pod -l app.kubernetes.io/component=controller --timeout=300s >/dev/null

# ── 3b. cert-manager (TLS=1) ───────────────────────────────────────────────────
TLS_SET=""
if [ "$TLS" = "1" ]; then
  log "Installing cert-manager ($CERT_MANAGER_VERSION) and a self-signed CA issuer 'predictive-ca'"
  kubectl apply -f "https://github.com/cert-manager/cert-manager/releases/download/$CERT_MANAGER_VERSION/cert-manager.yaml" >/dev/null
  for d in cert-manager cert-manager-cainjector cert-manager-webhook; do
    kubectl -n cert-manager rollout status "deployment/$d" --timeout=300s >/dev/null
  done
  # the webhook admits objects a few seconds after its rollout; retry the apply
  for i in $(seq 1 30); do
    kubectl apply -f - <<'EOF' >/dev/null 2>&1 && break
apiVersion: cert-manager.io/v1
kind: Issuer
metadata: {name: selfsigned, namespace: cert-manager}
spec: {selfSigned: {}}
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata: {name: predictive-ca, namespace: cert-manager}
spec:
  isCA: true
  commonName: predictive-ca
  secretName: predictive-ca
  duration: 8760h
  privateKey: {algorithm: ECDSA, size: 256}
  issuerRef: {name: selfsigned, kind: Issuer}
---
apiVersion: cert-manager.io/v1
kind: ClusterIssuer
metadata: {name: predictive-ca}
spec: {ca: {secretName: predictive-ca}}
EOF
    [ "$i" = 30 ] && die "cert-manager webhook did not admit the CA issuer"
    sleep 5
  done
  kubectl -n cert-manager wait --for=condition=Ready certificate/predictive-ca --timeout=180s >/dev/null
  TLS_SET="--set global.tls.enabled=true --set global.tls.issuerRef.name=predictive-ca"
fi

# ── 4. helm ────────────────────────────────────────────────────────────────────
log "Resolving chart dependencies (KEDA)"
helm repo add kedacore https://kedacore.github.io/charts >/dev/null 2>&1 || true
helm repo update kedacore >/dev/null
helm dependency build helm/predictive-stack >/dev/null

# Phase 1: everything except the ScaledObject — KEDA's CRDs come from the
# KEDA subchart's templates, so a ScaledObject in the same install would fail
# validation before the CRDs exist. No `helm --wait`: forecast-serving stays
# unready (503 on /ready) until a model is loaded, which happens later.
log "Installing release '$RELEASE' in namespace '$NAMESPACE'"
# shellcheck disable=SC2086
helm upgrade --install "$RELEASE" helm/predictive-stack \
  -n "$NAMESPACE" --create-namespace \
  --set global.imageTag="$IMAGE_TAG" \
  --set ingress.domain="$DOMAIN" \
  --set demo.scaling=none \
  --set demo.maxReplicas="$MAX_REPLICAS" \
  --set demo.predictive.stepSeconds="$STEP_SECONDS" \
  $TLS_SET $HELM_EXTRA \
  --timeout 10m >/dev/null
if [ "$TLS" = "1" ]; then
  log "Waiting for the certificates"
  kubectl -n "$NAMESPACE" wait --for=condition=Ready certificate --all --timeout=180s >/dev/null
fi

log "Waiting for the platform components"
for d in mlflow prometheus grafana predictive-scaler cpu-burner \
         keda-operator keda-operator-metrics-apiserver keda-admission-webhooks; do
  kubectl -n "$NAMESPACE" rollout status "deployment/$d" --timeout=600s >/dev/null \
    || { kubectl -n "$NAMESPACE" get pods; kubectl -n "$NAMESPACE" describe "deployment/$d" | tail -20; die "$d did not become ready"; }
done
# forecast-serving is alive (not ready) until a model is loaded
kubectl -n "$NAMESPACE" wait --for=jsonpath='{.status.phase}'=Running pod \
  -l app.kubernetes.io/name=forecast-serving --timeout=300s >/dev/null

# ── 5. first model ─────────────────────────────────────────────────────────────
if [ "$TRAIN" = "1" ]; then
  log "Training the first model from notebooks/data-jan-2022.json"
  kubectl -n "$NAMESPACE" create configmap train-data \
    --from-file=notebooks/data-jan-2022.json --dry-run=client -o yaml | kubectl apply -f - >/dev/null
fi

# Phase 2: the ScaledObject (KEDA is up now) and, on request, the training Job.
# shellcheck disable=SC2086
helm upgrade "$RELEASE" helm/predictive-stack -n "$NAMESPACE" --reuse-values \
  --set demo.scaling="$DEMO_SCALING" \
  --set training.bootstrap.enabled="$([ "$TRAIN" = "1" ] && echo true || echo false)" \
  $TLS_SET $HELM_EXTRA \
  --timeout 10m >/dev/null

if [ "$TRAIN" = "1" ]; then
  job=$(kubectl -n "$NAMESPACE" get jobs -l app=train-bootstrap --sort-by=.metadata.creationTimestamp \
          -o jsonpath='{.items[-1:].metadata.name}')
  log "Waiting for training job $job"
  if ! kubectl -n "$NAMESPACE" wait --for=condition=complete "job/$job" --timeout=20m; then
    kubectl -n "$NAMESPACE" logs "job/$job" --all-containers=true | tail -50
    die "training job failed"
  fi
  kubectl -n "$NAMESPACE" logs "job/$job" -c train | tail -5
  log "Waiting for forecast-serving to load the model"
  kubectl -n "$NAMESPACE" rollout status deployment/forecast-serving --timeout=300s >/dev/null
fi

# ── 6. smoke tests ─────────────────────────────────────────────────────────────
if [ "$SMOKE" = "1" ]; then
  log "Smoke: forecast-serving"
  for i in $(seq 1 60); do
    icurl "forecast.$DOMAIN" -f "http://forecast.$DOMAIN/ready" -o /dev/null && break
    [ "$i" = 60 ] && die "forecast-serving did not become ready"
    sleep 5
  done
  icurl "forecast.$DOMAIN" "http://forecast.$DOMAIN/model-info"; echo
  PRED=$(icurl "forecast.$DOMAIN" -f -X POST "http://forecast.$DOMAIN/predict" \
    -H 'Content-Type: application/json' \
    -d '{"values":[0.2,0.25,0.3,0.28,0.22,0.35,0.4,0.38,0.33,0.29]}')
  echo "  /predict → $PRED"
  echo "$PRED" | grep -q '"predictions"' || die "/predict returned no predictions"

  log "Smoke: predictive-scaler gRPC (GetMetricSpec + GetMetrics through KEDA's contract)"
  # Run the check inside the cluster with the scaler image: it has grpcio and the
  # generated stubs. GetMetrics needs Prometheus samples of cpu-burner, so give
  # it a little load first and retry until the window has data.
  kubectl -n "$NAMESPACE" run smoke-load --image=busybox:1.36 --restart=Never --rm -i --quiet --command -- \
    sh -c 'for i in $(seq 1 40); do wget -q -O- "http://cpu-burner:8080/work?ms=50" >/dev/null; done; echo load ok' || true
  kubectl -n "$NAMESPACE" create configmap scaler-smoke --from-file=smoke.py=scripts/scaler-smoke.py \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  kubectl -n "$NAMESPACE" delete pod scaler-smoke --ignore-not-found >/dev/null
  kubectl -n "$NAMESPACE" apply -f - <<EOF >/dev/null
apiVersion: v1
kind: Pod
metadata:
  name: scaler-smoke
spec:
  restartPolicy: Never
  containers:
    - name: smoke
      image: $REGISTRY/predictive-scaler:$IMAGE_TAG
      imagePullPolicy: IfNotPresent
      command: ["python", "/smoke/smoke.py"]
      env:
        - {name: PYTHONPATH, value: /app}      # the scaler package lives in the image's WORKDIR
        - {name: NS, value: "$NAMESPACE"}
        - {name: STEP, value: "$STEP_SECONDS"}
$([ "$TLS" = "1" ] && echo '        - {name: TLS_DIR, value: /etc/keda-tls}')
      volumeMounts:
        - {name: smoke, mountPath: /smoke}
$([ "$TLS" = "1" ] && echo '        - {name: keda-tls, mountPath: /etc/keda-tls, readOnly: true}')
  volumes:
    - name: smoke
      configMap: {name: scaler-smoke}
$([ "$TLS" = "1" ] && echo '    - name: keda-tls
      secret: {secretName: keda-scaler-client}')
EOF
  kubectl -n "$NAMESPACE" wait --for=jsonpath='{.status.phase}'=Succeeded pod/scaler-smoke --timeout=300s >/dev/null \
    || { kubectl -n "$NAMESPACE" logs pod/scaler-smoke; die "scaler smoke test failed"; }
  kubectl -n "$NAMESPACE" logs pod/scaler-smoke
  kubectl -n "$NAMESPACE" delete pod scaler-smoke --ignore-not-found >/dev/null

  echo
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "  Grafana      http://grafana.$DOMAIN      (admin/admin)"
  echo "  Prometheus   http://prometheus.$DOMAIN"
  echo "  MLflow       http://mlflow.$DOMAIN"
  echo "  Forecast API http://forecast.$DOMAIN/docs"
  echo "  kubectl -n $NAMESPACE get hpa,scaledobject,pods -w"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi
log "Done."
