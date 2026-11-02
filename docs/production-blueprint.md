# Production blueprint: from prototype to an enterprise-grade predictive autoscaler

Scope: `m0zart89/predictive-scaling-system` at `main` (commit `5f06420`,
v1.1 + PR #8). Audience: the platform/SRE team that will run it. Every
section names the architecture choice, the reason, and the change to this
repository (chart values, manifests, Python) that implements it.

Where the prototype already covers a point, this says so instead of
re-proposing it:

| Stated limitation | State on `main` | What remains |
|---|---|---|
| SQLite MLflow backend, local artifacts | still true (`helm/predictive-stack/templates/mlflow.yaml`: `--backend-store-uri=sqlite:////mlflow/mlflow.db`, artifacts on the same PVC) | §1.1 |
| Prometheus on `emptyDir` | PVC + 30-day retention exist (`prometheus.persistence`, `values-pilot.yaml`); single replica, no long-term store | §1.2 |
| CPU-only features | true | §3.2 |
| Single replica everywhere | true; the quality tracker in `serving/quality.py` is also per-pod state | §2.1 |
| Blind nightly retraining | true; `ForecastDriftHigh` fires but only pages | §3.1 |
| No AuthN/Z or transport security | `/reload` requires `X-Reload-Token`; NetworkPolicy for forecast-serving behind `security.networkPolicy`; gRPC and `/predict` are plaintext and unauthenticated | §2.2 |

**Status:** applied in PR #9 (`feat/production`), one commit per step, in the
order below; `values-prod.yaml` is the resulting profile and CI renders and
checks it, the kind e2e runs with mutual TLS. Not exercised by CI: the
external PostgreSQL/S3 and Thanos object store (no such services in CI) and
NetworkPolicy enforcement (kind's CNI does not enforce them).

The order below is the order to do it in. §1 and §2.1 are prerequisites
for a pilot that runs for weeks; §4 is the one that protects the business
when the model is wrong and should ship with the first production
workload.

---

## 1. State and storage

### 1.1 MLflow: PostgreSQL backend, object-store artifacts, stateless servers

**Choice.** Backend store on a managed PostgreSQL (RDS/Cloud SQL/Azure
Database; in-cluster CloudNativePG if there is no managed option).
Artifact root on S3 (or MinIO on-prem) with the tracking server in
*proxied artifact* mode (`--serve-artifacts --artifacts-destination
s3://…`), which the chart already uses against local disk. Proxied mode is
the right one here: training jobs and forecast-serving keep talking only
to the MLflow service, so the S3 credential lives in exactly one place
(the MLflow ServiceAccount via IRSA / Workload Identity), and the existing
code paths (`mlflow.sklearn.log_model`, `client.download_artifacts`) do
not change.

With Postgres + S3 the server is stateless: run 2 replicas, a PDB, and
gunicorn workers. Access control: MLflow 2.11 ships the `basic-auth`
app; for SSO put the UI behind oauth2-proxy and keep in-cluster clients
on a NetworkPolicy allow-list (training, serving, Grafana). The registry
is the system of record for which model is live, so it gets PITR on
Postgres and versioning + lifecycle on the bucket.

**Chart change** (`helm/predictive-stack/values.yaml`, new keys; the
template reads them instead of the SQLite flags):

```yaml
mlflow:
  replicas: 2
  backendStore:
    # URI from a Secret so the DSN never lands in values or `helm get values`
    existingSecret: mlflow-backend          # key: MLFLOW_BACKEND_STORE_URI
    # postgresql+psycopg2://mlflow:***@mlflow-db.internal:5432/mlflow?sslmode=require
  artifacts:
    destination: s3://acme-mlops-prod/predictive-scaling/mlflow
    # MinIO / on-prem: also set endpoint and credentials via the same Secret
    endpointUrl: ""                          # MLFLOW_S3_ENDPOINT_URL for MinIO
  serviceAccount:
    annotations:
      eks.amazonaws.com/role-arn: arn:aws:iam::123456789012:role/mlflow-artifacts
  auth:
    mode: basic                              # basic | none (behind oauth2-proxy)
    existingSecret: mlflow-basic-auth        # MLFLOW_FLASK_SERVER_SECRET_KEY, admin creds
  resources:
    requests: {cpu: 250m, memory: 512Mi}
    limits:   {cpu: "1",  memory: 1Gi}
  persistence:
    enabled: false                           # nothing local any more
```

Rendered container spec (what the template should produce):

```yaml
containers:
  - name: mlflow
    image: ghcr.io/mlflow/mlflow:v2.11.1
    command: ["mlflow", "server"]
    args:
      - --host=0.0.0.0
      - --port=5000
      - --backend-store-uri=$(MLFLOW_BACKEND_STORE_URI)
      - --default-artifact-root=mlflow-artifacts:/
      - --serve-artifacts
      - --artifacts-destination=s3://acme-mlops-prod/predictive-scaling/mlflow
      - --workers=4
      - --app-name=basic-auth
    envFrom:
      - secretRef: {name: mlflow-backend}      # MLFLOW_BACKEND_STORE_URI
      - secretRef: {name: mlflow-basic-auth}   # MLFLOW_FLASK_SERVER_SECRET_KEY, MLFLOW_AUTH_CONFIG_PATH
    env:
      - {name: AWS_DEFAULT_REGION, value: eu-central-1}
      # MinIO only: - {name: MLFLOW_S3_ENDPOINT_URL, value: https://minio.storage.svc:9000}
    readinessProbe: {httpGet: {path: /health, port: 5000}, periodSeconds: 10}
    livenessProbe:  {httpGet: {path: /health, port: 5000}, periodSeconds: 30, failureThreshold: 3}
```

Client side, with `basic-auth` on: training and serving read
`MLFLOW_TRACKING_USERNAME` / `MLFLOW_TRACKING_PASSWORD` from a Secret
(`mlflow-client` with a non-admin user that has `EDIT` on the experiment
and `MANAGE` on the registered model for training, `READ` for serving).
No code change: the MLflow client picks the variables up.

Run `mlflow db upgrade` as a Helm pre-upgrade hook Job, pinned to the same
image, so a chart upgrade that bumps MLflow cannot start servers against
an old schema.

**Serving must survive MLflow being down.** Today a forecast-serving pod
that restarts while MLflow is unreachable stays unready (it retries every
`MODEL_RETRY_SECONDS`). Add a last-known-good cache: after a successful
load, write `model/`, `scaler.pkl`, `forecast_meta.json` and the version
tags to `MODEL_CACHE_DIR` (an `emptyDir` is enough per pod; a small RWX
volume if you want it to survive rescheduling); at startup, if MLflow
fails, load from the cache and report `model_stage="cache"` in
`/model-info` and `forecast_model_info`. Alert on `model_stage="cache"`
for more than an hour. This is ~40 lines in `serving/app.py` around
`load_model()` and does not change the contract.

### 1.2 Prometheus: HA pair, long-term store, and recording rules for the training series

**Choice.** Two Prometheus replicas (Prometheus Operator) with a Thanos
sidecar shipping 2-hour blocks to the same object store, Thanos Query for
deduplicated reads and Thanos Store Gateway for anything older than local
retention. If the organisation already standardises on VictoriaMetrics,
`vmagent → vmsingle/vmcluster` is the cheaper equivalent and needs no code
change either: both expose `/api/v1/query_range`, which is all
`training/forecast_training/data.py` and the scaler use.

Two consumers, two endpoints:

| Consumer | Endpoint | Why |
|---|---|---|
| `predictive-scaler` (every 15–30 s, last day of one series) | the Prometheus pair directly (`prometheus-operated:9090`, both replicas behind the Service) | lowest latency; a day is always inside local retention |
| `forecast-training` (nightly or on alert, 14–60 days) | Thanos Query (`thanos-query:9090`) | deduplicated across replicas, reaches the object store for history beyond local retention, survives a replica rebuild |

**Record the demand series.** The training job currently runs the raw
workload query (`(sum(rate(container_cpu_usage_seconds_total{…}[1m]))) /
capacity`) as a 30-day `query_range`. Against a real cluster that is
expensive and fragile (cardinality churn, pod restarts, rate windows at
block boundaries). Put each workload's demand on a recording rule at the
model's step and train on the recorded series; the scaler can read the
same rule, which makes "scaler and model see the same numbers" true by
construction instead of by a Helm template:

```yaml
# PrometheusRule, one group per workload from .Values.workloads
groups:
  - name: predictive-stack.demand
    interval: 1m
    rules:
      - record: workload:demand:cpu_cores
        labels: {workload: api}
        expr: sum(rate(container_cpu_usage_seconds_total{namespace="shop", container="api"}[2m]))
      - record: workload:demand:model_units
        labels: {workload: api}
        expr: workload:demand:cpu_cores{workload="api"} / 4     # capacity
```

Then `training.cronjob.query` becomes
`workload:demand:model_units{workload="api"}` and the ScaledObject's
`query` the same expression. A rule evaluated every minute against a
5-minute step is cheap for 30 days (8 640 points) and identical whether
read from local TSDB or the store gateway.

**Prometheus CR** (Prometheus Operator; replaces the chart's hand-rolled
Deployment when the cluster has the operator, which a production cluster
should):

```yaml
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: predictive
  namespace: predictive
spec:
  replicas: 2
  replicaExternalLabelName: prometheus_replica     # Thanos dedups on this
  externalLabels: {cluster: prod-eu-1, stack: predictive}
  retention: 15d                                   # local; the store has the rest
  retentionSize: 80GB
  storage:
    volumeClaimTemplate:
      spec:
        storageClassName: gp3
        resources: {requests: {storage: 100Gi}}
  thanos:
    image: quay.io/thanos/thanos:v0.37.2
    objectStorageConfig:
      name: thanos-objstore                        # Secret: type: S3, config: {bucket, endpoint, region}
      key: objstore.yml
  serviceMonitorSelector: {matchLabels: {app.kubernetes.io/part-of: predictive-stack}}
  ruleSelector:           {matchLabels: {app.kubernetes.io/part-of: predictive-stack}}
  podMonitorSelector:     {matchLabels: {app.kubernetes.io/part-of: predictive-stack}}
  resources:
    requests: {cpu: 500m, memory: 2Gi}
    limits:   {memory: 4Gi}
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        - labelSelector: {matchLabels: {prometheus: predictive}}
          topologyKey: kubernetes.io/hostname
  securityContext: {runAsNonRoot: true, runAsUser: 65534, fsGroup: 65534}
```

Thanos Query/Store/Compactor come from the `bitnami/thanos` or
`thanos-operator` chart with the same `thanos-objstore` Secret; the
Compactor's retention (`--retention.resolution-raw=90d`) is what
guarantees the 30+ day window independent of pod lifetimes. Set
`training.cronjob.days: "28"` once the store holds four weeks: the
seasonal candidates then see four of every weekday.

Alert the data path itself: `absent_over_time(workload:demand:model_units[10m])`
per workload, `thanos_shipper_upload_failures_total > 0`, and
`prometheus_tsdb_head_series > 1.5e6` (cardinality guard). A model trained
on a series with holes is filled by `regularize()` with time-of-day means,
which is the right behaviour for one missing scrape and the wrong one for
a missing day.

---

## 2. High availability, scaling and security

### 2.1 forecast-serving and predictive-scaler as HA deployments

Three facts about this code decide the manifests:

1. **The forecast quality tracker is in-process state.** `serving/quality.py`
   matches a forecast to the actual value a *later* request carries. With
   2+ replicas behind one Service, consecutive requests for a series land
   on different pods and the rolling MAE, hence `ForecastDriftHigh`,
   silently degrades. Fix on the metrics side, not with sticky sessions:
   have the scaler export its per-step forecast as a gauge and compute the
   error in Prometheus with a positive `offset`. Stateless pods, exact
   matching, history for free.

   ```python
   # scaler/predictive_scaler/metrics.py (addition)
   self.forecast_demand = Gauge(
       "predictive_scaler_forecast_demand",
       "Forecast demand for each step of the horizon, in the query's units",
       ["namespace", "scaledobject", "step"])
   # engine._compute, after a successful forecast:
   for k, v in enumerate(horizon, start=1):
       self.metrics.forecast_demand.labels(**labels, step=str(k)).set(max(0.0, v * cfg.capacity))
   ```

   ```yaml
   # recording rules: error of the forecast made k steps ago vs demand now (step = 5m)
   - record: scaler:forecast_abs_error:step1
     expr: abs(predictive_scaler_forecast_demand{step="1"} offset 5m  - predictive_scaler_observed_demand)
   - record: scaler:forecast_abs_error:step6
     expr: abs(predictive_scaler_forecast_demand{step="6"} offset 30m - predictive_scaler_observed_demand)
   - record: scaler:forecast_mae_1h:step6
     expr: avg_over_time(scaler:forecast_abs_error:step6[1h])
   ```

   `ForecastDriftHigh` then compares `scaler:forecast_mae_1h:step6 /
   capacity` with `forecast_model_holdout_mae{horizon_step="6"}` (the
   model's own tag, already exported). Keep `serving/quality.py` for
   single-replica dev installs; in production set `FORECAST_MAX_SERIES=0`
   to switch it off.

2. **`/reload` reaches one pod.** The training job POSTs to the Service.
   With three replicas two keep serving the old version. Replace push with
   pull: a background thread in serving checks the registry alias every
   `MODEL_REFRESH_SECONDS` (60) and reloads when the version changes.
   (`/reload` was removed afterwards; the lifecycle worker is the one path.) Code, around the existing retry loop in
   `serving/app.py`:

   ```python
   def _refresh_forever():
       while True:
           time.sleep(MODEL_REFRESH_SECONDS)
           try:
               v, _, _ = _resolve_version(MlflowClient(tracking_uri=MLFLOW_TRACKING_URI))
               if _bundle is None or str(v.version) != _bundle.info["model_version"]:
                   load_model()
           except Exception as exc:            # registry down: keep serving what we have
               log.warning("alias check failed: %s", exc)
   threading.Thread(target=_refresh_forever, daemon=True).start()
   ```

   One uvicorn worker per pod (the default in `serving/Dockerfile`), scale
   with replicas: a second worker in the same pod would hold a second
   `_bundle` that `/reload` never touches.

3. **The scaler is already stateless enough.** Its decision cache is
   per-pod and bounded by `cacheSeconds`; KEDA calling two replicas in turn
   costs at most one extra Prometheus query per poll. The gRPC health
   service is implemented (`grpc_health.v1`), so native `grpc` probes
   apply.

**forecast-serving** (`helm/forecast-serving/values.yaml` overrides for
production; the deployment template needs the affinity, topology and PDB
blocks added):

```yaml
replicaCount: 3
podDisruptionBudget:
  enabled: true
  minAvailable: 2
resources:
  requests: {cpu: 250m, memory: 512Mi}
  limits:   {cpu: "2",  memory: 1Gi}        # no CPU throttling of inference on a spike
startupProbe:                               # model download + unpickle, MLflow may be slow
  httpGet: {path: /live, port: http}
  periodSeconds: 5
  failureThreshold: 60                      # 5 min
readinessProbe:
  httpGet: {path: /ready, port: http}       # 503 until a model is loaded
  periodSeconds: 10
  failureThreshold: 3
livenessProbe:
  httpGet: {path: /live, port: http}
  periodSeconds: 30
  failureThreshold: 3
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels: {app.kubernetes.io/name: forecast-serving}
        topologyKey: kubernetes.io/hostname
topologySpreadConstraints:
  - maxSkew: 1
    topologyKey: topology.kubernetes.io/zone
    whenUnsatisfiable: ScheduleAnyway
    labelSelector:
      matchLabels: {app.kubernetes.io/name: forecast-serving}
priorityClassName: platform-critical        # above the workloads it scales
env:
  MODEL_REFRESH_SECONDS: "60"
  MODEL_CACHE_DIR: /cache
  FORECAST_MAX_SERIES: "0"                  # quality via recording rules (above)
securityContext:
  readOnlyRootFilesystem: true              # /cache and /tmp as emptyDir
```

**predictive-scaler** (`helm/predictive-scaler/values.yaml`):

```yaml
replicaCount: 2
podDisruptionBudget:
  enabled: true
  maxUnavailable: 1
resources:
  requests: {cpu: 100m, memory: 128Mi}
  limits:   {cpu: 500m, memory: 256Mi}
readinessProbe:
  grpc: {port: 6000}                        # grpc_health.v1, SERVING once the servicer is up
  periodSeconds: 10
livenessProbe:
  grpc: {port: 6000}
  periodSeconds: 30
  failureThreshold: 3
affinity:
  podAntiAffinity:
    requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels: {app.kubernetes.io/name: predictive-scaler}
        topologyKey: kubernetes.io/hostname
priorityClassName: platform-critical
terminationGracePeriodSeconds: 20           # finish in-flight GetMetrics before SIGKILL
```

Pair this with a PDB-aware rollout: `maxUnavailable: 0, maxSurge: 1`
on both Deployments, so a chart upgrade never drops below the PDB floor.

**KEDA itself.** The umbrella chart's KEDA dependency defaults to one
operator and one metrics API server. Set `operator.replicaCount: 2`
(leader election is built in) and `metricsServer.replicaCount: 2`; the
metrics API server is on the HPA's hot path and a single replica means
every HPA in the cluster stalls during its restarts. Keep KEDA's
`fallback` on each ScaledObject: it is what holds replicas when the scaler
returns `UNAVAILABLE` (Prometheus down), and it is the one failure mode a
scaler replica count cannot cover.

### 2.2 Securing the two loops

Two mechanisms, layered: **NetworkPolicy** (who may open a connection)
and **mTLS** (who is on the other end, and nobody in between). Decide the
mTLS carrier first:

- **A service mesh is already in the cluster** (Linkerd, Istio ambient):
  inject the `predictive` and `keda` namespaces, write `Server` +
  `AuthorizationPolicy` objects that allow `keda-operator → predictive-scaler:6000`
  and `predictive-scaler → forecast-serving:8080`, done. No code change;
  KEDA's gRPC client and Python's `urllib` go through the sidecar/proxy.
- **No mesh**: use cert-manager-issued certificates and the native TLS
  support in both hops. That is the version shown below, because it has
  no dependency the chart cannot ship.

**Certificates** (one internal CA, cert-manager `ClusterIssuer: internal-ca`):

```yaml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata: {name: predictive-scaler-tls, namespace: predictive}
spec:
  secretName: predictive-scaler-tls
  issuerRef: {kind: ClusterIssuer, name: internal-ca}
  duration: 720h
  renewBefore: 240h
  usages: [server auth]
  dnsNames:
    - predictive-scaler.predictive.svc
    - predictive-scaler.predictive.svc.cluster.local
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata: {name: keda-scaler-client, namespace: keda}
spec:
  secretName: keda-scaler-client            # tls.crt, tls.key, ca.crt
  issuerRef: {kind: ClusterIssuer, name: internal-ca}
  usages: [client auth]
  commonName: keda-operator
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata: {name: forecast-serving-tls, namespace: predictive}
spec:
  secretName: forecast-serving-tls
  issuerRef: {kind: ClusterIssuer, name: internal-ca}
  usages: [server auth]
  dnsNames: [forecast-serving.predictive.svc, forecast-serving.predictive.svc.cluster.local]
---
apiVersion: cert-manager.io/v1
kind: Certificate
metadata: {name: predictive-scaler-client, namespace: predictive}
spec:
  secretName: predictive-scaler-client
  issuerRef: {kind: ClusterIssuer, name: internal-ca}
  usages: [client auth]
  commonName: predictive-scaler
```

**gRPC loop, KEDA → scaler.** Server side, `scaler/predictive_scaler/server.py`
(replace `add_insecure_port` when the three files are present):

```python
def _credentials():
    tls_dir = os.getenv("TLS_DIR")                      # /etc/scaler-tls, mounted Secret
    if not tls_dir:
        return None
    read = lambda n: open(os.path.join(tls_dir, n), "rb").read()
    return grpc.ssl_server_credentials(
        [(read("tls.key"), read("tls.crt"))],
        root_certificates=read("ca.crt"),
        require_client_auth=True,                         # mTLS: KEDA must present a cert
    )

creds = _credentials()
port = server.add_secure_port(address, creds) if creds else server.add_insecure_port(address)
```

Certificate rotation: cert-manager rewrites the Secret; a `grpc` server
does not reload files, so mount the Secret and let `reloader` (stakater)
or the chart's `checksum/tls` pod annotation roll the pods on change.
With `renewBefore: 240h` that is one rolling restart every 20 days,
covered by the PDB.

Client side is KEDA configuration only (KEDA ≥ 2.16: `enableTLS` on the
external trigger, certificates from a `TriggerAuthentication`). In the
chart's `_scaledobject.tpl`:

```yaml
  triggers:
    - type: external
      metricType: AverageValue
      metadata:
        scalerAddress: predictive-scaler.predictive.svc.cluster.local:6000
        enableTLS: "true"
        # … existing keys …
      authenticationRef:
        name: predictive-scaler-mtls
---
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata: {name: predictive-scaler-mtls, namespace: predictive}
spec:
  secretTargetRef:
    - parameter: caCert
      name: keda-scaler-client-copy          # the keda-namespace Secret, replicated into
      key: ca.crt                            # the workload namespace (ClusterTriggerAuthentication
    - parameter: tlsClientCert               # avoids the copy; use it when all workloads
      name: keda-scaler-client-copy          # share one CA)
      key: tls.crt
    - parameter: tlsClientKey
      name: keda-scaler-client-copy
      key: tls.key
```

**HTTP loop, scaler → forecast-serving.** uvicorn terminates TLS and
requires a client certificate:

```yaml
# forecast-serving container args
args: ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8443",
       "--ssl-certfile", "/etc/tls/tls.crt", "--ssl-keyfile", "/etc/tls/tls.key",
       "--ssl-ca-certs", "/etc/tls/ca.crt", "--ssl-cert-reqs", "2"]   # 2 = CERT_REQUIRED
```

`scaler/predictive_scaler/clients.py` is standard-library `urllib`; give
it an SSL context once and pass it to every `urlopen`:

```python
def _ssl_context():
    tls_dir = os.getenv("TLS_DIR")
    if not tls_dir:
        return None
    ctx = ssl.create_default_context(cafile=os.path.join(tls_dir, "ca.crt"))
    ctx.load_cert_chain(os.path.join(tls_dir, "tls.crt"), os.path.join(tls_dir, "tls.key"))
    ctx.minimum_version = ssl.TLSVersion.TLSv1_3
    return ctx

_CTX = _ssl_context()

def _read_json(req, timeout):
    with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as resp:
        return json.loads(resp.read().decode("utf-8"))
```

`forecastServiceURL` becomes `https://forecast-serving.predictive.svc.cluster.local:8443`.
Prometheus scrapes `/metrics` on the same port with `scheme: https` and
the CA from the Secret (`tlsConfig` on the ServiceMonitor). The training
job's `/reload` call keeps the token and gains the same client cert; with
pull-based refresh (§2.1) it can be removed from the CronJob entirely,
which also removes the `forecast-reload-token` Secret from the training
pod.

**NetworkPolicy, default deny.** Extend the chart's existing
`forecast-serving` policy to the whole namespace:

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: default-deny, namespace: predictive}
spec:
  podSelector: {}
  policyTypes: [Ingress, Egress]
  egress:
    - to: [{namespaceSelector: {matchLabels: {kubernetes.io/metadata.name: kube-system}}}]
      ports: [{port: 53, protocol: UDP}, {port: 53, protocol: TCP}]
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata: {name: predictive-scaler, namespace: predictive}
spec:
  podSelector: {matchLabels: {app.kubernetes.io/name: predictive-scaler}}
  policyTypes: [Ingress, Egress]
  ingress:
    - from:
        - namespaceSelector: {matchLabels: {kubernetes.io/metadata.name: keda}}
          podSelector: {matchLabels: {app: keda-operator}}
      ports: [{port: 6000}]
    - from: [{podSelector: {matchLabels: {app: prometheus}}}]
      ports: [{port: 9100}]
  egress:
    - to: [{podSelector: {matchLabels: {app: prometheus}}}]
      ports: [{port: 9090}]
    - to: [{podSelector: {matchLabels: {app.kubernetes.io/name: forecast-serving}}}]
      ports: [{port: 8443}]
    - to: [{namespaceSelector: {matchLabels: {kubernetes.io/metadata.name: kube-system}}}]
      ports: [{port: 53, protocol: UDP}]
```

(forecast-serving: ingress from the scaler, training jobs and Prometheus;
egress to MLflow only. Training jobs: egress to Prometheus/Thanos Query,
MLflow and serving.) These require a CNI that enforces policy; the chart
already says so for the existing one.

**The rest of the perimeter**, briefly because it is standard: image
digests pinned by the chart and verified by Kyverno/Sigstore
policy-controller (sign in CI with `cosign sign --yes` under GitHub OIDC;
the `build-image` job already has `id-token` reach), Secrets from External
Secrets Operator rather than `helm install --set`, `readOnlyRootFilesystem`
on every container (the serving image needs `/tmp` for MLflow downloads,
an `emptyDir`), and the Grafana/MLflow ingresses behind the organisation's
OIDC proxy. The `/reload` token and the training image's write access to
the registry are the two credentials that can change what runs in
production; rotate the first with the chart's existing
lookup-or-generate Secret and give the second to the training
ServiceAccount only.

---

## 3. Advanced MLOps: event-driven retraining and multivariate features

### 3.1 From a blind CronJob to alert-triggered Jobs

Keep the nightly CronJob as the floor (data accumulates, the comparison
rules guard against regressions) and add an event path. The trigger must
do three things the alert cannot: deduplicate (one retrain per cause, not
one per Alertmanager repeat), rate-limit (a model that drifts twice in an
hour has a data problem, not a training problem), and **exclude the
anomaly** that caused the alert from the training window — training on a
flash sale produces a model that forecasts flash sales.

**Alertmanager route** (the chart's rules already carry `ForecastDriftHigh`
and `ScalerFallbackRate`):

```yaml
route:
  routes:
    - receiver: retrain-trigger
      matchers: ['alertname =~ "ForecastDriftHigh|ForecastBreakerOpenLong"']
      group_by: [scaledobject]
      group_wait: 0s
      group_interval: 5m
      repeat_interval: 6h
      continue: true                       # still page the on-call
receivers:
  - name: retrain-trigger
    webhook_configs:
      - url: http://retrain-trigger.predictive.svc:8080/alertmanager
        send_resolved: true
        http_config:
          authorization: {type: Bearer, credentials_file: /etc/alertmanager/secrets/retrain-token}
```

**The receiver** is a 100-line service with one job: turn an alert into a
`batch/v1 Job` cloned from the CronJob's template, with the anomaly
excluded. Its RBAC is `get cronjobs`, `create/list jobs` in its own
namespace, nothing else. The pieces that matter:

```python
# training/retrain_trigger.py — FastAPI receiver for Alertmanager v4 webhooks
import os, time, hmac
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Header, HTTPException, Request
from kubernetes import client, config

NS = os.environ["NAMESPACE"]; CRONJOB = os.getenv("CRONJOB", "forecast-retrain")
TOKEN = os.environ["TRIGGER_TOKEN"]; MIN_GAP = timedelta(hours=float(os.getenv("MIN_GAP_HOURS", "6")))
PROM = os.getenv("PROMETHEUS_URL", "http://thanos-query:9090")
config.load_incluster_config(); batch = client.BatchV1Api()
app = FastAPI()

def active_or_recent(name_prefix: str) -> bool:
    jobs = batch.list_namespaced_job(NS, label_selector="app=forecast-retrain,trigger=alert").items
    now = datetime.now(timezone.utc)
    for j in jobs:
        if j.status.active or (j.status.completion_time and now - j.status.completion_time < MIN_GAP):
            return True
    return False

def breaker_windows(scaledobject: str, days: int) -> list[str]:
    """Ranges where the scaler's circuit breaker was open (§4): excluded from training."""
    import urllib.parse, urllib.request, json
    q = f'predictive_scaler_breaker_open{{scaledobject="{scaledobject}"}} == 1'
    end = time.time(); start = end - days * 86400
    url = f"{PROM}/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": q, "start": start, "end": end, "step": 60})
    body = json.load(urllib.request.urlopen(url, timeout=30))
    ranges, cur = [], None
    for ts, _ in (body["data"]["result"] or [{"values": []}])[0]["values"]:
        ts = float(ts)
        if cur and ts - cur[1] <= 90: cur[1] = ts
        else:
            if cur: ranges.append(cur)
            cur = [ts, ts]
    if cur: ranges.append(cur)
    # pad by one model horizon on each side; the training CLI takes START/END
    return [f"{int(a) - 1800}/{int(b) + 1800}" for a, b in ranges]

@app.post("/alertmanager")
async def alertmanager(req: Request, authorization: str = Header("")):
    if not hmac.compare_digest(authorization, f"Bearer {TOKEN}"):
        raise HTTPException(401)
    payload = await req.json()
    firing = [a for a in payload.get("alerts", []) if a["status"] == "firing"]
    started = []
    for alert in firing:
        so = alert["labels"].get("scaledobject", "")
        if active_or_recent(so):
            continue                                        # dedup + rate limit
        cj = batch.read_namespaced_cron_job(CRONJOB, NS)
        tpl = cj.spec.job_template
        container = tpl.spec.template.spec.containers[0]
        env = {e.name: e for e in (container.env or [])}
        days = int(env["TRAIN_DAYS"].value) if "TRAIN_DAYS" in env else 14
        container.args = (container.args or []) + [f"--exclude={r}" for r in breaker_windows(so, days)]
        container.env = list(env.values()) + [
            client.V1EnvVar(name="TRAIN_REASON", value=alert["labels"]["alertname"]),
            client.V1EnvVar(name="TRAIN_MIN_IMPROVEMENT", value="0.0"),   # drift: any better model wins
        ]
        job = client.V1Job(
            metadata=client.V1ObjectMeta(
                generate_name=f"retrain-{alert['labels']['alertname'].lower()}-",
                labels={"app": "forecast-retrain", "trigger": "alert", "scaledobject": so},
                annotations={"alert.fingerprint": alert["fingerprint"]}),
            spec=tpl.spec)
        batch.create_namespaced_job(NS, job)
        started.append(job.metadata.generate_name)
    return {"started": started}
```

Why these choices and not Argo Events / a Tekton trigger: the whole
behaviour fits in one file the team can read, the CronJob template stays
the single definition of "how we train" (the receiver clones it, it does
not re-declare it), and the exclusion logic is where the domain knowledge
is. If the platform already runs Argo Events, the same cloning logic
belongs in a Sensor's `k8s` trigger with a `resource` built from the
CronJob; the webhook EventSource replaces the FastAPI part.

**What the retrain does differently when triggered by drift.** The
`run_pipeline` rules do not change: level tracking, baseline, fair
comparison on fresh windows. Two settings do: `TRAIN_MIN_IMPROVEMENT=0`
(the champion is already known to be wrong), and `--exclude` for the
breaker windows. Log `TRAIN_REASON` as an MLflow run tag so the registry
history says why each version exists.

Add one alert for the new signal, because a breaker that stays open is a
model problem, not a traffic problem:

```yaml
- alert: ForecastBreakerOpenLong
  expr: max_over_time(predictive_scaler_breaker_open[2h]) == 1
    and predictive_scaler_breaker_open == 1
  for: 2h
  labels: {severity: warning}
  annotations:
    summary: "{{ $labels.scaledobject }}: reactive fallback for 2h, the model no longer fits"
    runbook_url: https://github.com/m0zart89/predictive-scaling-system/blob/main/docs/runbooks/forecast-breaker.md
```

### 3.2 Multivariate features: CPU, memory, request rate

The target stays one series: the demand in the ScaledObject's units,
which is what KEDA divides by `targetValue`. Memory and request rate are
**exogenous channels**: inputs that lead or explain the target. Request
rate is the one worth having first. It leads CPU by the request
processing time plus queueing, so a model that sees RPS climbing
forecasts the CPU ramp before CPU shows it; memory mostly matters as a
level (a leak, a cache warming) and as a second gate for workloads that
are memory-bound.

**Architecture.** Everything that is per-channel today becomes per-channel
in a list; the contract (`forecast_meta.json`) gains `channels` and the
request body gains a 2-D `values`. The feature code in
`training/forecast_training/features.py` is the shared implementation
serving imports, so one change covers both sides.

```python
# training/forecast_training/features.py (format_version 4)
@dataclass(frozen=True)
class ChannelSpec:
    name: str                    # "cpu" | "mem" | "rps" — the key in /predict and in the scaler config
    query: str                   # PromQL (recording rule), one series
    scale: str = "minmax"        # minmax | log1p-minmax (rps, queue depth)
    lags: Tuple[int, ...] = ()   # extra lags in steps beyond the window, e.g. (288,) for a day

@dataclass(frozen=True)
class ForecastMeta:
    time_steps: int = 10
    horizon: int = 1
    step_seconds: int = 300
    features: str = "window"
    target: str = "level"
    period_seconds: int = 86400
    season_steps: int = 0
    channels: Tuple[str, ...] = ("cpu",)    # channels[0] is the target
    format_version: int = FORMAT_VERSION

    @property
    def input_shape(self) -> Tuple[int, int]:
        return (len(self.channels), self.time_steps + self.season_steps)


def build_features_mv(X: np.ndarray, meta: ForecastMeta, last_ts=None) -> np.ndarray:
    """
    X: (n, channels, input_length) scaled. Target channel gets the full
    anchored/seasonal treatment of build_features; each exogenous channel
    contributes its anchored window (shape relative to its own last value)
    and its last value. Output (n, F).
    """
    n, c, length = X.shape
    parts = [build_features(X[:, 0, :], meta.features, last_ts, meta.period_seconds,
                            meta.time_steps, meta.horizon)]
    for j in range(1, c):
        w = X[:, j, -meta.time_steps:]
        last = w[:, -1:]
        parts.append(np.hstack([w[:, :-1] - last, last]))
    return np.hstack(parts)
```

```python
# training/forecast_training/data.py — one grid, many channels
@dataclass
class MultiSeries:
    timestamps: np.ndarray            # shared grid
    values: np.ndarray                # (channels, n)
    channels: Tuple[str, ...]
    filled: Dict[str, int]

def query_channels(prometheus: str, specs: Sequence[ChannelSpec], days: float,
                   step_seconds: int, exclude=()) -> MultiSeries:
    series = [regularize(query_prometheus(prometheus, s.query, days, step_seconds), step_seconds, exclude)
              for s in specs]
    t0 = max(s.timestamps[0] for s in series); t1 = min(s.timestamps[-1] for s in series)
    aligned = []
    for s in series:                                           # intersect the grids
        keep = (s.timestamps >= t0) & (s.timestamps <= t1)
        aligned.append(s.values[keep])
    ts = series[0].timestamps[(series[0].timestamps >= t0) & (series[0].timestamps <= t1)]
    return MultiSeries(ts, np.stack(aligned), tuple(s.name for s in specs),
                       {s.name: x.filled for s, x in zip(specs, series)})
```

Scaling: a `MinMaxScaler` per channel (store as `scaler/{channel}.pkl`,
or one `ColumnTransformer`); `log1p` before min-max for rates and queue
depths so a 10× burst does not flatten the rest of the range. The
delta target and `from_target` apply to channel 0 only. `level_tracking`
lifts channel 0 only, which is the right question ("does the forecast
follow the *demand* level"); lifting RPS with it is a second check worth
adding (`exogenous_tracking`), because a model that ignores RPS is not
wrong, just not multivariate.

**Serving and scaler contract.**

```json
POST /predict
{
  "values": {"cpu": [ … 298 … ], "rps": [ … 298 … ], "mem": [ … 298 … ]},
  "series": "shop/api", "timestamp": 1761900000, "step_seconds": 300
}
```

`prepare_input` runs per channel (same rules: tail of `input_length`,
last `time_steps` present, gaps interpolated); a missing exogenous
channel is a 422, because the model was trained with it. The scaler's
trigger metadata gains `queries` as a JSON map and `channel` order comes
from `/model-info`:

```yaml
metadata:
  query: workload:demand:model_units{workload="api"}        # channel 0, the target
  channels: '{"rps": "workload:rps{workload=\"api\"}", "mem": "workload:mem_ratio{workload=\"api\"}"}'
```

`fetch_history` is called once per channel on the same `end`, so the
grids line up by construction. Backwards compatibility: a model with
`channels == ("cpu",)` accepts the current 1-D `values`; the scaler sends
the map only when `channels` is set.

**Candidate selection does not change.** The same `train_and_select`
ranks `hgb_delta`, `hgb_q75_delta_season` and friends on the multivariate
features; add the univariate versions of each as candidates on the same
run so the registry shows, per workload, whether RPS actually bought
anything. On the Wikipedia series it would not (one channel exists); on
an API behind a load balancer it usually does on the ramps.

---

## 4. Anomaly and edge-case protection: a circuit breaker in the scaler

**What the breaker is for.** With `mode=max` the scaler already cannot
scale *below* current demand; a spike is handled reactively by
`max(forecast, observed)` whatever the model says. The damage a flash
sale or a DDoS does is subtler and comes afterwards:

1. The model's input now contains the spike. A delta model extrapolates
   the climb; a quantile model forecasts the 75th percentile of a
   distribution that just moved; a seasonal model will expect the spike
   again tomorrow at the same time. For the next horizon (30 min) to a
   day, the forecast is confidently wrong and the scaler over-provisions.
2. The nightly job trains on the spike (§3.1 handles this with
   `--exclude`, but only if the scaler records when the spike was).
3. Under DDoS, scaling to `maxReplicas` on a forecast is spending money
   on the attacker thirty minutes earlier than the HPA would.

So the breaker's job is: **detect that the present is not like the
history the model learned, stop using the forecast until it is, record
the interval, and hand over to reactive scaling cleanly.** Reactive here
means two things that must both be true: the external trigger reports
observed demand only, *and* the ScaledObject carries a second, plain
trigger so that KEDA keeps scaling on CPU even if the external scaler is
in trouble (KEDA takes the max across triggers):

```yaml
triggers:
  - type: external                 # predictive, as today
    metadata: { … }
  - type: cpu                      # reactive floor: always on, never needs the scaler
    metricType: Utilization
    metadata:
      value: "70"
```

### 4.1 Signals and thresholds

All computable inside `engine._compute` from data it already has: the
window, a day of history (`historySteps`), the forecasts it made earlier
(keep the last `horizon` of them per series), and the serving's own
quality figures (`/model-info.rolling_mae`, or the recording rules of §2.1).

| Signal | Definition | Trip condition | Why |
|---|---|---|---|
| **Residual shock** | `r = (observed − forecast made k steps ago for now) / σ_r`, σ_r = 1.4826 × MAD of the last 48 residuals (robust; a spike must not inflate its own yardstick) | `|r| > 4` on 2 consecutive polls, or `|r| > 8` once | The model is wrong *now*, by more than noise. |
| **Seasonal level shift** | `s = observed / value one period ago` (free: the history the scaler fetches for the seasonal models) | `s > 2.5` or `s < 0.4` on 3 consecutive polls, with observed above `activationValue` | Demand is far from what any seasonal pattern predicts; the model's lags are misleading. |
| **Slope** | `Δ = observed − previous observed`, compared with `Δ_max` = max step change in the last day | `Δ > 2 × Δ_max` once | A single-step jump the history never contained: flash sale, DDoS, failover of traffic from another region. |
| **Forecast health** | failures / calls over the last 5 min from the scaler's own counters | `> 20%` or p99 latency > `timeoutSeconds` | Serving is degraded; stop paying the latency on every poll. |
| **Model quality** (slow) | `rolling_mae(step=horizon) / holdout_mae(step=horizon)` from serving | `> 3` for 30 min | The model has drifted; this is `ForecastDriftHigh` as seen by the scaler. |

The first three catch the spike within one or two polls (15–30 s). The
last two are slower and mostly make the alert-driven retraining
unnecessary to wait for.

### 4.2 State machine

```
CLOSED ──trip──▶ OPEN ──(min_open elapsed)──▶ HALF_OPEN ──(calm for N polls)──▶ CLOSED
   ▲                                              │
   └──────────────────────── trip ────────────────┘
```

- **OPEN**: report `observed` only (`source="observed"`,
  `reason="breaker:<signal>"`); keep calling serving (cheap, and it keeps
  the residual series alive), but ignore the answer. Minimum open time
  `breakerMinOpenSeconds` (600): the model's input window is 50 min of
  5-minute samples, so anything shorter re-admits a model still looking at
  the spike.
- **HALF_OPEN**: compute the decision both ways, report observed, and
  close only when `|r| < 2` and `0.6 < s < 1.6` for `breakerCalmPolls` (6)
  consecutive polls. Any trip returns to OPEN and doubles `min_open`
  (capped at 1 h): a workload that oscillates gets the reactive path for
  longer, not a scaler that flaps.
- **CLOSED**: normal.

State is per `(namespace, scaledobject)`, in memory, per scaler replica.
Two replicas can briefly disagree; the cost is one poll of forecast-based
scaling on one of them, bounded by `cacheSeconds`. If that is not
acceptable, hash the ScaledObject name onto a replica (KEDA talks to the
Service; a headless Service plus a `grpc` `pick_first` on a per-workload
address is the simplest deterministic routing).

### 4.3 Implementation

```python
# scaler/predictive_scaler/breaker.py
from __future__ import annotations
import math, time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

CLOSED, OPEN, HALF_OPEN = "closed", "open", "half_open"


@dataclass
class BreakerConfig:
    residual_window: int = 48          # polls kept for the MAD
    residual_trip: float = 4.0         # |z| on two consecutive polls
    residual_trip_hard: float = 8.0    # |z| once
    level_shift_hi: float = 2.5        # observed / seasonal reference
    level_shift_lo: float = 0.4
    level_shift_polls: int = 3
    slope_factor: float = 2.0          # vs max one-step change in the history
    min_open_seconds: float = 600.0
    max_open_seconds: float = 3600.0
    calm_polls: int = 6
    calm_residual: float = 2.0


@dataclass
class _State:
    state: str = CLOSED
    opened_at: float = 0.0
    open_for: float = 600.0
    residuals: Deque[float] = field(default_factory=lambda: deque(maxlen=48))
    pending: Deque[Tuple[float, List[float], int]] = field(default_factory=lambda: deque(maxlen=32))
    hot_residual: int = 0
    hot_level: int = 0
    calm: int = 0
    last_observed: Optional[float] = None
    trips: Dict[str, int] = field(default_factory=dict)


class CircuitBreaker:
    """Decides, per ScaledObject, whether the forecast may be used right now."""

    def __init__(self, cfg: BreakerConfig, clock=time.time):
        self.cfg = cfg
        self._clock = clock
        self._s: Dict[str, _State] = {}

    def observe(self, key: str, observed: float, now_ts: float, step_seconds: int,
                history: List[Optional[float]], season_steps: int,
                forecast: Optional[List[float]]) -> Tuple[bool, str]:
        """
        Called once per decision with the data the engine already holds.
        Returns (use_forecast, reason). `forecast` is this poll's horizon
        (None when serving failed); `history` the grid the scaler fetched.
        """
        st = self._s.setdefault(key, _State(open_for=self.cfg.min_open_seconds))
        reasons = []

        # 1. residual of the forecast made for *now*
        z = self._residual_z(st, observed, now_ts, step_seconds)
        if z is not None:
            st.hot_residual = st.hot_residual + 1 if abs(z) > self.cfg.residual_trip else 0
            if abs(z) > self.cfg.residual_trip_hard or st.hot_residual >= 2:
                reasons.append(f"residual z={z:.1f}")

        # 2. level vs one season ago
        ref = self._seasonal_reference(history, season_steps)
        if ref and ref > 0 and observed > 0:
            s = observed / ref
            st.hot_level = st.hot_level + 1 if (s > self.cfg.level_shift_hi or s < self.cfg.level_shift_lo) else 0
            if st.hot_level >= self.cfg.level_shift_polls:
                reasons.append(f"level x{s:.1f} vs season")

        # 3. slope vs the largest step in the history
        if st.last_observed is not None:
            d = observed - st.last_observed
            d_max = self._max_step(history)
            if d_max > 0 and d > self.cfg.slope_factor * d_max:
                reasons.append(f"jump +{d:.3g} (> {self.cfg.slope_factor}x history max {d_max:.3g})")
        st.last_observed = observed

        if forecast:
            st.pending.append((now_ts, list(forecast), step_seconds))

        now = self._clock()
        if reasons:
            self._trip(st, now, reasons[0])
            return False, "breaker:" + "; ".join(reasons)
        if st.state == OPEN:
            if now - st.opened_at >= st.open_for:
                st.state, st.calm = HALF_OPEN, 0
            else:
                return False, f"breaker:open ({int(st.open_for - (now - st.opened_at))}s left)"
        if st.state == HALF_OPEN:
            calm = (z is None or abs(z) < self.cfg.calm_residual) and st.hot_level == 0
            st.calm = st.calm + 1 if calm else 0
            if st.calm >= self.cfg.calm_polls:
                st.state, st.open_for = CLOSED, self.cfg.min_open_seconds
                return True, "breaker:closed"
            return False, f"breaker:half_open ({st.calm}/{self.cfg.calm_polls} calm polls)"
        return True, ""

    def is_open(self, key: str) -> bool:
        st = self._s.get(key)
        return bool(st and st.state != CLOSED)

    # ── helpers ────────────────────────────────────────────────────────────
    def _trip(self, st: _State, now: float, why: str) -> None:
        if st.state == HALF_OPEN:                       # re-trip: back off
            st.open_for = min(st.open_for * 2, self.cfg.max_open_seconds)
        st.state, st.opened_at, st.calm = OPEN, now, 0
        st.trips[why.split(" ")[0]] = st.trips.get(why.split(" ")[0], 0) + 1

    def _residual_z(self, st: _State, observed: float, now_ts: float, step: int) -> Optional[float]:
        for made_at, horizon, s in list(st.pending):
            k = round((now_ts - made_at) / s)
            if 1 <= k <= len(horizon) and abs((now_ts - made_at) - k * s) <= 0.25 * s:
                r = observed - horizon[k - 1]
                st.residuals.append(r)
                break
        else:
            return None
        if len(st.residuals) < 12:
            return None
        med = _median(st.residuals)
        mad = _median([abs(x - med) for x in st.residuals]) * 1.4826
        return (r - med) / mad if mad > 1e-9 else (0.0 if abs(r - med) < 1e-9 else math.copysign(99, r - med))

    @staticmethod
    def _seasonal_reference(history: List[Optional[float]], season_steps: int) -> Optional[float]:
        if season_steps <= 0 or len(history) <= season_steps:
            return None
        v = history[-1 - season_steps]
        return v if v is not None else None

    @staticmethod
    def _max_step(history: List[Optional[float]]) -> float:
        prev, m = None, 0.0
        for v in history[:-1]:                          # exclude the current sample
            if v is not None and prev is not None:
                m = max(m, v - prev)
            prev = v
        return m


def _median(xs) -> float:
    s = sorted(xs); n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])
```

Engine integration (`engine._compute`, after the forecast call, before the
`mode` policy):

```python
use_forecast, breaker_reason = self.breaker.observe(
    key=f"{namespace}/{name}", observed=observed, now_ts=timestamps[-1],
    step_seconds=cfg.step_seconds, history=history,
    season_steps=max(0, cfg.history_steps - cfg.time_steps), forecast=horizon_or_none)
self.metrics.breaker_open.labels(**labels).set(0 if use_forecast else 1)
if not use_forecast and predicted is not None:
    predicted, source, reason = None, "observed", breaker_reason
    self.metrics.breaker_bypass.labels(**labels).inc()
```

Metrics (all per ScaledObject):

| Metric | Type | Use |
|---|---|---|
| `predictive_scaler_breaker_open` | gauge 0/1 | §3.1 exclusion windows; dashboards; `ForecastBreakerOpenLong` |
| `predictive_scaler_breaker_trips_total{reason}` | counter (`residual`, `level`, `jump`, `health`) | which signal fires; tune thresholds per workload from a week of data |
| `predictive_scaler_breaker_bypass_total` | counter | decisions that would have used a forecast and did not |
| `predictive_scaler_forecast_residual_z` | gauge | the live `z`, to set `residual_trip` with evidence |

Config lands in the trigger metadata with per-workload overrides
(`breakerResidualZ`, `breakerLevelShift`, `breakerMinOpenSeconds`), parsed
like the existing keys; disabled with `breaker: "off"` for the benchmark
runs that measure the model alone.

### 4.4 Guard rails the breaker does not replace

- **`maxReplicas` is the budget.** A DDoS should hit the ceiling, not the
  bill; keep it at what the service can honestly use. The breaker makes
  the climb reactive, it does not cap it.
- **Surge cap on the forecast path.** In `mode=max`, also cap the
  forecast at `observed × forecastSurgeFactor` (default 3): a model that
  asks for 10× current demand is wrong far more often than the demand is
  about to grow 10×. One line in the engine, one key in the metadata.
- **The HPA's own behaviour.** `scaleDown.stabilizationWindowSeconds`
  (300 in the chart) decides how long an over-provisioned spike costs
  after the breaker opens; for the reactive floor trigger keep
  `scaleUp.policies` at 100%/15 s as the chart has them.
- **Validate on the benchmark.** `bench/load.py` gets a `--spike
  HH:MM,duration,factor` option; the report adds *time to breaker open*,
  *replica-minutes during and after the spike* and *forecast bypasses*.
  Run it with the breaker on and off; the difference in replica-minutes in
  the hour after the spike is the number that justifies the feature.

---

## Rollout order and acceptance

| Step | Change | Done when |
|---|---|---|
| 1 | §2.1 recording-rule quality metrics, pull-based model refresh, model cache | 3 serving replicas, `forecast_model_info` shows the same version on all, `ForecastDriftHigh` computed from `scaler:forecast_mae_1h` |
| 2 | §1.1 MLflow on Postgres + S3, 2 replicas | `helm upgrade` with the MLflow pod deleted mid-training leaves a consistent registry; serving restarts with MLflow down and serves from cache |
| 3 | §1.2 Prometheus pair + Thanos, demand recording rules, training on Thanos Query with 28 days | a training run after a Prometheus PVC loss still finds 28 days |
| 4 | §4 breaker, `cpu` floor trigger, surge cap | spike benchmark: bypass within 2 polls, no forecast-driven replicas in the hour after, nightly retrain excludes the window |
| 5 | §2.2 mTLS on both loops, default-deny NetworkPolicy | `kubectl exec` from an unrelated pod cannot open 6000 or 8443; cert rotation rolls pods inside the PDB |
| 6 | §3.1 alert-driven retraining | `ForecastDriftHigh` creates one Job with the right `--exclude`, the second alert within 6 h creates none |
| 7 | §3.2 request-rate channel on one workload | registry shows the multivariate candidate winning the fair comparison, or not; the decision is recorded either way |

Everything above keeps the two invariants the prototype already has and
that make it safe to run next to production traffic: the forecast can
only add capacity (`mode=max`), and a model that cannot follow a new level
is never promoted (the level-tracking gate). The breaker adds the third:
when the present stops resembling the history, the model is out of the
loop until it does.
