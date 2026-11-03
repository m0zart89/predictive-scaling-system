# Relative delta alignment: models forecast fractions of capacity, not cores

> **Applied in PR #16.** Training (`CapacityFractionScaler`, `units` /
> `capacity_reference`, `TRAIN_UNITS` / `TRAIN_CAPACITY_REFERENCE`), serving
> (`units` in `/predict` and `/model-info`, load-time check), scaler
> (`capacity.py` resolver, `capacityQuery`, `ForecastAnswer`, drift metric)
> and the chart follow the sections below; the bootstrap Job keeps `scaled`
> because it trains on a saved export in raw cores.

Blueprint for Task 4 of the production refactor. It states what the pipeline
does today (with file references), where the scale mismatch actually is, the
target contract that removes it, and the code changes in training, serving
and the scaler, with snippets drawn from the modules they go into.

## 1. What exists today, and where it breaks

The three components already agree on one normalisation, `capacity`, kept in
one place by the chart:

| Stage | Code | Units |
|---|---|---|
| Training data | `helm/predictive-stack/templates/training.yaml` renders `TRAIN_QUERY = (<demand query>) / <capacity>` | demand ÷ capacity |
| Per-channel scaling | `training/forecast_training/model.py`: `MinMaxScaler().fit(channel)` on every channel, the target first | MinMax of *that training series* |
| Target | `features.py`: `to_target(X, Y, "delta")` = `Y − X[:, -1]` | change, in MinMax units |
| Serving | `serving/app.py` `Bundle.forecast`: scaler `transform` → features → `predict` → `from_target` → `inverse_transform` | back to demand ÷ capacity |
| Scaler | `scaler/predictive_scaler/engine.py`: input `v / cfg.capacity`, output `max(0, out * cfg.capacity)` | demand (cores) |

So the model never sees absolute cores, and `/predict` already answers in
demand ÷ capacity. Two things are still wrong:

1. **The delta is in MinMax units, not capacity units.** `MinMaxScaler`
   maps the training series' `[data_min, data_max]` to `[0, 1]`. A delta of
   0.1 means "a tenth of the range this workload covered in the training
   window", which depends on the data, not on the workload's capacity. The
   same model moved to a workload with a different range (or the same
   workload after a quiet week narrows the range) changes meaning silently.
   The inverse transform in serving hides it, but the holdout MAE, the
   breaker's residual signal and the drift threshold all live in those
   data-dependent units.
2. **`capacity` is a static trigger key.** It is a number in
   `ScaledObject.spec.triggers[].metadata` (`scalerDefaults.capacity` or
   `workloads[].capacity`). With heterogeneous pods (1-core and 8-core
   variants of one Deployment, or a limit change in a rollout) the envelope
   drifts away from the number the model and the scaler were told, and
   nothing notices.

The fix is not a new model; it is a unit contract: the target channel is
expressed as a **fraction of a reference capacity** end to end, the model
states that in its metadata, and the scaler resolves the reference capacity
at runtime instead of trusting a hard-coded value.

## 2. The target contract

For a window ending at `t` and a horizon step `k`:

```
                 CPU_{t+k} − CPU_t
    Target_k  =  ─────────────────          (dimensionless, typically in [−1, 1])
                    Capacity_ref
```

This is the requested `(CPU_t − CPU_{t−1}) / Capacity`, anchored at the last
observed sample for every step of the horizon rather than chained step by
step: the scaler asks "how much will demand move from *now* in `k` steps",
and an anchored delta answers that without accumulating `k` one-step errors.
Serving reconstructs the level as
`level_{t+k} = CPU_t / Capacity_ref + Target_k`, and the scaler multiplies by
the same reference to get cores.

**`Capacity_ref` must be fixed over the training horizon, not the live
total of running pods.** If the reference followed `replicas × limit`, every
scale-up the scaler triggers would shrink the normalised series it is
forecasting from, a feedback loop the model would learn. The reference is
the workload's *envelope*: `maxReplicas × CPU limit per pod` (what the chart
can derive), or an explicit number for workloads whose pods differ in size.
Heterogeneity between pods is then handled where it belongs, in
`targetValue` (cores one replica carries), while the model stays unit-free.

Model metadata (`forecast_meta.json`, format 5) states the contract:

```json
{"format": 5, "units": "capacity_fraction", "target": "delta",
 "capacity_reference": 12.0, "time_steps": 10, "horizon": 6, "step_seconds": 300, ...}
```

`units: "scaled"` is what every model trained before format 5 implies; the
loaders keep accepting it, so nothing has to be retrained on the day of the
rollout.

## 3. Training: identity scaling for the target channel

The input series is already `demand / Capacity_ref` (the chart's
`TRAIN_QUERY`). The only change the target needs is to stop re-scaling that
channel with `MinMaxScaler` and use a scaler that is the identity on
`[0, 1]`, so the existing `to_target(..., "delta")` *is* the contract above.
Exogenous channels (request rate, queue depth) keep `MinMaxScaler`: their
range has no capacity to be relative to.

`training/forecast_training/model.py`:

```python
from sklearn.base import BaseEstimator, TransformerMixin


class CapacityFractionScaler(BaseEstimator, TransformerMixin):
    """
    Identity transform for a series already expressed as a fraction of the
    reference capacity. Exposes data_min_/data_max_ like MinMaxScaler so
    serving's warm-up and the breaker's level checks need no special case,
    and remembers what it saw so a model can report the range it learned on.
    """

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.data_min_ = np.nanmin(X, axis=0)
        self.data_max_ = np.nanmax(X, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)

    def inverse_transform(self, X):
        return np.asarray(X, dtype=float)


def fit_scalers(chans: np.ndarray, units: str) -> list:
    """Channel 0 is the target. capacity_fraction: identity on the target,
    MinMax on the exogenous channels; scaled: MinMax everywhere (format ≤ 4)."""
    scalers = []
    for i, ch in enumerate(chans):
        if i == 0 and units == "capacity_fraction":
            scalers.append(CapacityFractionScaler().fit(ch.reshape(-1, 1)))
        else:
            scalers.append(MinMaxScaler().fit(ch.reshape(-1, 1)))
    return scalers
```

In `train_candidate` the one line
`scalers = [MinMaxScaler().fit(ch.reshape(-1, 1)) for ch in chans]` becomes
`scalers = fit_scalers(chans, units)`, and `ForecastMeta` gains
`units` and `capacity_reference`; the pipeline tags the registered version
with both (`units`, `capacity_reference`) next to `holdout_mae_step_*`. The
CronJob passes `TRAIN_UNITS=capacity_fraction` and
`TRAIN_CAPACITY_REFERENCE=<capacity>` (the same value the chart already puts
into `TRAIN_QUERY`), so the number is recorded with the model, not only
divided into its data.

A guard worth keeping in `fit`: with `units == "capacity_fraction"` refuse
to train when `nanmax(target) > 1.5`, because a series that exceeds its
reference capacity by half means the reference is wrong, and a model trained
on it will scale every workload that uses it by the same wrong factor.

What this buys beyond units: `holdout_mae_step_k` is now a fraction of
capacity, so a holdout MAE of 0.04 means "4 % of the envelope" on every
workload, and the `ForecastDriftHigh` threshold (twice the holdout MAE)
compares like with like across workloads and model versions.

## 4. Serving: say what the numbers mean

`Bundle.forecast` needs no change: it already calls `transform` and
`inverse_transform` on whatever scaler the bundle carries. What changes is
the contract the service exposes:

```python
class PredictResponse(BaseModel):
    prediction: float
    predictions: list[float]
    step_seconds: int
    units: str                 # "capacity_fraction" (format 5) | "scaled" (older models)
    capacity_reference: Optional[float] = None   # the reference the model was trained against
    model: str
    stage: str
    model_version: str
```

filled from `b.meta.get("units", "scaled")` and
`b.meta.get("capacity_reference")`, and the same two fields in
`/model-info`. `_validated()` rejects a format-5 bundle whose target scaler
is not the identity (`units` says fraction but the pickle is a
`MinMaxScaler`): a mismatch there is exactly the silent error this task
removes.

## 5. Scaler: read the coefficient, multiply by the runtime reference

Two changes in `scaler/predictive_scaler/`: the reference capacity becomes
resolvable at runtime, and the engine checks the model's units before it
multiplies.

### 5.1 Trigger metadata

| Key | Default | Meaning |
|---|---|---|
| `capacity` | `1` | static reference, as today; the fallback when the query below fails |
| `capacityQuery` | — | PromQL returning the reference envelope in cores, e.g. `max(kube_pod_container_resource_limits{namespace="shop", container="api", resource="cpu"}) * 20` (limit per pod × maxReplicas) |
| `capacityRefreshSeconds` | `300` | how long a resolved value is reused |
| `capacityDriftFactor` | `2` | a resolved value more than this factor away from the model's `capacity_reference` is logged and ignored (the static `capacity` is used) |

The umbrella chart renders `capacityQuery` for every `workloads[]` entry
from the Deployment's CPU limit and `maxReplicas`, and passes the same
number to the CronJob as `TRAIN_CAPACITY_REFERENCE`, so training and
scaling read one definition.

### 5.2 Resolving the reference (`clients.py` / `engine.py`)

```python
@dataclass(frozen=True)
class CapacityRef:
    value: float
    source: str          # "query" | "static"
    resolved_at: float


class CapacityResolver:
    """
    The reference capacity of a ScaledObject in cores. From `capacityQuery`
    when set (cached capacityRefreshSeconds), else the static `capacity`.
    Never raises: a Prometheus error, an empty answer or a non-positive value
    falls back to the static number and is logged once per key, so a bad
    capacity query degrades to today's behaviour, not to a scaling outage.
    """

    def __init__(self, prometheus: PrometheusClient, clock=time.monotonic):
        self._prom = prometheus
        self._clock = clock
        self._lock = threading.RLock()
        self._cache: Dict[str, CapacityRef] = {}
        self._warned: set = set()

    def resolve(self, key: str, cfg: ScalerConfig) -> CapacityRef:
        if not cfg.capacity_query:
            return CapacityRef(cfg.capacity, "static", self._clock())
        now = self._clock()
        with self._lock:
            hit = self._cache.get(key)
            if hit and now - hit.resolved_at < cfg.capacity_refresh_seconds:
                return hit
        try:
            value = self._prom.query_scalar(cfg.prometheus_address, cfg.capacity_query, cfg.timeout_seconds)
        except PrometheusError as exc:
            return self._fallback(key, cfg, f"capacityQuery failed: {exc}")
        if value is None or not math.isfinite(value) or value <= 0:
            return self._fallback(key, cfg, f"capacityQuery returned {value!r}")
        ref = CapacityRef(float(value), "query", now)
        with self._lock:
            self._cache[key] = ref
            self._warned.discard(key)
        return ref

    def _fallback(self, key: str, cfg: ScalerConfig, why: str) -> CapacityRef:
        with self._lock:
            first = key not in self._warned
            self._warned.add(key)
        if first:
            log.warning("event=capacity_fallback scaledobject=%s static_capacity=%g reason=%r",
                        key, cfg.capacity, why)
        return CapacityRef(cfg.capacity, "static", self._clock())
```

`PrometheusClient.query_scalar` is a ten-line instant query
(`/api/v1/query`) next to the existing `fetch_history`.

### 5.3 The multiplication, with the units check (`engine.py`, `_forecast`)

Today the engine does `predicted = max(0.0, model_out * cfg.capacity)` and
divides the input by the same static number. With the resolver and the
response's `units`:

```python
        ref = self.capacity.resolve(key, cfg)          # cores; "query" or "static"
        model_input = [None if v is None else v / ref.value for v in history]
        ...
        answer = self.forecast.predict(cfg.forecast_url, model_input, cfg.forecast_timeout_seconds,
                                       series=key, timestamp=timestamps[-1], step_seconds=cfg.step_seconds)
        # answer.predictions: the horizon as fractions of the reference capacity
        # answer.units / answer.capacity_reference: what the model says it speaks
        if answer.units == "capacity_fraction" and answer.capacity_reference:
            drift = ref.value / answer.capacity_reference
            if drift > cfg.capacity_drift_factor or drift < 1 / cfg.capacity_drift_factor:
                # the envelope moved by more than the model can be trusted across:
                # the coefficient still applies to *a* capacity, but the shape it
                # learned belongs to another; scale with the static number, flag it
                log.warning("event=capacity_drift scaledobject=%s runtime=%g trained=%g factor=%.2f",
                            key, ref.value, answer.capacity_reference, drift)
                self.metrics.capacity_drift.labels(**labels).set(drift)
        elif answer.units != "capacity_fraction":
            log.info("event=legacy_units scaledobject=%s units=%s", key, answer.units)   # format ≤ 4: MinMax units

        horizon_demand = [max(0.0, coeff * ref.value) for coeff in answer.predictions]   # fraction × cores = cores
        model_out = max(answer.predictions) if cfg.forecast_aggregation == "max" else answer.predictions[0]
        predicted = max(0.0, model_out * ref.value)
        self.metrics.capacity_reference.labels(**labels, source=ref.source).set(ref.value)
        log.info("event=forecast_scaled scaledobject=%s coefficient=%.4f capacity=%g source=%s demand=%.4g",
                 key, model_out, ref.value, ref.source, predicted)
```

The rest of the pipeline (surge cap, hysteresis gate, EWMA, `mode=max`
floor, KEDA's `ceil(reported / targetValue)`) is unchanged and in cores.
`ForecastClient.predict` returns a small `ForecastAnswer(predictions, units,
capacity_reference)` instead of a bare list; older services without the
fields yield `units="scaled"`, which the engine treats as today.

Two consequences of the units check deserve stating:

- The **breaker's level and residual signals** keep working in fractions
  (`observed / ref.value`), which is what they are meant to be: a residual
  z-score in capacity units is comparable across workloads.
- The scaler-side **forecast ledger** scores errors in fractions of the
  *runtime* reference. When `ref.source == "query"` and the envelope changes
  between the forecast and its scoring, the error moves with it. The ledger
  records `ref.value` with the forecast and scores against that, so a limit
  change does not look like model drift.

## 6. Heterogeneous pods, end to end

A Deployment with 1-core and 8-core pod variants, `maxReplicas: 20`,
`capacityQuery` = `max(limit) × 20` = 160 cores, `targetValue: "0.7"` set as
70 % of the *small* pod's core (the conservative choice: a replica is admitted
when the smallest pod would be at 70 %):

| Step | Value |
|---|---|
| Observed demand (sum of CPU over all pods) | 38 cores |
| Model input | 38 / 160 = 0.2375 |
| Model output, step 3 | +0.0625 (a 6.25 % rise of the envelope) |
| Serving level | 0.2375 + 0.0625 = 0.30 |
| Scaler demand | 0.30 × 160 = 48 cores |
| KEDA desired replicas | ceil(48 / 0.7) = 69, clamped to 20 by `maxReplicas` |

A cluster with only 8-core pods and the same workload shape sets
`targetValue: "5.6"` (70 % of 8): same model, same coefficient, 9 replicas.
Pod size never reaches the model; the reference capacity never reaches the
replica count except through `targetValue`.

## 7. Rollout

1. **Training** (`model.py`, `registry.py`, `__main__.py`): `CapacityFractionScaler`,
   `units` / `capacity_reference` in `ForecastMeta` (format 5) and in the
   version tags, `TRAIN_UNITS` / `TRAIN_CAPACITY_REFERENCE` from the chart,
   the `> 1.5` guard. Tests: a format-5 bundle round-trips through
   `to_target` / `from_target` with the identity scaler and reproduces
   `(CPU_{t+k} − CPU_t) / C` to machine precision; a format-4 model still loads.
2. **Serving** (`app.py`): `units` and `capacity_reference` in `/predict`
   and `/model-info`; `_validated` rejects a format-5 bundle with a MinMax
   target scaler. Tests on both formats.
3. **Scaler** (`config.py`, `clients.py`, `engine.py`, `metrics.py`):
   `capacityQuery`, `capacityRefreshSeconds`, `capacityDriftFactor`;
   `CapacityResolver`; `ForecastAnswer`; `predictive_scaler_capacity_reference{source}`
   and `predictive_scaler_capacity_drift`; ledger scored against the recorded
   reference. Tests: query vs static, fallback on Prometheus error, drift
   warning, a legacy answer without `units`.
4. **Chart**: `capacityQuery` per workload from the Deployment's limit ×
   `maxReplicas` (operators can override), the same number into
   `TRAIN_CAPACITY_REFERENCE`; CI greps both.
5. **Migration**: deploy serving and scaler first (both accept `units:
   scaled`), then the training image; the next nightly run registers a
   format-5 champion and the alias moves. No cut-over window: a format-4
   champion keeps serving with the MinMax inverse transform until then.

Order 1 → 3 can ship as one PR per component; nothing in 1 is live until 5.
