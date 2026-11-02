# ForecastBreakerOpenLong

**Severity:** warning · **Service:** predictive-scaler

The scaler's circuit breaker (`scaler/predictive_scaler/breaker.py`) has kept
the forecast out of the scaling decision for two hours on one ScaledObject:
`predictive_scaler_breaker_open == 1`. The workload is scaled reactively
meanwhile — on observed demand (`mode=max` reports it), plus the CPU floor
trigger when `reactiveFloor` is enabled — so this is a degradation of the
predictive behaviour, not an outage.

## What it means

The series keeps looking unlike the history the model learned. Which signal
keeps tripping is in `predictive_scaler_breaker_trips_total{reason}`:

| reason | meaning | typical cause |
|---|---|---|
| `jump` | a one-step rise larger than `breakerJumpFactor` × any rise in the fetched history | traffic failover, flash sale, DDoS, a new release with different cost per request |
| `level` | observed / value one season ago outside `1/breakerLevelShift … breakerLevelShift` for `breakerCalmPolls` polls | the daily pattern moved (campaign, holiday, a region's traffic added) |
| `residual` | the forecast made for now is off by more than `breakerResidualZ` robust standard deviations | the model has drifted; the same thing `ForecastDriftHigh` measures |

Two hours of it means the series is not coming back to the pattern: the
model is stale, not the traffic odd.

## What to do

1. Look at the `Predictive scaling` dashboard for the ScaledObject: observed
   vs forecast demand, `predictive_scaler_breaker_open`, the trip counter.
2. If the new level is the new normal (a launch, a migration), retrain now
   on recent data: `kubectl -n <ns> create job --from=cronjob/forecast-retrain retrain-now`.
   The alert-driven trigger (`retrain-trigger`, when deployed) already does
   this on `ForecastBreakerOpenLong` and excludes the breaker's open intervals
   from the training window.
3. If the traffic is abnormal (attack, a client gone wild), leave the breaker
   open: that is the design. Fix the traffic; the breaker closes on its own
   after `breakerCalmPolls` calm polls once the hold time elapsed.
4. If it trips on noise (`residual` on a workload with bursty demand), raise
   `breakerResidualZ` for that workload in `.Values.workloads[].breakerResidualZ`,
   or set `breaker: "off"` and rely on `mode=max` plus the CPU floor.

## Related

- `ForecastDriftHigh` — the model's error as measured by the scaler's own
  scoring and serving's tracker.
- `docs/production-blueprint.md` §4 — design of the breaker and its thresholds.
