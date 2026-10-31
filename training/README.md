# forecast-training

Retraining pipeline for the workload forecasting model. Replaces the manual
notebook run with a job that can run nightly and only ever moves the
`@champion` alias to a model that is both safe and better.

```
source ─► regularise ─► train candidates ─► holdout RMSE + level tracking
                                                    │
                     compare with @champion on data it has not seen
                                                    │
               register version, tag data_end_ts, move @champion ─► POST /reload
```

## Run

```bash
pip install -r requirements.txt
export MLFLOW_TRACKING_URI=http://mlflow:5000

# from the saved January export
python -m forecast_training --export ../notebooks/data-jan-2022.json \
    --exclude 2022-01-17T00:00:00Z/2022-01-17T08:00:00Z

# from live Prometheus, then reload serving
python -m forecast_training --prometheus http://prometheus:9090 \
    --query 'sum(rate(process_cpu_seconds_total{app="cpu-burner"}[5m]))' \
    --days 14 --reload-url http://forecast-serving:8080
```

Every flag has an environment variable (`TRAIN_*`, `MODEL_NAME`, `MODEL_ALIAS`,
`SERVING_RELOAD_URL`) — see `python -m forecast_training --help` and
the chart's CronJob (`helm/predictive-stack/templates/training.yaml`). The JSON
summary on stdout says what happened:
`promoted`, `kept_champion`, `rejected` or `dry_run`, with the reason.

## Candidates

| Name | Features | Target | Notes |
|---|---|---|---|
| `rf_level` | scaled window | next values | what the notebook promoted |
| `rf_delta` | window relative to last value, plus last value | change from last value | |
| `hgb_delta` | same | same | gradient boosting, one model per horizon step |
| `ridge_delta` | same | same | linear baseline |
| `hgb_delta_tod` | anchored + sin/cos of the seasonal phase | same | knows what time it is: can move before a daily ramp shows in the window |
| `rf_delta_tod` | same | same | |

All predict the whole horizon (default 6 steps = 30 min). The `_tod`
candidates need timestamps (both sources provide them) and a seasonal cycle:
`--period-seconds`, 86400 for real days, the compressed day length when time
is compressed as in the benchmark. The scaler sends the last sample's
timestamp with every `/predict`, so serving evaluates the phase for the right
instant.

## Promotion rules

1. **Level tracking.** Each candidate's holdout windows are lifted twice, by
   the training maximum m and by 2m, and `mean(f(X+2m) − f(X+m)) / m` is
   measured: 1.0 = forecast follows the new level, 0 = pinned to the training
   range. Below `--min-tracking` (0.5) a candidate is not eligible, whatever
   its RMSE. A pinned model forecasts *less* load than is arriving, which for
   an autoscaler means scaling down into a spike.
2. The best eligible candidate must beat the persistence baseline.
3. No champion yet → promote. Champion fails level tracking → replace it.
4. Otherwise compare both models on windows after the holdout split **and**
   after the champion's `data_end_ts` tag, i.e. data neither model was fitted
   to. Fewer than `--min-fresh-windows` (48) such windows → keep the champion.
   Promote if the new RMSE is lower by at least `--min-improvement`.

## Model contract (shared with serving)

A registered version has three artifacts:

- `model/` — the sklearn estimator (pickled with scikit-learn 1.4.2; keep
  `requirements.txt` in step with `serving/requirements.txt`)
- `scaler/scaler.pkl` — MinMaxScaler fitted on the training values
- `forecast_meta.json` — `time_steps`, `horizon`, `step_seconds`,
  `features` (`window` | `anchored` | `anchored_tod`), `target`
  (`level` | `delta`), `period_seconds` (seasonal cycle for `anchored_tod`)

`forecast_training/features.py` is the single implementation of the
transforms: the serving image copies it (`serving/Dockerfile` builds from the
repository root), so keep it free of dependencies beyond numpy.
`tests/test_training_serving.py` at the repo root checks that a registered
model serves exactly the forecast the pipeline computed, and that
notebook-registered models (no meta, `Production` stage) still load.

## Real traces

`forecast_training.traces` converts public traffic data into the export
shape the pipeline reads, so the same code can be shown on something other
than the synthetic demo:

```bash
# hourly page views of a whole Wikipedia project: a real daily + weekly pattern, no key needed
python -m forecast_training.traces wikipedia --project en.wikipedia --days 30 --out traces/views.json
MLFLOW_TRACKING_URI=sqlite:///mlflow.db python -m forecast_training --export traces/views.json \
    --step-seconds 3600 --time-steps 24 --horizon 6 --period-seconds 86400 --dry-run

# any timestamp/value CSV, e.g. Alibaba cluster-trace-v2018 machine_usage.csv summed over machines
python -m forecast_training.traces csv --path machine_usage.csv --ts-col 1 --value-col 2 \
    --ts-offset 1514736000 --scale 0.01 --out traces/alibaba-cpu.json
```

`.github/workflows/traces.yml` runs the Wikipedia case weekly for three
projects (English, German, Japanese: the same daily shape at different UTC
hours) and fails if no candidate beats persistence; the candidate table is in
the job output and artifacts.

## Tests

```bash
pytest tests/          # unit tests + end-to-end against a throwaway SQLite MLflow
```
