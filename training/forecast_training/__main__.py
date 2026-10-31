"""
Retrain the workload forecasting model.

    # from a saved Prometheus export (the notebook's data)
    python -m forecast_training --export notebooks/data-jan-2022.json \
        --exclude 2022-01-17T00:00:00Z/2022-01-17T08:00:00Z

    # from live Prometheus (what the CronJob runs)
    python -m forecast_training --prometheus http://prometheus:9090 \
        --query 'avg(rate(container_cpu_usage_seconds_total{namespace="demo"}[5m]))' \
        --days 14 --reload-url http://forecast-serving:8080

The MLflow server comes from MLFLOW_TRACKING_URI. Prints a JSON summary on
stdout; exits 0 whether or not a new model was promoted, 1 on errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .data import DataError, load_export, parse_exclude, query_prometheus, regularize
from .model import CANDIDATE_NAMES
from .registry import TrainConfig, reload_serving, run_pipeline


def _env(name: str, default=None):
    v = os.getenv(name)
    return v if v not in (None, "") else default


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="forecast_training", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group()
    src.add_argument("--export", default=_env("TRAIN_EXPORT"),
                     help="path to a saved Prometheus query_range JSON")
    src.add_argument("--prometheus", default=_env("TRAIN_PROMETHEUS_URL"),
                     help="Prometheus base URL to query")
    p.add_argument("--query", default=_env("TRAIN_QUERY"),
                   help="PromQL returning one series (required with --prometheus)")
    p.add_argument("--days", type=float, default=float(_env("TRAIN_DAYS", 14)),
                   help="history to train on, with --prometheus (default 14)")
    p.add_argument("--exclude", action="append", default=[], metavar="START/END",
                   help="time range to treat as missing (ISO or unix); repeatable")

    p.add_argument("--time-steps", type=int, default=int(_env("TRAIN_TIME_STEPS", 10)))
    p.add_argument("--horizon", type=int, default=int(_env("TRAIN_HORIZON", 6)),
                   help="steps to forecast; 6 × 5 min = 30 min (default 6)")
    p.add_argument("--step-seconds", type=int, default=int(_env("TRAIN_STEP_SECONDS", 300)))
    p.add_argument("--period-seconds", type=int, default=int(_env("TRAIN_PERIOD_SECONDS", 86400)),
                   help="seasonal cycle for the time-of-day features: 86400 for real days, the "
                        "compressed day length when time is compressed (default 86400)")
    p.add_argument("--test-fraction", type=float, default=float(_env("TRAIN_TEST_FRACTION", 0.2)))
    p.add_argument("--candidates", default=_env("TRAIN_CANDIDATES", ",".join(CANDIDATE_NAMES)),
                   help=f"comma-separated subset of: {', '.join(CANDIDATE_NAMES)}")
    p.add_argument("--min-tracking", type=float, default=float(_env("TRAIN_MIN_TRACKING", 0.5)),
                   help="minimum level tracking for a candidate to be eligible (default 0.5)")
    p.add_argument("--min-improvement", type=float,
                   default=float(_env("TRAIN_MIN_IMPROVEMENT", 0.0)),
                   help="relative RMSE improvement needed to replace the champion (e.g. 0.02)")
    p.add_argument("--min-fresh-windows", type=int,
                   default=int(_env("TRAIN_MIN_FRESH_WINDOWS", 48)),
                   help="windows newer than the champion's training data needed to compare "
                        "against it (default 48 = 4 h at 5 min)")
    p.add_argument("--no-refit", action="store_true",
                   help="register the model fitted on the training split instead of refitting "
                        "on all data")

    p.add_argument("--model-name", default=_env("MODEL_NAME", "cpu-forecasting"))
    p.add_argument("--alias", default=_env("MODEL_ALIAS", "champion"))
    p.add_argument("--experiment", default=_env("MLFLOW_EXPERIMENT", "cpu-forecasting"))
    p.add_argument("--reload-url", default=_env("SERVING_RELOAD_URL"),
                   help="forecast-serving base URL to POST /reload after a promotion")
    p.add_argument("--dry-run", action="store_true",
                   help="train, compare and log, but do not register or move the alias")
    p.add_argument("--summary", help="also write the JSON summary to this file")
    return p


def main(argv=None) -> int:
    logging.basicConfig(level=_env("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ", stream=sys.stderr)
    log = logging.getLogger("forecast-training")
    args = build_parser().parse_args(argv)

    try:
        exclude = [parse_exclude(s) for s in args.exclude]
        if args.export:
            points, source = load_export(args.export), f"export:{os.path.basename(args.export)}"
        elif args.prometheus:
            if not args.query:
                raise ValueError("--query is required with --prometheus")
            points = query_prometheus(args.prometheus, args.query, args.days, args.step_seconds)
            source = f"prometheus:{args.days:g}d"
        else:
            raise ValueError("one of --export or --prometheus is required")

        series = regularize(points, args.step_seconds, exclude)
        log.info("series: %d points, %d filled", len(series), series.filled)

        cfg = TrainConfig(
            model_name=args.model_name, alias=args.alias, experiment=args.experiment,
            time_steps=args.time_steps, horizon=args.horizon, step_seconds=args.step_seconds,
            period_seconds=args.period_seconds, test_fraction=args.test_fraction, min_tracking=args.min_tracking,
            min_improvement=args.min_improvement, min_fresh_windows=args.min_fresh_windows,
            refit=not args.no_refit, dry_run=args.dry_run,
            candidates=[c.strip() for c in args.candidates.split(",") if c.strip()],
        )
        outcome = run_pipeline(series.values, cfg, source, series.filled, series.timestamps)
    except (DataError, ValueError, OSError) as exc:
        log.error("%s", exc)
        return 1

    summary = outcome.to_dict()
    if outcome.promoted and args.reload_url:
        summary["serving_reloaded"] = reload_serving(args.reload_url)

    log.info("%s: %s", outcome.decision, outcome.reason)
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.summary:
        with open(args.summary, "w") as f:
            f.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
