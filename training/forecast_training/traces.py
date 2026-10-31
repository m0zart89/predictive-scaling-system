"""
Turn public workload traces into the export format the pipeline trains on.

The pipeline reads a Prometheus `query_range` response (one series of
(timestamp, value)); `regularize` puts it on the step grid and fills gaps.
This module writes that shape from two kinds of source, so the same training
code can be shown on real traffic, not only on the synthetic demo:

    # hourly page views of a whole Wikipedia project (a real daily + weekly
    # pattern, free, no key; per-article data is daily only)
    python -m forecast_training.traces wikipedia --project en.wikipedia --days 30 \
        --out traces/wikipedia-en.json
    python -m forecast_training --export traces/wikipedia-en.json \
        --step-seconds 3600 --time-steps 24 --horizon 6 --dry-run

    # any CSV with a timestamp and a value column, e.g. Alibaba
    # cluster-trace-v2018 machine_usage.csv (machine_id, time_stamp,
    # cpu_util_percent, ...), summed over machines per timestamp
    python -m forecast_training.traces csv --path machine_usage.csv \
        --ts-col 1 --value-col 2 --ts-offset 1514736000 --out traces/alibaba-cpu.json

The Wikimedia REST API: https://wikimedia.org/api/rest_v1/#/Pageviews%20data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Sequence, Tuple

Point = Tuple[int, float]

WIKIMEDIA = "https://wikimedia.org/api/rest_v1/metrics/pageviews/aggregate"
USER_AGENT = "predictive-scaling-system/1.1 (https://github.com/m0zart89/predictive-scaling-system)"


# ── sources ────────────────────────────────────────────────────────────────────

def parse_wikimedia(body: dict) -> List[Point]:
    """`items` of a pageviews response → (unix hour, views)."""
    points = []
    for item in body.get("items", []):
        ts = datetime.strptime(item["timestamp"], "%Y%m%d%H").replace(tzinfo=timezone.utc)
        points.append((int(ts.timestamp()), float(item["views"])))
    return points


def wikipedia_pageviews(project: str = "en.wikipedia", days: int = 30,
                        end: datetime | None = None, timeout: float = 60.0) -> List[Point]:
    """Hourly user views of a whole project over the last `days` whole days (ending yesterday)."""
    today = (end or datetime.now(timezone.utc)).replace(hour=0, minute=0, second=0, microsecond=0)
    last = today - timedelta(hours=1)                # yesterday 23:00; today is not final yet
    first = today - timedelta(days=days)             # 00:00, `days` days ago
    url = "/".join([WIKIMEDIA, urllib.parse.quote(project, safe=""), "all-access", "user",
                    "hourly", first.strftime("%Y%m%d%H"), last.strftime("%Y%m%d%H")])
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
            return parse_wikimedia(json.load(resp))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:500]
        raise RuntimeError(f"Wikimedia API returned HTTP {exc.code} for {url}: {detail}") from exc


def csv_series(path: str, ts_col, value_col, aggregate: str = "sum", ts_offset: int = 0,
               ts_format: str | None = None, delimiter: str = ",", header: bool | None = None,
               scale: float = 1.0) -> List[Point]:
    """
    One series from a CSV: rows with the same timestamp are combined with
    `aggregate` (sum | mean | max). Columns are names (needs a header) or
    0-based indexes. `ts_offset` is added to numeric timestamps (traces often
    count seconds from the trace start); `ts_format` parses string timestamps
    with strptime (UTC).
    """
    sums: Dict[int, List[float]] = defaultdict(list)
    with open(path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        first = next(reader, None)
        if first is None:
            raise ValueError(f"{path} is empty")
        # a header has no numeric cell at all; a data row with a string id still has some
        has_header = header if header is not None else not _has_numbers(first)
        names = first if has_header else None
        rows: Iterable[Sequence[str]] = reader if has_header else _chain(first, reader)
        ti, vi = _col(ts_col, names), _col(value_col, names)
        for row in rows:
            if len(row) <= max(ti, vi):
                continue
            try:
                raw_ts, value = row[ti].strip(), float(row[vi])
            except ValueError:
                continue
            if ts_format:
                ts = int(datetime.strptime(raw_ts, ts_format).replace(tzinfo=timezone.utc).timestamp())
            else:
                ts = int(float(raw_ts)) + ts_offset
            sums[ts].append(value * scale)
    if not sums:
        raise ValueError(f"{path}: no usable rows")
    agg = {"sum": sum, "mean": lambda xs: sum(xs) / len(xs), "max": max}[aggregate]
    return sorted((ts, float(agg(xs))) for ts, xs in sums.items())


def _has_numbers(row: Sequence[str]) -> bool:
    for cell in row:
        try:
            float(cell)
            return True
        except ValueError:
            continue
    return False


def _chain(first, rest):
    yield first
    yield from rest


def _col(spec, names) -> int:
    if isinstance(spec, int) or str(spec).isdigit():
        return int(spec)
    if names is None or spec not in names:
        raise ValueError(f"column {spec!r} needs a header row that contains it")
    return names.index(spec)


# ── export ─────────────────────────────────────────────────────────────────────

def to_export(points: Sequence[Point], metric: dict | None = None) -> dict:
    """The query_range response shape that forecast_training --export reads."""
    return {"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": metric or {}, "values": [[ts, str(v)] for ts, v in points]}]}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="forecast_training.traces", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="source", required=True)
    w = sub.add_parser("wikipedia", help="hourly page views of a whole project")
    w.add_argument("--project", default="en.wikipedia", help="e.g. en.wikipedia, de.wikipedia")
    w.add_argument("--days", type=int, default=30)
    w.add_argument("--out", required=True)
    c = sub.add_parser("csv", help="a timestamp/value CSV, rows per timestamp combined")
    c.add_argument("--path", required=True)
    c.add_argument("--ts-col", required=True, help="column name or 0-based index")
    c.add_argument("--value-col", required=True, help="column name or 0-based index")
    c.add_argument("--aggregate", choices=["sum", "mean", "max"], default="sum")
    c.add_argument("--ts-offset", type=int, default=0, help="added to numeric timestamps")
    c.add_argument("--ts-format", help="strptime format for string timestamps (UTC)")
    c.add_argument("--delimiter", default=",")
    c.add_argument("--scale", type=float, default=1.0, help="multiply values (e.g. 0.01 for percent)")
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.source == "wikipedia":
        points = wikipedia_pageviews(args.project, args.days)
        metric = {"source": "wikimedia-pageviews", "project": args.project}
    else:
        points = csv_series(args.path, args.ts_col, args.value_col, args.aggregate, args.ts_offset,
                            args.ts_format, args.delimiter, scale=args.scale)
        metric = {"source": "csv", "path": args.path}
    with open(args.out, "w") as f:
        json.dump(to_export(points, metric), f)
    span = (points[-1][0] - points[0][0]) / 86400 if len(points) > 1 else 0
    print(f"{len(points)} points, {span:.1f} days, {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
