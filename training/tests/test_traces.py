"""forecast_training.traces: public traces → the export shape the pipeline reads."""

import json

import pytest

from forecast_training.data import load_export, regularize
from forecast_training.traces import csv_series, main, parse_wikimedia, to_export


def test_wikimedia_items_become_hourly_points(tmp_path):
    body = {"items": [
        {"project": "en.wikipedia", "article": "Kubernetes", "timestamp": "2025100100", "views": 120},
        {"project": "en.wikipedia", "article": "Kubernetes", "timestamp": "2025100101", "views": 95},
    ]}
    points = parse_wikimedia(body)
    assert points == [(1759276800, 120.0), (1759280400, 95.0)]
    assert points[1][0] - points[0][0] == 3600

    out = tmp_path / "wiki.json"
    out.write_text(json.dumps(to_export(points, {"article": "Kubernetes"})))
    assert load_export(str(out)) == points          # round-trips through the pipeline's reader


def test_csv_rows_with_the_same_timestamp_are_combined(tmp_path):
    p = tmp_path / "machine_usage.csv"            # Alibaba-like: no header, seconds from trace start
    p.write_text("m_1,0,40,10\nm_2,0,20,10\nm_1,10,50,10\nm_2,10,30,10\nm_3,10,bad,10\n")
    points = csv_series(str(p), 1, 2, aggregate="sum", ts_offset=1_000_000, scale=0.01)
    assert points == [(1_000_000, pytest.approx(0.6)), (1_000_010, pytest.approx(0.8))]
    assert csv_series(str(p), 1, 2, aggregate="max")[1] == (10, 50.0)   # no offset here

    q = tmp_path / "named.csv"                     # header, string timestamps
    q.write_text("time,cpu\n2025-10-01 00:00:00,1.5\n2025-10-01 00:05:00,2.5\n")
    points = csv_series(str(q), "time", "cpu", ts_format="%Y-%m-%d %H:%M:%S")
    assert points == [(1759276800, 1.5), (1759277100, 2.5)]
    with pytest.raises(ValueError, match="header"):
        csv_series(str(p), "time", "cpu")


def test_cli_writes_an_export_the_pipeline_can_regularize(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("".join(f"{i * 3600},{10 + (i % 24)}\n" for i in range(48)))
    out = tmp_path / "t.json"
    assert main(["csv", "--path", str(p), "--ts-col", "0", "--value-col", "1",
                 "--ts-offset", "1700000000", "--out", str(out)]) == 0
    series = regularize(load_export(str(out)), 3600)
    assert len(series) == 48 and series.filled == 0
