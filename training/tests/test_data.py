import json

import numpy as np
import pytest

from forecast_training.data import DataError, load_export, parse_exclude, regularize

STEP = 300


def test_regular_input_is_unchanged():
    pts = [(1000 + i * STEP, float(i)) for i in range(10)]
    s = regularize(pts, STEP)
    assert s.filled == 0
    assert list(s.values) == [float(i) for i in range(10)]
    assert np.all(np.diff(s.timestamps) == STEP)


def test_jittered_and_unsorted_points_snap_to_grid():
    pts = [(1000 + 2 * STEP + 7, 2.0), (1000, 0.0), (1000 + STEP - 4, 1.0)]
    s = regularize(pts, STEP)
    assert list(s.values) == [0.0, 1.0, 2.0]


def test_duplicates_in_a_slot_are_averaged():
    s = regularize([(0, 1.0), (10, 3.0), (STEP, 5.0)], STEP)
    assert list(s.values) == [2.0, 5.0]


def test_gap_is_filled_from_same_time_of_day():
    day = 86400 // STEP
    # two days, constant 1.0 except the same slot on day 1 = 9.0; drop it on day 2
    pts = [(i * STEP, 9.0 if i % day == 5 else 1.0) for i in range(2 * day) if i != day + 5]
    s = regularize(pts, STEP)
    assert s.filled == 1
    assert s.values[day + 5] == pytest.approx(9.0)   # profile, not interpolation (1.0)


def test_excluded_range_is_replaced():
    day = 86400 // STEP
    pts = [(i * STEP, 1.0) for i in range(2 * day)]
    pts[day + 10] = (pts[day + 10][0], 0.0)           # an "outage" reading
    s = regularize(pts, STEP, exclude=[(pts[day + 10][0], pts[day + 10][0])])
    assert s.values[day + 10] == pytest.approx(1.0)
    assert s.filled == 1


def test_interpolates_when_time_of_day_has_no_data():
    s = regularize([(0, 0.0), (3 * STEP, 3.0)], STEP)
    assert list(s.values) == pytest.approx([0.0, 1.0, 2.0, 3.0])


def test_all_excluded_is_an_error():
    with pytest.raises(DataError):
        regularize([(0, 1.0), (STEP, 1.0)], STEP, exclude=[(0, STEP)])


def test_empty_is_an_error():
    with pytest.raises(DataError):
        regularize([], STEP)


def test_load_export(tmp_path):
    body = {"resultType": "matrix", "result": [{"metric": {}, "values": [
        [1641773100, "0.1"], [1641773400, "NaN"], [1641773700, "0.3"]]}]}
    p = tmp_path / "export.json"
    p.write_text(json.dumps(body))
    assert load_export(str(p)) == [(1641773100, 0.1), (1641773700, 0.3)]


def test_load_export_rejects_multiple_series(tmp_path):
    body = {"result": [{"values": []}, {"values": []}]}
    p = tmp_path / "export.json"
    p.write_text(json.dumps(body))
    with pytest.raises(DataError, match="sum"):
        load_export(str(p))


@pytest.mark.parametrize("spec,expected", [
    ("2022-01-17T00:00:00Z/2022-01-17T08:00:00Z", (1642377600, 1642406400)),
    ("2022-01-17T00:00:00/2022-01-17T08:00:00", (1642377600, 1642406400)),
    ("100/200", (100, 200)),
])
def test_parse_exclude(spec, expected):
    assert parse_exclude(spec) == expected


@pytest.mark.parametrize("spec", ["2022-01-17", "200/100"])
def test_parse_exclude_rejects(spec):
    with pytest.raises(ValueError):
        parse_exclude(spec)
