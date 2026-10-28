"""Client tests against real local HTTP servers (no mocking of urllib)."""

import json
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from predictive_scaler.clients import ForecastClient, ForecastError, PrometheusClient, PrometheusError


class Responder:
    """A tiny HTTP server whose response is set per test."""

    def __init__(self):
        self.status = 200
        self.body = {}
        self.requests = []
        responder = self

        class Handler(BaseHTTPRequestHandler):
            def _reply(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = self.rfile.read(length) if length else b""
                responder.requests.append((self.command, self.path, payload))
                raw = responder.body if isinstance(responder.body, bytes) \
                    else json.dumps(responder.body).encode()
                self.send_response(responder.status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            do_GET = _reply
            do_POST = _reply

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def http():
    r = Responder()
    yield r
    r.close()


def matrix(*series):
    return {"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": {}, "values": [[1700000000 + 30 * i, str(v)] for i, v in enumerate(s)]}
        for s in series
    ]}}


# ── Prometheus ─────────────────────────────────────────────────────────────────

def test_prometheus_window_is_parsed_and_trimmed(http):
    http.body = matrix([0.1, 0.2, 0.3, 0.4])
    values = PrometheusClient().fetch_window(http.url, "up", points=3, step_seconds=30,
                                             timeout=2, now=1700000100)
    assert values == [0.2, 0.3, 0.4]

    _, path, _ = http.requests[0]
    q = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
    assert urllib.parse.urlparse(path).path == "/api/v1/query_range"
    assert q["query"] == ["up"]
    assert q["step"] == ["30"]
    assert float(q["end"][0]) - float(q["start"][0]) == pytest.approx(90)


def test_prometheus_keeps_only_the_trailing_run_after_a_hole(http):
    # a NaN (division by zero) or missing sample is a hole: the window restarts after it
    http.body = matrix(["0.1", "NaN", "+Inf", "0.3", "0.4"])
    ts, values = PrometheusClient().fetch_series(http.url, "q", 10, 30, 2)
    assert values == [0.3, 0.4]
    assert ts == [1700000090.0, 1700000120.0]
    assert PrometheusClient().fetch_window(http.url, "q", 10, 30, 2) == [0.3, 0.4]


def test_prometheus_samples_must_be_step_seconds_apart(http):
    http.body = {"status": "success", "data": {"resultType": "matrix", "result": [
        {"metric": {}, "values": [[1700000000, "0.1"], [1700000030, "0.2"],
                                  [1700000090, "0.3"], [1700000120, "0.4"]]}]}}
    ts, values = PrometheusClient().fetch_series(http.url, "q", 10, 30, 2)
    assert values == [0.3, 0.4] and ts[-1] == 1700000120.0


def test_prometheus_empty_result_is_empty_list(http):
    http.body = matrix()
    assert PrometheusClient().fetch_window(http.url, "q", 10, 30, 2) == []
    assert PrometheusClient().fetch_series(http.url, "q", 10, 30, 2) == ([], [])


def test_prometheus_rejects_multiple_series(http):
    http.body = matrix([0.1], [0.2])
    with pytest.raises(PrometheusError, match="sum"):
        PrometheusClient().fetch_window(http.url, "q", 10, 30, 2)


def test_prometheus_query_error(http):
    http.status = 400
    http.body = {"status": "error", "error": "parse error"}
    with pytest.raises(PrometheusError, match="400"):
        PrometheusClient().fetch_window(http.url, "q(", 10, 30, 2)


def test_prometheus_error_status_in_body(http):
    http.body = {"status": "error", "error": "query timed out"}
    with pytest.raises(PrometheusError, match="timed out"):
        PrometheusClient().fetch_window(http.url, "q", 10, 30, 2)


def test_prometheus_unreachable():
    with pytest.raises(PrometheusError, match="unreachable"):
        PrometheusClient().fetch_window("http://127.0.0.1:1", "q", 10, 30, 1)


# ── Forecast service ───────────────────────────────────────────────────────────

def test_forecast_posts_values_and_reads_prediction(http):
    # a service that predates multi-step forecasts: only `prediction`
    http.body = {"prediction": 0.42, "model": "cpu-forecasting", "stage": "Production"}
    assert ForecastClient().predict(http.url, [0.1, 0.2], timeout=2) == [0.42]
    method, path, payload = http.requests[0]
    assert (method, path) == ("POST", "/predict")
    assert json.loads(payload) == {"values": [0.1, 0.2]}


def test_forecast_reads_the_whole_horizon(http):
    http.body = {"prediction": 0.4, "predictions": [0.4, 0.6, 0.5], "step_seconds": 300}
    assert ForecastClient().predict(http.url, [0.1], timeout=2) == [0.4, 0.6, 0.5]


@pytest.mark.parametrize("status,reason", [(503, "not_ready"), (422, "bad_response"),
                                           (500, "bad_response")])
def test_forecast_http_errors_are_classified(http, status, reason):
    http.status = status
    http.body = {"detail": "nope"}
    with pytest.raises(ForecastError) as err:
        ForecastClient().predict(http.url, [0.1], timeout=2)
    assert err.value.reason == reason


@pytest.mark.parametrize("body", [{}, {"prediction": "abc"}, {"prediction": None},
                                  {"prediction": 0.1, "predictions": [0.1, "NaN"]},
                                  [0.1], b"not json"])
def test_forecast_bad_payload(http, body):
    http.body = body
    with pytest.raises(ForecastError) as err:
        ForecastClient().predict(http.url, [0.1], timeout=2)
    assert err.value.reason == "bad_response"


def test_forecast_unreachable():
    with pytest.raises(ForecastError) as err:
        ForecastClient().predict("http://127.0.0.1:1", [0.1], timeout=1)
    assert err.value.reason == "unavailable"


def test_forecast_sends_series_and_timestamp_when_given(http):
    http.body = {"prediction": 0.4}
    ForecastClient().predict(http.url, [0.1], timeout=2, series="demo/cpu-burner", timestamp=1700.5,
                             step_seconds=30)
    body = json.loads(http.requests[-1][2])
    assert body["series"] == "demo/cpu-burner" and body["timestamp"] == 1700.5
    assert body["step_seconds"] == 30
    ForecastClient().predict(http.url, [0.1], timeout=2)
    assert set(json.loads(http.requests[-1][2])) == {"values"}
