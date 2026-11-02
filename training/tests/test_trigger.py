"""retrain trigger: Alertmanager webhook → a Job cloned from the CronJob, with the
breaker's open intervals excluded; dedup, rate limit, auth. Real HTTP servers
stand in for the Kubernetes API and Prometheus."""

import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from forecast_training import trigger as t

CRONJOB = {
    "metadata": {"name": "forecast-retrain"},
    "spec": {"schedule": "17 3 * * *", "jobTemplate": {"spec": {
        "backoffLimit": 1,
        "template": {"metadata": {"labels": {"app": "forecast-retrain"}}, "spec": {
            "restartPolicy": "Never",
            "containers": [{"name": "train", "image": "forecast-training:test",
                            "args": ["--candidates=hgb_delta"],
                            "env": [{"name": "TRAIN_DAYS", "value": "7"},
                                    {"name": "TRAIN_MIN_IMPROVEMENT", "value": "0.02"}]}]}}}}}}


class FakeApi:
    """Kubernetes API + Prometheus on one local server."""

    def __init__(self):
        self.jobs = []
        self.breaker = []          # (ts, "1") samples for the breaker series
        api = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                path = urllib.parse.urlparse(self.path)
                if path.path.endswith("/cronjobs/forecast-retrain"):
                    return self._json(200, CRONJOB)
                if path.path.endswith("/jobs"):
                    return self._json(200, {"items": api.jobs})
                if path.path == "/api/v1/query_range":
                    api.last_query = urllib.parse.parse_qs(path.query)
                    result = [{"metric": {}, "values": api.breaker}] if api.breaker else []
                    return self._json(200, {"status": "success", "data": {"resultType": "matrix", "result": result}})
                self._json(404, {})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                api.created = body
                api.jobs.append({**body, "status": {"active": 1}})
                self._json(201, body)

            def _json(self, code, body):
                raw = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)

            def log_message(self, *a):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def api():
    a = FakeApi()
    yield a
    a.close()


@pytest.fixture
def cfg(api):
    return t.TriggerConfig(namespace="predictive", token="s3cret", prometheus_url=api.url,
                           min_gap=timedelta(hours=6))


@pytest.fixture
def receiver(api, cfg):
    kube = t.KubeClient(base_url=api.url, token="sa-token")
    server = HTTPServer(("127.0.0.1", 0), t.make_handler(cfg, kube))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def post(url, payload, token="s3cret"):
    req = urllib.request.Request(f"{url}/alertmanager", data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {token}"} if token else {})})
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


def alert(name="ForecastBreakerOpenLong", status="firing", **labels):
    return {"status": status, "labels": {"alertname": name, **labels}, "fingerprint": "abc123"}


def test_firing_alert_starts_a_job_cloned_from_the_cronjob_with_exclusions(api, receiver):
    now = 1_700_000_000
    api.breaker = [[now - 7200 + 60 * i, "1"] for i in range(10)] + [[now - 600, "1"]]   # two open intervals
    code, body = post(receiver, {"alerts": [alert(scaledobject="cpu-burner")]})
    assert code == 200 and len(body["started"]) == 1
    job = api.created
    assert job["kind"] == "Job" and job["metadata"]["labels"]["trigger"] == "alert"
    assert job["metadata"]["labels"]["scaledobject"] == "cpu-burner"
    assert job["metadata"]["annotations"]["predictive-stack/alert-fingerprint"] == "abc123"
    c = job["spec"]["template"]["spec"]["containers"][0]
    excludes = [a for a in c["args"] if a.startswith("--exclude=")]
    assert c["args"][0] == "--candidates=hgb_delta" and len(excludes) == 2
    a, b = (int(x) for x in excludes[0].split("=")[1].split("/"))
    assert a == now - 7200 - 1800 and b == now - 7200 + 540 + 1800            # padded by 30 min
    env = {e["name"]: e["value"] for e in c["env"]}
    assert env["TRAIN_REASON"] == "ForecastBreakerOpenLong" and env["TRAIN_MIN_IMPROVEMENT"] == "0"
    assert env["TRAIN_DAYS"] == "7"
    assert api.last_query["query"] == ['max(predictive_scaler_breaker_open{scaledobject="cpu-burner"}) == 1']
    assert float(api.last_query["end"][0]) - float(api.last_query["start"][0]) == pytest.approx(7 * 86400)
    assert job["spec"]["template"]["metadata"]["labels"]["trigger"] == "alert"


def test_second_alert_is_deduplicated_while_a_job_is_active_or_recent(api, receiver):
    assert post(receiver, {"alerts": [alert()]})[1]["started"]
    assert post(receiver, {"alerts": [alert()]})[1]["started"] == []          # active
    api.jobs = [{"metadata": {"name": "old"}, "status": {
        "completionTime": (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")}}]
    assert post(receiver, {"alerts": [alert()]})[1]["started"] == []          # finished 2 h ago < 6 h
    api.jobs = [{"metadata": {"name": "old"}, "status": {
        "completionTime": (datetime.now(timezone.utc) - timedelta(hours=7)).strftime("%Y-%m-%dT%H:%M:%SZ")}}]
    assert len(post(receiver, {"alerts": [alert()]})[1]["started"]) == 1       # gap elapsed


def test_only_configured_firing_alerts_count_and_auth_is_required(api, receiver):
    assert post(receiver, {"alerts": [alert(status="resolved")]})[1]["started"] == []
    assert post(receiver, {"alerts": [alert(name="ForecastServiceDown")]})[1]["started"] == []
    assert post(receiver, {"alerts": [alert()]}, token="wrong")[0] == 401
    assert post(receiver, {"alerts": [alert()]}, token=None)[0] == 401
    assert api.jobs == []
    with urllib.request.urlopen(f"{receiver}/healthz", timeout=5) as resp:
        assert json.loads(resp.read()) == {"status": "ok"}


def test_prometheus_down_still_trains_without_exclusions(api, cfg):
    kube = t.KubeClient(base_url=api.url, token="sa-token")
    cfg.prometheus_url = "http://127.0.0.1:1"
    started = t.handle_alerts({"alerts": [alert(name="ForecastDriftHigh")]}, cfg, kube)
    assert len(started) == 1
    c = api.created["spec"]["template"]["spec"]["containers"][0]
    assert not any(a.startswith("--exclude") for a in c["args"])
    assert api.created["metadata"]["labels"]["scaledobject"] == ""


def test_breaker_windows_merge_adjacent_samples(api):
    api.breaker = [[100, "1"], [160, "1"], [220, "1"], [1000, "1"]]
    assert t.breaker_windows(api.url, "", days=1, pad=10, step=60, now=2000) == ["90/230", "990/1010"]
    api.breaker = []
    assert t.breaker_windows(api.url, "x", days=1, now=2000) == []
    assert t.recent_or_active([], timedelta(hours=1)) is None
