"""
Alert-driven retraining: an Alertmanager webhook receiver that turns a firing
alert into a Kubernetes Job cloned from the retraining CronJob.

    python -m forecast_training.trigger          # in the training image

Why a receiver instead of a plain CronJob trigger: it does three things the
alert cannot. It deduplicates (one retrain per cause, not one per
Alertmanager repeat), it rate-limits (a model that drifts twice an hour has
a data problem, not a training problem), and it excludes the anomaly that
caused the alert from the training window: the intervals in which the
scaler's circuit breaker was open (`predictive_scaler_breaker_open == 1`)
become `--exclude START/END` arguments, so a flash sale never teaches the
model to forecast flash sales.

Standard library only (the training image stays small): http.server for
the webhook, urllib against the Kubernetes API with the pod's ServiceAccount
token and against Prometheus. RBAC needed: get cronjobs; list, create jobs.

Environment:
  NAMESPACE          the release namespace (default: the pod's)
  CRONJOB            CronJob to clone (default forecast-retrain)
  TRIGGER_TOKEN      bearer token Alertmanager must send (required)
  PROMETHEUS_URL     where the breaker metric lives (default http://prometheus:9090)
  MIN_GAP_HOURS      no second alert-triggered Job within this gap (default 6)
  ALERT_NAMES        comma-separated alert names acted on
                     (default ForecastDriftHigh,ForecastBreakerOpenLong)
  EXCLUDE_PAD_SECONDS pad around each breaker interval (default 1800)
  PORT               listen port (default 8080)
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional, Sequence

log = logging.getLogger("retrain-trigger")

SA_DIR = "/var/run/secrets/kubernetes.io/serviceaccount"


# ── Kubernetes API (in-cluster, stdlib) ────────────────────────────────────────

class KubeClient:
    def __init__(self, base_url: Optional[str] = None, token: Optional[str] = None,
                 ca_file: Optional[str] = None, timeout: float = 15.0):
        host, port = os.getenv("KUBERNETES_SERVICE_HOST"), os.getenv("KUBERNETES_SERVICE_PORT", "443")
        self.base_url = base_url or (f"https://{host}:{port}" if host else "https://kubernetes.default.svc")
        self.token = token if token is not None else _read(os.path.join(SA_DIR, "token"))
        self.timeout = timeout
        self._ctx = None
        if self.base_url.startswith("https"):
            ca = ca_file or os.path.join(SA_DIR, "ca.crt")
            self._ctx = ssl.create_default_context(cafile=ca) if os.path.exists(ca) else ssl.create_default_context()

    def request(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base_url + path, data=data, method=method,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Content-Type": "application/json", "Accept": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout, context=self._ctx) as resp:  # noqa: S310
                return json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise RuntimeError(f"{method} {path} -> HTTP {exc.code}: {detail}") from exc

    def get_cronjob(self, ns: str, name: str) -> dict:
        return self.request("GET", f"/apis/batch/v1/namespaces/{ns}/cronjobs/{name}")

    def list_jobs(self, ns: str, selector: str) -> List[dict]:
        q = urllib.parse.urlencode({"labelSelector": selector})
        return self.request("GET", f"/apis/batch/v1/namespaces/{ns}/jobs?{q}").get("items", [])

    def create_job(self, ns: str, job: dict) -> dict:
        return self.request("POST", f"/apis/batch/v1/namespaces/{ns}/jobs", job)


def _read(path: str) -> str:
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


# ── the decision ───────────────────────────────────────────────────────────────

@dataclass
class TriggerConfig:
    namespace: str
    cronjob: str = "forecast-retrain"
    token: str = ""
    prometheus_url: str = "http://prometheus:9090"
    min_gap: timedelta = timedelta(hours=6)
    alert_names: Sequence[str] = ("ForecastDriftHigh", "ForecastBreakerOpenLong")
    exclude_pad: int = 1800
    labels: Dict[str, str] = field(default_factory=lambda: {"app": "forecast-retrain", "trigger": "alert"})

    @classmethod
    def from_env(cls) -> "TriggerConfig":
        ns = os.getenv("NAMESPACE") or _read(os.path.join(SA_DIR, "namespace")) or "default"
        names = [n.strip() for n in os.getenv("ALERT_NAMES", "ForecastDriftHigh,ForecastBreakerOpenLong").split(",")
                 if n.strip()]
        return cls(namespace=ns, cronjob=os.getenv("CRONJOB", "forecast-retrain"),
                   token=os.getenv("TRIGGER_TOKEN", "") or _read(os.getenv("TRIGGER_TOKEN_FILE", "")),
                   prometheus_url=os.getenv("PROMETHEUS_URL", "http://prometheus:9090").rstrip("/"),
                   min_gap=timedelta(hours=float(os.getenv("MIN_GAP_HOURS", "6"))),
                   alert_names=tuple(names), exclude_pad=int(os.getenv("EXCLUDE_PAD_SECONDS", "1800")))


def recent_or_active(jobs: Sequence[dict], min_gap: timedelta, now: Optional[datetime] = None) -> Optional[str]:
    """Name of an alert-triggered Job still running or finished within min_gap, else None."""
    now = now or datetime.now(timezone.utc)
    for j in jobs:
        status = j.get("status") or {}
        name = (j.get("metadata") or {}).get("name", "?")
        if status.get("active"):
            return name
        done = status.get("completionTime") or _last_condition_time(status)
        if done and now - _parse_time(done) < min_gap:
            return name
    return None


def _last_condition_time(status: dict) -> Optional[str]:
    times = [c.get("lastTransitionTime") for c in status.get("conditions") or [] if c.get("status") == "True"]
    return max(times) if times else None


def _parse_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def breaker_windows(prometheus_url: str, scaledobject: str, days: float, pad: int = 1800,
                    step: int = 60, now: Optional[float] = None, timeout: float = 30.0) -> List[str]:
    """
    START/END ranges (unix seconds) in which the scaler's circuit breaker was
    open for `scaledobject` (any ScaledObject when empty) over the last
    `days`, each padded by `pad` seconds; the training CLI's --exclude format.
    """
    end = now if now is not None else time.time()
    selector = f'{{scaledobject="{scaledobject}"}}' if scaledobject else ""
    query = f"max(predictive_scaler_breaker_open{selector}) == 1"
    url = f"{prometheus_url}/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": query, "start": f"{end - days * 86400:.0f}", "end": f"{end:.0f}", "step": str(step)})
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310
        body = json.load(resp)
    if body.get("status") != "success":
        raise RuntimeError(f"Prometheus error: {body.get('error', 'unknown')}")
    stamps: List[float] = []
    for series in body.get("data", {}).get("result", []):
        stamps.extend(float(ts) for ts, _ in series.get("values", []))
    ranges: List[List[float]] = []
    for ts in sorted(set(stamps)):
        if ranges and ts - ranges[-1][1] <= 1.5 * step:
            ranges[-1][1] = ts
        else:
            ranges.append([ts, ts])
    return [f"{int(a) - pad}/{int(b) + pad}" for a, b in ranges]


def job_from_cronjob(cronjob: dict, alert: dict, excludes: Sequence[str], cfg: TriggerConfig) -> dict:
    """A Job manifest cloned from the CronJob's template, with the alert's
    reason, the exclusions and no improvement bar (the champion is known to
    be wrong)."""
    labels = alert.get("labels") or {}
    alertname = labels.get("alertname", "alert")
    spec = json.loads(json.dumps(cronjob["spec"]["jobTemplate"]["spec"]))   # deep copy
    pod = spec["template"]["spec"]
    container = pod["containers"][0]
    container["args"] = list(container.get("args") or []) + [f"--exclude={r}" for r in excludes]
    env = [e for e in container.get("env") or [] if e.get("name") not in ("TRAIN_REASON", "TRAIN_MIN_IMPROVEMENT")]
    env += [{"name": "TRAIN_REASON", "value": alertname},
            {"name": "TRAIN_MIN_IMPROVEMENT", "value": "0"}]
    container["env"] = env
    pod_labels = (spec["template"].setdefault("metadata", {}).setdefault("labels", {}))
    pod_labels.update(cfg.labels)
    name = f"retrain-{alertname.lower()[:20]}-{int(time.time())}"
    return {
        "apiVersion": "batch/v1", "kind": "Job",
        "metadata": {
            "name": name,
            "labels": {**cfg.labels, "scaledobject": _label_safe(labels.get("scaledobject", "")),
                       "alertname": alertname},
            "annotations": {"predictive-stack/alert-fingerprint": alert.get("fingerprint", ""),
                            "predictive-stack/excluded": ",".join(excludes)},
        },
        "spec": spec,
    }


def _label_safe(value: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in value)[:63]


def handle_alerts(payload: dict, cfg: TriggerConfig, kube: KubeClient,
                  windows=breaker_windows) -> List[str]:
    """Alertmanager v4 webhook payload → names of the Jobs started."""
    started: List[str] = []
    firing = [a for a in payload.get("alerts", []) if a.get("status") == "firing"
              and (a.get("labels") or {}).get("alertname") in cfg.alert_names]
    if not firing:
        return started
    selector = ",".join(f"{k}={v}" for k, v in cfg.labels.items())
    busy = recent_or_active(kube.list_jobs(cfg.namespace, selector), cfg.min_gap)
    if busy:
        log.info("retrain not started: %s is active or finished less than %s ago", busy, cfg.min_gap)
        return started
    alert = firing[0]                                   # one Job per webhook: dedup across alerts too
    scaledobject = (alert.get("labels") or {}).get("scaledobject", "")
    cronjob = kube.get_cronjob(cfg.namespace, cfg.cronjob)
    days = _train_days(cronjob)
    try:
        excludes = windows(cfg.prometheus_url, scaledobject, days, cfg.exclude_pad)
    except Exception as exc:                           # no Prometheus: train anyway, say so
        log.warning("breaker windows unavailable (%s); training without exclusions", exc)
        excludes = []
    job = job_from_cronjob(cronjob, alert, excludes, cfg)
    created = kube.create_job(cfg.namespace, job)
    name = (created.get("metadata") or {}).get("name") or job["metadata"]["name"]
    log.info("started %s for %s (%s), excluding %d interval(s)", name,
             alert["labels"].get("alertname"), scaledobject or "all workloads", len(excludes))
    started.append(name)
    return started


def _train_days(cronjob: dict) -> float:
    for e in cronjob["spec"]["jobTemplate"]["spec"]["template"]["spec"]["containers"][0].get("env") or []:
        if e.get("name") == "TRAIN_DAYS" and e.get("value"):
            try:
                return float(e["value"])
            except ValueError:
                break
    return 14.0


# ── HTTP ───────────────────────────────────────────────────────────────────────

def make_handler(cfg: TriggerConfig, kube: KubeClient, windows=breaker_windows):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/healthz":
                self._reply(200, {"status": "ok"})
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/alertmanager":
                return self._reply(404, {"error": "not found"})
            auth = self.headers.get("Authorization", "")
            if not cfg.token or not hmac.compare_digest(auth, f"Bearer {cfg.token}"):
                return self._reply(401, {"error": "unauthorized"})
            length = int(self.headers.get("Content-Length") or 0)
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self._reply(400, {"error": "invalid JSON"})
            try:
                started = handle_alerts(payload, cfg, kube, windows)
            except Exception as exc:
                log.exception("retrain trigger failed")
                return self._reply(500, {"error": str(exc)})
            self._reply(200, {"started": started})

        def _reply(self, code: int, body: dict):
            raw = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, fmt, *args):
            log.debug(fmt, *args)

    return Handler


def serve(cfg: TriggerConfig, kube: KubeClient, port: int) -> None:
    server = ThreadingHTTPServer(("0.0.0.0", port), make_handler(cfg, kube))
    log.info("retrain trigger listening on :%d (namespace %s, cronjob %s, alerts %s)",
             port, cfg.namespace, cfg.cronjob, ",".join(cfg.alert_names))
    server.serve_forever()


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
                        datefmt="%Y-%m-%dT%H:%M:%SZ", stream=sys.stderr)
    cfg = TriggerConfig.from_env()
    if not cfg.token:
        log.error("TRIGGER_TOKEN (or TRIGGER_TOKEN_FILE) is required")
        return 1
    serve(cfg, KubeClient(), int(os.getenv("PORT", "8080")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
