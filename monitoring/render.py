#!/usr/bin/env python3
"""
Dashboards and alert rules as code: validate, and render the Grafana ConfigMap.

    python monitoring/render.py          # validate, then write the ConfigMap into grafana.yaml
    python monitoring/render.py --check  # validate; fail if grafana.yaml is out of date (CI)

Sources of truth:

* monitoring/dashboards/*.json      Grafana dashboards. Rendered into the
                                    `grafana-dashboards` ConfigMap between the
                                    GENERATED markers in monitoring/grafana.yaml,
                                    which the Grafana Deployment mounts.
* monitoring/prometheus-rules.yaml  Alerting rules. Every alert must link a
                                    runbook in docs/runbooks/<AlertName>.md.

Checks (standard library + PyYAML, no Grafana or promtool needed):

* each dashboard has a title, a unique uid, and panels with unique ids, a
  gridPos inside the 24-column grid, and at least one target with an `expr`;
* every PromQL expression references at least one metric the repo exports
  (names are collected from the Python sources) or a library/Kubernetes
  metric from the allowlist, and nothing that looks like a metric name but is
  neither — the cheap way to catch a typo before Grafana shows "No data";
* every alert has a runbook_url pointing to an existing file in docs/runbooks/.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DASHBOARD_DIR = os.path.join(ROOT, "monitoring", "dashboards")
GRAFANA_YAML = os.path.join(ROOT, "monitoring", "grafana.yaml")
RULES_YAML = os.path.join(ROOT, "monitoring", "prometheus-rules.yaml")
RUNBOOK_DIR = os.path.join(ROOT, "docs", "runbooks")
RUNBOOK_URL_PREFIX = "https://github.com/m0zart89/predictive-scaling-system/blob/main/docs/runbooks/"
CONFIGMAP_NAME = "grafana-dashboards"

BEGIN = "# ── BEGIN GENERATED (python monitoring/render.py) — edit monitoring/dashboards/*.json instead ──"
END = "# ── END GENERATED ──"

# Files whose Counter/Gauge/Histogram names the repo exports.
METRIC_SOURCES = [
    "serving/app.py", "serving/quality.py",
    "scaler/predictive_scaler/metrics.py", "demo/cpu-burner/app.py",
]
# Metrics from libraries and the platform that dashboards may use.
LIBRARY_METRICS = {
    "up", "scrape_duration_seconds",
    "http_requests_total", "http_request_duration_seconds",
    "http_request_size_bytes", "http_response_size_bytes",
    "process_cpu_seconds_total", "process_resident_memory_bytes",
    "process_virtual_memory_bytes", "process_open_fds", "process_start_time_seconds",
    "python_gc_objects_collected_total", "python_info",
    "container_cpu_usage_seconds_total", "container_memory_working_set_bytes",
    "kube_deployment_status_replicas", "kube_horizontalpodautoscaler_status_current_replicas",
    "kube_horizontalpodautoscaler_status_desired_replicas",
    "keda_scaler_metrics_value", "keda_scaler_errors_total",
}
HISTOGRAM_SUFFIXES = ("_bucket", "_count", "_sum", "_created")


class Problem(Exception):
    pass


# ── metric names ───────────────────────────────────────────────────────────────

def exported_metrics() -> set:
    """Metric names declared as Counter/Gauge/Histogram(...) in the sources."""
    names = set()
    pattern = re.compile(r'(?:Counter|Gauge|Histogram|Summary)\(\s*\n?\s*"([a-zA-Z_:][a-zA-Z0-9_:]*)"')
    for rel in METRIC_SOURCES:
        with open(os.path.join(ROOT, rel)) as f:
            names.update(pattern.findall(f.read()))
    return names


def metric_names_in(expr: str) -> list:
    """Identifiers in a PromQL expression that look like metric names."""
    out = []
    for m in re.finditer(r"(?<![\w$:])([a-zA-Z_:][a-zA-Z0-9_:]*)(?![\w:(])", expr):
        tok = m.group(1)
        # label matchers (`handler="..."`), bare `by (...)` groupings and
        # keywords: anything followed by `=`, `!`, `~` or inside parentheses
        # after by/without is a label, not a metric.
        rest = expr[m.end():m.end() + 2]
        if rest[:1] in ('"', "=", "!", "~") or rest[:2] in ("=~", "!~", "!="):
            continue
        out.append(tok)
    return out


_LABEL_GROUPS = re.compile(r"\b(?:by|without|on|ignoring|group_left|group_right)\s*\([^)]*\)")
_STRINGS = re.compile(r'"(?:\\.|[^"\\])*"')
_PROMQL_WORDS = {"by", "without", "on", "ignoring", "group_left", "group_right", "offset",
                 "bool", "and", "or", "unless", "le", "quantile", "sum", "min", "max", "avg",
                 "count", "rate", "irate", "increase", "histogram_quantile", "absent",
                 "label_values", "ceil", "floor", "round", "abs", "clamp_min", "clamp_max",
                 "delta", "deriv", "predict_linear", "topk", "bottomk", "stddev", "time",
                 "vector", "scalar", "changes", "resets", "s", "m", "h", "d", "w", "y",
                 "ms", "inf", "nan"}


def check_expr(expr: str, known: set, where: str) -> None:
    cleaned = _LABEL_GROUPS.sub(" ", _STRINGS.sub('""', expr))
    cleaned = re.sub(r"\$[A-Za-z_][A-Za-z0-9_]*", " ", cleaned)   # Grafana variables
    cleaned = re.sub(r"label_values\((.*),\s*[A-Za-z_]\w*\)", r"\1", cleaned)  # (metric, label)
    found = [t for t in metric_names_in(cleaned) if t not in _PROMQL_WORDS]
    if not found:
        raise Problem(f"{where}: no metric name found in {expr!r}")
    for tok in found:
        base = tok
        for suffix in HISTOGRAM_SUFFIXES:
            if tok.endswith(suffix):
                base = tok[: -len(suffix)]
                break
        if base not in known:
            raise Problem(f"{where}: unknown metric {tok!r} in {expr!r}")


# ── dashboards ─────────────────────────────────────────────────────────────────

def load_dashboards() -> dict:
    paths = sorted(glob.glob(os.path.join(DASHBOARD_DIR, "*.json")))
    if not paths:
        raise Problem(f"no dashboards in {DASHBOARD_DIR}")
    out = {}
    for path in paths:
        with open(path) as f:
            try:
                out[os.path.basename(path)] = json.load(f)
            except json.JSONDecodeError as exc:
                raise Problem(f"{path}: invalid JSON: {exc}") from exc
    return out


def check_dashboards(dashboards: dict, known: set) -> None:
    uids = {}
    for name, d in dashboards.items():
        for key in ("title", "uid", "panels"):
            if not d.get(key):
                raise Problem(f"{name}: missing {key!r}")
        if d["uid"] in uids:
            raise Problem(f"{name}: uid {d['uid']!r} already used by {uids[d['uid']]}")
        uids[d["uid"]] = name
        if "id" in d and d["id"] is not None:
            raise Problem(f"{name}: drop the numeric 'id'; provisioning assigns it")

        ids = set()
        for p in d["panels"]:
            where = f"{name} panel {p.get('id')} ({p.get('title')!r})"
            if p.get("id") in ids:
                raise Problem(f"{where}: duplicate panel id")
            ids.add(p.get("id"))
            if not p.get("title") or not p.get("type"):
                raise Problem(f"{where}: needs title and type")
            g = p.get("gridPos") or {}
            if not all(isinstance(g.get(k), int) for k in ("x", "y", "w", "h")):
                raise Problem(f"{where}: gridPos must have integer x, y, w, h")
            if g["x"] < 0 or g["w"] < 1 or g["x"] + g["w"] > 24 or g["h"] < 1:
                raise Problem(f"{where}: gridPos {g} leaves the 24-column grid")
            if p["type"] == "row":
                continue
            targets = p.get("targets") or []
            if not targets:
                raise Problem(f"{where}: no targets")
            for t in targets:
                if not t.get("expr"):
                    raise Problem(f"{where}: target without expr")
                check_expr(t["expr"], known, where)

        for v in (d.get("templating") or {}).get("list", []):
            q = v.get("query")
            q = q.get("query") if isinstance(q, dict) else q
            if v.get("type") == "query" and q:
                check_expr(q, known, f"{name} variable {v.get('name')!r}")


# ── alert rules and runbooks ───────────────────────────────────────────────────

def check_rules() -> list:
    import yaml

    with open(RULES_YAML) as f:
        cm = yaml.safe_load(f)
    alerts = []
    for fname, text in (cm.get("data") or {}).items():
        groups = yaml.safe_load(text).get("groups") or []
        for g in groups:
            for r in g.get("rules") or []:
                if "alert" not in r:
                    continue
                name = r["alert"]
                where = f"{fname}: alert {name}"
                if not r.get("expr"):
                    raise Problem(f"{where}: no expr")
                ann = r.get("annotations") or {}
                url = ann.get("runbook_url")
                if url != f"{RUNBOOK_URL_PREFIX}{name}.md":
                    raise Problem(f"{where}: runbook_url must be {RUNBOOK_URL_PREFIX}{name}.md")
                if not os.path.isfile(os.path.join(RUNBOOK_DIR, f"{name}.md")):
                    raise Problem(f"{where}: docs/runbooks/{name}.md does not exist")
                if not ann.get("summary"):
                    raise Problem(f"{where}: no summary annotation")
                if (r.get("labels") or {}).get("severity") not in ("info", "warning", "critical"):
                    raise Problem(f"{where}: severity label must be info|warning|critical")
                alerts.append(name)
    if not alerts:
        raise Problem(f"{RULES_YAML}: no alerts found")
    return alerts


# ── rendering ──────────────────────────────────────────────────────────────────

def render_configmap(dashboards: dict) -> str:
    lines = [BEGIN, "apiVersion: v1", "kind: ConfigMap", "metadata:",
             f"  name: {CONFIGMAP_NAME}", "data:"]
    for name, d in dashboards.items():
        lines.append(f"  {name}: |")
        lines.extend("    " + line for line in json.dumps(d, indent=2, sort_keys=True).splitlines())
    lines.append(END)
    return "\n".join(lines) + "\n"


def splice(text: str, block: str) -> str:
    start, end = text.find(BEGIN), text.find(END)
    if start < 0 or end < 0 or end < start:
        raise Problem(f"{GRAFANA_YAML}: GENERATED markers not found")
    end += len(END) + 1   # include the newline after the END marker
    return text[:start] + block + text[end:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="validate only; fail if grafana.yaml is stale")
    args = ap.parse_args(argv)

    try:
        known = exported_metrics() | LIBRARY_METRICS
        dashboards = load_dashboards()
        check_dashboards(dashboards, known)
        alerts = check_rules()

        with open(GRAFANA_YAML) as f:
            current = f.read()
        updated = splice(current, render_configmap(dashboards))
        if args.check:
            if updated != current:
                raise Problem(f"{os.path.relpath(GRAFANA_YAML, ROOT)} is out of date: "
                              "run python monitoring/render.py and commit")
        elif updated != current:
            with open(GRAFANA_YAML, "w") as f:
                f.write(updated)
            print(f"updated {os.path.relpath(GRAFANA_YAML, ROOT)}")
    except Problem as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"ok: {len(dashboards)} dashboards ({', '.join(dashboards)}), "
          f"{len(alerts)} alerts with runbooks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
