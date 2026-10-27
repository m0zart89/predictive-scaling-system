"""
Thin HTTP clients for Prometheus and the forecast service.

Standard library only (urllib): the scaler sits in the scaling path of other
workloads, so its dependency surface is kept as small as possible.
"""

from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import List, Optional


class PrometheusError(RuntimeError):
    """Prometheus is unreachable, rejected the query, or returned unusable data."""


class ForecastError(RuntimeError):
    """The forecast service is unreachable or returned an unusable answer."""

    def __init__(self, message: str, reason: str = "unavailable"):
        super().__init__(message)
        # Short label for metrics: unavailable | not_ready | bad_response
        self.reason = reason


def _read_json(req: urllib.request.Request, timeout: float) -> dict:
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL comes from operator config)
        return json.loads(resp.read().decode("utf-8"))


class PrometheusClient:
    def fetch_window(self, address: str, query: str, points: int, step_seconds: int,
                     timeout: float, now: Optional[float] = None) -> List[float]:
        """
        Return up to `points` most recent samples of `query`, oldest first.

        The query must resolve to exactly one series — aggregate it in PromQL
        (sum/avg) so the scaler never has to guess how to combine series.
        Non-finite samples (NaN from division by zero, etc.) are dropped.
        """
        end = time.time() if now is None else now
        # [end - points*step, end] yields points+1 samples, so one missing sample
        # at the edge (scrape gap) still leaves a full window.
        start = end - points * step_seconds
        params = urllib.parse.urlencode({
            "query": query,
            "start": f"{start:.3f}",
            "end": f"{end:.3f}",
            "step": str(step_seconds),
        })
        url = f"{address}/api/v1/query_range?{params}"
        try:
            body = _read_json(urllib.request.Request(url), timeout)
        except urllib.error.HTTPError as exc:
            raise PrometheusError(f"Prometheus returned HTTP {exc.code} for query_range") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PrometheusError(f"Prometheus unreachable at {address}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise PrometheusError("Prometheus returned invalid JSON") from exc

        if body.get("status") != "success":
            raise PrometheusError(f"Prometheus query failed: {body.get('error', 'unknown error')}")

        result = body.get("data", {}).get("result", [])
        if len(result) > 1:
            raise PrometheusError(
                f"query returned {len(result)} series; aggregate it to a single series "
                "(e.g. wrap it in sum(...))"
            )
        if not result:
            return []

        values: List[float] = []
        for _, raw in result[0].get("values", []):
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                values.append(v)
        return values[-points:]


class ForecastClient:
    def predict(self, base_url: str, values: List[float], timeout: float,
                series: Optional[str] = None, timestamp: Optional[float] = None,
                step_seconds: Optional[int] = None) -> List[float]:
        """
        Call POST /predict and return the forecast for each step of the model's
        horizon (model space). Models with a single-step horizon, and services
        that predate multi-step forecasts, yield a one-element list.

        `series`, `timestamp` (unix time of the last sample) and `step_seconds`
        (their spacing) let the service score this forecast against the actual
        values a later window carries; older services ignore them.
        """
        body_out: dict = {"values": values}
        if series:
            body_out["series"] = series
        if timestamp is not None:
            body_out["timestamp"] = timestamp
        if step_seconds is not None:
            body_out["step_seconds"] = step_seconds
        payload = json.dumps(body_out).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            body = _read_json(req, timeout)
        except urllib.error.HTTPError as exc:
            reason = "not_ready" if exc.code == 503 else "bad_response"
            raise ForecastError(f"forecast service returned HTTP {exc.code}", reason) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise ForecastError(f"forecast service unreachable at {base_url}: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise ForecastError("forecast service returned invalid JSON", "bad_response") from exc

        try:
            raw = body.get("predictions") or [body["prediction"]]
            predictions = [float(v) for v in raw]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ForecastError("forecast response has no numeric 'prediction'", "bad_response") from exc
        if not predictions or not all(math.isfinite(v) for v in predictions):
            raise ForecastError("forecast service returned a non-finite prediction", "bad_response")
        return predictions
