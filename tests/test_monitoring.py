"""monitoring/render.py: dashboards and alert rules stay consistent with the code."""

import importlib.util
import os

import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
spec = importlib.util.spec_from_file_location("render", os.path.join(ROOT, "monitoring", "render.py"))
render = importlib.util.module_from_spec(spec)
spec.loader.exec_module(render)


def test_repo_dashboards_and_rules_validate():
    assert render.main(["--check"]) == 0


def test_exported_metric_names_are_picked_up_from_sources():
    names = render.exported_metrics()
    assert {"forecast_rolling_mae", "forecast_abs_error", "predictive_scaler_decisions_total",
            "forecast_model_info", "cpu_burner_request_duration_seconds"} <= names


@pytest.mark.parametrize("expr", [
    'rate(predictive_scaler_decisions_total{source="observed"}[5m])',
    'histogram_quantile(0.9, sum by (le, horizon_step) (rate(forecast_abs_error_bucket[1h])))',
    'count by (namespace, app) (up{namespace=~"$namespace", app!=""} == 1)',
    'label_values(predictive_scaler_reported_demand{namespace=~"$namespace"}, scaledobject)',
])
def test_known_expressions_pass(expr):
    render.check_expr(expr, render.exported_metrics() | render.LIBRARY_METRICS, "t")


@pytest.mark.parametrize("expr", [
    'sum(rate(http_requests_total{job="forecast-serving", status_code=~"5.."}[5m]))',
    'forecast_rolling_mae{step="1"}',
    "forecast_rolling_mae_typo",
    'rate(predictive_scaler_decision_total[5m])',
    "sum by (le) (rate(forecast_abs_errors_bucket[1h]))",
    "1 + 1",
])
def test_unknown_metric_names_fail(expr):
    with pytest.raises(render.Problem):
        render.check_expr(expr, render.exported_metrics() | render.LIBRARY_METRICS, "t")
