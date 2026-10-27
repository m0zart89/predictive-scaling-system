import pytest

from fakes import BASE_META, make_config
from predictive_scaler.config import ConfigError, parse_metadata


def test_defaults():
    cfg = make_config()
    assert cfg.capacity == 1.0
    assert cfg.time_steps == 10
    assert cfg.step_seconds == 300
    assert cfg.mode == "max"
    assert cfg.activation_value == 0.0
    assert cfg.cache_seconds == 15.0
    assert cfg.metric_name == "predicted-demand"
    assert cfg.target_value == 0.5


def test_trailing_slashes_are_stripped():
    cfg = make_config(prometheusServerAddress="http://prom:9090/",
                      forecastServiceURL="http://fs:8080/")
    assert cfg.prometheus_address == "http://prom:9090"
    assert cfg.forecast_url == "http://fs:8080"


@pytest.mark.parametrize("missing", ["prometheusServerAddress", "query",
                                     "forecastServiceURL", "targetValue"])
def test_required_fields(missing):
    meta = {k: v for k, v in BASE_META.items() if k != missing}
    with pytest.raises(ConfigError, match=missing):
        parse_metadata(meta)


@pytest.mark.parametrize("key,value", [
    ("targetValue", "0"),
    ("targetValue", "-1"),
    ("targetValue", "abc"),
    ("targetValue", "nan"),
    ("capacity", "0"),
    ("timeSteps", "0"),
    ("timeSteps", "2.5"),
    ("stepSeconds", "-30"),
    ("activationValue", "-0.1"),
    ("mode", "aggressive"),
])
def test_invalid_values(key, value):
    with pytest.raises(ConfigError, match=key):
        make_config(**{key: value})


def test_zero_allowed_where_it_makes_sense():
    cfg = make_config(activationValue="0", cacheSeconds="0")
    assert cfg.activation_value == 0
    assert cfg.cache_seconds == 0


def test_mode_is_case_insensitive():
    assert make_config(mode="Forecast").mode == "forecast"


def test_cache_key_changes_with_settings():
    assert make_config().cache_key == make_config().cache_key
    assert make_config().cache_key != make_config(targetValue="0.6").cache_key
