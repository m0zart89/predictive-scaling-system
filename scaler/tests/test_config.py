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


def test_forecast_aggregation():
    assert make_config().forecast_aggregation == "next"
    assert make_config(forecastAggregation="MAX").forecast_aggregation == "max"
    with pytest.raises(ConfigError, match="forecastAggregation"):
        make_config(forecastAggregation="mean")


def test_history_steps_defaults_to_time_steps_and_must_not_be_shorter():
    assert make_config().history_steps == 10
    assert make_config(timeSteps="6").history_steps == 6
    cfg = make_config(timeSteps="10", historySteps="298")     # a day of 5-minute samples more
    assert cfg.history_steps == 298 and cfg.time_steps == 10
    assert cfg.cache_key != make_config().cache_key           # part of the settings
    with pytest.raises(ConfigError, match="historySteps"):
        make_config(timeSteps="10", historySteps="9")
    with pytest.raises(ConfigError, match="historySteps"):
        make_config(historySteps="0")


def test_channels_are_a_json_map_and_exclude_the_target():
    cfg = make_config(channels='{"rps": "sum(rate(http_requests_total[1m]))", "mem": "avg(mem)"}')
    assert cfg.channels == (("rps", "sum(rate(http_requests_total[1m]))"), ("mem", "avg(mem)"))
    assert cfg.target_channel == "cpu" and make_config().channels == ()
    assert make_config(targetChannel="demand").target_channel == "demand"
    for bad in ('not json', '[]', '{}', '{"rps": ""}', '{"rps": 3}'):
        with pytest.raises(ConfigError, match="channels"):
            make_config(channels=bad)
    with pytest.raises(ConfigError, match="target channel"):
        make_config(channels='{"cpu": "x"}')


def test_capacity_query_settings():
    from fakes import make_config
    from predictive_scaler.config import ConfigError
    import pytest

    cfg = make_config()
    assert cfg.capacity_query == "" and cfg.capacity_refresh_seconds == 300.0 and cfg.capacity_drift_factor == 2.0
    cfg = make_config(capacityQuery=" max(x) * 3 ", capacityRefreshSeconds="0", capacityDriftFactor="0")
    assert cfg.capacity_query == "max(x) * 3" and cfg.capacity_refresh_seconds == 0 and cfg.capacity_drift_factor == 0
    with pytest.raises(ConfigError, match="capacityDriftFactor"):
        make_config(capacityDriftFactor="0.5")
