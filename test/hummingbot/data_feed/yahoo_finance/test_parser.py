import hashlib
from datetime import datetime, timezone
from decimal import Decimal

import pytest

from hummingbot.data_feed.yahoo_finance.parser import (
    YahooChartParseError,
    YahooChartParser,
)

from .conftest import (
    TARGET_SESSION_DATE,
    dump_payload,
    fixture_payload,
    fixture_text,
)


UTC = timezone.utc
RECEIVED_AT = datetime(2026, 7, 17, 20, 2, 0, 100000, tzinfo=UTC)
SOURCE_URL = "https://query2.finance.yahoo.com/v8/finance/chart/SNDK"


def parse_sndk(raw_text: str | None = None):
    raw_text = fixture_text("sndk_chart.json") if raw_text is None else raw_text
    return YahooChartParser().parse(
        raw_text=raw_text,
        expected_symbol="SNDK",
        target_session_date=TARGET_SESSION_DATE,
        received_at_utc=RECEIVED_AT,
        source_url=SOURCE_URL,
    )


def test_selects_only_target_day_unadjusted_regular_close_with_decimal_and_raw_hash():
    raw_text = fixture_text("sndk_chart.json")

    observation = parse_sndk(raw_text)

    assert observation.symbol == "SNDK"
    assert observation.target_session_date == TARGET_SESSION_DATE
    assert observation.close == Decimal("250.125")
    assert isinstance(observation.close, Decimal)
    assert observation.close != Decimal("750.125")  # adjclose
    assert observation.close != Decimal("999.875")  # regularMarketPrice / last trade
    assert observation.bar_timestamp_utc == datetime(2026, 7, 17, 13, 30, tzinfo=UTC)
    assert observation.regular_market_time_utc == datetime(2026, 7, 17, 19, 59, 58, tzinfo=UTC)
    assert observation.received_at_utc == RECEIVED_AT
    assert observation.raw_response_hash == hashlib.sha256(raw_text.encode("utf-8")).hexdigest()
    assert observation.source_url == SOURCE_URL


def test_low_liquidity_etf_early_regular_market_time_is_recorded_not_rejected():
    raw_text = fixture_text("snxx_chart.json")

    observation = YahooChartParser().parse(
        raw_text=raw_text,
        expected_symbol="SNXX",
        target_session_date=TARGET_SESSION_DATE,
        received_at_utc=datetime(2026, 7, 17, 20, 2, 0, 250000, tzinfo=UTC),
        source_url="https://query2.finance.yahoo.com/v8/finance/chart/SNXX",
    )

    assert observation.close == Decimal("30.0625")
    assert observation.regular_market_time_utc == datetime(2026, 7, 17, 19, 47, 11, tzinfo=UTC)


def test_stale_prior_day_response_is_rejected_instead_of_using_latest_available_bar():
    payload = fixture_payload("sndk_chart.json")
    result = payload["chart"]["result"][0]
    result["timestamp"].pop()
    for values in result["indicators"]["quote"][0].values():
        values.pop()
    result["indicators"]["adjclose"][0]["adjclose"].pop()

    with pytest.raises(YahooChartParseError, match="target session date"):
        parse_sndk(dump_payload(payload))


def test_duplicate_target_exchange_date_is_rejected_even_when_arrays_align():
    payload = fixture_payload("sndk_chart.json")
    result = payload["chart"]["result"][0]
    result["timestamp"].append(result["timestamp"][-1])
    for values in result["indicators"]["quote"][0].values():
        values.append(values[-1])
    result["indicators"]["adjclose"][0]["adjclose"].append(12345)

    with pytest.raises(YahooChartParseError, match="duplicate target session date"):
        parse_sndk(dump_payload(payload))


@pytest.mark.parametrize(
    ("array_path", "expected_pattern"),
    [
        (("quote", "close"), "array alignment"),
        (("quote", "volume"), "array alignment"),
        (("adjclose", "adjclose"), "array alignment"),
    ],
)
def test_misaligned_timestamp_quote_or_adjusted_arrays_fail_closed(array_path, expected_pattern):
    payload = fixture_payload("sndk_chart.json")
    indicators = payload["chart"]["result"][0]["indicators"]
    container, field = array_path
    indicators[container][0][field].pop()

    with pytest.raises(YahooChartParseError, match=expected_pattern):
        parse_sndk(dump_payload(payload))


@pytest.mark.parametrize("replacement", ["null", "NaN", "Infinity", "-Infinity", "0", "-1", '"250.125"', "true"])
def test_null_nonfinite_nonpositive_or_non_numeric_target_close_fails_closed(replacement):
    raw_text = fixture_text("sndk_chart.json").replace("250.125", replacement, 1)

    with pytest.raises(YahooChartParseError, match="close|JSON constant"):
        parse_sndk(raw_text)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("symbol", "SNXX"),
        ("symbol", "sndk"),
        ("currency", "EUR"),
        ("exchangeTimezoneName", "UTC"),
    ],
)
def test_symbol_currency_and_exchange_timezone_metadata_must_match_exactly(field, value):
    payload = fixture_payload("sndk_chart.json")
    payload["chart"]["result"][0]["meta"][field] = value

    with pytest.raises(YahooChartParseError, match=field):
        parse_sndk(dump_payload(payload))


@pytest.mark.parametrize(
    "raw_text",
    [
        "<html><body>Will be right back</body></html>",
        "{not-json",
        "",
    ],
)
def test_html_empty_and_malformed_json_are_rejected(raw_text):
    with pytest.raises(YahooChartParseError):
        parse_sndk(raw_text)


def test_chart_error_is_rejected_without_attempting_to_interpret_result():
    payload = fixture_payload("sndk_chart.json")
    payload["chart"]["error"] = {"code": "Not Found", "description": "synthetic error"}

    with pytest.raises(YahooChartParseError, match="chart.error"):
        parse_sndk(dump_payload(payload))


@pytest.mark.parametrize(
    "mutation",
    [
        lambda result: result.update({"timestamp": "1784295000"}),
        lambda result: result["indicators"].update({"quote": []}),
        lambda result: result["indicators"].update({"quote": [{"close": [250.125]}]}),
        lambda result: result.update({"events": []}),
        lambda result: result["events"]["dividends"]["1784208600"].update({"amount": "0.25"}),
    ],
)
def test_malformed_timestamp_indicator_and_corporate_action_shapes_fail_closed(mutation):
    payload = fixture_payload("sndk_chart.json")
    mutation(payload["chart"]["result"][0])

    with pytest.raises(YahooChartParseError):
        parse_sndk(dump_payload(payload))


def test_boolean_or_fractional_bar_timestamp_is_rejected():
    for invalid_timestamp in (True, 1784295000.5):
        payload = fixture_payload("sndk_chart.json")
        payload["chart"]["result"][0]["timestamp"][-1] = invalid_timestamp

        with pytest.raises(YahooChartParseError, match="timestamp"):
            parse_sndk(dump_payload(payload))


def test_null_daily_close_never_falls_back_to_adjusted_close_or_last_trade():
    payload = fixture_payload("sndk_chart.json")
    payload["chart"]["result"][0]["indicators"]["quote"][0]["close"][-1] = None

    with pytest.raises(YahooChartParseError, match="close"):
        parse_sndk(dump_payload(payload))
