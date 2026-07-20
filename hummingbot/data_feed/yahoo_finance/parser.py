import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
from typing import Any
from zoneinfo import ZoneInfo

from hummingbot.strategy_v2.leveraged_etf_arbitrage.decimal_policy import validate_yahoo_decimal


UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_EVENT_KEY_PATTERN = re.compile(r"^[1-9][0-9]*$")
_SPLIT_RATIO_PATTERN = re.compile(
    r"^(?P<numerator>[1-9][0-9]*(?:\.[0-9]*[1-9])?):"
    r"(?P<denominator>[1-9][0-9]*(?:\.[0-9]*[1-9])?)$"
)


class YahooChartParseError(ValueError):
    """Raised when a Yahoo chart response cannot be trusted as a regular close."""


def _is_utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)


def _require_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or not _is_utc(value):
        raise ValueError(f"{field_name} must be an aware UTC datetime")
    return value


def _require_positive_decimal(value: object, field_name: str) -> Decimal:
    return validate_yahoo_decimal(value, field_name, positive=True)


@dataclass(frozen=True, slots=True)
class YahooCloseObservation:
    symbol: str
    target_session_date: date
    close: Decimal
    bar_timestamp_utc: datetime
    regular_market_time_utc: datetime
    received_at_utc: datetime
    raw_response_hash: str
    source_url: str

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(self.target_session_date, date) or isinstance(self.target_session_date, datetime):
            raise TypeError("target_session_date must be a date")
        _require_positive_decimal(self.close, "close")
        _require_utc(self.bar_timestamp_utc, "bar_timestamp_utc")
        _require_utc(self.regular_market_time_utc, "regular_market_time_utc")
        _require_utc(self.received_at_utc, "received_at_utc")
        if not isinstance(self.raw_response_hash, str) or _SHA256_PATTERN.fullmatch(self.raw_response_hash) is None:
            raise ValueError("raw_response_hash must be lowercase SHA-256 hexadecimal")
        if not isinstance(self.source_url, str) or not self.source_url:
            raise ValueError("source_url must be a non-empty string")


class YahooChartParser:
    """Strict parser for Yahoo's unadjusted daily chart close."""

    def parse(
        self,
        raw_text: str,
        expected_symbol: str,
        target_session_date: date,
        received_at_utc: datetime,
        source_url: str,
    ) -> YahooCloseObservation:
        if not isinstance(raw_text, str) or not raw_text.strip():
            raise YahooChartParseError("Yahoo chart response is empty")
        if not isinstance(expected_symbol, str) or not expected_symbol:
            raise YahooChartParseError("expected symbol must be a non-empty string")
        if not isinstance(target_session_date, date) or isinstance(target_session_date, datetime):
            raise YahooChartParseError("target session date must be a date")
        try:
            _require_utc(received_at_utc, "received_at_utc")
        except (TypeError, ValueError) as exception:
            raise YahooChartParseError(str(exception)) from exception
        if raw_text.lstrip().startswith("<") or "Will be right back" in raw_text:
            raise YahooChartParseError("Yahoo returned an HTML error page")

        def reject_constant(value: str) -> None:
            raise YahooChartParseError(f"invalid JSON constant {value}")

        try:
            payload = json.loads(raw_text, parse_float=Decimal, parse_constant=reject_constant)
        except YahooChartParseError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError) as exception:
            raise YahooChartParseError("Yahoo chart response is malformed JSON") from exception

        result = self._chart_result(payload)
        meta = self._mapping(result.get("meta"), "chart result meta")
        self._validate_meta(meta, expected_symbol)
        regular_market_time = self._unix_timestamp(meta.get("regularMarketTime"), "regularMarketTime")

        timestamps = result.get("timestamp")
        if not isinstance(timestamps, list) or not timestamps:
            raise YahooChartParseError("timestamp must be a non-empty array")
        bar_timestamps = [self._unix_timestamp(value, "timestamp") for value in timestamps]

        indicators = self._mapping(result.get("indicators"), "indicators")
        quote_groups = indicators.get("quote")
        if not isinstance(quote_groups, list) or len(quote_groups) != 1:
            raise YahooChartParseError("indicators.quote must contain exactly one object")
        quote = self._mapping(quote_groups[0], "indicators.quote[0]")
        quote_arrays = self._aligned_quote_arrays(quote, len(timestamps))
        self._validate_adjusted_alignment(indicators, len(timestamps))
        self._validate_events(result.get("events"))

        exchange_dates = [timestamp.astimezone(NEW_YORK).date() for timestamp in bar_timestamps]
        target_indices = [
            index for index, session_date in enumerate(exchange_dates) if session_date == target_session_date
        ]
        if not target_indices:
            raise YahooChartParseError("target session date is absent from Yahoo daily bars")
        if len(target_indices) != 1:
            raise YahooChartParseError("duplicate target session date in Yahoo daily bars")
        if len(exchange_dates) != len(set(exchange_dates)):
            raise YahooChartParseError("duplicate exchange date in Yahoo daily bars")

        target_index = target_indices[0]
        close = self._numeric_decimal(
            quote_arrays["close"][target_index],
            "target unadjusted close",
            positive=True,
            nullable=False,
        )
        assert close is not None
        return YahooCloseObservation(
            symbol=expected_symbol,
            target_session_date=target_session_date,
            close=close,
            bar_timestamp_utc=bar_timestamps[target_index],
            regular_market_time_utc=regular_market_time,
            received_at_utc=received_at_utc,
            raw_response_hash=hashlib.sha256(raw_text.encode("utf-8")).hexdigest(),
            source_url=source_url,
        )

    def _chart_result(self, payload: object) -> Mapping[str, Any]:
        root = self._mapping(payload, "response root")
        chart = self._mapping(root.get("chart"), "chart")
        if "error" not in chart or chart.get("error") is not None:
            raise YahooChartParseError("chart.error is not null")
        results = chart.get("result")
        if not isinstance(results, list) or len(results) != 1:
            raise YahooChartParseError("chart.result must contain exactly one result")
        return self._mapping(results[0], "chart.result[0]")

    def _validate_meta(self, meta: Mapping[str, Any], expected_symbol: str) -> None:
        expected_values = {
            "symbol": expected_symbol,
            "currency": "USD",
            "exchangeTimezoneName": "America/New_York",
        }
        for field_name, expected_value in expected_values.items():
            if meta.get(field_name) != expected_value:
                raise YahooChartParseError(f"meta.{field_name} does not match {expected_value}")

    def _aligned_quote_arrays(self, quote: Mapping[str, Any], expected_length: int) -> dict[str, list[Any]]:
        arrays: dict[str, list[Any]] = {}
        for field_name in ("open", "high", "low", "close", "volume"):
            values = quote.get(field_name)
            if not isinstance(values, list) or len(values) != expected_length:
                raise YahooChartParseError(f"timestamp/quote array alignment failed for {field_name}")
            arrays[field_name] = values
            for value in values:
                if field_name == "volume":
                    self._volume(value)
                else:
                    self._numeric_decimal(value, field_name, positive=True, nullable=True)
        return arrays

    def _validate_adjusted_alignment(self, indicators: Mapping[str, Any], expected_length: int) -> None:
        if "adjclose" not in indicators:
            return
        groups = indicators["adjclose"]
        if not isinstance(groups, list) or len(groups) != 1:
            raise YahooChartParseError("indicators.adjclose must contain exactly one object")
        adjusted = self._mapping(groups[0], "indicators.adjclose[0]").get("adjclose")
        if not isinstance(adjusted, list) or len(adjusted) != expected_length:
            raise YahooChartParseError("timestamp/adjusted-close array alignment failed")
        for value in adjusted:
            self._numeric_decimal(value, "adjusted close", positive=True, nullable=True)

    def _validate_events(self, events: object) -> None:
        if events is None:
            return
        event_groups = self._mapping(events, "events")
        allowed_groups = {"dividends", "splits", "capitalGains"}
        for group_name, raw_events in event_groups.items():
            if group_name not in allowed_groups:
                raise YahooChartParseError(f"unsupported corporate action group {group_name}")
            group = self._mapping(raw_events, f"events.{group_name}")
            for event_key, raw_event in group.items():
                if not isinstance(event_key, str) or _EVENT_KEY_PATTERN.fullmatch(event_key) is None:
                    raise YahooChartParseError(
                        f"events.{group_name} event key must be a positive integer Unix timestamp"
                    )
                try:
                    key_value = int(event_key)
                except ValueError as exception:
                    raise YahooChartParseError(
                        f"events.{group_name} event key must be a positive integer Unix timestamp"
                    ) from exception
                key_timestamp = self._unix_timestamp(key_value, f"events.{group_name}.{event_key} key")
                event = self._mapping(raw_event, f"events.{group_name}.{event_key}")
                event_timestamp = self._unix_timestamp(
                    event.get("date"),
                    f"events.{group_name}.{event_key}.date",
                )
                if event_timestamp != key_timestamp:
                    raise YahooChartParseError(
                        f"events.{group_name}.{event_key} event key does not match date"
                    )
                if group_name in {"dividends", "capitalGains"}:
                    self._numeric_decimal(
                        event.get("amount"),
                        f"events.{group_name}.{event_key}.amount",
                        positive=False,
                        nullable=False,
                    )
                else:
                    numerator = self._numeric_decimal(
                        event.get("numerator"),
                        f"events.{group_name}.{event_key}.numerator",
                        positive=True,
                        nullable=False,
                    )
                    denominator = self._numeric_decimal(
                        event.get("denominator"),
                        f"events.{group_name}.{event_key}.denominator",
                        positive=True,
                        nullable=False,
                    )
                    split_ratio = event.get("splitRatio")
                    if not isinstance(split_ratio, str):
                        raise YahooChartParseError(
                            f"events.{group_name}.{event_key}.splitRatio must use numerator:denominator syntax"
                        )
                    ratio_match = _SPLIT_RATIO_PATTERN.fullmatch(split_ratio)
                    if ratio_match is None:
                        raise YahooChartParseError(
                            f"events.{group_name}.{event_key}.splitRatio must use numerator:denominator syntax"
                        )
                    ratio_numerator = self._numeric_decimal(
                        Decimal(ratio_match.group("numerator")),
                        f"events.{group_name}.{event_key}.splitRatio numerator",
                        positive=True,
                        nullable=False,
                    )
                    ratio_denominator = self._numeric_decimal(
                        Decimal(ratio_match.group("denominator")),
                        f"events.{group_name}.{event_key}.splitRatio denominator",
                        positive=True,
                        nullable=False,
                    )
                    if ratio_numerator != numerator or ratio_denominator != denominator:
                        raise YahooChartParseError(
                            f"events.{group_name}.{event_key}.splitRatio is inconsistent with split fields"
                        )

    @staticmethod
    def _mapping(value: object, field_name: str) -> Mapping[str, Any]:
        if not isinstance(value, Mapping):
            raise YahooChartParseError(f"{field_name} must be an object")
        return value

    @staticmethod
    def _unix_timestamp(value: object, field_name: str) -> datetime:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise YahooChartParseError(f"{field_name} must be a positive integer Unix timestamp")
        try:
            return datetime.fromtimestamp(value, tz=UTC)
        except (OverflowError, OSError, ValueError) as exception:
            raise YahooChartParseError(f"{field_name} is outside the supported timestamp range") from exception

    @staticmethod
    def _numeric_decimal(
        value: object,
        field_name: str,
        *,
        positive: bool,
        nullable: bool,
    ) -> Decimal | None:
        if value is None and nullable:
            return None
        if isinstance(value, bool) or not isinstance(value, (Decimal, int)):
            raise YahooChartParseError(f"{field_name} must be a JSON number")
        parsed = value if isinstance(value, Decimal) else Decimal(value)
        try:
            return validate_yahoo_decimal(
                parsed,
                field_name,
                positive=positive,
                nonnegative=not positive,
            )
        except (TypeError, ValueError) as exception:
            raise YahooChartParseError(str(exception)) from exception

    @staticmethod
    def _volume(value: object) -> int | None:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise YahooChartParseError("volume must be a nonnegative integer or null")
        return value
