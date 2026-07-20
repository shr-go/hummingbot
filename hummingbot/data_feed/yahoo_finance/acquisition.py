import asyncio
import hashlib
import inspect
import json
import random
import re
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Optional

from hummingbot.data_feed.yahoo_finance.parser import YahooCloseObservation
from hummingbot.data_feed.yahoo_finance.provider import (
    YahooChartProvider,
    YahooCycleBudget,
    YahooDeadlineExceeded,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.calendar import (
    CalendarRangeError,
    XnysNavCalendar,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import NavConfig
from hummingbot.strategy_v2.leveraged_etf_arbitrage.decimal_policy import (
    MAX_DECIMAL_FIXED_LENGTH,
    YAHOO_MAX_DECIMAL_FIXED_LENGTH,
    validate_bounded_decimal,
    validate_yahoo_decimal,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.math import calculate_hedge_ratio


UTC = timezone.utc
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CYCLE_PATTERN = re.compile(r"^xnys-[0-9]{4}-[0-9]{2}-[0-9]{2}$")
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.^=-]+$")
_DATE_PATTERN = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])$")
_UTC_PATTERN = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.[0-9]{6}Z$"
)
_CHECKPOINT_FIELDS = {
    "schema_version",
    "cycle_id",
    "target_session_date",
    "official_close_utc",
    "deadline_utc",
    "attempt",
    "next_poll_utc",
    "confirmation_count",
    "candidate_stock_close",
    "candidate_etf_close",
    "candidate_stock_raw_response_hash",
    "candidate_etf_raw_response_hash",
    "stock_received_at_utc",
    "etf_received_at_utc",
    "revision",
}
_CONFIRMATION_EVIDENCE_FIELDS = {
    "confirmation_index",
    "stock_symbol",
    "etf_symbol",
    "stock_close",
    "etf_close",
    "stock_bar_timestamp_utc",
    "etf_bar_timestamp_utc",
    "stock_regular_market_time_utc",
    "etf_regular_market_time_utc",
    "stock_received_at_utc",
    "etf_received_at_utc",
    "stock_raw_response_hash",
    "etf_raw_response_hash",
    "stock_source_url",
    "etf_source_url",
}
_RECOVERY_FIELDS = _CHECKPOINT_FIELDS | {
    "integrity_version",
    "stock_symbol",
    "etf_symbol",
    "confirmation_evidence",
    "integrity_hash",
}
_CANDIDATE_EVIDENCE_FIELDS = {
    "evidence_version",
    "cycle_id",
    "pair_id",
    "target_session_date",
    "official_close_utc",
    "anchor_source",
    "stock_symbol",
    "etf_symbol",
    "stock_close",
    "etf_close",
    "stock_bar_timestamp_utc",
    "etf_bar_timestamp_utc",
    "stock_regular_market_time_utc",
    "etf_regular_market_time_utc",
    "stock_received_at_utc",
    "etf_received_at_utc",
    "acquisition_started_at_utc",
    "finalized_at_utc",
    "deadline_utc",
    "stock_raw_response_hash",
    "etf_raw_response_hash",
    "etf_daily_multiplier",
    "hedge_ratio",
    "anchor_confirmation_count",
    "observed_confirmation_count",
    "anchor_confirmation_interval_seconds",
    "anchor_min_finalize_delay_seconds",
    "anchor_pair_fetch_max_skew_seconds",
    "confirmation_evidence",
    "evidence_hash",
}
_ANCHOR_EVIDENCE_VERSION = 2


class CheckpointIntegrityError(ValueError):
    """Raised when untrusted checkpoint or anchor evidence violates its contract."""


class AnchorAcquisitionStatus(str, Enum):
    WAITING = "WAITING"
    POLLING = "POLLING"
    FINALIZABLE = "FINALIZABLE"
    ANCHOR_UNAVAILABLE = "ANCHOR_UNAVAILABLE"


class FinalizedAnchorStatus(str, Enum):
    UNCHANGED = "UNCHANGED"
    DATA_REVISION = "DATA_REVISION"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


def _utc_now() -> datetime:
    return datetime.now(tz=UTC)


def _random_jitter(upper_bound: float) -> float:
    return random.uniform(0.0, upper_bound)


def _is_utc(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() == UTC.utcoffset(value)


def _validate_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime) or not _is_utc(value):
        raise ValueError(f"{field_name} must be an aware UTC datetime")
    return value


def _format_utc(value: datetime) -> str:
    _validate_utc(value, "UTC value")
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _parse_utc(value: object, field_name: str) -> datetime:
    if not isinstance(value, str) or _UTC_PATTERN.fullmatch(value) is None:
        raise CheckpointIntegrityError(f"{field_name} must be a canonical UTC instant")
    try:
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    except ValueError as exception:
        raise CheckpointIntegrityError(f"{field_name} must be a valid UTC instant") from exception


def _validate_positive_decimal(value: object, field_name: str) -> Decimal:
    return validate_yahoo_decimal(value, field_name, positive=True)


def _format_decimal(value: Decimal) -> str:
    _validate_positive_decimal(value, "decimal")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _format_financial_decimal(value: Decimal) -> str:
    validate_bounded_decimal(value, "financial decimal")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _parse_canonical_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, str):
        raise CheckpointIntegrityError(f"{field_name} must be a canonical decimal string")
    if len(value) > YAHOO_MAX_DECIMAL_FIXED_LENGTH:
        raise CheckpointIntegrityError(f"{field_name} exceeds the canonical decimal serialized length")
    try:
        parsed = Decimal(value)
    except InvalidOperation as exception:
        raise CheckpointIntegrityError(f"{field_name} must be a canonical decimal string") from exception
    try:
        canonical = _format_decimal(parsed)
    except (TypeError, ValueError) as exception:
        raise CheckpointIntegrityError(f"{field_name} must be a positive canonical decimal") from exception
    if canonical != value:
        raise CheckpointIntegrityError(f"{field_name} must be a canonical decimal string")
    return parsed


def _parse_canonical_financial_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, str) or len(value) > MAX_DECIMAL_FIXED_LENGTH:
        raise CheckpointIntegrityError(f"{field_name} must be a bounded canonical decimal string")
    try:
        parsed = Decimal(value)
        canonical = _format_financial_decimal(parsed)
    except (InvalidOperation, TypeError, ValueError) as exception:
        raise CheckpointIntegrityError(f"{field_name} must be a bounded canonical decimal string") from exception
    if canonical != value:
        raise CheckpointIntegrityError(f"{field_name} must be a canonical decimal string")
    return parsed


def _validate_hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")
    return value


def _validate_symbol(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SYMBOL_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} is not a supported Yahoo symbol")
    return value


def _validate_source_url(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 512 or not value.startswith("https://"):
        raise ValueError(f"{field_name} must be a bounded HTTPS URL")
    return value


def _parse_date(value: object, field_name: str) -> date:
    if not isinstance(value, str) or _DATE_PATTERN.fullmatch(value) is None:
        raise CheckpointIntegrityError(f"{field_name} must be a canonical date")
    try:
        return date.fromisoformat(value)
    except ValueError as exception:
        raise CheckpointIntegrityError(f"{field_name} must be a valid canonical date") from exception


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    canonical_payload = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


def anchor_evidence_hash(
    cycle_id: str,
    stock_raw_response_hash: str,
    etf_raw_response_hash: str,
) -> str:
    if not isinstance(cycle_id, str) or _CYCLE_PATTERN.fullmatch(cycle_id) is None:
        raise ValueError("cycle_id must be an XNYS cycle identifier")
    _validate_hash(stock_raw_response_hash, "stock raw response hash")
    _validate_hash(etf_raw_response_hash, "ETF raw response hash")
    canonical_payload = json.dumps(
        {
            "cycle_id": cycle_id,
            "etf_raw_response_hash": etf_raw_response_hash,
            "stock_raw_response_hash": stock_raw_response_hash,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AnchorConfirmationEvidence:
    confirmation_index: int
    stock_symbol: str
    etf_symbol: str
    stock_close: Decimal
    etf_close: Decimal
    stock_bar_timestamp_utc: datetime
    etf_bar_timestamp_utc: datetime
    stock_regular_market_time_utc: datetime
    etf_regular_market_time_utc: datetime
    stock_received_at_utc: datetime
    etf_received_at_utc: datetime
    stock_raw_response_hash: str
    etf_raw_response_hash: str
    stock_source_url: str
    etf_source_url: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.confirmation_index, bool)
            or not isinstance(self.confirmation_index, int)
            or self.confirmation_index < 1
        ):
            raise ValueError("confirmation evidence index must be a positive integer")
        _validate_symbol(self.stock_symbol, "confirmation stock symbol")
        _validate_symbol(self.etf_symbol, "confirmation ETF symbol")
        _validate_positive_decimal(self.stock_close, "confirmation stock close")
        _validate_positive_decimal(self.etf_close, "confirmation ETF close")
        for value, field_name in (
            (self.stock_bar_timestamp_utc, "confirmation stock bar timestamp"),
            (self.etf_bar_timestamp_utc, "confirmation ETF bar timestamp"),
            (self.stock_regular_market_time_utc, "confirmation stock regular market time"),
            (self.etf_regular_market_time_utc, "confirmation ETF regular market time"),
            (self.stock_received_at_utc, "confirmation stock received time"),
            (self.etf_received_at_utc, "confirmation ETF received time"),
        ):
            _validate_utc(value, field_name)
        _validate_hash(self.stock_raw_response_hash, "confirmation stock raw response hash")
        _validate_hash(self.etf_raw_response_hash, "confirmation ETF raw response hash")
        _validate_source_url(self.stock_source_url, "confirmation stock source URL")
        _validate_source_url(self.etf_source_url, "confirmation ETF source URL")

    @classmethod
    def from_observations(
        cls,
        confirmation_index: int,
        stock: YahooCloseObservation,
        etf: YahooCloseObservation,
    ) -> "AnchorConfirmationEvidence":
        return cls(
            confirmation_index=confirmation_index,
            stock_symbol=stock.symbol,
            etf_symbol=etf.symbol,
            stock_close=stock.close,
            etf_close=etf.close,
            stock_bar_timestamp_utc=stock.bar_timestamp_utc,
            etf_bar_timestamp_utc=etf.bar_timestamp_utc,
            stock_regular_market_time_utc=stock.regular_market_time_utc,
            etf_regular_market_time_utc=etf.regular_market_time_utc,
            stock_received_at_utc=stock.received_at_utc,
            etf_received_at_utc=etf.received_at_utc,
            stock_raw_response_hash=stock.raw_response_hash,
            etf_raw_response_hash=etf.raw_response_hash,
            stock_source_url=stock.source_url,
            etf_source_url=etf.source_url,
        )

    @classmethod
    def from_fields(cls, fields: Mapping[str, Any]) -> "AnchorConfirmationEvidence":
        if not isinstance(fields, Mapping) or set(fields) != _CONFIRMATION_EVIDENCE_FIELDS:
            raise CheckpointIntegrityError("confirmation evidence fields do not match version 2")
        try:
            return cls(
                confirmation_index=fields["confirmation_index"],
                stock_symbol=fields["stock_symbol"],
                etf_symbol=fields["etf_symbol"],
                stock_close=_parse_canonical_decimal(fields["stock_close"], "stock_close"),
                etf_close=_parse_canonical_decimal(fields["etf_close"], "etf_close"),
                stock_bar_timestamp_utc=_parse_utc(
                    fields["stock_bar_timestamp_utc"], "stock_bar_timestamp_utc"
                ),
                etf_bar_timestamp_utc=_parse_utc(
                    fields["etf_bar_timestamp_utc"], "etf_bar_timestamp_utc"
                ),
                stock_regular_market_time_utc=_parse_utc(
                    fields["stock_regular_market_time_utc"], "stock_regular_market_time_utc"
                ),
                etf_regular_market_time_utc=_parse_utc(
                    fields["etf_regular_market_time_utc"], "etf_regular_market_time_utc"
                ),
                stock_received_at_utc=_parse_utc(fields["stock_received_at_utc"], "stock_received_at_utc"),
                etf_received_at_utc=_parse_utc(fields["etf_received_at_utc"], "etf_received_at_utc"),
                stock_raw_response_hash=fields["stock_raw_response_hash"],
                etf_raw_response_hash=fields["etf_raw_response_hash"],
                stock_source_url=fields["stock_source_url"],
                etf_source_url=fields["etf_source_url"],
            )
        except CheckpointIntegrityError:
            raise
        except (TypeError, ValueError) as exception:
            raise CheckpointIntegrityError(f"invalid confirmation evidence: {exception}") from exception

    def to_fields(self) -> dict[str, Any]:
        return {
            "confirmation_index": self.confirmation_index,
            "stock_symbol": self.stock_symbol,
            "etf_symbol": self.etf_symbol,
            "stock_close": _format_decimal(self.stock_close),
            "etf_close": _format_decimal(self.etf_close),
            "stock_bar_timestamp_utc": _format_utc(self.stock_bar_timestamp_utc),
            "etf_bar_timestamp_utc": _format_utc(self.etf_bar_timestamp_utc),
            "stock_regular_market_time_utc": _format_utc(self.stock_regular_market_time_utc),
            "etf_regular_market_time_utc": _format_utc(self.etf_regular_market_time_utc),
            "stock_received_at_utc": _format_utc(self.stock_received_at_utc),
            "etf_received_at_utc": _format_utc(self.etf_received_at_utc),
            "stock_raw_response_hash": self.stock_raw_response_hash,
            "etf_raw_response_hash": self.etf_raw_response_hash,
            "stock_source_url": self.stock_source_url,
            "etf_source_url": self.etf_source_url,
        }


@dataclass(frozen=True, slots=True)
class AnchorPollingCheckpoint:
    schema_version: int
    cycle_id: str
    target_session_date: date
    official_close_utc: datetime
    deadline_utc: datetime
    attempt: int
    next_poll_utc: datetime
    confirmation_count: int
    candidate_stock_close: Decimal | None
    candidate_etf_close: Decimal | None
    candidate_stock_raw_response_hash: str | None
    candidate_etf_raw_response_hash: str | None
    stock_received_at_utc: datetime | None
    etf_received_at_utc: datetime | None
    revision: int
    integrity_version: int = 1
    stock_symbol: str | None = None
    etf_symbol: str | None = None
    confirmation_evidence: tuple[AnchorConfirmationEvidence, ...] = ()
    integrity_hash: str | None = None

    def __post_init__(self) -> None:
        try:
            self._validate()
        except CheckpointIntegrityError:
            raise
        except (TypeError, ValueError) as exception:
            raise CheckpointIntegrityError(str(exception)) from exception

    def _validate(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise CheckpointIntegrityError("schema version must be 1")
        if not isinstance(self.target_session_date, date) or isinstance(self.target_session_date, datetime):
            raise CheckpointIntegrityError("target session date must be a date")
        expected_cycle_id = f"xnys-{self.target_session_date.isoformat()}"
        if self.cycle_id != expected_cycle_id:
            raise CheckpointIntegrityError("cycle ID does not match the target session date")
        _validate_utc(self.official_close_utc, "official close")
        _validate_utc(self.deadline_utc, "deadline")
        _validate_utc(self.next_poll_utc, "next poll")
        if self.official_close_utc.date() != self.target_session_date:
            raise CheckpointIntegrityError("official close does not match the target session date")
        if self.deadline_utc <= self.official_close_utc:
            raise CheckpointIntegrityError("deadline must be later than official close")
        if self.deadline_utc - self.official_close_utc > timedelta(seconds=600):
            raise CheckpointIntegrityError("deadline exceeds the 600-second anchor bound")
        if self.next_poll_utc < self.official_close_utc:
            raise CheckpointIntegrityError("next poll cannot precede official close")
        if self.next_poll_utc > self.deadline_utc:
            raise CheckpointIntegrityError("next poll cannot be later than the deadline")
        for value, field_name, minimum in (
            (self.attempt, "attempt", 0),
            (self.confirmation_count, "confirmation count", 0),
            (self.revision, "revision", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise CheckpointIntegrityError(f"{field_name} must be an integer of at least {minimum}")
        if self.confirmation_count > self.attempt:
            raise CheckpointIntegrityError("confirmation count cannot exceed attempt")

        candidate_fields = (
            self.candidate_stock_close,
            self.candidate_etf_close,
            self.candidate_stock_raw_response_hash,
            self.candidate_etf_raw_response_hash,
            self.stock_received_at_utc,
            self.etf_received_at_utc,
        )
        present_count = sum(value is not None for value in candidate_fields)
        if present_count not in (0, len(candidate_fields)):
            raise CheckpointIntegrityError("paired candidate checkpoint fields must be all null or all present")
        if present_count == 0:
            if self.confirmation_count != 0:
                raise CheckpointIntegrityError("confirmation count requires a paired candidate")
        else:
            if self.confirmation_count < 1:
                raise CheckpointIntegrityError("confirmation count must be positive for a paired candidate")
            _validate_positive_decimal(self.candidate_stock_close, "candidate stock close")
            _validate_positive_decimal(self.candidate_etf_close, "candidate ETF close")
            _validate_hash(self.candidate_stock_raw_response_hash, "candidate stock raw response hash")
            _validate_hash(self.candidate_etf_raw_response_hash, "candidate ETF raw response hash")
            _validate_utc(self.stock_received_at_utc, "stock received time")
            _validate_utc(self.etf_received_at_utc, "ETF received time")
            for received_at, field_name in (
                (self.stock_received_at_utc, "stock received time"),
                (self.etf_received_at_utc, "ETF received time"),
            ):
                if not self.official_close_utc <= received_at < self.deadline_utc:
                    raise CheckpointIntegrityError(
                        f"{field_name} must be inside the post-close acquisition window"
                    )
            if self.next_poll_utc < max(self.stock_received_at_utc, self.etf_received_at_utc):
                raise CheckpointIntegrityError("next poll cannot precede the latest candidate receive time")

        if isinstance(self.integrity_version, bool) or self.integrity_version not in (1, 2):
            raise CheckpointIntegrityError("checkpoint integrity version must be 1 or 2")
        if not isinstance(self.confirmation_evidence, tuple):
            raise CheckpointIntegrityError("checkpoint confirmation evidence must be an immutable tuple")
        if self.integrity_version == 1:
            if (
                self.stock_symbol is not None
                or self.etf_symbol is not None
                or self.confirmation_evidence
                or self.integrity_hash is not None
            ):
                raise CheckpointIntegrityError("version-1 checkpoint cannot contain version-2 recovery evidence")
            return

        if self.confirmation_count == 0:
            if self.stock_symbol is not None or self.etf_symbol is not None or self.confirmation_evidence:
                raise CheckpointIntegrityError("empty checkpoint cannot contain confirmation evidence")
        else:
            _validate_symbol(self.stock_symbol, "checkpoint stock symbol")
            _validate_symbol(self.etf_symbol, "checkpoint ETF symbol")
            if len(self.confirmation_evidence) != self.confirmation_count:
                raise CheckpointIntegrityError("checkpoint evidence count does not match confirmation count")
            for expected_index, record in enumerate(self.confirmation_evidence, start=1):
                if not isinstance(record, AnchorConfirmationEvidence):
                    raise CheckpointIntegrityError("checkpoint confirmation evidence has an invalid record")
                if record.confirmation_index != expected_index:
                    raise CheckpointIntegrityError("checkpoint confirmation evidence order is invalid")
                if record.stock_symbol != self.stock_symbol or record.etf_symbol != self.etf_symbol:
                    raise CheckpointIntegrityError("checkpoint confirmation evidence symbols do not match")
                if record.stock_close != self.candidate_stock_close or record.etf_close != self.candidate_etf_close:
                    raise CheckpointIntegrityError("checkpoint confirmation evidence closes do not match")
                for received_at in (record.stock_received_at_utc, record.etf_received_at_utc):
                    if not self.official_close_utc <= received_at < self.deadline_utc:
                        raise CheckpointIntegrityError("checkpoint evidence receive time is outside its window")
            latest = self.confirmation_evidence[-1]
            if (
                latest.stock_raw_response_hash != self.candidate_stock_raw_response_hash
                or latest.etf_raw_response_hash != self.candidate_etf_raw_response_hash
                or latest.stock_received_at_utc != self.stock_received_at_utc
                or latest.etf_received_at_utc != self.etf_received_at_utc
            ):
                raise CheckpointIntegrityError("checkpoint latest summary does not match its evidence trail")

        expected_integrity_hash = _canonical_sha256(self._integrity_payload())
        if self.integrity_hash is None:
            object.__setattr__(self, "integrity_hash", expected_integrity_hash)
        else:
            _validate_hash(self.integrity_hash, "checkpoint integrity hash")
            if self.integrity_hash != expected_integrity_hash:
                raise CheckpointIntegrityError("checkpoint integrity hash does not match its evidence payload")

    def _integrity_payload(self) -> dict[str, Any]:
        return {
            "integrity_version": self.integrity_version,
            "cycle_id": self.cycle_id,
            "target_session_date": self.target_session_date.isoformat(),
            "official_close_utc": _format_utc(self.official_close_utc),
            "deadline_utc": _format_utc(self.deadline_utc),
            "confirmation_count": self.confirmation_count,
            "candidate_stock_close": (
                None if self.candidate_stock_close is None else _format_decimal(self.candidate_stock_close)
            ),
            "candidate_etf_close": (
                None if self.candidate_etf_close is None else _format_decimal(self.candidate_etf_close)
            ),
            "candidate_stock_raw_response_hash": self.candidate_stock_raw_response_hash,
            "candidate_etf_raw_response_hash": self.candidate_etf_raw_response_hash,
            "stock_received_at_utc": (
                None if self.stock_received_at_utc is None else _format_utc(self.stock_received_at_utc)
            ),
            "etf_received_at_utc": (
                None if self.etf_received_at_utc is None else _format_utc(self.etf_received_at_utc)
            ),
            "stock_symbol": self.stock_symbol,
            "etf_symbol": self.etf_symbol,
            "confirmation_evidence": [record.to_fields() for record in self.confirmation_evidence],
        }

    @classmethod
    def from_contract_fields(cls, fields: Mapping[str, Any]) -> "AnchorPollingCheckpoint":
        if not isinstance(fields, Mapping):
            raise CheckpointIntegrityError("checkpoint contract fields must be an object")
        if set(fields) != _CHECKPOINT_FIELDS:
            raise CheckpointIntegrityError("checkpoint contract fields do not match AnchorPollingCheckpointV1")
        target_value = fields["target_session_date"]
        if not isinstance(target_value, str) or _DATE_PATTERN.fullmatch(target_value) is None:
            raise CheckpointIntegrityError("target session date must be a canonical date")
        try:
            target_session_date = date.fromisoformat(target_value)
        except ValueError as exception:
            raise CheckpointIntegrityError("target session date must be a valid canonical date") from exception

        def optional_decimal(field_name: str) -> Decimal | None:
            value = fields[field_name]
            return None if value is None else _parse_canonical_decimal(value, field_name)

        def optional_utc(field_name: str) -> datetime | None:
            value = fields[field_name]
            return None if value is None else _parse_utc(value, field_name)

        try:
            return cls(
                schema_version=fields["schema_version"],
                cycle_id=fields["cycle_id"],
                target_session_date=target_session_date,
                official_close_utc=_parse_utc(fields["official_close_utc"], "official_close_utc"),
                deadline_utc=_parse_utc(fields["deadline_utc"], "deadline_utc"),
                attempt=fields["attempt"],
                next_poll_utc=_parse_utc(fields["next_poll_utc"], "next_poll_utc"),
                confirmation_count=fields["confirmation_count"],
                candidate_stock_close=optional_decimal("candidate_stock_close"),
                candidate_etf_close=optional_decimal("candidate_etf_close"),
                candidate_stock_raw_response_hash=fields["candidate_stock_raw_response_hash"],
                candidate_etf_raw_response_hash=fields["candidate_etf_raw_response_hash"],
                stock_received_at_utc=optional_utc("stock_received_at_utc"),
                etf_received_at_utc=optional_utc("etf_received_at_utc"),
                revision=fields["revision"],
            )
        except CheckpointIntegrityError:
            raise
        except (TypeError, ValueError) as exception:
            raise CheckpointIntegrityError(f"invalid checkpoint contract fields: {exception}") from exception

    def to_contract_fields(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "cycle_id": self.cycle_id,
            "target_session_date": self.target_session_date.isoformat(),
            "official_close_utc": _format_utc(self.official_close_utc),
            "deadline_utc": _format_utc(self.deadline_utc),
            "attempt": self.attempt,
            "next_poll_utc": _format_utc(self.next_poll_utc),
            "confirmation_count": self.confirmation_count,
            "candidate_stock_close": (
                None if self.candidate_stock_close is None else _format_decimal(self.candidate_stock_close)
            ),
            "candidate_etf_close": (
                None if self.candidate_etf_close is None else _format_decimal(self.candidate_etf_close)
            ),
            "candidate_stock_raw_response_hash": self.candidate_stock_raw_response_hash,
            "candidate_etf_raw_response_hash": self.candidate_etf_raw_response_hash,
            "stock_received_at_utc": (
                None if self.stock_received_at_utc is None else _format_utc(self.stock_received_at_utc)
            ),
            "etf_received_at_utc": None if self.etf_received_at_utc is None else _format_utc(self.etf_received_at_utc),
            "revision": self.revision,
        }

    @classmethod
    def from_recovery_fields(cls, fields: Mapping[str, Any]) -> "AnchorPollingCheckpoint":
        if not isinstance(fields, Mapping) or set(fields) != _RECOVERY_FIELDS:
            raise CheckpointIntegrityError("checkpoint recovery fields do not match version 2")
        raw_records = fields["confirmation_evidence"]
        if not isinstance(raw_records, list):
            raise CheckpointIntegrityError("checkpoint confirmation evidence must be an array")
        contract_fields = {field_name: fields[field_name] for field_name in _CHECKPOINT_FIELDS}
        base = cls.from_contract_fields(contract_fields)
        try:
            return replace(
                base,
                integrity_version=fields["integrity_version"],
                stock_symbol=fields["stock_symbol"],
                etf_symbol=fields["etf_symbol"],
                confirmation_evidence=tuple(
                    AnchorConfirmationEvidence.from_fields(record) for record in raw_records
                ),
                integrity_hash=fields["integrity_hash"],
            )
        except CheckpointIntegrityError:
            raise
        except (TypeError, ValueError) as exception:
            raise CheckpointIntegrityError(f"invalid checkpoint recovery fields: {exception}") from exception

    def to_recovery_fields(self) -> dict[str, Any]:
        fields = self.to_contract_fields()
        fields.update(
            {
                "integrity_version": self.integrity_version,
                "stock_symbol": self.stock_symbol,
                "etf_symbol": self.etf_symbol,
                "confirmation_evidence": [record.to_fields() for record in self.confirmation_evidence],
                "integrity_hash": self.integrity_hash,
            }
        )
        return fields


@dataclass(frozen=True, slots=True)
class AnchorCandidate:
    evidence_version: int
    cycle_id: str
    pair_id: str
    target_session_date: date
    official_close_utc: datetime
    anchor_source: str
    stock_symbol: str
    etf_symbol: str
    stock_close: Decimal
    etf_close: Decimal
    stock_bar_timestamp_utc: datetime
    etf_bar_timestamp_utc: datetime
    stock_regular_market_time_utc: datetime
    etf_regular_market_time_utc: datetime
    stock_received_at_utc: datetime
    etf_received_at_utc: datetime
    acquisition_started_at_utc: datetime
    finalized_at_utc: datetime
    deadline_utc: datetime
    stock_raw_response_hash: str
    etf_raw_response_hash: str
    etf_daily_multiplier: Decimal
    hedge_ratio: Decimal
    anchor_confirmation_count: int
    observed_confirmation_count: int
    anchor_confirmation_interval_seconds: int
    anchor_min_finalize_delay_seconds: int
    anchor_pair_fetch_max_skew_seconds: int
    confirmation_evidence: tuple[AnchorConfirmationEvidence, ...]
    evidence_hash: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.evidence_version, bool) or self.evidence_version != _ANCHOR_EVIDENCE_VERSION:
            raise ValueError(f"candidate evidence version must be {_ANCHOR_EVIDENCE_VERSION}")
        if not isinstance(self.target_session_date, date) or isinstance(self.target_session_date, datetime):
            raise TypeError("candidate target session date must be a date")
        expected_cycle_id = f"xnys-{self.target_session_date.isoformat()}"
        if self.cycle_id != expected_cycle_id:
            raise ValueError("candidate cycle ID does not match target session date")
        if not isinstance(self.pair_id, str) or not self.pair_id or len(self.pair_id) > 128:
            raise ValueError("candidate pair ID must be a bounded non-empty string")
        if not isinstance(self.anchor_source, str) or not self.anchor_source:
            raise ValueError("candidate anchor source must be a non-empty string")
        _validate_symbol(self.stock_symbol, "stock symbol")
        _validate_symbol(self.etf_symbol, "ETF symbol")
        _validate_positive_decimal(self.stock_close, "stock close")
        _validate_positive_decimal(self.etf_close, "ETF close")
        validate_bounded_decimal(self.etf_daily_multiplier, "ETF daily multiplier", positive=True)
        validate_bounded_decimal(self.hedge_ratio, "hedge ratio", positive=True)
        for value, field_name in (
            (self.official_close_utc, "official close"),
            (self.stock_bar_timestamp_utc, "stock bar timestamp"),
            (self.etf_bar_timestamp_utc, "ETF bar timestamp"),
            (self.stock_regular_market_time_utc, "stock regular market time"),
            (self.etf_regular_market_time_utc, "ETF regular market time"),
            (self.stock_received_at_utc, "stock received time"),
            (self.etf_received_at_utc, "ETF received time"),
            (self.acquisition_started_at_utc, "acquisition started time"),
            (self.finalized_at_utc, "finalized time"),
            (self.deadline_utc, "deadline"),
        ):
            _validate_utc(value, field_name)
        if self.finalized_at_utc >= self.deadline_utc:
            raise ValueError("candidate must finalize before its deadline")
        if max(self.stock_received_at_utc, self.etf_received_at_utc) > self.finalized_at_utc:
            raise ValueError("candidate cannot finalize before its responses are received")
        _validate_hash(self.stock_raw_response_hash, "stock raw response hash")
        _validate_hash(self.etf_raw_response_hash, "ETF raw response hash")
        for value, field_name, minimum in (
            (self.anchor_confirmation_count, "required confirmation count", 2),
            (self.observed_confirmation_count, "observed confirmation count", 1),
            (self.anchor_confirmation_interval_seconds, "confirmation interval", 1),
            (self.anchor_min_finalize_delay_seconds, "minimum finalize delay", 1),
            (self.anchor_pair_fetch_max_skew_seconds, "pair fetch maximum skew", 1),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{field_name} must be an integer of at least {minimum}")
        if not isinstance(self.confirmation_evidence, tuple):
            raise TypeError("candidate confirmation evidence must be an immutable tuple")
        if any(not isinstance(record, AnchorConfirmationEvidence) for record in self.confirmation_evidence):
            raise TypeError("candidate confirmation evidence contains an invalid record")
        if self.evidence_hash is None:
            object.__setattr__(self, "evidence_hash", _canonical_sha256(self._validated_payload()))
        else:
            _validate_hash(self.evidence_hash, "evidence hash")

    def _validated_payload(self) -> dict[str, Any]:
        if self.evidence_version != _ANCHOR_EVIDENCE_VERSION:
            raise ValueError("candidate evidence version is unsupported")
        if self.cycle_id != f"xnys-{self.target_session_date.isoformat()}":
            raise ValueError("candidate cycle ID does not match target session date")
        if self.pair_id != f"{self.stock_symbol.lower()}_{self.etf_symbol.lower()}":
            raise ValueError("candidate pair ID does not match its symbols")
        if self.stock_symbol == self.etf_symbol:
            raise ValueError("candidate stock and ETF symbols must differ")
        if self.anchor_source != "yahoo_finance_chart_http":
            raise ValueError("candidate anchor source is unsupported")
        if self.official_close_utc.date() != self.target_session_date:
            raise ValueError("candidate official close does not match target session date")
        if self.acquisition_started_at_utc != self.official_close_utc:
            raise ValueError("candidate acquisition start must equal the official close")
        if not self.official_close_utc < self.deadline_utc:
            raise ValueError("candidate deadline must follow official close")
        if self.deadline_utc - self.official_close_utc > timedelta(seconds=600):
            raise ValueError("candidate deadline exceeds the bounded anchor window")
        if self.observed_confirmation_count != self.anchor_confirmation_count:
            raise ValueError("final candidate must contain every required confirmation")
        if len(self.confirmation_evidence) != self.observed_confirmation_count:
            raise ValueError("candidate evidence count does not match observed confirmation count")
        if not self.confirmation_evidence:
            raise ValueError("candidate must contain confirmation evidence")

        previous_received_at: datetime | None = None
        for expected_index, record in enumerate(self.confirmation_evidence, start=1):
            if record.confirmation_index != expected_index:
                raise ValueError("candidate confirmation evidence order is invalid")
            if record.stock_symbol != self.stock_symbol or record.etf_symbol != self.etf_symbol:
                raise ValueError("candidate confirmation evidence symbols do not match")
            if record.stock_close != self.stock_close or record.etf_close != self.etf_close:
                raise ValueError("candidate confirmation evidence closes do not match")
            for session_timestamp in (
                record.stock_bar_timestamp_utc,
                record.etf_bar_timestamp_utc,
                record.stock_regular_market_time_utc,
                record.etf_regular_market_time_utc,
            ):
                if session_timestamp.date() != self.target_session_date:
                    raise ValueError("candidate confirmation provenance does not match target session date")
            current_received_at = max(record.stock_received_at_utc, record.etf_received_at_utc)
            if not self.official_close_utc <= min(
                record.stock_received_at_utc, record.etf_received_at_utc
            ) or current_received_at >= self.deadline_utc:
                raise ValueError("candidate confirmation evidence is outside the acquisition window")
            skew = abs((record.stock_received_at_utc - record.etf_received_at_utc).total_seconds())
            if skew > self.anchor_pair_fetch_max_skew_seconds:
                raise ValueError("candidate confirmation evidence exceeds the configured receive skew")
            if previous_received_at is not None:
                elapsed = (current_received_at - previous_received_at).total_seconds()
                if elapsed < self.anchor_confirmation_interval_seconds:
                    raise ValueError("candidate confirmations are too close together")
            previous_received_at = current_received_at

        latest = self.confirmation_evidence[-1]
        latest_fields_match = (
            self.stock_bar_timestamp_utc == latest.stock_bar_timestamp_utc
            and self.etf_bar_timestamp_utc == latest.etf_bar_timestamp_utc
            and self.stock_regular_market_time_utc == latest.stock_regular_market_time_utc
            and self.etf_regular_market_time_utc == latest.etf_regular_market_time_utc
            and self.stock_received_at_utc == latest.stock_received_at_utc
            and self.etf_received_at_utc == latest.etf_received_at_utc
            and self.stock_raw_response_hash == latest.stock_raw_response_hash
            and self.etf_raw_response_hash == latest.etf_raw_response_hash
        )
        if not latest_fields_match:
            raise ValueError("candidate latest fields do not match its confirmation trail")
        expected_finalized_at = max(
            self.official_close_utc + timedelta(seconds=self.anchor_min_finalize_delay_seconds),
            latest.stock_received_at_utc,
            latest.etf_received_at_utc,
        )
        if self.finalized_at_utc != expected_finalized_at:
            raise ValueError("candidate finalization timestamp is not deterministic")
        expected_hedge_ratio = calculate_hedge_ratio(
            self.stock_close,
            self.etf_close,
            self.etf_daily_multiplier,
        )
        if self.hedge_ratio != expected_hedge_ratio:
            raise ValueError("candidate hedge ratio does not match its bound anchors and multiplier")

        return {
            "evidence_version": self.evidence_version,
            "cycle_id": self.cycle_id,
            "pair_id": self.pair_id,
            "target_session_date": self.target_session_date.isoformat(),
            "official_close_utc": _format_utc(self.official_close_utc),
            "anchor_source": self.anchor_source,
            "stock_symbol": self.stock_symbol,
            "etf_symbol": self.etf_symbol,
            "stock_close": _format_decimal(self.stock_close),
            "etf_close": _format_decimal(self.etf_close),
            "stock_bar_timestamp_utc": _format_utc(self.stock_bar_timestamp_utc),
            "etf_bar_timestamp_utc": _format_utc(self.etf_bar_timestamp_utc),
            "stock_regular_market_time_utc": _format_utc(self.stock_regular_market_time_utc),
            "etf_regular_market_time_utc": _format_utc(self.etf_regular_market_time_utc),
            "stock_received_at_utc": _format_utc(self.stock_received_at_utc),
            "etf_received_at_utc": _format_utc(self.etf_received_at_utc),
            "acquisition_started_at_utc": _format_utc(self.acquisition_started_at_utc),
            "finalized_at_utc": _format_utc(self.finalized_at_utc),
            "deadline_utc": _format_utc(self.deadline_utc),
            "stock_raw_response_hash": self.stock_raw_response_hash,
            "etf_raw_response_hash": self.etf_raw_response_hash,
            "etf_daily_multiplier": _format_financial_decimal(self.etf_daily_multiplier),
            "hedge_ratio": _format_financial_decimal(self.hedge_ratio),
            "anchor_confirmation_count": self.anchor_confirmation_count,
            "observed_confirmation_count": self.observed_confirmation_count,
            "anchor_confirmation_interval_seconds": self.anchor_confirmation_interval_seconds,
            "anchor_min_finalize_delay_seconds": self.anchor_min_finalize_delay_seconds,
            "anchor_pair_fetch_max_skew_seconds": self.anchor_pair_fetch_max_skew_seconds,
            "confirmation_evidence": [record.to_fields() for record in self.confirmation_evidence],
        }

    def verify_evidence_hash(self) -> bool:
        try:
            return self.evidence_hash == _canonical_sha256(self._validated_payload())
        except (CheckpointIntegrityError, TypeError, ValueError):
            return False

    def to_evidence_fields(self) -> dict[str, Any]:
        try:
            payload = self._validated_payload()
        except (TypeError, ValueError) as exception:
            raise CheckpointIntegrityError(f"candidate evidence payload is invalid: {exception}") from exception
        if self.evidence_hash != _canonical_sha256(payload):
            raise CheckpointIntegrityError("candidate evidence hash does not match its canonical payload")
        return {**payload, "evidence_hash": self.evidence_hash}

    @classmethod
    def from_evidence_fields(cls, fields: Mapping[str, Any]) -> "AnchorCandidate":
        if not isinstance(fields, Mapping) or set(fields) != _CANDIDATE_EVIDENCE_FIELDS:
            raise CheckpointIntegrityError("candidate evidence fields do not match version 2")
        raw_records = fields["confirmation_evidence"]
        if not isinstance(raw_records, list):
            raise CheckpointIntegrityError("candidate confirmation evidence must be an array")
        try:
            candidate = cls(
                evidence_version=fields["evidence_version"],
                cycle_id=fields["cycle_id"],
                pair_id=fields["pair_id"],
                target_session_date=_parse_date(fields["target_session_date"], "target_session_date"),
                official_close_utc=_parse_utc(fields["official_close_utc"], "official_close_utc"),
                anchor_source=fields["anchor_source"],
                stock_symbol=fields["stock_symbol"],
                etf_symbol=fields["etf_symbol"],
                stock_close=_parse_canonical_decimal(fields["stock_close"], "stock_close"),
                etf_close=_parse_canonical_decimal(fields["etf_close"], "etf_close"),
                stock_bar_timestamp_utc=_parse_utc(
                    fields["stock_bar_timestamp_utc"], "stock_bar_timestamp_utc"
                ),
                etf_bar_timestamp_utc=_parse_utc(
                    fields["etf_bar_timestamp_utc"], "etf_bar_timestamp_utc"
                ),
                stock_regular_market_time_utc=_parse_utc(
                    fields["stock_regular_market_time_utc"], "stock_regular_market_time_utc"
                ),
                etf_regular_market_time_utc=_parse_utc(
                    fields["etf_regular_market_time_utc"], "etf_regular_market_time_utc"
                ),
                stock_received_at_utc=_parse_utc(fields["stock_received_at_utc"], "stock_received_at_utc"),
                etf_received_at_utc=_parse_utc(fields["etf_received_at_utc"], "etf_received_at_utc"),
                acquisition_started_at_utc=_parse_utc(
                    fields["acquisition_started_at_utc"], "acquisition_started_at_utc"
                ),
                finalized_at_utc=_parse_utc(fields["finalized_at_utc"], "finalized_at_utc"),
                deadline_utc=_parse_utc(fields["deadline_utc"], "deadline_utc"),
                stock_raw_response_hash=fields["stock_raw_response_hash"],
                etf_raw_response_hash=fields["etf_raw_response_hash"],
                etf_daily_multiplier=_parse_canonical_financial_decimal(
                    fields["etf_daily_multiplier"], "etf_daily_multiplier"
                ),
                hedge_ratio=_parse_canonical_financial_decimal(fields["hedge_ratio"], "hedge_ratio"),
                anchor_confirmation_count=fields["anchor_confirmation_count"],
                observed_confirmation_count=fields["observed_confirmation_count"],
                anchor_confirmation_interval_seconds=fields["anchor_confirmation_interval_seconds"],
                anchor_min_finalize_delay_seconds=fields["anchor_min_finalize_delay_seconds"],
                anchor_pair_fetch_max_skew_seconds=fields["anchor_pair_fetch_max_skew_seconds"],
                confirmation_evidence=tuple(
                    AnchorConfirmationEvidence.from_fields(record) for record in raw_records
                ),
                evidence_hash=fields["evidence_hash"],
            )
        except CheckpointIntegrityError:
            raise
        except (TypeError, ValueError) as exception:
            raise CheckpointIntegrityError(f"invalid candidate evidence fields: {exception}") from exception
        if not candidate.verify_evidence_hash():
            raise CheckpointIntegrityError("candidate evidence hash does not match its canonical payload")
        return candidate


@dataclass(frozen=True, slots=True)
class RevisionObservation:
    cycle_id: str
    stock_close: Decimal
    etf_close: Decimal
    stock_raw_response_hash: str
    etf_raw_response_hash: str
    evidence_hash: str
    observed_at_utc: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.cycle_id, str) or _CYCLE_PATTERN.fullmatch(self.cycle_id) is None:
            raise ValueError("revision cycle ID must be an XNYS cycle identifier")
        _validate_positive_decimal(self.stock_close, "revision stock close")
        _validate_positive_decimal(self.etf_close, "revision ETF close")
        _validate_hash(self.stock_raw_response_hash, "revision stock raw response hash")
        _validate_hash(self.etf_raw_response_hash, "revision ETF raw response hash")
        _validate_hash(self.evidence_hash, "revision evidence hash")
        _validate_utc(self.observed_at_utc, "revision observed time")
        expected_evidence_hash = anchor_evidence_hash(
            cycle_id=self.cycle_id,
            stock_raw_response_hash=self.stock_raw_response_hash,
            etf_raw_response_hash=self.etf_raw_response_hash,
        )
        if self.evidence_hash != expected_evidence_hash:
            raise ValueError("revision evidence hash does not match its immutable response hashes")


@dataclass(frozen=True, slots=True)
class AnchorAcquisitionResult:
    status: AnchorAcquisitionStatus
    checkpoint: AnchorPollingCheckpoint
    candidate: AnchorCandidate | None = None
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class FinalizedAnchorAssessment:
    status: FinalizedAnchorStatus
    candidate: AnchorCandidate
    revision_observation: RevisionObservation | None = None


@dataclass(frozen=True, slots=True)
class _PairLegOutcome:
    observation: YahooCloseObservation
    completed_monotonic: float


class YahooAnchorAcquisition:
    """Checkpoint-driven, persistence-free Yahoo anchor state machine."""

    def __init__(
        self,
        nav_config: NavConfig,
        provider: Optional[YahooChartProvider] = None,
        utc_clock: Callable[[], datetime] = _utc_now,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float], float] = _random_jitter,
        etf_daily_multiplier: Decimal = Decimal("2"),
    ):
        if not isinstance(nav_config, NavConfig):
            raise TypeError("nav_config must be a NavConfig")
        self._nav_config = nav_config
        self._utc_clock = utc_clock
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
        self._etf_daily_multiplier = validate_bounded_decimal(
            etf_daily_multiplier,
            "ETF daily multiplier",
            positive=True,
        )
        self._calendar = XnysNavCalendar(nav_config=nav_config, clock=utc_clock)
        self._provider = provider or YahooChartProvider(
            nav_config=nav_config,
            utc_clock=utc_clock,
            monotonic_clock=monotonic_clock,
            sleep=sleep,
            jitter=jitter,
        )
        self._active_budget_key: tuple[str, datetime] | None = None
        self._active_budget: YahooCycleBudget | None = None
        self._poll_saturation_attempt = self._calculate_poll_saturation_attempt()

    def create_checkpoint(
        self,
        cycle_id: str,
        target_session_date: date,
        official_close_utc: datetime,
    ) -> AnchorPollingCheckpoint:
        self._validate_exact_official_close(target_session_date, official_close_utc)
        checkpoint = AnchorPollingCheckpoint(
            schema_version=1,
            cycle_id=cycle_id,
            target_session_date=target_session_date,
            official_close_utc=official_close_utc,
            deadline_utc=official_close_utc + timedelta(seconds=self._nav_config.anchor_wait_timeout_seconds),
            attempt=0,
            next_poll_utc=official_close_utc,
            confirmation_count=0,
            candidate_stock_close=None,
            candidate_etf_close=None,
            candidate_stock_raw_response_hash=None,
            candidate_etf_raw_response_hash=None,
            stock_received_at_utc=None,
            etf_received_at_utc=None,
            revision=1,
            integrity_version=_ANCHOR_EVIDENCE_VERSION,
        )
        self._active_budget_key = (checkpoint.cycle_id, checkpoint.deadline_utc)
        self._active_budget = YahooCycleBudget.start(
            deadline_utc=checkpoint.deadline_utc,
            utc_clock=self._utc_clock,
            monotonic_clock=self._monotonic_clock,
        )
        return checkpoint

    async def advance(
        self,
        checkpoint: AnchorPollingCheckpoint,
        stock_symbol: str,
        etf_symbol: str,
        *,
        budget: YahooCycleBudget | None = None,
    ) -> AnchorAcquisitionResult:
        self._validate_checkpoint_for_config(checkpoint)
        self._validate_symbol(stock_symbol, "stock symbol")
        self._validate_symbol(etf_symbol, "ETF symbol")
        if checkpoint.confirmation_count > 0 and (
            checkpoint.stock_symbol != stock_symbol or checkpoint.etf_symbol != etf_symbol
        ):
            raise CheckpointIntegrityError("checkpoint evidence symbols do not match the requested pair")
        budget = budget or self._budget_for(checkpoint)
        if budget.deadline_utc != checkpoint.deadline_utc:
            raise ValueError("cycle budget does not match checkpoint deadline")
        now = _validate_utc(self._utc_clock(), "UTC clock")
        try:
            remaining = budget.ensure_remaining("Yahoo anchor round")
        except YahooDeadlineExceeded:
            return self._deadline_result(checkpoint)
        if now >= checkpoint.deadline_utc:
            return self._deadline_result(checkpoint)

        if now < checkpoint.next_poll_utc:
            return AnchorAcquisitionResult(status=AnchorAcquisitionStatus.WAITING, checkpoint=checkpoint)
        if not self._has_conservative_budget(checkpoint, remaining):
            return AnchorAcquisitionResult(
                status=AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE,
                checkpoint=checkpoint,
                failure_reason="insufficient conservative budget for remaining confirmations",
            )

        try:
            outcomes = await budget.wait(
                self._fetch_pair(checkpoint, stock_symbol, etf_symbol, budget),
                "paired Yahoo gather",
            )
        except asyncio.CancelledError:
            raise
        except YahooDeadlineExceeded:
            return self._deadline_result(checkpoint, "Yahoo pair reached the absolute deadline")
        for outcome in outcomes:
            if isinstance(outcome, asyncio.CancelledError):
                raise outcome
        failures = [outcome for outcome in outcomes if isinstance(outcome, BaseException)]
        successful_legs = [outcome for outcome in outcomes if isinstance(outcome, _PairLegOutcome)]
        round_time = max(
            [now, _validate_utc(self._utc_clock(), "UTC clock")]
            + [leg.observation.received_at_utc for leg in successful_legs]
        )
        if failures:
            reason = "; ".join(str(failure) or type(failure).__name__ for failure in failures)
            return self._failed_round(checkpoint, round_time, reason)

        stock_leg, etf_leg = outcomes
        if not isinstance(stock_leg, _PairLegOutcome) or not isinstance(etf_leg, _PairLegOutcome):
            return self._failed_round(checkpoint, round_time, "Yahoo provider returned an invalid observation")
        stock_observation = stock_leg.observation
        etf_observation = etf_leg.observation
        try:
            self._validate_observation_window(stock_observation, stock_symbol, checkpoint)
            self._validate_observation_window(etf_observation, etf_symbol, checkpoint)
        except (TypeError, ValueError) as exception:
            return self._failed_round(checkpoint, round_time, str(exception))
        round_time = max(round_time, stock_observation.received_at_utc, etf_observation.received_at_utc)
        if round_time >= checkpoint.deadline_utc:
            return self._deadline_result(checkpoint, "Yahoo pair completed at or after the absolute deadline")
        try:
            budget.ensure_remaining("paired Yahoo validation")
        except YahooDeadlineExceeded:
            return self._deadline_result(checkpoint, "Yahoo pair reached the absolute deadline")

        utc_skew = abs((stock_observation.received_at_utc - etf_observation.received_at_utc).total_seconds())
        if utc_skew > self._nav_config.anchor_pair_fetch_max_skew_seconds:
            return self._failed_round(
                checkpoint,
                round_time,
                "paired Yahoo receive skew exceeds configured maximum",
            )
        monotonic_skew = abs(stock_leg.completed_monotonic - etf_leg.completed_monotonic)
        if monotonic_skew > self._nav_config.anchor_pair_fetch_max_skew_seconds:
            return self._failed_round(
                checkpoint,
                round_time,
                "paired Yahoo monotonic completion skew exceeds configured maximum",
            )
        return self._accept_pair(checkpoint, stock_observation, etf_observation, round_time)

    async def run(
        self,
        checkpoint: AnchorPollingCheckpoint,
        stock_symbol: str,
        etf_symbol: str,
        on_checkpoint: Optional[Callable[[AnchorPollingCheckpoint], Any]] = None,
    ) -> AnchorAcquisitionResult:
        self._validate_checkpoint_for_config(checkpoint)
        current = checkpoint
        budget = self._budget_for(checkpoint)
        while True:
            if budget.remaining_seconds() <= 0:
                return self._deadline_result(current)
            result = await self.advance(
                current,
                stock_symbol,
                etf_symbol,
                budget=budget,
            )
            if result.checkpoint.revision != current.revision and on_checkpoint is not None:
                try:
                    await self._run_checkpoint_callback(on_checkpoint, result.checkpoint, budget)
                except YahooDeadlineExceeded:
                    return self._deadline_result(
                        result.checkpoint,
                        "checkpoint callback reached the absolute deadline",
                    )
            current = result.checkpoint
            if result.status in {
                AnchorAcquisitionStatus.FINALIZABLE,
                AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE,
            }:
                return result

            now = _validate_utc(self._utc_clock(), "UTC clock")
            delay = max(0.0, (current.next_poll_utc - now).total_seconds())
            try:
                await budget.wait(
                    self._sleep(min(delay, budget.ensure_remaining("inter-round sleep"))),
                    "inter-round sleep",
                )
            except YahooDeadlineExceeded:
                return self._deadline_result(current)

    def assess_finalized(
        self,
        candidate: AnchorCandidate,
        stock_observation: YahooCloseObservation,
        etf_observation: YahooCloseObservation,
    ) -> FinalizedAnchorAssessment:
        if not isinstance(candidate, AnchorCandidate):
            raise TypeError("candidate must be an AnchorCandidate")
        if not candidate.verify_evidence_hash():
            return FinalizedAnchorAssessment(
                status=FinalizedAnchorStatus.RECOVERY_REQUIRED,
                candidate=candidate,
            )
        if (
            candidate.anchor_source != self._nav_config.anchor_source
            or candidate.etf_daily_multiplier != self._etf_daily_multiplier
            or candidate.anchor_confirmation_count != self._nav_config.anchor_confirmation_count
            or candidate.anchor_confirmation_interval_seconds
            != self._nav_config.anchor_confirmation_interval_seconds
            or candidate.anchor_min_finalize_delay_seconds
            != self._nav_config.anchor_min_finalize_delay_seconds
            or candidate.anchor_pair_fetch_max_skew_seconds
            != self._nav_config.anchor_pair_fetch_max_skew_seconds
        ):
            return FinalizedAnchorAssessment(
                status=FinalizedAnchorStatus.RECOVERY_REQUIRED,
                candidate=candidate,
            )
        self._validate_observation(stock_observation, candidate.stock_symbol, candidate.target_session_date)
        self._validate_observation(etf_observation, candidate.etf_symbol, candidate.target_session_date)
        skew = abs((stock_observation.received_at_utc - etf_observation.received_at_utc).total_seconds())
        if skew > self._nav_config.anchor_pair_fetch_max_skew_seconds:
            raise ValueError("revision observation exceeds paired receive skew limit")
        if stock_observation.close == candidate.stock_close and etf_observation.close == candidate.etf_close:
            return FinalizedAnchorAssessment(status=FinalizedAnchorStatus.UNCHANGED, candidate=candidate)

        evidence_hash = anchor_evidence_hash(
            cycle_id=candidate.cycle_id,
            stock_raw_response_hash=stock_observation.raw_response_hash,
            etf_raw_response_hash=etf_observation.raw_response_hash,
        )
        revision = RevisionObservation(
            cycle_id=candidate.cycle_id,
            stock_close=stock_observation.close,
            etf_close=etf_observation.close,
            stock_raw_response_hash=stock_observation.raw_response_hash,
            etf_raw_response_hash=etf_observation.raw_response_hash,
            evidence_hash=evidence_hash,
            observed_at_utc=max(stock_observation.received_at_utc, etf_observation.received_at_utc),
        )
        return FinalizedAnchorAssessment(
            status=FinalizedAnchorStatus.DATA_REVISION,
            candidate=candidate,
            revision_observation=revision,
        )

    def _accept_pair(
        self,
        checkpoint: AnchorPollingCheckpoint,
        stock: YahooCloseObservation,
        etf: YahooCloseObservation,
        round_time: datetime,
    ) -> AnchorAcquisitionResult:
        attempt = checkpoint.attempt + 1
        current_received_at = max(stock.received_at_utc, etf.received_at_utc)
        same_candidate = (
            checkpoint.candidate_stock_close == stock.close and checkpoint.candidate_etf_close == etf.close
        )
        too_soon = False
        if checkpoint.confirmation_count == 0 or not same_candidate:
            confirmation_count = 1
            confirmation_evidence = (AnchorConfirmationEvidence.from_observations(1, stock, etf),)
        elif checkpoint.confirmation_count >= self._nav_config.anchor_confirmation_count:
            confirmation_count = checkpoint.confirmation_count
            confirmation_evidence = checkpoint.confirmation_evidence
        else:
            prior_received_at = max(checkpoint.stock_received_at_utc, checkpoint.etf_received_at_utc)
            confirmation_elapsed = (current_received_at - prior_received_at).total_seconds()
            if confirmation_elapsed >= self._nav_config.anchor_confirmation_interval_seconds:
                confirmation_count = checkpoint.confirmation_count + 1
                confirmation_evidence = checkpoint.confirmation_evidence + (
                    AnchorConfirmationEvidence.from_observations(
                        confirmation_count,
                        stock,
                        etf,
                    ),
                )
            else:
                too_soon = True
                confirmation_count = checkpoint.confirmation_count
                confirmation_evidence = checkpoint.confirmation_evidence

        latest = confirmation_evidence[-1]
        stock_close = latest.stock_close
        etf_close = latest.etf_close
        stock_hash = latest.stock_raw_response_hash
        etf_hash = latest.etf_raw_response_hash
        stock_received_at = latest.stock_received_at_utc
        etf_received_at = latest.etf_received_at_utc

        if too_soon:
            next_poll = max(stock_received_at, etf_received_at) + timedelta(
                seconds=self._nav_config.anchor_confirmation_interval_seconds
            )
        else:
            interval = max(
                self._nav_config.anchor_confirmation_interval_seconds,
                self._poll_interval(attempt),
            )
            next_poll = current_received_at + timedelta(seconds=interval)
        minimum_finalize_at = checkpoint.official_close_utc + timedelta(
            seconds=self._nav_config.anchor_min_finalize_delay_seconds
        )
        if confirmation_count >= self._nav_config.anchor_confirmation_count and round_time < minimum_finalize_at:
            next_poll = max(next_poll, minimum_finalize_at)
        next_poll = min(checkpoint.deadline_utc, next_poll)
        updated = replace(
            checkpoint,
            attempt=attempt,
            next_poll_utc=next_poll,
            confirmation_count=confirmation_count,
            candidate_stock_close=stock_close,
            candidate_etf_close=etf_close,
            candidate_stock_raw_response_hash=stock_hash,
            candidate_etf_raw_response_hash=etf_hash,
            stock_received_at_utc=stock_received_at,
            etf_received_at_utc=etf_received_at,
            revision=checkpoint.revision + 1,
            integrity_version=_ANCHOR_EVIDENCE_VERSION,
            stock_symbol=latest.stock_symbol,
            etf_symbol=latest.etf_symbol,
            confirmation_evidence=confirmation_evidence,
            integrity_hash=None,
        )
        if (
            confirmation_count < self._nav_config.anchor_confirmation_count
            or round_time < minimum_finalize_at
        ):
            return AnchorAcquisitionResult(status=AnchorAcquisitionStatus.POLLING, checkpoint=updated)

        finalized_at = max(
            minimum_finalize_at,
            latest.stock_received_at_utc,
            latest.etf_received_at_utc,
        )
        candidate = AnchorCandidate(
            evidence_version=_ANCHOR_EVIDENCE_VERSION,
            cycle_id=checkpoint.cycle_id,
            pair_id=f"{latest.stock_symbol.lower()}_{latest.etf_symbol.lower()}",
            target_session_date=checkpoint.target_session_date,
            official_close_utc=checkpoint.official_close_utc,
            anchor_source=self._nav_config.anchor_source,
            stock_symbol=latest.stock_symbol,
            etf_symbol=latest.etf_symbol,
            stock_close=latest.stock_close,
            etf_close=latest.etf_close,
            stock_bar_timestamp_utc=latest.stock_bar_timestamp_utc,
            etf_bar_timestamp_utc=latest.etf_bar_timestamp_utc,
            stock_regular_market_time_utc=latest.stock_regular_market_time_utc,
            etf_regular_market_time_utc=latest.etf_regular_market_time_utc,
            stock_received_at_utc=latest.stock_received_at_utc,
            etf_received_at_utc=latest.etf_received_at_utc,
            acquisition_started_at_utc=checkpoint.official_close_utc,
            finalized_at_utc=finalized_at,
            deadline_utc=checkpoint.deadline_utc,
            stock_raw_response_hash=latest.stock_raw_response_hash,
            etf_raw_response_hash=latest.etf_raw_response_hash,
            etf_daily_multiplier=self._etf_daily_multiplier,
            hedge_ratio=calculate_hedge_ratio(
                latest.stock_close,
                latest.etf_close,
                self._etf_daily_multiplier,
            ),
            anchor_confirmation_count=self._nav_config.anchor_confirmation_count,
            observed_confirmation_count=confirmation_count,
            anchor_confirmation_interval_seconds=self._nav_config.anchor_confirmation_interval_seconds,
            anchor_min_finalize_delay_seconds=self._nav_config.anchor_min_finalize_delay_seconds,
            anchor_pair_fetch_max_skew_seconds=self._nav_config.anchor_pair_fetch_max_skew_seconds,
            confirmation_evidence=confirmation_evidence,
        )
        return AnchorAcquisitionResult(
            status=AnchorAcquisitionStatus.FINALIZABLE,
            checkpoint=updated,
            candidate=candidate,
        )

    def _failed_round(
        self,
        checkpoint: AnchorPollingCheckpoint,
        round_time: datetime,
        reason: str,
    ) -> AnchorAcquisitionResult:
        failed_checkpoint = self._cleared_checkpoint(checkpoint, round_time)
        if round_time >= checkpoint.deadline_utc:
            status = AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE
        else:
            status = AnchorAcquisitionStatus.POLLING
        return AnchorAcquisitionResult(
            status=status,
            checkpoint=failed_checkpoint,
            failure_reason=reason,
        )

    def _cleared_checkpoint(
        self,
        checkpoint: AnchorPollingCheckpoint,
        round_time: datetime,
    ) -> AnchorPollingCheckpoint:
        attempt = checkpoint.attempt + 1
        next_poll = min(
            checkpoint.deadline_utc,
            round_time + timedelta(seconds=self._poll_interval(attempt)),
        )
        return replace(
            checkpoint,
            attempt=attempt,
            next_poll_utc=next_poll,
            confirmation_count=0,
            candidate_stock_close=None,
            candidate_etf_close=None,
            candidate_stock_raw_response_hash=None,
            candidate_etf_raw_response_hash=None,
            stock_received_at_utc=None,
            etf_received_at_utc=None,
            revision=checkpoint.revision + 1,
            integrity_version=_ANCHOR_EVIDENCE_VERSION,
            stock_symbol=None,
            etf_symbol=None,
            confirmation_evidence=(),
            integrity_hash=None,
        )

    def _poll_interval(self, attempt: int) -> int:
        if isinstance(attempt, bool) or not isinstance(attempt, int):
            raise TypeError("attempt must be an integer")
        if attempt >= self._poll_saturation_attempt:
            return self._nav_config.anchor_poll_max_interval_seconds
        exponent = max(0, int(attempt) - 1)
        return self._nav_config.anchor_poll_initial_interval_seconds * (2 ** exponent)

    def _calculate_poll_saturation_attempt(self) -> int:
        attempt = 1
        interval = self._nav_config.anchor_poll_initial_interval_seconds
        maximum = self._nav_config.anchor_poll_max_interval_seconds
        while interval < maximum:
            interval = min(maximum, interval * 2)
            attempt += 1
        return attempt

    def _has_conservative_budget(self, checkpoint: AnchorPollingCheckpoint, remaining_seconds: float) -> bool:
        remaining_rounds = max(
            1,
            self._nav_config.anchor_confirmation_count - checkpoint.confirmation_count,
        )
        round_budget = (
            len(self._nav_config.yahoo_base_urls) * self._nav_config.anchor_http_request_timeout_seconds
            + self._nav_config.anchor_pair_fetch_max_skew_seconds
        )
        between_rounds = max(
            self._nav_config.anchor_confirmation_interval_seconds,
            self._nav_config.anchor_poll_max_interval_seconds,
        )
        required_seconds = remaining_rounds * round_budget + max(0, remaining_rounds - 1) * between_rounds
        return remaining_seconds >= required_seconds

    def _budget_for(self, checkpoint: AnchorPollingCheckpoint) -> YahooCycleBudget:
        key = (checkpoint.cycle_id, checkpoint.deadline_utc)
        if self._active_budget is None or self._active_budget_key != key:
            self._active_budget = YahooCycleBudget.start(
                deadline_utc=checkpoint.deadline_utc,
                utc_clock=self._utc_clock,
                monotonic_clock=self._monotonic_clock,
            )
            self._active_budget_key = key
        return self._active_budget

    async def _fetch_pair(
        self,
        checkpoint: AnchorPollingCheckpoint,
        stock_symbol: str,
        etf_symbol: str,
        budget: YahooCycleBudget,
    ) -> list[Any]:
        async def fetch_leg(symbol: str):
            observation = await self._provider.fetch_close(
                symbol,
                checkpoint.target_session_date,
                checkpoint.deadline_utc,
                budget=budget,
            )
            if not isinstance(observation, YahooCloseObservation):
                return observation
            return _PairLegOutcome(
                observation=observation,
                completed_monotonic=self._monotonic_clock(),
            )

        tasks = [
            asyncio.create_task(fetch_leg(stock_symbol)),
            asyncio.create_task(fetch_leg(etf_symbol)),
        ]

        def cancel_sibling_after_leg_cancellation(completed: asyncio.Task) -> None:
            if completed.cancelled():
                for task in tasks:
                    if task is not completed and not task.done():
                        task.cancel()

        for task in tasks:
            task.add_done_callback(cancel_sibling_after_leg_cancellation)
        try:
            return await asyncio.gather(*tasks, return_exceptions=True)
        finally:
            pending = [task for task in tasks if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

    async def _run_checkpoint_callback(
        self,
        callback: Callable[[AnchorPollingCheckpoint], Any],
        checkpoint: AnchorPollingCheckpoint,
        budget: YahooCycleBudget,
    ) -> None:
        if inspect.iscoroutinefunction(callback):
            await budget.wait(callback(checkpoint), "checkpoint callback")
            return
        callback_result = await budget.wait(
            asyncio.to_thread(callback, checkpoint),
            "checkpoint callback",
        )
        if inspect.isawaitable(callback_result):
            await budget.wait(callback_result, "checkpoint callback awaitable")

    @staticmethod
    def _deadline_result(
        checkpoint: AnchorPollingCheckpoint,
        reason: str = "absolute Yahoo anchor deadline elapsed",
    ) -> AnchorAcquisitionResult:
        return AnchorAcquisitionResult(
            status=AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE,
            checkpoint=checkpoint,
            failure_reason=reason,
        )

    def _validate_exact_official_close(
        self,
        target_session_date: date,
        official_close_utc: datetime,
    ) -> None:
        try:
            expected_close = self._calendar.official_close_for_session(target_session_date)
        except (CalendarRangeError, TypeError, ValueError) as exception:
            raise CheckpointIntegrityError(
                "target session date is not a valid XNYS session"
            ) from exception
        if official_close_utc != expected_close:
            raise CheckpointIntegrityError(
                "official close does not match the exact XNYS session schedule"
            )

    def _validate_checkpoint_for_config(self, checkpoint: AnchorPollingCheckpoint) -> None:
        if not isinstance(checkpoint, AnchorPollingCheckpoint):
            raise TypeError("checkpoint must be an AnchorPollingCheckpoint")
        self._validate_exact_official_close(
            checkpoint.target_session_date,
            checkpoint.official_close_utc,
        )
        expected_deadline = checkpoint.official_close_utc + timedelta(
            seconds=self._nav_config.anchor_wait_timeout_seconds
        )
        if checkpoint.deadline_utc != expected_deadline:
            raise CheckpointIntegrityError("checkpoint deadline does not match the configured original window")
        if checkpoint.confirmation_count > self._nav_config.anchor_confirmation_count:
            raise CheckpointIntegrityError("checkpoint confirmation count exceeds the configured requirement")
        if checkpoint.confirmation_count > 0 and checkpoint.integrity_version != _ANCHOR_EVIDENCE_VERSION:
            raise CheckpointIntegrityError(
                "confirmed legacy checkpoint lacks the full recovery evidence trail"
            )
        if checkpoint.stock_received_at_utc is not None:
            skew = abs((checkpoint.stock_received_at_utc - checkpoint.etf_received_at_utc).total_seconds())
            if skew > self._nav_config.anchor_pair_fetch_max_skew_seconds:
                raise CheckpointIntegrityError("checkpoint paired receive skew exceeds the configured maximum")
        previous_received_at: datetime | None = None
        for record in checkpoint.confirmation_evidence:
            current_received_at = max(record.stock_received_at_utc, record.etf_received_at_utc)
            skew = abs((record.stock_received_at_utc - record.etf_received_at_utc).total_seconds())
            if skew > self._nav_config.anchor_pair_fetch_max_skew_seconds:
                raise CheckpointIntegrityError("checkpoint evidence exceeds the configured paired receive skew")
            if previous_received_at is not None:
                elapsed = (current_received_at - previous_received_at).total_seconds()
                if elapsed < self._nav_config.anchor_confirmation_interval_seconds:
                    raise CheckpointIntegrityError("checkpoint evidence violates the confirmation interval")
            previous_received_at = current_received_at

    def _validate_observation_window(
        self,
        observation: YahooCloseObservation,
        expected_symbol: str,
        checkpoint: AnchorPollingCheckpoint,
    ) -> None:
        self._validate_observation(
            observation,
            expected_symbol,
            checkpoint.target_session_date,
        )
        if not checkpoint.official_close_utc <= observation.received_at_utc < checkpoint.deadline_utc:
            raise ValueError("Yahoo observation receive time is outside the post-close acquisition window")

    @staticmethod
    def _validate_symbol(symbol: object, field_name: str) -> str:
        if not isinstance(symbol, str) or _SYMBOL_PATTERN.fullmatch(symbol) is None:
            raise ValueError(f"{field_name} is not a supported Yahoo symbol")
        return symbol

    @staticmethod
    def _validate_observation(
        observation: YahooCloseObservation,
        expected_symbol: str,
        target_session_date: date,
    ) -> None:
        if observation.symbol != expected_symbol:
            raise ValueError("Yahoo observation symbol does not match requested symbol")
        if observation.target_session_date != target_session_date:
            raise ValueError("Yahoo observation target session date does not match checkpoint")
