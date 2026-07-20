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


class CheckpointIntegrityError(ValueError):
    """Raised when untrusted checkpoint fields violate the F001 contract."""


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
    if not isinstance(value, Decimal):
        raise ValueError(f"{field_name} must be a Decimal")
    if not value.is_finite() or value <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return value


def _format_decimal(value: Decimal) -> str:
    _validate_positive_decimal(value, "decimal")
    rendered = format(value, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered


def _parse_canonical_decimal(value: object, field_name: str) -> Decimal:
    if not isinstance(value, str):
        raise CheckpointIntegrityError(f"{field_name} must be a canonical decimal string")
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


def _validate_hash(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a lowercase SHA-256 hash")
    return value


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
            return
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


@dataclass(frozen=True, slots=True)
class AnchorCandidate:
    cycle_id: str
    target_session_date: date
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
    finalized_at_utc: datetime
    deadline_utc: datetime
    stock_raw_response_hash: str
    etf_raw_response_hash: str
    evidence_hash: str

    def __post_init__(self) -> None:
        if not isinstance(self.target_session_date, date) or isinstance(self.target_session_date, datetime):
            raise TypeError("candidate target session date must be a date")
        expected_cycle_id = f"xnys-{self.target_session_date.isoformat()}"
        if self.cycle_id != expected_cycle_id:
            raise ValueError("candidate cycle ID does not match target session date")
        for symbol, field_name in ((self.stock_symbol, "stock symbol"), (self.etf_symbol, "ETF symbol")):
            if not isinstance(symbol, str) or _SYMBOL_PATTERN.fullmatch(symbol) is None:
                raise ValueError(f"{field_name} is not a supported Yahoo symbol")
        _validate_positive_decimal(self.stock_close, "stock close")
        _validate_positive_decimal(self.etf_close, "ETF close")
        for value, field_name in (
            (self.stock_bar_timestamp_utc, "stock bar timestamp"),
            (self.etf_bar_timestamp_utc, "ETF bar timestamp"),
            (self.stock_regular_market_time_utc, "stock regular market time"),
            (self.etf_regular_market_time_utc, "ETF regular market time"),
            (self.stock_received_at_utc, "stock received time"),
            (self.etf_received_at_utc, "ETF received time"),
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
        _validate_hash(self.evidence_hash, "evidence hash")

    def verify_evidence_hash(self) -> bool:
        expected = anchor_evidence_hash(
            cycle_id=self.cycle_id,
            stock_raw_response_hash=self.stock_raw_response_hash,
            etf_raw_response_hash=self.etf_raw_response_hash,
        )
        return self.evidence_hash == expected


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
    ):
        if not isinstance(nav_config, NavConfig):
            raise TypeError("nav_config must be a NavConfig")
        self._nav_config = nav_config
        self._utc_clock = utc_clock
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
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
            stock_close = stock.close
            etf_close = etf.close
            stock_hash = stock.raw_response_hash
            etf_hash = etf.raw_response_hash
            stock_received_at = stock.received_at_utc
            etf_received_at = etf.received_at_utc
        else:
            prior_received_at = max(checkpoint.stock_received_at_utc, checkpoint.etf_received_at_utc)
            confirmation_elapsed = (current_received_at - prior_received_at).total_seconds()
            if confirmation_elapsed >= self._nav_config.anchor_confirmation_interval_seconds:
                confirmation_count = min(
                    self._nav_config.anchor_confirmation_count,
                    checkpoint.confirmation_count + 1,
                )
                stock_close = stock.close
                etf_close = etf.close
                stock_hash = stock.raw_response_hash
                etf_hash = etf.raw_response_hash
                stock_received_at = stock.received_at_utc
                etf_received_at = etf.received_at_utc
            else:
                too_soon = True
                confirmation_count = checkpoint.confirmation_count
                stock_close = checkpoint.candidate_stock_close
                etf_close = checkpoint.candidate_etf_close
                stock_hash = checkpoint.candidate_stock_raw_response_hash
                etf_hash = checkpoint.candidate_etf_raw_response_hash
                stock_received_at = checkpoint.stock_received_at_utc
                etf_received_at = checkpoint.etf_received_at_utc

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
        )
        if (
            confirmation_count < self._nav_config.anchor_confirmation_count
            or round_time < minimum_finalize_at
        ):
            return AnchorAcquisitionResult(status=AnchorAcquisitionStatus.POLLING, checkpoint=updated)

        evidence_hash = anchor_evidence_hash(
            cycle_id=checkpoint.cycle_id,
            stock_raw_response_hash=stock.raw_response_hash,
            etf_raw_response_hash=etf.raw_response_hash,
        )
        candidate = AnchorCandidate(
            cycle_id=checkpoint.cycle_id,
            target_session_date=checkpoint.target_session_date,
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
            finalized_at_utc=max(round_time, stock.received_at_utc, etf.received_at_utc),
            deadline_utc=checkpoint.deadline_utc,
            stock_raw_response_hash=stock.raw_response_hash,
            etf_raw_response_hash=etf.raw_response_hash,
            evidence_hash=evidence_hash,
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
        if checkpoint.stock_received_at_utc is not None:
            skew = abs((checkpoint.stock_received_at_utc - checkpoint.etf_received_at_utc).total_seconds())
            if skew > self._nav_config.anchor_pair_fetch_max_skew_seconds:
                raise CheckpointIntegrityError("checkpoint paired receive skew exceeds the configured maximum")

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
