from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal, NamedTuple, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator
from sqlalchemy import text
from sqlalchemy.engine import Connection
from sqlalchemy.exc import IntegrityError

from hummingbot.model.sql_connection_manager import SQLConnectionManager
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import (
    CanonicalNonNegativeDecimal,
    CanonicalPositiveDecimal,
    CanonicalUtcInstant,
    CanonicalWireModel,
    LeveragedEtfPairExecutorSnapshotV1,
    LeveragedEtfPairExecutorStateV1,
    LeveragedEtfPairOperation,
    LeveragedEtfPairState,
    SchemaVersionV1,
    Sha256Hex,
    StableIdentifier,
    TradingPair,
)

_CANONICAL_UTC_PATTERN = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])" r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.[0-9]{6}Z$"
)
_SESSION_DATE_PATTERN = re.compile(r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])$")
_SYMBOL_PATTERN = re.compile(r"^[A-Z0-9.^=-]+$")
_TERMINAL_EXECUTOR_STATES = frozenset(
    {
        LeveragedEtfPairState.COMPLETED,
        LeveragedEtfPairState.ABORTED_NO_FILL,
        LeveragedEtfPairState.FAILED_SAFE,
    }
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_utc_text(value: str, field_name: str) -> str:
    if _CANONICAL_UTC_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be canonical UTC")
    return value


def _verify_canonical_json(payload_json: str, payload_hash: str, label: str) -> Any:
    def reject_non_finite_constant(value: str):
        raise ValueError(f"{label} contains non-finite JSON constant {value}")

    try:
        value = json.loads(payload_json, parse_constant=reject_non_finite_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exception:
        raise ValueError(f"{label} contains malformed JSON") from exception
    if _canonical_json(value) != payload_json:
        raise ValueError(f"{label} JSON is not canonical")
    if _sha256_text(payload_json) != payload_hash:
        raise ValueError(f"{label} hash mismatch")
    return value


class JournalIntegrityError(RuntimeError):
    pass


class JournalConflictError(RuntimeError):
    pass


class AnchorRevisionConflict(RuntimeError):
    pass


class AnchorIntegrityError(RuntimeError):
    pass


class CanonicalOpaquePayload(BaseModel):
    """Canonical, versioned payload whose contents remain repository-opaque."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: StrictInt = Field(ge=1)
    kind: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    payload_json: str
    payload_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_canonical_payload(self) -> CanonicalOpaquePayload:
        value = _verify_canonical_json(self.payload_json, self.payload_hash, f"{self.kind} payload")
        if not isinstance(value, Mapping):
            raise ValueError("opaque payload must be a JSON object")
        embedded_version = value.get("schema_version")
        if (
            not isinstance(embedded_version, int)
            or isinstance(embedded_version, bool)
            or embedded_version != self.schema_version
        ):
            raise ValueError("wrapper and embedded schema_version must match exactly")
        if "kind" in value and value["kind"] != self.kind:
            raise ValueError("wrapper and embedded payload kind must match exactly")
        return self

    @classmethod
    def from_value(cls, schema_version: int, kind: str, value: Any) -> CanonicalOpaquePayload:
        payload_json = _canonical_json(value)
        return cls(
            schema_version=schema_version,
            kind=kind,
            payload_json=payload_json,
            payload_hash=_sha256_text(payload_json),
        )

    @classmethod
    def from_canonical_json(
        cls,
        schema_version: int,
        kind: str,
        payload_json: str,
        payload_hash: str,
    ) -> CanonicalOpaquePayload:
        return cls(
            schema_version=schema_version,
            kind=kind,
            payload_json=payload_json,
            payload_hash=payload_hash,
        )

    def value(self) -> Any:
        return _verify_canonical_json(self.payload_json, self.payload_hash, f"{self.kind} payload")


class JournalEventType(str, Enum):
    PREPARED = "PREPARED"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    REJECTED = "REJECTED"
    SUBMIT_UNKNOWN = "SUBMIT_UNKNOWN"
    ORDER_CREATED = "ORDER_CREATED"
    FILL = "FILL"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCEL_CONFIRMED = "CANCEL_CONFIRMED"
    HEDGE_REQUESTED = "HEDGE_REQUESTED"
    HEDGE_CONFIRMED = "HEDGE_CONFIRMED"
    ROLLBACK_REQUESTED = "ROLLBACK_REQUESTED"
    ROLLBACK_CONFIRMED = "ROLLBACK_CONFIRMED"
    RECONCILIATION = "RECONCILIATION"
    STATE_TRANSITION = "STATE_TRANSITION"


class JournalSideEffect(str, Enum):
    ETF_MAKER = "ETF_MAKER"
    STOCK_HEDGE = "STOCK_HEDGE"
    ETF_ROLLBACK = "ETF_ROLLBACK"
    CANCEL = "CANCEL"


class ReconciliationOutcome(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    NOT_FOUND = "NOT_FOUND"
    UNKNOWN = "UNKNOWN"
    CONSISTENT_NO_FILL = "CONSISTENT_NO_FILL"
    CONFLICT = "CONFLICT"


def _canonical_decimal_text(value: Decimal) -> str:
    if value == 0:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


class SideEffectIdentityV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    executor_id: StableIdentifier
    operation: LeveragedEtfPairOperation
    action: JournalSideEffect
    leg: Literal["ETF", "STOCK"]
    logical_quantity: CanonicalPositiveDecimal
    order_quantity: CanonicalPositiveDecimal
    attempt: StrictInt = Field(ge=1)
    intent_id: StableIdentifier
    idempotency_key: StableIdentifier
    connector_name: StableIdentifier
    trading_pair: TradingPair
    client_order_id: StableIdentifier
    deadline_utc: CanonicalUtcInstant

    @model_validator(mode="after")
    def validate_side_effect_identity(self) -> SideEffectIdentityV1:
        required_leg = {
            JournalSideEffect.ETF_MAKER: "ETF",
            JournalSideEffect.STOCK_HEDGE: "STOCK",
            JournalSideEffect.ETF_ROLLBACK: "ETF",
            JournalSideEffect.CANCEL: "ETF",
        }.get(self.action)
        if required_leg is not None and self.leg != required_leg:
            raise ValueError(f"{self.action.value} requires the {required_leg} leg")
        logical_quantity = _canonical_decimal_text(self.logical_quantity)
        expected_key = f"{self.executor_id}:{self.operation.value}:{self.leg}:" f"{logical_quantity}:{self.attempt}"
        if self.idempotency_key != expected_key:
            raise ValueError("idempotency_key does not match the canonical side-effect identity")
        if self.order_quantity > self.logical_quantity:
            raise ValueError("order_quantity cannot exceed logical_quantity")
        return self

    @property
    def exposure_increasing(self) -> bool:
        return self.operation in {
            LeveragedEtfPairOperation.OPEN,
            LeveragedEtfPairOperation.ADD,
        } and self.action in {
            JournalSideEffect.ETF_MAKER,
            JournalSideEffect.STOCK_HEDGE,
        }


class _IdentityJournalPayloadV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    identity: SideEffectIdentityV1
    next_state: LeveragedEtfPairExecutorStateV1


class PreparedJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["PREPARED"] = "PREPARED"


class AcknowledgedJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["ACKNOWLEDGED"] = "ACKNOWLEDGED"


class RejectedJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["REJECTED"] = "REJECTED"
    reason: str = Field(min_length=1, max_length=1024)


class SubmitUnknownJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["SUBMIT_UNKNOWN"] = "SUBMIT_UNKNOWN"
    uncertainty_started_at_utc: CanonicalUtcInstant


class OrderCreatedJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["ORDER_CREATED"] = "ORDER_CREATED"
    exchange_order_id: StableIdentifier


class FillJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["FILL"] = "FILL"
    exchange_order_id: Optional[StableIdentifier]
    exchange_trade_id: StableIdentifier
    price: CanonicalPositiveDecimal
    fill_quantity: CanonicalPositiveDecimal
    cumulative_filled_quantity: CanonicalPositiveDecimal
    outcome: Literal["PARTIAL", "FILLED"]
    terminal: bool

    @model_validator(mode="after")
    def validate_fill_outcome(self) -> FillJournalPayloadV1:
        if self.fill_quantity > self.cumulative_filled_quantity:
            raise ValueError("fill_quantity cannot exceed cumulative_filled_quantity")
        if self.terminal != (self.outcome == "FILLED"):
            raise ValueError("fill terminal flag must be derived from outcome")
        return self


class CancelJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["CANCEL"] = "CANCEL"
    phase: Literal["REQUESTED", "CONFIRMED"]
    final_cumulative_filled_quantity: CanonicalNonNegativeDecimal

    @model_validator(mode="after")
    def validate_cancel_action(self) -> CancelJournalPayloadV1:
        if self.identity.action != JournalSideEffect.CANCEL:
            raise ValueError("cancel payload requires CANCEL side-effect identity")
        return self


class HedgeJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["HEDGE"] = "HEDGE"
    phase: Literal["REQUESTED", "CONFIRMED"]

    @model_validator(mode="after")
    def validate_hedge_action(self) -> HedgeJournalPayloadV1:
        if self.identity.action != JournalSideEffect.STOCK_HEDGE:
            raise ValueError("hedge payload requires STOCK_HEDGE side-effect identity")
        return self


class RollbackJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["ROLLBACK"] = "ROLLBACK"
    phase: Literal["REQUESTED", "CONFIRMED"]

    @model_validator(mode="after")
    def validate_rollback_action(self) -> RollbackJournalPayloadV1:
        if self.identity.action != JournalSideEffect.ETF_ROLLBACK:
            raise ValueError("rollback payload requires ETF_ROLLBACK side-effect identity")
        return self


class ReconciliationJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["RECONCILIATION"] = "RECONCILIATION"
    outcome: ReconciliationOutcome
    terminal: bool

    @model_validator(mode="after")
    def validate_reconciliation_outcome(self) -> ReconciliationJournalPayloadV1:
        terminal_outcomes = {
            ReconciliationOutcome.FILLED,
            ReconciliationOutcome.CANCELED,
            ReconciliationOutcome.EXPIRED,
            ReconciliationOutcome.REJECTED,
            ReconciliationOutcome.CONSISTENT_NO_FILL,
        }
        if self.terminal != (self.outcome in terminal_outcomes):
            raise ValueError("reconciliation terminal flag must be derived from outcome")
        return self


class StateTransitionJournalPayloadV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    kind: Literal["STATE_TRANSITION"] = "STATE_TRANSITION"
    reason: str = Field(min_length=1, max_length=1024)
    next_state: LeveragedEtfPairExecutorStateV1


JournalPayloadV1 = Annotated[
    Union[
        PreparedJournalPayloadV1,
        AcknowledgedJournalPayloadV1,
        RejectedJournalPayloadV1,
        SubmitUnknownJournalPayloadV1,
        OrderCreatedJournalPayloadV1,
        FillJournalPayloadV1,
        CancelJournalPayloadV1,
        HedgeJournalPayloadV1,
        RollbackJournalPayloadV1,
        ReconciliationJournalPayloadV1,
        StateTransitionJournalPayloadV1,
    ],
    Field(discriminator="kind"),
]


class JournalEventV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    event_id: StableIdentifier
    event_type: JournalEventType
    intent_id: Optional[StableIdentifier] = None
    idempotency_key: Optional[StableIdentifier] = None
    connector_name: Optional[StableIdentifier] = None
    trading_pair: Optional[TradingPair] = None
    client_order_id: Optional[StableIdentifier] = None
    exchange_order_id: Optional[StableIdentifier] = None
    exchange_trade_id: Optional[StableIdentifier] = None
    payload: JournalPayloadV1
    created_at_utc: CanonicalUtcInstant

    @model_validator(mode="before")
    @classmethod
    def unwrap_canonical_payload(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload = value.get("payload")
        if isinstance(payload, CanonicalOpaquePayload):
            value = {**value, "payload": payload.value()}
        elif isinstance(payload, Mapping) and {"payload_json", "payload_hash", "kind"} <= set(payload):
            value = {**value, "payload": CanonicalOpaquePayload.model_validate(payload).value()}
        return value

    @model_validator(mode="after")
    def validate_identity_fields(self) -> JournalEventV1:
        expected_payload_types = {
            JournalEventType.PREPARED: PreparedJournalPayloadV1,
            JournalEventType.ACKNOWLEDGED: AcknowledgedJournalPayloadV1,
            JournalEventType.REJECTED: RejectedJournalPayloadV1,
            JournalEventType.SUBMIT_UNKNOWN: SubmitUnknownJournalPayloadV1,
            JournalEventType.ORDER_CREATED: OrderCreatedJournalPayloadV1,
            JournalEventType.FILL: FillJournalPayloadV1,
            JournalEventType.CANCEL_REQUESTED: CancelJournalPayloadV1,
            JournalEventType.CANCEL_CONFIRMED: CancelJournalPayloadV1,
            JournalEventType.HEDGE_REQUESTED: HedgeJournalPayloadV1,
            JournalEventType.HEDGE_CONFIRMED: HedgeJournalPayloadV1,
            JournalEventType.ROLLBACK_REQUESTED: RollbackJournalPayloadV1,
            JournalEventType.ROLLBACK_CONFIRMED: RollbackJournalPayloadV1,
            JournalEventType.RECONCILIATION: ReconciliationJournalPayloadV1,
            JournalEventType.STATE_TRANSITION: StateTransitionJournalPayloadV1,
        }
        expected_payload_type = expected_payload_types[self.event_type]
        if not isinstance(self.payload, expected_payload_type):
            raise ValueError(f"{self.event_type.value} has a mismatched typed payload")

        identity = getattr(self.payload, "identity", None)
        if identity is not None:
            comparisons = {
                "intent_id": self.intent_id,
                "idempotency_key": self.idempotency_key,
                "connector_name": self.connector_name,
                "trading_pair": self.trading_pair,
                "client_order_id": self.client_order_id,
            }
            for field_name, outer_value in comparisons.items():
                if outer_value != getattr(identity, field_name):
                    raise ValueError(f"event {field_name} disagrees with side-effect identity")

        expected_phase = {
            JournalEventType.CANCEL_REQUESTED: "REQUESTED",
            JournalEventType.CANCEL_CONFIRMED: "CONFIRMED",
            JournalEventType.HEDGE_REQUESTED: "REQUESTED",
            JournalEventType.HEDGE_CONFIRMED: "CONFIRMED",
            JournalEventType.ROLLBACK_REQUESTED: "REQUESTED",
            JournalEventType.ROLLBACK_CONFIRMED: "CONFIRMED",
        }.get(self.event_type)
        if expected_phase is not None and getattr(self.payload, "phase", None) != expected_phase:
            raise ValueError(f"{self.event_type.value} payload phase is invalid")

        if isinstance(self.payload, OrderCreatedJournalPayloadV1):
            if self.exchange_order_id != self.payload.exchange_order_id:
                raise ValueError("order-created exchange_order_id disagrees with payload")
        if isinstance(self.payload, FillJournalPayloadV1):
            if self.exchange_order_id != self.payload.exchange_order_id:
                raise ValueError("fill exchange_order_id disagrees with payload")
            if self.exchange_trade_id != self.payload.exchange_trade_id:
                raise ValueError("fill exchange_trade_id disagrees with payload")
        elif self.exchange_trade_id is not None:
            raise ValueError("exchange_trade_id is only valid for typed FILL events")
        return self


class CommittedJournalEventV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    executor_id: StableIdentifier
    sequence: StrictInt = Field(ge=1)
    event: JournalEventV1
    mutation_hash: Sha256Hex


class IncompleteIntentV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    executor_id: StableIdentifier
    intent_id: StableIdentifier
    status: JournalEventType
    last_sequence: StrictInt = Field(ge=1)
    prepared_event: JournalEventV1


class ReservationIdentityPayloadV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    kind: Literal["LEVERAGE_RESERVATION"] = "LEVERAGE_RESERVATION"
    reservation_key: StableIdentifier
    executor_id: StableIdentifier
    connector_name: StableIdentifier
    trading_pair: TradingPair
    leg: Literal["ETF", "STOCK"]
    logical_quantity: CanonicalPositiveDecimal


class StrategyReservationV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    reservation_id: StableIdentifier
    executor_id: StableIdentifier
    reservation_key: StableIdentifier
    connector_name: StableIdentifier
    trading_pair: TradingPair
    leg: Literal["ETF", "STOCK"]
    quantity: CanonicalPositiveDecimal
    leverage: StrictInt = Field(ge=1)
    notional_cap: CanonicalPositiveDecimal
    payload: ReservationIdentityPayloadV1
    created_at_utc: CanonicalUtcInstant
    updated_at_utc: CanonicalUtcInstant
    released_at_utc: Optional[CanonicalUtcInstant]

    @model_validator(mode="before")
    @classmethod
    def unwrap_reservation_payload(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        payload = value.get("payload")
        if isinstance(payload, CanonicalOpaquePayload):
            value = {**value, "payload": payload.value()}
        elif isinstance(payload, Mapping) and {"payload_json", "payload_hash", "kind"} <= set(payload):
            value = {**value, "payload": CanonicalOpaquePayload.model_validate(payload).value()}
        return value

    @model_validator(mode="after")
    def validate_reservation_identity(self) -> StrategyReservationV1:
        comparisons = {
            "reservation_key": self.reservation_key,
            "executor_id": self.executor_id,
            "connector_name": self.connector_name,
            "trading_pair": self.trading_pair,
            "leg": self.leg,
            "logical_quantity": self.quantity,
        }
        for field_name, expected in comparisons.items():
            if getattr(self.payload, field_name) != expected:
                raise ValueError(f"reservation payload {field_name} disagrees with row identity")
        if self.updated_at_utc < self.created_at_utc:
            raise ValueError("reservation updated_at_utc precedes created_at_utc")
        if self.released_at_utc is not None:
            if self.released_at_utc < self.created_at_utc:
                raise ValueError("reservation released_at_utc precedes created_at_utc")
            if self.updated_at_utc != self.released_at_utc:
                raise ValueError("released reservation updated_at_utc must equal released_at_utc")
        return self


class AnchorPollingCheckpointV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    cycle_id: StableIdentifier
    target_session_date: str
    official_close_utc: CanonicalUtcInstant
    deadline_utc: CanonicalUtcInstant
    attempt: StrictInt = Field(ge=0)
    next_poll_utc: CanonicalUtcInstant
    confirmation_count: StrictInt = Field(ge=0)
    candidate_stock_close: Optional[CanonicalPositiveDecimal]
    candidate_etf_close: Optional[CanonicalPositiveDecimal]
    candidate_stock_raw_response_hash: Optional[Sha256Hex]
    candidate_etf_raw_response_hash: Optional[Sha256Hex]
    stock_received_at_utc: Optional[CanonicalUtcInstant]
    etf_received_at_utc: Optional[CanonicalUtcInstant]
    revision: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def validate_checkpoint(self) -> AnchorPollingCheckpointV1:
        if _SESSION_DATE_PATTERN.fullmatch(self.target_session_date) is None:
            raise ValueError("target_session_date is not canonical")
        paired = (
            self.candidate_stock_close,
            self.candidate_etf_close,
            self.candidate_stock_raw_response_hash,
            self.candidate_etf_raw_response_hash,
            self.stock_received_at_utc,
            self.etf_received_at_utc,
        )
        if any(value is None for value in paired) and any(value is not None for value in paired):
            raise ValueError("anchor candidate fields must be paired")
        return self


class AnchorRecordV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    cycle_id: StableIdentifier
    yahoo_stock_symbol: str
    yahoo_etf_symbol: str
    binance_stock_trading_pair: TradingPair
    binance_etf_trading_pair: TradingPair
    target_session_date: str
    s0: CanonicalPositiveDecimal
    l0: CanonicalPositiveDecimal
    h: CanonicalPositiveDecimal
    stock_bar_timestamp_utc: CanonicalUtcInstant
    etf_bar_timestamp_utc: CanonicalUtcInstant
    stock_regular_market_time_utc: CanonicalUtcInstant
    etf_regular_market_time_utc: CanonicalUtcInstant
    stock_received_at_utc: CanonicalUtcInstant
    etf_received_at_utc: CanonicalUtcInstant
    finalized_at_utc: CanonicalUtcInstant
    deadline_utc: CanonicalUtcInstant
    stock_raw_response_hash: Sha256Hex
    etf_raw_response_hash: Sha256Hex
    evidence_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_anchor_record(self) -> AnchorRecordV1:
        if _SESSION_DATE_PATTERN.fullmatch(self.target_session_date) is None:
            raise ValueError("target_session_date is not canonical")
        if _SYMBOL_PATTERN.fullmatch(self.yahoo_stock_symbol) is None:
            raise ValueError("yahoo_stock_symbol is invalid")
        if _SYMBOL_PATTERN.fullmatch(self.yahoo_etf_symbol) is None:
            raise ValueError("yahoo_etf_symbol is invalid")
        return self


AnchorStateV1 = Union[AnchorPollingCheckpointV1, AnchorRecordV1]


class OpaqueAnchorCheckpointV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    cycle_id: StableIdentifier
    target_session_date: str
    official_close_utc: CanonicalUtcInstant
    deadline_utc: CanonicalUtcInstant
    revision: StrictInt = Field(ge=1)
    payload: CanonicalOpaquePayload

    @model_validator(mode="after")
    def validate_metadata(self) -> OpaqueAnchorCheckpointV1:
        if _SESSION_DATE_PATTERN.fullmatch(self.target_session_date) is None:
            raise ValueError("target_session_date is not canonical")
        if self.payload.kind != "ANCHOR_CHECKPOINT":
            raise ValueError("checkpoint payload kind must be ANCHOR_CHECKPOINT")
        return self


class OpaqueAnchorFinalizedV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    cycle_id: StableIdentifier
    target_session_date: str
    official_close_utc: Optional[CanonicalUtcInstant]
    deadline_utc: CanonicalUtcInstant
    revision: StrictInt = Field(ge=1)
    evidence_hash: Sha256Hex
    payload: CanonicalOpaquePayload

    @model_validator(mode="after")
    def validate_metadata(self) -> OpaqueAnchorFinalizedV1:
        if _SESSION_DATE_PATTERN.fullmatch(self.target_session_date) is None:
            raise ValueError("target_session_date is not canonical")
        if self.payload.kind != "ANCHOR_RECORD":
            raise ValueError("finalized payload kind must be ANCHOR_RECORD")
        return self


OpaqueAnchorStateV1 = Union[OpaqueAnchorCheckpointV1, OpaqueAnchorFinalizedV1]


class AnchorRevisionObservationV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    cycle_id: StableIdentifier
    evidence_hash: Sha256Hex
    observed_at_utc: str

    @model_validator(mode="after")
    def validate_observed_time(self) -> AnchorRevisionObservationV1:
        _validate_utc_text(self.observed_at_utc, "observed_at_utc")
        return self


class _DecodedJournalMutation(NamedTuple):
    committed: CommittedJournalEventV1
    snapshot_before: LeveragedEtfPairExecutorSnapshotV1
    snapshot_after: LeveragedEtfPairExecutorSnapshotV1


class _TransactionalRepository:
    def __init__(
        self,
        sql_manager: SQLConnectionManager,
        before_commit: Optional[Callable[[Connection], None]] = None,
        recovery_read_hook: Optional[Callable[[Connection], None]] = None,
    ):
        self._sql_manager = sql_manager
        self._before_commit = before_commit
        self._recovery_read_hook = recovery_read_hook

    @property
    def sql_manager(self) -> SQLConnectionManager:
        return self._sql_manager

    def _write(self, operation: Callable[[Connection], Any]) -> Any:
        connection = self._sql_manager.engine.connect()
        transaction = connection.begin()
        try:
            connection.exec_driver_sql("BEGIN IMMEDIATE")
            result = operation(connection)
            if self._before_commit is not None:
                self._before_commit(connection)
            transaction.commit()
            return result
        except BaseException:
            if transaction.is_active:
                transaction.rollback()
            raise
        finally:
            connection.close()

    def _read_transaction(self, operation: Callable[[Connection], Any]) -> Any:
        connection = self._sql_manager.engine.connect()
        transaction = connection.begin()
        try:
            connection.exec_driver_sql("BEGIN")
            result = operation(connection)
            transaction.commit()
            return result
        except BaseException:
            if transaction.is_active:
                transaction.rollback()
            raise
        finally:
            connection.close()


class LeveragedEtfJournalRepository(_TransactionalRepository):
    """SQLite journal whose committed events are the sole snapshot reducer input."""

    _SNAPSHOT_SELECT = """
        SELECT executor_id, controller_id, pair_id, nav_cycle_id, schema_version,
               state, snapshot_json, snapshot_hash, last_journal_sequence,
               created_at_utc, updated_at_utc
        FROM LeveragedEtfExecutorSnapshot
        WHERE executor_id = :executor_id
    """
    _EVENT_SELECT = """
        SELECT event_id, executor_id, sequence, event_type, intent_id,
               idempotency_key, connector_name, trading_pair, client_order_id,
               exchange_order_id, exchange_trade_id, payload_json, payload_hash,
               created_at_utc
        FROM LeveragedEtfJournalEvent
    """
    _IMMUTABLE_SNAPSHOT_FIELDS = (
        "schema_version",
        "executor_id",
        "controller_id",
        "pair_id",
        "nav_cycle_id",
        "operation",
        "direction",
        "etf_connector_name",
        "etf_trading_pair",
        "stock_connector_name",
        "stock_trading_pair",
        "s0",
        "l0",
        "h",
        "created_raw_bp",
        "created_net_bp",
        "target_gross_notional",
        "etf_target_quantity",
        "stock_target_quantity",
        "config_hash",
        "created_at_utc",
    )
    _ALWAYS_TERMINAL_INTENT_EVENTS = frozenset(
        {
            JournalEventType.REJECTED,
            JournalEventType.CANCEL_CONFIRMED,
            JournalEventType.HEDGE_CONFIRMED,
            JournalEventType.ROLLBACK_CONFIRMED,
        }
    )
    _REQUIRES_PREPARED_INTENT = frozenset(
        {
            JournalEventType.ACKNOWLEDGED,
            JournalEventType.REJECTED,
            JournalEventType.SUBMIT_UNKNOWN,
            JournalEventType.ORDER_CREATED,
            JournalEventType.FILL,
            JournalEventType.CANCEL_REQUESTED,
            JournalEventType.CANCEL_CONFIRMED,
            JournalEventType.HEDGE_REQUESTED,
            JournalEventType.HEDGE_CONFIRMED,
            JournalEventType.ROLLBACK_REQUESTED,
            JournalEventType.ROLLBACK_CONFIRMED,
        }
    )

    @staticmethod
    def _decode_snapshot_row(row: Mapping[str, Any]) -> LeveragedEtfPairExecutorSnapshotV1:
        try:
            payload = _verify_canonical_json(row["snapshot_json"], row["snapshot_hash"], "executor snapshot")
            snapshot = LeveragedEtfPairExecutorSnapshotV1.model_validate(payload)
        except Exception as exception:
            if isinstance(exception, JournalIntegrityError):
                raise
            raise JournalIntegrityError(f"executor snapshot integrity failure: {exception}") from exception
        serialized = snapshot.model_dump(mode="json")
        expected_columns = {
            "executor_id": serialized["executor_id"],
            "controller_id": serialized["controller_id"],
            "pair_id": serialized["pair_id"],
            "nav_cycle_id": serialized["nav_cycle_id"],
            "schema_version": serialized["schema_version"],
            "state": serialized["state"],
            "last_journal_sequence": serialized["last_journal_sequence"],
            "created_at_utc": serialized["created_at_utc"],
            "updated_at_utc": serialized["updated_at_utc"],
        }
        for column_name, expected in expected_columns.items():
            if row[column_name] != expected:
                raise JournalIntegrityError(f"executor snapshot column {column_name} disagrees with payload")
        return snapshot

    @classmethod
    def _load_snapshot_connection(
        cls,
        connection: Connection,
        executor_id: str,
    ) -> Optional[LeveragedEtfPairExecutorSnapshotV1]:
        row = connection.execute(text(cls._SNAPSHOT_SELECT), {"executor_id": executor_id}).mappings().one_or_none()
        return None if row is None else cls._decode_snapshot_row(row)

    @staticmethod
    def _event_terminal(event: JournalEventV1) -> bool:
        if event.event_type in LeveragedEtfJournalRepository._ALWAYS_TERMINAL_INTENT_EVENTS:
            return True
        return (
            isinstance(event.payload, (FillJournalPayloadV1, ReconciliationJournalPayloadV1)) and event.payload.terminal
        )

    @classmethod
    def _decode_event_row(cls, row: Mapping[str, Any]) -> _DecodedJournalMutation:
        try:
            mutation = _verify_canonical_json(row["payload_json"], row["payload_hash"], "journal mutation")
            if not isinstance(mutation, dict) or mutation.get("schema_version") != 1:
                raise ValueError("journal mutation envelope is invalid")
            event = JournalEventV1.model_validate(mutation["event"])
            snapshot_before = LeveragedEtfPairExecutorSnapshotV1.model_validate(mutation["snapshot_before"])
            snapshot_after = LeveragedEtfPairExecutorSnapshotV1.model_validate(mutation["snapshot_after"])
            if snapshot_before.canonical_sha256() != mutation["snapshot_before_hash"]:
                raise ValueError("snapshot-before hash mismatch")
            if snapshot_after.canonical_sha256() != mutation["snapshot_after_hash"]:
                raise ValueError("snapshot-after hash mismatch")
        except Exception as exception:
            raise JournalIntegrityError(f"journal event hash or payload integrity failure: {exception}") from exception

        event_json = event.model_dump(mode="json")
        expected_columns = {
            "event_id": event_json["event_id"],
            "event_type": event_json["event_type"],
            "intent_id": event_json["intent_id"],
            "idempotency_key": event_json["idempotency_key"],
            "connector_name": event_json["connector_name"],
            "trading_pair": event_json["trading_pair"],
            "client_order_id": event_json["client_order_id"],
            "exchange_order_id": event_json["exchange_order_id"],
            "exchange_trade_id": event_json["exchange_trade_id"],
            "created_at_utc": event_json["created_at_utc"],
        }
        for column_name, expected in expected_columns.items():
            if row[column_name] != expected:
                raise JournalIntegrityError(f"journal column {column_name} disagrees with payload")
        if row["executor_id"] != snapshot_before.executor_id or row["executor_id"] != snapshot_after.executor_id:
            raise JournalIntegrityError("journal executor ID disagrees with snapshot chain")
        if row["sequence"] != snapshot_after.last_journal_sequence:
            raise JournalIntegrityError("journal sequence disagrees with reduced snapshot")
        return _DecodedJournalMutation(
            committed=CommittedJournalEventV1(
                executor_id=row["executor_id"],
                sequence=row["sequence"],
                event=event,
                mutation_hash=row["payload_hash"],
            ),
            snapshot_before=snapshot_before,
            snapshot_after=snapshot_after,
        )

    @classmethod
    def _event_rows(cls, connection: Connection, executor_id: Optional[str] = None):
        statement = cls._EVENT_SELECT
        parameters = {}
        if executor_id is not None:
            statement += " WHERE executor_id = :executor_id"
            parameters["executor_id"] = executor_id
        statement += " ORDER BY executor_id, sequence"
        return connection.execute(text(statement), parameters).mappings().all()

    @classmethod
    def _decoded_events(
        cls,
        connection: Connection,
        executor_id: str,
    ) -> Tuple[_DecodedJournalMutation, ...]:
        return tuple(cls._decode_event_row(row) for row in cls._event_rows(connection, executor_id))

    @classmethod
    def _replay_decoded(
        cls,
        persisted: LeveragedEtfPairExecutorSnapshotV1,
        decoded: Tuple[_DecodedJournalMutation, ...],
    ) -> LeveragedEtfPairExecutorSnapshotV1:
        if not decoded:
            if persisted.last_journal_sequence != 0:
                raise JournalIntegrityError("persisted snapshot sequence has no journal history")
            return persisted

        reduced = decoded[0].snapshot_before
        if reduced.executor_id != persisted.executor_id:
            raise JournalIntegrityError("journal replay begins with a foreign executor snapshot")
        if reduced.last_journal_sequence != 0:
            raise JournalIntegrityError("journal replay does not begin at sequence zero")
        prior: list[_DecodedJournalMutation] = []
        for expected_sequence, mutation in enumerate(decoded, start=1):
            if mutation.committed.executor_id != persisted.executor_id:
                raise JournalIntegrityError("journal event has a foreign executor owner")
            if mutation.committed.sequence != expected_sequence:
                raise JournalIntegrityError("journal sequence is not contiguous")
            if mutation.snapshot_before != reduced:
                raise JournalIntegrityError("journal snapshot hash chain diverges")
            cls._validate_intent_transition(mutation.committed.event, tuple(prior))
            cls._validate_snapshot_transition(
                mutation.snapshot_before,
                mutation.committed.event,
                mutation.snapshot_after,
            )
            reduced = mutation.snapshot_after
            prior.append(mutation)
        if reduced != persisted:
            raise JournalIntegrityError("persisted snapshot disagrees with journal replay")
        return reduced

    def _verified_recovery_state(
        self,
        connection: Connection,
        executor_id: Optional[str] = None,
    ) -> dict[str, Tuple[LeveragedEtfPairExecutorSnapshotV1, Tuple[_DecodedJournalMutation, ...]]]:
        if executor_id is None:
            snapshot_statement = self._SNAPSHOT_SELECT.replace(
                "WHERE executor_id = :executor_id",
                "ORDER BY executor_id",
            )
            snapshot_parameters = {}
        else:
            snapshot_statement = self._SNAPSHOT_SELECT
            snapshot_parameters = {"executor_id": executor_id}
        snapshot_rows = connection.execute(text(snapshot_statement), snapshot_parameters).mappings().all()
        snapshots = {row["executor_id"]: self._decode_snapshot_row(row) for row in snapshot_rows}

        if self._recovery_read_hook is not None:
            self._recovery_read_hook(connection)

        event_rows = self._event_rows(connection, executor_id)
        by_executor: dict[str, list[_DecodedJournalMutation]] = {}
        for row in event_rows:
            mutation = self._decode_event_row(row)
            by_executor.setdefault(mutation.committed.executor_id, []).append(mutation)
        orphan_executor_ids = set(by_executor) - set(snapshots)
        if orphan_executor_ids:
            owner = sorted(orphan_executor_ids)[0]
            raise JournalIntegrityError(f"orphan journal events have no executor snapshot: {owner}")

        verified = {}
        for current_executor_id, snapshot in snapshots.items():
            decoded = tuple(by_executor.get(current_executor_id, ()))
            self._replay_decoded(snapshot, decoded)
            verified[current_executor_id] = (snapshot, decoded)
        return verified

    @staticmethod
    def _logical_exposure_key(event: JournalEventV1) -> Optional[Tuple[str, str, str]]:
        if event.event_type != JournalEventType.PREPARED or not isinstance(event.payload, PreparedJournalPayloadV1):
            return None
        identity = event.payload.identity
        if not identity.exposure_increasing:
            return None
        return identity.operation.value, identity.leg, _canonical_decimal_text(identity.logical_quantity)

    @classmethod
    def _incomplete_from_decoded(
        cls,
        executor_id: str,
        decoded: Tuple[_DecodedJournalMutation, ...],
    ) -> Tuple[IncompleteIntentV1, ...]:
        prepared = {}
        latest = {}
        for mutation in decoded:
            event = mutation.committed.event
            if event.intent_id is None:
                continue
            if event.event_type == JournalEventType.PREPARED:
                prepared[event.intent_id] = event
            latest[event.intent_id] = mutation.committed
        incomplete = []
        for intent_id, committed in latest.items():
            prepared_event = prepared.get(intent_id)
            if prepared_event is None:
                raise JournalIntegrityError(f"intent {intent_id} has no PREPARED event")
            if not cls._event_terminal(committed.event):
                incomplete.append(
                    IncompleteIntentV1(
                        executor_id=executor_id,
                        intent_id=intent_id,
                        status=committed.event.event_type,
                        last_sequence=committed.sequence,
                        prepared_event=prepared_event,
                    )
                )
        return tuple(sorted(incomplete, key=lambda value: (value.executor_id, value.last_sequence, value.intent_id)))

    @classmethod
    def _validate_intent_transition(
        cls,
        event: JournalEventV1,
        decoded: Tuple[_DecodedJournalMutation, ...],
    ) -> None:
        events = tuple(mutation.committed.event for mutation in decoded)
        if event.event_type == JournalEventType.PREPARED:
            assert event.intent_id is not None
            assert event.idempotency_key is not None
            if any(existing.intent_id == event.intent_id for existing in events):
                raise JournalConflictError(f"intent {event.intent_id} already exists")
            if any(
                existing.event_type == JournalEventType.PREPARED and existing.idempotency_key == event.idempotency_key
                for existing in events
            ):
                raise JournalConflictError(f"stable idempotency key {event.idempotency_key} already exists")
            logical_key = cls._logical_exposure_key(event)
            if logical_key is not None:
                for incomplete in cls._incomplete_from_decoded(
                    decoded[0].committed.executor_id if decoded else "unknown",
                    decoded,
                ):
                    if cls._logical_exposure_key(incomplete.prepared_event) == logical_key:
                        raise JournalConflictError(
                            "logical quantity already has a non-terminal exposure-increasing intent"
                        )
            return

        if event.intent_id is None:
            if event.event_type in cls._REQUIRES_PREPARED_INTENT:
                raise JournalConflictError(f"{event.event_type.value} requires a PREPARED intent")
            return
        matching = [existing for existing in events if existing.intent_id == event.intent_id]
        if not matching or matching[0].event_type != JournalEventType.PREPARED:
            raise JournalConflictError(f"intent {event.intent_id} has no PREPARED event")
        if cls._event_terminal(matching[-1]):
            raise JournalConflictError(f"intent {event.intent_id} is already terminal")

        prepared_identity = getattr(matching[0].payload, "identity", None)
        event_identity = getattr(event.payload, "identity", None)
        if prepared_identity is None or event_identity is None or event_identity != prepared_identity:
            raise JournalConflictError(f"intent {event.intent_id} side-effect identity or action changed")

    @classmethod
    def _validate_snapshot_transition(
        cls,
        current: LeveragedEtfPairExecutorSnapshotV1,
        event: JournalEventV1,
        next_snapshot: LeveragedEtfPairExecutorSnapshotV1,
    ) -> LeveragedEtfPairExecutorSnapshotV1:
        if next_snapshot.last_journal_sequence != current.last_journal_sequence + 1:
            raise JournalConflictError("next snapshot journal sequence is not current + 1")
        for field_name in cls._IMMUTABLE_SNAPSHOT_FIELDS:
            if getattr(current, field_name) != getattr(next_snapshot, field_name):
                raise JournalIntegrityError(f"snapshot immutable field {field_name} changed")
        if next_snapshot.updated_at_utc < current.updated_at_utc:
            raise JournalIntegrityError("snapshot updated_at_utc moved backwards")
        if event.created_at_utc != next_snapshot.updated_at_utc:
            raise JournalIntegrityError("event timestamp disagrees with reducer state timestamp")

        next_state = event.payload.next_state
        if next_state.executor_id != current.executor_id:
            raise JournalIntegrityError("reducer state executor ID disagrees with current executor")
        if next_state.last_journal_sequence != current.last_journal_sequence + 1:
            raise JournalConflictError("reducer state journal sequence is not current + 1")
        if next_state.updated_at_utc != event.created_at_utc:
            raise JournalIntegrityError("reducer state timestamp disagrees with journal event")

        expected_payload = current.model_dump(mode="json")
        expected_payload.update(next_state.model_dump(mode="json"))
        expected = LeveragedEtfPairExecutorSnapshotV1.model_validate(expected_payload)
        if next_snapshot != expected:
            raise JournalIntegrityError("caller snapshot disagrees with deterministic journal reducer")

        cls._validate_side_effect_ownership(current, event)
        cls._validate_quantity_transition(current, event, expected)
        cls._validate_state_transition(current, event, expected)
        return expected

    @staticmethod
    def _validate_side_effect_ownership(
        current: LeveragedEtfPairExecutorSnapshotV1,
        event: JournalEventV1,
    ) -> None:
        identity = getattr(event.payload, "identity", None)
        if identity is None:
            return
        if identity.executor_id != current.executor_id or identity.operation != current.operation:
            raise JournalIntegrityError("journal side-effect executor or operation identity changed")
        connector = current.etf_connector_name if identity.leg == "ETF" else current.stock_connector_name
        trading_pair = current.etf_trading_pair if identity.leg == "ETF" else current.stock_trading_pair
        target = current.etf_target_quantity if identity.leg == "ETF" else current.stock_target_quantity
        if identity.connector_name != connector or identity.trading_pair != trading_pair:
            raise JournalIntegrityError("journal side-effect connector role identity changed")
        if identity.logical_quantity > target or identity.order_quantity > target:
            raise JournalIntegrityError("journal side-effect quantity exceeds the immutable leg target")

    @staticmethod
    def _validate_quantity_transition(
        current: LeveragedEtfPairExecutorSnapshotV1,
        event: JournalEventV1,
        expected: LeveragedEtfPairExecutorSnapshotV1,
    ) -> None:
        if expected.leverage_reservation != current.leverage_reservation:
            raise JournalIntegrityError("executor leverage reservation changed during journal reduction")
        for prefix in ("etf", "stock"):
            target = getattr(expected, f"{prefix}_target_quantity")
            submitted = getattr(expected, f"{prefix}_submitted_quantity")
            filled = getattr(expected, f"{prefix}_filled_quantity")
            remaining = getattr(expected, f"{prefix}_remaining_quantity")
            if not Decimal("0") <= filled <= submitted <= target:
                raise JournalIntegrityError(f"{prefix} quantity reducer invariant is invalid")
            if remaining != target - filled:
                raise JournalIntegrityError(f"{prefix} remaining quantity is not derived from fills")
            if submitted < getattr(current, f"{prefix}_submitted_quantity"):
                raise JournalIntegrityError(f"{prefix} submitted quantity moved backwards")
            if filled < getattr(current, f"{prefix}_filled_quantity"):
                raise JournalIntegrityError(f"{prefix} filled quantity moved backwards")

        for field_name in ("maker_order_ids", "stock_order_ids"):
            before = getattr(current, field_name)
            after = getattr(expected, field_name)
            if after[: len(before)] != before:
                raise JournalIntegrityError(f"{field_name} is not append-only")

        if expected.state in _TERMINAL_EXECUTOR_STATES:
            if expected.close_reason is None:
                raise JournalIntegrityError("terminal executor state requires a close reason")
        elif expected.close_reason != current.close_reason:
            raise JournalIntegrityError("non-terminal journal event changed the close reason")

        payload = event.payload
        if isinstance(payload, FillJournalPayloadV1):
            prefix = "etf" if payload.identity.leg == "ETF" else "stock"
            other = "stock" if prefix == "etf" else "etf"
            expected_filled = getattr(current, f"{prefix}_filled_quantity") + payload.fill_quantity
            if getattr(expected, f"{prefix}_filled_quantity") != expected_filled:
                raise JournalIntegrityError("fill quantity was not reduced exactly once")
            if getattr(expected, f"{prefix}_filled_quantity") != payload.cumulative_filled_quantity:
                raise JournalIntegrityError("fill cumulative quantity disagrees with reducer state")
            if payload.cumulative_filled_quantity > payload.identity.order_quantity:
                raise JournalIntegrityError("fill cumulative quantity exceeds the prepared order quantity")
            if payload.terminal and payload.cumulative_filled_quantity != payload.identity.order_quantity:
                raise JournalIntegrityError("terminal fill does not equal the prepared order quantity")
            submitted_delta = getattr(expected, f"{prefix}_submitted_quantity") - getattr(
                current, f"{prefix}_submitted_quantity"
            )
            if submitted_delta > payload.identity.order_quantity:
                raise JournalIntegrityError("fill increased submitted quantity beyond the prepared order")
            for suffix in ("submitted_quantity", "filled_quantity", "remaining_quantity"):
                if getattr(expected, f"{other}_{suffix}") != getattr(current, f"{other}_{suffix}"):
                    raise JournalIntegrityError("fill event changed the non-filled leg quantities")
            if (
                expected.maker_order_ids != current.maker_order_ids
                or expected.stock_order_ids != current.stock_order_ids
            ):
                raise JournalIntegrityError("fill event changed order identity history")
            return

        if isinstance(payload, OrderCreatedJournalPayloadV1):
            prefix = "etf" if payload.identity.leg == "ETF" else "stock"
            other = "stock" if prefix == "etf" else "etf"
            selected_orders = expected.maker_order_ids if prefix == "etf" else expected.stock_order_ids
            current_orders = current.maker_order_ids if prefix == "etf" else current.stock_order_ids
            if len(selected_orders) != len(current_orders) + 1:
                raise JournalIntegrityError("order-created event must append exactly one order identity")
            appended = selected_orders[-1]
            if (
                appended.client_order_id != payload.identity.client_order_id
                or appended.exchange_order_id != payload.exchange_order_id
            ):
                raise JournalIntegrityError("order-created identity disagrees with appended order history")
            other_orders = expected.stock_order_ids if prefix == "etf" else expected.maker_order_ids
            current_other_orders = current.stock_order_ids if prefix == "etf" else current.maker_order_ids
            if other_orders != current_other_orders:
                raise JournalIntegrityError("order-created event changed the other leg order history")
            submitted_delta = getattr(expected, f"{prefix}_submitted_quantity") - getattr(
                current, f"{prefix}_submitted_quantity"
            )
            if submitted_delta > payload.identity.order_quantity:
                raise JournalIntegrityError("order-created submitted quantity exceeds the prepared order")
            for suffix in ("filled_quantity", "remaining_quantity"):
                if getattr(expected, f"{prefix}_{suffix}") != getattr(current, f"{prefix}_{suffix}"):
                    raise JournalIntegrityError("order-created event changed fill facts")
            for suffix in ("submitted_quantity", "filled_quantity", "remaining_quantity"):
                if getattr(expected, f"{other}_{suffix}") != getattr(current, f"{other}_{suffix}"):
                    raise JournalIntegrityError("order-created event changed the other leg quantities")
            if expected.hedge_dust_quantity != current.hedge_dust_quantity:
                raise JournalIntegrityError("order-created event changed hedge dust")
            return

        quantity_fields = (
            "etf_submitted_quantity",
            "etf_filled_quantity",
            "etf_remaining_quantity",
            "stock_submitted_quantity",
            "stock_filled_quantity",
            "stock_remaining_quantity",
            "hedge_dust_quantity",
            "maker_order_ids",
            "stock_order_ids",
        )
        if any(getattr(expected, field_name) != getattr(current, field_name) for field_name in quantity_fields):
            raise JournalIntegrityError("non-fill event changed reducer-owned quantity or order facts")

    @staticmethod
    def _validate_state_transition(
        current: LeveragedEtfPairExecutorSnapshotV1,
        event: JournalEventV1,
        expected: LeveragedEtfPairExecutorSnapshotV1,
    ) -> None:
        source = current.state
        target = expected.state
        identity = getattr(event.payload, "identity", None)
        action = None if identity is None else identity.action

        if source in _TERMINAL_EXECUTOR_STATES:
            raise JournalConflictError(f"terminal executor state {source.value} cannot accept new journal events")

        pending_state = {
            JournalSideEffect.ETF_MAKER: LeveragedEtfPairState.MAKER_SUBMITTING,
            JournalSideEffect.CANCEL: LeveragedEtfPairState.MAKER_CANCEL_PENDING,
            JournalSideEffect.STOCK_HEDGE: LeveragedEtfPairState.STOCK_HEDGE_PENDING,
            JournalSideEffect.ETF_ROLLBACK: LeveragedEtfPairState.ETF_ROLLBACK_PENDING,
        }
        prepared_sources = {
            JournalSideEffect.ETF_MAKER: {
                LeveragedEtfPairState.CREATED,
                LeveragedEtfPairState.PREFLIGHT,
                LeveragedEtfPairState.MAKER_WORKING,
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
            },
            JournalSideEffect.CANCEL: {
                LeveragedEtfPairState.MAKER_SUBMITTING,
                LeveragedEtfPairState.MAKER_WORKING,
            },
            JournalSideEffect.STOCK_HEDGE: {LeveragedEtfPairState.STOCK_HEDGE_PENDING},
            JournalSideEffect.ETF_ROLLBACK: {
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                LeveragedEtfPairState.RECONCILING,
            },
        }
        reconciliation_sources = {
            JournalSideEffect.ETF_MAKER: {
                LeveragedEtfPairState.MAKER_SUBMITTING,
                LeveragedEtfPairState.MAKER_WORKING,
                LeveragedEtfPairState.MAKER_CANCEL_PENDING,
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
            },
            JournalSideEffect.CANCEL: {
                LeveragedEtfPairState.MAKER_CANCEL_PENDING,
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
            },
            JournalSideEffect.STOCK_HEDGE: {
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
            },
            JournalSideEffect.ETF_ROLLBACK: {
                LeveragedEtfPairState.ETF_ROLLBACK_PENDING,
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
            },
        }
        reconciliation_targets = {
            JournalSideEffect.ETF_MAKER: {
                LeveragedEtfPairState.MAKER_WORKING,
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
                LeveragedEtfPairState.ABORTED_NO_FILL,
            },
            JournalSideEffect.CANCEL: {
                LeveragedEtfPairState.MAKER_CANCEL_PENDING,
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
                LeveragedEtfPairState.ABORTED_NO_FILL,
            },
            JournalSideEffect.STOCK_HEDGE: {
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                LeveragedEtfPairState.ETF_ROLLBACK_PENDING,
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
                LeveragedEtfPairState.COMPLETED,
                LeveragedEtfPairState.FAILED_SAFE,
            },
            JournalSideEffect.ETF_ROLLBACK: {
                LeveragedEtfPairState.ETF_ROLLBACK_PENDING,
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
                LeveragedEtfPairState.COMPLETED,
                LeveragedEtfPairState.FAILED_SAFE,
            },
        }

        if event.event_type == JournalEventType.PREPARED:
            legal_sources = prepared_sources.get(action, set())
            legal_targets = {pending_state[action]} if action in pending_state else set()
        elif event.event_type in {JournalEventType.ACKNOWLEDGED, JournalEventType.ORDER_CREATED}:
            legal_sources = {pending_state[action]} if action in pending_state else set()
            if action == JournalSideEffect.ETF_MAKER:
                legal_sources.update(
                    {
                        LeveragedEtfPairState.MAKER_WORKING,
                        LeveragedEtfPairState.MAKER_CANCEL_PENDING,
                        LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                    }
                )
                legal_targets = (
                    {LeveragedEtfPairState.MAKER_WORKING}
                    if source in {LeveragedEtfPairState.MAKER_SUBMITTING, LeveragedEtfPairState.MAKER_WORKING}
                    else {source}
                )
            else:
                legal_targets = legal_sources
        elif event.event_type == JournalEventType.SUBMIT_UNKNOWN:
            legal_sources = {pending_state[action]} if action in pending_state else set()
            if action == JournalSideEffect.ETF_MAKER:
                legal_sources.update(
                    {
                        LeveragedEtfPairState.MAKER_CANCEL_PENDING,
                        LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                    }
                )
            legal_targets = {
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
            }
            if source in {
                LeveragedEtfPairState.MAKER_CANCEL_PENDING,
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
            }:
                legal_targets.add(source)
        elif event.event_type == JournalEventType.FILL:
            legal_sources = reconciliation_sources.get(action, set())
            legal_targets = {
                JournalSideEffect.ETF_MAKER: {LeveragedEtfPairState.STOCK_HEDGE_PENDING},
                JournalSideEffect.STOCK_HEDGE: {
                    LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                    LeveragedEtfPairState.COMPLETED,
                },
                JournalSideEffect.ETF_ROLLBACK: {
                    LeveragedEtfPairState.ETF_ROLLBACK_PENDING,
                    LeveragedEtfPairState.COMPLETED,
                    LeveragedEtfPairState.FAILED_SAFE,
                },
            }.get(action, set())
        elif event.event_type in {JournalEventType.CANCEL_REQUESTED, JournalEventType.CANCEL_CONFIRMED}:
            legal_sources = {LeveragedEtfPairState.MAKER_CANCEL_PENDING}
            legal_targets = {
                LeveragedEtfPairState.MAKER_CANCEL_PENDING,
                LeveragedEtfPairState.ABORTED_NO_FILL,
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
            }
        elif event.event_type in {JournalEventType.HEDGE_REQUESTED, JournalEventType.HEDGE_CONFIRMED}:
            legal_sources = {LeveragedEtfPairState.STOCK_HEDGE_PENDING}
            legal_targets = {LeveragedEtfPairState.STOCK_HEDGE_PENDING, LeveragedEtfPairState.COMPLETED}
        elif event.event_type in {JournalEventType.ROLLBACK_REQUESTED, JournalEventType.ROLLBACK_CONFIRMED}:
            legal_sources = {LeveragedEtfPairState.ETF_ROLLBACK_PENDING}
            legal_targets = {
                LeveragedEtfPairState.ETF_ROLLBACK_PENDING,
                LeveragedEtfPairState.ABORTED_NO_FILL,
                LeveragedEtfPairState.COMPLETED,
                LeveragedEtfPairState.FAILED_SAFE,
            }
        elif event.event_type == JournalEventType.RECONCILIATION:
            legal_sources = reconciliation_sources.get(action, set())
            legal_targets = reconciliation_targets.get(action, set())
        elif event.event_type == JournalEventType.REJECTED:
            legal_sources = {pending_state[action]} if action in pending_state else set()
            legal_targets = reconciliation_targets.get(action, set())
        else:
            transition_targets = {
                LeveragedEtfPairState.CREATED: {
                    LeveragedEtfPairState.CREATED,
                    LeveragedEtfPairState.PREFLIGHT,
                    LeveragedEtfPairState.ABORTED_NO_FILL,
                    LeveragedEtfPairState.RECOVERY_REQUIRED,
                },
                LeveragedEtfPairState.PREFLIGHT: {
                    LeveragedEtfPairState.PREFLIGHT,
                    LeveragedEtfPairState.ABORTED_NO_FILL,
                    LeveragedEtfPairState.RECOVERY_REQUIRED,
                },
                LeveragedEtfPairState.MAKER_SUBMITTING: {
                    LeveragedEtfPairState.MAKER_SUBMITTING,
                    LeveragedEtfPairState.RECONCILING,
                    LeveragedEtfPairState.RECOVERY_REQUIRED,
                },
                LeveragedEtfPairState.MAKER_WORKING: {
                    LeveragedEtfPairState.MAKER_WORKING,
                    LeveragedEtfPairState.RECONCILING,
                    LeveragedEtfPairState.RECOVERY_REQUIRED,
                },
                LeveragedEtfPairState.MAKER_CANCEL_PENDING: {
                    LeveragedEtfPairState.MAKER_CANCEL_PENDING,
                    LeveragedEtfPairState.RECONCILING,
                    LeveragedEtfPairState.RECOVERY_REQUIRED,
                },
                LeveragedEtfPairState.STOCK_HEDGE_PENDING: {
                    LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                    LeveragedEtfPairState.RECONCILING,
                    LeveragedEtfPairState.RECOVERY_REQUIRED,
                },
                LeveragedEtfPairState.ETF_ROLLBACK_PENDING: {
                    LeveragedEtfPairState.ETF_ROLLBACK_PENDING,
                    LeveragedEtfPairState.RECONCILING,
                    LeveragedEtfPairState.RECOVERY_REQUIRED,
                },
                LeveragedEtfPairState.RECONCILING: {
                    LeveragedEtfPairState.RECONCILING,
                    LeveragedEtfPairState.RECOVERY_REQUIRED,
                },
                LeveragedEtfPairState.RECOVERY_REQUIRED: {
                    LeveragedEtfPairState.RECOVERY_REQUIRED,
                    LeveragedEtfPairState.RECONCILING,
                },
            }
            legal_sources = {source}
            legal_targets = transition_targets.get(source, set())

        if source not in legal_sources:
            raise JournalConflictError(
                f"illegal {event.event_type.value} source state {source.value} for side-effect action"
            )
        if target not in legal_targets:
            raise JournalConflictError(
                f"illegal {event.event_type.value} reducer state transition {source.value} -> {target.value}"
            )

    def create_executor(
        self,
        snapshot: LeveragedEtfPairExecutorSnapshotV1,
    ) -> LeveragedEtfPairExecutorSnapshotV1:
        if snapshot.last_journal_sequence != 0:
            raise JournalConflictError("new executor snapshot must begin at journal sequence zero")

        def operation(connection: Connection):
            existing = self._load_snapshot_connection(connection, snapshot.executor_id)
            if existing is not None:
                if existing == snapshot:
                    return existing
                raise JournalIntegrityError(f"executor {snapshot.executor_id} already exists with different state")
            serialized = snapshot.model_dump(mode="json")
            payload_json = snapshot.canonical_json()
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfExecutorSnapshot (
                        executor_id, controller_id, pair_id, nav_cycle_id, schema_version,
                        state, snapshot_json, snapshot_hash, last_journal_sequence,
                        created_at_utc, updated_at_utc
                    ) VALUES (
                        :executor_id, :controller_id, :pair_id, :nav_cycle_id, :schema_version,
                        :state, :snapshot_json, :snapshot_hash, :last_journal_sequence,
                        :created_at_utc, :updated_at_utc
                    )
                    """),
                {
                    "executor_id": serialized["executor_id"],
                    "controller_id": serialized["controller_id"],
                    "pair_id": serialized["pair_id"],
                    "nav_cycle_id": serialized["nav_cycle_id"],
                    "schema_version": serialized["schema_version"],
                    "state": serialized["state"],
                    "snapshot_json": payload_json,
                    "snapshot_hash": _sha256_text(payload_json),
                    "last_journal_sequence": serialized["last_journal_sequence"],
                    "created_at_utc": serialized["created_at_utc"],
                    "updated_at_utc": serialized["updated_at_utc"],
                },
            )
            return snapshot

        try:
            return self._write(operation)
        except IntegrityError as exception:
            raise JournalConflictError(f"executor {snapshot.executor_id} could not be created") from exception

    def load_snapshot(self, executor_id: str) -> Optional[LeveragedEtfPairExecutorSnapshotV1]:
        def operation(connection: Connection):
            verified = self._verified_recovery_state(connection, executor_id)
            value = verified.get(executor_id)
            return None if value is None else value[0]

        return self._read_transaction(operation)

    def append_and_reduce(
        self,
        executor_id: str,
        event: JournalEventV1,
        next_snapshot: LeveragedEtfPairExecutorSnapshotV1,
    ) -> CommittedJournalEventV1:
        def operation(connection: Connection):
            current = self._load_snapshot_connection(connection, executor_id)
            if current is None:
                raise JournalConflictError(f"executor {executor_id} does not exist")
            decoded = self._decoded_events(connection, executor_id)
            self._replay_decoded(current, decoded)

            duplicate_row = (
                connection.execute(
                    text(self._EVENT_SELECT + " WHERE event_id = :event_id"),
                    {"event_id": event.event_id},
                )
                .mappings()
                .one_or_none()
            )
            if duplicate_row is not None:
                duplicate = self._decode_event_row(duplicate_row)
                if duplicate.committed.executor_id != executor_id:
                    raise JournalConflictError("event identity is owned by a different executor")
                if duplicate.committed.event == event and duplicate.snapshot_after == next_snapshot:
                    return duplicate.committed
                raise JournalIntegrityError("event identity has a conflicting payload")

            if event.exchange_trade_id is not None:
                if not isinstance(event.payload, FillJournalPayloadV1):
                    raise JournalIntegrityError("only a typed FILL event may own an exchange trade ID")
                trade_row = (
                    connection.execute(
                        text(
                            self._EVENT_SELECT
                            + " WHERE connector_name = :connector_name AND trading_pair = :trading_pair "
                            "AND exchange_trade_id = :exchange_trade_id"
                        ),
                        {
                            "connector_name": event.connector_name,
                            "trading_pair": event.trading_pair,
                            "exchange_trade_id": event.exchange_trade_id,
                        },
                    )
                    .mappings()
                    .one_or_none()
                )
                if trade_row is not None:
                    duplicate = self._decode_event_row(trade_row)
                    if duplicate.committed.executor_id != executor_id:
                        raise JournalConflictError("exchange trade identity is owned by a different executor")
                    raise JournalIntegrityError("exchange trade identity is already owned by a different event")

            if next_snapshot.executor_id != executor_id:
                raise JournalIntegrityError("next snapshot executor ID does not match repository key")
            self._validate_intent_transition(event, decoded)
            reduced_snapshot = self._validate_snapshot_transition(current, event, next_snapshot)

            sequence = current.last_journal_sequence + 1
            mutation = {
                "schema_version": 1,
                "event": event.model_dump(mode="json"),
                "snapshot_before": current.model_dump(mode="json"),
                "snapshot_before_hash": current.canonical_sha256(),
                "snapshot_after": reduced_snapshot.model_dump(mode="json"),
                "snapshot_after_hash": reduced_snapshot.canonical_sha256(),
            }
            payload_json = _canonical_json(mutation)
            payload_hash = _sha256_text(payload_json)
            event_json = event.model_dump(mode="json")
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfJournalEvent (
                        event_id, executor_id, sequence, event_type, intent_id,
                        idempotency_key, connector_name, trading_pair, client_order_id,
                        exchange_order_id, exchange_trade_id, payload_json, payload_hash,
                        created_at_utc
                    ) VALUES (
                        :event_id, :executor_id, :sequence, :event_type, :intent_id,
                        :idempotency_key, :connector_name, :trading_pair, :client_order_id,
                        :exchange_order_id, :exchange_trade_id, :payload_json, :payload_hash,
                        :created_at_utc
                    )
                    """),
                {
                    **{key: event_json[key] for key in event_json if key != "payload"},
                    "executor_id": executor_id,
                    "sequence": sequence,
                    "payload_json": payload_json,
                    "payload_hash": payload_hash,
                },
            )
            snapshot_json = reduced_snapshot.canonical_json()
            next_json = reduced_snapshot.model_dump(mode="json")
            updated = connection.execute(
                text("""
                    UPDATE LeveragedEtfExecutorSnapshot
                    SET state = :state,
                        snapshot_json = :snapshot_json,
                        snapshot_hash = :snapshot_hash,
                        last_journal_sequence = :next_sequence,
                        updated_at_utc = :updated_at_utc
                    WHERE executor_id = :executor_id
                      AND last_journal_sequence = :current_sequence
                    """),
                {
                    "state": next_json["state"],
                    "snapshot_json": snapshot_json,
                    "snapshot_hash": _sha256_text(snapshot_json),
                    "next_sequence": sequence,
                    "updated_at_utc": next_json["updated_at_utc"],
                    "executor_id": executor_id,
                    "current_sequence": current.last_journal_sequence,
                },
            )
            if updated.rowcount != 1:
                raise JournalConflictError("snapshot compare-and-set failed")
            return CommittedJournalEventV1(
                executor_id=executor_id,
                sequence=sequence,
                event=event,
                mutation_hash=payload_hash,
            )

        try:
            return self._write(operation)
        except IntegrityError as exception:
            raise JournalConflictError("journal identity or reservation constraint rejected the append") from exception

    def events(self, executor_id: str) -> Tuple[CommittedJournalEventV1, ...]:
        def operation(connection: Connection):
            verified = self._verified_recovery_state(connection, executor_id)
            value = verified.get(executor_id)
            return () if value is None else tuple(mutation.committed for mutation in value[1])

        return self._read_transaction(operation)

    def replay(self, executor_id: str) -> LeveragedEtfPairExecutorSnapshotV1:
        def operation(connection: Connection):
            verified = self._verified_recovery_state(connection, executor_id)
            value = verified.get(executor_id)
            if value is None:
                raise JournalConflictError(f"executor {executor_id} does not exist")
            return value[0]

        return self._read_transaction(operation)

    def incomplete_intents(self, executor_id: Optional[str] = None) -> Tuple[IncompleteIntentV1, ...]:
        def operation(connection: Connection):
            verified = self._verified_recovery_state(connection, executor_id)
            result = []
            for current_executor_id, (_, mutations) in verified.items():
                result.extend(self._incomplete_from_decoded(current_executor_id, mutations))
            return tuple(sorted(result, key=lambda value: (value.executor_id, value.last_sequence, value.intent_id)))

        return self._read_transaction(operation)

    def incomplete_executors(self) -> Tuple[LeveragedEtfPairExecutorSnapshotV1, ...]:
        def operation(connection: Connection):
            verified = self._verified_recovery_state(connection)
            incomplete_intent_ids = {
                current_executor_id
                for current_executor_id, (_, mutations) in verified.items()
                if self._incomplete_from_decoded(current_executor_id, mutations)
            }
            return tuple(
                snapshot
                for current_executor_id, (snapshot, _) in sorted(verified.items())
                if snapshot.state not in _TERMINAL_EXECUTOR_STATES or current_executor_id in incomplete_intent_ids
            )

        return self._read_transaction(operation)

    @staticmethod
    def _decode_reservation_row(row: Mapping[str, Any]) -> StrategyReservationV1:
        try:
            payload = _verify_canonical_json(row["payload_json"], row["payload_hash"], "strategy reservation")
            reservation = StrategyReservationV1.model_validate(payload)
        except Exception as exception:
            raise JournalIntegrityError(f"reservation integrity failure: {exception}") from exception
        serialized = reservation.model_dump(mode="json")
        for column_name in (
            "reservation_id",
            "executor_id",
            "reservation_key",
            "connector_name",
            "trading_pair",
            "leg",
            "quantity",
            "leverage",
            "notional_cap",
            "created_at_utc",
            "updated_at_utc",
            "released_at_utc",
        ):
            if row[column_name] != serialized[column_name]:
                raise JournalIntegrityError(f"reservation column {column_name} disagrees with payload")
        return reservation

    def reserve(self, reservation: StrategyReservationV1) -> StrategyReservationV1:
        def operation(connection: Connection):
            serialized = reservation.model_dump(mode="json")
            expected_key = (
                f"{reservation.executor_id}:{reservation.leg}:" f"{_canonical_decimal_text(reservation.quantity)}"
            )
            if reservation.reservation_key != expected_key:
                raise JournalConflictError("reservation key is not the canonical logical identity")
            if reservation.released_at_utc is not None:
                raise JournalConflictError("reservation cannot be created in a released state")
            if reservation.updated_at_utc != reservation.created_at_utc:
                raise JournalConflictError("reservation creation updated time must equal created time")

            snapshot = self._load_snapshot_connection(connection, reservation.executor_id)
            if snapshot is None:
                raise JournalConflictError(f"reservation executor {reservation.executor_id} does not exist")
            decoded = self._decoded_events(connection, reservation.executor_id)
            self._replay_decoded(snapshot, decoded)
            if reservation.leg == "ETF":
                expected_connector = snapshot.etf_connector_name
                expected_pair = snapshot.etf_trading_pair
                expected_quantity = snapshot.leverage_reservation.etf_quantity
                expected_leverage = snapshot.leverage_reservation.etf_leverage
                expected_notional_cap = snapshot.leverage_reservation.etf_notional_cap
            else:
                expected_connector = snapshot.stock_connector_name
                expected_pair = snapshot.stock_trading_pair
                expected_quantity = snapshot.leverage_reservation.stock_quantity
                expected_leverage = snapshot.leverage_reservation.stock_leverage
                expected_notional_cap = snapshot.leverage_reservation.stock_notional_cap
            if reservation.connector_name != expected_connector or reservation.trading_pair != expected_pair:
                raise JournalIntegrityError("reservation connector or trading-pair role disagrees with executor")
            if reservation.quantity > expected_quantity:
                raise JournalIntegrityError("reservation quantity exceeds the executor leverage reservation")
            if reservation.leverage != expected_leverage:
                raise JournalIntegrityError("reservation leverage disagrees with the executor reservation")
            if reservation.notional_cap != expected_notional_cap:
                raise JournalIntegrityError("reservation notional cap disagrees with the executor reservation")

            existing_rows = (
                connection.execute(
                    text("""
                    SELECT * FROM LeveragedEtfStrategyReservation
                    WHERE reservation_id = :reservation_id OR reservation_key = :reservation_key
                    """),
                    {
                        "reservation_id": reservation.reservation_id,
                        "reservation_key": reservation.reservation_key,
                    },
                )
                .mappings()
                .all()
            )
            if existing_rows:
                existing = self._decode_reservation_row(existing_rows[0])
                if len(existing_rows) == 1 and existing == reservation:
                    return existing
                raise JournalConflictError("reservation identity has conflicting payload")
            active = (
                connection.execute(
                    text("""
                    SELECT * FROM LeveragedEtfStrategyReservation
                    WHERE executor_id = :executor_id
                      AND connector_name = :connector_name
                      AND trading_pair = :trading_pair
                      AND leg = :leg
                      AND released_at_utc IS NULL
                    """),
                    reservation.model_dump(mode="json"),
                )
                .mappings()
                .one_or_none()
            )
            if active is not None:
                raise JournalConflictError("reservation active leg already exists")
            payload_json = reservation.canonical_json()
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfStrategyReservation (
                        reservation_id, executor_id, reservation_key, connector_name,
                        trading_pair, leg, quantity, leverage, notional_cap, payload_json,
                        payload_hash, created_at_utc, updated_at_utc, released_at_utc
                    ) VALUES (
                        :reservation_id, :executor_id, :reservation_key, :connector_name,
                        :trading_pair, :leg, :quantity, :leverage, :notional_cap, :payload_json,
                        :payload_hash, :created_at_utc, :updated_at_utc, :released_at_utc
                    )
                    """),
                {
                    **serialized,
                    "payload_json": payload_json,
                    "payload_hash": _sha256_text(payload_json),
                },
            )
            return reservation

        try:
            return self._write(operation)
        except IntegrityError as exception:
            raise JournalConflictError("reservation uniqueness constraint rejected the write") from exception

    def release_reservation(self, reservation_id: str, released_at_utc: str) -> StrategyReservationV1:
        _validate_utc_text(released_at_utc, "released_at_utc")

        def operation(connection: Connection):
            row = (
                connection.execute(
                    text("SELECT * FROM LeveragedEtfStrategyReservation WHERE reservation_id = :reservation_id"),
                    {"reservation_id": reservation_id},
                )
                .mappings()
                .one_or_none()
            )
            if row is None:
                raise JournalConflictError(f"reservation {reservation_id} does not exist")
            current = self._decode_reservation_row(row)
            current_json = current.model_dump(mode="json")
            if current_json["released_at_utc"] is not None:
                if current_json["released_at_utc"] == released_at_utc:
                    return current
                raise JournalConflictError("reservation was already released at a different time")
            if released_at_utc < current_json["created_at_utc"] or released_at_utc < current_json["updated_at_utc"]:
                raise JournalConflictError("reservation release time must be monotonic after creation")
            updated = StrategyReservationV1.model_validate(
                {
                    **current_json,
                    "updated_at_utc": released_at_utc,
                    "released_at_utc": released_at_utc,
                }
            )
            serialized = updated.model_dump(mode="json")
            payload_json = updated.canonical_json()
            result = connection.execute(
                text("""
                    UPDATE LeveragedEtfStrategyReservation
                    SET payload_json = :payload_json,
                        payload_hash = :payload_hash,
                        updated_at_utc = :updated_at_utc,
                        released_at_utc = :released_at_utc
                    WHERE reservation_id = :reservation_id AND released_at_utc IS NULL
                    """),
                {
                    "payload_json": payload_json,
                    "payload_hash": _sha256_text(payload_json),
                    "updated_at_utc": serialized["updated_at_utc"],
                    "released_at_utc": serialized["released_at_utc"],
                    "reservation_id": reservation_id,
                },
            )
            if result.rowcount != 1:
                raise JournalConflictError("reservation release compare-and-set failed")
            return updated

        return self._write(operation)

    def active_reservations(self, executor_id: Optional[str] = None) -> Tuple[StrategyReservationV1, ...]:
        statement = "SELECT * FROM LeveragedEtfStrategyReservation WHERE released_at_utc IS NULL"
        parameters = {}
        if executor_id is not None:
            statement += " AND executor_id = :executor_id"
            parameters["executor_id"] = executor_id
        statement += " ORDER BY executor_id, connector_name, trading_pair, leg, reservation_id"
        with self._sql_manager.engine.connect() as connection:
            rows = connection.execute(text(statement), parameters).mappings().all()
        return tuple(self._decode_reservation_row(row) for row in rows)


class AnchorRepositoryV1(_TransactionalRepository):
    """Sole synchronous anchor repository with typed-v1 wrappers over one opaque path."""

    _ANCHOR_SELECT = """
        SELECT cycle_id, schema_version, state_kind, revision, target_session_date,
               official_close_utc, deadline_utc, evidence_hash, payload_json,
               payload_hash, created_at_utc, updated_at_utc
        FROM LeveragedEtfAnchorState
        WHERE cycle_id = :cycle_id
    """

    @staticmethod
    def _validate_opaque_metadata(state: OpaqueAnchorStateV1) -> None:
        value = state.payload.value()
        if not isinstance(value, Mapping):
            raise AnchorIntegrityError("anchor payload must be a JSON object")
        comparisons = {
            "cycle_id": state.cycle_id,
            "target_session_date": state.target_session_date,
            "deadline_utc": state.model_dump(mode="json")["deadline_utc"],
        }
        if isinstance(state, OpaqueAnchorCheckpointV1):
            comparisons["official_close_utc"] = state.model_dump(mode="json")["official_close_utc"]
            comparisons["revision"] = state.revision
        else:
            comparisons["evidence_hash"] = state.evidence_hash
        for key, expected in comparisons.items():
            if key in value and value[key] != expected:
                raise AnchorIntegrityError(f"anchor payload {key} disagrees with storage metadata")

    @staticmethod
    def _assert_cycle_identity(
        current: OpaqueAnchorStateV1,
        proposed: OpaqueAnchorStateV1,
        *,
        inherit_missing_official_close: bool = False,
    ) -> None:
        current_json = current.model_dump(mode="json")
        proposed_json = proposed.model_dump(mode="json")
        if current.cycle_id != proposed.cycle_id:
            raise AnchorIntegrityError("anchor cycle identity cannot change")
        if current.target_session_date != proposed.target_session_date:
            raise AnchorIntegrityError("anchor target session identity cannot change")
        if current_json["deadline_utc"] != proposed_json["deadline_utc"]:
            raise AnchorIntegrityError("anchor deadline identity cannot change")
        proposed_close = proposed_json["official_close_utc"]
        if inherit_missing_official_close and proposed_close is None:
            proposed_close = current_json["official_close_utc"]
        if current_json["official_close_utc"] != proposed_close:
            raise AnchorIntegrityError("anchor official close identity cannot change")

    @classmethod
    def _decode_anchor_row(cls, row: Mapping[str, Any]) -> OpaqueAnchorStateV1:
        if row["schema_version"] != 1:
            raise AnchorIntegrityError("anchor storage schema version is unsupported")
        try:
            value = _verify_canonical_json(row["payload_json"], row["payload_hash"], "anchor state")
            if not isinstance(value, Mapping):
                raise ValueError("anchor payload must be an object")
            payload_version = value.get("schema_version")
            if not isinstance(payload_version, int) or isinstance(payload_version, bool) or payload_version < 1:
                raise ValueError("anchor payload schema_version is invalid")
            if row["state_kind"] == "CHECKPOINT":
                if row["evidence_hash"] is not None or row["official_close_utc"] is None:
                    raise ValueError("checkpoint storage metadata is invalid")
                payload = CanonicalOpaquePayload.from_canonical_json(
                    payload_version,
                    "ANCHOR_CHECKPOINT",
                    row["payload_json"],
                    row["payload_hash"],
                )
                state: OpaqueAnchorStateV1 = OpaqueAnchorCheckpointV1(
                    cycle_id=row["cycle_id"],
                    target_session_date=row["target_session_date"],
                    official_close_utc=row["official_close_utc"],
                    deadline_utc=row["deadline_utc"],
                    revision=row["revision"],
                    payload=payload,
                )
            elif row["state_kind"] == "FINALIZED":
                if row["evidence_hash"] is None:
                    raise ValueError("finalized storage metadata lacks evidence hash")
                payload = CanonicalOpaquePayload.from_canonical_json(
                    payload_version,
                    "ANCHOR_RECORD",
                    row["payload_json"],
                    row["payload_hash"],
                )
                state = OpaqueAnchorFinalizedV1(
                    cycle_id=row["cycle_id"],
                    target_session_date=row["target_session_date"],
                    official_close_utc=row["official_close_utc"],
                    deadline_utc=row["deadline_utc"],
                    revision=row["revision"],
                    evidence_hash=row["evidence_hash"],
                    payload=payload,
                )
            else:
                raise ValueError("anchor state_kind is invalid")
            cls._validate_opaque_metadata(state)
            return state
        except AnchorIntegrityError:
            raise
        except Exception as exception:
            raise AnchorIntegrityError(f"anchor state hash or payload integrity failure: {exception}") from exception

    @classmethod
    def _load_opaque_connection(cls, connection: Connection, cycle_id: str) -> Optional[OpaqueAnchorStateV1]:
        row = connection.execute(text(cls._ANCHOR_SELECT), {"cycle_id": cycle_id}).mappings().one_or_none()
        return None if row is None else cls._decode_anchor_row(row)

    def load_opaque(self, cycle_id: str) -> Optional[OpaqueAnchorStateV1]:
        with self._sql_manager.engine.connect() as connection:
            return self._load_opaque_connection(connection, cycle_id)

    def load(self, cycle_id: str) -> Optional[AnchorStateV1]:
        opaque = self.load_opaque(cycle_id)
        if opaque is None:
            return None
        if opaque.payload.schema_version != 1:
            raise AnchorIntegrityError(
                f"anchor payload version {opaque.payload.schema_version} requires a downstream lossless adapter"
            )
        try:
            if isinstance(opaque, OpaqueAnchorCheckpointV1):
                checkpoint = AnchorPollingCheckpointV1.model_validate(opaque.payload.value())
                expected = (
                    checkpoint.cycle_id,
                    checkpoint.target_session_date,
                    checkpoint.official_close_utc,
                    checkpoint.deadline_utc,
                    checkpoint.revision,
                )
                actual = (
                    opaque.cycle_id,
                    opaque.target_session_date,
                    opaque.official_close_utc,
                    opaque.deadline_utc,
                    opaque.revision,
                )
                if expected != actual:
                    raise AnchorIntegrityError("typed checkpoint metadata disagrees with stored envelope")
                return checkpoint
            record = AnchorRecordV1.model_validate(opaque.payload.value())
            if (
                record.cycle_id != opaque.cycle_id
                or record.target_session_date != opaque.target_session_date
                or record.deadline_utc != opaque.deadline_utc
                or record.evidence_hash != opaque.evidence_hash
            ):
                raise AnchorIntegrityError("typed anchor metadata disagrees with stored envelope")
            return record
        except AnchorIntegrityError:
            raise
        except Exception as exception:
            raise AnchorIntegrityError(f"malformed anchor v1 payload: {exception}") from exception

    def compare_and_set_opaque_checkpoint(
        self,
        checkpoint: OpaqueAnchorCheckpointV1,
        expected_revision: int,
    ) -> OpaqueAnchorCheckpointV1:
        if expected_revision < 0 or checkpoint.revision != expected_revision + 1:
            raise AnchorRevisionConflict("checkpoint revision must be exactly expected_revision + 1")
        self._validate_opaque_metadata(checkpoint)
        checkpoint_json = checkpoint.model_dump(mode="json")
        payload_value = checkpoint.payload.value()
        updated_at_utc = (
            payload_value.get("next_poll_utc", checkpoint_json["official_close_utc"])
            if isinstance(payload_value, Mapping)
            else checkpoint_json["official_close_utc"]
        )
        _validate_utc_text(updated_at_utc, "checkpoint updated_at_utc")

        def operation(connection: Connection):
            current = self._load_opaque_connection(connection, checkpoint.cycle_id)
            parameters = {
                "cycle_id": checkpoint.cycle_id,
                "schema_version": 1,
                "state_kind": "CHECKPOINT",
                "revision": checkpoint.revision,
                "target_session_date": checkpoint.target_session_date,
                "official_close_utc": checkpoint_json["official_close_utc"],
                "deadline_utc": checkpoint_json["deadline_utc"],
                "payload_json": checkpoint.payload.payload_json,
                "payload_hash": checkpoint.payload.payload_hash,
                "updated_at_utc": updated_at_utc,
                "expected_revision": expected_revision,
            }
            if current is None:
                if expected_revision != 0:
                    raise AnchorRevisionConflict("checkpoint does not exist at expected revision")
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfAnchorState (
                            cycle_id, schema_version, state_kind, revision,
                            target_session_date, official_close_utc, deadline_utc,
                            evidence_hash, payload_json, payload_hash,
                            created_at_utc, updated_at_utc
                        ) VALUES (
                            :cycle_id, :schema_version, :state_kind, :revision,
                            :target_session_date, :official_close_utc, :deadline_utc,
                            NULL, :payload_json, :payload_hash,
                            :updated_at_utc, :updated_at_utc
                        )
                        """),
                    parameters,
                )
                return checkpoint
            if not isinstance(current, OpaqueAnchorCheckpointV1):
                raise AnchorIntegrityError("finalized anchor cannot be replaced by a checkpoint")
            self._assert_cycle_identity(current, checkpoint)
            if current.revision != expected_revision:
                raise AnchorRevisionConflict(
                    f"checkpoint revision {current.revision} does not match expected {expected_revision}"
                )
            updated = connection.execute(
                text("""
                    UPDATE LeveragedEtfAnchorState
                    SET revision = :revision,
                        target_session_date = :target_session_date,
                        official_close_utc = :official_close_utc,
                        deadline_utc = :deadline_utc,
                        payload_json = :payload_json,
                        payload_hash = :payload_hash,
                        updated_at_utc = :updated_at_utc
                    WHERE cycle_id = :cycle_id
                      AND state_kind = 'CHECKPOINT'
                      AND revision = :expected_revision
                    """),
                parameters,
            )
            if updated.rowcount != 1:
                raise AnchorRevisionConflict("checkpoint compare-and-set lost a concurrent race")
            return checkpoint

        try:
            return self._write(operation)
        except IntegrityError as exception:
            raise AnchorRevisionConflict("checkpoint compare-and-set violated storage identity") from exception

    def compare_and_set_checkpoint(
        self,
        checkpoint: AnchorPollingCheckpointV1,
        expected_revision: int,
    ) -> AnchorStateV1:
        serialized = checkpoint.model_dump(mode="json")
        opaque = OpaqueAnchorCheckpointV1(
            cycle_id=checkpoint.cycle_id,
            target_session_date=checkpoint.target_session_date,
            official_close_utc=serialized["official_close_utc"],
            deadline_utc=serialized["deadline_utc"],
            revision=checkpoint.revision,
            payload=CanonicalOpaquePayload.from_value(1, "ANCHOR_CHECKPOINT", serialized),
        )
        self.compare_and_set_opaque_checkpoint(opaque, expected_revision)
        return checkpoint

    def finalize_opaque_if_absent(
        self,
        record: OpaqueAnchorFinalizedV1,
        expected_revision: int,
    ) -> OpaqueAnchorFinalizedV1:
        if expected_revision < 0:
            raise AnchorRevisionConflict("expected revision must be non-negative")
        self._validate_opaque_metadata(record)
        record_json = record.model_dump(mode="json")
        payload_value = record.payload.value()
        updated_at_utc = (
            payload_value.get("finalized_at_utc", record_json["deadline_utc"])
            if isinstance(payload_value, Mapping)
            else record_json["deadline_utc"]
        )
        _validate_utc_text(updated_at_utc, "anchor finalized_at_utc")

        def operation(connection: Connection):
            current = self._load_opaque_connection(connection, record.cycle_id)
            if isinstance(current, OpaqueAnchorFinalizedV1):
                self._assert_cycle_identity(
                    current,
                    record,
                    inherit_missing_official_close=True,
                )
                if current.evidence_hash != record.evidence_hash:
                    raise AnchorIntegrityError("different final evidence cannot overwrite an anchor")
                if current.payload != record.payload:
                    raise AnchorIntegrityError("same evidence hash has conflicting finalized payload")
                return current
            parameters = {
                "cycle_id": record.cycle_id,
                "schema_version": 1,
                "state_kind": "FINALIZED",
                "revision": record.revision,
                "target_session_date": record.target_session_date,
                "official_close_utc": record_json["official_close_utc"],
                "deadline_utc": record_json["deadline_utc"],
                "evidence_hash": record.evidence_hash,
                "payload_json": record.payload.payload_json,
                "payload_hash": record.payload.payload_hash,
                "updated_at_utc": updated_at_utc,
                "expected_revision": expected_revision,
            }
            if current is None:
                if expected_revision != 0 or record.revision != 1:
                    raise AnchorRevisionConflict("absent anchor can only finalize from revision zero")
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfAnchorState (
                            cycle_id, schema_version, state_kind, revision,
                            target_session_date, official_close_utc, deadline_utc,
                            evidence_hash, payload_json, payload_hash,
                            created_at_utc, updated_at_utc
                        ) VALUES (
                            :cycle_id, :schema_version, :state_kind, :revision,
                            :target_session_date, :official_close_utc, :deadline_utc,
                            :evidence_hash, :payload_json, :payload_hash,
                            :updated_at_utc, :updated_at_utc
                        )
                        """),
                    parameters,
                )
                return record
            if not isinstance(current, OpaqueAnchorCheckpointV1):
                raise AnchorIntegrityError("anchor state kind is invalid")
            self._assert_cycle_identity(
                current,
                record,
                inherit_missing_official_close=True,
            )
            if current.revision != expected_revision:
                raise AnchorRevisionConflict(
                    f"checkpoint revision {current.revision} does not match expected {expected_revision}"
                )
            if record.revision != current.revision:
                raise AnchorRevisionConflict("final record revision must match the checkpoint revision")
            parameters["revision"] = current.revision
            parameters["official_close_utc"] = record_json["official_close_utc"] or _utc_text(
                current.official_close_utc
            )
            updated = connection.execute(
                text("""
                    UPDATE LeveragedEtfAnchorState
                    SET state_kind = :state_kind,
                        revision = :revision,
                        target_session_date = :target_session_date,
                        official_close_utc = :official_close_utc,
                        deadline_utc = :deadline_utc,
                        evidence_hash = :evidence_hash,
                        payload_json = :payload_json,
                        payload_hash = :payload_hash,
                        updated_at_utc = :updated_at_utc
                    WHERE cycle_id = :cycle_id
                      AND state_kind = 'CHECKPOINT'
                      AND revision = :expected_revision
                    """),
                parameters,
            )
            if updated.rowcount != 1:
                raise AnchorRevisionConflict("anchor finalization lost a concurrent race")
            return OpaqueAnchorFinalizedV1.model_validate(
                {
                    **record_json,
                    "revision": current.revision,
                    "official_close_utc": parameters["official_close_utc"],
                }
            )

        try:
            return self._write(operation)
        except IntegrityError as exception:
            raise AnchorIntegrityError("anchor finalization violated immutable identity") from exception

    def finalize_if_absent(
        self,
        record: AnchorRecordV1,
        expected_revision: int,
    ) -> AnchorRecordV1:
        serialized = record.model_dump(mode="json")
        opaque = OpaqueAnchorFinalizedV1(
            cycle_id=record.cycle_id,
            target_session_date=record.target_session_date,
            official_close_utc=None,
            deadline_utc=serialized["deadline_utc"],
            revision=max(1, expected_revision),
            evidence_hash=record.evidence_hash,
            payload=CanonicalOpaquePayload.from_value(1, "ANCHOR_RECORD", serialized),
        )
        finalized = self.finalize_opaque_if_absent(opaque, expected_revision)
        try:
            return AnchorRecordV1.model_validate(finalized.payload.value())
        except Exception as exception:
            raise AnchorIntegrityError(f"finalized v1 record is malformed: {exception}") from exception

    def append_revision_observation(
        self,
        cycle_id: str,
        evidence_hash: str,
        observed_at: str,
    ) -> None:
        observation = AnchorRevisionObservationV1(
            cycle_id=cycle_id,
            evidence_hash=evidence_hash,
            observed_at_utc=observed_at,
        )

        def operation(connection: Connection):
            current = self._load_opaque_connection(connection, cycle_id)
            if current is None:
                raise AnchorIntegrityError(f"anchor cycle {cycle_id} does not exist")
            if not isinstance(current, OpaqueAnchorFinalizedV1):
                raise AnchorIntegrityError("revision observations require a finalized anchor")
            existing = connection.execute(
                text("""
                    SELECT 1 FROM LeveragedEtfAnchorRevisionObservation
                    WHERE cycle_id = :cycle_id
                      AND evidence_hash = :evidence_hash
                      AND observed_at_utc = :observed_at_utc
                    """),
                observation.model_dump(mode="json"),
            ).one_or_none()
            if existing is not None:
                return None
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfAnchorRevisionObservation (
                        cycle_id, evidence_hash, observed_at_utc
                    ) VALUES (:cycle_id, :evidence_hash, :observed_at_utc)
                    """),
                observation.model_dump(mode="json"),
            )
            return None

        try:
            self._write(operation)
        except IntegrityError as exception:
            raise AnchorIntegrityError("anchor revision observation violated append-only identity") from exception

    def revision_observations(self, cycle_id: str) -> Tuple[AnchorRevisionObservationV1, ...]:
        with self._sql_manager.engine.connect() as connection:
            rows = (
                connection.execute(
                    text("""
                    SELECT cycle_id, evidence_hash, observed_at_utc
                    FROM LeveragedEtfAnchorRevisionObservation
                    WHERE cycle_id = :cycle_id
                    ORDER BY observed_at_utc, evidence_hash
                    """),
                    {"cycle_id": cycle_id},
                )
                .mappings()
                .all()
            )
        return tuple(AnchorRevisionObservationV1.model_validate(dict(row)) for row in rows)
