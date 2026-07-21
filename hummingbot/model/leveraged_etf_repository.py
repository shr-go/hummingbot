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
_ANCHOR_PAIR_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,127}$")
_ANCHOR_CYCLE_ID_PATTERN = re.compile(r"^xnys-[0-9]{4}-[0-9]{2}-[0-9]{2}$")
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
    order_cumulative_filled_quantity: CanonicalPositiveDecimal
    leg_cumulative_filled_quantity: CanonicalPositiveDecimal
    outcome: Literal["PARTIAL", "FILLED"]

    @model_validator(mode="after")
    def validate_fill_outcome(self) -> FillJournalPayloadV1:
        if self.fill_quantity > self.order_cumulative_filled_quantity:
            raise ValueError("fill_quantity cannot exceed order cumulative fill")
        if self.fill_quantity > self.leg_cumulative_filled_quantity:
            raise ValueError("fill_quantity cannot exceed leg cumulative fill")
        return self

    @property
    def terminal(self) -> bool:
        return self.outcome == "FILLED" and self.identity.action == JournalSideEffect.ETF_MAKER


class CancelJournalPayloadV1(_IdentityJournalPayloadV1):
    kind: Literal["CANCEL"] = "CANCEL"
    phase: Literal["REQUESTED", "CONFIRMED"]
    target_intent_id: Optional[StableIdentifier] = None
    target_client_order_id: Optional[StableIdentifier] = None
    target_exchange_order_id: Optional[StableIdentifier] = None
    final_order_cumulative_filled_quantity: CanonicalNonNegativeDecimal

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
    exchange_order_id: Optional[StableIdentifier] = None
    order_cumulative_filled_quantity: CanonicalNonNegativeDecimal = Decimal("0")

    @property
    def terminal(self) -> bool:
        return self.outcome in {
            ReconciliationOutcome.FILLED,
            ReconciliationOutcome.CANCELED,
            ReconciliationOutcome.EXPIRED,
            ReconciliationOutcome.REJECTED,
            ReconciliationOutcome.CONSISTENT_NO_FILL,
        }


class StateTransitionJournalPayloadV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    kind: Literal["STATE_TRANSITION"] = "STATE_TRANSITION"
    reason: str = Field(min_length=1, max_length=1024)
    target_state: LeveragedEtfPairState


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
            payload = payload.value()
        elif isinstance(payload, Mapping) and {"payload_json", "payload_hash", "kind"} <= set(payload):
            payload = CanonicalOpaquePayload.model_validate(payload).value()
        if not isinstance(payload, Mapping):
            return value

        normalized = dict(payload)
        legacy_state = normalized.pop("next_state", None)
        kind = normalized.get("kind")
        if kind == "FILL":
            legacy_cumulative = normalized.pop("cumulative_filled_quantity", None)
            if legacy_cumulative is not None:
                normalized["order_cumulative_filled_quantity"] = legacy_cumulative
                normalized["leg_cumulative_filled_quantity"] = legacy_cumulative
            normalized.pop("terminal", None)
        elif kind == "CANCEL":
            legacy_cumulative = normalized.pop("final_cumulative_filled_quantity", None)
            if legacy_cumulative is not None:
                normalized.setdefault("final_order_cumulative_filled_quantity", legacy_cumulative)
            normalized.setdefault("final_order_cumulative_filled_quantity", "0")
        elif kind == "RECONCILIATION":
            normalized.pop("terminal", None)
            normalized.setdefault("exchange_order_id", value.get("exchange_order_id"))
            normalized.setdefault("order_cumulative_filled_quantity", "0")
        elif kind == "STATE_TRANSITION" and isinstance(legacy_state, Mapping):
            normalized.setdefault("target_state", legacy_state.get("state"))
        return {**value, "payload": normalized}

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
        if isinstance(self.payload, ReconciliationJournalPayloadV1):
            if self.exchange_order_id != self.payload.exchange_order_id:
                raise ValueError("reconciliation exchange_order_id disagrees with payload")
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


def _validate_checkpoint_payload_revision(kind: str, value: Any) -> None:
    if kind == "ANCHOR_CHECKPOINT" and (
        not isinstance(value, Mapping) or type(value.get("revision")) is not int
    ):
        raise ValueError("opaque anchor checkpoint payload revision must be an exact built-in integer")


def _validate_canonical_anchor_payload_v2(
    *,
    kind: str,
    contract_version_field: str,
    contract_version: int,
    payload_json: str,
    payload_hash: str,
) -> Mapping[str, Any]:
    value = _verify_canonical_json(payload_json, payload_hash, f"{kind} payload")
    if not isinstance(value, Mapping):
        raise ValueError("opaque anchor payload must be a JSON object")
    _validate_checkpoint_payload_revision(kind, value)
    declared_version = value.get(contract_version_field)
    if (
        not isinstance(declared_version, int)
        or isinstance(declared_version, bool)
        or declared_version != contract_version
    ):
        raise ValueError("opaque anchor payload contract version is invalid")
    if "kind" in value and value["kind"] != kind:
        raise ValueError("wrapper and embedded anchor payload kind must match exactly")
    return value


class CanonicalOpaqueAnchorPayloadV2(BaseModel):
    """Canonical domain payload with an explicit, adapter-supplied version discriminator."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    kind: str = Field(min_length=1, max_length=64, pattern=r"^[A-Z][A-Z0-9_]*$")
    contract_version_field: Literal["schema_version", "integrity_version", "evidence_version"]
    contract_version: StrictInt = Field(ge=1)
    payload_json: str
    payload_hash: Sha256Hex

    @model_validator(mode="after")
    def validate_canonical_payload(self) -> CanonicalOpaqueAnchorPayloadV2:
        _validate_canonical_anchor_payload_v2(
            kind=self.kind,
            contract_version_field=self.contract_version_field,
            contract_version=self.contract_version,
            payload_json=self.payload_json,
            payload_hash=self.payload_hash,
        )
        return self

    @classmethod
    def from_value(
        cls,
        *,
        kind: str,
        contract_version_field: str,
        contract_version: int,
        value: Any,
    ) -> CanonicalOpaqueAnchorPayloadV2:
        _validate_checkpoint_payload_revision(kind, value)
        payload_json = _canonical_json(value)
        return cls(
            kind=kind,
            contract_version_field=contract_version_field,
            contract_version=contract_version,
            payload_json=payload_json,
            payload_hash=_sha256_text(payload_json),
        )

    @classmethod
    def from_canonical_json(
        cls,
        *,
        kind: str,
        contract_version_field: str,
        contract_version: int,
        payload_json: str,
        payload_hash: str,
    ) -> CanonicalOpaqueAnchorPayloadV2:
        return cls(
            kind=kind,
            contract_version_field=contract_version_field,
            contract_version=contract_version,
            payload_json=payload_json,
            payload_hash=payload_hash,
        )

    def value(self) -> Mapping[str, Any]:
        return _validate_canonical_anchor_payload_v2(
            kind=self.kind,
            contract_version_field=self.contract_version_field,
            contract_version=self.contract_version,
            payload_json=self.payload_json,
            payload_hash=self.payload_hash,
        )


class AnchorStorageKeyV2(BaseModel):
    """F004-owned opaque storage namespace; F006 maps the F003 key into this DTO."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    pair_id: str = Field(min_length=1, max_length=128)
    cycle_id: str

    @model_validator(mode="after")
    def validate_identity(self) -> AnchorStorageKeyV2:
        if _ANCHOR_PAIR_ID_PATTERN.fullmatch(self.pair_id) is None:
            raise ValueError("anchor pair_id must be a bounded canonical identifier")
        if _ANCHOR_CYCLE_ID_PATTERN.fullmatch(self.cycle_id) is None:
            raise ValueError("anchor cycle_id must be an XNYS cycle identifier")
        return self


def _validate_pair_scoped_payload_identity(
    key: AnchorStorageKeyV2,
    payload: CanonicalOpaqueAnchorPayloadV2,
) -> Mapping[str, Any]:
    value = payload.value()
    if value.get("pair_id") != key.pair_id or value.get("cycle_id") != key.cycle_id:
        raise AnchorIntegrityError("anchor storage key does not match payload pair/cycle identity")
    return value


class OpaqueAnchorCheckpointV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    key: AnchorStorageKeyV2
    target_session_date: str
    official_close_utc: str
    deadline_utc: str
    revision: StrictInt = Field(ge=1)
    payload: CanonicalOpaqueAnchorPayloadV2

    @model_validator(mode="after")
    def validate_metadata(self) -> OpaqueAnchorCheckpointV2:
        if _SESSION_DATE_PATTERN.fullmatch(self.target_session_date) is None:
            raise ValueError("target_session_date is not canonical")
        _validate_utc_text(self.official_close_utc, "anchor official_close_utc")
        _validate_utc_text(self.deadline_utc, "anchor deadline_utc")
        if self.payload.kind != "ANCHOR_CHECKPOINT":
            raise ValueError("checkpoint payload kind must be ANCHOR_CHECKPOINT")
        value = _validate_pair_scoped_payload_identity(self.key, self.payload)
        comparisons = {
            "target_session_date": self.target_session_date,
            "official_close_utc": self.official_close_utc,
            "deadline_utc": self.deadline_utc,
            "revision": self.revision,
        }
        for field_name, expected in comparisons.items():
            actual = value.get(field_name)
            if type(actual) is not type(expected) or actual != expected:
                raise AnchorIntegrityError(f"anchor checkpoint {field_name} disagrees with envelope metadata")
        return self


class OpaqueAnchorFinalizedV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    key: AnchorStorageKeyV2
    target_session_date: str
    official_close_utc: str
    deadline_utc: str
    revision: StrictInt = Field(ge=1)
    evidence_hash: Sha256Hex
    payload: CanonicalOpaqueAnchorPayloadV2

    @model_validator(mode="after")
    def validate_metadata(self) -> OpaqueAnchorFinalizedV2:
        if _SESSION_DATE_PATTERN.fullmatch(self.target_session_date) is None:
            raise ValueError("target_session_date is not canonical")
        _validate_utc_text(self.official_close_utc, "anchor official_close_utc")
        _validate_utc_text(self.deadline_utc, "anchor deadline_utc")
        if self.payload.kind != "ANCHOR_RECORD":
            raise ValueError("finalized payload kind must be ANCHOR_RECORD")
        value = _validate_pair_scoped_payload_identity(self.key, self.payload)
        comparisons = {
            "target_session_date": self.target_session_date,
            "official_close_utc": self.official_close_utc,
            "deadline_utc": self.deadline_utc,
            "evidence_hash": self.evidence_hash,
        }
        for field_name, expected in comparisons.items():
            if value.get(field_name) != expected:
                raise AnchorIntegrityError(f"finalized anchor {field_name} disagrees with envelope metadata")
        return self


OpaqueAnchorStateV2 = Union[OpaqueAnchorCheckpointV2, OpaqueAnchorFinalizedV2]


class OpaqueAnchorRevisionObservationV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[2] = 2
    key: AnchorStorageKeyV2
    evidence_hash: Sha256Hex
    observed_at_utc: str
    payload: CanonicalOpaqueAnchorPayloadV2

    @model_validator(mode="after")
    def validate_metadata(self) -> OpaqueAnchorRevisionObservationV2:
        _validate_utc_text(self.observed_at_utc, "anchor revision observed_at_utc")
        if self.payload.kind != "ANCHOR_REVISION_OBSERVATION":
            raise ValueError("revision payload kind must be ANCHOR_REVISION_OBSERVATION")
        value = _validate_pair_scoped_payload_identity(self.key, self.payload)
        if value.get("evidence_hash") != self.evidence_hash:
            raise AnchorIntegrityError("anchor revision evidence_hash disagrees with envelope metadata")
        if value.get("observed_at_utc") != self.observed_at_utc:
            raise AnchorIntegrityError("anchor revision observed_at_utc disagrees with envelope metadata")
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
            JournalEventType.RECONCILIATION,
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
        if event.event_type in {
            JournalEventType.REJECTED,
            JournalEventType.CANCEL_CONFIRMED,
            JournalEventType.HEDGE_CONFIRMED,
            JournalEventType.ROLLBACK_CONFIRMED,
        }:
            return True
        return (
            isinstance(event.payload, (FillJournalPayloadV1, ReconciliationJournalPayloadV1)) and event.payload.terminal
        )

    @classmethod
    def _intent_terminal(cls, events: Tuple[JournalEventV1, ...]) -> bool:
        """Derive terminal progress from facts without depending on their arrival order."""

        return any(cls._event_terminal(event) for event in events)

    @staticmethod
    def _late_submission_fact(event: JournalEventV1) -> bool:
        return event.event_type in {
            JournalEventType.ACKNOWLEDGED,
            JournalEventType.ORDER_CREATED,
        }

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
            expected = cls._reduce_snapshot(mutation.snapshot_before, mutation.committed.event, tuple(prior))
            if mutation.snapshot_after != expected:
                raise JournalIntegrityError("persisted journal snapshot disagrees with authoritative event reduction")
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

        self._validate_persisted_global_ownership(connection)
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
    def _validate_persisted_global_ownership(connection: Connection) -> None:
        rows = connection.execute(text("""
                SELECT executor_id, intent_id, connector_name, trading_pair,
                       client_order_id, exchange_order_id
                FROM LeveragedEtfJournalEvent
                WHERE client_order_id IS NOT NULL OR exchange_order_id IS NOT NULL
                ORDER BY executor_id, sequence
            """)).mappings()
        client_owners: dict[Tuple[str, str], Tuple[str, Optional[str]]] = {}
        exchange_owners: dict[Tuple[str, str, str], Tuple[str, Optional[str]]] = {}
        for row in rows:
            owner = (row["executor_id"], row["intent_id"])
            if row["connector_name"] is not None and row["client_order_id"] is not None:
                key = (row["connector_name"], row["client_order_id"])
                if key in client_owners and client_owners[key] != owner:
                    raise JournalIntegrityError("persisted client order identity has conflicting global owners")
                client_owners[key] = owner
            if (
                row["connector_name"] is not None
                and row["trading_pair"] is not None
                and row["exchange_order_id"] is not None
            ):
                key = (row["connector_name"], row["trading_pair"], row["exchange_order_id"])
                if key in exchange_owners and exchange_owners[key] != owner:
                    raise JournalIntegrityError("persisted exchange order identity has conflicting global owners")
                exchange_owners[key] = owner

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
        by_intent: dict[str, list[CommittedJournalEventV1]] = {}
        for mutation in decoded:
            event = mutation.committed.event
            if event.intent_id is None:
                continue
            if event.event_type == JournalEventType.PREPARED:
                prepared[event.intent_id] = event
            by_intent.setdefault(event.intent_id, []).append(mutation.committed)
        incomplete = []
        for intent_id, committed_events in by_intent.items():
            prepared_event = prepared.get(intent_id)
            if prepared_event is None:
                raise JournalIntegrityError(f"intent {intent_id} has no PREPARED event")
            if not cls._intent_terminal(tuple(committed.event for committed in committed_events)):
                committed = committed_events[-1]
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
                existing.event_type == JournalEventType.PREPARED
                and existing.idempotency_key == event.idempotency_key
                and getattr(existing.payload, "identity", None).action == event.payload.identity.action
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

        prepared_identity = getattr(matching[0].payload, "identity", None)
        event_identity = getattr(event.payload, "identity", None)
        if prepared_identity is None or event_identity is None or event_identity != prepared_identity:
            raise JournalConflictError(f"intent {event.intent_id} side-effect identity or action changed")

        has_fill_fact = any(isinstance(existing.payload, FillJournalPayloadV1) for existing in matching)
        if cls._late_submission_fact(event) and has_fill_fact:
            if any(existing.event_type == event.event_type for existing in matching):
                raise JournalConflictError(f"intent {event.intent_id} already recorded {event.event_type.value}")
            return
        if cls._intent_terminal(tuple(matching)):
            raise JournalConflictError(f"intent {event.intent_id} is already terminal")

        action = prepared_identity.action
        latest_type = matching[-1].event_type
        common_submission = {
            JournalEventType.ACKNOWLEDGED,
            JournalEventType.ORDER_CREATED,
            JournalEventType.FILL,
            JournalEventType.REJECTED,
            JournalEventType.SUBMIT_UNKNOWN,
            JournalEventType.RECONCILIATION,
        }
        if action == JournalSideEffect.ETF_MAKER:
            legal = {
                JournalEventType.PREPARED: common_submission,
                JournalEventType.ACKNOWLEDGED: common_submission
                - {JournalEventType.ACKNOWLEDGED, JournalEventType.REJECTED},
                JournalEventType.ORDER_CREATED: {
                    JournalEventType.ACKNOWLEDGED,
                    JournalEventType.FILL,
                    JournalEventType.SUBMIT_UNKNOWN,
                    JournalEventType.RECONCILIATION,
                },
                JournalEventType.FILL: {JournalEventType.FILL, JournalEventType.RECONCILIATION},
                JournalEventType.SUBMIT_UNKNOWN: {JournalEventType.RECONCILIATION},
                JournalEventType.RECONCILIATION: {
                    JournalEventType.RECONCILIATION,
                    JournalEventType.FILL,
                    JournalEventType.ORDER_CREATED,
                },
            }
        elif action == JournalSideEffect.CANCEL:
            legal = {
                JournalEventType.PREPARED: {JournalEventType.CANCEL_REQUESTED},
                JournalEventType.CANCEL_REQUESTED: {
                    JournalEventType.CANCEL_CONFIRMED,
                    JournalEventType.SUBMIT_UNKNOWN,
                    JournalEventType.RECONCILIATION,
                },
                JournalEventType.SUBMIT_UNKNOWN: {JournalEventType.RECONCILIATION},
                JournalEventType.RECONCILIATION: {
                    JournalEventType.RECONCILIATION,
                    JournalEventType.CANCEL_CONFIRMED,
                },
            }
        elif action == JournalSideEffect.STOCK_HEDGE:
            legal = {
                JournalEventType.PREPARED: {JournalEventType.HEDGE_REQUESTED},
                JournalEventType.HEDGE_REQUESTED: common_submission | {JournalEventType.HEDGE_CONFIRMED},
                JournalEventType.ACKNOWLEDGED: (common_submission | {JournalEventType.HEDGE_CONFIRMED})
                - {JournalEventType.ACKNOWLEDGED, JournalEventType.REJECTED},
                JournalEventType.ORDER_CREATED: {
                    JournalEventType.ACKNOWLEDGED,
                    JournalEventType.FILL,
                    JournalEventType.SUBMIT_UNKNOWN,
                    JournalEventType.RECONCILIATION,
                    JournalEventType.HEDGE_CONFIRMED,
                },
                JournalEventType.FILL: {
                    JournalEventType.FILL,
                    JournalEventType.RECONCILIATION,
                    JournalEventType.HEDGE_CONFIRMED,
                },
                JournalEventType.SUBMIT_UNKNOWN: {JournalEventType.RECONCILIATION},
                JournalEventType.RECONCILIATION: {
                    JournalEventType.RECONCILIATION,
                    JournalEventType.FILL,
                    JournalEventType.HEDGE_CONFIRMED,
                },
            }
        else:
            legal = {
                JournalEventType.PREPARED: {JournalEventType.ROLLBACK_REQUESTED},
                JournalEventType.ROLLBACK_REQUESTED: common_submission | {JournalEventType.ROLLBACK_CONFIRMED},
                JournalEventType.ACKNOWLEDGED: (common_submission | {JournalEventType.ROLLBACK_CONFIRMED})
                - {JournalEventType.ACKNOWLEDGED, JournalEventType.REJECTED},
                JournalEventType.ORDER_CREATED: {
                    JournalEventType.ACKNOWLEDGED,
                    JournalEventType.FILL,
                    JournalEventType.SUBMIT_UNKNOWN,
                    JournalEventType.RECONCILIATION,
                    JournalEventType.ROLLBACK_CONFIRMED,
                },
                JournalEventType.FILL: {
                    JournalEventType.FILL,
                    JournalEventType.RECONCILIATION,
                    JournalEventType.ROLLBACK_CONFIRMED,
                },
                JournalEventType.SUBMIT_UNKNOWN: {JournalEventType.RECONCILIATION},
                JournalEventType.RECONCILIATION: {
                    JournalEventType.RECONCILIATION,
                    JournalEventType.FILL,
                    JournalEventType.ROLLBACK_CONFIRMED,
                },
            }
        if event.event_type not in legal.get(latest_type, set()):
            raise JournalConflictError(
                f"illegal {action.value} intent phase {latest_type.value} -> {event.event_type.value}"
            )

    @classmethod
    def _events_with_candidate(
        cls,
        decoded: Tuple[_DecodedJournalMutation, ...],
        candidate: Optional[JournalEventV1] = None,
    ) -> Tuple[JournalEventV1, ...]:
        events = tuple(mutation.committed.event for mutation in decoded)
        return events if candidate is None else (*events, candidate)

    @classmethod
    def _action_fill_total(
        cls,
        decoded: Tuple[_DecodedJournalMutation, ...],
        action: JournalSideEffect,
        candidate: Optional[JournalEventV1] = None,
    ) -> Decimal:
        return sum(
            (
                event.payload.fill_quantity
                for event in cls._events_with_candidate(decoded, candidate)
                if isinstance(event.payload, FillJournalPayloadV1) and event.payload.identity.action == action
            ),
            Decimal("0"),
        )

    @classmethod
    def _action_logical_target(
        cls,
        decoded: Tuple[_DecodedJournalMutation, ...],
        identity: SideEffectIdentityV1,
    ) -> Decimal:
        targets = tuple(
            event.payload.identity.logical_quantity
            for event in cls._events_with_candidate(decoded)
            if isinstance(event.payload, PreparedJournalPayloadV1)
            and event.payload.identity.action == identity.action
            and event.payload.identity.leg == identity.leg
        )
        if not targets:
            raise JournalIntegrityError("fill action has no authoritative prepared leg target")
        return max(targets)

    @classmethod
    def _intent_fill_total(
        cls,
        decoded: Tuple[_DecodedJournalMutation, ...],
        intent_id: str,
        candidate: Optional[JournalEventV1] = None,
    ) -> Decimal:
        return sum(
            (
                event.payload.fill_quantity
                for event in cls._events_with_candidate(decoded, candidate)
                if event.intent_id == intent_id and isinstance(event.payload, FillJournalPayloadV1)
            ),
            Decimal("0"),
        )

    @classmethod
    def _order_fill_total(
        cls,
        decoded: Tuple[_DecodedJournalMutation, ...],
        connector_name: str,
        trading_pair: str,
        exchange_order_id: str,
        candidate: Optional[JournalEventV1] = None,
    ) -> Decimal:
        return sum(
            (
                event.payload.fill_quantity
                for event in cls._events_with_candidate(decoded, candidate)
                if isinstance(event.payload, FillJournalPayloadV1)
                and event.connector_name == connector_name
                and event.trading_pair == trading_pair
                and event.exchange_order_id == exchange_order_id
            ),
            Decimal("0"),
        )

    @classmethod
    def _all_intents_terminal_after(
        cls,
        decoded: Tuple[_DecodedJournalMutation, ...],
        event: JournalEventV1,
    ) -> bool:
        by_intent: dict[str, list[JournalEventV1]] = {}
        for candidate in cls._events_with_candidate(decoded, event):
            if candidate.intent_id is not None:
                by_intent.setdefault(candidate.intent_id, []).append(candidate)
        return all(cls._intent_terminal(tuple(intent_events)) for intent_events in by_intent.values())

    @classmethod
    def _action_intents_terminal_after(
        cls,
        decoded: Tuple[_DecodedJournalMutation, ...],
        event: JournalEventV1,
        action: JournalSideEffect,
    ) -> bool:
        by_intent: dict[str, list[JournalEventV1]] = {}
        for candidate in cls._events_with_candidate(decoded, event):
            identity = getattr(candidate.payload, "identity", None)
            if candidate.intent_id is not None and identity is not None and identity.action == action:
                by_intent.setdefault(candidate.intent_id, []).append(candidate)
        return all(cls._intent_terminal(tuple(intent_events)) for intent_events in by_intent.values())

    @classmethod
    def _exposure_totals(
        cls,
        decoded: Tuple[_DecodedJournalMutation, ...],
        event: JournalEventV1,
    ) -> Tuple[Decimal, Decimal, Decimal]:
        return (
            cls._action_fill_total(decoded, JournalSideEffect.ETF_MAKER, event),
            cls._action_fill_total(decoded, JournalSideEffect.STOCK_HEDGE, event),
            cls._action_fill_total(decoded, JournalSideEffect.ETF_ROLLBACK, event),
        )

    @classmethod
    def _exposure_is_balanced(
        cls,
        current: LeveragedEtfPairExecutorSnapshotV1,
        decoded: Tuple[_DecodedJournalMutation, ...],
        event: JournalEventV1,
    ) -> bool:
        maker_filled, stock_filled, rollback_filled = cls._exposure_totals(decoded, event)
        if rollback_filled > maker_filled:
            return False
        if maker_filled == 0 and stock_filled == 0 and rollback_filled == 0:
            return True
        if maker_filled == rollback_filled and stock_filled == 0:
            return True
        prepared_identities = (
            candidate.payload.identity
            for candidate in cls._events_with_candidate(decoded, event)
            if isinstance(candidate.payload, PreparedJournalPayloadV1)
        )
        cumulative_logical: dict[JournalSideEffect, Decimal] = {}
        for identity in prepared_identities:
            cumulative_logical[identity.action] = max(
                cumulative_logical.get(identity.action, Decimal("0")),
                identity.logical_quantity,
            )
        if (
            rollback_filled == 0
            and maker_filled == cumulative_logical.get(JournalSideEffect.ETF_MAKER, Decimal("0"))
            and stock_filled == cumulative_logical.get(JournalSideEffect.STOCK_HEDGE, Decimal("0"))
            and maker_filled > 0
            and stock_filled > 0
        ):
            return True
        return (maker_filled - rollback_filled) * current.stock_target_quantity == (
            stock_filled * current.etf_target_quantity
        )

    @classmethod
    def _reduce_snapshot(
        cls,
        current: LeveragedEtfPairExecutorSnapshotV1,
        event: JournalEventV1,
        decoded: Tuple[_DecodedJournalMutation, ...],
    ) -> LeveragedEtfPairExecutorSnapshotV1:
        if event.created_at_utc < current.updated_at_utc:
            raise JournalIntegrityError("journal event timestamp moved backwards")
        cls._validate_side_effect_ownership(current, event)
        cls._validate_event_facts(current, event, decoded)

        serialized = current.model_dump(mode="json")
        next_sequence = current.last_journal_sequence + 1
        serialized["last_journal_sequence"] = next_sequence
        serialized["updated_at_utc"] = event.model_dump(mode="json")["created_at_utc"]
        identity = getattr(event.payload, "identity", None)

        def append_order_reference(prefix: str, exchange_order_id: Optional[str]) -> None:
            if identity is None or exchange_order_id is None:
                return
            field_name = "maker_order_ids" if prefix == "etf" else "stock_order_ids"
            references = list(serialized[field_name])
            if any(reference["exchange_order_id"] == exchange_order_id for reference in references):
                return
            references.append(
                {
                    "sequence": next_sequence,
                    "client_order_id": identity.client_order_id,
                    "exchange_order_id": exchange_order_id,
                }
            )
            serialized[field_name] = references

        if isinstance(event.payload, OrderCreatedJournalPayloadV1):
            prefix = "etf" if identity.leg == "ETF" else "stock"
            append_order_reference(prefix, event.payload.exchange_order_id)
            submitted_field = f"{prefix}_submitted_quantity"
            serialized[submitted_field] = _canonical_decimal_text(
                max(Decimal(serialized[submitted_field]), identity.logical_quantity)
            )
        elif isinstance(event.payload, FillJournalPayloadV1):
            if identity.action == JournalSideEffect.ETF_MAKER:
                prefix = "etf"
            elif identity.action == JournalSideEffect.STOCK_HEDGE:
                prefix = "stock"
            else:
                prefix = None
            if prefix is not None:
                append_order_reference(prefix, event.payload.exchange_order_id)
                filled_field = f"{prefix}_filled_quantity"
                submitted_field = f"{prefix}_submitted_quantity"
                new_filled = Decimal(serialized[filled_field]) + event.payload.fill_quantity
                serialized[filled_field] = _canonical_decimal_text(new_filled)
                serialized[submitted_field] = _canonical_decimal_text(
                    max(Decimal(serialized[submitted_field]), new_filled)
                )

        serialized["etf_remaining_quantity"] = _canonical_decimal_text(
            current.etf_target_quantity - Decimal(serialized["etf_filled_quantity"])
        )
        serialized["stock_remaining_quantity"] = _canonical_decimal_text(
            current.stock_target_quantity - Decimal(serialized["stock_filled_quantity"])
        )
        state = cls._derive_state(current, event, decoded)
        serialized["state"] = state.value
        if state in _TERMINAL_EXECUTOR_STATES:
            serialized["close_reason"] = cls._terminal_close_reason(state, event)
        expected = LeveragedEtfPairExecutorSnapshotV1.model_validate(serialized)
        cls._validate_terminal_snapshot(expected, event, decoded)
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

    @classmethod
    def _validate_event_facts(
        cls,
        current: LeveragedEtfPairExecutorSnapshotV1,
        event: JournalEventV1,
        decoded: Tuple[_DecodedJournalMutation, ...],
    ) -> None:
        payload = event.payload
        identity = getattr(payload, "identity", None)
        if isinstance(payload, PreparedJournalPayloadV1) and identity.deadline_utc < event.created_at_utc:
            raise JournalConflictError("prepared side-effect deadline precedes the journal event time")
        if identity is None:
            return

        intent_events = tuple(
            candidate for candidate in cls._events_with_candidate(decoded) if candidate.intent_id == identity.intent_id
        )
        bound_exchange_order_ids = {
            candidate.exchange_order_id for candidate in intent_events if candidate.exchange_order_id is not None
        }
        if len(bound_exchange_order_ids) > 1:
            raise JournalIntegrityError("intent has more than one bound exchange order identity")
        if event.exchange_order_id is not None and bound_exchange_order_ids:
            if event.exchange_order_id not in bound_exchange_order_ids:
                raise JournalConflictError("event exchange order is not bound to the prepared intent")

        if isinstance(payload, FillJournalPayloadV1):
            if payload.exchange_order_id is None:
                raise JournalIntegrityError("fill requires an authoritative exchange order identity")
            prior_order_fill = cls._order_fill_total(
                decoded,
                identity.connector_name,
                identity.trading_pair,
                payload.exchange_order_id,
            )
            expected_order_fill = prior_order_fill + payload.fill_quantity
            if payload.order_cumulative_filled_quantity != expected_order_fill:
                raise JournalIntegrityError("fill per-order cumulative quantity disagrees with accepted trades")
            prior_leg_fill = cls._action_fill_total(decoded, identity.action)
            if payload.leg_cumulative_filled_quantity != prior_leg_fill + payload.fill_quantity:
                raise JournalIntegrityError("fill leg cumulative quantity disagrees with accepted trades")
            immutable_leg_target = (
                current.etf_target_quantity if identity.leg == "ETF" else current.stock_target_quantity
            )
            authoritative_leg_target = min(
                immutable_leg_target,
                cls._action_logical_target(decoded, identity),
            )
            if payload.leg_cumulative_filled_quantity > authoritative_leg_target:
                raise JournalIntegrityError("fill leg cumulative quantity exceeds the authoritative leg target")
            if payload.order_cumulative_filled_quantity > identity.order_quantity:
                raise JournalIntegrityError("fill per-order cumulative quantity exceeds prepared order quantity")
            if payload.outcome == "FILLED":
                if payload.order_cumulative_filled_quantity != identity.order_quantity:
                    raise JournalIntegrityError("FILLED outcome requires the complete prepared order quantity")
            elif payload.order_cumulative_filled_quantity >= identity.order_quantity:
                raise JournalIntegrityError("PARTIAL outcome must remain below the prepared order quantity")

        if isinstance(payload, ReconciliationJournalPayloadV1):
            if payload.exchange_order_id is None:
                recorded_order_fill = cls._intent_fill_total(decoded, identity.intent_id)
                if bound_exchange_order_ids:
                    raise JournalIntegrityError("reconciliation of a bound order requires its exchange identity")
            else:
                recorded_order_fill = cls._order_fill_total(
                    decoded,
                    identity.connector_name,
                    identity.trading_pair,
                    payload.exchange_order_id,
                )
            if payload.order_cumulative_filled_quantity != recorded_order_fill:
                raise JournalIntegrityError("reconciliation cumulative fill disagrees with accepted fill facts")
            if recorded_order_fill > identity.order_quantity:
                raise JournalIntegrityError("reconciliation cumulative fill exceeds prepared order quantity")
            if payload.outcome == ReconciliationOutcome.FILLED:
                if payload.exchange_order_id is None or recorded_order_fill != identity.order_quantity:
                    raise JournalIntegrityError("reconciliation FILLED requires a fully recorded bound order")
            elif payload.outcome == ReconciliationOutcome.PARTIALLY_FILLED:
                if not Decimal("0") < recorded_order_fill < identity.order_quantity:
                    raise JournalIntegrityError("PARTIALLY_FILLED reconciliation contradicts accepted fills")
            elif (
                payload.outcome
                in {
                    ReconciliationOutcome.NEW,
                    ReconciliationOutcome.NOT_FOUND,
                    ReconciliationOutcome.CONSISTENT_NO_FILL,
                }
                and recorded_order_fill != 0
            ):
                raise JournalIntegrityError(f"{payload.outcome.value} reconciliation contradicts accepted fills")

        if isinstance(payload, CancelJournalPayloadV1):
            if payload.target_intent_id is None or payload.target_client_order_id is None:
                raise JournalIntegrityError("cancel requires a prepared target intent and client order identity")
            target_events = tuple(
                candidate
                for candidate in cls._events_with_candidate(decoded)
                if candidate.intent_id == payload.target_intent_id
            )
            if not target_events or not isinstance(target_events[0].payload, PreparedJournalPayloadV1):
                raise JournalConflictError("cancel target intent has no PREPARED event")
            target_identity = target_events[0].payload.identity
            if (
                target_identity.action != JournalSideEffect.ETF_MAKER
                or target_identity.client_order_id != payload.target_client_order_id
                or target_identity.connector_name != identity.connector_name
            ):
                raise JournalIntegrityError("cancel target identity does not match the maker order")
            target_bound_ids = {
                candidate.exchange_order_id for candidate in target_events if candidate.exchange_order_id is not None
            }
            if payload.target_exchange_order_id is not None:
                if target_bound_ids != {payload.target_exchange_order_id}:
                    raise JournalIntegrityError("cancel target exchange order is not bound to the maker intent")
                recorded_fill = cls._order_fill_total(
                    decoded,
                    target_identity.connector_name,
                    target_identity.trading_pair,
                    payload.target_exchange_order_id,
                )
            else:
                if target_bound_ids:
                    raise JournalIntegrityError("cancel of a bound maker order requires its exchange identity")
                recorded_fill = cls._intent_fill_total(decoded, payload.target_intent_id)
            if payload.final_order_cumulative_filled_quantity != recorded_fill:
                raise JournalIntegrityError("cancel cumulative fill disagrees with accepted maker fills")

        if event.event_type in {JournalEventType.HEDGE_CONFIRMED, JournalEventType.ROLLBACK_CONFIRMED}:
            recorded_fill = cls._intent_fill_total(decoded, identity.intent_id)
            if recorded_fill != identity.order_quantity:
                raise JournalIntegrityError(f"{event.event_type.value} requires a completely recorded side-effect fill")

    @classmethod
    def _derive_state(
        cls,
        current: LeveragedEtfPairExecutorSnapshotV1,
        event: JournalEventV1,
        decoded: Tuple[_DecodedJournalMutation, ...],
    ) -> LeveragedEtfPairState:
        source = current.state
        late_after_fill = cls._late_submission_fact(event) and any(
            candidate.intent_id == event.intent_id and isinstance(candidate.payload, FillJournalPayloadV1)
            for candidate in cls._events_with_candidate(decoded)
        )
        if source in _TERMINAL_EXECUTOR_STATES:
            if late_after_fill:
                return source
            raise JournalConflictError(f"terminal executor state {source.value} cannot accept journal events")
        payload = event.payload
        identity = getattr(payload, "identity", None)
        action = None if identity is None else identity.action
        if late_after_fill:
            return source

        prepared_sources = {
            JournalSideEffect.ETF_MAKER: {
                LeveragedEtfPairState.CREATED,
                LeveragedEtfPairState.PREFLIGHT,
                LeveragedEtfPairState.MAKER_WORKING,
            },
            JournalSideEffect.CANCEL: {
                LeveragedEtfPairState.MAKER_SUBMITTING,
                LeveragedEtfPairState.MAKER_WORKING,
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                LeveragedEtfPairState.RECONCILING,
            },
            JournalSideEffect.STOCK_HEDGE: {LeveragedEtfPairState.STOCK_HEDGE_PENDING},
            JournalSideEffect.ETF_ROLLBACK: {
                LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                LeveragedEtfPairState.RECONCILING,
                LeveragedEtfPairState.RECOVERY_REQUIRED,
            },
        }
        active_sources = {
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
        pending_state = {
            JournalSideEffect.ETF_MAKER: LeveragedEtfPairState.MAKER_SUBMITTING,
            JournalSideEffect.CANCEL: LeveragedEtfPairState.MAKER_CANCEL_PENDING,
            JournalSideEffect.STOCK_HEDGE: LeveragedEtfPairState.STOCK_HEDGE_PENDING,
            JournalSideEffect.ETF_ROLLBACK: LeveragedEtfPairState.ETF_ROLLBACK_PENDING,
        }

        if event.event_type == JournalEventType.STATE_TRANSITION:
            target = payload.target_state
            if target not in _TERMINAL_EXECUTOR_STATES:
                legal_targets = {
                    LeveragedEtfPairState.CREATED: {
                        LeveragedEtfPairState.CREATED,
                        LeveragedEtfPairState.PREFLIGHT,
                        LeveragedEtfPairState.RECOVERY_REQUIRED,
                    },
                    LeveragedEtfPairState.PREFLIGHT: {
                        LeveragedEtfPairState.PREFLIGHT,
                        LeveragedEtfPairState.RECOVERY_REQUIRED,
                    },
                    LeveragedEtfPairState.MAKER_SUBMITTING: {
                        LeveragedEtfPairState.RECONCILING,
                        LeveragedEtfPairState.RECOVERY_REQUIRED,
                    },
                    LeveragedEtfPairState.MAKER_WORKING: {
                        LeveragedEtfPairState.RECONCILING,
                        LeveragedEtfPairState.RECOVERY_REQUIRED,
                    },
                    LeveragedEtfPairState.MAKER_CANCEL_PENDING: {
                        LeveragedEtfPairState.RECONCILING,
                        LeveragedEtfPairState.RECOVERY_REQUIRED,
                    },
                    LeveragedEtfPairState.STOCK_HEDGE_PENDING: {
                        LeveragedEtfPairState.RECONCILING,
                        LeveragedEtfPairState.RECOVERY_REQUIRED,
                    },
                    LeveragedEtfPairState.ETF_ROLLBACK_PENDING: {
                        LeveragedEtfPairState.RECONCILING,
                        LeveragedEtfPairState.RECOVERY_REQUIRED,
                    },
                    LeveragedEtfPairState.RECONCILING: {
                        LeveragedEtfPairState.RECONCILING,
                        LeveragedEtfPairState.RECOVERY_REQUIRED,
                    },
                    LeveragedEtfPairState.RECOVERY_REQUIRED: {
                        LeveragedEtfPairState.RECONCILING,
                        LeveragedEtfPairState.RECOVERY_REQUIRED,
                    },
                }
                if target not in legal_targets.get(source, set()):
                    raise JournalConflictError(f"illegal explicit state transition {source.value} -> {target.value}")
            return target

        if action is None:
            raise JournalIntegrityError(f"{event.event_type.value} lacks a typed side-effect identity")
        legal_sources = (
            prepared_sources[action] if event.event_type == JournalEventType.PREPARED else active_sources[action]
        )
        if source not in legal_sources:
            raise JournalConflictError(
                f"illegal {event.event_type.value} source state {source.value} for {action.value}"
            )
        if event.event_type == JournalEventType.PREPARED:
            return pending_state[action]
        if event.event_type in {
            JournalEventType.CANCEL_REQUESTED,
            JournalEventType.CANCEL_CONFIRMED,
        }:
            return LeveragedEtfPairState.MAKER_CANCEL_PENDING
        if event.event_type == JournalEventType.HEDGE_REQUESTED:
            return LeveragedEtfPairState.STOCK_HEDGE_PENDING
        if event.event_type == JournalEventType.ROLLBACK_REQUESTED:
            return LeveragedEtfPairState.ETF_ROLLBACK_PENDING
        if event.event_type == JournalEventType.SUBMIT_UNKNOWN:
            return LeveragedEtfPairState.RECONCILING
        if event.event_type == JournalEventType.REJECTED:
            return {
                JournalSideEffect.ETF_MAKER: LeveragedEtfPairState.PREFLIGHT,
                JournalSideEffect.CANCEL: LeveragedEtfPairState.MAKER_CANCEL_PENDING,
                JournalSideEffect.STOCK_HEDGE: LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                JournalSideEffect.ETF_ROLLBACK: LeveragedEtfPairState.RECOVERY_REQUIRED,
            }[action]
        if event.event_type in {JournalEventType.ACKNOWLEDGED, JournalEventType.ORDER_CREATED}:
            return (
                LeveragedEtfPairState.MAKER_WORKING if action == JournalSideEffect.ETF_MAKER else pending_state[action]
            )
        if event.event_type == JournalEventType.FILL:
            return {
                JournalSideEffect.ETF_MAKER: LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                JournalSideEffect.STOCK_HEDGE: LeveragedEtfPairState.STOCK_HEDGE_PENDING,
                JournalSideEffect.ETF_ROLLBACK: LeveragedEtfPairState.ETF_ROLLBACK_PENDING,
            }[action]

        balanced = cls._exposure_is_balanced(current, decoded, event)
        all_terminal = cls._all_intents_terminal_after(decoded, event)
        hedge_intents_terminal = cls._action_intents_terminal_after(
            decoded,
            event,
            JournalSideEffect.STOCK_HEDGE,
        )
        maker_filled, stock_filled, rollback_filled = cls._exposure_totals(decoded, event)
        completely_filled = (
            maker_filled == current.etf_target_quantity
            and stock_filled == current.stock_target_quantity
            and rollback_filled == 0
        )
        if event.event_type == JournalEventType.HEDGE_CONFIRMED:
            if balanced and all_terminal and completely_filled:
                return LeveragedEtfPairState.COMPLETED
            if balanced and hedge_intents_terminal:
                return LeveragedEtfPairState.MAKER_WORKING
            return LeveragedEtfPairState.STOCK_HEDGE_PENDING
        if event.event_type == JournalEventType.ROLLBACK_CONFIRMED:
            return (
                LeveragedEtfPairState.FAILED_SAFE
                if balanced and all_terminal
                else LeveragedEtfPairState.RECOVERY_REQUIRED
            )
        if isinstance(payload, ReconciliationJournalPayloadV1):
            if payload.outcome in {
                ReconciliationOutcome.NOT_FOUND,
                ReconciliationOutcome.UNKNOWN,
                ReconciliationOutcome.CONFLICT,
            }:
                return LeveragedEtfPairState.RECOVERY_REQUIRED
            if action == JournalSideEffect.ETF_MAKER:
                if maker_filled > 0 and not balanced:
                    return LeveragedEtfPairState.STOCK_HEDGE_PENDING
                if source == LeveragedEtfPairState.MAKER_CANCEL_PENDING and all_terminal and maker_filled == 0:
                    return LeveragedEtfPairState.ABORTED_NO_FILL
                return LeveragedEtfPairState.MAKER_WORKING
            if action == JournalSideEffect.STOCK_HEDGE:
                if balanced and all_terminal and completely_filled:
                    return LeveragedEtfPairState.COMPLETED
                if balanced and hedge_intents_terminal:
                    return LeveragedEtfPairState.MAKER_WORKING
                return LeveragedEtfPairState.STOCK_HEDGE_PENDING
            if action == JournalSideEffect.ETF_ROLLBACK:
                return (
                    LeveragedEtfPairState.FAILED_SAFE
                    if balanced and all_terminal
                    else LeveragedEtfPairState.ETF_ROLLBACK_PENDING
                )
            return LeveragedEtfPairState.MAKER_CANCEL_PENDING
        raise JournalConflictError(f"unsupported authoritative reducer event {event.event_type.value}")

    @staticmethod
    def _terminal_close_reason(state: LeveragedEtfPairState, event: JournalEventV1) -> str:
        if isinstance(event.payload, StateTransitionJournalPayloadV1):
            return event.payload.reason
        return {
            LeveragedEtfPairState.COMPLETED: "all target fills confirmed and pair exposure balanced",
            LeveragedEtfPairState.ABORTED_NO_FILL: "all side effects terminal with no accepted fill",
            LeveragedEtfPairState.FAILED_SAFE: "rollback confirmed and residual pair exposure is zero",
        }[state]

    @classmethod
    def _validate_terminal_snapshot(
        cls,
        snapshot: LeveragedEtfPairExecutorSnapshotV1,
        event: JournalEventV1,
        decoded: Tuple[_DecodedJournalMutation, ...],
    ) -> None:
        if snapshot.state not in _TERMINAL_EXECUTOR_STATES:
            return
        if snapshot.close_reason is None:
            raise JournalIntegrityError("terminal executor state requires a close reason")
        if not cls._all_intents_terminal_after(decoded, event):
            raise JournalIntegrityError("terminal executor state requires every prepared intent to be terminal")
        maker_filled, stock_filled, rollback_filled = cls._exposure_totals(decoded, event)
        if rollback_filled > maker_filled or not cls._exposure_is_balanced(snapshot, decoded, event):
            raise JournalIntegrityError("terminal executor state has residual unhedged pair exposure")
        if snapshot.state == LeveragedEtfPairState.COMPLETED:
            if (
                maker_filled != snapshot.etf_target_quantity
                or stock_filled != snapshot.stock_target_quantity
                or rollback_filled != 0
                or snapshot.etf_remaining_quantity != 0
                or snapshot.stock_remaining_quantity != 0
            ):
                raise JournalIntegrityError("COMPLETED requires both gross target quantities to be fully filled")
        elif snapshot.state == LeveragedEtfPairState.ABORTED_NO_FILL:
            if maker_filled != 0 or stock_filled != 0 or rollback_filled != 0:
                raise JournalIntegrityError("ABORTED_NO_FILL cannot contain an accepted fill")

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

    @staticmethod
    def _validate_global_event_ownership(
        connection: Connection,
        executor_id: str,
        event: JournalEventV1,
    ) -> None:
        owner = (executor_id, event.intent_id)
        if event.connector_name is not None and event.client_order_id is not None:
            rows = connection.execute(
                text("""
                    SELECT executor_id, intent_id
                    FROM LeveragedEtfJournalEvent
                    WHERE connector_name = :connector_name
                      AND client_order_id = :client_order_id
                """),
                {
                    "connector_name": event.connector_name,
                    "client_order_id": event.client_order_id,
                },
            ).mappings()
            if any((row["executor_id"], row["intent_id"]) != owner for row in rows):
                raise JournalConflictError("client order identity is owned by a different executor or intent")
        if event.connector_name is not None and event.trading_pair is not None and event.exchange_order_id is not None:
            rows = connection.execute(
                text("""
                    SELECT executor_id, intent_id
                    FROM LeveragedEtfJournalEvent
                    WHERE connector_name = :connector_name
                      AND trading_pair = :trading_pair
                      AND exchange_order_id = :exchange_order_id
                """),
                {
                    "connector_name": event.connector_name,
                    "trading_pair": event.trading_pair,
                    "exchange_order_id": event.exchange_order_id,
                },
            ).mappings()
            if any((row["executor_id"], row["intent_id"]) != owner for row in rows):
                raise JournalConflictError("exchange order identity is owned by a different executor or intent")

    def append_and_reduce(
        self,
        executor_id: str,
        event: JournalEventV1,
        expected_snapshot: Optional[LeveragedEtfPairExecutorSnapshotV1] = None,
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
                if duplicate.committed.event == event and (
                    expected_snapshot is None or duplicate.snapshot_after == expected_snapshot
                ):
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

            self._validate_persisted_global_ownership(connection)
            self._validate_global_event_ownership(connection, executor_id, event)
            if (
                expected_snapshot is not None
                and expected_snapshot.last_journal_sequence != current.last_journal_sequence + 1
            ):
                raise JournalConflictError("expected snapshot sequence is not current + 1")
            self._validate_intent_transition(event, decoded)
            reduced_snapshot = self._reduce_snapshot(current, event, decoded)
            if expected_snapshot is not None:
                if expected_snapshot.executor_id != executor_id:
                    raise JournalIntegrityError("expected snapshot executor ID does not match repository key")
                if expected_snapshot != reduced_snapshot:
                    raise JournalIntegrityError("reducer snapshot assertion disagrees with authoritative event facts")

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
    """Retired cycle-only runtime repository; historical V1 DTOs remain readable."""

    @staticmethod
    def _fail_closed() -> None:
        raise AnchorIntegrityError(
            "legacy cycle-only anchor repository is disabled; a pair-scoped storage key is required"
        )

    def load_opaque(self, cycle_id: str) -> None:
        self._fail_closed()

    def load(self, cycle_id: str) -> None:
        self._fail_closed()

    def compare_and_set_opaque_checkpoint(self, *args, **kwargs) -> None:
        self._fail_closed()

    def compare_and_set_checkpoint(self, *args, **kwargs) -> None:
        self._fail_closed()

    def finalize_opaque_if_absent(self, *args, **kwargs) -> None:
        self._fail_closed()

    def finalize_if_absent(self, *args, **kwargs) -> None:
        self._fail_closed()

    def append_revision_observation(self, *args, **kwargs) -> None:
        self._fail_closed()

    def revision_observations(self, *args, **kwargs) -> None:
        self._fail_closed()


class PairScopedAnchorRepository(_TransactionalRepository):
    """Canonical pair/cycle-scoped opaque anchor repository."""

    _ANCHOR_SELECT = """
        SELECT pair_id, cycle_id, schema_version, state_kind, revision,
               target_session_date, official_close_utc, deadline_utc,
               evidence_hash, payload_version_field, payload_contract_version,
               payload_json, payload_hash, created_at_utc, updated_at_utc
        FROM LeveragedEtfAnchorState
        WHERE pair_id = :pair_id AND cycle_id = :cycle_id
    """
    _OBSERVATION_SELECT = """
        SELECT pair_id, cycle_id, evidence_hash, observed_at_utc, schema_version,
               payload_version_field, payload_contract_version, payload_json,
               payload_hash
        FROM LeveragedEtfAnchorRevisionObservation
    """

    @staticmethod
    def _require_key(key: AnchorStorageKeyV2) -> AnchorStorageKeyV2:
        if not isinstance(key, AnchorStorageKeyV2):
            raise AnchorIntegrityError("pair-scoped anchor operations require AnchorStorageKeyV2")
        return key

    @staticmethod
    def _require_expected_revision(expected_revision: int, *, minimum: int) -> int:
        if type(expected_revision) is not int or expected_revision < minimum:
            raise AnchorRevisionConflict(f"expected revision must be an integer of at least {minimum}")
        return expected_revision

    @staticmethod
    def _revalidate_checkpoint_envelope(checkpoint: OpaqueAnchorCheckpointV2) -> OpaqueAnchorCheckpointV2:
        if type(checkpoint) is not OpaqueAnchorCheckpointV2:
            raise AnchorIntegrityError("checkpoint writes require an exact OpaqueAnchorCheckpointV2 envelope")
        try:
            checkpoint.validate_metadata()
            primitive = checkpoint.model_dump(mode="python", round_trip=True, warnings="error")
            return OpaqueAnchorCheckpointV2.model_validate(primitive)
        except AnchorIntegrityError:
            raise
        except Exception as exception:
            raise AnchorIntegrityError(f"anchor checkpoint envelope integrity failure: {exception}") from exception

    @classmethod
    def _assert_key_matches(
        cls,
        key: AnchorStorageKeyV2,
        state: Union[
            OpaqueAnchorCheckpointV2,
            OpaqueAnchorFinalizedV2,
            OpaqueAnchorRevisionObservationV2,
        ],
    ) -> None:
        cls._require_key(key)
        if state.key != key:
            raise AnchorIntegrityError("anchor storage key does not match envelope pair/cycle identity")

    @staticmethod
    def _assert_anchor_identity(
        current: OpaqueAnchorStateV2,
        proposed: OpaqueAnchorStateV2,
    ) -> None:
        if current.key != proposed.key:
            raise AnchorIntegrityError("anchor pair/cycle identity cannot change")
        if current.target_session_date != proposed.target_session_date:
            raise AnchorIntegrityError("anchor target session identity cannot change")
        if current.official_close_utc != proposed.official_close_utc:
            raise AnchorIntegrityError("anchor official close identity cannot change")
        if current.deadline_utc != proposed.deadline_utc:
            raise AnchorIntegrityError("anchor deadline identity cannot change")

    @classmethod
    def _decode_anchor_row(cls, row: Mapping[str, Any]) -> OpaqueAnchorStateV2:
        try:
            if row["schema_version"] != 2:
                raise ValueError("anchor storage schema version is unsupported")
            key = AnchorStorageKeyV2(
                pair_id=row["pair_id"],
                cycle_id=row["cycle_id"],
            )
            if row["state_kind"] == "CHECKPOINT":
                if row["evidence_hash"] is not None:
                    raise ValueError("checkpoint storage metadata contains final evidence")
                payload_kind = "ANCHOR_CHECKPOINT"
            elif row["state_kind"] == "FINALIZED":
                if row["evidence_hash"] is None:
                    raise ValueError("finalized storage metadata lacks evidence hash")
                payload_kind = "ANCHOR_RECORD"
            else:
                raise ValueError("anchor state_kind is invalid")
            payload = CanonicalOpaqueAnchorPayloadV2.from_canonical_json(
                kind=payload_kind,
                contract_version_field=row["payload_version_field"],
                contract_version=row["payload_contract_version"],
                payload_json=row["payload_json"],
                payload_hash=row["payload_hash"],
            )
            common = {
                "key": key,
                "target_session_date": row["target_session_date"],
                "official_close_utc": row["official_close_utc"],
                "deadline_utc": row["deadline_utc"],
                "revision": row["revision"],
                "payload": payload,
            }
            if row["state_kind"] == "CHECKPOINT":
                return OpaqueAnchorCheckpointV2(**common)
            return OpaqueAnchorFinalizedV2(
                **common,
                evidence_hash=row["evidence_hash"],
            )
        except AnchorIntegrityError:
            raise
        except Exception as exception:
            raise AnchorIntegrityError(
                f"anchor state hash, version, or payload integrity failure: {exception}"
            ) from exception

    @classmethod
    def _decode_observation_row(
        cls,
        row: Mapping[str, Any],
    ) -> OpaqueAnchorRevisionObservationV2:
        try:
            if row["schema_version"] != 2:
                raise ValueError("anchor observation storage schema version is unsupported")
            payload = CanonicalOpaqueAnchorPayloadV2.from_canonical_json(
                kind="ANCHOR_REVISION_OBSERVATION",
                contract_version_field=row["payload_version_field"],
                contract_version=row["payload_contract_version"],
                payload_json=row["payload_json"],
                payload_hash=row["payload_hash"],
            )
            return OpaqueAnchorRevisionObservationV2(
                key=AnchorStorageKeyV2(
                    pair_id=row["pair_id"],
                    cycle_id=row["cycle_id"],
                ),
                evidence_hash=row["evidence_hash"],
                observed_at_utc=row["observed_at_utc"],
                payload=payload,
            )
        except AnchorIntegrityError:
            raise
        except Exception as exception:
            raise AnchorIntegrityError(
                f"anchor observation hash, version, or payload integrity failure: {exception}"
            ) from exception

    @classmethod
    def _load_opaque_connection(
        cls,
        connection: Connection,
        key: AnchorStorageKeyV2,
    ) -> Optional[OpaqueAnchorStateV2]:
        cls._require_key(key)
        row = (
            connection.execute(
                text(cls._ANCHOR_SELECT),
                {"pair_id": key.pair_id, "cycle_id": key.cycle_id},
            )
            .mappings()
            .one_or_none()
        )
        return None if row is None else cls._decode_anchor_row(row)

    def load_opaque(self, key: AnchorStorageKeyV2) -> Optional[OpaqueAnchorStateV2]:
        self._require_key(key)
        with self._sql_manager.engine.connect() as connection:
            return self._load_opaque_connection(connection, key)

    @staticmethod
    def _payload_parameters(payload: CanonicalOpaqueAnchorPayloadV2) -> dict[str, Any]:
        return {
            "payload_version_field": payload.contract_version_field,
            "payload_contract_version": payload.contract_version,
            "payload_json": payload.payload_json,
            "payload_hash": payload.payload_hash,
        }

    def compare_and_set_opaque_checkpoint(
        self,
        key: AnchorStorageKeyV2,
        checkpoint: OpaqueAnchorCheckpointV2,
        expected_revision: int,
    ) -> OpaqueAnchorCheckpointV2:
        checkpoint = self._revalidate_checkpoint_envelope(checkpoint)
        self._assert_key_matches(key, checkpoint)
        self._require_expected_revision(expected_revision, minimum=0)
        if checkpoint.revision != expected_revision + 1:
            raise AnchorRevisionConflict("checkpoint revision must be exactly expected_revision + 1")
        payload_value = checkpoint.payload.value()
        updated_at_utc = payload_value.get("next_poll_utc")
        if not isinstance(updated_at_utc, str):
            raise AnchorIntegrityError("checkpoint payload lacks canonical next_poll_utc")
        _validate_utc_text(updated_at_utc, "checkpoint updated_at_utc")

        def operation(connection: Connection) -> OpaqueAnchorCheckpointV2:
            current = self._load_opaque_connection(connection, key)
            parameters = {
                "pair_id": key.pair_id,
                "cycle_id": key.cycle_id,
                "schema_version": 2,
                "state_kind": "CHECKPOINT",
                "revision": checkpoint.revision,
                "target_session_date": checkpoint.target_session_date,
                "official_close_utc": checkpoint.official_close_utc,
                "deadline_utc": checkpoint.deadline_utc,
                "updated_at_utc": updated_at_utc,
                "expected_revision": expected_revision,
                **self._payload_parameters(checkpoint.payload),
            }
            if current is None:
                if expected_revision != 0:
                    raise AnchorRevisionConflict("checkpoint does not exist at expected revision")
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfAnchorState (
                            pair_id, cycle_id, schema_version, state_kind, revision,
                            target_session_date, official_close_utc, deadline_utc,
                            evidence_hash, payload_version_field,
                            payload_contract_version, payload_json, payload_hash,
                            created_at_utc, updated_at_utc
                        ) VALUES (
                            :pair_id, :cycle_id, :schema_version, :state_kind,
                            :revision, :target_session_date, :official_close_utc,
                            :deadline_utc, NULL, :payload_version_field,
                            :payload_contract_version, :payload_json, :payload_hash,
                            :updated_at_utc, :updated_at_utc
                        )
                    """),
                    parameters,
                )
                return checkpoint
            if isinstance(current, OpaqueAnchorFinalizedV2):
                raise AnchorIntegrityError("finalized anchor cannot be replaced by a checkpoint")
            self._assert_anchor_identity(current, checkpoint)
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
                        payload_version_field = :payload_version_field,
                        payload_contract_version = :payload_contract_version,
                        payload_json = :payload_json,
                        payload_hash = :payload_hash,
                        updated_at_utc = :updated_at_utc
                    WHERE pair_id = :pair_id
                      AND cycle_id = :cycle_id
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
            raise AnchorRevisionConflict(
                "checkpoint compare-and-set violated pair-scoped storage identity"
            ) from exception

    def finalize_opaque_if_absent(
        self,
        key: AnchorStorageKeyV2,
        record: OpaqueAnchorFinalizedV2,
        expected_revision: int,
    ) -> OpaqueAnchorFinalizedV2:
        self._assert_key_matches(key, record)
        self._require_expected_revision(expected_revision, minimum=1)
        if record.revision != expected_revision:
            raise AnchorRevisionConflict("final record revision must equal a positive expected checkpoint revision")
        payload_value = record.payload.value()
        updated_at_utc = payload_value.get("finalized_at_utc")
        if not isinstance(updated_at_utc, str):
            raise AnchorIntegrityError("finalized payload lacks canonical finalized_at_utc")
        _validate_utc_text(updated_at_utc, "anchor finalized_at_utc")

        def operation(connection: Connection) -> OpaqueAnchorFinalizedV2:
            current = self._load_opaque_connection(connection, key)
            if isinstance(current, OpaqueAnchorFinalizedV2):
                self._assert_anchor_identity(current, record)
                if current.evidence_hash != record.evidence_hash:
                    raise AnchorIntegrityError("different final evidence cannot overwrite an anchor")
                if current.payload != record.payload:
                    raise AnchorIntegrityError("same evidence hash has conflicting finalized payload")
                return current
            if current is None:
                raise AnchorRevisionConflict("anchor finalization requires an existing pair-scoped checkpoint")
            self._assert_anchor_identity(current, record)
            if current.revision != expected_revision:
                raise AnchorRevisionConflict(
                    f"checkpoint revision {current.revision} does not match expected {expected_revision}"
                )
            parameters = {
                "pair_id": key.pair_id,
                "cycle_id": key.cycle_id,
                "state_kind": "FINALIZED",
                "revision": current.revision,
                "target_session_date": record.target_session_date,
                "official_close_utc": record.official_close_utc,
                "deadline_utc": record.deadline_utc,
                "evidence_hash": record.evidence_hash,
                "updated_at_utc": updated_at_utc,
                "expected_revision": expected_revision,
                **self._payload_parameters(record.payload),
            }
            updated = connection.execute(
                text("""
                    UPDATE LeveragedEtfAnchorState
                    SET state_kind = :state_kind,
                        revision = :revision,
                        target_session_date = :target_session_date,
                        official_close_utc = :official_close_utc,
                        deadline_utc = :deadline_utc,
                        evidence_hash = :evidence_hash,
                        payload_version_field = :payload_version_field,
                        payload_contract_version = :payload_contract_version,
                        payload_json = :payload_json,
                        payload_hash = :payload_hash,
                        updated_at_utc = :updated_at_utc
                    WHERE pair_id = :pair_id
                      AND cycle_id = :cycle_id
                      AND state_kind = 'CHECKPOINT'
                      AND revision = :expected_revision
                """),
                parameters,
            )
            if updated.rowcount != 1:
                raise AnchorRevisionConflict("anchor finalization lost a concurrent race")
            return record

        try:
            return self._write(operation)
        except IntegrityError as exception:
            raise AnchorIntegrityError("anchor finalization violated immutable pair-scoped identity") from exception

    def append_opaque_revision_observation(
        self,
        key: AnchorStorageKeyV2,
        observation: OpaqueAnchorRevisionObservationV2,
    ) -> None:
        self._assert_key_matches(key, observation)

        def operation(connection: Connection) -> None:
            current = self._load_opaque_connection(connection, key)
            if current is None:
                raise AnchorIntegrityError(f"anchor pair/cycle {key.pair_id}/{key.cycle_id} does not exist")
            if not isinstance(current, OpaqueAnchorFinalizedV2):
                raise AnchorIntegrityError("revision observations require a finalized anchor")
            parameters = {
                "pair_id": key.pair_id,
                "cycle_id": key.cycle_id,
                "evidence_hash": observation.evidence_hash,
                "observed_at_utc": observation.observed_at_utc,
                "schema_version": 2,
                **self._payload_parameters(observation.payload),
            }
            existing = (
                connection.execute(
                    text(self._OBSERVATION_SELECT + """
                      WHERE pair_id = :pair_id
                        AND cycle_id = :cycle_id
                        AND evidence_hash = :evidence_hash
                        AND observed_at_utc = :observed_at_utc
                    """),
                    parameters,
                )
                .mappings()
                .one_or_none()
            )
            if existing is not None:
                persisted = self._decode_observation_row(existing)
                if persisted != observation:
                    raise AnchorIntegrityError("anchor revision identity has conflicting immutable payload")
                return None
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfAnchorRevisionObservation (
                        pair_id, cycle_id, evidence_hash, observed_at_utc,
                        schema_version, payload_version_field,
                        payload_contract_version, payload_json, payload_hash
                    ) VALUES (
                        :pair_id, :cycle_id, :evidence_hash, :observed_at_utc,
                        :schema_version, :payload_version_field,
                        :payload_contract_version, :payload_json, :payload_hash
                    )
                """),
                parameters,
            )
            return None

        try:
            self._write(operation)
        except IntegrityError as exception:
            raise AnchorIntegrityError(
                "anchor revision observation violated append-only pair-scoped identity"
            ) from exception

    def opaque_revision_observations(
        self,
        key: AnchorStorageKeyV2,
    ) -> Tuple[OpaqueAnchorRevisionObservationV2, ...]:
        self._require_key(key)
        with self._sql_manager.engine.connect() as connection:
            rows = (
                connection.execute(
                    text(self._OBSERVATION_SELECT + """
                          WHERE pair_id = :pair_id AND cycle_id = :cycle_id
                          ORDER BY observed_at_utc, evidence_hash
                        """),
                    {"pair_id": key.pair_id, "cycle_id": key.cycle_id},
                )
                .mappings()
                .all()
            )
        return tuple(self._decode_observation_row(row) for row in rows)
