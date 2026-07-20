from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Literal, NamedTuple, Optional, Tuple, Union

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
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _utc_text(value: datetime) -> str:
    return value.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _validate_utc_text(value: str, field_name: str) -> str:
    if _CANONICAL_UTC_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be canonical UTC")
    return value


def _verify_canonical_json(payload_json: str, payload_hash: str, label: str) -> Any:
    try:
        value = json.loads(payload_json)
    except (TypeError, json.JSONDecodeError) as exception:
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
        _verify_canonical_json(self.payload_json, self.payload_hash, f"{self.kind} payload")
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
        return json.loads(self.payload_json)


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
    payload: CanonicalOpaquePayload
    created_at_utc: CanonicalUtcInstant

    @model_validator(mode="after")
    def validate_identity_fields(self) -> JournalEventV1:
        if self.event_type == JournalEventType.PREPARED:
            if self.intent_id is None or self.idempotency_key is None:
                raise ValueError("PREPARED requires intent_id and idempotency_key")
        if self.exchange_trade_id is not None and (self.connector_name is None or self.trading_pair is None):
            raise ValueError("exchange_trade_id requires connector_name and trading_pair")
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


class StrategyReservationV1(CanonicalWireModel):
    schema_version: SchemaVersionV1 = 1
    reservation_id: StableIdentifier
    executor_id: StableIdentifier
    reservation_key: StableIdentifier
    connector_name: StableIdentifier
    trading_pair: TradingPair
    leg: Literal["ETF", "STOCK"]
    quantity: CanonicalNonNegativeDecimal
    leverage: StrictInt = Field(ge=1)
    notional_cap: CanonicalPositiveDecimal
    payload: CanonicalOpaquePayload
    created_at_utc: CanonicalUtcInstant
    updated_at_utc: CanonicalUtcInstant
    released_at_utc: Optional[CanonicalUtcInstant]


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
    ):
        self._sql_manager = sql_manager
        self._before_commit = before_commit

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
        value = event.payload.value()
        return isinstance(value, Mapping) and value.get("terminal") is True

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

    @staticmethod
    def _logical_exposure_key(event: JournalEventV1) -> Optional[Tuple[str, str, str]]:
        if event.event_type != JournalEventType.PREPARED:
            return None
        value = event.payload.value()
        if not isinstance(value, Mapping) or value.get("exposure_increasing") is not True:
            return None
        required = ("operation", "leg", "logical_quantity")
        if any(not isinstance(value.get(field), str) or not value[field] for field in required):
            raise JournalIntegrityError("exposure-increasing PREPARED payload lacks logical identity")
        return value["operation"], value["leg"], value["logical_quantity"]

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

    @classmethod
    def _validate_snapshot_transition(
        cls,
        current: LeveragedEtfPairExecutorSnapshotV1,
        next_snapshot: LeveragedEtfPairExecutorSnapshotV1,
    ) -> None:
        if next_snapshot.last_journal_sequence != current.last_journal_sequence + 1:
            raise JournalConflictError("next snapshot journal sequence is not current + 1")
        for field_name in cls._IMMUTABLE_SNAPSHOT_FIELDS:
            if getattr(current, field_name) != getattr(next_snapshot, field_name):
                raise JournalIntegrityError(f"snapshot immutable field {field_name} changed")
        if next_snapshot.updated_at_utc < current.updated_at_utc:
            raise JournalIntegrityError("snapshot updated_at_utc moved backwards")

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
        with self._sql_manager.engine.connect() as connection:
            return self._load_snapshot_connection(connection, executor_id)

    def append_and_reduce(
        self,
        executor_id: str,
        event: JournalEventV1,
        next_snapshot: LeveragedEtfPairExecutorSnapshotV1,
    ) -> CommittedJournalEventV1:
        def operation(connection: Connection):
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
                if duplicate.committed.event == event and duplicate.snapshot_after == next_snapshot:
                    return duplicate.committed
                raise JournalIntegrityError("event identity has a conflicting payload")

            if event.exchange_trade_id is not None:
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
                    if duplicate.committed.event.payload == event.payload:
                        return duplicate.committed
                    raise JournalIntegrityError("exchange trade identity has a conflicting payload")

            current = self._load_snapshot_connection(connection, executor_id)
            if current is None:
                raise JournalConflictError(f"executor {executor_id} does not exist")
            if next_snapshot.executor_id != executor_id:
                raise JournalIntegrityError("next snapshot executor ID does not match repository key")
            decoded = self._decoded_events(connection, executor_id)
            self._validate_intent_transition(event, decoded)
            self._validate_snapshot_transition(current, next_snapshot)

            sequence = current.last_journal_sequence + 1
            mutation = {
                "schema_version": 1,
                "event": event.model_dump(mode="json"),
                "snapshot_before": current.model_dump(mode="json"),
                "snapshot_before_hash": current.canonical_sha256(),
                "snapshot_after": next_snapshot.model_dump(mode="json"),
                "snapshot_after_hash": next_snapshot.canonical_sha256(),
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
            snapshot_json = next_snapshot.canonical_json()
            next_json = next_snapshot.model_dump(mode="json")
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
        with self._sql_manager.engine.connect() as connection:
            return tuple(mutation.committed for mutation in self._decoded_events(connection, executor_id))

    def replay(self, executor_id: str) -> LeveragedEtfPairExecutorSnapshotV1:
        with self._sql_manager.engine.connect() as connection:
            persisted = self._load_snapshot_connection(connection, executor_id)
            if persisted is None:
                raise JournalConflictError(f"executor {executor_id} does not exist")
            mutations = self._decoded_events(connection, executor_id)
        if not mutations:
            if persisted.last_journal_sequence != 0:
                raise JournalIntegrityError("snapshot sequence has no journal history")
            return persisted
        reduced = mutations[0].snapshot_before
        if reduced.last_journal_sequence != 0:
            raise JournalIntegrityError("journal replay does not begin at sequence zero")
        for expected_sequence, mutation in enumerate(mutations, start=1):
            if mutation.committed.sequence != expected_sequence:
                raise JournalIntegrityError("journal sequence is not contiguous")
            if mutation.snapshot_before != reduced:
                raise JournalIntegrityError("journal snapshot hash chain diverges")
            reduced = mutation.snapshot_after
        if reduced != persisted:
            raise JournalIntegrityError("persisted snapshot disagrees with journal replay")
        return reduced

    def incomplete_intents(self, executor_id: Optional[str] = None) -> Tuple[IncompleteIntentV1, ...]:
        with self._sql_manager.engine.connect() as connection:
            rows = self._event_rows(connection, executor_id)
        by_executor = {}
        for row in rows:
            mutation = self._decode_event_row(row)
            by_executor.setdefault(mutation.committed.executor_id, []).append(mutation)
        result = []
        for current_executor_id, mutations in by_executor.items():
            result.extend(self._incomplete_from_decoded(current_executor_id, tuple(mutations)))
        return tuple(sorted(result, key=lambda value: (value.executor_id, value.last_sequence, value.intent_id)))

    def incomplete_executors(self) -> Tuple[LeveragedEtfPairExecutorSnapshotV1, ...]:
        with self._sql_manager.engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        self._SNAPSHOT_SELECT.replace(
                            "WHERE executor_id = :executor_id",
                            "ORDER BY executor_id",
                        )
                    )
                )
                .mappings()
                .all()
            )
        snapshots = tuple(self._decode_snapshot_row(row) for row in rows)
        incomplete_intent_ids = {intent.executor_id for intent in self.incomplete_intents()}
        return tuple(
            snapshot
            for snapshot in snapshots
            if snapshot.state not in _TERMINAL_EXECUTOR_STATES or snapshot.executor_id in incomplete_intent_ids
        )

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
            serialized = reservation.model_dump(mode="json")
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
            if current.revision != expected_revision:
                raise AnchorRevisionConflict(
                    f"checkpoint revision {current.revision} does not match expected {expected_revision}"
                )
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
