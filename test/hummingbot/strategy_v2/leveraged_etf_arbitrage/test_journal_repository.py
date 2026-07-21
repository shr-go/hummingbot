import hashlib
import json
import math
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.model.leveraged_etf_persistence import SQLITE_GUARD_DDL
from hummingbot.model.leveraged_etf_repository import (
    AnchorIntegrityError,
    AnchorPollingCheckpointV1,
    AnchorRecordV1,
    AnchorRepositoryV1,
    AnchorRevisionConflict,
    AnchorRevisionObservationV1,
    AnchorStorageKeyV2,
    CanonicalOpaqueAnchorPayloadV2,
    CanonicalOpaquePayload,
    JournalConflictError,
    JournalEventType,
    JournalEventV1,
    JournalIntegrityError,
    LeveragedEtfJournalRepository,
    OpaqueAnchorCheckpointV1,
    OpaqueAnchorCheckpointV2,
    OpaqueAnchorFinalizedV1,
    OpaqueAnchorFinalizedV2,
    OpaqueAnchorRevisionObservationV2,
    PairScopedAnchorRepository,
    StrategyReservationV1,
)
from hummingbot.model.sql_connection_manager import SQLConnectionManager, SQLConnectionType
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import LeveragedEtfPairExecutorSnapshotV1

CREATED_AT = "2026-07-17T14:01:02.000000Z"
UPDATED_AT = "2026-07-17T14:01:03.123456Z"
LATER_AT = "2026-07-17T14:01:04.123456Z"
DEADLINE_AT = "2026-07-17T14:02:00.000000Z"

_STATE_FIELDS = (
    "schema_version",
    "executor_id",
    "state",
    "etf_submitted_quantity",
    "etf_filled_quantity",
    "etf_remaining_quantity",
    "stock_submitted_quantity",
    "stock_filled_quantity",
    "stock_remaining_quantity",
    "hedge_dust_quantity",
    "maker_order_ids",
    "stock_order_ids",
    "leverage_reservation",
    "updated_at_utc",
    "close_reason",
    "last_journal_sequence",
)


def _contract_vectors() -> dict:
    relative_path = Path("contracts/equity_leveraged_etf/v1/round_trip_vectors.json")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise AssertionError(f"F001 contract document not found: {relative_path}")


@pytest.fixture(scope="module")
def vectors() -> dict:
    return _contract_vectors()


def _client_config() -> ClientConfigAdapter:
    return ClientConfigAdapter(ClientConfigMap())


def _open_manager(db_path: Path) -> SQLConnectionManager:
    return SQLConnectionManager(
        _client_config(),
        SQLConnectionType.TRADE_FILLS,
        db_path=str(db_path),
    )


@pytest.fixture
def manager(tmp_path: Path):
    value = _open_manager(tmp_path / "repository.sqlite")
    try:
        yield value
    finally:
        value.engine.dispose()


class _V1WireAnchorStorageAdapter:
    """Test-only bridge proving historical V1 DTOs over the canonical pair-scoped store."""

    def __init__(self, manager: SQLConnectionManager, pair_id: str = "sndk_snxx"):
        self.repository = PairScopedAnchorRepository(manager)
        self.pair_id = pair_id

    def _key(self, cycle_id: str) -> AnchorStorageKeyV2:
        return AnchorStorageKeyV2(pair_id=self.pair_id, cycle_id=cycle_id)

    def _payload(self, payload: CanonicalOpaquePayload) -> CanonicalOpaqueAnchorPayloadV2:
        value = {**payload.value(), "pair_id": self.pair_id}
        return CanonicalOpaqueAnchorPayloadV2.from_value(
            kind=payload.kind,
            contract_version_field="schema_version",
            contract_version=payload.schema_version,
            value=value,
        )

    @staticmethod
    def _legacy_payload(payload: CanonicalOpaqueAnchorPayloadV2) -> CanonicalOpaquePayload:
        value = dict(payload.value())
        value.pop("pair_id")
        if payload.kind == "ANCHOR_RECORD" and "yahoo_stock_symbol" in value:
            value.pop("official_close_utc", None)
        return CanonicalOpaquePayload.from_value(
            schema_version=payload.contract_version,
            kind=payload.kind,
            value=value,
        )

    def compare_and_set_opaque_checkpoint(
        self,
        checkpoint: OpaqueAnchorCheckpointV1,
        expected_revision: int,
    ) -> OpaqueAnchorCheckpointV1:
        key = self._key(checkpoint.cycle_id)
        stored = OpaqueAnchorCheckpointV2(
            key=key,
            target_session_date=checkpoint.target_session_date,
            official_close_utc=checkpoint.model_dump(mode="json")["official_close_utc"],
            deadline_utc=checkpoint.model_dump(mode="json")["deadline_utc"],
            revision=checkpoint.revision,
            payload=self._payload(checkpoint.payload),
        )
        self.repository.compare_and_set_opaque_checkpoint(key, stored, expected_revision)
        return checkpoint

    def compare_and_set_checkpoint(
        self,
        checkpoint: AnchorPollingCheckpointV1,
        expected_revision: int,
    ) -> AnchorPollingCheckpointV1:
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
        key = self._key(record.cycle_id)
        serialized = record.model_dump(mode="json")
        stored = OpaqueAnchorFinalizedV2(
            key=key,
            target_session_date=record.target_session_date,
            official_close_utc=serialized["official_close_utc"],
            deadline_utc=serialized["deadline_utc"],
            revision=record.revision,
            evidence_hash=record.evidence_hash,
            payload=self._payload(record.payload),
        )
        finalized = self.repository.finalize_opaque_if_absent(key, stored, expected_revision)
        return OpaqueAnchorFinalizedV1(
            cycle_id=record.cycle_id,
            target_session_date=finalized.target_session_date,
            official_close_utc=finalized.official_close_utc,
            deadline_utc=finalized.deadline_utc,
            revision=finalized.revision,
            evidence_hash=finalized.evidence_hash,
            payload=self._legacy_payload(finalized.payload),
        )

    def finalize_if_absent(
        self,
        record: AnchorRecordV1,
        expected_revision: int,
    ) -> AnchorRecordV1:
        serialized = record.model_dump(mode="json")
        current = self.repository.load_opaque(self._key(record.cycle_id))
        if current is None:
            raise AnchorRevisionConflict("historical V1 finalization requires a stored checkpoint")
        finalized = self.finalize_opaque_if_absent(
            OpaqueAnchorFinalizedV1(
                cycle_id=record.cycle_id,
                target_session_date=record.target_session_date,
                official_close_utc=current.official_close_utc,
                deadline_utc=serialized["deadline_utc"],
                revision=expected_revision,
                evidence_hash=record.evidence_hash,
                payload=CanonicalOpaquePayload.from_value(
                    1,
                    "ANCHOR_RECORD",
                    {**serialized, "official_close_utc": current.official_close_utc},
                ),
            ),
            expected_revision,
        )
        return AnchorRecordV1.model_validate(finalized.payload.value())

    def load_opaque(self, cycle_id: str):
        stored = self.repository.load_opaque(self._key(cycle_id))
        if stored is None:
            return None
        common = {
            "cycle_id": cycle_id,
            "target_session_date": stored.target_session_date,
            "official_close_utc": stored.official_close_utc,
            "deadline_utc": stored.deadline_utc,
            "revision": stored.revision,
            "payload": self._legacy_payload(stored.payload),
        }
        if isinstance(stored, OpaqueAnchorCheckpointV2):
            return OpaqueAnchorCheckpointV1(**common)
        return OpaqueAnchorFinalizedV1(**common, evidence_hash=stored.evidence_hash)

    def load(self, cycle_id: str):
        stored = self.load_opaque(cycle_id)
        if stored is None:
            return None
        value = stored.payload.value()
        if isinstance(stored, OpaqueAnchorCheckpointV1):
            return AnchorPollingCheckpointV1.model_validate(value)
        return AnchorRecordV1.model_validate(value)

    def append_revision_observation(self, cycle_id: str, evidence_hash: str, observed_at: str) -> None:
        key = self._key(cycle_id)
        value = {
            "schema_version": 1,
            "pair_id": self.pair_id,
            "cycle_id": cycle_id,
            "evidence_hash": evidence_hash,
            "observed_at_utc": observed_at,
        }
        self.repository.append_opaque_revision_observation(
            key,
            OpaqueAnchorRevisionObservationV2(
                key=key,
                evidence_hash=evidence_hash,
                observed_at_utc=observed_at,
                payload=CanonicalOpaqueAnchorPayloadV2.from_value(
                    kind="ANCHOR_REVISION_OBSERVATION",
                    contract_version_field="schema_version",
                    contract_version=1,
                    value=value,
                ),
            ),
        )

    def revision_observations(self, cycle_id: str):
        values = self.repository.opaque_revision_observations(self._key(cycle_id))
        return tuple(
            AnchorRevisionObservationV1(
                cycle_id=cycle_id,
                evidence_hash=value.evidence_hash,
                observed_at_utc=value.observed_at_utc,
            )
            for value in values
        )


def _initial_snapshot(vectors: dict) -> LeveragedEtfPairExecutorSnapshotV1:
    payload = dict(vectors["fixtures"]["executor_active"])
    payload.update(
        {
            "state": "CREATED",
            "etf_submitted_quantity": "0",
            "etf_filled_quantity": "0",
            "etf_remaining_quantity": payload["etf_target_quantity"],
            "stock_submitted_quantity": "0",
            "stock_filled_quantity": "0",
            "stock_remaining_quantity": payload["stock_target_quantity"],
            "maker_order_ids": [],
            "stock_order_ids": [],
            "updated_at_utc": CREATED_AT,
            "last_journal_sequence": 0,
        }
    )
    return LeveragedEtfPairExecutorSnapshotV1.model_validate(payload)


def _prepared_snapshot(snapshot: LeveragedEtfPairExecutorSnapshotV1) -> LeveragedEtfPairExecutorSnapshotV1:
    payload = snapshot.model_dump(mode="json")
    payload.update(
        {
            "state": "MAKER_SUBMITTING",
            "updated_at_utc": UPDATED_AT,
            "last_journal_sequence": 1,
        }
    )
    return LeveragedEtfPairExecutorSnapshotV1.model_validate(payload)


def _snapshot_at(
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
    sequence: int,
    state: str | None = None,
) -> LeveragedEtfPairExecutorSnapshotV1:
    payload = snapshot.model_dump(mode="json")
    payload["last_journal_sequence"] = sequence
    payload["updated_at_utc"] = UPDATED_AT
    if state is not None:
        payload["state"] = state
    return LeveragedEtfPairExecutorSnapshotV1.model_validate(payload)


def _snapshot_with_leg_fill(
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
    *,
    sequence: int,
    state: str,
    leg: str,
    submitted: str,
    filled: str,
    client_order_id: str | None = None,
    exchange_order_id: str | None = None,
    updated_at_utc: str = UPDATED_AT,
) -> LeveragedEtfPairExecutorSnapshotV1:
    payload = snapshot.model_dump(mode="json")
    target_field = "etf_target_quantity" if leg == "ETF" else "stock_target_quantity"
    prefix = "etf" if leg == "ETF" else "stock"
    payload.update(
        {
            "state": state,
            f"{prefix}_submitted_quantity": submitted,
            f"{prefix}_filled_quantity": filled,
            f"{prefix}_remaining_quantity": str(Decimal(payload[target_field]) - Decimal(filled)),
            "updated_at_utc": updated_at_utc,
            "last_journal_sequence": sequence,
        }
    )
    if exchange_order_id is not None:
        assert client_order_id is not None
        order_field = "maker_order_ids" if leg == "ETF" else "stock_order_ids"
        payload[order_field] = [
            {
                "sequence": sequence,
                "client_order_id": client_order_id,
                "exchange_order_id": exchange_order_id,
            }
        ]
    return LeveragedEtfPairExecutorSnapshotV1.model_validate(payload)


def _state_payload(snapshot: LeveragedEtfPairExecutorSnapshotV1) -> dict:
    serialized = snapshot.model_dump(mode="json")
    return {field_name: serialized[field_name] for field_name in _STATE_FIELDS}


def _side_effect_identity(
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
    *,
    action: str = "ETF_MAKER",
    leg: str = "ETF",
    logical_quantity: str = "40",
    order_quantity: str | None = None,
    attempt: int = 1,
    intent_id: str = "intent-maker-1",
    client_order_id: str = "exec-sndk-snxx-0001-maker-1",
) -> dict:
    serialized = snapshot.model_dump(mode="json")
    operation = serialized["operation"]
    connector_name = serialized["etf_connector_name"] if leg == "ETF" else serialized["stock_connector_name"]
    trading_pair = serialized["etf_trading_pair"] if leg == "ETF" else serialized["stock_trading_pair"]
    return {
        "schema_version": 1,
        "executor_id": snapshot.executor_id,
        "operation": operation,
        "action": action,
        "leg": leg,
        "logical_quantity": logical_quantity,
        "order_quantity": logical_quantity if order_quantity is None else order_quantity,
        "attempt": attempt,
        "intent_id": intent_id,
        "idempotency_key": f"{snapshot.executor_id}:{operation}:{leg}:{logical_quantity}:{attempt}",
        "connector_name": connector_name,
        "trading_pair": trading_pair,
        "client_order_id": client_order_id,
        "deadline_utc": DEADLINE_AT,
    }


def _journal_payload(kind: str, value: dict) -> CanonicalOpaquePayload:
    return CanonicalOpaquePayload.from_value(
        schema_version=1,
        kind=kind,
        value={"schema_version": 1, "kind": kind, **value},
    )


def _payload_value(payload) -> dict:
    if hasattr(payload, "value"):
        return payload.value()
    return payload.model_dump(mode="json")


def _prepared_event(
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
    next_snapshot: LeveragedEtfPairExecutorSnapshotV1,
    *,
    payload_value: str = "40",
    action: str = "ETF_MAKER",
    leg: str = "ETF",
    attempt: int = 1,
    event_id: str = "event-maker-prepared-1",
    intent_id: str = "intent-maker-1",
    client_order_id: str = "exec-sndk-snxx-0001-maker-1",
    identity_overrides: dict | None = None,
    payload_overrides: dict | None = None,
) -> JournalEventV1:
    identity = _side_effect_identity(
        snapshot,
        action=action,
        leg=leg,
        logical_quantity=payload_value,
        attempt=attempt,
        intent_id=intent_id,
        client_order_id=client_order_id,
    )
    identity.update(identity_overrides or {})
    payload_value_dict = {
        "identity": identity,
        "next_state": _state_payload(next_snapshot),
        **(payload_overrides or {}),
    }
    return JournalEventV1(
        event_id=event_id,
        event_type=JournalEventType.PREPARED,
        intent_id=identity.get("intent_id"),
        idempotency_key=identity.get("idempotency_key"),
        connector_name=identity.get("connector_name"),
        trading_pair=identity.get("trading_pair"),
        client_order_id=identity.get("client_order_id"),
        payload=_journal_payload("PREPARED", payload_value_dict),
        created_at_utc=next_snapshot.model_dump(mode="json")["updated_at_utc"],
    )


def _followup_event(
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
    next_snapshot: LeveragedEtfPairExecutorSnapshotV1,
    event_type: JournalEventType,
    event_id: str,
    *,
    action: str = "ETF_MAKER",
    leg: str = "ETF",
    logical_quantity: str = "40",
    attempt: int = 1,
    intent_id: str = "intent-maker-1",
    client_order_id: str = "exec-sndk-snxx-0001-maker-1",
    terminal: bool = False,
    exchange_order_id: str | None = None,
    exchange_trade_id: str | None = None,
    fill_quantity: str = "1",
    cumulative_filled_quantity: str = "1",
) -> JournalEventV1:
    identity = _side_effect_identity(
        snapshot,
        action=action,
        leg=leg,
        logical_quantity=logical_quantity,
        attempt=attempt,
        intent_id=intent_id,
        client_order_id=client_order_id,
    )
    payload_kind = {
        JournalEventType.ACKNOWLEDGED: "ACKNOWLEDGED",
        JournalEventType.REJECTED: "REJECTED",
        JournalEventType.SUBMIT_UNKNOWN: "SUBMIT_UNKNOWN",
        JournalEventType.ORDER_CREATED: "ORDER_CREATED",
        JournalEventType.FILL: "FILL",
        JournalEventType.CANCEL_REQUESTED: "CANCEL",
        JournalEventType.CANCEL_CONFIRMED: "CANCEL",
        JournalEventType.HEDGE_REQUESTED: "HEDGE",
        JournalEventType.HEDGE_CONFIRMED: "HEDGE",
        JournalEventType.ROLLBACK_REQUESTED: "ROLLBACK",
        JournalEventType.ROLLBACK_CONFIRMED: "ROLLBACK",
        JournalEventType.RECONCILIATION: "RECONCILIATION",
    }[event_type]
    payload_value = {
        "identity": identity,
        "next_state": _state_payload(next_snapshot),
    }
    if event_type == JournalEventType.SUBMIT_UNKNOWN:
        payload_value["uncertainty_started_at_utc"] = next_snapshot.model_dump(mode="json")["updated_at_utc"]
    elif event_type == JournalEventType.REJECTED:
        payload_value["reason"] = "connector rejected request"
    elif event_type == JournalEventType.ORDER_CREATED:
        payload_value["exchange_order_id"] = exchange_order_id
    elif event_type == JournalEventType.FILL:
        payload_value.update(
            {
                "exchange_order_id": exchange_order_id,
                "exchange_trade_id": exchange_trade_id,
                "price": "30",
                "fill_quantity": fill_quantity,
                "cumulative_filled_quantity": cumulative_filled_quantity,
                "outcome": "FILLED" if terminal else "PARTIAL",
                "terminal": terminal,
            }
        )
    elif event_type in {JournalEventType.CANCEL_REQUESTED, JournalEventType.CANCEL_CONFIRMED}:
        payload_value.update(
            {
                "phase": "REQUESTED" if event_type == JournalEventType.CANCEL_REQUESTED else "CONFIRMED",
                "final_cumulative_filled_quantity": cumulative_filled_quantity,
            }
        )
    elif event_type in {JournalEventType.HEDGE_REQUESTED, JournalEventType.HEDGE_CONFIRMED}:
        payload_value["phase"] = "REQUESTED" if event_type == JournalEventType.HEDGE_REQUESTED else "CONFIRMED"
    elif event_type in {JournalEventType.ROLLBACK_REQUESTED, JournalEventType.ROLLBACK_CONFIRMED}:
        payload_value["phase"] = "REQUESTED" if event_type == JournalEventType.ROLLBACK_REQUESTED else "CONFIRMED"
    elif event_type == JournalEventType.RECONCILIATION:
        payload_value["outcome"] = "CANCELED" if terminal else "NEW"
        payload_value["terminal"] = terminal
    return JournalEventV1(
        event_id=event_id,
        event_type=event_type,
        intent_id=identity["intent_id"],
        idempotency_key=identity["idempotency_key"],
        connector_name=identity["connector_name"],
        trading_pair=identity["trading_pair"],
        client_order_id=identity["client_order_id"],
        exchange_order_id=exchange_order_id,
        exchange_trade_id=exchange_trade_id,
        payload=_journal_payload(payload_kind, payload_value),
        created_at_utc=next_snapshot.model_dump(mode="json")["updated_at_utc"],
    )


def _fact_event(
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
    event_type: JournalEventType,
    event_id: str,
    *,
    action: str = "ETF_MAKER",
    leg: str = "ETF",
    logical_quantity: str = "1",
    order_quantity: str | None = None,
    attempt: int = 1,
    intent_id: str = "intent-maker-fact-1",
    client_order_id: str = "exec-sndk-snxx-0001-maker-fact-1",
    deadline_utc: str = DEADLINE_AT,
    created_at_utc: str = UPDATED_AT,
    exchange_order_id: str | None = None,
    exchange_trade_id: str | None = None,
    fill_quantity: str = "1",
    order_cumulative_filled_quantity: str = "1",
    leg_cumulative_filled_quantity: str = "1",
    outcome: str | None = None,
    target_intent_id: str | None = None,
    target_client_order_id: str | None = None,
    target_exchange_order_id: str | None = None,
) -> JournalEventV1:
    """Build a v1 journal event containing exchange facts, never a next snapshot."""

    identity = _side_effect_identity(
        snapshot,
        action=action,
        leg=leg,
        logical_quantity=logical_quantity,
        order_quantity=order_quantity,
        attempt=attempt,
        intent_id=intent_id,
        client_order_id=client_order_id,
    )
    identity["deadline_utc"] = deadline_utc
    payload_kind = {
        JournalEventType.PREPARED: "PREPARED",
        JournalEventType.ACKNOWLEDGED: "ACKNOWLEDGED",
        JournalEventType.REJECTED: "REJECTED",
        JournalEventType.SUBMIT_UNKNOWN: "SUBMIT_UNKNOWN",
        JournalEventType.ORDER_CREATED: "ORDER_CREATED",
        JournalEventType.FILL: "FILL",
        JournalEventType.CANCEL_REQUESTED: "CANCEL",
        JournalEventType.CANCEL_CONFIRMED: "CANCEL",
        JournalEventType.HEDGE_REQUESTED: "HEDGE",
        JournalEventType.HEDGE_CONFIRMED: "HEDGE",
        JournalEventType.ROLLBACK_REQUESTED: "ROLLBACK",
        JournalEventType.ROLLBACK_CONFIRMED: "ROLLBACK",
        JournalEventType.RECONCILIATION: "RECONCILIATION",
    }[event_type]
    payload_value = {"schema_version": 1, "kind": payload_kind, "identity": identity}
    if event_type == JournalEventType.REJECTED:
        payload_value["reason"] = "connector rejected request"
    elif event_type == JournalEventType.SUBMIT_UNKNOWN:
        payload_value["uncertainty_started_at_utc"] = created_at_utc
    elif event_type == JournalEventType.ORDER_CREATED:
        payload_value["exchange_order_id"] = exchange_order_id
    elif event_type == JournalEventType.FILL:
        payload_value.update(
            {
                "exchange_order_id": exchange_order_id,
                "exchange_trade_id": exchange_trade_id,
                "price": "30",
                "fill_quantity": fill_quantity,
                "order_cumulative_filled_quantity": order_cumulative_filled_quantity,
                "leg_cumulative_filled_quantity": leg_cumulative_filled_quantity,
                "outcome": outcome or "FILLED",
            }
        )
    elif event_type in {JournalEventType.CANCEL_REQUESTED, JournalEventType.CANCEL_CONFIRMED}:
        payload_value.update(
            {
                "phase": "REQUESTED" if event_type == JournalEventType.CANCEL_REQUESTED else "CONFIRMED",
                "target_intent_id": target_intent_id,
                "target_client_order_id": target_client_order_id,
                "target_exchange_order_id": target_exchange_order_id,
                "final_order_cumulative_filled_quantity": order_cumulative_filled_quantity,
            }
        )
    elif event_type in {JournalEventType.HEDGE_REQUESTED, JournalEventType.HEDGE_CONFIRMED}:
        payload_value["phase"] = "REQUESTED" if event_type == JournalEventType.HEDGE_REQUESTED else "CONFIRMED"
    elif event_type in {JournalEventType.ROLLBACK_REQUESTED, JournalEventType.ROLLBACK_CONFIRMED}:
        payload_value["phase"] = "REQUESTED" if event_type == JournalEventType.ROLLBACK_REQUESTED else "CONFIRMED"
    elif event_type == JournalEventType.RECONCILIATION:
        payload_value.update(
            {
                "exchange_order_id": exchange_order_id,
                "order_cumulative_filled_quantity": order_cumulative_filled_quantity,
                "outcome": outcome or "NEW",
            }
        )
    return JournalEventV1.model_validate(
        {
            "schema_version": 1,
            "event_id": event_id,
            "event_type": event_type.value,
            "intent_id": identity["intent_id"],
            "idempotency_key": identity["idempotency_key"],
            "connector_name": identity["connector_name"],
            "trading_pair": identity["trading_pair"],
            "client_order_id": identity["client_order_id"],
            "exchange_order_id": exchange_order_id,
            "exchange_trade_id": exchange_trade_id,
            "payload": payload_value,
            "created_at_utc": created_at_utc,
        }
    )


def _reservation(
    reservation_id: str = "reservation-1",
    reservation_key: str | None = None,
    *,
    executor_id: str = "exec-sndk-snxx-0001",
    connector_name: str = "binance_perpetual",
    trading_pair: str = "SNXX-USDT",
    leg: str = "ETF",
    quantity: str = "40",
    leverage: int = 20,
    notional_cap: str = "1000000",
    created_at_utc: str = CREATED_AT,
    updated_at_utc: str = CREATED_AT,
    released_at_utc: str | None = None,
    payload_overrides: dict | None = None,
) -> StrategyReservationV1:
    reservation_key = reservation_key or f"{executor_id}:{leg}:{quantity}"
    payload_value = {
        "schema_version": 1,
        "kind": "LEVERAGE_RESERVATION",
        "reservation_key": reservation_key,
        "executor_id": executor_id,
        "connector_name": connector_name,
        "trading_pair": trading_pair,
        "leg": leg,
        "logical_quantity": quantity,
        **(payload_overrides or {}),
    }
    return StrategyReservationV1(
        reservation_id=reservation_id,
        executor_id=executor_id,
        reservation_key=reservation_key,
        connector_name=connector_name,
        trading_pair=trading_pair,
        leg=leg,
        quantity=quantity,
        leverage=leverage,
        notional_cap=notional_cap,
        payload=CanonicalOpaquePayload.from_value(
            schema_version=1,
            kind="LEVERAGE_RESERVATION",
            value=payload_value,
        ),
        created_at_utc=created_at_utc,
        updated_at_utc=updated_at_utc,
        released_at_utc=released_at_utc,
    )


def test_journal_append_replay_and_stable_idempotency(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository.create_executor(initial)

    event = _prepared_event(initial, prepared)
    committed = repository.append_and_reduce(initial.executor_id, event, prepared)
    duplicate = repository.append_and_reduce(initial.executor_id, event, prepared)

    assert committed == duplicate
    assert committed.sequence == 1
    assert repository.load_snapshot(initial.executor_id) == prepared
    assert repository.replay(initial.executor_id) == prepared
    assert len(repository.events(initial.executor_id)) == 1

    with pytest.raises(JournalIntegrityError, match="event|payload|idempotency"):
        repository.append_and_reduce(
            initial.executor_id,
            _prepared_event(initial, prepared, payload_value="41"),
            prepared,
        )


def test_journal_commit_failure_rolls_back_event_and_snapshot(manager: SQLConnectionManager, vectors: dict):
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository = LeveragedEtfJournalRepository(manager)
    repository.create_executor(initial)

    def fail_commit(_connection) -> None:
        raise RuntimeError("forced commit failure")

    failing_repository = LeveragedEtfJournalRepository(manager, before_commit=fail_commit)
    with pytest.raises(RuntimeError, match="forced commit failure"):
        failing_repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)

    assert repository.events(initial.executor_id) == ()
    assert repository.load_snapshot(initial.executor_id) == initial


def test_concurrent_duplicate_event_has_one_committed_sequence(manager: SQLConnectionManager, vectors: dict):
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    LeveragedEtfJournalRepository(manager).create_executor(initial)

    def append_duplicate(_index: int):
        return LeveragedEtfJournalRepository(manager).append_and_reduce(
            initial.executor_id,
            _prepared_event(initial, prepared),
            prepared,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(append_duplicate, range(2)))

    repository = LeveragedEtfJournalRepository(manager)
    assert results[0] == results[1]
    assert tuple(event.sequence for event in repository.events(initial.executor_id)) == (1,)


def test_intent_transition_incomplete_query_and_logical_quantity_collision(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository.create_executor(initial)
    repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)
    conflicting_snapshot = _snapshot_at(prepared, 2)
    conflicting = _prepared_event(
        initial,
        conflicting_snapshot,
        attempt=2,
        event_id="event-maker-prepared-2",
        intent_id="intent-maker-2",
        client_order_id="exec-sndk-snxx-0001-maker-2",
    )
    with pytest.raises(JournalConflictError, match="logical|non-terminal|intent"):
        repository.append_and_reduce(
            initial.executor_id,
            conflicting,
            conflicting_snapshot,
        )

    acknowledged = _snapshot_at(prepared, 2, "MAKER_WORKING")
    repository.append_and_reduce(
        initial.executor_id,
        _followup_event(initial, acknowledged, JournalEventType.ACKNOWLEDGED, "event-maker-ack-1"),
        acknowledged,
    )
    incomplete = repository.incomplete_intents(initial.executor_id)
    assert tuple((intent.intent_id, intent.status) for intent in incomplete) == (
        ("intent-maker-1", JournalEventType.ACKNOWLEDGED),
    )

    reconciled = _snapshot_at(acknowledged, 3)
    repository.append_and_reduce(
        initial.executor_id,
        _followup_event(
            initial,
            reconciled,
            JournalEventType.RECONCILIATION,
            "event-maker-reconciled-1",
            terminal=True,
        ),
        reconciled,
    )
    assert repository.incomplete_intents(initial.executor_id) == ()


def test_ack_without_prepared_is_rejected_without_snapshot_mutation(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)

    with pytest.raises(JournalConflictError, match="PREPARED|intent"):
        repository.append_and_reduce(
            initial.executor_id,
            _followup_event(
                initial,
                _snapshot_at(initial, 1, "MAKER_WORKING"),
                JournalEventType.ACKNOWLEDGED,
                "event-maker-ack-orphan",
            ),
            _snapshot_at(initial, 1, "MAKER_WORKING"),
        )
    assert repository.events(initial.executor_id) == ()
    assert repository.load_snapshot(initial.executor_id) == initial


def test_side_effect_status_without_intent_is_rejected_without_snapshot_mutation(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    next_snapshot = _snapshot_at(initial, 1, "MAKER_WORKING")
    complete_ack = _followup_event(
        initial,
        next_snapshot,
        JournalEventType.ACKNOWLEDGED,
        "event-maker-ack-without-intent",
    ).model_dump(mode="json")
    complete_ack["intent_id"] = None

    with pytest.raises((ValidationError, JournalConflictError), match="intent|PREPARED|identity"):
        orphan_ack = JournalEventV1.model_validate(complete_ack)
        repository.append_and_reduce(initial.executor_id, orphan_ack, next_snapshot)
    assert repository.events(initial.executor_id) == ()
    assert repository.load_snapshot(initial.executor_id) == initial


def test_fill_identity_is_deduplicated_and_out_of_order_snapshot_is_rejected(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository.create_executor(initial)
    repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)
    filled_snapshot = _snapshot_with_leg_fill(
        prepared,
        sequence=2,
        state="STOCK_HEDGE_PENDING",
        leg="ETF",
        submitted="1",
        filled="1",
        client_order_id="exec-sndk-snxx-0001-maker-1",
        exchange_order_id="exchange-order-1",
    )
    fill = _followup_event(
        initial,
        filled_snapshot,
        JournalEventType.FILL,
        "event-maker-fill-1",
        exchange_order_id="exchange-order-1",
        exchange_trade_id="exchange-trade-1",
    )
    committed = repository.append_and_reduce(initial.executor_id, fill, filled_snapshot)
    assert fill.payload.__class__.__name__ == "FillJournalPayloadV1"
    duplicate_fill = JournalEventV1.model_validate(
        {
            **fill.model_dump(mode="json"),
            "event_id": "event-maker-fill-duplicate",
        }
    )

    with pytest.raises((JournalConflictError, JournalIntegrityError), match="trade|event|identity|payload"):
        repository.append_and_reduce(
            initial.executor_id,
            duplicate_fill,
            _snapshot_at(filled_snapshot, 3),
        )
    assert repository.load_snapshot(initial.executor_id) == filled_snapshot
    assert tuple(event.sequence for event in repository.events(initial.executor_id)) == (1, 2)

    distinct_fill = JournalEventV1.model_validate(
        {
            **fill.model_dump(mode="json"),
            "event_id": "event-maker-fill-conflict",
            "payload": _journal_payload(
                "FILL",
                {
                    **_payload_value(fill.payload),
                    "fill_quantity": "2",
                    "cumulative_filled_quantity": "2",
                },
            ).model_dump(mode="json"),
        }
    )
    with pytest.raises(JournalIntegrityError, match="trade|payload"):
        repository.append_and_reduce(
            initial.executor_id,
            distinct_fill,
            _snapshot_at(filled_snapshot, 3),
        )

    with pytest.raises(JournalConflictError, match="sequence"):
        repository.append_and_reduce(
            initial.executor_id,
            _followup_event(
                initial,
                _snapshot_at(filled_snapshot, 4),
                JournalEventType.ACKNOWLEDGED,
                "event-maker-ack-out-of-order",
            ),
            _snapshot_at(filled_snapshot, 4),
        )
    assert repository.load_snapshot(initial.executor_id) == filled_snapshot


def test_reservation_uniqueness_release_and_incomplete_executor_query(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)

    assert repository.reserve(_reservation()) == _reservation()
    assert repository.reserve(_reservation()) == _reservation()
    with pytest.raises(JournalConflictError, match="reservation|active|leg"):
        repository.reserve(_reservation("reservation-2", "exec-sndk-snxx-0001:ETF:40:2"))

    assert repository.active_reservations(initial.executor_id) == (_reservation(),)
    assert repository.incomplete_executors() == (initial,)
    released = repository.release_reservation("reservation-1", UPDATED_AT)
    assert released.released_at_utc is not None
    assert repository.release_reservation("reservation-1", UPDATED_AT) == released
    assert repository.active_reservations(initial.executor_id) == ()


def test_journal_replay_detects_hash_corruption_after_reopen(tmp_path: Path, vectors: dict):
    db_path = tmp_path / "replay.sqlite"
    manager = _open_manager(db_path)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository = LeveragedEtfJournalRepository(manager)
    repository.create_executor(initial)
    repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        assert LeveragedEtfJournalRepository(reopened).replay(initial.executor_id) == prepared
    finally:
        reopened.engine.dispose()

    corrupted = _open_manager(db_path)
    try:
        with sqlite3.connect(db_path) as connection:
            connection.execute("DROP TRIGGER lepf_journal_no_update")
            connection.execute(
                "UPDATE LeveragedEtfJournalEvent SET payload_hash = ? WHERE event_id = ?",
                ("f" * 64, "event-maker-prepared-1"),
            )
        with pytest.raises(JournalIntegrityError, match="hash"):
            LeveragedEtfJournalRepository(corrupted).replay(initial.executor_id)
    finally:
        corrupted.engine.dispose()


def test_anchor_checkpoint_cas_and_hash_integrity(manager: SQLConnectionManager, vectors: dict):
    repository = _V1WireAnchorStorageAdapter(manager)
    empty = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_empty"])
    confirmed = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_confirmed"])

    assert repository.load(empty.cycle_id) is None
    assert repository.compare_and_set_checkpoint(empty, expected_revision=0) == empty
    with pytest.raises(AnchorRevisionConflict):
        repository.compare_and_set_checkpoint(confirmed, expected_revision=0)
    assert repository.load(empty.cycle_id) == empty
    assert repository.compare_and_set_checkpoint(confirmed, expected_revision=1) == confirmed

    with manager.engine.begin() as connection:
        connection.execute(
            text("UPDATE LeveragedEtfAnchorState SET payload_hash = :payload_hash WHERE cycle_id = :cycle_id"),
            {"payload_hash": "f" * 64, "cycle_id": empty.cycle_id},
        )
    with pytest.raises(AnchorIntegrityError, match="hash"):
        repository.load(empty.cycle_id)


def test_anchor_opaque_versioned_checkpoint_round_trip_and_cas(manager: SQLConnectionManager):
    repository = _V1WireAnchorStorageAdapter(manager)
    payload = CanonicalOpaquePayload.from_value(
        schema_version=2,
        kind="ANCHOR_CHECKPOINT",
        value={
            "schema_version": 2,
            "cycle_id": "xnys-2026-07-18",
            "target_session_date": "2026-07-18",
            "official_close_utc": "2026-07-18T20:00:00.000000Z",
            "deadline_utc": "2026-07-18T20:10:00.000000Z",
            "next_poll_utc": "2026-07-18T20:00:00.000000Z",
            "revision": 1,
            "provider_state": {"cursor": "opaque", "samples": ["250", "30"]},
        },
    )
    checkpoint = OpaqueAnchorCheckpointV1(
        cycle_id="xnys-2026-07-18",
        target_session_date="2026-07-18",
        official_close_utc="2026-07-18T20:00:00.000000Z",
        deadline_utc="2026-07-18T20:10:00.000000Z",
        revision=1,
        payload=payload,
    )

    assert repository.compare_and_set_opaque_checkpoint(checkpoint, expected_revision=0) == checkpoint
    assert repository.load_opaque(checkpoint.cycle_id) == checkpoint
    assert repository.load_opaque(checkpoint.cycle_id).payload.value() == payload.value()

    updated_payload = CanonicalOpaquePayload.from_value(
        schema_version=2,
        kind="ANCHOR_CHECKPOINT",
        value={
            **payload.value(),
            "revision": 2,
            "provider_state": {"cursor": "next", "samples": ["250", "30"]},
        },
    )
    updated = OpaqueAnchorCheckpointV1.model_validate(
        {**checkpoint.model_dump(mode="json"), "revision": 2, "payload": updated_payload.model_dump(mode="json")}
    )
    with pytest.raises(AnchorRevisionConflict):
        repository.compare_and_set_opaque_checkpoint(updated, expected_revision=0)
    assert repository.compare_and_set_opaque_checkpoint(updated, expected_revision=1) == updated
    assert repository.load_opaque(checkpoint.cycle_id) == updated


def test_anchor_finalize_is_idempotent_and_conflict_is_immutable_after_reopen(tmp_path: Path, vectors: dict):
    db_path = tmp_path / "anchor.sqlite"
    manager = _open_manager(db_path)
    repository = _V1WireAnchorStorageAdapter(manager)
    confirmed = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_confirmed"])
    record = AnchorRecordV1.model_validate(vectors["fixtures"]["anchor_record"])
    conflict = AnchorRecordV1.model_validate(vectors["fixtures"]["anchor_record_conflict"])
    repository.compare_and_set_checkpoint(
        AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_empty"]),
        expected_revision=0,
    )
    repository.compare_and_set_checkpoint(confirmed, expected_revision=1)

    assert repository.finalize_if_absent(record, expected_revision=2) == record
    assert repository.finalize_if_absent(record, expected_revision=2) == record
    with pytest.raises(AnchorIntegrityError, match="evidence|final"):
        repository.finalize_if_absent(conflict, expected_revision=2)
    assert repository.load(record.cycle_id) == record
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        assert _V1WireAnchorStorageAdapter(reopened).load(record.cycle_id) == record
    finally:
        reopened.engine.dispose()


def test_anchor_revision_observations_append_without_mutating_final_record(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = _V1WireAnchorStorageAdapter(manager)
    empty = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_empty"])
    confirmed = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_confirmed"])
    record = AnchorRecordV1.model_validate(vectors["fixtures"]["anchor_record"])
    repository.compare_and_set_checkpoint(empty, expected_revision=0)
    repository.compare_and_set_checkpoint(confirmed, expected_revision=1)
    repository.finalize_if_absent(record, expected_revision=2)

    repository.append_revision_observation(record.cycle_id, "5" * 64, "2026-07-17T20:05:00.000000Z")
    repository.append_revision_observation(record.cycle_id, "6" * 64, "2026-07-17T20:06:00.000000Z")

    assert repository.load(record.cycle_id) == record
    assert tuple(
        (value.evidence_hash, value.observed_at_utc) for value in repository.revision_observations(record.cycle_id)
    ) == (
        ("5" * 64, "2026-07-17T20:05:00.000000Z"),
        ("6" * 64, "2026-07-17T20:06:00.000000Z"),
    )


def _pair_scoped_checkpoint(pair_id: str, revision: int) -> tuple[AnchorStorageKeyV2, OpaqueAnchorCheckpointV2]:
    cycle_id = "xnys-2026-07-18"
    key = AnchorStorageKeyV2(pair_id=pair_id, cycle_id=cycle_id)
    fields = {
        "integrity_version": 3,
        "pair_id": pair_id,
        "cycle_id": cycle_id,
        "target_session_date": "2026-07-18",
        "official_close_utc": "2026-07-18T20:00:00.000000Z",
        "deadline_utc": "2026-07-18T20:10:00.000000Z",
        "next_poll_utc": f"2026-07-18T20:00:0{revision - 1}.000000Z",
        "revision": revision,
    }
    return key, OpaqueAnchorCheckpointV2(
        key=key,
        target_session_date=fields["target_session_date"],
        official_close_utc=fields["official_close_utc"],
        deadline_utc=fields["deadline_utc"],
        revision=revision,
        payload=CanonicalOpaqueAnchorPayloadV2.from_value(
            kind="ANCHOR_CHECKPOINT",
            contract_version_field="integrity_version",
            contract_version=3,
            value=fields,
        ),
    )


def _pair_scoped_final(
    key: AnchorStorageKeyV2,
    revision: int,
    evidence_hash: str,
) -> OpaqueAnchorFinalizedV2:
    fields = {
        "evidence_version": 3,
        "pair_id": key.pair_id,
        "cycle_id": key.cycle_id,
        "target_session_date": "2026-07-18",
        "official_close_utc": "2026-07-18T20:00:00.000000Z",
        "deadline_utc": "2026-07-18T20:10:00.000000Z",
        "finalized_at_utc": "2026-07-18T20:02:00.000000Z",
        "evidence_hash": evidence_hash,
    }
    return OpaqueAnchorFinalizedV2(
        key=key,
        target_session_date=fields["target_session_date"],
        official_close_utc=fields["official_close_utc"],
        deadline_utc=fields["deadline_utc"],
        revision=revision,
        evidence_hash=evidence_hash,
        payload=CanonicalOpaqueAnchorPayloadV2.from_value(
            kind="ANCHOR_RECORD",
            contract_version_field="evidence_version",
            contract_version=3,
            value=fields,
        ),
    )


def _pair_scoped_revision(
    key: AnchorStorageKeyV2,
    evidence_hash: str,
    observed_at_utc: str,
) -> OpaqueAnchorRevisionObservationV2:
    fields = {
        "schema_version": 2,
        "pair_id": key.pair_id,
        "cycle_id": key.cycle_id,
        "evidence_hash": evidence_hash,
        "observed_at_utc": observed_at_utc,
    }
    return OpaqueAnchorRevisionObservationV2(
        key=key,
        evidence_hash=evidence_hash,
        observed_at_utc=observed_at_utc,
        payload=CanonicalOpaqueAnchorPayloadV2.from_value(
            kind="ANCHOR_REVISION_OBSERVATION",
            contract_version_field="schema_version",
            contract_version=2,
            value=fields,
        ),
    )


def test_pair_scoped_concurrent_cas_finalization_and_revisions_are_isolated(
    manager: SQLConnectionManager,
):
    repository = PairScopedAnchorRepository(manager)
    initial = tuple(_pair_scoped_checkpoint(pair_id, revision=1) for pair_id in ("sndk_snxx", "intc_intw"))

    def create(item):
        key, checkpoint = item
        return PairScopedAnchorRepository(manager).compare_and_set_opaque_checkpoint(
            key,
            checkpoint,
            expected_revision=0,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert tuple(executor.map(create, initial)) == tuple(checkpoint for _, checkpoint in initial)

    updated = tuple(_pair_scoped_checkpoint(key.pair_id, revision=2) for key, _ in initial)
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert tuple(
            executor.map(
                lambda item: PairScopedAnchorRepository(manager).compare_and_set_opaque_checkpoint(
                    item[0], item[1], expected_revision=1
                ),
                updated,
            )
        ) == tuple(checkpoint for _, checkpoint in updated)

    finalized = tuple(
        (key, _pair_scoped_final(key, revision=2, evidence_hash=str(index) * 64))
        for index, (key, _) in enumerate(updated, start=1)
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        assert tuple(
            executor.map(
                lambda item: PairScopedAnchorRepository(manager).finalize_opaque_if_absent(
                    item[0], item[1], expected_revision=2
                ),
                finalized,
            )
        ) == tuple(record for _, record in finalized)

    revisions = tuple(
        (
            key,
            _pair_scoped_revision(
                key,
                evidence_hash=character * 64,
                observed_at_utc=f"2026-07-18T20:0{index + 2}:00.000000Z",
            ),
        )
        for index, ((key, _), character) in enumerate(zip(finalized, ("a", "b")))
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        tuple(
            executor.map(
                lambda item: PairScopedAnchorRepository(manager).append_opaque_revision_observation(item[0], item[1]),
                revisions,
            )
        )

    for (key, record), (_, revision) in zip(finalized, revisions):
        assert repository.load_opaque(key) == record
        assert repository.opaque_revision_observations(key) == (revision,)


@pytest.mark.parametrize("invalid_expected_revision", [False, 0.0, "0", None])
def test_pair_scoped_cas_rejects_non_integer_expected_revision(
    manager: SQLConnectionManager,
    invalid_expected_revision,
):
    repository = PairScopedAnchorRepository(manager)
    key, checkpoint = _pair_scoped_checkpoint("sndk_snxx", revision=1)

    with pytest.raises(AnchorRevisionConflict, match="revision|integer"):
        repository.compare_and_set_opaque_checkpoint(
            key,
            checkpoint,
            expected_revision=invalid_expected_revision,
        )
    assert repository.load_opaque(key) is None


def _terminal_snapshot(
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
    sequence: int,
    state: str = "COMPLETED",
) -> LeveragedEtfPairExecutorSnapshotV1:
    payload = snapshot.model_dump(mode="json")
    if state == "COMPLETED":
        payload.update(
            {
                "etf_submitted_quantity": payload["etf_target_quantity"],
                "etf_filled_quantity": payload["etf_target_quantity"],
                "etf_remaining_quantity": "0",
                "stock_submitted_quantity": payload["stock_target_quantity"],
                "stock_filled_quantity": payload["stock_target_quantity"],
                "stock_remaining_quantity": "0",
            }
        )
    payload.update(
        {
            "state": state,
            "close_reason": "round-2 terminal probe",
            "updated_at_utc": LATER_AT,
            "last_journal_sequence": sequence,
        }
    )
    return LeveragedEtfPairExecutorSnapshotV1.model_validate(payload)


def _snapshot_for_executor(
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
    executor_id: str,
) -> LeveragedEtfPairExecutorSnapshotV1:
    payload = snapshot.model_dump(mode="json")
    payload["executor_id"] = executor_id
    return LeveragedEtfPairExecutorSnapshotV1.model_validate(payload)


def _canonical_json_hash(value: dict) -> tuple[str, str]:
    payload_json = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
    return payload_json, hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _overwrite_snapshot(
    db_path: Path,
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
) -> None:
    serialized = snapshot.model_dump(mode="json")
    payload_json, payload_hash = _canonical_json_hash(serialized)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE LeveragedEtfExecutorSnapshot
            SET state = ?, snapshot_json = ?, snapshot_hash = ?,
                last_journal_sequence = ?, updated_at_utc = ?
            WHERE executor_id = ?
            """,
            (
                serialized["state"],
                payload_json,
                payload_hash,
                serialized["last_journal_sequence"],
                serialized["updated_at_utc"],
                serialized["executor_id"],
            ),
        )


def test_journal_payloads_are_typed_and_discriminated(vectors: dict):
    initial = _initial_snapshot(vectors)
    payload_cases = []
    payload_cases.append((_prepared_event(initial, _prepared_snapshot(initial)), "PreparedJournalPayloadV1"))
    payload_cases.extend(
        (
            _followup_event(
                initial,
                _snapshot_at(initial, 1, next_state),
                event_type,
                f"event-{event_type.value.lower()}",
                action=action,
                leg=leg,
                logical_quantity=logical_quantity,
                exchange_order_id="exchange-order-typed" if event_type == JournalEventType.ORDER_CREATED else None,
                exchange_trade_id="exchange-trade-typed" if event_type == JournalEventType.FILL else None,
            ),
            expected_type,
        )
        for event_type, expected_type, action, leg, logical_quantity, next_state in (
            (JournalEventType.ACKNOWLEDGED, "AcknowledgedJournalPayloadV1", "ETF_MAKER", "ETF", "40", "MAKER_WORKING"),
            (JournalEventType.SUBMIT_UNKNOWN, "SubmitUnknownJournalPayloadV1", "ETF_MAKER", "ETF", "40", "RECONCILING"),
            (JournalEventType.FILL, "FillJournalPayloadV1", "ETF_MAKER", "ETF", "40", "STOCK_HEDGE_PENDING"),
            (
                JournalEventType.CANCEL_REQUESTED,
                "CancelJournalPayloadV1",
                "CANCEL",
                "ETF",
                "40",
                "MAKER_CANCEL_PENDING",
            ),
            (
                JournalEventType.HEDGE_REQUESTED,
                "HedgeJournalPayloadV1",
                "STOCK_HEDGE",
                "STOCK",
                "49.796",
                "STOCK_HEDGE_PENDING",
            ),
            (
                JournalEventType.ROLLBACK_REQUESTED,
                "RollbackJournalPayloadV1",
                "ETF_ROLLBACK",
                "ETF",
                "40",
                "ETF_ROLLBACK_PENDING",
            ),
            (
                JournalEventType.RECONCILIATION,
                "ReconciliationJournalPayloadV1",
                "ETF_MAKER",
                "ETF",
                "40",
                "RECONCILING",
            ),
        )
    )

    assert tuple(payload.__class__.__name__ for event, _ in payload_cases for payload in (event.payload,)) == tuple(
        expected_type for _, expected_type in payload_cases
    )
    payload_schema = JournalEventV1.model_json_schema()["properties"]["payload"]
    assert "discriminator" in payload_schema


def test_prepared_cannot_jump_created_executor_to_completed(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    illegal = _terminal_snapshot(initial, 1)
    repository.create_executor(initial)

    with pytest.raises((JournalConflictError, JournalIntegrityError), match="state|transition|reducer"):
        repository.append_and_reduce(initial.executor_id, _prepared_event(initial, illegal), illegal)
    assert repository.events(initial.executor_id) == ()
    assert repository.load_snapshot(initial.executor_id) == initial


def test_maker_prepared_cannot_be_completed_by_hedge_event(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    illegal = _snapshot_at(prepared, 2, "STOCK_HEDGE_PENDING")
    repository.create_executor(initial)
    repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)
    hedge = _followup_event(
        initial,
        illegal,
        JournalEventType.HEDGE_CONFIRMED,
        "event-foreign-hedge-confirmed",
        action="STOCK_HEDGE",
        leg="STOCK",
        logical_quantity="49.796",
        intent_id="intent-maker-1",
    )

    with pytest.raises((JournalConflictError, JournalIntegrityError), match="action|identity|intent|operation"):
        repository.append_and_reduce(initial.executor_id, hedge, illegal)
    assert tuple(event.sequence for event in repository.events(initial.executor_id)) == (1,)
    assert repository.load_snapshot(initial.executor_id) == prepared


@pytest.mark.parametrize(
    "mutation",
    (
        "missing_logical_quantity",
        "false_exposure_flag",
        "missing_deadline",
        "operation_mismatch",
        "idempotency_mismatch",
        "connector_mismatch",
    ),
)
def test_malformed_prepared_identity_is_rejected_atomically(
    manager: SQLConnectionManager,
    vectors: dict,
    mutation: str,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository.create_executor(initial)
    identity = _side_effect_identity(initial)
    if mutation == "missing_logical_quantity":
        identity.pop("logical_quantity")
    elif mutation == "false_exposure_flag":
        identity["exposure_increasing"] = False
    elif mutation == "missing_deadline":
        identity.pop("deadline_utc")
    elif mutation == "operation_mismatch":
        identity["operation"] = "CLOSE"
    elif mutation == "idempotency_mismatch":
        identity["idempotency_key"] = "arbitrary-idempotency-key"
    elif mutation == "connector_mismatch":
        identity["connector_name"] = "other_connector"
    event_payload = _journal_payload(
        "PREPARED",
        {
            "identity": identity,
            "next_state": _state_payload(prepared),
        },
    )

    with pytest.raises(
        (ValidationError, JournalConflictError, JournalIntegrityError),
        match="identity|exposure|logical|deadline|operation|idempotency|connector|payload",
    ):
        event = JournalEventV1(
            event_id=f"event-malformed-{mutation}",
            event_type=JournalEventType.PREPARED,
            intent_id="intent-maker-1",
            idempotency_key="exec-sndk-snxx-0001:OPEN:ETF:40:1",
            connector_name="binance_perpetual",
            trading_pair="SNXX-USDT",
            client_order_id="exec-sndk-snxx-0001-maker-1",
            payload=event_payload,
            created_at_utc=UPDATED_AT,
        )
        repository.append_and_reduce(initial.executor_id, event, prepared)
    assert repository.events(initial.executor_id) == ()
    assert repository.load_snapshot(initial.executor_id) == initial


@pytest.mark.parametrize(
    ("event_type", "action", "leg", "logical_quantity", "illegal_state"),
    (
        (JournalEventType.ACKNOWLEDGED, "ETF_MAKER", "ETF", "40", "COMPLETED"),
        (JournalEventType.SUBMIT_UNKNOWN, "ETF_MAKER", "ETF", "40", "COMPLETED"),
        (JournalEventType.FILL, "ETF_MAKER", "ETF", "40", "COMPLETED"),
        (JournalEventType.CANCEL_CONFIRMED, "CANCEL", "ETF", "40", "ABORTED_NO_FILL"),
        (JournalEventType.HEDGE_CONFIRMED, "STOCK_HEDGE", "STOCK", "49.796", "STOCK_HEDGE_PENDING"),
        (JournalEventType.ROLLBACK_CONFIRMED, "ETF_ROLLBACK", "ETF", "40", "ETF_ROLLBACK_PENDING"),
        (JournalEventType.RECONCILIATION, "ETF_MAKER", "ETF", "40", "CREATED"),
    ),
)
def test_illegal_cross_operation_and_state_matrix_has_no_mutation(
    manager: SQLConnectionManager,
    vectors: dict,
    event_type: JournalEventType,
    action: str,
    leg: str,
    logical_quantity: str,
    illegal_state: str,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository.create_executor(initial)
    repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)
    illegal = _snapshot_at(prepared, 2, illegal_state)
    event = _followup_event(
        initial,
        illegal,
        event_type,
        f"event-illegal-{event_type.value.lower()}",
        action=action,
        leg=leg,
        logical_quantity=logical_quantity,
        exchange_order_id="exchange-order-illegal" if event_type == JournalEventType.FILL else None,
        exchange_trade_id="exchange-trade-illegal" if event_type == JournalEventType.FILL else None,
    )

    with pytest.raises(
        (JournalConflictError, JournalIntegrityError), match="state|transition|action|identity|quantity|reducer"
    ):
        repository.append_and_reduce(initial.executor_id, event, illegal)
    assert tuple(item.sequence for item in repository.events(initial.executor_id)) == (1,)
    assert repository.load_snapshot(initial.executor_id) == prepared


def test_exchange_trade_duplicate_requires_complete_same_event_identity(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    filled = _snapshot_with_leg_fill(
        prepared,
        sequence=2,
        state="STOCK_HEDGE_PENDING",
        leg="ETF",
        submitted="1",
        filled="1",
        client_order_id="exec-sndk-snxx-0001-maker-1",
        exchange_order_id="exchange-order-owned",
    )
    event = _followup_event(
        initial,
        filled,
        JournalEventType.FILL,
        "event-trade-owned",
        exchange_order_id="exchange-order-owned",
        exchange_trade_id="exchange-trade-owned",
    )
    repository.create_executor(initial)
    repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)
    committed = repository.append_and_reduce(initial.executor_id, event, filled)

    assert repository.append_and_reduce(initial.executor_id, event, filled) == committed
    conflicting = JournalEventV1.model_validate({**event.model_dump(mode="json"), "event_id": "event-trade-renamed"})
    with pytest.raises((JournalConflictError, JournalIntegrityError), match="trade|event|identity"):
        repository.append_and_reduce(initial.executor_id, conflicting, _snapshot_at(filled, 3))
    assert repository.load_snapshot(initial.executor_id) == filled


def test_exchange_trade_duplicate_cannot_return_foreign_executor_event(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial_a = _initial_snapshot(vectors)
    initial_b = _snapshot_for_executor(initial_a, "exec-sndk-snxx-0002")
    prepared_a = _prepared_snapshot(initial_a)
    filled_a = _snapshot_with_leg_fill(
        prepared_a,
        sequence=2,
        state="STOCK_HEDGE_PENDING",
        leg="ETF",
        submitted="1",
        filled="1",
        client_order_id="exec-sndk-snxx-0001-maker-1",
        exchange_order_id="exchange-order-shared",
    )
    fill_a = _followup_event(
        initial_a,
        filled_a,
        JournalEventType.FILL,
        "event-trade-owner-a",
        exchange_order_id="exchange-order-shared",
        exchange_trade_id="exchange-trade-shared",
    )
    repository.create_executor(initial_a)
    repository.create_executor(initial_b)
    repository.append_and_reduce(initial_a.executor_id, _prepared_event(initial_a, prepared_a), prepared_a)
    repository.append_and_reduce(initial_a.executor_id, fill_a, filled_a)
    foreign_event_data = fill_a.model_dump(mode="json")
    foreign_event_data.update(
        {
            "event_id": "event-trade-foreign-b",
            "intent_id": "intent-foreign-b",
            "idempotency_key": "exec-sndk-snxx-0002:OPEN:ETF:40:1",
            "client_order_id": "exec-sndk-snxx-0002-maker-1",
        }
    )
    proposed_b = _snapshot_with_leg_fill(
        initial_b,
        sequence=1,
        state="STOCK_HEDGE_PENDING",
        leg="ETF",
        submitted="1",
        filled="1",
    )

    with pytest.raises(
        (ValidationError, JournalConflictError, JournalIntegrityError), match="executor|owner|identity|trade|PREPARED"
    ):
        foreign_event = JournalEventV1.model_validate(foreign_event_data)
        repository.append_and_reduce(initial_b.executor_id, foreign_event, proposed_b)
    assert repository.events(initial_b.executor_id) == ()
    assert repository.load_snapshot(initial_b.executor_id) == initial_b


def test_exchange_trade_id_is_rejected_for_non_fill_event(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository.create_executor(initial)
    repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)
    with pytest.raises(
        (ValidationError, JournalConflictError, JournalIntegrityError), match="trade|FILL|identity|action"
    ):
        illegal = _followup_event(
            initial,
            _snapshot_at(prepared, 2, "STOCK_HEDGE_PENDING"),
            JournalEventType.HEDGE_CONFIRMED,
            "event-non-fill-trade-id",
            action="STOCK_HEDGE",
            leg="STOCK",
            logical_quantity="49.796",
            exchange_trade_id="exchange-trade-not-a-fill",
        )
        repository.append_and_reduce(initial.executor_id, illegal, _snapshot_at(prepared, 2, "STOCK_HEDGE_PENDING"))
    assert tuple(item.sequence for item in repository.events(initial.executor_id)) == (1,)


def test_cross_executor_exchange_trade_race_has_one_owner(manager: SQLConnectionManager, vectors: dict):
    initial_a = _initial_snapshot(vectors)
    initial_b = _snapshot_for_executor(initial_a, "exec-sndk-snxx-0002")
    repository = LeveragedEtfJournalRepository(manager)
    repository.create_executor(initial_a)
    repository.create_executor(initial_b)
    prepared_a = _prepared_snapshot(initial_a)
    prepared_b = _prepared_snapshot(initial_b)
    repository.append_and_reduce(initial_a.executor_id, _prepared_event(initial_a, prepared_a), prepared_a)
    repository.append_and_reduce(
        initial_b.executor_id,
        _prepared_event(
            initial_b,
            prepared_b,
            event_id="event-maker-prepared-b",
            intent_id="intent-maker-b",
            client_order_id="exec-sndk-snxx-0002-maker-1",
        ),
        prepared_b,
    )
    filled_a = _snapshot_with_leg_fill(
        prepared_a,
        sequence=2,
        state="STOCK_HEDGE_PENDING",
        leg="ETF",
        submitted="1",
        filled="1",
        client_order_id="exec-sndk-snxx-0001-maker-1",
        exchange_order_id="exchange-order-race-a",
    )
    filled_b = _snapshot_with_leg_fill(
        prepared_b,
        sequence=2,
        state="STOCK_HEDGE_PENDING",
        leg="ETF",
        submitted="1",
        filled="1",
        client_order_id="exec-sndk-snxx-0002-maker-1",
        exchange_order_id="exchange-order-race-b",
    )
    fill_a = _followup_event(
        initial_a,
        filled_a,
        JournalEventType.FILL,
        "event-race-fill-a",
        exchange_order_id="exchange-order-race-a",
        exchange_trade_id="exchange-trade-race",
    )
    fill_b = _followup_event(
        initial_b,
        filled_b,
        JournalEventType.FILL,
        "event-race-fill-b",
        intent_id="intent-maker-b",
        client_order_id="exec-sndk-snxx-0002-maker-1",
        exchange_order_id="exchange-order-race-b",
        exchange_trade_id="exchange-trade-race",
    )

    def append(executor_id: str, event: JournalEventV1, snapshot: LeveragedEtfPairExecutorSnapshotV1):
        try:
            return LeveragedEtfJournalRepository(manager).append_and_reduce(executor_id, event, snapshot)
        except (JournalConflictError, JournalIntegrityError) as exception:
            return exception

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(
            executor.map(
                lambda arguments: append(*arguments),
                ((initial_a.executor_id, fill_a, filled_a), (initial_b.executor_id, fill_b, filled_b)),
            )
        )
    committed = tuple(result for result in results if not isinstance(result, Exception))
    rejected = tuple(result for result in results if isinstance(result, Exception))

    assert len(committed) == 1
    assert len(rejected) == 1
    snapshots = (repository.load_snapshot(initial_a.executor_id), repository.load_snapshot(initial_b.executor_id))
    assert sorted(snapshot.last_journal_sequence for snapshot in snapshots) == [1, 2]


def _create_terminal_maker_intent(
    repository: LeveragedEtfJournalRepository,
    initial: LeveragedEtfPairExecutorSnapshotV1,
) -> LeveragedEtfPairExecutorSnapshotV1:
    prepared = _prepared_snapshot(initial)
    acknowledged = _snapshot_at(prepared, 2, "MAKER_WORKING")
    reconciled = _snapshot_at(acknowledged, 3, "MAKER_WORKING")
    repository.create_executor(initial)
    repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)
    repository.append_and_reduce(
        initial.executor_id,
        _followup_event(initial, acknowledged, JournalEventType.ACKNOWLEDGED, "event-recovery-ack"),
        acknowledged,
    )
    repository.append_and_reduce(
        initial.executor_id,
        _followup_event(
            initial,
            reconciled,
            JournalEventType.RECONCILIATION,
            "event-recovery-terminal",
            terminal=True,
        ),
        reconciled,
    )
    return reconciled


@pytest.mark.parametrize("discovery_method", ("incomplete_intents", "incomplete_executors"))
def test_recovery_discovery_replays_snapshot_before_filtering(
    tmp_path: Path,
    vectors: dict,
    discovery_method: str,
):
    db_path = tmp_path / f"recovery-divergence-{discovery_method}.sqlite"
    manager = _open_manager(db_path)
    initial = _initial_snapshot(vectors)
    repository = LeveragedEtfJournalRepository(manager)
    reconciled = _create_terminal_maker_intent(repository, initial)
    tampered = LeveragedEtfPairExecutorSnapshotV1.model_validate(
        {**reconciled.model_dump(mode="json"), "state": "COMPLETED", "close_reason": "tampered terminal"}
    )
    manager.engine.dispose()
    _overwrite_snapshot(db_path, tampered)
    reopened = _open_manager(db_path)
    try:
        with pytest.raises(JournalIntegrityError, match="replay|persisted|journal|chain"):
            method = getattr(LeveragedEtfJournalRepository(reopened), discovery_method)
            method(initial.executor_id) if discovery_method == "incomplete_intents" else method()
    finally:
        reopened.engine.dispose()


@pytest.mark.parametrize("corruption", ("missing_sequence", "extra_journal_sequence", "orphan_event"))
def test_recovery_discovery_rejects_gap_extra_and_orphan_rows(
    tmp_path: Path,
    vectors: dict,
    corruption: str,
):
    db_path = tmp_path / f"recovery-{corruption}.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    acknowledged = _snapshot_at(prepared, 2, "MAKER_WORKING")
    repository.create_executor(initial)
    repository.append_and_reduce(initial.executor_id, _prepared_event(initial, prepared), prepared)
    repository.append_and_reduce(
        initial.executor_id,
        _followup_event(initial, acknowledged, JournalEventType.ACKNOWLEDGED, "event-recovery-gap-ack"),
        acknowledged,
    )
    manager.engine.dispose()
    if corruption == "extra_journal_sequence":
        _overwrite_snapshot(db_path, prepared)
    else:
        with sqlite3.connect(db_path) as connection:
            if corruption == "missing_sequence":
                connection.execute("DROP TRIGGER lepf_journal_no_delete")
                connection.execute("DELETE FROM LeveragedEtfJournalEvent WHERE sequence = 1")
                connection.execute(SQLITE_GUARD_DDL["lepf_journal_no_delete"])
            else:
                connection.execute("DROP TRIGGER lepf_snapshot_referenced_no_delete")
                connection.execute(
                    "DELETE FROM LeveragedEtfExecutorSnapshot WHERE executor_id = ?",
                    (initial.executor_id,),
                )
                connection.execute(SQLITE_GUARD_DDL["lepf_snapshot_referenced_no_delete"])
    reopened = _open_manager(db_path)
    try:
        recovery_repository = LeveragedEtfJournalRepository(reopened)
        with pytest.raises(JournalIntegrityError, match="sequence|orphan|snapshot|journal|replay|intent"):
            recovery_repository.incomplete_executors()
        with pytest.raises(JournalIntegrityError, match="sequence|orphan|snapshot|journal|replay|intent"):
            recovery_repository.incomplete_intents(initial.executor_id)
    finally:
        reopened.engine.dispose()


def test_recovery_discovery_and_append_share_one_database_snapshot(manager: SQLConnectionManager, vectors: dict):
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    LeveragedEtfJournalRepository(manager).create_executor(initial)
    read_entered = threading.Event()
    release_read = threading.Event()
    writer_reached_commit = threading.Event()

    def recovery_hook(_connection) -> None:
        read_entered.set()
        assert release_read.wait(timeout=10)

    def before_writer_commit(_connection) -> None:
        writer_reached_commit.set()

    recovery_repository = LeveragedEtfJournalRepository(manager, recovery_read_hook=recovery_hook)
    writer_repository = LeveragedEtfJournalRepository(manager, before_commit=before_writer_commit)
    with ThreadPoolExecutor(max_workers=2) as executor:
        discovery = executor.submit(recovery_repository.incomplete_executors)
        assert read_entered.wait(timeout=10)
        append = executor.submit(
            writer_repository.append_and_reduce,
            initial.executor_id,
            _prepared_event(initial, prepared),
            prepared,
        )
        assert writer_reached_commit.wait(timeout=10)
        assert not append.done()
        release_read.set()
        assert discovery.result(timeout=10) == (initial,)
        assert append.result(timeout=10).sequence == 1


@pytest.mark.parametrize(
    "mutation",
    (
        "connector",
        "pair",
        "key",
        "quantity_payload",
        "leverage",
        "notional_cap",
        "pre_released",
        "pre_released_backdated",
        "non_creation_update",
        "updated_backdated",
    ),
)
def test_reservation_role_identity_and_creation_lifecycle_are_atomic(
    manager: SQLConnectionManager,
    vectors: dict,
    mutation: str,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    reservation_kwargs = {
        "connector": {"connector_name": "other_connector"},
        "pair": {"trading_pair": "BTC-USDT"},
        "key": {"reservation_key": "arbitrary-reservation-key"},
        "quantity_payload": {"quantity": "39", "payload_overrides": {"logical_quantity": "40"}},
        "leverage": {"leverage": 10},
        "notional_cap": {"notional_cap": "999999"},
        "pre_released": {"released_at_utc": UPDATED_AT},
        "pre_released_backdated": {"released_at_utc": "2026-07-17T14:01:01.000000Z"},
        "non_creation_update": {"updated_at_utc": LATER_AT},
        "updated_backdated": {"updated_at_utc": "2026-07-17T14:01:01.000000Z"},
    }[mutation]

    with pytest.raises(
        (ValidationError, JournalConflictError, JournalIntegrityError),
        match="reservation|role|connector|pair|key|quantity|release|created|updated|identity",
    ):
        reservation = _reservation(**reservation_kwargs)
        repository.reserve(reservation)
    assert repository.active_reservations() == ()


def test_reservation_reopen_and_concurrent_release_are_monotonic(tmp_path: Path, vectors: dict):
    db_path = tmp_path / "reservation-reopen.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    reservation = _reservation()
    repository.reserve(reservation)
    manager.engine.dispose()
    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        assert reopened_repository.active_reservations(initial.executor_id) == (reservation,)
        with pytest.raises(JournalConflictError, match="release|time|monotonic|created"):
            reopened_repository.release_reservation(
                reservation.reservation_id,
                "2026-07-17T14:01:01.000000Z",
            )
        assert reopened_repository.active_reservations(initial.executor_id) == (reservation,)

        def release(timestamp: str):
            try:
                return LeveragedEtfJournalRepository(reopened).release_reservation(
                    reservation.reservation_id,
                    timestamp,
                )
            except JournalConflictError as exception:
                return exception

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = tuple(executor.map(release, (UPDATED_AT, LATER_AT)))
        released = tuple(result for result in results if isinstance(result, StrategyReservationV1))
        rejected = tuple(result for result in results if isinstance(result, JournalConflictError))
        assert len(released) == 1
        assert len(rejected) == 1
        assert reopened_repository.active_reservations(initial.executor_id) == ()
        with pytest.raises(JournalConflictError, match="released|time|monotonic"):
            reopened_repository.release_reservation(reservation.reservation_id, CREATED_AT)
    finally:
        reopened.engine.dispose()


@pytest.mark.parametrize("non_finite", (math.nan, math.inf, -math.inf), ids=("nan", "infinity", "negative-infinity"))
def test_canonical_opaque_payload_rejects_non_finite_json(non_finite: float):
    with pytest.raises((ValueError, ValidationError), match="finite|constant|JSON|canonical"):
        CanonicalOpaquePayload.from_value(
            schema_version=1,
            kind="ANCHOR_CHECKPOINT",
            value={"schema_version": 1, "value": non_finite},
        )
    raw_json = (
        '{"schema_version":1,"value":'
        + ("NaN" if math.isnan(non_finite) else "Infinity" if non_finite > 0 else "-Infinity")
        + "}"
    )
    with pytest.raises((ValueError, ValidationError), match="finite|constant|JSON|canonical"):
        CanonicalOpaquePayload.from_canonical_json(
            schema_version=1,
            kind="ANCHOR_CHECKPOINT",
            payload_json=raw_json,
            payload_hash=hashlib.sha256(raw_json.encode("utf-8")).hexdigest(),
        )


@pytest.mark.parametrize("kind", ("ANCHOR_CHECKPOINT", "ANCHOR_RECORD", "LEVERAGE_RESERVATION"))
def test_opaque_wrapper_and_embedded_schema_versions_must_match(kind: str):
    with pytest.raises((ValueError, ValidationError), match="schema_version|version"):
        CanonicalOpaquePayload.from_value(
            schema_version=2,
            kind=kind,
            value={"schema_version": 1, "kind": kind},
        )


@pytest.mark.parametrize("schema_version", (1, 2))
@pytest.mark.parametrize("state_kind", ("CHECKPOINT", "FINALIZED"))
def test_matching_opaque_anchor_versions_round_trip_after_reopen(
    tmp_path: Path,
    schema_version: int,
    state_kind: str,
):
    db_path = tmp_path / f"opaque-{schema_version}-{state_kind.lower()}.sqlite"
    manager = _open_manager(db_path)
    repository = _V1WireAnchorStorageAdapter(manager)
    cycle_id = f"xnys-2026-07-{18 + schema_version}"
    target_session_date = f"2026-07-{18 + schema_version}"
    official_close_utc = f"2026-07-{18 + schema_version}T20:00:00.000000Z"
    deadline_utc = f"2026-07-{18 + schema_version}T20:10:00.000000Z"
    if state_kind == "CHECKPOINT":
        payload = CanonicalOpaquePayload.from_value(
            schema_version,
            "ANCHOR_CHECKPOINT",
            {
                "schema_version": schema_version,
                "cycle_id": cycle_id,
                "target_session_date": target_session_date,
                "official_close_utc": official_close_utc,
                "deadline_utc": deadline_utc,
                "next_poll_utc": official_close_utc,
                "revision": 1,
                "provider_state": {"cursor": f"v{schema_version}"},
            },
        )
        state = OpaqueAnchorCheckpointV1(
            cycle_id=cycle_id,
            target_session_date=target_session_date,
            official_close_utc=official_close_utc,
            deadline_utc=deadline_utc,
            revision=1,
            payload=payload,
        )
        repository.compare_and_set_opaque_checkpoint(state, expected_revision=0)
    else:
        evidence_hash = str(schema_version) * 64
        checkpoint_payload = CanonicalOpaquePayload.from_value(
            schema_version,
            "ANCHOR_CHECKPOINT",
            {
                "schema_version": schema_version,
                "cycle_id": cycle_id,
                "target_session_date": target_session_date,
                "official_close_utc": official_close_utc,
                "deadline_utc": deadline_utc,
                "next_poll_utc": official_close_utc,
                "revision": 1,
                "provider_state": {"cursor": f"v{schema_version}"},
            },
        )
        repository.compare_and_set_opaque_checkpoint(
            OpaqueAnchorCheckpointV1(
                cycle_id=cycle_id,
                target_session_date=target_session_date,
                official_close_utc=official_close_utc,
                deadline_utc=deadline_utc,
                revision=1,
                payload=checkpoint_payload,
            ),
            expected_revision=0,
        )
        payload = CanonicalOpaquePayload.from_value(
            schema_version,
            "ANCHOR_RECORD",
            {
                "schema_version": schema_version,
                "cycle_id": cycle_id,
                "target_session_date": target_session_date,
                "official_close_utc": official_close_utc,
                "deadline_utc": deadline_utc,
                "evidence_hash": evidence_hash,
                "finalized_at_utc": official_close_utc,
                "provider_state": {"cursor": f"v{schema_version}"},
            },
        )
        state = OpaqueAnchorFinalizedV1(
            cycle_id=cycle_id,
            target_session_date=target_session_date,
            official_close_utc=official_close_utc,
            deadline_utc=deadline_utc,
            revision=1,
            evidence_hash=evidence_hash,
            payload=payload,
        )
        repository.finalize_opaque_if_absent(state, expected_revision=1)
    manager.engine.dispose()
    reopened = _open_manager(db_path)
    try:
        assert _V1WireAnchorStorageAdapter(reopened).load_opaque(cycle_id) == state
    finally:
        reopened.engine.dispose()


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("target_session_date", "2026-07-18"),
        ("official_close_utc", "2026-07-17T20:01:00.000000Z"),
        ("deadline_utc", "2026-07-17T20:11:00.000000Z"),
    ),
)
def test_anchor_checkpoint_cycle_identity_is_frozen_without_mutation(
    manager: SQLConnectionManager,
    vectors: dict,
    field_name: str,
    replacement: str,
):
    repository = _V1WireAnchorStorageAdapter(manager)
    empty = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_empty"])
    confirmed_data = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_confirmed"]).model_dump(
        mode="json"
    )
    confirmed_data[field_name] = replacement
    changed = AnchorPollingCheckpointV1.model_validate(confirmed_data)
    repository.compare_and_set_checkpoint(empty, expected_revision=0)

    with pytest.raises(AnchorIntegrityError, match="cycle|session|official|deadline|identity"):
        repository.compare_and_set_checkpoint(changed, expected_revision=1)
    assert repository.load(empty.cycle_id) == empty


@pytest.mark.parametrize(
    ("field_name", "replacement"),
    (
        ("target_session_date", "2026-07-18"),
        ("deadline_utc", "2026-07-17T20:11:00.000000Z"),
    ),
)
def test_anchor_typed_finalization_preserves_checkpoint_identity_after_reopen(
    tmp_path: Path,
    vectors: dict,
    field_name: str,
    replacement: str,
):
    db_path = tmp_path / f"anchor-final-{field_name}.sqlite"
    manager = _open_manager(db_path)
    repository = _V1WireAnchorStorageAdapter(manager)
    empty = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_empty"])
    confirmed = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_confirmed"])
    repository.compare_and_set_checkpoint(empty, expected_revision=0)
    repository.compare_and_set_checkpoint(confirmed, expected_revision=1)
    manager.engine.dispose()
    reopened = _open_manager(db_path)
    try:
        reopened_repository = _V1WireAnchorStorageAdapter(reopened)
        record_data = AnchorRecordV1.model_validate(vectors["fixtures"]["anchor_record"]).model_dump(mode="json")
        record_data[field_name] = replacement
        changed_record = AnchorRecordV1.model_validate(record_data)
        with pytest.raises(AnchorIntegrityError, match="cycle|session|deadline|identity"):
            reopened_repository.finalize_if_absent(changed_record, expected_revision=2)
        assert reopened_repository.load(confirmed.cycle_id) == confirmed
    finally:
        reopened.engine.dispose()


def test_anchor_opaque_finalization_preserves_official_close(manager: SQLConnectionManager, vectors: dict):
    repository = _V1WireAnchorStorageAdapter(manager)
    empty = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_empty"])
    confirmed = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_confirmed"])
    record = AnchorRecordV1.model_validate(vectors["fixtures"]["anchor_record"])
    repository.compare_and_set_checkpoint(empty, expected_revision=0)
    repository.compare_and_set_checkpoint(confirmed, expected_revision=1)
    opaque = OpaqueAnchorFinalizedV1(
        cycle_id=record.cycle_id,
        target_session_date=record.target_session_date,
        official_close_utc="2026-07-17T20:01:00.000000Z",
        deadline_utc=record.deadline_utc,
        revision=2,
        evidence_hash=record.evidence_hash,
        payload=CanonicalOpaquePayload.from_value(1, "ANCHOR_RECORD", record.model_dump(mode="json")),
    )

    with pytest.raises(AnchorIntegrityError, match="official|close|cycle|identity"):
        repository.finalize_opaque_if_absent(opaque, expected_revision=2)
    assert repository.load(confirmed.cycle_id) == confirmed


def test_anchor_concurrent_valid_and_identity_conflicting_cas_keeps_valid_state(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = _V1WireAnchorStorageAdapter(manager)
    empty = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_empty"])
    confirmed = AnchorPollingCheckpointV1.model_validate(vectors["fixtures"]["checkpoint_confirmed"])
    invalid = AnchorPollingCheckpointV1.model_validate(
        {**confirmed.model_dump(mode="json"), "deadline_utc": "2026-07-17T20:11:00.000000Z"}
    )
    repository.compare_and_set_checkpoint(empty, expected_revision=0)

    def update(checkpoint: AnchorPollingCheckpointV1):
        try:
            return _V1WireAnchorStorageAdapter(manager).compare_and_set_checkpoint(
                checkpoint,
                expected_revision=1,
            )
        except (AnchorIntegrityError, AnchorRevisionConflict) as exception:
            return exception

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(update, (confirmed, invalid)))
    assert sum(isinstance(result, AnchorPollingCheckpointV1) for result in results) == 1
    assert sum(isinstance(result, (AnchorIntegrityError, AnchorRevisionConflict)) for result in results) == 1
    assert repository.load(empty.cycle_id) == confirmed


def _append_fact(
    repository: LeveragedEtfJournalRepository,
    snapshot: LeveragedEtfPairExecutorSnapshotV1,
    event: JournalEventV1,
) -> LeveragedEtfPairExecutorSnapshotV1:
    repository.append_and_reduce(snapshot.executor_id, event)
    reduced = repository.load_snapshot(snapshot.executor_id)
    assert reduced is not None
    return reduced


def _assert_fact_rejected_atomically(
    repository: LeveragedEtfJournalRepository,
    executor_id: str,
    event: JournalEventV1,
) -> None:
    before_snapshot = repository.load_snapshot(executor_id)
    before_events = repository.events(executor_id)
    with pytest.raises((JournalConflictError, JournalIntegrityError)):
        repository.append_and_reduce(executor_id, event)
    assert repository.load_snapshot(executor_id) == before_snapshot
    assert repository.events(executor_id) == before_events


def test_journal_event_facts_authoritatively_derive_snapshot(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    prepared_event = _fact_event(initial, JournalEventType.PREPARED, "event-facts-prepared")

    assert "next_state" not in prepared_event.payload.model_dump(mode="json")
    reduced = _append_fact(repository, initial, prepared_event)
    assert reduced.state.value == "MAKER_SUBMITTING"
    assert reduced.last_journal_sequence == 1
    assert reduced.etf_filled_quantity == Decimal("0")

    acknowledged = _fact_event(initial, JournalEventType.ACKNOWLEDGED, "event-facts-ack")
    forged_terminal = _terminal_snapshot(reduced, 2)
    with pytest.raises(JournalIntegrityError, match="assert|derived|snapshot|reducer"):
        repository.append_and_reduce(initial.executor_id, acknowledged, forged_terminal)
    assert repository.load_snapshot(initial.executor_id) == reduced


def test_zero_fill_hedge_confirmation_cannot_terminalize_executor(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    current = _append_fact(
        repository,
        initial,
        _fact_event(initial, JournalEventType.PREPARED, "event-zero-hedge-maker-prepared"),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-zero-hedge-maker-created",
            exchange_order_id="exchange-zero-hedge-maker",
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-zero-hedge-maker-fill",
            exchange_order_id="exchange-zero-hedge-maker",
            exchange_trade_id="trade-zero-hedge-maker",
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.PREPARED,
            "event-zero-hedge-prepared",
            action="STOCK_HEDGE",
            leg="STOCK",
            logical_quantity="1.2449",
            intent_id="intent-zero-hedge",
            client_order_id="exec-sndk-snxx-0001-zero-hedge",
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.HEDGE_REQUESTED,
            "event-zero-hedge-requested",
            action="STOCK_HEDGE",
            leg="STOCK",
            logical_quantity="1.2449",
            intent_id="intent-zero-hedge",
            client_order_id="exec-sndk-snxx-0001-zero-hedge",
        ),
    )
    confirmation = _fact_event(
        initial,
        JournalEventType.HEDGE_CONFIRMED,
        "event-zero-hedge-confirmed",
        action="STOCK_HEDGE",
        leg="STOCK",
        logical_quantity="1.2449",
        intent_id="intent-zero-hedge",
        client_order_id="exec-sndk-snxx-0001-zero-hedge",
    )

    with pytest.raises((JournalConflictError, JournalIntegrityError), match="fill|hedge|confirm|exposure"):
        repository.append_and_reduce(initial.executor_id, confirmation)
    assert repository.load_snapshot(initial.executor_id) == current
    assert current.state.value == "STOCK_HEDGE_PENDING"
    assert current.stock_filled_quantity == Decimal("0")


def test_cancel_request_stays_pending_until_target_maker_is_terminal(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    current = _append_fact(
        repository,
        initial,
        _fact_event(initial, JournalEventType.PREPARED, "event-cancel-make-prepared"),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.ACKNOWLEDGED, "event-cancel-maker-ack"),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.PREPARED,
            "event-cancel-prepared",
            action="CANCEL",
            intent_id="intent-cancel-1",
            client_order_id="exec-sndk-snxx-0001-cancel-1",
        ),
    )
    requested = _fact_event(
        initial,
        JournalEventType.CANCEL_REQUESTED,
        "event-cancel-requested",
        action="CANCEL",
        intent_id="intent-cancel-1",
        client_order_id="exec-sndk-snxx-0001-cancel-1",
        order_cumulative_filled_quantity="0",
        target_intent_id="intent-maker-fact-1",
        target_client_order_id="exec-sndk-snxx-0001-maker-fact-1",
    )
    current = _append_fact(repository, current, requested)

    assert current.state.value == "MAKER_CANCEL_PENDING"
    assert current.close_reason is None
    assert tuple((intent.intent_id, intent.status.value) for intent in repository.incomplete_intents()) == (
        ("intent-maker-fact-1", "ACKNOWLEDGED"),
        ("intent-cancel-1", "CANCEL_REQUESTED"),
    )


def test_reconciliation_filled_requires_recorded_fill_facts(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    current = _append_fact(
        repository,
        initial,
        _fact_event(initial, JournalEventType.PREPARED, "event-reconcile-prepared"),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-reconcile-order",
            exchange_order_id="exchange-reconcile-order",
        ),
    )
    reconciliation = _fact_event(
        initial,
        JournalEventType.RECONCILIATION,
        "event-reconcile-impossible-filled",
        exchange_order_id="exchange-reconcile-order",
        order_cumulative_filled_quantity="0",
        outcome="FILLED",
    )

    with pytest.raises((JournalConflictError, JournalIntegrityError), match="fill|cumulative|reconcil|outcome"):
        repository.append_and_reduce(initial.executor_id, reconciliation)
    assert repository.load_snapshot(initial.executor_id) == current
    assert current.etf_filled_quantity == Decimal("0")


def test_prepared_deadline_must_not_precede_event_time(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    expired = _fact_event(
        initial,
        JournalEventType.PREPARED,
        "event-expired-prepared",
        deadline_utc=CREATED_AT,
        created_at_utc=UPDATED_AT,
    )

    with pytest.raises((JournalConflictError, JournalIntegrityError), match="deadline|expired|time"):
        repository.append_and_reduce(initial.executor_id, expired)
    assert repository.events(initial.executor_id) == ()


def test_fill_exchange_order_must_match_intent_binding(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    current = _append_fact(
        repository,
        initial,
        _fact_event(initial, JournalEventType.PREPARED, "event-bound-prepared"),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-bound-order-a",
            exchange_order_id="exchange-bound-a",
        ),
    )
    wrong_order_fill = _fact_event(
        initial,
        JournalEventType.FILL,
        "event-bound-fill-b",
        exchange_order_id="exchange-bound-b",
        exchange_trade_id="trade-bound-b",
    )

    with pytest.raises((JournalConflictError, JournalIntegrityError), match="order|bound|intent|identity"):
        repository.append_and_reduce(initial.executor_id, wrong_order_fill)
    assert repository.load_snapshot(initial.executor_id) == current


@pytest.mark.parametrize("identity_kind", ("client", "exchange_order"))
def test_cross_executor_order_identity_race_has_one_persistent_owner(
    manager: SQLConnectionManager,
    vectors: dict,
    identity_kind: str,
):
    repository = LeveragedEtfJournalRepository(manager)
    initial_a = _initial_snapshot(vectors)
    initial_b = _snapshot_for_executor(initial_a, "exec-sndk-snxx-race-b")
    repository.create_executor(initial_a)
    repository.create_executor(initial_b)

    if identity_kind == "client":
        events = (
            _fact_event(
                initial_a,
                JournalEventType.PREPARED,
                "event-client-owner-a",
                client_order_id="globally-shared-client",
            ),
            _fact_event(
                initial_b,
                JournalEventType.PREPARED,
                "event-client-owner-b",
                intent_id="intent-maker-owner-b",
                client_order_id="globally-shared-client",
            ),
        )
    else:
        prepared_a = _fact_event(initial_a, JournalEventType.PREPARED, "event-order-owner-prepared-a")
        prepared_b = _fact_event(
            initial_b,
            JournalEventType.PREPARED,
            "event-order-owner-prepared-b",
            intent_id="intent-maker-owner-b",
            client_order_id="exec-sndk-snxx-race-b-maker",
        )
        repository.append_and_reduce(initial_a.executor_id, prepared_a)
        repository.append_and_reduce(initial_b.executor_id, prepared_b)
        events = (
            _fact_event(
                initial_a,
                JournalEventType.ORDER_CREATED,
                "event-order-owner-a",
                exchange_order_id="globally-shared-exchange-order",
            ),
            _fact_event(
                initial_b,
                JournalEventType.ORDER_CREATED,
                "event-order-owner-b",
                intent_id="intent-maker-owner-b",
                client_order_id="exec-sndk-snxx-race-b-maker",
                exchange_order_id="globally-shared-exchange-order",
            ),
        )

    def append(owner: LeveragedEtfPairExecutorSnapshotV1, event: JournalEventV1):
        try:
            return LeveragedEtfJournalRepository(manager).append_and_reduce(owner.executor_id, event)
        except (JournalConflictError, JournalIntegrityError) as exception:
            return exception

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda item: append(*item), ((initial_a, events[0]), (initial_b, events[1]))))
    assert sum(not isinstance(result, Exception) for result in results) == 1
    assert sum(isinstance(result, Exception) for result in results) == 1


def test_recovery_rejects_persisted_cross_executor_client_owner_conflict(tmp_path: Path, vectors: dict):
    db_path = tmp_path / "global-owner-corruption.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial_a = _initial_snapshot(vectors)
    initial_b = _snapshot_for_executor(initial_a, "exec-sndk-snxx-owner-b")
    repository.create_executor(initial_a)
    repository.create_executor(initial_b)
    repository.append_and_reduce(
        initial_a.executor_id,
        _fact_event(initial_a, JournalEventType.PREPARED, "event-owner-corrupt-a"),
    )
    repository.append_and_reduce(
        initial_b.executor_id,
        _fact_event(
            initial_b,
            JournalEventType.PREPARED,
            "event-owner-corrupt-b",
            intent_id="intent-owner-corrupt-b",
            client_order_id="exec-sndk-snxx-owner-b-maker",
        ),
    )
    manager.engine.dispose()

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TRIGGER lepf_journal_no_update")
        connection.execute("DROP TRIGGER lepf_journal_client_order_owner_insert")
        row = connection.execute(
            "SELECT payload_json FROM LeveragedEtfJournalEvent WHERE event_id = ?",
            ("event-owner-corrupt-b",),
        ).fetchone()
        mutation = json.loads(row[0])
        mutation["event"]["client_order_id"] = "exec-sndk-snxx-0001-maker-fact-1"
        mutation["event"]["payload"]["identity"]["client_order_id"] = "exec-sndk-snxx-0001-maker-fact-1"
        payload_json, payload_hash = _canonical_json_hash(mutation)
        connection.execute(
            """
            UPDATE LeveragedEtfJournalEvent
            SET client_order_id = ?, payload_json = ?, payload_hash = ?
            WHERE event_id = ?
            """,
            ("exec-sndk-snxx-0001-maker-fact-1", payload_json, payload_hash, "event-owner-corrupt-b"),
        )
        connection.execute(SQLITE_GUARD_DDL["lepf_journal_client_order_owner_insert"])
        connection.execute(SQLITE_GUARD_DDL["lepf_journal_no_update"])

    reopened = _open_manager(db_path)
    try:
        with pytest.raises(JournalIntegrityError, match="client|owner|identity|global"):
            LeveragedEtfJournalRepository(reopened).incomplete_executors()
    finally:
        reopened.engine.dispose()


def test_multi_slice_partial_retry_fill_totals_replay_and_reopen(tmp_path: Path, vectors: dict):
    db_path = tmp_path / "multi-slice.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    current = initial

    maker_one = {
        "logical_quantity": "1",
        "order_quantity": "1",
        "attempt": 1,
        "intent_id": "intent-maker-slice-1",
        "client_order_id": "client-maker-slice-1",
    }
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.PREPARED, "event-maker-slice-1-prepared", **maker_one),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-maker-slice-1-created",
            exchange_order_id="exchange-maker-slice-1",
            **maker_one,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-maker-slice-1-fill-a",
            exchange_order_id="exchange-maker-slice-1",
            exchange_trade_id="trade-maker-slice-1-a",
            fill_quantity="0.4",
            order_cumulative_filled_quantity="0.4",
            leg_cumulative_filled_quantity="0.4",
            outcome="PARTIAL",
            **maker_one,
        ),
    )
    final_first_fill = _fact_event(
        initial,
        JournalEventType.FILL,
        "event-maker-slice-1-fill-b",
        exchange_order_id="exchange-maker-slice-1",
        exchange_trade_id="trade-maker-slice-1-b",
        fill_quantity="0.6",
        order_cumulative_filled_quantity="1",
        leg_cumulative_filled_quantity="1",
        outcome="FILLED",
        **maker_one,
    )
    current = _append_fact(repository, current, final_first_fill)
    committed_count = len(repository.events(initial.executor_id))
    repository.append_and_reduce(initial.executor_id, final_first_fill)
    assert len(repository.events(initial.executor_id)) == committed_count

    rejected_hedge = {
        "action": "STOCK_HEDGE",
        "leg": "STOCK",
        "logical_quantity": "1.2449",
        "order_quantity": "1.2449",
        "attempt": 1,
        "intent_id": "intent-stock-slice-1-rejected",
        "client_order_id": "client-stock-slice-1-rejected",
    }
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.PREPARED, "event-stock-retry-prepared", **rejected_hedge),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.HEDGE_REQUESTED, "event-stock-retry-requested", **rejected_hedge),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.REJECTED, "event-stock-retry-rejected", **rejected_hedge),
    )

    hedge_one = {
        **rejected_hedge,
        "attempt": 2,
        "intent_id": "intent-stock-slice-1",
        "client_order_id": "client-stock-slice-1",
    }
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.PREPARED, "event-stock-slice-1-prepared", **hedge_one),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.HEDGE_REQUESTED, "event-stock-slice-1-requested", **hedge_one),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-stock-slice-1-created",
            exchange_order_id="exchange-stock-slice-1",
            **hedge_one,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-stock-slice-1-fill-a",
            exchange_order_id="exchange-stock-slice-1",
            exchange_trade_id="trade-stock-slice-1-a",
            fill_quantity="0.5",
            order_cumulative_filled_quantity="0.5",
            leg_cumulative_filled_quantity="0.5",
            outcome="PARTIAL",
            **hedge_one,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-stock-slice-1-fill-b",
            exchange_order_id="exchange-stock-slice-1",
            exchange_trade_id="trade-stock-slice-1-b",
            fill_quantity="0.7449",
            order_cumulative_filled_quantity="1.2449",
            leg_cumulative_filled_quantity="1.2449",
            outcome="FILLED",
            **hedge_one,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.HEDGE_CONFIRMED, "event-stock-slice-1-confirmed", **hedge_one),
    )
    assert current.state.value == "MAKER_WORKING"

    maker_two = {
        "logical_quantity": "2",
        "order_quantity": "1",
        "attempt": 2,
        "intent_id": "intent-maker-slice-2",
        "client_order_id": "client-maker-slice-2",
    }
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.PREPARED, "event-maker-slice-2-prepared", **maker_two),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-maker-slice-2-created",
            exchange_order_id="exchange-maker-slice-2",
            **maker_two,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-maker-slice-2-fill",
            exchange_order_id="exchange-maker-slice-2",
            exchange_trade_id="trade-maker-slice-2",
            fill_quantity="1",
            order_cumulative_filled_quantity="1",
            leg_cumulative_filled_quantity="2",
            outcome="FILLED",
            **maker_two,
        ),
    )

    hedge_two = {
        "action": "STOCK_HEDGE",
        "leg": "STOCK",
        "logical_quantity": "2.4898",
        "order_quantity": "1.2449",
        "attempt": 3,
        "intent_id": "intent-stock-slice-2",
        "client_order_id": "client-stock-slice-2",
    }
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.PREPARED, "event-stock-slice-2-prepared", **hedge_two),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.HEDGE_REQUESTED, "event-stock-slice-2-requested", **hedge_two),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-stock-slice-2-created",
            exchange_order_id="exchange-stock-slice-2",
            **hedge_two,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-stock-slice-2-fill",
            exchange_order_id="exchange-stock-slice-2",
            exchange_trade_id="trade-stock-slice-2",
            fill_quantity="1.2449",
            order_cumulative_filled_quantity="1.2449",
            leg_cumulative_filled_quantity="2.4898",
            outcome="FILLED",
            **hedge_two,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.HEDGE_CONFIRMED, "event-stock-slice-2-confirmed", **hedge_two),
    )

    assert current.state.value == "MAKER_WORKING"
    assert current.etf_submitted_quantity == Decimal("2")
    assert current.etf_filled_quantity == Decimal("2")
    assert current.stock_submitted_quantity == Decimal("2.4898")
    assert current.stock_filled_quantity == Decimal("2.4898")
    assert tuple(order.exchange_order_id for order in current.maker_order_ids) == (
        "exchange-maker-slice-1",
        "exchange-maker-slice-2",
    )
    assert tuple(order.exchange_order_id for order in current.stock_order_ids) == (
        "exchange-stock-slice-1",
        "exchange-stock-slice-2",
    )
    assert repository.replay(initial.executor_id) == current
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        assert LeveragedEtfJournalRepository(reopened).replay(initial.executor_id) == current
    finally:
        reopened.engine.dispose()


def test_concurrent_hedge_reverse_order_interleaved_fills_use_executor_leg_target(
    tmp_path: Path,
    vectors: dict,
):
    db_path = tmp_path / "concurrent-hedge-fills.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)

    maker = {
        "logical_quantity": "2",
        "order_quantity": "2",
        "intent_id": "intent-concurrent-maker",
        "client_order_id": "client-concurrent-maker",
    }
    current = _append_fact(
        repository,
        initial,
        _fact_event(initial, JournalEventType.PREPARED, "event-concurrent-maker-prepared", **maker),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-concurrent-maker-created",
            exchange_order_id="exchange-concurrent-maker",
            **maker,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-concurrent-maker-fill",
            exchange_order_id="exchange-concurrent-maker",
            exchange_trade_id="trade-concurrent-maker",
            fill_quantity="2",
            order_cumulative_filled_quantity="2",
            leg_cumulative_filled_quantity="2",
            outcome="FILLED",
            **maker,
        ),
    )

    older_hedge = {
        "action": "STOCK_HEDGE",
        "leg": "STOCK",
        "logical_quantity": "1.2449",
        "order_quantity": "1.2449",
        "attempt": 1,
        "intent_id": "intent-concurrent-hedge-older",
        "client_order_id": "client-concurrent-hedge-older",
    }
    newer_hedge = {
        "action": "STOCK_HEDGE",
        "leg": "STOCK",
        "logical_quantity": "2.4898",
        "order_quantity": "1.2449",
        "attempt": 2,
        "intent_id": "intent-concurrent-hedge-newer",
        "client_order_id": "client-concurrent-hedge-newer",
    }
    for suffix, hedge, exchange_order_id in (
        ("older", older_hedge, "exchange-concurrent-hedge-older"),
        ("newer", newer_hedge, "exchange-concurrent-hedge-newer"),
    ):
        current = _append_fact(
            repository,
            current,
            _fact_event(
                initial,
                JournalEventType.PREPARED,
                f"event-concurrent-hedge-{suffix}-prepared",
                **hedge,
            ),
        )
        current = _append_fact(
            repository,
            current,
            _fact_event(
                initial,
                JournalEventType.HEDGE_REQUESTED,
                f"event-concurrent-hedge-{suffix}-requested",
                **hedge,
            ),
        )
        current = _append_fact(
            repository,
            current,
            _fact_event(
                initial,
                JournalEventType.ORDER_CREATED,
                f"event-concurrent-hedge-{suffix}-created",
                exchange_order_id=exchange_order_id,
                **hedge,
            ),
        )

    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-concurrent-hedge-newer-fill-a",
            exchange_order_id="exchange-concurrent-hedge-newer",
            exchange_trade_id="trade-concurrent-hedge-newer-a",
            fill_quantity="0.4",
            order_cumulative_filled_quantity="0.4",
            leg_cumulative_filled_quantity="0.4",
            outcome="PARTIAL",
            **newer_hedge,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-concurrent-hedge-older-fill-a",
            exchange_order_id="exchange-concurrent-hedge-older",
            exchange_trade_id="trade-concurrent-hedge-older-a",
            fill_quantity="0.5",
            order_cumulative_filled_quantity="0.5",
            leg_cumulative_filled_quantity="0.9",
            outcome="PARTIAL",
            **older_hedge,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-concurrent-hedge-newer-fill-b",
            exchange_order_id="exchange-concurrent-hedge-newer",
            exchange_trade_id="trade-concurrent-hedge-newer-b",
            fill_quantity="0.8449",
            order_cumulative_filled_quantity="1.2449",
            leg_cumulative_filled_quantity="1.7449",
            outcome="FILLED",
            **newer_hedge,
        ),
    )

    def assert_rejected_atomically(event: JournalEventV1) -> None:
        before_snapshot = repository.load_snapshot(initial.executor_id)
        before_events = repository.events(initial.executor_id)
        with pytest.raises((JournalConflictError, JournalIntegrityError)):
            repository.append_and_reduce(initial.executor_id, event)
        assert repository.load_snapshot(initial.executor_id) == before_snapshot
        assert repository.events(initial.executor_id) == before_events

    assert_rejected_atomically(
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-concurrent-hedge-wrong-order",
            exchange_order_id="exchange-concurrent-hedge-newer",
            exchange_trade_id="trade-concurrent-hedge-wrong-order",
            fill_quantity="0.7449",
            order_cumulative_filled_quantity="1.2449",
            leg_cumulative_filled_quantity="2.4898",
            outcome="FILLED",
            **older_hedge,
        )
    )
    assert_rejected_atomically(
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-concurrent-hedge-wrong-intent",
            exchange_order_id="exchange-concurrent-hedge-older",
            exchange_trade_id="trade-concurrent-hedge-wrong-intent",
            fill_quantity="0.1",
            order_cumulative_filled_quantity="1.3449",
            leg_cumulative_filled_quantity="1.8449",
            outcome="FILLED",
            **newer_hedge,
        )
    )
    assert_rejected_atomically(
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-concurrent-hedge-over-order-cap",
            exchange_order_id="exchange-concurrent-hedge-older",
            exchange_trade_id="trade-concurrent-hedge-over-order-cap",
            fill_quantity="0.8",
            order_cumulative_filled_quantity="1.3",
            leg_cumulative_filled_quantity="2.5449",
            outcome="FILLED",
            **older_hedge,
        )
    )
    assert_rejected_atomically(
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-concurrent-hedge-false-cumulative",
            exchange_order_id="exchange-concurrent-hedge-older",
            exchange_trade_id="trade-concurrent-hedge-older-b",
            fill_quantity="0.7449",
            order_cumulative_filled_quantity="1.2449",
            leg_cumulative_filled_quantity="2.4",
            outcome="FILLED",
            **older_hedge,
        )
    )

    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-concurrent-hedge-older-fill-b",
            exchange_order_id="exchange-concurrent-hedge-older",
            exchange_trade_id="trade-concurrent-hedge-older-b",
            fill_quantity="0.7449",
            order_cumulative_filled_quantity="1.2449",
            leg_cumulative_filled_quantity="2.4898",
            outcome="FILLED",
            **older_hedge,
        ),
    )

    assert current.state.value == "STOCK_HEDGE_PENDING"
    assert current.stock_submitted_quantity == Decimal("2.4898")
    assert current.stock_filled_quantity == Decimal("2.4898")
    assert tuple(order.exchange_order_id for order in current.stock_order_ids) == (
        "exchange-concurrent-hedge-older",
        "exchange-concurrent-hedge-newer",
    )
    assert repository.replay(initial.executor_id) == current
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        assert LeveragedEtfJournalRepository(reopened).replay(initial.executor_id) == current
    finally:
        reopened.engine.dispose()


@pytest.mark.parametrize(
    ("first_fill_quantity", "first_outcome"),
    (("0.4", "PARTIAL"), ("1", "FILLED")),
    ids=("partial-fill-first", "full-fill-first"),
)
def test_fill_before_ack_and_order_created_preserves_authoritative_progress(
    tmp_path: Path,
    vectors: dict,
    first_fill_quantity: str,
    first_outcome: str,
):
    db_path = tmp_path / f"fill-before-order-{first_outcome.lower()}.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    maker = {
        "logical_quantity": "1",
        "order_quantity": "1",
        "intent_id": f"intent-fill-before-order-{first_outcome.lower()}",
        "client_order_id": f"client-fill-before-order-{first_outcome.lower()}",
    }
    current = _append_fact(
        repository,
        initial,
        _fact_event(
            initial,
            JournalEventType.PREPARED,
            f"event-fill-before-order-{first_outcome.lower()}-prepared",
            **maker,
        ),
    )
    first_fill = _fact_event(
        initial,
        JournalEventType.FILL,
        f"event-fill-before-order-{first_outcome.lower()}-fill-a",
        exchange_order_id=f"exchange-fill-before-order-{first_outcome.lower()}",
        exchange_trade_id=f"trade-fill-before-order-{first_outcome.lower()}-a",
        fill_quantity=first_fill_quantity,
        order_cumulative_filled_quantity=first_fill_quantity,
        leg_cumulative_filled_quantity=first_fill_quantity,
        outcome=first_outcome,
        **maker,
    )
    current = _append_fact(repository, current, first_fill)

    assert current.state.value == "STOCK_HEDGE_PENDING"
    assert current.etf_filled_quantity == Decimal(first_fill_quantity)
    assert tuple(order.exchange_order_id for order in current.maker_order_ids) == (
        f"exchange-fill-before-order-{first_outcome.lower()}",
    )
    event_count = len(repository.events(initial.executor_id))
    repository.append_and_reduce(initial.executor_id, first_fill)
    assert len(repository.events(initial.executor_id)) == event_count
    assert repository.load_snapshot(initial.executor_id) == current

    def assert_rejected_atomically(event: JournalEventV1) -> None:
        before_snapshot = repository.load_snapshot(initial.executor_id)
        before_events = repository.events(initial.executor_id)
        with pytest.raises((JournalConflictError, JournalIntegrityError)):
            repository.append_and_reduce(initial.executor_id, event)
        assert repository.load_snapshot(initial.executor_id) == before_snapshot
        assert repository.events(initial.executor_id) == before_events

    assert_rejected_atomically(
        _fact_event(
            initial,
            JournalEventType.ACKNOWLEDGED,
            f"event-fill-before-order-{first_outcome.lower()}-conflicting-ack",
            client_order_id=f"{maker['client_order_id']}-conflict",
            logical_quantity="1",
            order_quantity="1",
            intent_id=maker["intent_id"],
        )
    )
    acknowledged = _fact_event(
        initial,
        JournalEventType.ACKNOWLEDGED,
        f"event-fill-before-order-{first_outcome.lower()}-ack",
        **maker,
    )
    current = _append_fact(repository, current, acknowledged)
    assert current.state.value == "STOCK_HEDGE_PENDING"
    assert current.etf_filled_quantity == Decimal(first_fill_quantity)
    acknowledged_count = len(repository.events(initial.executor_id))
    repository.append_and_reduce(initial.executor_id, acknowledged)
    assert len(repository.events(initial.executor_id)) == acknowledged_count

    assert_rejected_atomically(
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            f"event-fill-before-order-{first_outcome.lower()}-conflicting-created",
            exchange_order_id=f"exchange-fill-before-order-{first_outcome.lower()}-conflict",
            **maker,
        )
    )
    created = _fact_event(
        initial,
        JournalEventType.ORDER_CREATED,
        f"event-fill-before-order-{first_outcome.lower()}-created",
        exchange_order_id=f"exchange-fill-before-order-{first_outcome.lower()}",
        **maker,
    )
    current = _append_fact(repository, current, created)
    assert current.state.value == "STOCK_HEDGE_PENDING"
    assert current.etf_filled_quantity == Decimal(first_fill_quantity)
    assert len(current.maker_order_ids) == 1
    created_count = len(repository.events(initial.executor_id))
    repository.append_and_reduce(initial.executor_id, created)
    assert len(repository.events(initial.executor_id)) == created_count

    if first_outcome == "PARTIAL":
        current = _append_fact(
            repository,
            current,
            _fact_event(
                initial,
                JournalEventType.FILL,
                "event-fill-before-order-partial-fill-b",
                exchange_order_id="exchange-fill-before-order-partial",
                exchange_trade_id="trade-fill-before-order-partial-b",
                fill_quantity="0.6",
                order_cumulative_filled_quantity="1",
                leg_cumulative_filled_quantity="1",
                outcome="FILLED",
                **maker,
            ),
        )

    assert current.state.value == "STOCK_HEDGE_PENDING"
    assert current.etf_filled_quantity == Decimal("1")
    assert repository.incomplete_intents() == ()
    assert repository.replay(initial.executor_id) == current
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        assert reopened_repository.replay(initial.executor_id) == current
        assert reopened_repository.incomplete_intents() == ()
    finally:
        reopened.engine.dispose()


def _concurrent_hedges_filled_in_reverse_order(
    db_path: Path,
    vectors: dict,
) -> tuple[
    SQLConnectionManager,
    LeveragedEtfJournalRepository,
    LeveragedEtfPairExecutorSnapshotV1,
    LeveragedEtfPairExecutorSnapshotV1,
    dict[str, dict],
]:
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    maker = {
        "logical_quantity": "2",
        "order_quantity": "2",
        "intent_id": "intent-r5-concurrent-maker",
        "client_order_id": "client-r5-concurrent-maker",
    }
    current = _append_fact(
        repository,
        initial,
        _fact_event(initial, JournalEventType.PREPARED, "event-r5-concurrent-maker-prepared", **maker),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-r5-concurrent-maker-created",
            exchange_order_id="exchange-r5-concurrent-maker",
            **maker,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-r5-concurrent-maker-fill",
            exchange_order_id="exchange-r5-concurrent-maker",
            exchange_trade_id="trade-r5-concurrent-maker",
            fill_quantity="2",
            order_cumulative_filled_quantity="2",
            leg_cumulative_filled_quantity="2",
            outcome="FILLED",
            **maker,
        ),
    )
    hedges = {
        "older": {
            "action": "STOCK_HEDGE",
            "leg": "STOCK",
            "logical_quantity": "1.2449",
            "order_quantity": "1.2449",
            "attempt": 1,
            "intent_id": "intent-r5-concurrent-older",
            "client_order_id": "client-r5-concurrent-older",
            "exchange_order_id": "exchange-r5-concurrent-older",
        },
        "newer": {
            "action": "STOCK_HEDGE",
            "leg": "STOCK",
            "logical_quantity": "2.4898",
            "order_quantity": "1.2449",
            "attempt": 2,
            "intent_id": "intent-r5-concurrent-newer",
            "client_order_id": "client-r5-concurrent-newer",
            "exchange_order_id": "exchange-r5-concurrent-newer",
        },
    }
    for name in ("older", "newer"):
        hedge = {key: value for key, value in hedges[name].items() if key != "exchange_order_id"}
        current = _append_fact(
            repository,
            current,
            _fact_event(
                initial,
                JournalEventType.PREPARED,
                f"event-r5-concurrent-{name}-prepared",
                **hedge,
            ),
        )
        current = _append_fact(
            repository,
            current,
            _fact_event(
                initial,
                JournalEventType.HEDGE_REQUESTED,
                f"event-r5-concurrent-{name}-requested",
                **hedge,
            ),
        )
        current = _append_fact(
            repository,
            current,
            _fact_event(
                initial,
                JournalEventType.ORDER_CREATED,
                f"event-r5-concurrent-{name}-created",
                exchange_order_id=hedges[name]["exchange_order_id"],
                **hedge,
            ),
        )
    fills = (
        ("newer", "a", "0.4", "0.4", "0.4", "PARTIAL"),
        ("older", "a", "0.5", "0.5", "0.9", "PARTIAL"),
        ("newer", "b", "0.8449", "1.2449", "1.7449", "FILLED"),
        ("older", "b", "0.7449", "1.2449", "2.4898", "FILLED"),
    )
    for name, suffix, fill_quantity, order_cumulative, leg_cumulative, outcome in fills:
        hedge = {key: value for key, value in hedges[name].items() if key != "exchange_order_id"}
        current = _append_fact(
            repository,
            current,
            _fact_event(
                initial,
                JournalEventType.FILL,
                f"event-r5-concurrent-{name}-fill-{suffix}",
                exchange_order_id=hedges[name]["exchange_order_id"],
                exchange_trade_id=f"trade-r5-concurrent-{name}-{suffix}",
                fill_quantity=fill_quantity,
                order_cumulative_filled_quantity=order_cumulative,
                leg_cumulative_filled_quantity=leg_cumulative,
                outcome=outcome,
                **hedge,
            ),
        )
    return manager, repository, initial, current, hedges


@pytest.mark.parametrize(
    ("first_name", "first_type", "second_name", "second_type"),
    (
        ("newer", JournalEventType.HEDGE_CONFIRMED, "older", JournalEventType.RECONCILIATION),
        ("older", JournalEventType.RECONCILIATION, "newer", JournalEventType.HEDGE_CONFIRMED),
    ),
    ids=("newer-confirmed-first", "older-reconciled-first"),
)
def test_concurrent_hedge_terminal_facts_close_in_either_order(
    tmp_path: Path,
    vectors: dict,
    first_name: str,
    first_type: JournalEventType,
    second_name: str,
    second_type: JournalEventType,
):
    db_path = tmp_path / f"concurrent-terminal-{first_name}.sqlite"
    manager, repository, initial, current, hedges = _concurrent_hedges_filled_in_reverse_order(db_path, vectors)

    def terminal_fact(name: str, event_type: JournalEventType, suffix: str) -> JournalEventV1:
        hedge = {key: value for key, value in hedges[name].items() if key != "exchange_order_id"}
        parameters = {}
        if event_type == JournalEventType.RECONCILIATION:
            parameters = {
                "exchange_order_id": hedges[name]["exchange_order_id"],
                "order_cumulative_filled_quantity": "1.2449",
                "outcome": "FILLED",
            }
        return _fact_event(
            initial,
            event_type,
            f"event-r5-terminal-{suffix}-{name}",
            **parameters,
            **hedge,
        )

    current = _append_fact(repository, current, terminal_fact(first_name, first_type, "first"))
    assert current.state.value == "STOCK_HEDGE_PENDING"
    assert tuple(intent.intent_id for intent in repository.incomplete_intents(initial.executor_id)) == (
        hedges[second_name]["intent_id"],
    )

    second_hedge = {key: value for key, value in hedges[second_name].items() if key != "exchange_order_id"}
    _assert_fact_rejected_atomically(
        repository,
        initial.executor_id,
        _fact_event(
            initial,
            JournalEventType.RECONCILIATION,
            f"event-r5-terminal-conflicting-{second_name}",
            exchange_order_id=hedges[second_name]["exchange_order_id"],
            order_cumulative_filled_quantity="1",
            outcome="FILLED",
            **second_hedge,
        ),
    )

    current = _append_fact(repository, current, terminal_fact(second_name, second_type, "second"))
    assert current.state.value == "MAKER_WORKING"
    assert current.etf_filled_quantity == Decimal("2")
    assert current.stock_filled_quantity == Decimal("2.4898")
    assert current.etf_filled_quantity * Decimal("1.2449") == current.stock_filled_quantity
    assert repository.incomplete_intents(initial.executor_id) == ()
    terminal_audit = tuple(
        (committed.event.intent_id, committed.event.event_type)
        for committed in repository.events(initial.executor_id)
        if committed.event.event_id.startswith("event-r5-terminal-")
    )
    assert terminal_audit == (
        (hedges[first_name]["intent_id"], first_type),
        (hedges[second_name]["intent_id"], second_type),
    )
    assert repository.replay(initial.executor_id) == current
    committed_events = repository.events(initial.executor_id)
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        assert reopened_repository.replay(initial.executor_id) == current
        assert reopened_repository.incomplete_intents(initial.executor_id) == ()
        assert reopened_repository.events(initial.executor_id) == committed_events
    finally:
        reopened.engine.dispose()


@pytest.mark.parametrize(
    ("fill_quantity", "fill_outcome"),
    (("0.4", "PARTIAL"), ("1", "FILLED")),
    ids=("partial-fill", "full-fill"),
)
def test_late_maker_facts_survive_intervening_rollback_state(
    tmp_path: Path,
    vectors: dict,
    fill_quantity: str,
    fill_outcome: str,
):
    db_path = tmp_path / f"late-maker-rollback-{fill_outcome.lower()}.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    maker = {
        "logical_quantity": "1",
        "order_quantity": "1",
        "intent_id": f"intent-r5-late-maker-{fill_outcome.lower()}",
        "client_order_id": f"client-r5-late-maker-{fill_outcome.lower()}",
    }
    current = _append_fact(
        repository,
        initial,
        _fact_event(
            initial,
            JournalEventType.PREPARED,
            f"event-r5-late-maker-{fill_outcome.lower()}-prepared",
            **maker,
        ),
    )
    exchange_order_id = f"exchange-r5-late-maker-{fill_outcome.lower()}"
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            f"event-r5-late-maker-{fill_outcome.lower()}-fill",
            exchange_order_id=exchange_order_id,
            exchange_trade_id=f"trade-r5-late-maker-{fill_outcome.lower()}",
            fill_quantity=fill_quantity,
            order_cumulative_filled_quantity=fill_quantity,
            leg_cumulative_filled_quantity=fill_quantity,
            outcome=fill_outcome,
            **maker,
        ),
    )
    rollback = {
        "action": "ETF_ROLLBACK",
        "leg": "ETF",
        "logical_quantity": fill_quantity,
        "order_quantity": fill_quantity,
        "intent_id": f"intent-r5-rollback-{fill_outcome.lower()}",
        "client_order_id": f"client-r5-rollback-{fill_outcome.lower()}",
    }
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.PREPARED,
            f"event-r5-rollback-{fill_outcome.lower()}-prepared",
            **rollback,
        ),
    )
    assert current.state.value == "ETF_ROLLBACK_PENDING"
    frozen_exposure = (current.etf_filled_quantity, current.stock_filled_quantity)

    _assert_fact_rejected_atomically(
        repository,
        initial.executor_id,
        _fact_event(
            initial,
            JournalEventType.ACKNOWLEDGED,
            f"event-r5-late-maker-{fill_outcome.lower()}-wrong-ack",
            logical_quantity="1",
            order_quantity="1",
            intent_id=maker["intent_id"],
            client_order_id=f"{maker['client_order_id']}-conflict",
        ),
    )
    acknowledged = _fact_event(
        initial,
        JournalEventType.ACKNOWLEDGED,
        f"event-r5-late-maker-{fill_outcome.lower()}-ack",
        **maker,
    )
    current = _append_fact(repository, current, acknowledged)
    assert current.state.value == "ETF_ROLLBACK_PENDING"
    assert (current.etf_filled_quantity, current.stock_filled_quantity) == frozen_exposure
    event_count = len(repository.events(initial.executor_id))
    repository.append_and_reduce(initial.executor_id, acknowledged)
    assert len(repository.events(initial.executor_id)) == event_count

    _assert_fact_rejected_atomically(
        repository,
        initial.executor_id,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            f"event-r5-late-maker-{fill_outcome.lower()}-wrong-created",
            exchange_order_id=f"{exchange_order_id}-conflict",
            **maker,
        ),
    )
    created = _fact_event(
        initial,
        JournalEventType.ORDER_CREATED,
        f"event-r5-late-maker-{fill_outcome.lower()}-created",
        exchange_order_id=exchange_order_id,
        **maker,
    )
    current = _append_fact(repository, current, created)
    assert current.state.value == "ETF_ROLLBACK_PENDING"
    assert (current.etf_filled_quantity, current.stock_filled_quantity) == frozen_exposure
    assert tuple(order.exchange_order_id for order in current.maker_order_ids) == (exchange_order_id,)
    event_count = len(repository.events(initial.executor_id))
    repository.append_and_reduce(initial.executor_id, created)
    assert len(repository.events(initial.executor_id)) == event_count
    late_audit = tuple(
        committed.event.event_id
        for committed in repository.events(initial.executor_id)
        if "late-maker" in committed.event.event_id
    )
    assert late_audit[-2:] == (acknowledged.event_id, created.event_id)
    assert repository.replay(initial.executor_id) == current
    committed_events = repository.events(initial.executor_id)
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        assert reopened_repository.replay(initial.executor_id) == current
        assert reopened_repository.events(initial.executor_id) == committed_events
    finally:
        reopened.engine.dispose()


def test_late_stock_facts_survive_post_confirmation_maker_state(tmp_path: Path, vectors: dict):
    db_path = tmp_path / "late-stock-post-confirmation.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    repository.create_executor(initial)
    maker = {
        "logical_quantity": "1",
        "order_quantity": "1",
        "intent_id": "intent-r5-post-confirm-maker",
        "client_order_id": "client-r5-post-confirm-maker",
    }
    current = _append_fact(
        repository,
        initial,
        _fact_event(initial, JournalEventType.PREPARED, "event-r5-post-confirm-maker-prepared", **maker),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-r5-post-confirm-maker-created",
            exchange_order_id="exchange-r5-post-confirm-maker",
            **maker,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-r5-post-confirm-maker-fill",
            exchange_order_id="exchange-r5-post-confirm-maker",
            exchange_trade_id="trade-r5-post-confirm-maker",
            fill_quantity="1",
            order_cumulative_filled_quantity="1",
            leg_cumulative_filled_quantity="1",
            outcome="FILLED",
            **maker,
        ),
    )
    hedge = {
        "action": "STOCK_HEDGE",
        "leg": "STOCK",
        "logical_quantity": "1.2449",
        "order_quantity": "1.2449",
        "intent_id": "intent-r5-post-confirm-hedge",
        "client_order_id": "client-r5-post-confirm-hedge",
    }
    current = _append_fact(
        repository,
        current,
        _fact_event(initial, JournalEventType.PREPARED, "event-r5-post-confirm-hedge-prepared", **hedge),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.HEDGE_REQUESTED,
            "event-r5-post-confirm-hedge-requested",
            **hedge,
        ),
    )
    stock_exchange_order_id = "exchange-r5-post-confirm-hedge"
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.FILL,
            "event-r5-post-confirm-hedge-fill",
            exchange_order_id=stock_exchange_order_id,
            exchange_trade_id="trade-r5-post-confirm-hedge",
            fill_quantity="1.2449",
            order_cumulative_filled_quantity="1.2449",
            leg_cumulative_filled_quantity="1.2449",
            outcome="FILLED",
            **hedge,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.HEDGE_CONFIRMED,
            "event-r5-post-confirm-hedge-confirmed",
            **hedge,
        ),
    )
    assert current.state.value == "MAKER_WORKING"
    assert repository.incomplete_intents(initial.executor_id) == ()
    frozen_exposure = (current.etf_filled_quantity, current.stock_filled_quantity)

    _assert_fact_rejected_atomically(
        repository,
        initial.executor_id,
        _fact_event(
            initial,
            JournalEventType.ACKNOWLEDGED,
            "event-r5-post-confirm-wrong-ack",
            action="STOCK_HEDGE",
            leg="STOCK",
            logical_quantity="1.2449",
            order_quantity="1.2449",
            intent_id=hedge["intent_id"],
            client_order_id=f"{hedge['client_order_id']}-conflict",
        ),
    )
    acknowledged = _fact_event(
        initial,
        JournalEventType.ACKNOWLEDGED,
        "event-r5-post-confirm-ack",
        **hedge,
    )
    current = _append_fact(repository, current, acknowledged)
    assert current.state.value == "MAKER_WORKING"
    assert (current.etf_filled_quantity, current.stock_filled_quantity) == frozen_exposure
    event_count = len(repository.events(initial.executor_id))
    repository.append_and_reduce(initial.executor_id, acknowledged)
    assert len(repository.events(initial.executor_id)) == event_count

    _assert_fact_rejected_atomically(
        repository,
        initial.executor_id,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-r5-post-confirm-wrong-created",
            exchange_order_id=f"{stock_exchange_order_id}-conflict",
            **hedge,
        ),
    )
    created = _fact_event(
        initial,
        JournalEventType.ORDER_CREATED,
        "event-r5-post-confirm-created",
        exchange_order_id=stock_exchange_order_id,
        **hedge,
    )
    current = _append_fact(repository, current, created)
    assert current.state.value == "MAKER_WORKING"
    assert (current.etf_filled_quantity, current.stock_filled_quantity) == frozen_exposure
    assert tuple(order.exchange_order_id for order in current.stock_order_ids) == (stock_exchange_order_id,)
    event_count = len(repository.events(initial.executor_id))
    repository.append_and_reduce(initial.executor_id, created)
    assert len(repository.events(initial.executor_id)) == event_count
    assert repository.incomplete_intents(initial.executor_id) == ()
    late_audit = tuple(
        committed.event.event_id
        for committed in repository.events(initial.executor_id)
        if committed.event.event_id.startswith("event-r5-post-confirm-")
    )
    assert late_audit[-2:] == (acknowledged.event_id, created.event_id)
    assert repository.replay(initial.executor_id) == current
    committed_events = repository.events(initial.executor_id)
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        assert reopened_repository.replay(initial.executor_id) == current
        assert reopened_repository.incomplete_intents(initial.executor_id) == ()
        assert reopened_repository.events(initial.executor_id) == committed_events
    finally:
        reopened.engine.dispose()


def _opaque_final_anchor(
    *,
    official_close_utc: str,
    payload_official_close_utc: str,
    provider_state: str = "initial",
) -> OpaqueAnchorFinalizedV1:
    cycle_id = "xnys-2026-07-21"
    target_session_date = "2026-07-21"
    deadline_utc = "2026-07-21T20:10:00.000000Z"
    evidence_hash = "7" * 64
    return OpaqueAnchorFinalizedV1(
        cycle_id=cycle_id,
        target_session_date=target_session_date,
        official_close_utc=official_close_utc,
        deadline_utc=deadline_utc,
        revision=1,
        evidence_hash=evidence_hash,
        payload=CanonicalOpaquePayload.from_value(
            2,
            "ANCHOR_RECORD",
            {
                "schema_version": 2,
                "cycle_id": cycle_id,
                "target_session_date": target_session_date,
                "official_close_utc": payload_official_close_utc,
                "deadline_utc": deadline_utc,
                "evidence_hash": evidence_hash,
                "finalized_at_utc": "2026-07-21T20:05:00.000000Z",
                "provider_state": provider_state,
            },
        ),
    )


def _seed_checkpoint_for_opaque_final(
    repository: _V1WireAnchorStorageAdapter,
    final: OpaqueAnchorFinalizedV1,
) -> None:
    repository.compare_and_set_opaque_checkpoint(
        OpaqueAnchorCheckpointV1(
            cycle_id=final.cycle_id,
            target_session_date=final.target_session_date,
            official_close_utc=final.model_dump(mode="json")["official_close_utc"],
            deadline_utc=final.model_dump(mode="json")["deadline_utc"],
            revision=1,
            payload=CanonicalOpaquePayload.from_value(
                2,
                "ANCHOR_CHECKPOINT",
                {
                    "schema_version": 2,
                    "cycle_id": final.cycle_id,
                    "target_session_date": final.target_session_date,
                    "official_close_utc": final.model_dump(mode="json")["official_close_utc"],
                    "deadline_utc": final.model_dump(mode="json")["deadline_utc"],
                    "next_poll_utc": final.model_dump(mode="json")["official_close_utc"],
                    "revision": 1,
                },
            ),
        ),
        expected_revision=0,
    )


def test_opaque_anchor_rejects_official_close_mismatch_on_absent_insert(manager: SQLConnectionManager):
    repository = _V1WireAnchorStorageAdapter(manager)
    mismatched = _opaque_final_anchor(
        official_close_utc="2026-07-21T20:00:00.000000Z",
        payload_official_close_utc="2026-07-21T20:01:00.000000Z",
    )

    with pytest.raises(AnchorIntegrityError, match="official|close|identity|metadata"):
        repository.finalize_opaque_if_absent(mismatched, expected_revision=0)
    assert repository.load_opaque(mismatched.cycle_id) is None


def test_opaque_anchor_rejects_official_close_mismatch_when_finalizing_checkpoint(
    manager: SQLConnectionManager,
):
    repository = _V1WireAnchorStorageAdapter(manager)
    final = _opaque_final_anchor(
        official_close_utc="2026-07-21T20:00:00.000000Z",
        payload_official_close_utc="2026-07-21T20:01:00.000000Z",
    )
    checkpoint = OpaqueAnchorCheckpointV1(
        cycle_id=final.cycle_id,
        target_session_date=final.target_session_date,
        official_close_utc="2026-07-21T20:00:00.000000Z",
        deadline_utc=final.deadline_utc,
        revision=1,
        payload=CanonicalOpaquePayload.from_value(
            2,
            "ANCHOR_CHECKPOINT",
            {
                "schema_version": 2,
                "cycle_id": final.cycle_id,
                "target_session_date": final.target_session_date,
                "official_close_utc": "2026-07-21T20:00:00.000000Z",
                "deadline_utc": "2026-07-21T20:10:00.000000Z",
                "next_poll_utc": "2026-07-21T20:00:00.000000Z",
                "revision": 1,
            },
        ),
    )
    repository.compare_and_set_opaque_checkpoint(checkpoint, expected_revision=0)

    with pytest.raises(AnchorIntegrityError, match="official|close|identity|metadata"):
        repository.finalize_opaque_if_absent(final, expected_revision=1)
    assert repository.load_opaque(final.cycle_id) == checkpoint


def test_opaque_anchor_rejects_official_close_mismatch_on_idempotent_retry(
    manager: SQLConnectionManager,
):
    repository = _V1WireAnchorStorageAdapter(manager)
    valid = _opaque_final_anchor(
        official_close_utc="2026-07-21T20:00:00.000000Z",
        payload_official_close_utc="2026-07-21T20:00:00.000000Z",
    )
    _seed_checkpoint_for_opaque_final(repository, valid)
    repository.finalize_opaque_if_absent(valid, expected_revision=1)
    mismatched = _opaque_final_anchor(
        official_close_utc="2026-07-21T20:00:00.000000Z",
        payload_official_close_utc="2026-07-21T20:01:00.000000Z",
    )

    with pytest.raises(AnchorIntegrityError, match="official|close|identity|metadata"):
        repository.finalize_opaque_if_absent(mismatched, expected_revision=1)
    assert repository.load_opaque(valid.cycle_id) == valid


def test_opaque_anchor_reopen_rejects_persisted_official_close_mismatch(tmp_path: Path):
    db_path = tmp_path / "anchor-close-corruption.sqlite"
    manager = _open_manager(db_path)
    repository = _V1WireAnchorStorageAdapter(manager)
    valid = _opaque_final_anchor(
        official_close_utc="2026-07-21T20:00:00.000000Z",
        payload_official_close_utc="2026-07-21T20:00:00.000000Z",
    )
    _seed_checkpoint_for_opaque_final(repository, valid)
    repository.finalize_opaque_if_absent(valid, expected_revision=1)
    manager.engine.dispose()

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TRIGGER lepf_anchor_finalized_no_update")
        payload = valid.payload.value()
        payload["official_close_utc"] = "2026-07-21T20:01:00.000000Z"
        payload_json, payload_hash = _canonical_json_hash(payload)
        connection.execute(
            "UPDATE LeveragedEtfAnchorState SET payload_json = ?, payload_hash = ? WHERE cycle_id = ?",
            (payload_json, payload_hash, valid.cycle_id),
        )
        connection.execute(SQLITE_GUARD_DDL["lepf_anchor_finalized_no_update"])

    reopened = _open_manager(db_path)
    try:
        with pytest.raises(AnchorIntegrityError, match="official|close|identity|metadata"):
            _V1WireAnchorStorageAdapter(reopened).load_opaque(valid.cycle_id)
    finally:
        reopened.engine.dispose()
