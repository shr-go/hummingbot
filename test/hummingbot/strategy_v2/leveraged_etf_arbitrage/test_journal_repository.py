import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import text

from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.model.leveraged_etf_repository import (
    AnchorIntegrityError,
    AnchorPollingCheckpointV1,
    AnchorRecordV1,
    AnchorRepositoryV1,
    AnchorRevisionConflict,
    CanonicalOpaquePayload,
    JournalConflictError,
    JournalEventType,
    JournalEventV1,
    JournalIntegrityError,
    LeveragedEtfJournalRepository,
    OpaqueAnchorCheckpointV1,
    StrategyReservationV1,
)
from hummingbot.model.sql_connection_manager import SQLConnectionManager, SQLConnectionType
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import LeveragedEtfPairExecutorSnapshotV1

CREATED_AT = "2026-07-17T14:01:02.000000Z"
UPDATED_AT = "2026-07-17T14:01:03.123456Z"


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


def _prepared_event(payload_value: str = "40") -> JournalEventV1:
    return JournalEventV1(
        event_id="event-maker-prepared-1",
        event_type=JournalEventType.PREPARED,
        intent_id="intent-maker-1",
        idempotency_key="exec-sndk-snxx-0001:OPEN:ETF:40:1",
        connector_name="binance_perpetual",
        trading_pair="SNXX-USDT",
        client_order_id="exec-sndk-snxx-0001-maker-1",
        payload=CanonicalOpaquePayload.from_value(
            schema_version=1,
            kind="ORDER_INTENT",
            value={
                "exposure_increasing": True,
                "leg": "ETF",
                "logical_quantity": payload_value,
                "operation": "OPEN",
            },
        ),
        created_at_utc=UPDATED_AT,
    )


def _followup_event(
    event_type: JournalEventType,
    event_id: str,
    *,
    terminal: bool = False,
) -> JournalEventV1:
    return JournalEventV1(
        event_id=event_id,
        event_type=event_type,
        intent_id="intent-maker-1",
        idempotency_key="exec-sndk-snxx-0001:OPEN:ETF:40:1",
        connector_name="binance_perpetual",
        trading_pair="SNXX-USDT",
        client_order_id="exec-sndk-snxx-0001-maker-1",
        payload=CanonicalOpaquePayload.from_value(
            schema_version=1,
            kind="ORDER_STATUS",
            value={"terminal": terminal},
        ),
        created_at_utc=UPDATED_AT,
    )


def _reservation(
    reservation_id: str = "reservation-1",
    reservation_key: str = "exec-sndk-snxx-0001:ETF:40",
) -> StrategyReservationV1:
    return StrategyReservationV1(
        reservation_id=reservation_id,
        executor_id="exec-sndk-snxx-0001",
        reservation_key=reservation_key,
        connector_name="binance_perpetual",
        trading_pair="SNXX-USDT",
        leg="ETF",
        quantity="40",
        leverage=20,
        notional_cap="1000000",
        payload=CanonicalOpaquePayload.from_value(
            schema_version=1,
            kind="LEVERAGE_RESERVATION",
            value={"logical_quantity": "40"},
        ),
        created_at_utc=CREATED_AT,
        updated_at_utc=CREATED_AT,
        released_at_utc=None,
    )


def test_journal_append_replay_and_stable_idempotency(manager: SQLConnectionManager, vectors: dict):
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository.create_executor(initial)

    committed = repository.append_and_reduce(initial.executor_id, _prepared_event(), prepared)
    duplicate = repository.append_and_reduce(initial.executor_id, _prepared_event(), prepared)

    assert committed == duplicate
    assert committed.sequence == 1
    assert repository.load_snapshot(initial.executor_id) == prepared
    assert repository.replay(initial.executor_id) == prepared
    assert len(repository.events(initial.executor_id)) == 1

    with pytest.raises(JournalIntegrityError, match="event|payload|idempotency"):
        repository.append_and_reduce(initial.executor_id, _prepared_event("41"), prepared)


def test_journal_commit_failure_rolls_back_event_and_snapshot(manager: SQLConnectionManager, vectors: dict):
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    repository = LeveragedEtfJournalRepository(manager)
    repository.create_executor(initial)

    def fail_commit(_connection) -> None:
        raise RuntimeError("forced commit failure")

    failing_repository = LeveragedEtfJournalRepository(manager, before_commit=fail_commit)
    with pytest.raises(RuntimeError, match="forced commit failure"):
        failing_repository.append_and_reduce(initial.executor_id, _prepared_event(), prepared)

    assert repository.events(initial.executor_id) == ()
    assert repository.load_snapshot(initial.executor_id) == initial


def test_concurrent_duplicate_event_has_one_committed_sequence(manager: SQLConnectionManager, vectors: dict):
    initial = _initial_snapshot(vectors)
    prepared = _prepared_snapshot(initial)
    LeveragedEtfJournalRepository(manager).create_executor(initial)

    def append_duplicate(_index: int):
        return LeveragedEtfJournalRepository(manager).append_and_reduce(
            initial.executor_id,
            _prepared_event(),
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
    repository.append_and_reduce(initial.executor_id, _prepared_event(), prepared)

    conflicting_payload = _prepared_event().model_dump(mode="json")
    conflicting_payload.update(
        {
            "event_id": "event-maker-prepared-2",
            "intent_id": "intent-maker-2",
            "idempotency_key": "exec-sndk-snxx-0001:OPEN:ETF:40:2",
        }
    )
    with pytest.raises(JournalConflictError, match="logical|non-terminal|intent"):
        repository.append_and_reduce(
            initial.executor_id,
            JournalEventV1.model_validate(conflicting_payload),
            _snapshot_at(prepared, 2),
        )

    acknowledged = _snapshot_at(prepared, 2, "MAKER_WORKING")
    repository.append_and_reduce(
        initial.executor_id,
        _followup_event(JournalEventType.ACKNOWLEDGED, "event-maker-ack-1"),
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
            _followup_event(JournalEventType.ACKNOWLEDGED, "event-maker-ack-orphan"),
            _snapshot_at(initial, 1),
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
    orphan_ack = JournalEventV1(
        event_id="event-maker-ack-without-intent",
        event_type=JournalEventType.ACKNOWLEDGED,
        connector_name="binance_perpetual",
        trading_pair="SNXX-USDT",
        client_order_id="exec-sndk-snxx-0001-maker-1",
        payload=CanonicalOpaquePayload.from_value(
            schema_version=1,
            kind="ORDER_STATUS",
            value={"terminal": False},
        ),
        created_at_utc=UPDATED_AT,
    )

    with pytest.raises(JournalConflictError, match="intent|PREPARED"):
        repository.append_and_reduce(initial.executor_id, orphan_ack, _snapshot_at(initial, 1))
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
    repository.append_and_reduce(initial.executor_id, _prepared_event(), prepared)
    fill = JournalEventV1(
        event_id="event-maker-fill-1",
        event_type=JournalEventType.FILL,
        intent_id="intent-maker-1",
        connector_name="binance_perpetual",
        trading_pair="SNXX-USDT",
        client_order_id="exec-sndk-snxx-0001-maker-1",
        exchange_order_id="exchange-order-1",
        exchange_trade_id="exchange-trade-1",
        payload=CanonicalOpaquePayload.from_value(
            schema_version=1,
            kind="ORDER_FILL",
            value={"price": "30", "quantity": "1", "terminal": False},
        ),
        created_at_utc=UPDATED_AT,
    )
    filled_snapshot = _snapshot_at(prepared, 2, "MAKER_PARTIALLY_FILLED")
    committed = repository.append_and_reduce(initial.executor_id, fill, filled_snapshot)
    duplicate_fill = JournalEventV1.model_validate(
        {
            **fill.model_dump(mode="json"),
            "event_id": "event-maker-fill-duplicate",
        }
    )

    assert repository.append_and_reduce(
        initial.executor_id,
        duplicate_fill,
        _snapshot_at(filled_snapshot, 3),
    ) == committed
    assert repository.load_snapshot(initial.executor_id) == filled_snapshot
    assert tuple(event.sequence for event in repository.events(initial.executor_id)) == (1, 2)

    distinct_fill = JournalEventV1.model_validate(
        {
            **fill.model_dump(mode="json"),
            "event_id": "event-maker-fill-conflict",
            "payload": CanonicalOpaquePayload.from_value(
                schema_version=1,
                kind="ORDER_FILL",
                value={"price": "30", "quantity": "2", "terminal": False},
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
            _followup_event(JournalEventType.ACKNOWLEDGED, "event-maker-ack-out-of-order"),
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
    repository.append_and_reduce(initial.executor_id, _prepared_event(), prepared)
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
    repository = AnchorRepositoryV1(manager)
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
    repository = AnchorRepositoryV1(manager)
    payload = CanonicalOpaquePayload.from_value(
        schema_version=2,
        kind="ANCHOR_CHECKPOINT",
        value={
            "schema_version": 2,
            "cycle_id": "xnys-2026-07-18",
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
        value={**payload.value(), "provider_state": {"cursor": "next", "samples": ["250", "30"]}},
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
    repository = AnchorRepositoryV1(manager)
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
        assert AnchorRepositoryV1(reopened).load(record.cycle_id) == record
    finally:
        reopened.engine.dispose()


def test_anchor_revision_observations_append_without_mutating_final_record(
    manager: SQLConnectionManager,
    vectors: dict,
):
    repository = AnchorRepositoryV1(manager)
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
