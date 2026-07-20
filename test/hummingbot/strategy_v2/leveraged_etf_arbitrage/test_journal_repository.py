import json
import sqlite3
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
    JournalEventType,
    JournalEventV1,
    JournalIntegrityError,
    LeveragedEtfJournalRepository,
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
