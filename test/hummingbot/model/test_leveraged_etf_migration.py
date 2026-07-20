import asyncio
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from hummingbot.client.config.client_config_map import ClientConfigMap, MarketDataCollectionConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.connector.markets_recorder import MarketsRecorder
from hummingbot.model.sql_connection_manager import SQLConnectionManager, SQLConnectionType

TARGET_VERSION = "20260719"
TARGET_TABLES = {
    "LeveragedEtfAnchorRevisionObservation",
    "LeveragedEtfAnchorState",
    "LeveragedEtfExecutorSnapshot",
    "LeveragedEtfJournalEvent",
    "LeveragedEtfStrategyReservation",
}
LEGACY_FIXTURE = Path(__file__).with_name("fixtures") / "leveraged_etf_legacy_20230516.sql"
HASH_A = "a" * 64
HASH_B = "b" * 64
CREATED_AT = "2026-07-19T20:00:00.000000Z"
UPDATED_AT = "2026-07-19T20:00:01.000000Z"


COMPATIBLE_PARTIAL_SNAPSHOT_SQL = f"""
CREATE TABLE LeveragedEtfExecutorSnapshot (
    executor_id TEXT NOT NULL PRIMARY KEY,
    controller_id TEXT NOT NULL,
    pair_id TEXT NOT NULL,
    nav_cycle_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    state TEXT NOT NULL,
    snapshot_json TEXT NOT NULL,
    snapshot_hash TEXT NOT NULL,
    last_journal_sequence BIGINT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    CONSTRAINT ck_lepf_snapshot_schema_version CHECK (schema_version = 1),
    CONSTRAINT ck_lepf_snapshot_sequence CHECK (last_journal_sequence >= 0),
    CONSTRAINT ck_lepf_snapshot_hash CHECK (
        length(snapshot_hash) = 64
        AND snapshot_hash = lower(snapshot_hash)
        AND snapshot_hash NOT GLOB '*[^0-9a-f]*'
    )
);

INSERT INTO LeveragedEtfExecutorSnapshot (
    executor_id,
    controller_id,
    pair_id,
    nav_cycle_id,
    schema_version,
    state,
    snapshot_json,
    snapshot_hash,
    last_journal_sequence,
    created_at_utc,
    updated_at_utc
) VALUES (
    'partial-executor',
    'partial-controller',
    'sndk_snxx',
    'XNYS-2026-07-17',
    1,
    'CREATED',
    '{{}}',
    '{HASH_A}',
    0,
    '{CREATED_AT}',
    '{UPDATED_AT}'
);
"""


INCOMPATIBLE_NULLABLE_SNAPSHOT_SQL = f"""
CREATE TABLE LeveragedEtfExecutorSnapshot (
    executor_id TEXT PRIMARY KEY,
    controller_id TEXT NOT NULL,
    pair_id TEXT NOT NULL,
    nav_cycle_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    state TEXT NOT NULL,
    snapshot_json TEXT,
    snapshot_hash TEXT NOT NULL,
    last_journal_sequence BIGINT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL
);

INSERT INTO LeveragedEtfExecutorSnapshot (
    executor_id,
    controller_id,
    pair_id,
    nav_cycle_id,
    schema_version,
    state,
    snapshot_json,
    snapshot_hash,
    last_journal_sequence,
    created_at_utc,
    updated_at_utc
) VALUES (
    'null-payload-executor',
    'legacy-controller',
    'sndk_snxx',
    'XNYS-2026-07-17',
    1,
    'CREATED',
    NULL,
    '{HASH_A}',
    0,
    '{CREATED_AT}',
    '{UPDATED_AT}'
);
"""


def _client_config() -> ClientConfigAdapter:
    return ClientConfigAdapter(ClientConfigMap())


def _materialize_legacy_database(tmp_path: Path, extra_sql: str = "") -> Path:
    db_path = tmp_path / "legacy.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.executescript(LEGACY_FIXTURE.read_text(encoding="utf-8"))
        if extra_sql:
            connection.executescript(extra_sql)
    return db_path


def _open_manager(db_path: Path) -> SQLConnectionManager:
    return SQLConnectionManager(
        _client_config(),
        SQLConnectionType.TRADE_FILLS,
        db_path=str(db_path),
    )


def _database_version(manager: SQLConnectionManager) -> str:
    with manager.engine.connect() as connection:
        return connection.execute(text("SELECT value FROM Metadata WHERE key = 'local_db_version'")).scalar_one()


def _table_names(manager: SQLConnectionManager) -> set[str]:
    return set(inspect(manager.engine).get_table_names())


def _insert_snapshot(connection, executor_id: str = "executor-1") -> None:
    connection.execute(
        text("""
            INSERT INTO LeveragedEtfExecutorSnapshot (
                executor_id,
                controller_id,
                pair_id,
                nav_cycle_id,
                schema_version,
                state,
                snapshot_json,
                snapshot_hash,
                last_journal_sequence,
                created_at_utc,
                updated_at_utc
            ) VALUES (
                :executor_id,
                'controller-1',
                'sndk_snxx',
                'XNYS-2026-07-17',
                1,
                'CREATED',
                '{}',
                :payload_hash,
                0,
                :created_at,
                :updated_at
            )
            """),
        {
            "executor_id": executor_id,
            "payload_hash": HASH_A,
            "created_at": CREATED_AT,
            "updated_at": UPDATED_AT,
        },
    )


def _insert_journal_event(connection, event_id: str, executor_id: str) -> None:
    connection.execute(
        text("""
            INSERT INTO LeveragedEtfJournalEvent (
                event_id,
                executor_id,
                sequence,
                event_type,
                intent_id,
                idempotency_key,
                connector_name,
                trading_pair,
                client_order_id,
                exchange_order_id,
                exchange_trade_id,
                payload_json,
                payload_hash,
                created_at_utc
            ) VALUES (
                :event_id,
                :executor_id,
                1,
                'STATE_TRANSITION',
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                NULL,
                '{}',
                :payload_hash,
                :created_at
            )
            """),
        {
            "event_id": event_id,
            "executor_id": executor_id,
            "payload_hash": HASH_A,
            "created_at": CREATED_AT,
        },
    )


def test_fresh_database_registers_versioned_tables_constraints_and_indexes(tmp_path: Path):
    manager = _open_manager(tmp_path / "fresh.sqlite")
    try:
        inspector = inspect(manager.engine)
        assert manager.LOCAL_DB_VERSION_VALUE == TARGET_VERSION
        assert _database_version(manager) == TARGET_VERSION
        assert TARGET_TABLES.issubset(_table_names(manager))

        snapshot_columns = {column["name"]: column for column in inspector.get_columns("LeveragedEtfExecutorSnapshot")}
        assert snapshot_columns["executor_id"]["nullable"] is False
        assert snapshot_columns["snapshot_json"]["nullable"] is False
        assert snapshot_columns["snapshot_hash"]["nullable"] is False
        assert inspector.get_pk_constraint("LeveragedEtfExecutorSnapshot")["constrained_columns"] == ["executor_id"]

        snapshot_checks = {
            constraint["name"] for constraint in inspector.get_check_constraints("LeveragedEtfExecutorSnapshot")
        }
        assert {
            "ck_lepf_snapshot_hash",
            "ck_lepf_snapshot_schema_version",
            "ck_lepf_snapshot_sequence",
        }.issubset(snapshot_checks)

        snapshot_indexes = {index["name"] for index in inspector.get_indexes("LeveragedEtfExecutorSnapshot")}
        assert {
            "lepf_snapshot_controller_state",
            "lepf_snapshot_pair_cycle",
            "lepf_snapshot_updated",
        }.issubset(snapshot_indexes)

        anchor_checks = {
            constraint["name"] for constraint in inspector.get_check_constraints("LeveragedEtfAnchorState")
        }
        assert "ck_lepf_anchor_official_close_state" in anchor_checks

        journal_indexes = {index["name"] for index in inspector.get_indexes("LeveragedEtfJournalEvent")}
        assert {
            "lepf_journal_executor_type_sequence",
            "lepf_journal_idempotency_sequence",
            "lepf_journal_intent_sequence",
            "lepf_journal_order_lookup",
            "lepf_journal_trade_dedup",
        }.issubset(journal_indexes)

        journal_foreign_keys = inspector.get_foreign_keys("LeveragedEtfJournalEvent")
        assert journal_foreign_keys[0]["referred_table"] == "LeveragedEtfExecutorSnapshot"
        reservation_foreign_keys = inspector.get_foreign_keys("LeveragedEtfStrategyReservation")
        assert reservation_foreign_keys[0]["referred_table"] == "LeveragedEtfExecutorSnapshot"
        observation_foreign_keys = inspector.get_foreign_keys("LeveragedEtfAnchorRevisionObservation")
        assert observation_foreign_keys[0]["referred_table"] == "LeveragedEtfAnchorState"
    finally:
        manager.engine.dispose()


def test_real_legacy_database_migrates_without_data_loss_and_reopens(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)

    manager = _open_manager(db_path)
    try:
        assert _database_version(manager) == TARGET_VERSION
        assert TARGET_TABLES.issubset(_table_names(manager))
        with manager.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT value FROM Metadata WHERE key = 'legacy_sentinel'")).scalar_one()
                == "preserve-me"
            )
            assert (
                connection.execute(
                    text("SELECT custom_info FROM Executors WHERE id = 'legacy-position-executor'")
                ).scalar_one()
                == '{"legacy":true}'
            )
    finally:
        manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        assert _database_version(reopened) == TARGET_VERSION
        assert TARGET_TABLES.issubset(_table_names(reopened))
        with reopened.engine.connect() as connection:
            assert connection.execute(text("PRAGMA integrity_check")).scalar_one() == "ok"
            assert connection.execute(text("SELECT count(*) FROM Executors")).scalar_one() == 1
    finally:
        reopened.engine.dispose()


def test_partial_schema_and_repeated_migration_are_idempotent(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path, COMPATIBLE_PARTIAL_SNAPSHOT_SQL)

    manager = _open_manager(db_path)
    try:
        assert TARGET_TABLES.issubset(_table_names(manager))
        with manager.engine.connect() as connection:
            assert (
                connection.execute(
                    text(
                        "SELECT snapshot_hash FROM LeveragedEtfExecutorSnapshot "
                        "WHERE executor_id = 'partial-executor'"
                    )
                ).scalar_one()
                == HASH_A
            )
    finally:
        manager.engine.dispose()

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE Metadata SET value = '20230516' WHERE key = 'local_db_version'")

    repeated = _open_manager(db_path)
    try:
        assert _database_version(repeated) == TARGET_VERSION
        assert TARGET_TABLES.issubset(_table_names(repeated))
        with repeated.engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM LeveragedEtfExecutorSnapshot")).scalar_one() == 1
            assert connection.execute(text("PRAGMA integrity_check")).scalar_one() == "ok"
    finally:
        repeated.engine.dispose()

    assert not Path(f"{db_path}.new").exists()


def test_constraints_reject_duplicate_stable_ids_invalid_foreign_keys_and_null_payloads(tmp_path: Path):
    manager = _open_manager(tmp_path / "constraints.sqlite")
    try:
        with manager.engine.begin() as connection:
            _insert_snapshot(connection)

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                _insert_snapshot(connection)

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                _insert_journal_event(connection, "orphan-event", "missing-executor")

        with manager.engine.begin() as connection:
            _insert_journal_event(connection, "stable-event", "executor-1")

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                _insert_journal_event(connection, "stable-event", "executor-1")

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfExecutorSnapshot (
                            executor_id, controller_id, pair_id, nav_cycle_id,
                            schema_version, state, snapshot_json, snapshot_hash,
                            last_journal_sequence, created_at_utc, updated_at_utc
                        ) VALUES (
                            'null-payload', 'controller-1', 'sndk_snxx', 'XNYS-2026-07-17',
                            1, 'CREATED', NULL, :payload_hash, 0, :created_at, :updated_at
                        )
                        """),
                    {"payload_hash": HASH_A, "created_at": CREATED_AT, "updated_at": UPDATED_AT},
                )

        with manager.engine.begin() as connection:
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfStrategyReservation (
                        reservation_id, executor_id, reservation_key, connector_name,
                        trading_pair, leg, quantity, leverage, notional_cap,
                        payload_json, payload_hash, created_at_utc, updated_at_utc, released_at_utc
                    ) VALUES (
                        'reservation-1', 'executor-1', 'stable-reservation-key', 'binance_perpetual',
                        'SNXX-USDT', 'ETF', '1', 20, '1000', '{}', :payload_hash,
                        :created_at, :updated_at, NULL
                    )
                    """),
                {"payload_hash": HASH_A, "created_at": CREATED_AT, "updated_at": UPDATED_AT},
            )

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfStrategyReservation (
                            reservation_id, executor_id, reservation_key, connector_name,
                            trading_pair, leg, quantity, leverage, notional_cap,
                            payload_json, payload_hash, created_at_utc, updated_at_utc, released_at_utc
                        ) VALUES (
                            'reservation-2', 'executor-1', 'stable-reservation-key', 'binance_perpetual',
                            'SNXX-USDT', 'ETF', '1', 20, '1000', '{}', :payload_hash,
                            :created_at, :updated_at, NULL
                        )
                        """),
                    {"payload_hash": HASH_A, "created_at": CREATED_AT, "updated_at": UPDATED_AT},
                )

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(text("DELETE FROM LeveragedEtfExecutorSnapshot " "WHERE executor_id = 'executor-1'"))

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE LeveragedEtfStrategyReservation "
                        "SET reservation_id = 'rewritten-reservation' "
                        "WHERE reservation_id = 'reservation-1'"
                    )
                )

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfAnchorRevisionObservation (
                            cycle_id, evidence_hash, observed_at_utc
                        ) VALUES ('missing-cycle', :evidence_hash, :observed_at)
                        """),
                    {"evidence_hash": HASH_B, "observed_at": UPDATED_AT},
                )

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfAnchorState (
                            cycle_id, schema_version, state_kind, revision, target_session_date,
                            official_close_utc, deadline_utc, evidence_hash, payload_json,
                            payload_hash, created_at_utc, updated_at_utc
                        ) VALUES (
                            'checkpoint-without-close', 1, 'CHECKPOINT', 1, '2026-07-17',
                            NULL, '2026-07-17T20:10:00.000000Z', NULL, '{}',
                            :payload_hash, :created_at, :updated_at
                        )
                        """),
                    {"payload_hash": HASH_A, "created_at": CREATED_AT, "updated_at": UPDATED_AT},
                )
    finally:
        manager.engine.dispose()


def test_append_only_guards_survive_database_reopen(tmp_path: Path):
    db_path = tmp_path / "append-only.sqlite"
    manager = _open_manager(db_path)
    try:
        with manager.engine.begin() as connection:
            _insert_snapshot(connection)
            _insert_journal_event(connection, "event-1", "executor-1")
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfAnchorState (
                        cycle_id, schema_version, state_kind, revision, target_session_date,
                        official_close_utc, deadline_utc, evidence_hash, payload_json,
                        payload_hash, created_at_utc, updated_at_utc
                    ) VALUES (
                        'XNYS-2026-07-17', 1, 'FINALIZED', 2, '2026-07-17',
                        NULL, '2026-07-17T20:10:00.000000Z', :evidence_hash, '{}',
                        :payload_hash, :created_at, :updated_at
                    )
                    """),
                {
                    "evidence_hash": HASH_B,
                    "payload_hash": HASH_A,
                    "created_at": CREATED_AT,
                    "updated_at": UPDATED_AT,
                },
            )
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfAnchorRevisionObservation (
                        cycle_id, evidence_hash, observed_at_utc
                    ) VALUES ('XNYS-2026-07-17', :evidence_hash, :observed_at)
                    """),
                {"evidence_hash": HASH_A, "observed_at": UPDATED_AT},
            )
    finally:
        manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        with pytest.raises(IntegrityError):
            with reopened.engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE LeveragedEtfJournalEvent SET payload_json = :payload_json " "WHERE event_id = 'event-1'"
                    ),
                    {"payload_json": '{"changed":true}'},
                )
        with pytest.raises(IntegrityError):
            with reopened.engine.begin() as connection:
                connection.execute(
                    text("DELETE FROM LeveragedEtfAnchorRevisionObservation " "WHERE cycle_id = 'XNYS-2026-07-17'")
                )
        with pytest.raises(IntegrityError):
            with reopened.engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE LeveragedEtfAnchorState SET evidence_hash = :evidence_hash "
                        "WHERE cycle_id = 'XNYS-2026-07-17'"
                    ),
                    {"evidence_hash": "c" * 64},
                )
    finally:
        reopened.engine.dispose()


def test_incompatible_null_legacy_payload_rolls_back_and_can_be_repaired(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path, INCOMPATIBLE_NULLABLE_SNAPSHOT_SQL)

    with pytest.raises(RuntimeError, match="incompatible leveraged ETF persistence schema"):
        _open_manager(db_path)

    assert not Path(f"{db_path}.new").exists()

    with sqlite3.connect(db_path) as connection:
        version = connection.execute("SELECT value FROM Metadata WHERE key = 'local_db_version'").fetchone()[0]
        assert version == "20230516"
        assert (
            connection.execute("SELECT value FROM Metadata WHERE key = 'legacy_sentinel'").fetchone()[0]
            == "preserve-me"
        )
        assert (
            connection.execute(
                "SELECT snapshot_json FROM LeveragedEtfExecutorSnapshot " "WHERE executor_id = 'null-payload-executor'"
            ).fetchone()[0]
            is None
        )
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        assert "LeveragedEtfJournalEvent" not in tables
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"

        connection.execute("DROP TABLE LeveragedEtfExecutorSnapshot")

    repaired = _open_manager(db_path)
    try:
        assert _database_version(repaired) == TARGET_VERSION
        assert TARGET_TABLES.issubset(_table_names(repaired))
        with repaired.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT value FROM Metadata WHERE key = 'legacy_sentinel'")).scalar_one()
                == "preserve-me"
            )
    finally:
        repaired.engine.dispose()


def test_markets_recorder_initialization_uses_the_registered_persistence_schema(tmp_path: Path):
    manager = _open_manager(tmp_path / "recorder.sqlite")
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        recorder = MarketsRecorder(
            sql=manager,
            markets=[],
            config_file_path="migration-test",
            strategy_name="equity_leveraged_etf_arbitrage",
            market_data_collection=MarketDataCollectionConfigMap(
                market_data_collection_enabled=False,
                market_data_collection_interval=60,
                market_data_collection_depth=20,
            ),
        )

        assert recorder.sql_manager is manager
        assert TARGET_TABLES.issubset(_table_names(recorder.sql_manager))
    finally:
        MarketsRecorder._shared_instance = None
        asyncio.set_event_loop(None)
        loop.close()
        manager.engine.dispose()
