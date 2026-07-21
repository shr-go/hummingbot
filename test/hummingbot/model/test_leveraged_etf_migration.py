import asyncio
import hashlib
import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import inspect, text
from sqlalchemy.exc import IntegrityError

from hummingbot.client.config.client_config_map import ClientConfigMap, MarketDataCollectionConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.connector.markets_recorder import MarketsRecorder
from hummingbot.model.sql_connection_manager import DatabaseMigrationError, SQLConnectionManager, SQLConnectionType

TARGET_VERSION = "20260721"
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
        typeof(snapshot_hash) = 'text'
        AND instr(snapshot_hash, char(0)) = 0
        AND length(snapshot_hash) = 64
        AND snapshot_hash = lower(snapshot_hash)
        AND snapshot_hash NOT GLOB '*[^0123456789abcdef]*'
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


INTERMEDIATE_20260719_ANCHOR_SQL = """
CREATE TABLE LeveragedEtfAnchorState (
    cycle_id TEXT NOT NULL,
    schema_version INTEGER NOT NULL,
    state_kind TEXT NOT NULL,
    revision INTEGER NOT NULL,
    target_session_date TEXT NOT NULL,
    official_close_utc TEXT,
    deadline_utc TEXT NOT NULL,
    evidence_hash TEXT,
    payload_json TEXT NOT NULL,
    payload_hash TEXT NOT NULL,
    created_at_utc TEXT NOT NULL,
    updated_at_utc TEXT NOT NULL,
    PRIMARY KEY (cycle_id),
    CONSTRAINT ck_lepf_anchor_schema_version CHECK (schema_version = 1),
    CONSTRAINT ck_lepf_anchor_state_kind CHECK (state_kind IN ('CHECKPOINT', 'FINALIZED')),
    CONSTRAINT ck_lepf_anchor_revision CHECK (revision >= 1),
    CONSTRAINT ck_lepf_anchor_evidence_state CHECK (
        (state_kind = 'CHECKPOINT' AND evidence_hash IS NULL)
        OR (state_kind = 'FINALIZED' AND evidence_hash IS NOT NULL)
    ),
    CONSTRAINT ck_lepf_anchor_official_close_state CHECK (
        state_kind = 'FINALIZED' OR official_close_utc IS NOT NULL
    ),
    CONSTRAINT ck_lepf_anchor_evidence_hash CHECK (
        evidence_hash IS NULL OR (
            typeof(evidence_hash) = 'text'
            AND instr(evidence_hash, char(0)) = 0
            AND length(evidence_hash) = 64
            AND evidence_hash = lower(evidence_hash)
            AND evidence_hash NOT GLOB '*[^0123456789abcdef]*'
        )
    ),
    CONSTRAINT ck_lepf_anchor_payload_hash CHECK (
        typeof(payload_hash) = 'text'
        AND instr(payload_hash, char(0)) = 0
        AND length(payload_hash) = 64
        AND payload_hash = lower(payload_hash)
        AND payload_hash NOT GLOB '*[^0123456789abcdef]*'
    ),
    CONSTRAINT uq_lepf_anchor_evidence_hash UNIQUE (evidence_hash)
);

CREATE INDEX lepf_anchor_deadline_state
    ON LeveragedEtfAnchorState (deadline_utc, state_kind);
CREATE INDEX lepf_anchor_session_date
    ON LeveragedEtfAnchorState (target_session_date);

CREATE TABLE LeveragedEtfAnchorRevisionObservation (
    cycle_id TEXT NOT NULL,
    evidence_hash TEXT NOT NULL,
    observed_at_utc TEXT NOT NULL,
    PRIMARY KEY (cycle_id, evidence_hash, observed_at_utc),
    CONSTRAINT ck_lepf_anchor_observation_hash CHECK (
        typeof(evidence_hash) = 'text'
        AND instr(evidence_hash, char(0)) = 0
        AND length(evidence_hash) = 64
        AND evidence_hash = lower(evidence_hash)
        AND evidence_hash NOT GLOB '*[^0123456789abcdef]*'
    ),
    CONSTRAINT fk_lepf_anchor_observation_cycle FOREIGN KEY (cycle_id)
        REFERENCES LeveragedEtfAnchorState (cycle_id) ON DELETE RESTRICT
);

CREATE INDEX lepf_anchor_observation_cycle_time
    ON LeveragedEtfAnchorRevisionObservation (cycle_id, observed_at_utc);

CREATE TRIGGER lepf_anchor_observation_cycle_fk_insert
BEFORE INSERT ON LeveragedEtfAnchorRevisionObservation
FOR EACH ROW
WHEN NOT EXISTS (
    SELECT 1 FROM LeveragedEtfAnchorState
    WHERE cycle_id = NEW.cycle_id
)
BEGIN
    SELECT RAISE(ABORT, 'LeveragedEtfAnchorRevisionObservation cycle_id does not exist');
END;

CREATE TRIGGER lepf_anchor_observation_identity_insert
BEFORE INSERT ON LeveragedEtfAnchorRevisionObservation
FOR EACH ROW
WHEN EXISTS (
    SELECT 1 FROM LeveragedEtfAnchorRevisionObservation
    WHERE cycle_id = NEW.cycle_id
      AND evidence_hash = NEW.evidence_hash
      AND observed_at_utc = NEW.observed_at_utc
)
BEGIN
    SELECT RAISE(ABORT, 'LeveragedEtfAnchorRevisionObservation identity already exists');
END;

CREATE TRIGGER lepf_anchor_observation_no_update
BEFORE UPDATE ON LeveragedEtfAnchorRevisionObservation
BEGIN
    SELECT RAISE(ABORT, 'LeveragedEtfAnchorRevisionObservation is append-only');
END;

CREATE TRIGGER lepf_anchor_observation_no_delete
BEFORE DELETE ON LeveragedEtfAnchorRevisionObservation
BEGIN
    SELECT RAISE(ABORT, 'LeveragedEtfAnchorRevisionObservation is append-only');
END;

CREATE TRIGGER lepf_anchor_finalized_no_update
BEFORE UPDATE ON LeveragedEtfAnchorState
FOR EACH ROW
WHEN OLD.state_kind = 'FINALIZED'
BEGIN
    SELECT RAISE(ABORT, 'finalized LeveragedEtfAnchorState is immutable');
END;

CREATE TRIGGER lepf_anchor_identity_insert
BEFORE INSERT ON LeveragedEtfAnchorState
FOR EACH ROW
WHEN EXISTS (
    SELECT 1 FROM LeveragedEtfAnchorState
    WHERE cycle_id = NEW.cycle_id
       OR (
            NEW.evidence_hash IS NOT NULL
            AND evidence_hash = NEW.evidence_hash
       )
)
BEGIN
    SELECT RAISE(ABORT, 'LeveragedEtfAnchorState identity already exists');
END;

CREATE TRIGGER lepf_anchor_cycle_id_no_update
BEFORE UPDATE OF cycle_id ON LeveragedEtfAnchorState
BEGIN
    SELECT RAISE(ABORT, 'LeveragedEtfAnchorState cycle_id is stable');
END;

CREATE TRIGGER lepf_anchor_state_no_delete
BEFORE DELETE ON LeveragedEtfAnchorState
BEGIN
    SELECT RAISE(ABORT, 'LeveragedEtfAnchorState is durable');
END;
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


def _materialize_intermediate_20260719_database(tmp_path: Path, *, populated: bool) -> Path:
    db_path = _materialize_legacy_database(tmp_path)
    manager = _open_manager(db_path)
    manager.engine.dispose()

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE LeveragedEtfAnchorRevisionObservation")
        connection.execute("DROP TABLE LeveragedEtfAnchorState")
        connection.executescript(INTERMEDIATE_20260719_ANCHOR_SQL)
        connection.execute("UPDATE Metadata SET value = '20260719' WHERE key = 'local_db_version'")
        if populated:
            connection.execute(
                """
                INSERT INTO LeveragedEtfAnchorState (
                    cycle_id, schema_version, state_kind, revision,
                    target_session_date, official_close_utc, deadline_utc,
                    evidence_hash, payload_json, payload_hash,
                    created_at_utc, updated_at_utc
                ) VALUES (?, 1, 'FINALIZED', 3, '2026-07-17', NULL, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "xnys-2026-07-17",
                    "2026-07-17T20:10:00.000000Z",
                    HASH_B,
                    '{"legacy":"ambiguous"}',
                    HASH_A,
                    CREATED_AT,
                    UPDATED_AT,
                ),
            )
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


def _assert_only_canonical_anchor_schema(manager: SQLConnectionManager) -> None:
    inspector = inspect(manager.engine)
    anchor_tables = {
        table_name for table_name in inspector.get_table_names() if table_name.startswith("LeveragedEtfAnchor")
    }
    assert anchor_tables == {
        "LeveragedEtfAnchorRevisionObservation",
        "LeveragedEtfAnchorState",
    }
    assert [column["name"] for column in inspector.get_columns("LeveragedEtfAnchorState")] == [
        "pair_id",
        "cycle_id",
        "schema_version",
        "state_kind",
        "revision",
        "target_session_date",
        "official_close_utc",
        "deadline_utc",
        "evidence_hash",
        "payload_version_field",
        "payload_contract_version",
        "payload_json",
        "payload_hash",
        "created_at_utc",
        "updated_at_utc",
    ]
    assert inspector.get_pk_constraint("LeveragedEtfAnchorState")["constrained_columns"] == [
        "pair_id",
        "cycle_id",
    ]


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
        _assert_only_canonical_anchor_schema(manager)

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
        assert {
            "ck_lepf_anchor_official_close_state",
            "ck_lepf_anchor_pair_id",
            "ck_lepf_anchor_cycle_id",
            "ck_lepf_anchor_payload_contract_version",
        }.issubset(anchor_checks)
        assert inspector.get_pk_constraint("LeveragedEtfAnchorState")["constrained_columns"] == [
            "pair_id",
            "cycle_id",
        ]
        anchor_indexes = {index["name"] for index in inspector.get_indexes("LeveragedEtfAnchorState")}
        assert {
            "lepf_anchor_pair_cycle",
            "lepf_anchor_pair_deadline_state",
            "lepf_anchor_pair_session",
        }.issubset(anchor_indexes)

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
        assert observation_foreign_keys[0]["constrained_columns"] == ["pair_id", "cycle_id"]
        assert observation_foreign_keys[0]["referred_columns"] == ["pair_id", "cycle_id"]
    finally:
        manager.engine.dispose()


def test_real_legacy_database_migrates_without_data_loss_and_reopens(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)

    manager = _open_manager(db_path)
    try:
        assert _database_version(manager) == TARGET_VERSION
        assert TARGET_TABLES.issubset(_table_names(manager))
        _assert_only_canonical_anchor_schema(manager)
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


def test_empty_20260719_intermediate_anchor_schema_rebuilds_atomically_and_idempotently(
    tmp_path: Path,
):
    db_path = _materialize_intermediate_20260719_database(tmp_path, populated=False)

    manager = _open_manager(db_path)
    try:
        assert _database_version(manager) == TARGET_VERSION
        _assert_only_canonical_anchor_schema(manager)
        with manager.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT value FROM Metadata WHERE key = 'legacy_sentinel'")).scalar_one()
                == "preserve-me"
            )
            assert connection.execute(text("SELECT count(*) FROM Executors")).scalar_one() == 1
            assert connection.execute(text("SELECT count(*) FROM LeveragedEtfAnchorState")).scalar_one() == 0
            assert connection.execute(text("PRAGMA integrity_check")).scalar_one() == "ok"
    finally:
        manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        assert _database_version(reopened) == TARGET_VERSION
        _assert_only_canonical_anchor_schema(reopened)
        with reopened.engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM Executors")).scalar_one() == 1
            assert connection.execute(text("PRAGMA integrity_check")).scalar_one() == "ok"
    finally:
        reopened.engine.dispose()


def test_nonempty_20260719_intermediate_anchor_schema_fails_without_mutating_source(
    tmp_path: Path,
):
    db_path = _materialize_intermediate_20260719_database(tmp_path, populated=True)
    original_digest = hashlib.sha256(db_path.read_bytes()).hexdigest()

    with pytest.raises(
        DatabaseMigrationError,
        match="non-empty 20260719 cycle-only anchor schema is ambiguous",
    ):
        _open_manager(db_path)

    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == original_digest
    assert not tuple(tmp_path.glob(f".{db_path.name}.migration-*.sqlite"))
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT value FROM Metadata WHERE key = 'local_db_version'").fetchone() == (
            "20260719",
        )
        assert connection.execute(
            "SELECT cycle_id, revision, evidence_hash, payload_json " "FROM LeveragedEtfAnchorState"
        ).fetchall() == [("xnys-2026-07-17", 3, HASH_B, '{"legacy":"ambiguous"}')]
        assert connection.execute("SELECT value FROM Metadata WHERE key = 'legacy_sentinel'").fetchone() == (
            "preserve-me",
        )


def test_drifted_20260719_intermediate_shape_fails_without_mutating_source(tmp_path: Path):
    db_path = _materialize_intermediate_20260719_database(tmp_path, populated=False)
    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP INDEX lepf_anchor_session_date")
    original_digest = hashlib.sha256(db_path.read_bytes()).hexdigest()

    with pytest.raises(
        DatabaseMigrationError,
        match="20260719 intermediate anchor schema shape is incompatible",
    ):
        _open_manager(db_path)

    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == original_digest
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT value FROM Metadata WHERE key = 'local_db_version'").fetchone() == (
            "20260719",
        )
        assert connection.execute("SELECT count(*) FROM LeveragedEtfAnchorState").fetchone() == (0,)


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
                            pair_id, cycle_id, evidence_hash, observed_at_utc,
                            schema_version, payload_version_field,
                            payload_contract_version, payload_json, payload_hash
                        ) VALUES (
                            'sndk_snxx', 'xnys-2026-07-18', :evidence_hash, :observed_at,
                            2, 'schema_version', 2, '{}', :payload_hash
                        )
                        """),
                    {
                        "evidence_hash": HASH_B,
                        "observed_at": UPDATED_AT,
                        "payload_hash": HASH_A,
                    },
                )

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfAnchorState (
                            pair_id, cycle_id, schema_version, state_kind, revision,
                            target_session_date, official_close_utc, deadline_utc,
                            evidence_hash, payload_version_field,
                            payload_contract_version, payload_json, payload_hash,
                            created_at_utc, updated_at_utc
                        ) VALUES (
                            'sndk_snxx', 'xnys-2026-07-17', 2, 'CHECKPOINT', 1,
                            '2026-07-17', NULL, '2026-07-17T20:10:00.000000Z',
                            NULL, 'integrity_version', 3, '{}',
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
                        pair_id, cycle_id, schema_version, state_kind, revision,
                        target_session_date, official_close_utc, deadline_utc,
                        evidence_hash, payload_version_field,
                        payload_contract_version, payload_json, payload_hash,
                        created_at_utc, updated_at_utc
                    ) VALUES (
                        'sndk_snxx', 'xnys-2026-07-17', 2, 'FINALIZED', 2,
                        '2026-07-17', '2026-07-17T20:00:00.000000Z',
                        '2026-07-17T20:10:00.000000Z', :evidence_hash,
                        'evidence_version', 3, '{}',
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
                        pair_id, cycle_id, evidence_hash, observed_at_utc,
                        schema_version, payload_version_field,
                        payload_contract_version, payload_json, payload_hash
                    ) VALUES (
                        'sndk_snxx', 'xnys-2026-07-17', :evidence_hash, :observed_at,
                        2, 'schema_version', 2, '{}', :payload_hash
                    )
                    """),
                {
                    "evidence_hash": HASH_A,
                    "observed_at": UPDATED_AT,
                    "payload_hash": HASH_B,
                },
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
                    text("DELETE FROM LeveragedEtfAnchorRevisionObservation " "WHERE cycle_id = 'xnys-2026-07-17'")
                )
        with pytest.raises(IntegrityError):
            with reopened.engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE LeveragedEtfAnchorState SET evidence_hash = :evidence_hash "
                        "WHERE pair_id = 'sndk_snxx' AND cycle_id = 'xnys-2026-07-17'"
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
