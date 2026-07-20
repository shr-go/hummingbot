import fcntl
import multiprocessing
import os
import sqlite3
import stat
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import sqlite
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateTable

import hummingbot.model.db_migration.migrator as migrator_module
from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.model.db_migration.migrator import Migrator
from hummingbot.model.leveraged_etf_persistence import LeveragedEtfExecutorSnapshot, LeveragedEtfJournalEvent
from hummingbot.model.sql_connection_manager import DatabaseMigrationError, SQLConnectionManager, SQLConnectionType

TARGET_VERSION = "20260719"
LEGACY_VERSION = "20230516"
LEGACY_FIXTURE = Path(__file__).with_name("fixtures") / "leveraged_etf_legacy_20230516.sql"
HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
BAD_HASH = "g" * 64
CREATED_AT = "2026-07-19T20:00:00.000000Z"
UPDATED_AT = "2026-07-19T20:00:01.000000Z"


def _client_config() -> ClientConfigAdapter:
    return ClientConfigAdapter(ClientConfigMap())


def _open_manager(db_path: Path) -> SQLConnectionManager:
    return SQLConnectionManager(
        _client_config(),
        SQLConnectionType.TRADE_FILLS,
        db_path=str(db_path),
    )


def _open_unmanaged_manager(db_path: Path) -> SQLConnectionManager:
    return SQLConnectionManager(
        _client_config(),
        SQLConnectionType.TRADE_FILLS,
        db_path=str(db_path),
        called_from_migrator=True,
    )


def _materialize_legacy_database(tmp_path: Path, name: str = "legacy.sqlite") -> Path:
    db_path = tmp_path / name
    with sqlite3.connect(db_path) as connection:
        connection.executescript(LEGACY_FIXTURE.read_text(encoding="utf-8"))
    return db_path


def _version(db_path: Path) -> str | None:
    with sqlite3.connect(db_path) as connection:
        metadata_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'Metadata'"
        ).fetchone()
        if metadata_exists is None:
            return None
        row = connection.execute("SELECT value FROM Metadata WHERE key = 'local_db_version'").fetchone()
        return None if row is None else row[0]


def _table_names(db_path: Path) -> set[str]:
    with sqlite3.connect(db_path) as connection:
        return {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
            )
        }


def _assert_legacy_data_unchanged(db_path: Path) -> None:
    assert _version(db_path) == LEGACY_VERSION
    with sqlite3.connect(db_path) as connection:
        assert (
            connection.execute("SELECT value FROM Metadata WHERE key = 'legacy_sentinel'").fetchone()[0]
            == "preserve-me"
        )
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def _compiled_table_sql(table) -> str:
    return str(CreateTable(table).compile(dialect=sqlite.dialect()))


def _create_compiled_table(connection: sqlite3.Connection, table, transform=lambda value: value) -> None:
    connection.execute(transform(_compiled_table_sql(table)))


def _insert_compiled_column(create_sql: str, column_sql: str) -> str:
    marker = "\tPRIMARY KEY"
    assert marker in create_sql
    return create_sql.replace(marker, f"\t{column_sql}, \n{marker}", 1)


def _append_compiled_constraint(create_sql: str, constraint_sql: str) -> str:
    closing_index = create_sql.rfind("\n)")
    assert closing_index > 0
    return f"{create_sql[:closing_index]}, \n\t{constraint_sql}{create_sql[closing_index:]}"


def _insert_snapshot(connection, executor_id: str = "executor-1", snapshot_json: str = "original") -> None:
    connection.execute(
        text("""
            INSERT INTO LeveragedEtfExecutorSnapshot (
                executor_id, controller_id, pair_id, nav_cycle_id, schema_version,
                state, snapshot_json, snapshot_hash, last_journal_sequence,
                created_at_utc, updated_at_utc
            ) VALUES (
                :executor_id, 'controller-1', 'sndk_snxx', 'xnys-2026-07-17', 1,
                'CREATED', :snapshot_json, :snapshot_hash, 0, :created_at, :updated_at
            )
            """),
        {
            "executor_id": executor_id,
            "snapshot_json": snapshot_json,
            "snapshot_hash": HASH_A,
            "created_at": CREATED_AT,
            "updated_at": UPDATED_AT,
        },
    )


def _seed_protected_rows(manager: SQLConnectionManager) -> None:
    with manager.engine.begin() as connection:
        _insert_snapshot(connection)
        connection.execute(
            text("""
                INSERT INTO LeveragedEtfJournalEvent (
                    event_id, executor_id, sequence, event_type, intent_id,
                    idempotency_key, connector_name, trading_pair, client_order_id,
                    exchange_order_id, exchange_trade_id, payload_json, payload_hash,
                    created_at_utc
                ) VALUES (
                    'event-1', 'executor-1', 1, 'FILL', NULL, NULL,
                    'binance_perpetual', 'SNXX-USDT', NULL, NULL, 'trade-1',
                    'original', :payload_hash, :created_at
                )
                """),
            {"payload_hash": HASH_A, "created_at": CREATED_AT},
        )
        connection.execute(
            text("""
                INSERT INTO LeveragedEtfStrategyReservation (
                    reservation_id, executor_id, reservation_key, connector_name,
                    trading_pair, leg, quantity, leverage, notional_cap, payload_json,
                    payload_hash, created_at_utc, updated_at_utc, released_at_utc
                ) VALUES (
                    'reservation-1', 'executor-1', 'reservation-key-1',
                    'binance_perpetual', 'SNXX-USDT', 'ETF', '1', 20, '1000',
                    'original', :payload_hash, :created_at, :updated_at, NULL
                )
                """),
            {
                "payload_hash": HASH_A,
                "created_at": CREATED_AT,
                "updated_at": UPDATED_AT,
            },
        )
        connection.execute(
            text("""
                INSERT INTO LeveragedEtfAnchorState (
                    cycle_id, schema_version, state_kind, revision, target_session_date,
                    official_close_utc, deadline_utc, evidence_hash, payload_json,
                    payload_hash, created_at_utc, updated_at_utc
                ) VALUES (
                    'xnys-2026-07-17', 1, 'FINALIZED', 2, '2026-07-17', NULL,
                    '2026-07-17T20:10:00.000000Z', :evidence_hash,
                    'original', :payload_hash, :created_at, :updated_at
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
                ) VALUES ('xnys-2026-07-17', :evidence_hash, :observed_at)
                """),
            {"evidence_hash": HASH_C, "observed_at": UPDATED_AT},
        )


def _hold_migration_lock(lock_path: str, ready, release) -> None:
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        ready.set()
        release.wait(timeout=20)
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _commit_wal_row_then_crash(db_path: str) -> None:
    connection = sqlite3.connect(db_path)
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA wal_autocheckpoint = 0")
    connection.execute("INSERT INTO Metadata (key, value) VALUES ('wal_committed', 'must-survive')")
    connection.commit()
    os._exit(0)


def test_non_sqlite_mode_fails_before_engine_or_file_mutation(tmp_path: Path):
    db_path = tmp_path / "must-not-change.sqlite"
    original_bytes = b"not-a-database-and-must-remain-byte-exact"
    db_path.write_bytes(original_bytes)
    config = SimpleNamespace(
        db_mode=SimpleNamespace(get_url=lambda _: "postgresql+psycopg2://user:password@localhost/hummingbot")
    )

    with patch(
        "hummingbot.model.sql_connection_manager.create_engine",
        side_effect=AssertionError("unsupported dialect reached create_engine"),
    ) as create_engine:
        with pytest.raises(DatabaseMigrationError, match="SQLite"):
            SQLConnectionManager(
                config,
                SQLConnectionType.TRADE_FILLS,
                db_path=str(db_path),
            )

    create_engine.assert_not_called()
    assert db_path.read_bytes() == original_bytes


@pytest.mark.parametrize("layout", ["missing_metadata_table", "missing_version_row"])
def test_nonempty_unversioned_database_is_rejected_without_mutation(tmp_path: Path, layout: str):
    db_path = tmp_path / f"{layout}.sqlite"
    with sqlite3.connect(db_path) as connection:
        if layout == "missing_version_row":
            connection.execute("CREATE TABLE Metadata (key TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL)")
            connection.execute("INSERT INTO Metadata (key, value) VALUES ('legacy_sentinel', 'preserve-me')")
        connection.execute("CREATE TABLE ExistingData (id INTEGER PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO ExistingData (id, value) VALUES (1, 'preserve-me')")

    with pytest.raises(DatabaseMigrationError, match="unversioned"):
        _open_manager(db_path)

    assert _table_names(db_path) == (
        {"ExistingData"} if layout == "missing_metadata_table" else {"ExistingData", "Metadata"}
    )
    assert _version(db_path) is None
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT value FROM ExistingData WHERE id = 1").fetchone()[0] == "preserve-me"


@pytest.mark.parametrize("version", ["20240101", "20260720"])
def test_unknown_or_future_database_version_is_rejected_before_mutation(tmp_path: Path, version: str):
    db_path = _materialize_legacy_database(tmp_path, f"version-{version}.sqlite")
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE Metadata SET value = ? WHERE key = 'local_db_version'",
            (version,),
        )

    with pytest.raises(DatabaseMigrationError, match="version"):
        _open_manager(db_path)

    assert _version(db_path) == version
    assert "LeveragedEtfExecutorSnapshot" not in _table_names(db_path)
    with sqlite3.connect(db_path) as connection:
        assert (
            connection.execute("SELECT value FROM Metadata WHERE key = 'legacy_sentinel'").fetchone()[0]
            == "preserve-me"
        )


def test_missing_version_row_with_incompatible_partial_schema_is_not_stamped(tmp_path: Path):
    db_path = tmp_path / "missing-version-incompatible.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE Metadata (key TEXT NOT NULL PRIMARY KEY, value TEXT NOT NULL)")
        connection.execute("INSERT INTO Metadata (key, value) VALUES ('legacy_sentinel', 'preserve-me')")
        connection.execute(
            "CREATE TABLE LeveragedEtfExecutorSnapshot (" "executor_id TEXT PRIMARY KEY, snapshot_json TEXT NULL)"
        )

    with pytest.raises(DatabaseMigrationError, match="unversioned"):
        _open_manager(db_path)

    assert _version(db_path) is None
    assert _table_names(db_path) == {"Metadata", "LeveragedEtfExecutorSnapshot"}


def test_cross_process_migration_lock_rejects_concurrent_attempt_without_mutation(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    lock_path = db_path.parent / f".{db_path.name}.migration.lock"
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    holder = context.Process(
        target=_hold_migration_lock,
        args=(str(lock_path), ready, release),
    )
    holder.start()
    assert ready.wait(timeout=10)
    try:
        with pytest.raises(DatabaseMigrationError, match="lock"):
            unexpected = _open_manager(db_path)
            unexpected.engine.dispose()
    finally:
        release.set()
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join(timeout=5)

    assert holder.exitcode == 0
    _assert_legacy_data_unchanged(db_path)

    manager = _open_manager(db_path)
    manager.engine.dispose()
    assert _version(db_path) == TARGET_VERSION


def test_active_sqlite_writer_causes_fail_closed_migration_and_preserves_original(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    blocker = sqlite3.connect(db_path)
    blocker.execute("BEGIN IMMEDIATE")
    blocker.execute("INSERT INTO Metadata (key, value) VALUES ('uncommitted_writer', 'must-not-appear')")
    try:
        with pytest.raises(DatabaseMigrationError, match="active|lock|busy"):
            unexpected = _open_manager(db_path)
            unexpected.engine.dispose()
    finally:
        blocker.rollback()
        blocker.close()

    _assert_legacy_data_unchanged(db_path)
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT count(*) FROM Metadata WHERE key = 'uncommitted_writer'").fetchone()[0] == 0


def test_committed_wal_data_survives_migration_and_journal_mode_is_preserved(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    context = multiprocessing.get_context("fork")
    writer = context.Process(target=_commit_wal_row_then_crash, args=(str(db_path),))
    writer.start()
    writer.join(timeout=10)
    assert writer.exitcode == 0
    assert Path(f"{db_path}-wal").exists()

    manager = _open_manager(db_path)
    try:
        with manager.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT value FROM Metadata WHERE key = 'wal_committed'")).scalar_one()
                == "must-survive"
            )
            assert connection.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
    finally:
        manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        with reopened.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT value FROM Metadata WHERE key = 'wal_committed'")).scalar_one()
                == "must-survive"
            )
    finally:
        reopened.engine.dispose()


@pytest.mark.parametrize(
    "failure_hook",
    ["_sqlite_backup_database", "_copy_database_file", "_atomic_replace"],
)
def test_injected_migration_preparation_failure_cleans_up_reconnects_and_preserves_original(
    tmp_path: Path,
    failure_hook: str,
):
    db_path = _materialize_legacy_database(tmp_path, f"failure-{failure_hook}.sqlite")
    manager = _open_unmanaged_manager(db_path)
    try:
        with patch.object(
            migrator_module,
            failure_hook,
            side_effect=OSError(f"injected {failure_hook} failure"),
            create=True,
        ):
            assert (
                Migrator().migrate_db_to_version(
                    _client_config(),
                    manager,
                    int(LEGACY_VERSION),
                    int(TARGET_VERSION),
                )
                is False
            )

        _assert_legacy_data_unchanged(db_path)
        with manager.engine.connect() as connection:
            assert (
                connection.execute(text("SELECT value FROM Metadata WHERE key = 'legacy_sentinel'")).scalar_one()
                == "preserve-me"
            )
        assert not Path(f"{db_path}.new").exists()
        assert list(tmp_path.glob(f".{db_path.name}.migration-*.sqlite")) == []
    finally:
        manager.engine.dispose()


def test_unique_same_directory_temporary_paths_are_cleaned_after_replace_failures(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    manager = _open_unmanaged_manager(db_path)
    replacement_sources: list[Path] = []

    def reject_replace(source, destination):
        replacement_sources.append(Path(source))
        assert Path(source).parent == db_path.parent
        assert Path(destination) == db_path
        raise OSError("injected replace failure")

    try:
        with patch.object(
            migrator_module,
            "_atomic_replace",
            side_effect=reject_replace,
            create=True,
        ):
            for _ in range(2):
                assert (
                    Migrator().migrate_db_to_version(
                        _client_config(),
                        manager,
                        int(LEGACY_VERSION),
                        int(TARGET_VERSION),
                    )
                    is False
                )
                _assert_legacy_data_unchanged(db_path)
    finally:
        manager.engine.dispose()

    assert len(replacement_sources) == 2
    assert replacement_sources[0] != replacement_sources[1]
    assert all(path.name != f"{db_path.name}.new" for path in replacement_sources)
    assert all(not path.exists() for path in replacement_sources)


def test_migration_preserves_file_mode_and_durably_syncs_file_and_parent_directory(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    db_path.chmod(0o600)
    file_syncs: list[Path] = []
    directory_syncs: list[Path] = []

    with patch.object(
        migrator_module,
        "_fsync_file",
        side_effect=lambda path: file_syncs.append(Path(path)),
        create=True,
    ), patch.object(
        migrator_module,
        "_fsync_directory",
        side_effect=lambda path: directory_syncs.append(Path(path)),
        create=True,
    ):
        manager = _open_manager(db_path)
        manager.engine.dispose()

    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
    backups = list(tmp_path.glob(f"{db_path.name}.backup_*"))
    assert len(backups) == 1
    assert stat.S_IMODE(backups[0].stat().st_mode) == 0o600
    assert db_path in file_syncs
    assert db_path.parent in directory_syncs


def test_cascade_foreign_key_is_rejected_before_version_stamp(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
        journal_sql = _compiled_table_sql(LeveragedEtfJournalEvent.__table__)
        assert "ON DELETE RESTRICT" in journal_sql
        connection.execute(journal_sql.replace("ON DELETE RESTRICT", "ON DELETE CASCADE"))

    with pytest.raises(DatabaseMigrationError, match="foreign key|schema"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)
    assert "LeveragedEtfStrategyReservation" not in _table_names(db_path)


@pytest.mark.parametrize(
    "trigger_sql",
    [
        """
        CREATE TRIGGER lepf_snapshot_executor_id_no_update
        BEFORE UPDATE OF executor_id ON LeveragedEtfExecutorSnapshot
        BEGIN
            SELECT 1;
        END
        """,
        """
        CREATE TRIGGER lepf_snapshot_executor_id_no_update
        BEFORE DELETE ON Executors
        BEGIN
            SELECT 1;
        END
        """,
    ],
    ids=["same_target_noop_body", "wrong_target_timing_and_event"],
)
def test_same_named_incompatible_trigger_is_rejected_before_version_stamp(
    tmp_path: Path,
    trigger_sql: str,
):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
        connection.execute(trigger_sql)

    with pytest.raises(DatabaseMigrationError, match="trigger|schema"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)


@pytest.mark.parametrize(
    ("table_transform", "attached_object_sql"),
    [
        (
            lambda create_sql: _insert_compiled_column(create_sql, "sabotage_required TEXT NOT NULL"),
            None,
        ),
        (
            lambda create_sql: _append_compiled_constraint(
                create_sql,
                "CONSTRAINT sabotage_snapshot_json CHECK (snapshot_json <> 'blocked')",
            ),
            None,
        ),
        (
            lambda create_sql: _append_compiled_constraint(
                create_sql,
                "CHECK (snapshot_json <> 'blocked')",
            ),
            None,
        ),
        (
            lambda create_sql: _append_compiled_constraint(
                create_sql,
                "CONSTRAINT sabotage_controller_unique UNIQUE (controller_id)",
            ),
            None,
        ),
        (
            lambda create_sql: _append_compiled_constraint(
                create_sql,
                "UNIQUE (controller_id)",
            ),
            None,
        ),
        (
            lambda create_sql: _append_compiled_constraint(
                create_sql,
                'CONSTRAINT sabotage_executor_fk FOREIGN KEY (executor_id) REFERENCES "Executors" (id) '
                "ON DELETE CASCADE",
            ),
            None,
        ),
        (
            lambda create_sql: _append_compiled_constraint(
                create_sql,
                'FOREIGN KEY (executor_id) REFERENCES "Executors" (id) ON DELETE CASCADE',
            ),
            None,
        ),
        (
            lambda create_sql: create_sql,
            """
            CREATE UNIQUE INDEX sabotage_snapshot_state_unique
            ON LeveragedEtfExecutorSnapshot (state)
            """,
        ),
        (
            lambda create_sql: create_sql,
            """
            CREATE TRIGGER arbitrary_block_snapshot_inserts
            BEFORE INSERT ON LeveragedEtfExecutorSnapshot
            BEGIN
                SELECT RAISE(ABORT, 'sabotage');
            END
            """,
        ),
    ],
    ids=[
        "extra_required_column",
        "extra_named_check_constraint",
        "extra_unnamed_check_constraint",
        "extra_named_unique_constraint",
        "extra_unnamed_unique_constraint",
        "extra_named_foreign_key",
        "extra_unnamed_foreign_key",
        "extra_unique_index",
        "arbitrary_named_blocking_trigger",
    ],
)
def test_managed_schema_rejects_unexpected_behavior_before_version_stamp(
    tmp_path: Path,
    table_transform,
    attached_object_sql: str | None,
):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(
            connection,
            LeveragedEtfExecutorSnapshot.__table__,
            transform=table_transform,
        )
        if attached_object_sql is not None:
            connection.execute(attached_object_sql)

    with pytest.raises(DatabaseMigrationError, match="schema|column|constraint|foreign key|index|trigger"):
        unexpected = _open_manager(db_path)
        unexpected.engine.dispose()

    _assert_legacy_data_unchanged(db_path)
    assert "LeveragedEtfStrategyReservation" not in _table_names(db_path)


def test_schema_validation_accepts_equivalent_expressions_and_sqlite_autoindexes(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(
            connection,
            LeveragedEtfExecutorSnapshot.__table__,
            transform=lambda create_sql: create_sql.replace(
                "CHECK (schema_version = 1)",
                "CHECK (((schema_version == 1)))",
            ),
        )
        _create_compiled_table(connection, LeveragedEtfJournalEvent.__table__)
        connection.execute("""
            CREATE UNIQUE INDEX lepf_journal_trade_dedup
            ON LeveragedEtfJournalEvent (connector_name, trading_pair, exchange_trade_id)
            WHERE ((exchange_trade_id IS NOT NULL))
            """)

    manager = _open_manager(db_path)
    try:
        assert _version(db_path) == TARGET_VERSION
        with manager.engine.connect() as connection:
            origins = {row[3] for row in connection.execute(text('PRAGMA index_list("LeveragedEtfJournalEvent")'))}
            assert {"pk", "u"} <= origins
    finally:
        manager.engine.dispose()

    reopened = _open_manager(db_path)
    reopened.engine.dispose()


OR_REPLACE_CASES = [
    (
        "snapshot_executor_id",
        """
        INSERT OR REPLACE INTO LeveragedEtfExecutorSnapshot (
            executor_id, controller_id, pair_id, nav_cycle_id, schema_version,
            state, snapshot_json, snapshot_hash, last_journal_sequence,
            created_at_utc, updated_at_utc
        ) VALUES (
            'executor-1', 'controller-1', 'sndk_snxx', 'xnys-2026-07-17', 1,
            'CREATED', 'replaced', :hash_a, 0, :created_at, :updated_at
        )
        """,
        "SELECT snapshot_json FROM LeveragedEtfExecutorSnapshot WHERE executor_id = 'executor-1'",
        "original",
    ),
    (
        "journal_event_id",
        """
        INSERT OR REPLACE INTO LeveragedEtfJournalEvent (
            event_id, executor_id, sequence, event_type, payload_json,
            payload_hash, created_at_utc
        ) VALUES ('event-1', 'executor-1', 2, 'STATE_TRANSITION',
                  'replaced', :hash_a, :created_at)
        """,
        "SELECT payload_json FROM LeveragedEtfJournalEvent WHERE event_id = 'event-1'",
        "original",
    ),
    (
        "journal_executor_sequence",
        """
        INSERT OR REPLACE INTO LeveragedEtfJournalEvent (
            event_id, executor_id, sequence, event_type, payload_json,
            payload_hash, created_at_utc
        ) VALUES ('event-2', 'executor-1', 1, 'STATE_TRANSITION',
                  'replaced', :hash_a, :created_at)
        """,
        "SELECT event_id FROM LeveragedEtfJournalEvent WHERE executor_id = 'executor-1' AND sequence = 1",
        "event-1",
    ),
    (
        "journal_trade_identity",
        """
        INSERT OR REPLACE INTO LeveragedEtfJournalEvent (
            event_id, executor_id, sequence, event_type, connector_name,
            trading_pair, exchange_trade_id, payload_json, payload_hash, created_at_utc
        ) VALUES ('event-2', 'executor-1', 2, 'FILL', 'binance_perpetual',
                  'SNXX-USDT', 'trade-1', 'replaced', :hash_a, :created_at)
        """,
        "SELECT event_id FROM LeveragedEtfJournalEvent WHERE exchange_trade_id = 'trade-1'",
        "event-1",
    ),
    (
        "reservation_id",
        """
        INSERT OR REPLACE INTO LeveragedEtfStrategyReservation (
            reservation_id, executor_id, reservation_key, connector_name,
            trading_pair, leg, quantity, leverage, notional_cap, payload_json,
            payload_hash, created_at_utc, updated_at_utc, released_at_utc
        ) VALUES ('reservation-1', 'executor-1', 'reservation-key-2',
                  'binance_perpetual', 'SNDK-USDT', 'STOCK', '1', 20, '1000',
                  'replaced', :hash_a, :created_at, :updated_at, NULL)
        """,
        "SELECT payload_json FROM LeveragedEtfStrategyReservation WHERE reservation_id = 'reservation-1'",
        "original",
    ),
    (
        "reservation_key",
        """
        INSERT OR REPLACE INTO LeveragedEtfStrategyReservation (
            reservation_id, executor_id, reservation_key, connector_name,
            trading_pair, leg, quantity, leverage, notional_cap, payload_json,
            payload_hash, created_at_utc, updated_at_utc, released_at_utc
        ) VALUES ('reservation-2', 'executor-1', 'reservation-key-1',
                  'binance_perpetual', 'SNDK-USDT', 'STOCK', '1', 20, '1000',
                  'replaced', :hash_a, :created_at, :updated_at, NULL)
        """,
        "SELECT reservation_id FROM LeveragedEtfStrategyReservation WHERE reservation_key = 'reservation-key-1'",
        "reservation-1",
    ),
    (
        "reservation_active_leg",
        """
        INSERT OR REPLACE INTO LeveragedEtfStrategyReservation (
            reservation_id, executor_id, reservation_key, connector_name,
            trading_pair, leg, quantity, leverage, notional_cap, payload_json,
            payload_hash, created_at_utc, updated_at_utc, released_at_utc
        ) VALUES ('reservation-2', 'executor-1', 'reservation-key-2',
                  'binance_perpetual', 'SNXX-USDT', 'ETF', '1', 20, '1000',
                  'replaced', :hash_a, :created_at, :updated_at, NULL)
        """,
        "SELECT reservation_id FROM LeveragedEtfStrategyReservation WHERE connector_name = 'binance_perpetual' AND trading_pair = 'SNXX-USDT' AND leg = 'ETF' AND released_at_utc IS NULL",
        "reservation-1",
    ),
    (
        "anchor_cycle_id",
        """
        INSERT OR REPLACE INTO LeveragedEtfAnchorState (
            cycle_id, schema_version, state_kind, revision, target_session_date,
            official_close_utc, deadline_utc, evidence_hash, payload_json,
            payload_hash, created_at_utc, updated_at_utc
        ) VALUES ('xnys-2026-07-17', 1, 'FINALIZED', 3, '2026-07-17', NULL,
                  '2026-07-17T20:10:00.000000Z', :hash_c, 'replaced',
                  :hash_a, :created_at, :updated_at)
        """,
        "SELECT payload_json FROM LeveragedEtfAnchorState WHERE cycle_id = 'xnys-2026-07-17'",
        "original",
    ),
    (
        "anchor_evidence_hash",
        """
        INSERT OR REPLACE INTO LeveragedEtfAnchorState (
            cycle_id, schema_version, state_kind, revision, target_session_date,
            official_close_utc, deadline_utc, evidence_hash, payload_json,
            payload_hash, created_at_utc, updated_at_utc
        ) VALUES ('xnys-2026-07-18', 1, 'FINALIZED', 1, '2026-07-18', NULL,
                  '2026-07-18T20:10:00.000000Z', :hash_b, 'replaced',
                  :hash_a, :created_at, :updated_at)
        """,
        "SELECT cycle_id FROM LeveragedEtfAnchorState WHERE evidence_hash = :hash_b",
        "xnys-2026-07-17",
    ),
    (
        "anchor_observation_primary_key",
        """
        INSERT OR REPLACE INTO LeveragedEtfAnchorRevisionObservation (
            cycle_id, evidence_hash, observed_at_utc
        ) VALUES ('xnys-2026-07-17', :hash_c, :updated_at)
        """,
        "SELECT count(*) FROM LeveragedEtfAnchorRevisionObservation WHERE cycle_id = 'xnys-2026-07-17' AND evidence_hash = :hash_c AND observed_at_utc = :updated_at",
        1,
    ),
]


@pytest.mark.parametrize(
    ("case_name", "replacement_sql", "verification_sql", "expected"),
    OR_REPLACE_CASES,
    ids=[case[0] for case in OR_REPLACE_CASES],
)
def test_insert_or_replace_cannot_rewrite_protected_identity_after_reopen(
    tmp_path: Path,
    case_name: str,
    replacement_sql: str,
    verification_sql: str,
    expected,
):
    db_path = tmp_path / f"replace-{case_name}.sqlite"
    manager = _open_manager(db_path)
    _seed_protected_rows(manager)
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    parameters = {
        "hash_a": HASH_A,
        "hash_b": HASH_B,
        "hash_c": HASH_C,
        "created_at": CREATED_AT,
        "updated_at": UPDATED_AT,
    }
    try:
        with reopened.engine.connect() as connection:
            connection.execute(text("PRAGMA recursive_triggers = OFF"))
            assert connection.execute(text("PRAGMA recursive_triggers")).scalar_one() == 0

        with pytest.raises(IntegrityError):
            with reopened.engine.begin() as connection:
                connection.execute(text(replacement_sql), parameters)

        with reopened.engine.connect() as connection:
            assert connection.execute(text(verification_sql), parameters).scalar_one() == expected
    finally:
        reopened.engine.dispose()


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("quantity", "-1"),
        ("quantity", "01"),
        ("quantity", "1.0"),
        ("quantity", "1e3"),
        ("quantity", ".5"),
        ("quantity", "0."),
        ("quantity", "not-a-number"),
        ("notional_cap", "0"),
        ("notional_cap", "-1"),
        ("notional_cap", "01"),
        ("notional_cap", "1.0"),
        ("notional_cap", "1e3"),
        ("notional_cap", ".5"),
        ("notional_cap", "0."),
        ("notional_cap", "not-a-number"),
    ],
)
def test_reservation_numeric_fields_reject_noncanonical_or_out_of_domain_values(
    tmp_path: Path,
    column: str,
    invalid_value: str,
):
    manager = _open_manager(tmp_path / f"invalid-{column}.sqlite")
    try:
        with manager.engine.begin() as connection:
            _insert_snapshot(connection)

        values = {
            "quantity": "1",
            "notional_cap": "1000",
            column: invalid_value,
        }
        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfStrategyReservation (
                            reservation_id, executor_id, reservation_key, connector_name,
                            trading_pair, leg, quantity, leverage, notional_cap, payload_json,
                            payload_hash, created_at_utc, updated_at_utc, released_at_utc
                        ) VALUES (
                            'reservation-invalid', 'executor-1', 'reservation-key-invalid',
                            'binance_perpetual', 'SNXX-USDT', 'ETF', :quantity, 20,
                            :notional_cap, '{}', :payload_hash, :created_at, :updated_at, NULL
                        )
                        """),
                    {
                        **values,
                        "payload_hash": HASH_A,
                        "created_at": CREATED_AT,
                        "updated_at": UPDATED_AT,
                    },
                )
    finally:
        manager.engine.dispose()


@pytest.mark.parametrize(
    ("column", "invalid_value"),
    [
        ("quantity", "1\x00suffix"),
        ("notional_cap", "1000\x00suffix"),
    ],
)
def test_reservation_numeric_fields_reject_embedded_nul_suffix(
    tmp_path: Path,
    column: str,
    invalid_value: str,
):
    manager = _open_manager(tmp_path / f"nul-{column}.sqlite")
    try:
        with manager.engine.begin() as connection:
            _insert_snapshot(connection)

        values = {
            "quantity": "1",
            "notional_cap": "1000",
            column: invalid_value,
        }
        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfStrategyReservation (
                            reservation_id, executor_id, reservation_key, connector_name,
                            trading_pair, leg, quantity, leverage, notional_cap, payload_json,
                            payload_hash, created_at_utc, updated_at_utc, released_at_utc
                        ) VALUES (
                            'reservation-nul', 'executor-1', 'reservation-key-nul',
                            'binance_perpetual', 'SNXX-USDT', 'ETF', :quantity, 20,
                            :notional_cap, '{}', :payload_hash, :created_at, :updated_at, NULL
                        )
                        """),
                    {
                        **values,
                        "payload_hash": HASH_A,
                        "created_at": CREATED_AT,
                        "updated_at": UPDATED_AT,
                    },
                )
    finally:
        manager.engine.dispose()


BAD_HASH_INSERTS = {
    "snapshot_hash": """
        INSERT INTO LeveragedEtfExecutorSnapshot (
            executor_id, controller_id, pair_id, nav_cycle_id, schema_version, state,
            snapshot_json, snapshot_hash, last_journal_sequence, created_at_utc, updated_at_utc
        ) VALUES ('executor-bad-hash', 'controller-1', 'sndk_snxx', 'xnys-2026-07-17',
                  1, 'CREATED', '{}', :bad_hash, 0, :created_at, :updated_at)
    """,
    "journal_payload_hash": """
        INSERT INTO LeveragedEtfJournalEvent (
            event_id, executor_id, sequence, event_type, payload_json, payload_hash, created_at_utc
        ) VALUES ('event-bad-hash', 'executor-1', 2, 'STATE_TRANSITION', '{}', :bad_hash, :created_at)
    """,
    "reservation_payload_hash": """
        INSERT INTO LeveragedEtfStrategyReservation (
            reservation_id, executor_id, reservation_key, connector_name, trading_pair,
            leg, quantity, leverage, notional_cap, payload_json, payload_hash,
            created_at_utc, updated_at_utc, released_at_utc
        ) VALUES ('reservation-bad-hash', 'executor-1', 'reservation-key-bad-hash',
                  'binance_perpetual', 'SNDK-USDT', 'STOCK', '1', 20, '1000',
                  '{}', :bad_hash, :created_at, :updated_at, NULL)
    """,
    "anchor_payload_hash": """
        INSERT INTO LeveragedEtfAnchorState (
            cycle_id, schema_version, state_kind, revision, target_session_date,
            official_close_utc, deadline_utc, evidence_hash, payload_json,
            payload_hash, created_at_utc, updated_at_utc
        ) VALUES ('xnys-2026-07-18', 1, 'CHECKPOINT', 1, '2026-07-18',
                  '2026-07-18T20:00:00.000000Z', '2026-07-18T20:10:00.000000Z',
                  NULL, '{}', :bad_hash, :created_at, :updated_at)
    """,
    "anchor_evidence_hash": """
        INSERT INTO LeveragedEtfAnchorState (
            cycle_id, schema_version, state_kind, revision, target_session_date,
            official_close_utc, deadline_utc, evidence_hash, payload_json,
            payload_hash, created_at_utc, updated_at_utc
        ) VALUES ('xnys-2026-07-18', 1, 'FINALIZED', 1, '2026-07-18', NULL,
                  '2026-07-18T20:10:00.000000Z', :bad_hash, '{}', :hash_a,
                  :created_at, :updated_at)
    """,
    "anchor_observation_hash": """
        INSERT INTO LeveragedEtfAnchorRevisionObservation (
            cycle_id, evidence_hash, observed_at_utc
        ) VALUES ('xnys-2026-07-17', :bad_hash, :updated_at)
    """,
}


@pytest.mark.parametrize(("case_name", "insert_sql"), BAD_HASH_INSERTS.items())
def test_all_hash_constraints_reject_lowercase_non_hex_text(
    tmp_path: Path,
    case_name: str,
    insert_sql: str,
):
    manager = _open_manager(tmp_path / f"bad-hash-{case_name}.sqlite")
    try:
        with manager.engine.begin() as connection:
            _insert_snapshot(connection)
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfAnchorState (
                        cycle_id, schema_version, state_kind, revision, target_session_date,
                        official_close_utc, deadline_utc, evidence_hash, payload_json,
                        payload_hash, created_at_utc, updated_at_utc
                    ) VALUES ('xnys-2026-07-17', 1, 'FINALIZED', 1, '2026-07-17', NULL,
                              '2026-07-17T20:10:00.000000Z', :hash_b, '{}', :hash_a,
                              :created_at, :updated_at)
                    """),
                {
                    "hash_a": HASH_A,
                    "hash_b": HASH_B,
                    "created_at": CREATED_AT,
                    "updated_at": UPDATED_AT,
                },
            )

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text(insert_sql),
                    {
                        "bad_hash": BAD_HASH,
                        "hash_a": HASH_A,
                        "created_at": CREATED_AT,
                        "updated_at": UPDATED_AT,
                    },
                )
    finally:
        manager.engine.dispose()


@pytest.mark.parametrize(("case_name", "insert_sql"), BAD_HASH_INSERTS.items())
def test_all_hash_constraints_reject_embedded_nul_suffix(
    tmp_path: Path,
    case_name: str,
    insert_sql: str,
):
    manager = _open_manager(tmp_path / f"nul-hash-{case_name}.sqlite")
    try:
        with manager.engine.begin() as connection:
            _insert_snapshot(connection)
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfAnchorState (
                        cycle_id, schema_version, state_kind, revision, target_session_date,
                        official_close_utc, deadline_utc, evidence_hash, payload_json,
                        payload_hash, created_at_utc, updated_at_utc
                    ) VALUES ('xnys-2026-07-17', 1, 'FINALIZED', 1, '2026-07-17', NULL,
                              '2026-07-17T20:10:00.000000Z', :hash_b, '{}', :hash_a,
                              :created_at, :updated_at)
                    """),
                {
                    "hash_a": HASH_A,
                    "hash_b": HASH_B,
                    "created_at": CREATED_AT,
                    "updated_at": UPDATED_AT,
                },
            )

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text(insert_sql),
                    {
                        "bad_hash": f"{HASH_A}\x00g",
                        "hash_a": HASH_A,
                        "created_at": CREATED_AT,
                        "updated_at": UPDATED_AT,
                    },
                )
    finally:
        manager.engine.dispose()


@pytest.mark.parametrize(
    ("connector_name", "trading_pair"),
    [(None, "SNXX-USDT"), ("binance_perpetual", None), (None, None)],
)
def test_exchange_trade_id_requires_complete_deduplication_identity(
    tmp_path: Path,
    connector_name: str | None,
    trading_pair: str | None,
):
    manager = _open_manager(tmp_path / "trade-identity.sqlite")
    try:
        with manager.engine.begin() as connection:
            _insert_snapshot(connection)

        with pytest.raises(IntegrityError):
            with manager.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT INTO LeveragedEtfJournalEvent (
                            event_id, executor_id, sequence, event_type, connector_name,
                            trading_pair, exchange_trade_id, payload_json, payload_hash,
                            created_at_utc
                        ) VALUES ('trade-event', 'executor-1', 1, 'FILL', :connector_name,
                                  :trading_pair, 'trade-1', '{}', :payload_hash, :created_at)
                        """),
                    {
                        "connector_name": connector_name,
                        "trading_pair": trading_pair,
                        "payload_hash": HASH_A,
                        "created_at": CREATED_AT,
                    },
                )
    finally:
        manager.engine.dispose()


def test_released_reservation_allows_a_new_active_reservation_for_the_same_leg(tmp_path: Path):
    manager = _open_manager(tmp_path / "release-rereserve.sqlite")
    try:
        with manager.engine.begin() as connection:
            _insert_snapshot(connection)
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfStrategyReservation (
                        reservation_id, executor_id, reservation_key, connector_name,
                        trading_pair, leg, quantity, leverage, notional_cap, payload_json,
                        payload_hash, created_at_utc, updated_at_utc, released_at_utc
                    ) VALUES ('reservation-1', 'executor-1', 'reservation-key-1',
                              'binance_perpetual', 'SNXX-USDT', 'ETF', '1', 20, '1000',
                              '{}', :payload_hash, :created_at, :updated_at, NULL)
                    """),
                {
                    "payload_hash": HASH_A,
                    "created_at": CREATED_AT,
                    "updated_at": UPDATED_AT,
                },
            )
            connection.execute(
                text("""
                    UPDATE LeveragedEtfStrategyReservation
                    SET released_at_utc = :released_at, updated_at_utc = :released_at
                    WHERE reservation_id = 'reservation-1'
                    """),
                {"released_at": UPDATED_AT},
            )
            connection.execute(
                text("""
                    INSERT INTO LeveragedEtfStrategyReservation (
                        reservation_id, executor_id, reservation_key, connector_name,
                        trading_pair, leg, quantity, leverage, notional_cap, payload_json,
                        payload_hash, created_at_utc, updated_at_utc, released_at_utc
                    ) VALUES ('reservation-2', 'executor-1', 'reservation-key-2',
                              'binance_perpetual', 'SNXX-USDT', 'ETF', '2', 20, '2000',
                              '{}', :payload_hash, :created_at, :updated_at, NULL)
                    """),
                {
                    "payload_hash": HASH_A,
                    "created_at": CREATED_AT,
                    "updated_at": UPDATED_AT,
                },
            )

        with manager.engine.connect() as connection:
            assert connection.execute(text("SELECT count(*) FROM LeveragedEtfStrategyReservation")).scalar_one() == 2
            assert (
                connection.execute(
                    text("SELECT reservation_id FROM LeveragedEtfStrategyReservation " "WHERE released_at_utc IS NULL")
                ).scalar_one()
                == "reservation-2"
            )
    finally:
        manager.engine.dispose()
