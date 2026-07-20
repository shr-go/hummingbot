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
import hummingbot.model.leveraged_etf_persistence as persistence_module
from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.model.db_migration.migrator import Migrator
from hummingbot.model.leveraged_etf_persistence import (
    SQLITE_GUARD_DDL,
    LeveragedEtfAnchorState,
    LeveragedEtfExecutorSnapshot,
    LeveragedEtfJournalEvent,
    LeveragedEtfStrategyReservation,
)
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


def _materialize_encoded_legacy_database(tmp_path: Path, encoding: str) -> Path:
    db_path = tmp_path / f"legacy-{encoding.lower()}.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute(f"PRAGMA encoding = '{encoding}'")
        connection.executescript(LEGACY_FIXTURE.read_text(encoding="utf-8"))
        assert connection.execute("PRAGMA encoding").fetchone()[0].lower() == encoding.lower()
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


def _install_round5_sqlite_syntax_variant(connection: sqlite3.Connection, variant: str) -> str:
    table_name = LeveragedEtfExecutorSnapshot.__tablename__
    if variant == "check_operand_parentheses":
        _create_compiled_table(
            connection,
            LeveragedEtfExecutorSnapshot.__table__,
            transform=lambda create_sql: create_sql.replace(
                "CHECK (schema_version = 1)",
                "CHECK (([schema_version]) == ((1)))",
                1,
            ),
        )
        return "([schema_version]) == ((1))"

    _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
    if variant == "trigger_for_each_added":
        trigger_sql = SQLITE_GUARD_DDL["lepf_snapshot_executor_id_no_update"].replace(
            f"ON {table_name}",
            f"ON {table_name}\n        FOR EACH ROW",
            1,
        )
        connection.execute(trigger_sql)
        return "FOR EACH ROW"
    if variant == "trigger_for_each_omitted":
        trigger_sql = SQLITE_GUARD_DDL["lepf_snapshot_identity_insert"].replace(
            "FOR EACH ROW",
            "",
            1,
        )
        connection.execute(trigger_sql)
        return "BEFORE INSERT"
    if variant == "trigger_main_schema":
        trigger_sql = SQLITE_GUARD_DDL["lepf_snapshot_identity_insert"].replace(
            f"ON {table_name}",
            f"ON main.{table_name}",
            1,
        )
        trigger_sql = trigger_sql.replace(
            f"FROM {table_name}",
            f"FROM main.{table_name}",
            1,
        )
        connection.execute(trigger_sql)
        return f"ON main.{table_name}"
    if variant == "index_explicit_asc":
        connection.execute(f"CREATE INDEX lepf_snapshot_updated ON {table_name} (updated_at_utc ASC)")
        return "updated_at_utc ASC"
    if variant == "index_explicit_binary":
        connection.execute(f"CREATE INDEX lepf_snapshot_updated ON {table_name} (updated_at_utc COLLATE BINARY)")
        return "updated_at_utc COLLATE BINARY"
    if variant == "index_explicit_binary_asc":
        connection.execute(f"CREATE INDEX lepf_snapshot_updated ON {table_name} " "(updated_at_utc COLLATE BINARY ASC)")
        return "updated_at_utc COLLATE BINARY ASC"
    raise AssertionError(f"unknown round-five SQLite syntax variant: {variant}")


def _round5_variant_sql(db_path: Path, variant: str) -> str:
    object_type = (
        "table"
        if variant == "check_operand_parentheses"
        else ("trigger" if variant.startswith("trigger_") else "index")
    )
    object_name = {
        "table": LeveragedEtfExecutorSnapshot.__tablename__,
        "trigger": (
            "lepf_snapshot_executor_id_no_update"
            if variant == "trigger_for_each_added"
            else "lepf_snapshot_identity_insert"
        ),
        "index": "lepf_snapshot_updated",
    }[object_type]
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, object_name),
        ).fetchone()
    assert row is not None
    return row[0]


def _install_round5_meaningful_drift(connection: sqlite3.Connection, drift: str) -> None:
    table_name = LeveragedEtfExecutorSnapshot.__tablename__
    if drift == "check_value":
        _create_compiled_table(
            connection,
            LeveragedEtfExecutorSnapshot.__table__,
            transform=lambda create_sql: create_sql.replace(
                "CHECK (schema_version = 1)",
                "CHECK ((schema_version) = (2))",
                1,
            ),
        )
        return
    if drift == "index_auxiliary_rows":
        _create_compiled_table(
            connection,
            LeveragedEtfExecutorSnapshot.__table__,
            transform=lambda create_sql: f"{create_sql.rstrip()} WITHOUT ROWID",
        )
        connection.execute(f"CREATE INDEX lepf_snapshot_updated ON {table_name} (updated_at_utc COLLATE BINARY ASC)")
        auxiliary_rows = [
            row for row in connection.execute('PRAGMA index_xinfo("lepf_snapshot_updated")') if not row[5]
        ]
        assert auxiliary_rows and auxiliary_rows[0][2] == "executor_id"
        return

    _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
    if drift.startswith("trigger_"):
        trigger_sql = SQLITE_GUARD_DDL["lepf_snapshot_identity_insert"]
        if drift == "trigger_target":
            connection.execute("CREATE TABLE SnapshotShadow (executor_id TEXT)")
            trigger_sql = trigger_sql.replace(table_name, "SnapshotShadow", 1)
        elif drift == "trigger_body":
            trigger_sql = trigger_sql.replace("identity already exists", "different body")
        elif drift == "trigger_for_each_literal":
            trigger_sql = trigger_sql.replace("identity already exists", "FOR EACH ROW")
        elif drift == "trigger_main_literal":
            trigger_sql = trigger_sql.replace("identity already exists", f"main.{table_name}")
        else:
            raise AssertionError(f"unknown round-five trigger drift: {drift}")
        connection.execute(trigger_sql)
        return

    index_keys = {
        "index_desc": ("lepf_snapshot_updated", "updated_at_utc DESC", ""),
        "index_nocase": ("lepf_snapshot_updated", "updated_at_utc COLLATE NOCASE", ""),
        "index_expression": ("lepf_snapshot_updated", "lower(updated_at_utc)", ""),
        "index_predicate": (
            "lepf_snapshot_updated",
            "updated_at_utc",
            " WHERE updated_at_utc IS NOT NULL",
        ),
        "index_key_order": (
            "lepf_snapshot_controller_state",
            "state, controller_id",
            "",
        ),
    }
    if drift == "index_target_table":
        connection.execute("CREATE TABLE SnapshotIndexShadow (updated_at_utc TEXT)")
        connection.execute("CREATE INDEX lepf_snapshot_updated ON SnapshotIndexShadow (updated_at_utc)")
        return
    if drift not in index_keys:
        raise AssertionError(f"unknown round-five index drift: {drift}")
    index_name, key_sql, predicate_sql = index_keys[drift]
    connection.execute(f"CREATE INDEX {index_name} ON {table_name} ({key_sql}){predicate_sql}")


def _install_round6_keyword_parentheses_variant(connection: sqlite3.Connection, variant: str) -> tuple[str, str]:
    if variant == "managed_check_is_null":
        _create_compiled_table(
            connection,
            LeveragedEtfAnchorState.__table__,
            transform=lambda create_sql: create_sql.replace(
                "evidence_hash IS NULL",
                "evidence_hash IS (NULL)",
                1,
            ),
        )
        return "table", LeveragedEtfAnchorState.__tablename__
    if variant == "partial_index_is_null":
        _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
        _create_compiled_table(connection, LeveragedEtfStrategyReservation.__table__)
        connection.execute(f"""
            CREATE UNIQUE INDEX lepf_reservation_active_leg
            ON {LeveragedEtfStrategyReservation.__tablename__} (
                executor_id,
                connector_name,
                trading_pair,
                leg
            )
            WHERE released_at_utc IS (NULL)
            """)
        return "index", "lepf_reservation_active_leg"
    raise AssertionError(f"unknown round-six keyword-parentheses variant: {variant}")


ROUND7_KEYWORD_FUNCTIONS = ("glob", "like", "match", "regexp")
ROUND7_OPERAND_CONTEXTS = (
    ("and", "flag AND {name}(value)"),
    ("is", "value IS {name}(value)"),
    ("between_lower", "value BETWEEN {name}(low_value) AND 10"),
    ("between_upper", "value BETWEEN 0 AND {name}(high_value)"),
    ("case_base", "CASE {name}(flag) WHEN 1 THEN 2 ELSE 3 END"),
    ("case_then", "CASE WHEN flag THEN {name}(value) ELSE 0 END"),
    ("like", "'alpha' LIKE {name}(pattern)"),
    ("escape", "'a_b' LIKE 'a#_b' ESCAPE {name}(escape_char)"),
)


def _parenthesize_round7_keyword_call(expression: str, function_name: str) -> str:
    call_at = expression.index(f"{function_name}(")
    depth = 0
    for cursor in range(call_at + len(function_name), len(expression)):
        if expression[cursor] == "(":
            depth += 1
        elif expression[cursor] == ")":
            depth -= 1
            if depth == 0:
                return f"{expression[:call_at]}({expression[call_at : cursor + 1]}){expression[cursor + 1 :]}"
    raise AssertionError(f"unbalanced keyword-shaped function call: {expression}")


def _evaluate_round7_sqlite_expression(connection: sqlite3.Connection, expression: str):
    return connection.execute(f"""
        WITH vars (
            flag, value, low_value, high_value, pattern, escape_char
        ) AS (
            VALUES (1, 5, 1, 10, 'a%', '#')
        )
        SELECT {expression} FROM vars
        """).fetchone()[0]


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


def test_round4_wrong_target_trigger_with_embedded_identifier_token_is_rejected_before_stamp(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    shadow_table = "LeveragedEtfIfNotExistsExecutorSnapshot"
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
        connection.execute(f"CREATE TABLE {shadow_table} (executor_id TEXT)")
        connection.execute(
            SQLITE_GUARD_DDL["lepf_snapshot_identity_insert"].replace(
                "LeveragedEtfExecutorSnapshot",
                shadow_table,
                2,
            )
        )

    with pytest.raises(DatabaseMigrationError, match="trigger|schema"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)


def test_round4_trigger_comparison_preserves_embedded_literal_tokens(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
        connection.execute(
            SQLITE_GUARD_DDL["lepf_snapshot_identity_insert"].replace(
                "identity already exists",
                "ifnotexistsidentity already exists",
            )
        )

    with pytest.raises(DatabaseMigrationError, match="trigger|schema"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)


def test_round4_trigger_comparison_preserves_multi_table_body_references(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    journal_shadow = "LeveragedEtfIfNotExistsJournalEvent"
    reservation_shadow = "LeveragedEtfIfNotExistsStrategyReservation"
    trigger_sql = SQLITE_GUARD_DDL["lepf_snapshot_referenced_no_delete"]
    trigger_sql = trigger_sql.replace("LeveragedEtfJournalEvent", journal_shadow)
    trigger_sql = trigger_sql.replace("LeveragedEtfStrategyReservation", reservation_shadow)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
        connection.execute(f"CREATE TABLE {journal_shadow} (executor_id TEXT)")
        connection.execute(f"CREATE TABLE {reservation_shadow} (executor_id TEXT)")
        connection.execute(trigger_sql)

    with pytest.raises(DatabaseMigrationError, match="trigger|schema"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)


@pytest.mark.parametrize(
    "extra_check_sql",
    [
        "/* ' */ CHECK (snapshot_json <> 'blocked') /* ' */",
        "-- ' CHECK (ignored)\n\tCHECK (snapshot_json <> 'blocked')",
        "/* CHECK (ignored) \" [ ( */ CHECK ((snapshot_json <> 'blocked'))",
        "CHECK (snapshot_json <> 'escaped '' CHECK(token)')",
        'CHECK ("CHECK(snapshot_json)" IS NULL)',
    ],
    ids=[
        "block_comment_quote",
        "line_comment_quote",
        "block_comment_check_and_delimiters",
        "escaped_string_check_token",
        "quoted_identifier_check_token",
    ],
)
def test_round4_comment_quote_and_check_token_collisions_cannot_hide_extra_constraints(
    tmp_path: Path,
    extra_check_sql: str,
):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(
            connection,
            LeveragedEtfExecutorSnapshot.__table__,
            transform=lambda create_sql: _append_compiled_constraint(create_sql, extra_check_sql),
        )

    with pytest.raises(DatabaseMigrationError, match="schema|CHECK|constraint"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)


def test_round4_check_parser_accepts_comments_inside_an_equivalent_expression(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(
            connection,
            LeveragedEtfExecutorSnapshot.__table__,
            transform=lambda create_sql: create_sql.replace(
                "CHECK (schema_version = 1)",
                "CHECK (((schema_version /* CHECK ('ignored') */ == 1)))",
            ),
        )

    manager = _open_manager(db_path)
    manager.engine.dispose()
    assert _version(db_path) == TARGET_VERSION

    reopened = _open_manager(db_path)
    reopened.engine.dispose()


@pytest.mark.parametrize("generated_kind", ["STORED", "VIRTUAL"])
def test_round4_generated_snapshot_hash_column_is_rejected_before_version_stamp(
    tmp_path: Path,
    generated_kind: str,
):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(
            connection,
            LeveragedEtfExecutorSnapshot.__table__,
            transform=lambda create_sql: create_sql.replace(
                "snapshot_hash TEXT NOT NULL",
                f"snapshot_hash TEXT GENERATED ALWAYS AS ('{HASH_A}') {generated_kind} NOT NULL",
                1,
            ),
        )

    with pytest.raises(DatabaseMigrationError, match="schema|column"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)


@pytest.mark.parametrize(
    ("replacement", "case_name"),
    [
        ("snapshot_json VARCHAR NOT NULL", "declared_type"),
        ("snapshot_json TEXT COLLATE NOCASE NOT NULL", "column_collation"),
    ],
)
def test_round4_same_affinity_column_behavior_drift_is_rejected_before_version_stamp(
    tmp_path: Path,
    replacement: str,
    case_name: str,
):
    db_path = _materialize_legacy_database(tmp_path, f"column-{case_name}.sqlite")
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(
            connection,
            LeveragedEtfExecutorSnapshot.__table__,
            transform=lambda create_sql: create_sql.replace(
                "snapshot_json TEXT NOT NULL",
                replacement,
                1,
            ),
        )

    with pytest.raises(DatabaseMigrationError, match="schema|column"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)


@pytest.mark.parametrize(
    "index_key_sql",
    [
        "updated_at_utc COLLATE NOCASE",
        "updated_at_utc DESC",
        "lower(updated_at_utc)",
        "updated_at_utc) WHERE updated_at_utc IS NOT NULL --",
    ],
    ids=["collation", "descending", "expression", "predicate"],
)
def test_round4_explicit_index_key_semantics_are_rejected_before_version_stamp(
    tmp_path: Path,
    index_key_sql: str,
):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
        connection.execute("CREATE INDEX lepf_snapshot_updated " f"ON LeveragedEtfExecutorSnapshot ({index_key_sql})")

    with pytest.raises(DatabaseMigrationError, match="schema|index"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)


def test_round4_partial_unique_index_collation_is_rejected_before_version_stamp(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
        _create_compiled_table(connection, LeveragedEtfJournalEvent.__table__)
        connection.execute("""
            CREATE UNIQUE INDEX lepf_journal_trade_dedup
            ON LeveragedEtfJournalEvent (
                connector_name COLLATE NOCASE,
                trading_pair,
                exchange_trade_id
            )
            WHERE exchange_trade_id IS NOT NULL
            """)

    with pytest.raises(DatabaseMigrationError, match="schema|index"):
        _open_manager(db_path)

    _assert_legacy_data_unchanged(db_path)


def test_round4_unique_autoindex_collation_is_rejected_before_version_stamp(tmp_path: Path):
    db_path = _materialize_legacy_database(tmp_path)
    with sqlite3.connect(db_path) as connection:
        _create_compiled_table(connection, LeveragedEtfExecutorSnapshot.__table__)
        _create_compiled_table(
            connection,
            LeveragedEtfStrategyReservation.__table__,
            transform=lambda create_sql: create_sql.replace(
                "reservation_key TEXT NOT NULL",
                "reservation_key TEXT COLLATE NOCASE NOT NULL",
                1,
            ),
        )
        autoindex_name = connection.execute(
            "SELECT name FROM pragma_index_list('LeveragedEtfStrategyReservation') " "WHERE origin = 'u'"
        ).fetchone()[0]
        assert "NOCASE" in {row[4] for row in connection.execute(f'PRAGMA index_xinfo("{autoindex_name}")') if row[5]}

    with pytest.raises(DatabaseMigrationError, match="schema|column|index|UNIQUE"):
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


@pytest.mark.parametrize(
    "variant",
    [
        "check_operand_parentheses",
        "trigger_for_each_added",
        "trigger_for_each_omitted",
        "trigger_main_schema",
        "index_explicit_asc",
        "index_explicit_binary",
        "index_explicit_binary_asc",
    ],
)
def test_round5_sqlite_noop_syntax_survives_legacy_migration_and_current_reopen(
    tmp_path: Path,
    variant: str,
):
    db_path = _materialize_legacy_database(tmp_path, f"round5-equivalent-{variant}.sqlite")
    with sqlite3.connect(db_path) as connection:
        marker = _install_round5_sqlite_syntax_variant(connection, variant)

    manager = _open_manager(db_path)
    manager.engine.dispose()
    assert _version(db_path) == TARGET_VERSION

    persisted_sql = _round5_variant_sql(db_path, variant)
    assert marker in persisted_sql
    if variant == "trigger_for_each_omitted":
        assert "FOR EACH ROW" not in persisted_sql

    reopened = _open_manager(db_path)
    reopened.engine.dispose()
    assert _round5_variant_sql(db_path, variant) == persisted_sql


@pytest.mark.parametrize(
    "drift",
    [
        "check_value",
        "trigger_target",
        "trigger_body",
        "trigger_for_each_literal",
        "trigger_main_literal",
        "index_desc",
        "index_nocase",
        "index_expression",
        "index_predicate",
        "index_target_table",
        "index_key_order",
        "index_auxiliary_rows",
    ],
)
def test_round5_meaningful_schema_drift_remains_rejected_before_version_stamp(
    tmp_path: Path,
    drift: str,
):
    db_path = _materialize_legacy_database(tmp_path, f"round5-drift-{drift}.sqlite")
    with sqlite3.connect(db_path) as connection:
        _install_round5_meaningful_drift(connection, drift)

    with pytest.raises(DatabaseMigrationError, match="schema|CHECK|trigger|index"):
        unexpected = _open_manager(db_path)
        unexpected.engine.dispose()

    _assert_legacy_data_unchanged(db_path)


@pytest.mark.parametrize(
    "variant",
    ["managed_check_is_null", "partial_index_is_null"],
)
def test_round6_keyword_operand_parentheses_survive_legacy_migration_and_current_reopen(
    tmp_path: Path,
    variant: str,
):
    db_path = _materialize_legacy_database(tmp_path, f"round6-equivalent-{variant}.sqlite")
    with sqlite3.connect(db_path) as connection:
        object_type, object_name = _install_round6_keyword_parentheses_variant(connection, variant)

    manager = _open_manager(db_path)
    manager.engine.dispose()
    assert _version(db_path) == TARGET_VERSION

    with sqlite3.connect(db_path) as connection:
        persisted_sql = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
            (object_type, object_name),
        ).fetchone()[0]
    assert "IS (NULL)" in persisted_sql

    reopened = _open_manager(db_path)
    reopened.engine.dispose()
    with sqlite3.connect(db_path) as connection:
        assert (
            connection.execute(
                "SELECT sql FROM sqlite_master WHERE type = ? AND name = ?",
                (object_type, object_name),
            ).fetchone()[0]
            == persisted_sql
        )


@pytest.mark.parametrize(
    ("canonical_sql", "equivalent_sql"),
    [
        ("evidence_hash IS NULL", "evidence_hash IS (NULL)"),
        ("evidence_hash IS NOT NULL", "evidence_hash IS NOT (NULL)"),
        ("state GLOB 'F*'", "state GLOB ('F*')"),
        ("state NOT GLOB 'F*'", "state NOT GLOB ('F*')"),
        ("revision BETWEEN 1 AND 3", "revision BETWEEN (1) AND (3)"),
        ("is_valid AND is_ready", "is_valid AND (is_ready)"),
        ("is_valid OR is_ready", "is_valid OR (is_ready)"),
        ("NOT is_valid", "NOT (is_valid)"),
        ("name LIKE 'a%' ESCAPE '_'", "name LIKE ('a%') ESCAPE ('_')"),
    ],
)
def test_round6_atomic_parentheses_are_normalized_after_keyword_operators(
    canonical_sql: str,
    equivalent_sql: str,
):
    assert persistence_module._normalize_sql(equivalent_sql) == persistence_module._normalize_sql(canonical_sql)


@pytest.mark.parametrize(
    ("with_parentheses", "without_parentheses"),
    [
        ("custom_function(value)", "custom_function value"),
        ("glob(value)", "glob value"),
        ("value IN ('ETF')", "value IN 'ETF'"),
        ("EXISTS (SELECT 1)", "EXISTS SELECT 1"),
        ("value = (SELECT 1)", "value = SELECT 1"),
        ("left_value + (middle_value * right_value)", "left_value + middle_value * right_value"),
    ],
)
def test_round6_grammar_significant_parentheses_remain_distinct(
    with_parentheses: str,
    without_parentheses: str,
):
    assert persistence_module._normalize_sql(with_parentheses) != persistence_module._normalize_sql(without_parentheses)


def test_round6_trigger_body_parentheses_remain_token_significant():
    parenthesized_body = """
        CREATE TRIGGER preserve_body_parentheses
        BEFORE UPDATE ON LeveragedEtfExecutorSnapshot
        BEGIN
            SELECT (1);
        END
    """
    unparenthesized_body = parenthesized_body.replace("SELECT (1)", "SELECT 1")

    assert persistence_module._normalize_trigger_sql(parenthesized_body) != persistence_module._normalize_trigger_sql(
        unparenthesized_body
    )


def test_round7_case_base_atomic_grouping_is_sqlite_equivalent():
    canonical_sql = "CASE value WHEN 5 THEN 1 ELSE 0 END"
    equivalent_sql = "CASE (value) WHEN 5 THEN 1 ELSE 0 END"
    with sqlite3.connect(":memory:") as connection:
        assert _evaluate_round7_sqlite_expression(
            connection,
            equivalent_sql,
        ) == _evaluate_round7_sqlite_expression(connection, canonical_sql)

    assert persistence_module._normalize_sql(equivalent_sql) == persistence_module._normalize_sql(canonical_sql)


@pytest.mark.parametrize("function_name", ROUND7_KEYWORD_FUNCTIONS)
@pytest.mark.parametrize(
    ("context_name", "expression_template"),
    ROUND7_OPERAND_CONTEXTS,
    ids=[context_name for context_name, _ in ROUND7_OPERAND_CONTEXTS],
)
def test_round7_registered_keyword_function_grouping_is_sqlite_equivalent(
    function_name: str,
    context_name: str,
    expression_template: str,
):
    canonical_sql = expression_template.format(name=function_name)
    equivalent_sql = _parenthesize_round7_keyword_call(canonical_sql, function_name)
    with sqlite3.connect(":memory:") as connection:
        connection.create_function(function_name, 1, lambda value: value)
        assert _evaluate_round7_sqlite_expression(
            connection,
            equivalent_sql,
        ) == _evaluate_round7_sqlite_expression(connection, canonical_sql)

    assert persistence_module._normalize_sql(equivalent_sql) == persistence_module._normalize_sql(canonical_sql)


@pytest.mark.parametrize("function_name", ROUND7_KEYWORD_FUNCTIONS)
def test_round7_keyword_function_argument_lists_cannot_collapse_to_invalid_sql(function_name: str):
    canonical_sql = f"flag AND {function_name}(value)"
    invalid_sql = f"flag AND {function_name} value"
    with sqlite3.connect(":memory:") as connection:
        connection.create_function(function_name, 1, lambda value: value)
        _evaluate_round7_sqlite_expression(connection, canonical_sql)
        with pytest.raises(sqlite3.OperationalError):
            _evaluate_round7_sqlite_expression(connection, invalid_sql)

    assert persistence_module._normalize_sql(canonical_sql) != persistence_module._normalize_sql(invalid_sql)


@pytest.mark.parametrize("function_name", ROUND7_KEYWORD_FUNCTIONS)
def test_round7_keyword_function_calls_at_expression_start_keep_argument_lists(function_name: str):
    canonical_sql = f"{function_name}(value)"
    equivalent_sql = f"({canonical_sql})"
    invalid_sql = f"{function_name} value"

    assert persistence_module._normalize_sql(equivalent_sql) == persistence_module._normalize_sql(canonical_sql)
    assert persistence_module._normalize_sql(canonical_sql) != persistence_module._normalize_sql(invalid_sql)


@pytest.mark.parametrize(
    ("with_parentheses", "without_parentheses"),
    [
        ("value IN ('ETF')", "value IN 'ETF'"),
        ("EXISTS (SELECT 1)", "EXISTS SELECT 1"),
        ("value = (SELECT 1)", "value = SELECT 1"),
        ("left_value + (middle_value * right_value)", "left_value + middle_value * right_value"),
    ],
)
def test_round7_non_grouping_parentheses_remain_distinct(
    with_parentheses: str,
    without_parentheses: str,
):
    assert persistence_module._normalize_sql(with_parentheses) != persistence_module._normalize_sql(without_parentheses)


def test_round7_trigger_body_parentheses_and_literals_remain_token_significant():
    expected_sql = SQLITE_GUARD_DDL["lepf_snapshot_identity_insert"]
    changed_parentheses = expected_sql.replace("RAISE(ABORT,", "RAISE ABORT,", 1)
    changed_literal = expected_sql.replace("identity already exists", "glob(value)", 1)

    assert persistence_module._normalize_trigger_sql(expected_sql) != persistence_module._normalize_trigger_sql(
        changed_parentheses
    )
    assert persistence_module._normalize_trigger_sql(expected_sql) != persistence_module._normalize_trigger_sql(
        changed_literal
    )


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


ROUND4_INVALID_HASH_VALUES = (
    HASH_A + "\x00suffix",
    HASH_A.encode("ascii"),
    "é" * 64,
    HASH_A.upper(),
    BAD_HASH,
    "a" * 63,
)

ROUND4_INVALID_DECIMAL_VALUES = (
    "1\x00suffix",
    b"1",
    "１",
    "+1",
    "-1",
    "1e3",
    "NaN",
    "Infinity",
    "-Infinity",
    "00",
    "01",
    "0.0",
    "0.00",
    "1.0",
    ".5",
    "0.",
    " 1",
    "1 ",
)


def _seed_round4_hash_prerequisites(manager: SQLConnectionManager) -> None:
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


def _assert_round4_hash_insert(
    manager: SQLConnectionManager,
    insert_sql: str,
    candidate,
    accepted: bool,
) -> None:
    with manager.engine.connect() as connection:
        transaction = connection.begin()
        try:
            parameters = {
                "bad_hash": candidate,
                "hash_a": HASH_A,
                "created_at": CREATED_AT,
                "updated_at": UPDATED_AT,
            }
            if accepted:
                connection.execute(text(insert_sql), parameters)
            else:
                with pytest.raises(IntegrityError):
                    connection.execute(text(insert_sql), parameters)
        finally:
            transaction.rollback()


def _assert_round4_decimal_insert(
    manager: SQLConnectionManager,
    quantity,
    notional_cap,
    accepted: bool,
) -> None:
    with manager.engine.connect() as connection:
        transaction = connection.begin()
        try:
            statement = text("""
                INSERT INTO LeveragedEtfStrategyReservation (
                    reservation_id, executor_id, reservation_key, connector_name,
                    trading_pair, leg, quantity, leverage, notional_cap, payload_json,
                    payload_hash, created_at_utc, updated_at_utc, released_at_utc
                ) VALUES ('round4-reservation', 'executor-1', 'round4-reservation-key',
                          'binance_perpetual', 'SNXX-USDT', 'ETF', :quantity, 20,
                          :notional_cap, '{}', :payload_hash, :created_at, :updated_at,
                          :updated_at)
                """)
            parameters = {
                "quantity": quantity,
                "notional_cap": notional_cap,
                "payload_hash": HASH_A,
                "created_at": CREATED_AT,
                "updated_at": UPDATED_AT,
            }
            if accepted:
                connection.execute(statement, parameters)
            else:
                with pytest.raises(IntegrityError):
                    connection.execute(statement, parameters)
        finally:
            transaction.rollback()


@pytest.mark.parametrize("encoding", ["UTF-8", "UTF-16le", "UTF-16be"])
def test_round4_encoded_text_domain_matrix_survives_migration_and_reopen(
    tmp_path: Path,
    encoding: str,
):
    db_path = _materialize_encoded_legacy_database(tmp_path, encoding)
    manager = _open_manager(db_path)
    manager.engine.dispose()
    assert _version(db_path) == TARGET_VERSION

    reopened = _open_manager(db_path)
    try:
        with reopened.engine.connect() as connection:
            assert connection.execute(text("PRAGMA encoding")).scalar_one().lower() == encoding.lower()
        _seed_round4_hash_prerequisites(reopened)

        for insert_sql in BAD_HASH_INSERTS.values():
            _assert_round4_hash_insert(reopened, insert_sql, HASH_A, accepted=True)
            for invalid_hash in ROUND4_INVALID_HASH_VALUES:
                _assert_round4_hash_insert(reopened, insert_sql, invalid_hash, accepted=False)

        for valid_quantity in ("0", "1", "0.1", "10.01", "1000"):
            _assert_round4_decimal_insert(reopened, valid_quantity, "1000", accepted=True)
        for valid_notional_cap in ("1", "0.1", "10.01", "1000"):
            _assert_round4_decimal_insert(reopened, "1", valid_notional_cap, accepted=True)

        for invalid_quantity in ROUND4_INVALID_DECIMAL_VALUES:
            _assert_round4_decimal_insert(reopened, invalid_quantity, "1000", accepted=False)
        for invalid_notional_cap in ("0", *ROUND4_INVALID_DECIMAL_VALUES):
            _assert_round4_decimal_insert(reopened, "1", invalid_notional_cap, accepted=False)
    finally:
        reopened.engine.dispose()


def test_round4_snapshot_identity_guard_blocks_insert_or_replace_after_reopen(tmp_path: Path):
    db_path = tmp_path / "round4-replace.sqlite"
    manager = _open_manager(db_path)
    with manager.engine.begin() as connection:
        _insert_snapshot(connection)
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        with pytest.raises(IntegrityError):
            with reopened.engine.begin() as connection:
                connection.execute(
                    text("""
                        INSERT OR REPLACE INTO LeveragedEtfExecutorSnapshot (
                            executor_id, controller_id, pair_id, nav_cycle_id, schema_version,
                            state, snapshot_json, snapshot_hash, last_journal_sequence,
                            created_at_utc, updated_at_utc
                        ) VALUES ('executor-1', 'controller-1', 'sndk_snxx',
                                  'xnys-2026-07-17', 1, 'CREATED', 'replaced', :hash_a,
                                  0, :created_at, :updated_at)
                        """),
                    {"hash_a": HASH_A, "created_at": CREATED_AT, "updated_at": UPDATED_AT},
                )

        with reopened.engine.connect() as connection:
            assert (
                connection.execute(
                    text("SELECT snapshot_json FROM LeveragedEtfExecutorSnapshot " "WHERE executor_id = 'executor-1'")
                ).scalar_one()
                == "original"
            )
    finally:
        reopened.engine.dispose()


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
