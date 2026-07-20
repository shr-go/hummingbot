import fcntl
import logging
import os
import shutil
import sqlite3
import tempfile
from datetime import datetime, timezone
from inspect import getmembers, isabstract, isclass
from pathlib import Path
from typing import List, Optional

from sqlalchemy import text

from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.model.db_migration.base_transformation import DatabaseTransformation
from hummingbot.model.sql_connection_manager import SQLConnectionManager, SQLConnectionType


def _sqlite_backup_database(source_connection: sqlite3.Connection, destination_path: Path) -> None:
    destination_connection = sqlite3.connect(str(destination_path))
    try:
        source_connection.backup(destination_connection)
        destination_connection.commit()
    finally:
        destination_connection.close()


def _copy_database_file(source_path: Path, destination_path: Path) -> None:
    shutil.copy2(source_path, destination_path)


def _atomic_replace(source_path: Path, destination_path: Path) -> None:
    os.replace(source_path, destination_path)


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlink_database_files(path: Path) -> None:
    first_error = None
    for candidate in (
        path,
        Path(f"{path}-journal"),
        Path(f"{path}-shm"),
        Path(f"{path}-wal"),
    ):
        try:
            candidate.unlink(missing_ok=True)
        except Exception as exception:
            if first_error is None:
                first_error = exception
    if first_error is not None:
        raise first_error


def _create_unique_file(parent: Path, prefix: str, suffix: str) -> Path:
    descriptor, raw_path = tempfile.mkstemp(dir=parent, prefix=prefix, suffix=suffix)
    os.close(descriptor)
    return Path(raw_path)


def _acquire_migration_lock(database_path: Path) -> int:
    lock_path = database_path.parent / f".{database_path.name}.migration.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        raise RuntimeError(f"SQLite migration lock is already held: {lock_path}")
    return descriptor


def _release_migration_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _quiesce_sqlite_database(database_path: Path) -> tuple[sqlite3.Connection, str]:
    connection = sqlite3.connect(str(database_path), timeout=0, isolation_level=None)
    try:
        connection.execute("PRAGMA busy_timeout = 0")
        original_journal_mode = str(connection.execute("PRAGMA journal_mode").fetchone()[0]).lower()
        connection.execute("PRAGMA locking_mode = EXCLUSIVE")
        connection.execute("BEGIN EXCLUSIVE")
        connection.execute("COMMIT")

        if original_journal_mode == "wal":
            busy, _, _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy:
                raise RuntimeError("SQLite WAL checkpoint is busy because an active handle exists")
            resulting_mode = str(connection.execute("PRAGMA journal_mode = DELETE").fetchone()[0]).lower()
            if resulting_mode != "delete":
                raise RuntimeError("SQLite WAL database could not enter a copy-safe journal mode")
        return connection, original_journal_mode
    except Exception:
        connection.close()
        raise


def _restore_journal_mode(connection: sqlite3.Connection, journal_mode: str) -> None:
    if journal_mode == "wal":
        resulting_mode = str(connection.execute("PRAGMA journal_mode = WAL").fetchone()[0]).lower()
        if resulting_mode != "wal":
            raise RuntimeError("SQLite WAL journal mode could not be restored")


def _set_database_journal_mode(database_path: Path, journal_mode: str) -> None:
    if journal_mode not in {"delete", "wal"}:
        raise RuntimeError(f"unsupported SQLite journal mode for migration: {journal_mode}")
    connection = sqlite3.connect(str(database_path), timeout=0, isolation_level=None)
    try:
        resulting_mode = str(connection.execute(f"PRAGMA journal_mode = {journal_mode.upper()}").fetchone()[0]).lower()
        if resulting_mode != journal_mode:
            raise RuntimeError(f"SQLite journal mode {journal_mode} could not be preserved")
        if journal_mode == "wal":
            busy, _, _ = connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if busy:
                raise RuntimeError("migrated SQLite WAL could not be checkpointed")
    finally:
        connection.close()


class Migrator:
    @classmethod
    def _get_transformations(cls):
        import hummingbot.model.db_migration.transformations as transformations

        return [
            o
            for _, o in getmembers(
                transformations,
                predicate=lambda c: isclass(c) and issubclass(c, DatabaseTransformation) and not isabstract(c),
            )
        ]

    def __init__(self):
        self.transformations = [t(self) for t in self._get_transformations()]
        self.last_error: Optional[Exception] = None

    def migration_path(self, original_version: int, target_version: int) -> List[DatabaseTransformation]:
        known_versions = {transformation.to_version for transformation in self.transformations}
        if original_version not in known_versions or target_version not in known_versions:
            raise ValueError(f"unrecognized migration endpoint {original_version} -> {target_version}")
        if original_version >= target_version:
            raise ValueError(f"migration source must precede target: {original_version} -> {target_version}")

        path = sorted(
            transformation
            for transformation in self.transformations
            if transformation.does_apply_to_version(original_version, target_version)
        )
        current_version = original_version
        for transformation in path:
            if transformation.from_version > 0 and transformation.from_version != current_version:
                raise ValueError(
                    f"migration gap before {transformation.name}: "
                    f"expected {transformation.from_version}, got {current_version}"
                )
            current_version = transformation.to_version
        if not path or current_version != target_version:
            raise ValueError(f"migration path does not reach target {target_version} from {original_version}")
        return path

    def migrate_db_to_version(self, client_config_map: ClientConfigAdapter, db_handle, from_version, to_version):
        from hummingbot.model.leveraged_etf_persistence import ensure_leveraged_etf_persistence_schema

        self.last_error = None
        original_db_file = Path(db_handle.db_path).resolve()
        original_db_name = original_db_file.stem
        parent_directory = original_db_file.parent
        migration_file = None
        backup_file = None
        backup_complete = False
        lock_descriptor = None
        source_connection = None
        original_journal_mode = None
        replaced = False
        new_db_handle = None
        migration_successful = False
        try:
            relevant_transformations = self.migration_path(from_version, to_version)
            lock_descriptor = _acquire_migration_lock(original_db_file)

            # The application-managed pool must be closed before the SQLite exclusive
            # precondition. Non-cooperating handles make that precondition fail closed.
            db_handle.engine.dispose()
            source_connection, original_journal_mode = _quiesce_sqlite_database(original_db_file)

            source_version_row = source_connection.execute(
                "SELECT value FROM Metadata WHERE key = ?",
                (SQLConnectionManager.LOCAL_DB_VERSION_KEY,),
            ).fetchone()
            if source_version_row is None:
                raise RuntimeError("local database version row is missing under migration lock")
            locked_source_version = int(source_version_row[0])
            if locked_source_version == to_version:
                migration_successful = True
                return migration_successful
            if locked_source_version != from_version:
                raise RuntimeError(
                    f"database version changed while acquiring migration lock: "
                    f"expected {from_version}, found {locked_source_version}"
                )

            migration_file = _create_unique_file(
                parent_directory,
                prefix=f".{original_db_file.name}.migration-",
                suffix=".sqlite",
            )
            backup_file = _create_unique_file(
                parent_directory,
                prefix=(
                    f"{original_db_file.name}.backup_" f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S-%f')}-"
                ),
                suffix=".sqlite",
            )

            _sqlite_backup_database(source_connection, migration_file)
            shutil.copystat(original_db_file, migration_file)
            _copy_database_file(migration_file, backup_file)
            _fsync_file(backup_file)
            _fsync_directory(parent_directory)
            backup_complete = True

            new_db_handle = SQLConnectionManager(
                client_config_map,
                SQLConnectionType.TRADE_FILLS,
                db_path=str(migration_file),
                db_name=original_db_name,
                called_from_migrator=True,
            )
            logging.getLogger().info(f"Will run DB migration from {from_version} to {to_version}")

            for transformation in sorted(relevant_transformations):
                logging.getLogger().info(f"Applying {transformation.name} to DB...")
                new_db_handle = transformation.apply(new_db_handle)
                logging.getLogger().info(f"DONE with {transformation.name}")

            new_db_handle._metadata.create_all(new_db_handle.engine)
            with new_db_handle.engine.begin() as connection:
                ensure_leveraged_etf_persistence_schema(connection)
                update_result = connection.execute(
                    text("UPDATE Metadata SET value = :version WHERE key = :version_key"),
                    {
                        "version": str(to_version),
                        "version_key": SQLConnectionManager.LOCAL_DB_VERSION_KEY,
                    },
                )
                if update_result.rowcount != 1:
                    raise RuntimeError("local database version row is missing")

            new_db_handle.engine.dispose()
            shutil.copystat(original_db_file, migration_file)
            _set_database_journal_mode(migration_file, original_journal_mode)
            _fsync_file(migration_file)
            _atomic_replace(migration_file, original_db_file)
            replaced = True
            _fsync_file(original_db_file)
            _fsync_directory(parent_directory)
            migration_successful = True
        except Exception as exception:
            self.last_error = exception
            logging.getLogger().error(
                "Unexpected error while checking and upgrading the local database.", exc_info=True
            )
        finally:
            cleanup_errors = []

            def attempt_cleanup(label, operation):
                try:
                    operation()
                except Exception as exception:
                    cleanup_errors.append((label, exception))
                    logging.getLogger().error(
                        f"Failed to {label} after SQLite migration attempt.",
                        exc_info=True,
                    )

            attempt_cleanup("dispose original database handle", db_handle.engine.dispose)
            if new_db_handle is not None:
                attempt_cleanup("dispose migration database handle", new_db_handle.engine.dispose)
            if source_connection is not None:
                if not replaced and original_journal_mode is not None:
                    attempt_cleanup(
                        "restore source journal mode",
                        lambda: _restore_journal_mode(source_connection, original_journal_mode),
                    )
                attempt_cleanup("close source database handle", source_connection.close)
            if lock_descriptor is not None:
                attempt_cleanup(
                    "release cross-process migration lock",
                    lambda: _release_migration_lock(lock_descriptor),
                )
            if migration_file is not None:
                attempt_cleanup(
                    "remove migration temporary files",
                    lambda: _unlink_database_files(migration_file),
                )
            if backup_file is not None and not backup_complete:
                attempt_cleanup(
                    "remove incomplete backup files",
                    lambda: _unlink_database_files(backup_file),
                )

            reconnect_error = None
            try:
                db_handle.__init__(
                    client_config_map,
                    SQLConnectionType.TRADE_FILLS,
                    db_path=str(original_db_file),
                    db_name=original_db_name,
                    called_from_migrator=True,
                )
            except Exception as exception:
                reconnect_error = exception
                logging.getLogger().error(
                    f"Fatal error reconnecting DB {original_db_file}",
                    exc_info=True,
                )

            if cleanup_errors:
                migration_successful = False
                if self.last_error is None:
                    self.last_error = cleanup_errors[0][1]
            if reconnect_error is not None:
                raise reconnect_error
        return migration_successful
