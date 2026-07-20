import logging
from enum import Enum
from os.path import join
from typing import TYPE_CHECKING, Optional

from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine.base import Engine
from sqlalchemy.engine.url import make_url
from sqlalchemy.orm import Query, Session, sessionmaker
from sqlalchemy.schema import DropConstraint, ForeignKeyConstraint, Table

from hummingbot import data_path
from hummingbot.logger.logger import HummingbotLogger
from hummingbot.model import get_declarative_base
from hummingbot.model.metadata import Metadata as LocalMetadata
from hummingbot.model.transaction_base import TransactionBase

if TYPE_CHECKING:
    from hummingbot.client.config.config_helpers import ClientConfigAdapter


class SQLConnectionType(Enum):
    TRADE_FILLS = 1


class DatabaseMigrationError(RuntimeError):
    pass


class SQLConnectionManager(TransactionBase):
    _scm_logger: Optional[HummingbotLogger] = None
    _scm_trade_fills_instance: Optional["SQLConnectionManager"] = None

    LOCAL_DB_VERSION_KEY = "local_db_version"
    LOCAL_DB_VERSION_VALUE = "20260719"

    @classmethod
    def logger(cls) -> HummingbotLogger:
        if cls._scm_logger is None:
            cls._scm_logger = logging.getLogger(__name__)
        return cls._scm_logger

    @classmethod
    def get_declarative_base(cls):
        return get_declarative_base()

    @classmethod
    def get_trade_fills_instance(
        cls, client_config_map: "ClientConfigAdapter", db_name: Optional[str] = None
    ) -> "SQLConnectionManager":
        if cls._scm_trade_fills_instance is None:
            cls._scm_trade_fills_instance = SQLConnectionManager(
                client_config_map, SQLConnectionType.TRADE_FILLS, db_name=db_name
            )
        elif cls.create_db_path(db_name=db_name) != cls._scm_trade_fills_instance.db_path:
            cls._scm_trade_fills_instance = SQLConnectionManager(
                client_config_map, SQLConnectionType.TRADE_FILLS, db_name=db_name
            )
        return cls._scm_trade_fills_instance

    @classmethod
    def create_db_path(cls, db_path: Optional[str] = None, db_name: Optional[str] = None) -> str:
        if db_path is not None:
            return db_path
        if db_name is not None:
            return join(data_path(), f"{db_name}.sqlite")
        else:
            return join(data_path(), "hummingbot_trades.sqlite")

    def __init__(
        self,
        client_config_map: "ClientConfigAdapter",
        connection_type: SQLConnectionType,
        db_path: Optional[str] = None,
        db_name: Optional[str] = None,
        called_from_migrator=False,
    ):
        db_path = self.create_db_path(db_path, db_name)
        self.db_path = db_path

        if connection_type is SQLConnectionType.TRADE_FILLS:
            database_url = client_config_map.db_mode.get_url(self.db_path)
            if make_url(database_url).get_backend_name() != "sqlite":
                raise DatabaseMigrationError(
                    "Only SQLite databases are supported by the leveraged ETF persistence schema"
                )
            self._engine: Engine = create_engine(database_url)
            self._metadata: MetaData = self.get_declarative_base().metadata

        self._session_cls = sessionmaker(bind=self._engine)

        if connection_type is SQLConnectionType.TRADE_FILLS and (not called_from_migrator):
            self.check_and_migrate_db(client_config_map)
            self._drop_foreign_key_constraints_for_supported_dialects()

    def _drop_foreign_key_constraints_for_supported_dialects(self):
        # SQLite keeps foreign-key enforcement disabled because fills may arrive before orders.
        # Preserve the existing non-SQLite behavior for legacy recorder tables while keeping
        # the leveraged-ETF persistence layer's explicitly managed constraints intact.
        from hummingbot.model.leveraged_etf_persistence import LEVERAGED_ETF_PERSISTENCE_TABLES

        managed_table_names = {table.name for table in LEVERAGED_ETF_PERSISTENCE_TABLES}
        with self._engine.begin() as conn:
            inspector = inspect(conn)

            for tname, fkcs in reversed(inspector.get_sorted_table_and_fkc_names()):
                if tname in managed_table_names:
                    continue
                if fkcs:
                    if not self._engine.dialect.supports_alter:
                        continue
                    for fkc in fkcs:
                        fk_constraint = ForeignKeyConstraint((), (), name=fkc)
                        Table(tname, MetaData(), fk_constraint)
                        conn.execute(DropConstraint(fk_constraint))

    @property
    def engine(self) -> Engine:
        return self._engine

    def get_new_session(self) -> Session:
        return self._session_cls()

    def get_local_db_version(self, session: Session):
        query: Query = session.query(LocalMetadata).filter(LocalMetadata.key == self.LOCAL_DB_VERSION_KEY)
        result: Optional[LocalMetadata] = query.one_or_none()
        return result

    def check_and_migrate_db(self, client_config_map: "ClientConfigAdapter"):
        from hummingbot.model.db_migration.migrator import Migrator
        from hummingbot.model.leveraged_etf_persistence import (
            ensure_leveraged_etf_persistence_schema,
            validate_leveraged_etf_persistence_schema,
        )

        existing_tables = {name for name in inspect(self._engine).get_table_names() if not name.startswith("sqlite_")}
        if not existing_tables:
            try:
                self._metadata.create_all(self._engine)
                with self._engine.begin() as connection:
                    ensure_leveraged_etf_persistence_schema(connection)
                    connection.execute(
                        text("INSERT INTO Metadata (key, value) VALUES (:key, :value)"),
                        {
                            "key": self.LOCAL_DB_VERSION_KEY,
                            "value": self.LOCAL_DB_VERSION_VALUE,
                        },
                    )
            except Exception as exception:
                self._engine.dispose()
                raise DatabaseMigrationError("fresh SQLite schema creation or validation failed") from exception
            return

        if LocalMetadata.__tablename__ not in existing_tables:
            self._engine.dispose()
            raise DatabaseMigrationError("nonempty unversioned SQLite database has no Metadata table")

        try:
            with self._engine.connect() as connection:
                version_rows = connection.execute(
                    text("SELECT value FROM Metadata WHERE key = :key"),
                    {"key": self.LOCAL_DB_VERSION_KEY},
                ).fetchall()
        except Exception as exception:
            self._engine.dispose()
            raise DatabaseMigrationError("SQLite Metadata table is incompatible") from exception
        if len(version_rows) != 1:
            self._engine.dispose()
            raise DatabaseMigrationError("nonempty unversioned SQLite database has no unique local_db_version row")

        try:
            current_version = int(version_rows[0][0])
        except (TypeError, ValueError) as exception:
            self._engine.dispose()
            raise DatabaseMigrationError("SQLite database version is malformed") from exception
        target_version = int(self.LOCAL_DB_VERSION_VALUE)

        if current_version == target_version:
            try:
                with self._engine.connect() as connection:
                    validate_leveraged_etf_persistence_schema(connection)
            except Exception as exception:
                self._engine.dispose()
                raise DatabaseMigrationError(
                    "current SQLite database version has an incompatible schema"
                ) from exception
            return

        if current_version > target_version:
            self._engine.dispose()
            raise DatabaseMigrationError(
                f"SQLite database version {current_version} is newer than supported version {target_version}"
            )

        migrator = Migrator()
        try:
            migrator.migration_path(current_version, target_version)
        except ValueError as exception:
            self._engine.dispose()
            raise DatabaseMigrationError(
                f"SQLite database version {current_version} has no recognized migration path"
            ) from exception

        was_migration_successful = migrator.migrate_db_to_version(
            client_config_map, self, current_version, target_version
        )
        if not was_migration_successful:
            self._engine.dispose()
            failure_detail = "" if migrator.last_error is None else f": {migrator.last_error}"
            raise DatabaseMigrationError(
                "SQLite database migration failed because of a lock, active/busy handle, "
                f"copy/replace error, or incompatible schema{failure_detail}"
            ) from migrator.last_error

        try:
            with self._engine.connect() as connection:
                validate_leveraged_etf_persistence_schema(connection)
        except Exception as exception:
            self._engine.dispose()
            raise DatabaseMigrationError("migrated SQLite schema validation failed") from exception
