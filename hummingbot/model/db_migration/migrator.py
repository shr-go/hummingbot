import logging
from datetime import datetime, timezone
from inspect import getmembers, isabstract, isclass
from pathlib import Path
from shutil import copyfile

from sqlalchemy import text

from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.model.db_migration.base_transformation import DatabaseTransformation
from hummingbot.model.sql_connection_manager import SQLConnectionManager, SQLConnectionType


class Migrator:
    @classmethod
    def _get_transformations(cls):
        import hummingbot.model.db_migration.transformations as transformations
        return [o for _, o in getmembers(transformations,
                                         predicate=lambda c: isclass(c) and
                                         issubclass(c, DatabaseTransformation) and
                                         not isabstract(c))]

    def __init__(self):
        self.transformations = [t(self) for t in self._get_transformations()]

    def migrate_db_to_version(self, client_config_map: ClientConfigAdapter, db_handle, from_version, to_version):
        original_db_path = db_handle.db_path
        original_db_name = Path(original_db_path).stem
        backup_db_path = original_db_path + ".backup_" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f")
        new_db_path = original_db_path + ".new"
        new_db_file = Path(new_db_path)
        new_db_file.unlink(missing_ok=True)

        # Release all managed SQLite handles before taking the file snapshots.
        db_handle.engine.dispose()
        copyfile(original_db_path, new_db_path)
        copyfile(original_db_path, backup_db_path)

        new_db_handle = None
        migration_successful = False
        try:
            new_db_handle = SQLConnectionManager(
                client_config_map,
                SQLConnectionType.TRADE_FILLS,
                db_path=new_db_path,
                db_name=original_db_name,
                called_from_migrator=True,
            )
            relevant_transformations = [
                transformation
                for transformation in self.transformations
                if transformation.does_apply_to_version(from_version, to_version)
            ]
            if relevant_transformations:
                logging.getLogger().info(f"Will run DB migration from {from_version} to {to_version}")

            for transformation in sorted(relevant_transformations):
                logging.getLogger().info(f"Applying {transformation.name} to DB...")
                new_db_handle = transformation.apply(new_db_handle)
                logging.getLogger().info(f"DONE with {transformation.name}")

            with new_db_handle.engine.begin() as connection:
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
            new_db_file.replace(original_db_path)
            migration_successful = True
        except Exception:
            logging.getLogger().error(
                "Unexpected error while checking and upgrading the local database.", exc_info=True
            )
        finally:
            try:
                if new_db_handle is not None:
                    new_db_handle.engine.dispose()
                if not migration_successful:
                    new_db_file.unlink(missing_ok=True)
                db_handle.__init__(
                    client_config_map,
                    SQLConnectionType.TRADE_FILLS,
                    db_path=original_db_path,
                    db_name=original_db_name,
                    called_from_migrator=True,
                )
            except Exception as e:
                logging.getLogger().error(f"Fatal error migrating DB {original_db_path}")
                raise e
        return migration_successful
