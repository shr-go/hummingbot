from typing import Dict, Mapping, Tuple

from sqlalchemy import (
    DDL,
    BigInteger,
    CheckConstraint,
    Column,
    ForeignKey,
    Index,
    Integer,
    Text,
    UniqueConstraint,
    event,
    inspect,
    text,
)
from sqlalchemy.engine import Connection
from sqlalchemy.schema import Table

from hummingbot.model import HummingbotBase


class LeveragedEtfExecutorSnapshot(HummingbotBase):
    __tablename__ = "LeveragedEtfExecutorSnapshot"
    __table_args__ = (
        CheckConstraint("schema_version = 1", name="ck_lepf_snapshot_schema_version"),
        CheckConstraint("last_journal_sequence >= 0", name="ck_lepf_snapshot_sequence"),
        CheckConstraint(
            "length(snapshot_hash) = 64 AND snapshot_hash = lower(snapshot_hash)",
            name="ck_lepf_snapshot_hash",
        ),
        Index("lepf_snapshot_controller_state", "controller_id", "state"),
        Index("lepf_snapshot_pair_cycle", "pair_id", "nav_cycle_id"),
        Index("lepf_snapshot_updated", "updated_at_utc"),
    )

    executor_id = Column(Text, primary_key=True, nullable=False)
    controller_id = Column(Text, nullable=False)
    pair_id = Column(Text, nullable=False)
    nav_cycle_id = Column(Text, nullable=False)
    schema_version = Column(Integer, nullable=False)
    state = Column(Text, nullable=False)
    snapshot_json = Column(Text, nullable=False)
    snapshot_hash = Column(Text, nullable=False)
    last_journal_sequence = Column(BigInteger, nullable=False)
    created_at_utc = Column(Text, nullable=False)
    updated_at_utc = Column(Text, nullable=False)


class LeveragedEtfJournalEvent(HummingbotBase):
    __tablename__ = "LeveragedEtfJournalEvent"
    __table_args__ = (
        CheckConstraint("sequence >= 1", name="ck_lepf_journal_sequence"),
        CheckConstraint(
            "length(payload_hash) = 64 AND payload_hash = lower(payload_hash)",
            name="ck_lepf_journal_hash",
        ),
        UniqueConstraint("executor_id", "sequence", name="uq_lepf_journal_executor_sequence"),
        Index("lepf_journal_executor_type_sequence", "executor_id", "event_type", "sequence"),
        Index("lepf_journal_idempotency_sequence", "idempotency_key", "sequence"),
        Index("lepf_journal_intent_sequence", "intent_id", "sequence"),
        Index(
            "lepf_journal_order_lookup",
            "connector_name",
            "client_order_id",
            "sequence",
        ),
        Index(
            "lepf_journal_trade_dedup",
            "connector_name",
            "trading_pair",
            "exchange_trade_id",
            unique=True,
            sqlite_where=text("exchange_trade_id IS NOT NULL"),
        ),
    )

    event_id = Column(Text, primary_key=True, nullable=False)
    executor_id = Column(
        Text,
        ForeignKey(
            "LeveragedEtfExecutorSnapshot.executor_id",
            name="fk_lepf_journal_executor",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    sequence = Column(BigInteger, nullable=False)
    event_type = Column(Text, nullable=False)
    intent_id = Column(Text, nullable=True)
    idempotency_key = Column(Text, nullable=True)
    connector_name = Column(Text, nullable=True)
    trading_pair = Column(Text, nullable=True)
    client_order_id = Column(Text, nullable=True)
    exchange_order_id = Column(Text, nullable=True)
    exchange_trade_id = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=False)
    payload_hash = Column(Text, nullable=False)
    created_at_utc = Column(Text, nullable=False)


class LeveragedEtfStrategyReservation(HummingbotBase):
    __tablename__ = "LeveragedEtfStrategyReservation"
    __table_args__ = (
        CheckConstraint("leg IN ('ETF', 'STOCK')", name="ck_lepf_reservation_leg"),
        CheckConstraint("length(quantity) > 0", name="ck_lepf_reservation_quantity"),
        CheckConstraint("leverage >= 1", name="ck_lepf_reservation_leverage"),
        CheckConstraint("length(notional_cap) > 0", name="ck_lepf_reservation_notional_cap"),
        CheckConstraint(
            "length(payload_hash) = 64 AND payload_hash = lower(payload_hash)",
            name="ck_lepf_reservation_hash",
        ),
        UniqueConstraint("reservation_key", name="uq_lepf_reservation_key"),
        Index(
            "lepf_reservation_active_leg",
            "executor_id",
            "connector_name",
            "trading_pair",
            "leg",
            unique=True,
            sqlite_where=text("released_at_utc IS NULL"),
        ),
        Index("lepf_reservation_pair_release", "connector_name", "trading_pair", "released_at_utc"),
    )

    reservation_id = Column(Text, primary_key=True, nullable=False)
    executor_id = Column(
        Text,
        ForeignKey(
            "LeveragedEtfExecutorSnapshot.executor_id",
            name="fk_lepf_reservation_executor",
            ondelete="RESTRICT",
        ),
        nullable=False,
    )
    reservation_key = Column(Text, nullable=False)
    connector_name = Column(Text, nullable=False)
    trading_pair = Column(Text, nullable=False)
    leg = Column(Text, nullable=False)
    quantity = Column(Text, nullable=False)
    leverage = Column(Integer, nullable=False)
    notional_cap = Column(Text, nullable=False)
    payload_json = Column(Text, nullable=False)
    payload_hash = Column(Text, nullable=False)
    created_at_utc = Column(Text, nullable=False)
    updated_at_utc = Column(Text, nullable=False)
    released_at_utc = Column(Text, nullable=True)


class LeveragedEtfAnchorState(HummingbotBase):
    __tablename__ = "LeveragedEtfAnchorState"
    __table_args__ = (
        CheckConstraint("schema_version = 1", name="ck_lepf_anchor_schema_version"),
        CheckConstraint("state_kind IN ('CHECKPOINT', 'FINALIZED')", name="ck_lepf_anchor_state_kind"),
        CheckConstraint("revision >= 1", name="ck_lepf_anchor_revision"),
        CheckConstraint(
            "(state_kind = 'CHECKPOINT' AND evidence_hash IS NULL) OR "
            "(state_kind = 'FINALIZED' AND evidence_hash IS NOT NULL)",
            name="ck_lepf_anchor_evidence_state",
        ),
        CheckConstraint(
            "state_kind = 'FINALIZED' OR official_close_utc IS NOT NULL",
            name="ck_lepf_anchor_official_close_state",
        ),
        CheckConstraint(
            "evidence_hash IS NULL OR " "(length(evidence_hash) = 64 AND evidence_hash = lower(evidence_hash))",
            name="ck_lepf_anchor_evidence_hash",
        ),
        CheckConstraint(
            "length(payload_hash) = 64 AND payload_hash = lower(payload_hash)",
            name="ck_lepf_anchor_payload_hash",
        ),
        UniqueConstraint("evidence_hash", name="uq_lepf_anchor_evidence_hash"),
        Index("lepf_anchor_deadline_state", "deadline_utc", "state_kind"),
        Index("lepf_anchor_session_date", "target_session_date"),
    )

    cycle_id = Column(Text, primary_key=True, nullable=False)
    schema_version = Column(Integer, nullable=False)
    state_kind = Column(Text, nullable=False)
    revision = Column(Integer, nullable=False)
    target_session_date = Column(Text, nullable=False)
    official_close_utc = Column(Text, nullable=True)
    deadline_utc = Column(Text, nullable=False)
    evidence_hash = Column(Text, nullable=True)
    payload_json = Column(Text, nullable=False)
    payload_hash = Column(Text, nullable=False)
    created_at_utc = Column(Text, nullable=False)
    updated_at_utc = Column(Text, nullable=False)


class LeveragedEtfAnchorRevisionObservation(HummingbotBase):
    __tablename__ = "LeveragedEtfAnchorRevisionObservation"
    __table_args__ = (
        CheckConstraint(
            "length(evidence_hash) = 64 AND evidence_hash = lower(evidence_hash)",
            name="ck_lepf_anchor_observation_hash",
        ),
        Index("lepf_anchor_observation_cycle_time", "cycle_id", "observed_at_utc"),
    )

    cycle_id = Column(
        Text,
        ForeignKey(
            "LeveragedEtfAnchorState.cycle_id",
            name="fk_lepf_anchor_observation_cycle",
            ondelete="RESTRICT",
        ),
        primary_key=True,
        nullable=False,
    )
    evidence_hash = Column(Text, primary_key=True, nullable=False)
    observed_at_utc = Column(Text, primary_key=True, nullable=False)


LEVERAGED_ETF_PERSISTENCE_TABLES: Tuple[Table, ...] = (
    LeveragedEtfExecutorSnapshot.__table__,
    LeveragedEtfJournalEvent.__table__,
    LeveragedEtfStrategyReservation.__table__,
    LeveragedEtfAnchorState.__table__,
    LeveragedEtfAnchorRevisionObservation.__table__,
)


SQLITE_GUARD_DDL: Mapping[str, str] = {
    "lepf_snapshot_executor_id_no_update": """
        CREATE TRIGGER IF NOT EXISTS lepf_snapshot_executor_id_no_update
        BEFORE UPDATE OF executor_id ON LeveragedEtfExecutorSnapshot
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfExecutorSnapshot executor_id is stable');
        END
    """,
    "lepf_snapshot_referenced_no_delete": """
        CREATE TRIGGER IF NOT EXISTS lepf_snapshot_referenced_no_delete
        BEFORE DELETE ON LeveragedEtfExecutorSnapshot
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1 FROM LeveragedEtfJournalEvent
            WHERE executor_id = OLD.executor_id
        ) OR EXISTS (
            SELECT 1 FROM LeveragedEtfStrategyReservation
            WHERE executor_id = OLD.executor_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfExecutorSnapshot is referenced');
        END
    """,
    "lepf_journal_executor_fk_insert": """
        CREATE TRIGGER IF NOT EXISTS lepf_journal_executor_fk_insert
        BEFORE INSERT ON LeveragedEtfJournalEvent
        FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1 FROM LeveragedEtfExecutorSnapshot
            WHERE executor_id = NEW.executor_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfJournalEvent executor_id does not exist');
        END
    """,
    "lepf_reservation_executor_fk_insert": """
        CREATE TRIGGER IF NOT EXISTS lepf_reservation_executor_fk_insert
        BEFORE INSERT ON LeveragedEtfStrategyReservation
        FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1 FROM LeveragedEtfExecutorSnapshot
            WHERE executor_id = NEW.executor_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfStrategyReservation executor_id does not exist');
        END
    """,
    "lepf_reservation_executor_fk_update": """
        CREATE TRIGGER IF NOT EXISTS lepf_reservation_executor_fk_update
        BEFORE UPDATE OF executor_id ON LeveragedEtfStrategyReservation
        FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1 FROM LeveragedEtfExecutorSnapshot
            WHERE executor_id = NEW.executor_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfStrategyReservation executor_id does not exist');
        END
    """,
    "lepf_reservation_stable_ids_no_update": """
        CREATE TRIGGER IF NOT EXISTS lepf_reservation_stable_ids_no_update
        BEFORE UPDATE OF reservation_id, reservation_key ON LeveragedEtfStrategyReservation
        FOR EACH ROW
        WHEN NEW.reservation_id <> OLD.reservation_id
          OR NEW.reservation_key <> OLD.reservation_key
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfStrategyReservation identifiers are stable');
        END
    """,
    "lepf_reservation_no_delete": """
        CREATE TRIGGER IF NOT EXISTS lepf_reservation_no_delete
        BEFORE DELETE ON LeveragedEtfStrategyReservation
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfStrategyReservation is durable');
        END
    """,
    "lepf_anchor_observation_cycle_fk_insert": """
        CREATE TRIGGER IF NOT EXISTS lepf_anchor_observation_cycle_fk_insert
        BEFORE INSERT ON LeveragedEtfAnchorRevisionObservation
        FOR EACH ROW
        WHEN NOT EXISTS (
            SELECT 1 FROM LeveragedEtfAnchorState
            WHERE cycle_id = NEW.cycle_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfAnchorRevisionObservation cycle_id does not exist');
        END
    """,
    "lepf_journal_no_update": """
        CREATE TRIGGER IF NOT EXISTS lepf_journal_no_update
        BEFORE UPDATE ON LeveragedEtfJournalEvent
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfJournalEvent is append-only');
        END
    """,
    "lepf_journal_no_delete": """
        CREATE TRIGGER IF NOT EXISTS lepf_journal_no_delete
        BEFORE DELETE ON LeveragedEtfJournalEvent
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfJournalEvent is append-only');
        END
    """,
    "lepf_anchor_observation_no_update": """
        CREATE TRIGGER IF NOT EXISTS lepf_anchor_observation_no_update
        BEFORE UPDATE ON LeveragedEtfAnchorRevisionObservation
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfAnchorRevisionObservation is append-only');
        END
    """,
    "lepf_anchor_observation_no_delete": """
        CREATE TRIGGER IF NOT EXISTS lepf_anchor_observation_no_delete
        BEFORE DELETE ON LeveragedEtfAnchorRevisionObservation
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfAnchorRevisionObservation is append-only');
        END
    """,
    "lepf_anchor_finalized_no_update": """
        CREATE TRIGGER IF NOT EXISTS lepf_anchor_finalized_no_update
        BEFORE UPDATE ON LeveragedEtfAnchorState
        FOR EACH ROW
        WHEN OLD.state_kind = 'FINALIZED'
        BEGIN
            SELECT RAISE(ABORT, 'finalized LeveragedEtfAnchorState is immutable');
        END
    """,
    "lepf_anchor_cycle_id_no_update": """
        CREATE TRIGGER IF NOT EXISTS lepf_anchor_cycle_id_no_update
        BEFORE UPDATE OF cycle_id ON LeveragedEtfAnchorState
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfAnchorState cycle_id is stable');
        END
    """,
    "lepf_anchor_state_no_delete": """
        CREATE TRIGGER IF NOT EXISTS lepf_anchor_state_no_delete
        BEFORE DELETE ON LeveragedEtfAnchorState
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfAnchorState is durable');
        END
    """,
}


for table in LEVERAGED_ETF_PERSISTENCE_TABLES:
    for statement in SQLITE_GUARD_DDL.values():
        if f" ON {table.name}" in statement:
            event.listen(table, "after_create", DDL(statement).execute_if(dialect="sqlite"))


def _normalize_sql(expression: object) -> str:
    return "".join(str(expression).replace('"', "").replace("`", "").lower().split())


def _validate_table_shape(connection: Connection, table: Table) -> Tuple[str, ...]:
    inspector = inspect(connection)
    errors = []
    actual_columns: Dict[str, Mapping[str, object]] = {
        column["name"]: column for column in inspector.get_columns(table.name)
    }
    for expected in table.columns:
        actual = actual_columns.get(expected.name)
        if actual is None:
            errors.append(f"{table.name}.{expected.name} is missing")
            continue
        if bool(actual["nullable"]) != bool(expected.nullable):
            errors.append(
                f"{table.name}.{expected.name} nullable={actual['nullable']} " f"expected {expected.nullable}"
            )
        if actual["type"]._type_affinity is not expected.type._type_affinity:
            errors.append(
                f"{table.name}.{expected.name} type affinity "
                f"{actual['type']._type_affinity.__name__} expected {expected.type._type_affinity.__name__}"
            )

    expected_primary_key = tuple(column.name for column in table.primary_key.columns)
    actual_primary_key = tuple(inspector.get_pk_constraint(table.name)["constrained_columns"])
    if actual_primary_key != expected_primary_key:
        errors.append(f"{table.name} primary key {actual_primary_key} expected {expected_primary_key}")

    expected_checks = {
        constraint.name: _normalize_sql(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint) and constraint.name is not None
    }
    actual_checks = {
        constraint["name"]: _normalize_sql(constraint["sqltext"])
        for constraint in inspector.get_check_constraints(table.name)
        if constraint.get("name") is not None
    }
    for name, sqltext in expected_checks.items():
        if name not in actual_checks:
            errors.append(f"{table.name} check constraint {name} is missing")
        elif actual_checks[name] != sqltext:
            errors.append(f"{table.name} check constraint {name} has incompatible SQL")

    expected_uniques = {
        constraint.name: tuple(column.name for column in constraint.columns)
        for constraint in table.constraints
        if isinstance(constraint, UniqueConstraint) and constraint.name is not None
    }
    actual_uniques = {
        constraint["name"]: tuple(constraint["column_names"])
        for constraint in inspector.get_unique_constraints(table.name)
        if constraint.get("name") is not None
    }
    for name, columns in expected_uniques.items():
        if name not in actual_uniques:
            errors.append(f"{table.name} unique constraint {name} is missing")
        elif actual_uniques[name] != columns:
            errors.append(f"{table.name} unique constraint {name} has incompatible columns")

    expected_foreign_keys = {
        constraint.name: (
            tuple(element.parent.name for element in constraint.elements),
            tuple(element.target_fullname for element in constraint.elements),
        )
        for constraint in table.foreign_key_constraints
        if constraint.name is not None
    }
    actual_foreign_keys = {
        constraint["name"]: (
            tuple(constraint["constrained_columns"]),
            tuple(f"{constraint['referred_table']}.{column}" for column in constraint["referred_columns"]),
        )
        for constraint in inspector.get_foreign_keys(table.name)
        if constraint.get("name") is not None
    }
    for name, shape in expected_foreign_keys.items():
        if name not in actual_foreign_keys:
            errors.append(f"{table.name} foreign key {name} is missing")
        elif actual_foreign_keys[name] != shape:
            errors.append(f"{table.name} foreign key {name} has incompatible columns")

    actual_indexes = {index["name"]: index for index in inspector.get_indexes(table.name)}
    for index in table.indexes:
        actual = actual_indexes.get(index.name)
        if actual is None:
            errors.append(f"{table.name} index {index.name} is missing")
            continue
        expected_columns = tuple(column.name for column in index.columns)
        if tuple(actual["column_names"]) != expected_columns:
            errors.append(f"{table.name} index {index.name} has incompatible columns")
        if bool(actual["unique"]) != bool(index.unique):
            errors.append(f"{table.name} index {index.name} has incompatible uniqueness")
        if connection.dialect.name == "sqlite":
            expected_where = index.dialect_options["sqlite"].get("where")
            actual_where = actual.get("dialect_options", {}).get("sqlite_where")
            expected_where_value = "" if expected_where is None else expected_where
            actual_where_value = "" if actual_where is None else actual_where
            if _normalize_sql(expected_where_value) != _normalize_sql(actual_where_value):
                errors.append(f"{table.name} index {index.name} has incompatible predicate")

    return tuple(errors)


def ensure_leveraged_etf_persistence_schema(connection: Connection) -> None:
    """Create missing v1 tables/indexes/guards and reject incompatible partial tables."""

    for table in LEVERAGED_ETF_PERSISTENCE_TABLES:
        table.create(bind=connection, checkfirst=True)
        for index in sorted(table.indexes, key=lambda candidate: candidate.name):
            index.create(bind=connection, checkfirst=True)

    if connection.dialect.name == "sqlite":
        for statement in SQLITE_GUARD_DDL.values():
            connection.execute(text(statement))

    errors = []
    for table in LEVERAGED_ETF_PERSISTENCE_TABLES:
        errors.extend(_validate_table_shape(connection, table))

    if connection.dialect.name == "sqlite":
        actual_triggers = {
            row[0] for row in connection.execute(text("SELECT name FROM sqlite_master WHERE type = 'trigger'"))
        }
        for missing in sorted(set(SQLITE_GUARD_DDL) - actual_triggers):
            errors.append(f"SQLite trigger {missing} is missing")

    if errors:
        raise RuntimeError("incompatible leveraged ETF persistence schema: " + "; ".join(errors))
