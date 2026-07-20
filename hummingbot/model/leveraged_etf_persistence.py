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


def _sha256_check(column_name: str) -> str:
    return (
        f"typeof({column_name}) = 'text' "
        f"AND length({column_name}) = 64 "
        f"AND length(CAST({column_name} AS BLOB)) = 64 "
        f"AND {column_name} = lower({column_name}) "
        f"AND {column_name} NOT GLOB '*[^0-9a-f]*'"
    )


def _canonical_nonnegative_decimal_check(column_name: str) -> str:
    return f"""
        typeof({column_name}) = 'text'
        AND length({column_name}) = length(CAST({column_name} AS BLOB))
        AND (
            {column_name} = '0' OR (
                length({column_name}) > 0
                AND {column_name} NOT GLOB '*[^0-9.]*'
                AND length({column_name}) - length(replace({column_name}, '.', '')) <= 1
                AND (
                    (
                        instr({column_name}, '.') = 0
                        AND substr({column_name}, 1, 1) GLOB '[1-9]'
                    ) OR (
                        instr({column_name}, '.') > 0
                        AND instr({column_name}, '.') < length({column_name})
                        AND (
                            substr({column_name}, 1, instr({column_name}, '.') - 1) = '0'
                            OR substr({column_name}, 1, 1) GLOB '[1-9]'
                        )
                        AND substr({column_name}, instr({column_name}, '.') + 1)
                            NOT GLOB '*[^0-9]*'
                        AND substr({column_name}, -1, 1) GLOB '[1-9]'
                    )
                )
            )
        )
    """


def _canonical_positive_decimal_check(column_name: str) -> str:
    return f"({column_name} <> '0') AND ({_canonical_nonnegative_decimal_check(column_name)})"


class LeveragedEtfExecutorSnapshot(HummingbotBase):
    __tablename__ = "LeveragedEtfExecutorSnapshot"
    __table_args__ = (
        CheckConstraint("schema_version = 1", name="ck_lepf_snapshot_schema_version"),
        CheckConstraint("last_journal_sequence >= 0", name="ck_lepf_snapshot_sequence"),
        CheckConstraint(_sha256_check("snapshot_hash"), name="ck_lepf_snapshot_hash"),
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
        CheckConstraint(_sha256_check("payload_hash"), name="ck_lepf_journal_hash"),
        CheckConstraint(
            "exchange_trade_id IS NULL OR " "(connector_name IS NOT NULL AND trading_pair IS NOT NULL)",
            name="ck_lepf_journal_trade_identity",
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
        CheckConstraint(
            _canonical_nonnegative_decimal_check("quantity"),
            name="ck_lepf_reservation_quantity",
        ),
        CheckConstraint("leverage >= 1", name="ck_lepf_reservation_leverage"),
        CheckConstraint(
            _canonical_positive_decimal_check("notional_cap"),
            name="ck_lepf_reservation_notional_cap",
        ),
        CheckConstraint(_sha256_check("payload_hash"), name="ck_lepf_reservation_hash"),
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
            f"evidence_hash IS NULL OR ({_sha256_check('evidence_hash')})",
            name="ck_lepf_anchor_evidence_hash",
        ),
        CheckConstraint(_sha256_check("payload_hash"), name="ck_lepf_anchor_payload_hash"),
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
        CheckConstraint(_sha256_check("evidence_hash"), name="ck_lepf_anchor_observation_hash"),
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
    "lepf_snapshot_identity_insert": """
        CREATE TRIGGER IF NOT EXISTS lepf_snapshot_identity_insert
        BEFORE INSERT ON LeveragedEtfExecutorSnapshot
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1 FROM LeveragedEtfExecutorSnapshot
            WHERE executor_id = NEW.executor_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfExecutorSnapshot identity already exists');
        END
    """,
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
    "lepf_journal_identity_insert": """
        CREATE TRIGGER IF NOT EXISTS lepf_journal_identity_insert
        BEFORE INSERT ON LeveragedEtfJournalEvent
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1 FROM LeveragedEtfJournalEvent
            WHERE event_id = NEW.event_id
               OR (executor_id = NEW.executor_id AND sequence = NEW.sequence)
               OR (
                    NEW.exchange_trade_id IS NOT NULL
                    AND connector_name = NEW.connector_name
                    AND trading_pair = NEW.trading_pair
                    AND exchange_trade_id = NEW.exchange_trade_id
               )
        )
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfJournalEvent identity already exists');
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
    "lepf_reservation_identity_insert": """
        CREATE TRIGGER IF NOT EXISTS lepf_reservation_identity_insert
        BEFORE INSERT ON LeveragedEtfStrategyReservation
        FOR EACH ROW
        WHEN EXISTS (
            SELECT 1 FROM LeveragedEtfStrategyReservation
            WHERE reservation_id = NEW.reservation_id
               OR reservation_key = NEW.reservation_key
               OR (
                    NEW.released_at_utc IS NULL
                    AND released_at_utc IS NULL
                    AND executor_id = NEW.executor_id
                    AND connector_name = NEW.connector_name
                    AND trading_pair = NEW.trading_pair
                    AND leg = NEW.leg
               )
        )
        BEGIN
            SELECT RAISE(ABORT, 'LeveragedEtfStrategyReservation identity already exists');
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
    "lepf_anchor_observation_identity_insert": """
        CREATE TRIGGER IF NOT EXISTS lepf_anchor_observation_identity_insert
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
    "lepf_anchor_identity_insert": """
        CREATE TRIGGER IF NOT EXISTS lepf_anchor_identity_insert
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


def _has_redundant_outer_parentheses(expression: str) -> bool:
    if len(expression) < 2 or expression[0] != "(" or expression[-1] != ")":
        return False

    depth = 0
    in_string = False
    cursor = 0
    while cursor < len(expression):
        character = expression[cursor]
        if in_string:
            if character == "'":
                if cursor + 1 < len(expression) and expression[cursor + 1] == "'":
                    cursor += 2
                    continue
                in_string = False
        elif character == "'":
            in_string = True
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0 and cursor != len(expression) - 1:
                return False
        cursor += 1
    return depth == 0 and not in_string


def _normalize_sql(expression: object) -> str:
    """Canonicalize SQLite syntax while preserving string-literal semantics."""

    source = str(expression)
    normalized = []
    cursor = 0
    while cursor < len(source):
        character = source[cursor]
        if character.isspace():
            cursor += 1
            continue
        if character == "'":
            literal = [character]
            cursor += 1
            while cursor < len(source):
                literal.append(source[cursor])
                if source[cursor] == "'":
                    if cursor + 1 < len(source) and source[cursor + 1] == "'":
                        literal.append(source[cursor + 1])
                        cursor += 2
                        continue
                    cursor += 1
                    break
                cursor += 1
            normalized.extend(literal)
            continue
        if character in ('"', "`", "["):
            closing_quote = "]" if character == "[" else character
            cursor += 1
            identifier = []
            while cursor < len(source):
                if source[cursor] == closing_quote:
                    if cursor + 1 < len(source) and source[cursor + 1] == closing_quote:
                        identifier.append(closing_quote)
                        cursor += 2
                        continue
                    cursor += 1
                    break
                identifier.append(source[cursor])
                cursor += 1
            normalized.extend("".join(identifier).lower())
            continue
        if source.startswith("==", cursor):
            normalized.append("=")
            cursor += 2
            continue
        if source.startswith("!=", cursor):
            normalized.append("<>")
            cursor += 2
            continue
        normalized.append(character.lower())
        cursor += 1

    normalized_sql = "".join(normalized)
    while _has_redundant_outer_parentheses(normalized_sql):
        normalized_sql = normalized_sql[1:-1]
    return normalized_sql


def _normalize_trigger_sql(expression: object) -> str:
    return _normalize_sql(expression).replace("ifnotexists", "").rstrip(";")


def _normalize_foreign_key_options(options: Mapping[str, object]) -> Tuple[Tuple[str, str], ...]:
    normalized = []
    for name, value in options.items():
        if value is None:
            continue
        normalized_name = str(name).lower()
        normalized_value = str(value).upper()
        if normalized_name in ("ondelete", "onupdate") and normalized_value == "NO ACTION":
            continue
        if normalized_name == "deferrable" and normalized_value in ("0", "FALSE", "NOT DEFERRABLE"):
            continue
        if normalized_name == "match" and normalized_value == "SIMPLE":
            continue
        normalized.append((normalized_name, normalized_value))
    return tuple(sorted(normalized))


def _find_unquoted_sql_keyword(source: str, keyword: str, start: int) -> int:
    upper_keyword = keyword.upper()
    cursor = start
    while cursor < len(source):
        character = source[cursor]
        if character in ("'", '"', "`", "["):
            closing_quote = "]" if character == "[" else character
            cursor += 1
            while cursor < len(source):
                if source[cursor] == closing_quote:
                    if cursor + 1 < len(source) and source[cursor + 1] == closing_quote:
                        cursor += 2
                        continue
                    cursor += 1
                    break
                cursor += 1
            continue
        candidate = source[cursor : cursor + len(keyword)]
        preceding = source[cursor - 1] if cursor > 0 else ""
        following_at = cursor + len(keyword)
        following = source[following_at] if following_at < len(source) else ""
        if (
            candidate.upper() == upper_keyword
            and not (preceding.isalnum() or preceding == "_")
            and not (following.isalnum() or following == "_")
        ):
            return cursor
        cursor += 1
    return -1


def _extract_parenthesized_sql(source: str, opening_at: int) -> Tuple[str, int]:
    depth = 0
    string_quote = None
    cursor = opening_at
    while cursor < len(source):
        character = source[cursor]
        if string_quote is not None:
            if character == string_quote:
                if cursor + 1 < len(source) and source[cursor + 1] == string_quote:
                    cursor += 2
                    continue
                string_quote = None
        elif character in ("'", '"', "`"):
            string_quote = character
        elif character == "[":
            string_quote = "]"
        elif character == "(":
            depth += 1
        elif character == ")":
            depth -= 1
            if depth == 0:
                return source[opening_at + 1 : cursor], cursor + 1
        cursor += 1
    raise RuntimeError("malformed SQLite CHECK constraint")


def _sqlite_check_expressions(connection: Connection, table_name: str) -> Tuple[str, ...]:
    """Return every CHECK expression, including unnamed and nested constraints."""

    create_sql = connection.execute(
        text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :table_name"),
        {"table_name": table_name},
    ).scalar_one()
    expressions = []
    search_from = 0
    while True:
        check_at = _find_unquoted_sql_keyword(create_sql, "CHECK", search_from)
        if check_at < 0:
            break
        opening_at = create_sql.find("(", check_at + len("CHECK"))
        if opening_at < 0:
            raise RuntimeError(f"malformed SQLite CHECK constraint on {table_name}")
        expression, search_from = _extract_parenthesized_sql(create_sql, opening_at)
        expressions.append(expression)
    return tuple(expressions)


def _validate_table_shape(connection: Connection, table: Table) -> Tuple[str, ...]:
    inspector = inspect(connection)
    errors = []
    actual_columns: Dict[str, Mapping[str, object]] = {
        column["name"]: column for column in inspector.get_columns(table.name)
    }
    expected_column_names = {column.name for column in table.columns}
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
        if expected.server_default is None and actual.get("default") is not None:
            errors.append(f"{table.name}.{expected.name} has an unexpected default")
    for unexpected_column in sorted(set(actual_columns) - expected_column_names):
        errors.append(f"{table.name}.{unexpected_column} is an unexpected column")

    expected_primary_key = tuple(column.name.lower() for column in table.primary_key.columns)
    actual_primary_key = tuple(
        column_name.lower() for column_name in inspector.get_pk_constraint(table.name)["constrained_columns"]
    )
    if actual_primary_key != expected_primary_key:
        errors.append(f"{table.name} primary key {actual_primary_key} expected {expected_primary_key}")

    expected_checks = tuple(
        sorted(
            _normalize_sql(constraint.sqltext)
            for constraint in table.constraints
            if isinstance(constraint, CheckConstraint)
        )
    )
    actual_checks = tuple(
        sorted(_normalize_sql(sqltext) for sqltext in _sqlite_check_expressions(connection, table.name))
    )
    if actual_checks != expected_checks:
        errors.append(f"{table.name} CHECK constraint set is incompatible")

    expected_uniques = tuple(
        sorted(
            tuple(column.name.lower() for column in constraint.columns)
            for constraint in table.constraints
            if isinstance(constraint, UniqueConstraint)
        )
    )
    actual_uniques = tuple(
        sorted(
            tuple(column_name.lower() for column_name in constraint["column_names"])
            for constraint in inspector.get_unique_constraints(table.name)
        )
    )
    if actual_uniques != expected_uniques:
        errors.append(f"{table.name} UNIQUE constraint set is incompatible")

    expected_foreign_keys = tuple(
        sorted(
            (
                tuple(element.parent.name.lower() for element in constraint.elements),
                tuple(element.target_fullname.lower() for element in constraint.elements),
                _normalize_foreign_key_options(
                    {
                        "ondelete": constraint.ondelete,
                        "onupdate": constraint.onupdate,
                        "deferrable": constraint.deferrable,
                        "initially": constraint.initially,
                        "match": constraint.match,
                    }
                ),
            )
            for constraint in table.foreign_key_constraints
        )
    )
    actual_foreign_keys = tuple(
        sorted(
            (
                tuple(column_name.lower() for column_name in constraint["constrained_columns"]),
                tuple(
                    f"{constraint['referred_table']}.{column_name}".lower()
                    for column_name in constraint["referred_columns"]
                ),
                _normalize_foreign_key_options(constraint.get("options", {})),
            )
            for constraint in inspector.get_foreign_keys(table.name)
        )
    )
    if actual_foreign_keys != expected_foreign_keys:
        errors.append(f"{table.name} foreign key constraint set is incompatible")

    expected_indexes = {index.name: index for index in table.indexes}
    actual_indexes = {index["name"]: index for index in inspector.get_indexes(table.name)}
    explicit_index_names = {
        row[0]
        for row in connection.execute(
            text(
                "SELECT name FROM sqlite_master " "WHERE type = 'index' AND tbl_name = :table_name AND sql IS NOT NULL"
            ),
            {"table_name": table.name},
        )
    }
    for unexpected_index in sorted(explicit_index_names - set(expected_indexes)):
        errors.append(f"{table.name} index {unexpected_index} is unexpected")
    for index_name, index in expected_indexes.items():
        actual = actual_indexes.get(index_name)
        if index_name not in explicit_index_names or actual is None:
            errors.append(f"{table.name} index {index_name} is missing")
            continue
        expected_columns = tuple(column.name.lower() for column in index.columns)
        actual_index_columns = tuple(
            None if column_name is None else column_name.lower() for column_name in actual["column_names"]
        )
        if actual_index_columns != expected_columns:
            errors.append(f"{table.name} index {index_name} has incompatible columns")
        if bool(actual["unique"]) != bool(index.unique):
            errors.append(f"{table.name} index {index_name} has incompatible uniqueness")
        expected_where = index.dialect_options["sqlite"].get("where")
        actual_where = actual.get("dialect_options", {}).get("sqlite_where")
        expected_where_value = "" if expected_where is None else expected_where
        actual_where_value = "" if actual_where is None else actual_where
        if _normalize_sql(expected_where_value) != _normalize_sql(actual_where_value):
            errors.append(f"{table.name} index {index_name} has incompatible predicate")

    return tuple(errors)


def validate_leveraged_etf_persistence_schema(connection: Connection) -> None:
    """Validate the complete SQLite v1 schema without mutating it."""

    if connection.dialect.name != "sqlite":
        raise RuntimeError("leveraged ETF persistence supports SQLite only")

    errors = []
    actual_table_names = set(inspect(connection).get_table_names())
    for table in LEVERAGED_ETF_PERSISTENCE_TABLES:
        if table.name not in actual_table_names:
            errors.append(f"table {table.name} is missing")
        else:
            errors.extend(_validate_table_shape(connection, table))

    actual_triggers = {
        row[0]: (row[1], row[2])
        for row in connection.execute(text("SELECT name, tbl_name, sql FROM sqlite_master WHERE type = 'trigger'"))
    }
    expected_trigger_names = set(SQLITE_GUARD_DDL)
    managed_table_names = {table.name for table in LEVERAGED_ETF_PERSISTENCE_TABLES}
    for name, (table_name, _) in sorted(actual_triggers.items()):
        if table_name in managed_table_names and name not in expected_trigger_names:
            errors.append(f"SQLite trigger {name} on {table_name} is unexpected")
    for name, expected_sql in SQLITE_GUARD_DDL.items():
        actual_trigger = actual_triggers.get(name)
        if actual_trigger is None:
            errors.append(f"SQLite trigger {name} is missing")
        elif _normalize_trigger_sql(actual_trigger[1]) != _normalize_trigger_sql(expected_sql):
            errors.append(f"SQLite trigger {name} has incompatible definition")

    if errors:
        raise RuntimeError("incompatible leveraged ETF persistence schema: " + "; ".join(errors))


def ensure_leveraged_etf_persistence_schema(connection: Connection) -> None:
    """Create missing v1 SQLite objects, then validate their exact definitions."""

    if connection.dialect.name != "sqlite":
        raise RuntimeError("leveraged ETF persistence supports SQLite only")

    for table in LEVERAGED_ETF_PERSISTENCE_TABLES:
        table.create(bind=connection, checkfirst=True)
        for index in sorted(table.indexes, key=lambda candidate: candidate.name):
            index.create(bind=connection, checkfirst=True)

    for statement in SQLITE_GUARD_DDL.values():
        connection.execute(text(statement))

    validate_leveraged_etf_persistence_schema(connection)
