import sqlite3
from collections import Counter
from typing import Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple

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
from sqlalchemy.schema import CreateIndex, CreateTable, Table

from hummingbot.model import HummingbotBase


def _sha256_check(column_name: str) -> str:
    return (
        f"typeof({column_name}) = 'text' "
        f"AND instr({column_name}, char(0)) = 0 "
        f"AND length({column_name}) = 64 "
        f"AND {column_name} = lower({column_name}) "
        f"AND {column_name} NOT GLOB '*[^0123456789abcdef]*'"
    )


def _canonical_nonnegative_decimal_check(column_name: str) -> str:
    return f"""
        typeof({column_name}) = 'text'
        AND instr({column_name}, char(0)) = 0
        AND (
            {column_name} = '0' OR (
                length({column_name}) > 0
                AND {column_name} NOT GLOB '*[^0123456789.]*'
                AND length({column_name}) - length(replace({column_name}, '.', '')) <= 1
                AND (
                    (
                        instr({column_name}, '.') = 0
                        AND substr({column_name}, 1, 1) GLOB '[123456789]'
                    ) OR (
                        instr({column_name}, '.') > 0
                        AND instr({column_name}, '.') < length({column_name})
                        AND (
                            substr({column_name}, 1, instr({column_name}, '.') - 1) = '0'
                            OR substr({column_name}, 1, 1) GLOB '[123456789]'
                        )
                        AND substr({column_name}, instr({column_name}, '.') + 1)
                            NOT GLOB '*[^0123456789]*'
                        AND substr({column_name}, -1, 1) GLOB '[123456789]'
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


class _SQLiteToken(NamedTuple):
    kind: str
    value: str


CanonicalSql = Tuple[Tuple[str, str], ...]


def _sqlite_tokens(expression: object) -> Tuple[_SQLiteToken, ...]:
    """Tokenize the SQLite DDL subset used by the persistence schema."""

    source = str(expression)
    tokens: List[_SQLiteToken] = []
    cursor = 0
    while cursor < len(source):
        character = source[cursor]
        if character.isspace():
            cursor += 1
            continue
        if source.startswith("--", cursor):
            newline_at = source.find("\n", cursor + 2)
            cursor = len(source) if newline_at < 0 else newline_at + 1
            continue
        if source.startswith("/*", cursor):
            closing_at = source.find("*/", cursor + 2)
            if closing_at < 0:
                raise RuntimeError("unterminated SQLite block comment")
            cursor = closing_at + 2
            continue
        if character == "'":
            cursor += 1
            literal: List[str] = []
            while cursor < len(source):
                if source[cursor] == "'":
                    if cursor + 1 < len(source) and source[cursor + 1] == "'":
                        literal.append("'")
                        cursor += 2
                        continue
                    cursor += 1
                    break
                literal.append(source[cursor])
                cursor += 1
            else:
                raise RuntimeError("unterminated SQLite string literal")
            tokens.append(_SQLiteToken("string", "".join(literal)))
            continue
        if character in ('"', "`", "["):
            closing_quote = "]" if character == "[" else character
            cursor += 1
            identifier: List[str] = []
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
            else:
                raise RuntimeError("unterminated SQLite quoted identifier")
            tokens.append(_SQLiteToken("quoted_identifier", "".join(identifier).lower()))
            continue
        if character.isalpha() or character == "_" or ord(character) >= 128:
            start = cursor
            cursor += 1
            while cursor < len(source):
                candidate = source[cursor]
                if not (candidate.isalnum() or candidate in ("_", "$") or ord(candidate) >= 128):
                    break
                cursor += 1
            tokens.append(_SQLiteToken("word", source[start:cursor].lower()))
            continue
        if character.isdigit():
            start = cursor
            cursor += 1
            while cursor < len(source) and (source[cursor].isalnum() or source[cursor] in (".", "_")):
                cursor += 1
            tokens.append(_SQLiteToken("number", source[start:cursor].lower()))
            continue
        if source.startswith("==", cursor):
            tokens.append(_SQLiteToken("symbol", "="))
            cursor += 2
            continue
        if source.startswith("!=", cursor):
            tokens.append(_SQLiteToken("symbol", "<>"))
            cursor += 2
            continue
        matched_operator = next(
            (
                operator
                for operator in ("->>", "<=", ">=", "<>", "||", "<<", ">>", "->")
                if source.startswith(operator, cursor)
            ),
            None,
        )
        if matched_operator is not None:
            tokens.append(_SQLiteToken("symbol", matched_operator.lower()))
            cursor += len(matched_operator)
            continue
        tokens.append(_SQLiteToken("symbol", character.lower()))
        cursor += 1
    return tuple(tokens)


def _token_is_identifier(token: _SQLiteToken) -> bool:
    return token.kind in ("word", "quoted_identifier")


def _has_redundant_outer_parentheses(tokens: Sequence[Tuple[str, str]]) -> bool:
    if len(tokens) < 2 or tokens[0] != ("symbol", "(") or tokens[-1] != ("symbol", ")"):
        return False
    depth = 0
    for index, token in enumerate(tokens):
        if token == ("symbol", "("):
            depth += 1
        elif token == ("symbol", ")"):
            depth -= 1
            if depth < 0 or (depth == 0 and index != len(tokens) - 1):
                return False
    return depth == 0


def _canonicalize_sql_tokens(tokens: Sequence[_SQLiteToken]) -> CanonicalSql:
    canonical: List[Tuple[str, str]] = []
    for token in tokens:
        if token.kind in ("word", "quoted_identifier"):
            canonical.append(("identifier", token.value.lower()))
        else:
            canonical.append((token.kind, token.value))
    while _has_redundant_outer_parentheses(canonical):
        canonical = canonical[1:-1]
    return tuple(canonical)


def _is_atomic_sql_expression(tokens: Sequence[_SQLiteToken]) -> bool:
    canonical = _canonicalize_sql_tokens(tokens)
    if not canonical:
        return False
    if len(canonical) == 1:
        return canonical[0][0] in ("identifier", "number", "string")
    if len(canonical) >= 3 and all(
        token[0] == "identifier" if index % 2 == 0 else token == ("symbol", ".")
        for index, token in enumerate(canonical)
    ):
        return True
    function_at = 1
    while (
        function_at + 1 < len(canonical)
        and canonical[function_at] == ("symbol", ".")
        and canonical[function_at + 1][0] == "identifier"
    ):
        function_at += 2
    if canonical[0][0] != "identifier" or canonical[function_at] != ("symbol", "("):
        return False
    depth = 0
    for index, token in enumerate(canonical[function_at:], start=function_at):
        if token == ("symbol", "("):
            depth += 1
        elif token == ("symbol", ")"):
            depth -= 1
            if depth < 0 or (depth == 0 and index != len(canonical) - 1):
                return False
    return depth == 0


def _strip_redundant_operand_parentheses(tokens: Sequence[_SQLiteToken]) -> Tuple[_SQLiteToken, ...]:
    """Remove grouping around atomic operands without changing operator precedence."""

    normalized: List[_SQLiteToken] = []
    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        if token != _SQLiteToken("symbol", "("):
            normalized.append(token)
            cursor += 1
            continue
        inner, after_group = _extract_parenthesized_tokens(tokens, cursor)
        normalized_inner = _strip_redundant_operand_parentheses(inner)
        follows_identifier = bool(normalized and _token_is_identifier(normalized[-1]))
        if _is_atomic_sql_expression(normalized_inner) and not follows_identifier:
            normalized.extend(normalized_inner)
        else:
            normalized.append(_SQLiteToken("symbol", "("))
            normalized.extend(normalized_inner)
            normalized.append(_SQLiteToken("symbol", ")"))
        cursor = after_group
    return tuple(normalized)


def _canonicalize_sql_expression_tokens(tokens: Sequence[_SQLiteToken]) -> CanonicalSql:
    return _canonicalize_sql_tokens(_strip_redundant_operand_parentheses(tokens))


def _normalize_sql(expression: object) -> CanonicalSql:
    """Canonicalize SQLite syntax while preserving literal and identifier boundaries."""

    return _canonicalize_sql_expression_tokens(_sqlite_tokens(expression))


def _sqlite_trigger_target_span(tokens: Sequence[_SQLiteToken]) -> Tuple[int, int, int]:
    begin_at = next(
        (index for index, token in enumerate(tokens) if token == _SQLiteToken("word", "begin")),
        len(tokens),
    )
    on_at = next(
        (index for index, token in enumerate(tokens[:begin_at]) if token == _SQLiteToken("word", "on")),
        -1,
    )
    if on_at < 0 or on_at + 1 >= begin_at or not _token_is_identifier(tokens[on_at + 1]):
        raise RuntimeError("malformed SQLite trigger target")
    target_at = on_at + 1
    if (
        target_at + 2 < begin_at
        and tokens[target_at].value.lower() == "main"
        and tokens[target_at + 1] == _SQLiteToken("symbol", ".")
        and _token_is_identifier(tokens[target_at + 2])
    ):
        target_at += 2
    return on_at, target_at, target_at + 1


def _strip_trigger_main_schema_qualifiers(tokens: Sequence[_SQLiteToken]) -> Tuple[_SQLiteToken, ...]:
    managed_table_names = {table.name.lower() for table in LEVERAGED_ETF_PERSISTENCE_TABLES}
    normalized: List[_SQLiteToken] = []
    cursor = 0
    while cursor < len(tokens):
        token = tokens[cursor]
        normalized.append(token)
        if (
            token.kind == "word"
            and token.value in ("from", "join")
            and cursor + 3 < len(tokens)
            and _token_is_identifier(tokens[cursor + 1])
            and tokens[cursor + 1].value.lower() == "main"
            and tokens[cursor + 2] == _SQLiteToken("symbol", ".")
            and _token_is_identifier(tokens[cursor + 3])
            and tokens[cursor + 3].value.lower() in managed_table_names
        ):
            normalized.append(tokens[cursor + 3])
            cursor += 4
            continue
        cursor += 1
    return tuple(normalized)


def _normalize_trigger_sql(expression: object) -> CanonicalSql:
    tokens = list(_sqlite_tokens(expression))
    while tokens and tokens[-1] == _SQLiteToken("symbol", ";"):
        tokens.pop()
    if not tokens or tokens[0] != _SQLiteToken("word", "create"):
        raise RuntimeError("malformed SQLite CREATE TRIGGER statement")
    cursor = 1
    if (
        cursor < len(tokens)
        and tokens[cursor].kind == "word"
        and tokens[cursor].value
        in (
            "temp",
            "temporary",
        )
    ):
        cursor += 1
    if cursor >= len(tokens) or tokens[cursor] != _SQLiteToken("word", "trigger"):
        raise RuntimeError("malformed SQLite CREATE TRIGGER statement")
    optional_clause_at = cursor + 1
    if tuple(tokens[optional_clause_at : optional_clause_at + 3]) == (
        _SQLiteToken("word", "if"),
        _SQLiteToken("word", "not"),
        _SQLiteToken("word", "exists"),
    ):
        del tokens[optional_clause_at : optional_clause_at + 3]
    on_at, target_at, after_target = _sqlite_trigger_target_span(tokens)
    if tuple(tokens[after_target : after_target + 3]) == (
        _SQLiteToken("word", "for"),
        _SQLiteToken("word", "each"),
        _SQLiteToken("word", "row"),
    ):
        del tokens[after_target : after_target + 3]
    if target_at != on_at + 1:
        del tokens[on_at + 1 : target_at]
    tokens = list(_strip_trigger_main_schema_qualifiers(tokens))
    return _canonicalize_sql_tokens(tokens)


def _sqlite_trigger_target(expression: object) -> str:
    tokens = _sqlite_tokens(expression)
    _, target_at, _ = _sqlite_trigger_target_span(tokens)
    return tokens[target_at].value.lower()


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


def _extract_parenthesized_tokens(
    tokens: Sequence[_SQLiteToken], opening_at: int
) -> Tuple[Tuple[_SQLiteToken, ...], int]:
    if opening_at >= len(tokens) or tokens[opening_at] != _SQLiteToken("symbol", "("):
        raise RuntimeError("malformed SQLite parenthesized expression")
    depth = 0
    cursor = opening_at
    while cursor < len(tokens):
        token = tokens[cursor]
        if token == _SQLiteToken("symbol", "("):
            depth += 1
        elif token == _SQLiteToken("symbol", ")"):
            depth -= 1
            if depth == 0:
                return tuple(tokens[opening_at + 1 : cursor]), cursor + 1
            if depth < 0:
                break
        cursor += 1
    raise RuntimeError("malformed SQLite parenthesized expression")


def _split_top_level_tokens(tokens: Sequence[_SQLiteToken]) -> Tuple[Tuple[_SQLiteToken, ...], ...]:
    segments: List[Tuple[_SQLiteToken, ...]] = []
    start = 0
    depth = 0
    for index, token in enumerate(tokens):
        if token == _SQLiteToken("symbol", "("):
            depth += 1
        elif token == _SQLiteToken("symbol", ")"):
            depth -= 1
            if depth < 0:
                raise RuntimeError("malformed SQLite expression list")
        elif token == _SQLiteToken("symbol", ",") and depth == 0:
            segments.append(tuple(tokens[start:index]))
            start = index + 1
    if depth != 0:
        raise RuntimeError("malformed SQLite expression list")
    segments.append(tuple(tokens[start:]))
    return tuple(segments)


def _sqlite_check_expressions(connection: Connection, table_name: str) -> Tuple[CanonicalSql, ...]:
    """Return every CHECK expression, including unnamed and nested constraints."""

    create_sql = connection.execute(
        text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :table_name"),
        {"table_name": table_name},
    ).scalar_one()
    tokens = _sqlite_tokens(create_sql)
    expressions: List[CanonicalSql] = []
    search_from = 0
    while search_from < len(tokens):
        check_at = next(
            (index for index in range(search_from, len(tokens)) if tokens[index] == _SQLiteToken("word", "check")),
            -1,
        )
        if check_at < 0:
            break
        opening_at = check_at + 1
        if opening_at >= len(tokens) or tokens[opening_at] != _SQLiteToken("symbol", "("):
            raise RuntimeError(f"malformed SQLite CHECK constraint on {table_name}")
        expression, search_from = _extract_parenthesized_tokens(tokens, opening_at)
        expressions.append(_canonicalize_sql_expression_tokens(expression))
    return tuple(expressions)


def _normalized_declared_type(value: object) -> str:
    return " ".join(str(value).strip().upper().split())


def _sqlite_column_collations(create_sql: object, column_names: Sequence[str]) -> Dict[str, str]:
    tokens = _sqlite_tokens(create_sql)
    opening_at = next(
        (index for index, token in enumerate(tokens) if token == _SQLiteToken("symbol", "(")),
        -1,
    )
    if opening_at < 0:
        raise RuntimeError("malformed SQLite CREATE TABLE statement")
    table_body, _ = _extract_parenthesized_tokens(tokens, opening_at)
    expected_names = {name.lower() for name in column_names}
    collations: Dict[str, str] = {}
    for segment in _split_top_level_tokens(table_body):
        if not segment or not _token_is_identifier(segment[0]):
            continue
        column_name = segment[0].value.lower()
        if column_name not in expected_names:
            continue
        depth = 0
        declared_collations: List[str] = []
        for index, token in enumerate(segment[1:], start=1):
            if token == _SQLiteToken("symbol", "("):
                depth += 1
            elif token == _SQLiteToken("symbol", ")"):
                depth -= 1
            elif depth == 0 and token == _SQLiteToken("word", "collate"):
                if index + 1 >= len(segment) or not _token_is_identifier(segment[index + 1]):
                    raise RuntimeError(f"malformed SQLite COLLATE clause for {column_name}")
                declared_collations.append(segment[index + 1].value.lower())
        if depth != 0 or len(declared_collations) > 1:
            raise RuntimeError(f"malformed SQLite column definition for {column_name}")
        collations[column_name] = declared_collations[0] if declared_collations else "binary"
    return collations


def _sqlite_conflict_clauses(create_sql: object) -> Tuple[str, ...]:
    tokens = _sqlite_tokens(create_sql)
    clauses: List[str] = []
    for index in range(len(tokens) - 2):
        if tokens[index] == _SQLiteToken("word", "on") and tokens[index + 1] == _SQLiteToken("word", "conflict"):
            resolution = tokens[index + 2]
            if resolution.kind != "word" or resolution.value not in (
                "rollback",
                "abort",
                "fail",
                "ignore",
                "replace",
            ):
                raise RuntimeError("malformed SQLite ON CONFLICT clause")
            clauses.append(resolution.value)
    return tuple(sorted(clauses))


def _sqlite_table_options(create_sql: object) -> CanonicalSql:
    tokens = _sqlite_tokens(create_sql)
    opening_at = next(
        (index for index, token in enumerate(tokens) if token == _SQLiteToken("symbol", "(")),
        -1,
    )
    if opening_at < 0:
        raise RuntimeError("malformed SQLite CREATE TABLE statement")
    _, after_table_body = _extract_parenthesized_tokens(tokens, opening_at)
    options = list(tokens[after_table_body:])
    while options and options[-1] == _SQLiteToken("symbol", ";"):
        options.pop()
    return _canonicalize_sql_tokens(options)


class _IndexSignature(NamedTuple):
    unique: bool
    origin: str
    partial: bool
    xinfo: Tuple[Tuple[int, int, Optional[str], bool, Optional[str], bool], ...]
    definition: Optional[Tuple[str, Tuple[CanonicalSql, ...], CanonicalSql]]


def _canonicalize_index_key_tokens(tokens: Sequence[_SQLiteToken]) -> CanonicalSql:
    """Compare key expressions while index_xinfo owns effective order and collation."""

    expression_tokens = list(tokens)
    if expression_tokens and expression_tokens[-1] == _SQLiteToken("word", "asc"):
        expression_tokens.pop()
    if (
        len(expression_tokens) >= 2
        and expression_tokens[-2] == _SQLiteToken("word", "collate")
        and _token_is_identifier(expression_tokens[-1])
        and expression_tokens[-1].value.lower() == "binary"
    ):
        del expression_tokens[-2:]
    if not expression_tokens:
        raise RuntimeError("malformed SQLite CREATE INDEX key expression")
    return _canonicalize_sql_expression_tokens(expression_tokens)


def _index_definition_signature(
    create_sql: Optional[str],
) -> Optional[Tuple[str, Tuple[CanonicalSql, ...], CanonicalSql]]:
    if create_sql is None:
        return None
    tokens = _sqlite_tokens(create_sql)
    on_at = next(
        (index for index, token in enumerate(tokens) if token == _SQLiteToken("word", "on")),
        -1,
    )
    if on_at < 0 or on_at + 1 >= len(tokens) or not _token_is_identifier(tokens[on_at + 1]):
        raise RuntimeError("malformed SQLite CREATE INDEX statement")
    target_at = on_at + 1
    if (
        target_at + 2 < len(tokens)
        and tokens[target_at + 1] == _SQLiteToken("symbol", ".")
        and _token_is_identifier(tokens[target_at + 2])
    ):
        target_at += 2
    opening_at = target_at + 1
    if opening_at >= len(tokens) or tokens[opening_at] != _SQLiteToken("symbol", "("):
        raise RuntimeError("malformed SQLite CREATE INDEX key list")
    key_tokens, after_keys = _extract_parenthesized_tokens(tokens, opening_at)
    keys = tuple(_canonicalize_index_key_tokens(key) for key in _split_top_level_tokens(key_tokens))
    remainder = list(tokens[after_keys:])
    while remainder and remainder[-1] == _SQLiteToken("symbol", ";"):
        remainder.pop()
    if remainder:
        if remainder[0] != _SQLiteToken("word", "where"):
            raise RuntimeError("malformed SQLite CREATE INDEX predicate")
        predicate = _canonicalize_sql_expression_tokens(remainder[1:])
    else:
        predicate = tuple()
    return tokens[target_at].value.lower(), keys, predicate


def _index_xinfo_signature(
    rows: Sequence[Sequence[object]],
) -> Tuple[Tuple[int, int, Optional[str], bool, Optional[str], bool], ...]:
    return tuple(
        (
            int(row[0]),
            int(row[1]),
            None if row[2] is None else str(row[2]).lower(),
            bool(row[3]),
            None if row[4] is None else str(row[4]).lower(),
            bool(row[5]),
        )
        for row in rows
    )


def _actual_index_catalog(connection: Connection, table_name: str) -> Dict[str, _IndexSignature]:
    catalog: Dict[str, _IndexSignature] = {}
    rows = connection.execute(
        text('SELECT seq, name, "unique", origin, partial ' "FROM pragma_index_list(:table_name) ORDER BY seq"),
        {"table_name": table_name},
    ).fetchall()
    for row in rows:
        index_name = str(row[1])
        xinfo_rows = connection.execute(
            text('SELECT seqno, cid, name, "desc", coll, "key" ' "FROM pragma_index_xinfo(:index_name) ORDER BY seqno"),
            {"index_name": index_name},
        ).fetchall()
        create_sql = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE type = 'index' AND name = :index_name"),
            {"index_name": index_name},
        ).scalar_one_or_none()
        catalog[index_name] = _IndexSignature(
            unique=bool(row[2]),
            origin=str(row[3]).lower(),
            partial=bool(row[4]),
            xinfo=_index_xinfo_signature(xinfo_rows),
            definition=_index_definition_signature(create_sql),
        )
    return catalog


def _expected_index_catalog(table: Table, dialect) -> Dict[str, _IndexSignature]:
    database = sqlite3.connect(":memory:")
    try:
        database.execute(str(CreateTable(table).compile(dialect=dialect)))
        for index in sorted(table.indexes, key=lambda candidate: candidate.name):
            database.execute(str(CreateIndex(index).compile(dialect=dialect)))
        catalog: Dict[str, _IndexSignature] = {}
        rows = database.execute(
            'SELECT seq, name, "unique", origin, partial FROM pragma_index_list(?) ORDER BY seq',
            (table.name,),
        ).fetchall()
        for row in rows:
            index_name = str(row[1])
            xinfo_rows = database.execute(
                'SELECT seqno, cid, name, "desc", coll, "key" ' "FROM pragma_index_xinfo(?) ORDER BY seqno",
                (index_name,),
            ).fetchall()
            create_sql_row = database.execute(
                "SELECT sql FROM sqlite_master WHERE type = 'index' AND name = ?",
                (index_name,),
            ).fetchone()
            create_sql = None if create_sql_row is None else create_sql_row[0]
            catalog[index_name] = _IndexSignature(
                unique=bool(row[2]),
                origin=str(row[3]).lower(),
                partial=bool(row[4]),
                xinfo=_index_xinfo_signature(xinfo_rows),
                definition=_index_definition_signature(create_sql),
            )
        return catalog
    finally:
        database.close()


def _validate_table_shape(connection: Connection, table: Table) -> Tuple[str, ...]:
    inspector = inspect(connection)
    errors: List[str] = []
    actual_column_rows = connection.execute(
        text(
            'SELECT cid, name, type, "notnull", dflt_value, pk, hidden '
            "FROM pragma_table_xinfo(:table_name) ORDER BY cid"
        ),
        {"table_name": table.name},
    ).fetchall()
    actual_columns = {str(row[1]).lower(): row for row in actual_column_rows}
    expected_columns = {column.name.lower(): column for column in table.columns}
    actual_column_order = tuple(str(row[1]).lower() for row in actual_column_rows)
    expected_column_order = tuple(column.name.lower() for column in table.columns)
    if actual_column_order != expected_column_order:
        errors.append(f"{table.name} column order {actual_column_order} expected {expected_column_order}")
    expected_primary_key_positions = {
        column.name.lower(): position for position, column in enumerate(table.primary_key.columns, start=1)
    }
    for expected in table.columns:
        actual = actual_columns.get(expected.name.lower())
        if actual is None:
            errors.append(f"{table.name}.{expected.name} is missing")
            continue
        actual_not_null = bool(actual[3])
        expected_not_null = not bool(expected.nullable)
        if actual_not_null != expected_not_null:
            errors.append(f"{table.name}.{expected.name} notnull={actual_not_null} " f"expected {expected_not_null}")
        actual_type = _normalized_declared_type(actual[2])
        expected_type = _normalized_declared_type(expected.type.compile(dialect=connection.dialect))
        if actual_type != expected_type:
            errors.append(f"{table.name}.{expected.name} declared type {actual_type} expected {expected_type}")
        actual_default = None if actual[4] is None else _normalize_sql(actual[4])
        expected_default = None if expected.server_default is None else _normalize_sql(expected.server_default.arg)
        if actual_default != expected_default:
            errors.append(f"{table.name}.{expected.name} has an incompatible default")
        actual_primary_key_position = int(actual[5])
        expected_primary_key_position = expected_primary_key_positions.get(expected.name.lower(), 0)
        if actual_primary_key_position != expected_primary_key_position:
            errors.append(
                f"{table.name}.{expected.name} primary-key position "
                f"{actual_primary_key_position} expected {expected_primary_key_position}"
            )
        if int(actual[6]) != 0:
            errors.append(f"{table.name}.{expected.name} is generated or hidden")
    for unexpected_column in sorted(set(actual_columns) - set(expected_columns)):
        errors.append(f"{table.name}.{unexpected_column} is an unexpected column")

    create_sql = connection.execute(
        text("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = :table_name"),
        {"table_name": table.name},
    ).scalar_one()
    actual_collations = _sqlite_column_collations(create_sql, expected_column_order)
    expected_create_sql = str(CreateTable(table).compile(dialect=connection.dialect))
    expected_collations = _sqlite_column_collations(
        expected_create_sql,
        expected_column_order,
    )
    if actual_collations != expected_collations:
        errors.append(f"{table.name} column collation set is incompatible")
    if _sqlite_conflict_clauses(create_sql) != _sqlite_conflict_clauses(expected_create_sql):
        errors.append(f"{table.name} ON CONFLICT behavior is incompatible")
    if _sqlite_table_options(create_sql) != _sqlite_table_options(expected_create_sql):
        errors.append(f"{table.name} rowid/strict table options are incompatible")

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
    actual_checks = tuple(sorted(_sqlite_check_expressions(connection, table.name)))
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

    expected_index_catalog = _expected_index_catalog(table, connection.dialect)
    actual_index_catalog = _actual_index_catalog(connection, table.name)
    expected_explicit_indexes = {
        name: signature for name, signature in expected_index_catalog.items() if signature.origin == "c"
    }
    actual_explicit_indexes = {
        name: signature for name, signature in actual_index_catalog.items() if signature.origin == "c"
    }
    for unexpected_index in sorted(set(actual_explicit_indexes) - set(expected_explicit_indexes)):
        errors.append(f"{table.name} index {unexpected_index} is unexpected")
    for index_name, expected_signature in expected_explicit_indexes.items():
        actual_signature = actual_explicit_indexes.get(index_name)
        if actual_signature is None:
            errors.append(f"{table.name} index {index_name} is missing")
            continue
        if actual_signature != expected_signature:
            errors.append(f"{table.name} index {index_name} has incompatible definition")

    expected_internal_indexes = Counter(
        signature for signature in expected_index_catalog.values() if signature.origin != "c"
    )
    actual_internal_indexes = Counter(
        signature for signature in actual_index_catalog.values() if signature.origin != "c"
    )
    if actual_internal_indexes != expected_internal_indexes:
        errors.append(f"{table.name} constraint-owned index set is incompatible")

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
            continue
        expected_table_name = _sqlite_trigger_target(expected_sql)
        if str(actual_trigger[0]).lower() != expected_table_name:
            errors.append(f"SQLite trigger {name} has incompatible target")
        if _normalize_trigger_sql(actual_trigger[1]) != _normalize_trigger_sql(expected_sql):
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
