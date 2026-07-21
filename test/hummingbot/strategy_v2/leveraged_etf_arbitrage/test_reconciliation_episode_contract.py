import json
import sqlite3
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

import hummingbot.model.leveraged_etf_repository as durable_contract
from hummingbot.model.leveraged_etf_persistence import SQLITE_GUARD_DDL
from hummingbot.model.leveraged_etf_repository import (
    JournalConflictError,
    JournalEventType,
    JournalEventV1,
    JournalIntegrityError,
    LeveragedEtfJournalRepository,
)

from .test_journal_repository import (
    _append_fact,
    _assert_fact_rejected_atomically,
    _canonical_json_hash,
    _contract_vectors,
    _fact_event,
    _initial_snapshot,
    _open_manager,
    _snapshot_for_executor,
)


UNKNOWN_AT = "2026-07-17T14:01:04.000000Z"
SWEEP_ONE_AT = "2026-07-17T14:01:05.000000Z"
SWEEP_TWO_AT = "2026-07-17T14:01:06.000000Z"
RECONCILED_AT = "2026-07-17T14:01:07.000000Z"
EPISODE_T0 = "2026-07-17T14:01:08.000000Z"
EPISODE_PROGRESS_AT = "2026-07-17T14:01:09.000000Z"
EPISODE_RESTART_AT = "2026-07-17T14:01:10.000000Z"
EPISODE_CLOSE_AT = "2026-07-17T14:01:11.000000Z"


def _initial():
    return _initial_snapshot(_contract_vectors())


def _prepare_unknown(
    repository: LeveragedEtfJournalRepository,
    initial,
    *,
    logical_quantity: str = "2",
    intent_id: str = "intent-t005-unknown",
    client_order_id: str = "client-t005-unknown",
    event_prefix: str = "t005",
):
    current = _append_fact(
        repository,
        initial,
        _fact_event(
            initial,
            JournalEventType.PREPARED,
            f"event-{event_prefix}-prepared",
            logical_quantity=logical_quantity,
            intent_id=intent_id,
            client_order_id=client_order_id,
        ),
    )
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.SUBMIT_UNKNOWN,
            f"event-{event_prefix}-submit-unknown",
            logical_quantity=logical_quantity,
            intent_id=intent_id,
            client_order_id=client_order_id,
            created_at_utc=UNKNOWN_AT,
        ),
    )
    return current


def _reconciliation_event(
    initial,
    event_id: str,
    *,
    outcome: str,
    logical_quantity: str = "2",
    intent_id: str = "intent-t005-unknown",
    client_order_id: str = "client-t005-unknown",
    exchange_order_id: str | None = None,
    cumulative: str = "0",
    created_at_utc: str = RECONCILED_AT,
    evidence_state: str | None = None,
    proven_no_fill: dict | None = None,
) -> JournalEventV1:
    event = _fact_event(
        initial,
        JournalEventType.RECONCILIATION,
        event_id,
        logical_quantity=logical_quantity,
        intent_id=intent_id,
        client_order_id=client_order_id,
        exchange_order_id=exchange_order_id,
        order_cumulative_filled_quantity=cumulative,
        outcome=outcome,
        created_at_utc=created_at_utc,
    )
    serialized = event.model_dump(mode="json")
    if evidence_state is not None:
        serialized["payload"]["evidence_state"] = evidence_state
    if proven_no_fill is not None:
        serialized["payload"]["proven_no_fill"] = proven_no_fill
    return JournalEventV1.model_validate(serialized)


def _no_fill_proof(
    *,
    proof_id: str = "proof-t005-no-fill",
    exchange_order_id: str | None = None,
) -> dict:
    def sweep(suffix: str, observed_at_utc: str) -> dict:
        return {
            "schema_version": 1,
            "sweep_id": f"sweep-t005-{suffix}",
            "observed_at_utc": observed_at_utc,
            "post_grace": True,
            "order_status": {
                "schema_version": 1,
                "outcome": "NOT_FOUND",
                "exchange_order_id": exchange_order_id,
                "cumulative_filled_quantity": "0",
            },
            "trade_history": {
                "schema_version": 1,
                "exchange_order_id": exchange_order_id,
                "exchange_trade_ids": [],
                "cumulative_filled_quantity": "0",
            },
            "position_consistency": {
                "schema_version": 1,
                "consistent": True,
                "exposure_change_quantity": "0",
            },
        }

    return {
        "schema_version": 1,
        "proof_id": proof_id,
        "sweeps": [
            sweep("one", SWEEP_ONE_AT),
            sweep("two", SWEEP_TWO_AT),
        ],
    }


def _require_reconciliation_contract():
    evidence_type = getattr(durable_contract, "ReconciliationEvidenceState", None)
    proof_type = getattr(durable_contract, "ProvenNoFillEvidenceV1", None)
    if evidence_type is None or proof_type is None:
        pytest.skip("T005 reconciliation evidence contract is not implemented yet")
    return evidence_type, proof_type


def _require_episode_contract():
    status_type = getattr(durable_contract, "ExposureEpisodeStatus", None)
    record_type = getattr(durable_contract, "ExposureEpisodeAuditV1", None)
    if status_type is None or record_type is None:
        pytest.skip("T005 exposure episode audit contract is not implemented yet")
    return status_type, record_type


@pytest.mark.parametrize("outcome", ("UNKNOWN", "NOT_FOUND"))
def test_t005_unknown_reconciliation_remains_reconciling_and_keeps_intent(outcome: str, tmp_path: Path):
    manager = _open_manager(tmp_path / f"unknown-{outcome.lower()}.sqlite")
    try:
        repository = LeveragedEtfJournalRepository(manager)
        initial = _initial()
        repository.create_executor(initial)
        current = _prepare_unknown(repository, initial)
        assert current.state.value == "RECONCILING"

        current = _append_fact(
            repository,
            current,
            _reconciliation_event(initial, f"event-t005-{outcome.lower()}", outcome=outcome),
        )

        assert current.state.value == "RECONCILING"
        assert current.etf_filled_quantity == Decimal("0")
        incomplete = repository.incomplete_intents(initial.executor_id)
        assert tuple(value.intent_id for value in incomplete) == ("intent-t005-unknown",)
        assert incomplete[0].prepared_event.client_order_id == "client-t005-unknown"
        assert repository.events(initial.executor_id)[-1].event.payload.evidence_state.value == "UNKNOWN"

        duplicate_exposure = _fact_event(
            initial,
            JournalEventType.PREPARED,
            f"event-t005-duplicate-{outcome.lower()}",
            logical_quantity="2",
            attempt=2,
            intent_id=f"intent-t005-duplicate-{outcome.lower()}",
            client_order_id=f"client-t005-duplicate-{outcome.lower()}",
        )
        _assert_fact_rejected_atomically(repository, initial.executor_id, duplicate_exposure)
    finally:
        manager.engine.dispose()


def test_t005_existing_unknown_to_trade_fill_path_stays_authoritative_and_replayable(tmp_path: Path):
    db_path = tmp_path / "unknown-fill-regression.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial()
    repository.create_executor(initial)
    current = _prepare_unknown(repository, initial)
    current = _append_fact(
        repository,
        current,
        _reconciliation_event(
            initial,
            "event-t005-unknown-before-fill",
            outcome="UNKNOWN",
            exchange_order_id="exchange-t005-maker",
        ),
    )
    first_fill = _fact_event(
        initial,
        JournalEventType.FILL,
        "event-t005-fill-one",
        logical_quantity="2",
        intent_id="intent-t005-unknown",
        client_order_id="client-t005-unknown",
        exchange_order_id="exchange-t005-maker",
        exchange_trade_id="trade-t005-one",
        fill_quantity="0.75",
        order_cumulative_filled_quantity="0.75",
        leg_cumulative_filled_quantity="0.75",
        outcome="PARTIAL",
        created_at_utc="2026-07-17T14:01:08.000000Z",
    )
    current = _append_fact(repository, current, first_fill)
    current = _append_fact(
        repository,
        current,
        _fact_event(
            initial,
            JournalEventType.ORDER_CREATED,
            "event-t005-late-created",
            logical_quantity="2",
            intent_id="intent-t005-unknown",
            client_order_id="client-t005-unknown",
            exchange_order_id="exchange-t005-maker",
            created_at_utc="2026-07-17T14:01:09.000000Z",
        ),
    )
    second_fill = _fact_event(
        initial,
        JournalEventType.FILL,
        "event-t005-fill-two",
        logical_quantity="2",
        intent_id="intent-t005-unknown",
        client_order_id="client-t005-unknown",
        exchange_order_id="exchange-t005-maker",
        exchange_trade_id="trade-t005-two",
        fill_quantity="1.25",
        order_cumulative_filled_quantity="2",
        leg_cumulative_filled_quantity="2",
        outcome="FILLED",
        created_at_utc="2026-07-17T14:01:10.000000Z",
    )
    current = _append_fact(repository, current, second_fill)

    assert current.state.value == "STOCK_HEDGE_PENDING"
    assert current.etf_filled_quantity == Decimal("2")
    assert tuple(reference.exchange_order_id for reference in current.maker_order_ids) == (
        "exchange-t005-maker",
    )
    assert repository.incomplete_intents(initial.executor_id) == ()
    assert repository.append_and_reduce(initial.executor_id, first_fill).sequence == 4
    _assert_fact_rejected_atomically(
        repository,
        initial.executor_id,
        first_fill.model_copy(update={"event_id": "event-t005-fill-one-conflict"}),
    )
    _assert_fact_rejected_atomically(
        repository,
        initial.executor_id,
        _reconciliation_event(
            initial,
            "event-t005-redundant-terminal",
            outcome="FILLED",
            exchange_order_id="exchange-t005-maker",
            cumulative="2",
            created_at_utc="2026-07-17T14:01:11.000000Z",
        ),
    )
    evidence_type = getattr(durable_contract, "ReconciliationEvidenceState", None)
    if evidence_type is not None:
        fill_events = tuple(
            value.event for value in repository.events(initial.executor_id) if value.event.event_type == JournalEventType.FILL
        )
        assert all(event.payload.evidence_state is evidence_type.AUTHORITATIVE_FILL for event in fill_events)
    committed = repository.events(initial.executor_id)
    assert repository.replay(initial.executor_id) == current
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        assert reopened_repository.replay(initial.executor_id) == current
        assert reopened_repository.events(initial.executor_id) == committed
        assert reopened_repository.incomplete_intents(initial.executor_id) == ()
    finally:
        reopened.engine.dispose()


def test_t005_consistent_no_fill_without_proof_is_not_terminal(tmp_path: Path):
    manager = _open_manager(tmp_path / "unproven-no-fill.sqlite")
    try:
        repository = LeveragedEtfJournalRepository(manager)
        initial = _initial()
        repository.create_executor(initial)
        current = _prepare_unknown(repository, initial)
        current = _append_fact(
            repository,
            current,
            _reconciliation_event(
                initial,
                "event-t005-unproven-no-fill",
                outcome="CONSISTENT_NO_FILL",
            ),
        )

        assert current.state.value == "RECONCILING"
        assert tuple(value.intent_id for value in repository.incomplete_intents(initial.executor_id)) == (
            "intent-t005-unknown",
        )
        payload = repository.events(initial.executor_id)[-1].event.payload
        assert payload.evidence_state.value == "UNKNOWN"
        assert payload.proven_no_fill is None
    finally:
        manager.engine.dispose()


def test_t005_proven_no_fill_requires_and_replays_two_three_source_sweeps(tmp_path: Path):
    evidence_type = getattr(durable_contract, "ReconciliationEvidenceState", None)
    proof_type = getattr(durable_contract, "ProvenNoFillEvidenceV1", None)
    assert evidence_type is not None, "tri-state reconciliation evidence contract is missing"
    assert proof_type is not None, "versioned no-fill proof contract is missing"

    db_path = tmp_path / "proven-no-fill.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial()
    repository.create_executor(initial)
    current = _prepare_unknown(repository, initial)
    proof = _no_fill_proof()
    event = _reconciliation_event(
        initial,
        "event-t005-proven-no-fill",
        outcome="CONSISTENT_NO_FILL",
        evidence_state="PROVEN_NO_FILL",
        proven_no_fill=proof,
    )
    committed = repository.append_and_reduce(initial.executor_id, event)
    current = repository.load_snapshot(initial.executor_id)
    assert current is not None
    assert repository.append_and_reduce(initial.executor_id, event) == committed
    assert repository.incomplete_intents(initial.executor_id) == ()
    stored_payload = repository.events(initial.executor_id)[-1].event.payload
    assert stored_payload.evidence_state is evidence_type.PROVEN_NO_FILL
    assert stored_payload.proven_no_fill == proof_type.model_validate(proof)
    assert len(stored_payload.proven_no_fill.sweeps) == 2
    assert repository.replay(initial.executor_id) == current
    stored_events = repository.events(initial.executor_id)
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        assert reopened_repository.replay(initial.executor_id) == current
        assert reopened_repository.events(initial.executor_id) == stored_events
        assert reopened_repository.incomplete_intents(initial.executor_id) == ()
    finally:
        reopened.engine.dispose()


def test_t005_proven_no_fill_rejects_missing_malformed_or_contradictory_proof(tmp_path: Path):
    _require_reconciliation_contract()
    initial = _initial()

    malformed = []
    one_sweep = _no_fill_proof()
    one_sweep["sweeps"] = one_sweep["sweeps"][:1]
    malformed.append(one_sweep)
    duplicate_sweep = _no_fill_proof()
    duplicate_sweep["sweeps"][1]["sweep_id"] = duplicate_sweep["sweeps"][0]["sweep_id"]
    malformed.append(duplicate_sweep)
    reversed_time = _no_fill_proof()
    reversed_time["sweeps"][1]["observed_at_utc"] = SWEEP_ONE_AT
    malformed.append(reversed_time)
    inconsistent_status = _no_fill_proof()
    inconsistent_status["sweeps"][1]["order_status"]["outcome"] = "CANCELED"
    malformed.append(inconsistent_status)
    nonzero_order = _no_fill_proof()
    nonzero_order["sweeps"][0]["order_status"]["cumulative_filled_quantity"] = "0.1"
    malformed.append(nonzero_order)
    nonzero_trade = _no_fill_proof()
    nonzero_trade["sweeps"][1]["trade_history"]["cumulative_filled_quantity"] = "0.1"
    malformed.append(nonzero_trade)
    trade_present = _no_fill_proof()
    trade_present["sweeps"][1]["trade_history"]["exchange_trade_ids"] = ["unexpected-trade"]
    malformed.append(trade_present)
    inconsistent_position = _no_fill_proof()
    inconsistent_position["sweeps"][0]["position_consistency"]["consistent"] = False
    malformed.append(inconsistent_position)
    nonzero_position = _no_fill_proof()
    nonzero_position["sweeps"][0]["position_consistency"]["exposure_change_quantity"] = "0.1"
    malformed.append(nonzero_position)
    before_grace = _no_fill_proof()
    before_grace["sweeps"][0]["post_grace"] = False
    malformed.append(before_grace)

    for index, proof in enumerate(malformed):
        with pytest.raises(ValidationError):
            _reconciliation_event(
                initial,
                f"event-t005-malformed-proof-{index}",
                outcome="CONSISTENT_NO_FILL",
                evidence_state="PROVEN_NO_FILL",
                proven_no_fill=proof,
            )

    with pytest.raises(ValidationError):
        _reconciliation_event(
            initial,
            "event-t005-missing-proof",
            outcome="CONSISTENT_NO_FILL",
            evidence_state="PROVEN_NO_FILL",
        )
    with pytest.raises(ValidationError):
        _reconciliation_event(
            initial,
            "event-t005-proof-on-unknown",
            outcome="UNKNOWN",
            evidence_state="UNKNOWN",
            proven_no_fill=_no_fill_proof(),
        )
    with pytest.raises(ValidationError):
        _reconciliation_event(
            initial,
            "event-t005-fill-arm-on-reconciliation",
            outcome="UNKNOWN",
            evidence_state="AUTHORITATIVE_FILL",
        )

    manager = _open_manager(tmp_path / "proof-after-fill.sqlite")
    try:
        repository = LeveragedEtfJournalRepository(manager)
        repository.create_executor(initial)
        current = _prepare_unknown(repository, initial)
        current = _append_fact(
            repository,
            current,
            _reconciliation_event(
                initial,
                "event-t005-unknown-bound",
                outcome="UNKNOWN",
                exchange_order_id="exchange-t005-bound",
            ),
        )
        current = _append_fact(
            repository,
            current,
            _fact_event(
                initial,
                JournalEventType.FILL,
                "event-t005-known-partial",
                logical_quantity="2",
                intent_id="intent-t005-unknown",
                client_order_id="client-t005-unknown",
                exchange_order_id="exchange-t005-bound",
                exchange_trade_id="trade-t005-known-partial",
                fill_quantity="0.5",
                order_cumulative_filled_quantity="0.5",
                leg_cumulative_filled_quantity="0.5",
                outcome="PARTIAL",
                created_at_utc="2026-07-17T14:01:08.000000Z",
            ),
        )
        _assert_fact_rejected_atomically(
            repository,
            initial.executor_id,
            _reconciliation_event(
                initial,
                "event-t005-false-no-fill-after-fill",
                outcome="CONSISTENT_NO_FILL",
                exchange_order_id="exchange-t005-bound",
                cumulative="0",
                evidence_state="PROVEN_NO_FILL",
                proven_no_fill=_no_fill_proof(exchange_order_id="exchange-t005-bound"),
                created_at_utc="2026-07-17T14:01:09.000000Z",
            ),
        )

        terminal = _reconciliation_event(
            initial,
            "event-t005-terminal-canceled",
            outcome="CANCELED",
            exchange_order_id="exchange-t005-bound",
            cumulative="0.5",
            created_at_utc="2026-07-17T14:01:10.000000Z",
        )
        _append_fact(repository, current, terminal)
        assert repository.incomplete_intents(initial.executor_id) == ()
    finally:
        manager.engine.dispose()


@pytest.mark.parametrize("outcome", ("CANCELED", "EXPIRED", "REJECTED"))
def test_t005_zero_fill_terminal_status_requires_proof_and_keeps_late_fill_open(
    outcome: str,
    tmp_path: Path,
):
    manager = _open_manager(tmp_path / f"unproven-{outcome.lower()}.sqlite")
    try:
        repository = LeveragedEtfJournalRepository(manager)
        initial = _initial()
        repository.create_executor(initial)
        current = _prepare_unknown(repository, initial, event_prefix=f"t005-{outcome.lower()}")
        current = _append_fact(
            repository,
            current,
            _reconciliation_event(
                initial,
                f"event-t005-unproven-{outcome.lower()}",
                outcome=outcome,
                exchange_order_id=f"exchange-t005-{outcome.lower()}",
            ),
        )

        assert current.state.value == "RECONCILING"
        assert tuple(value.intent_id for value in repository.incomplete_intents(initial.executor_id)) == (
            "intent-t005-unknown",
        )
        assert repository.events(initial.executor_id)[-1].event.payload.terminal is False

        late_fill = _fact_event(
            initial,
            JournalEventType.FILL,
            f"event-t005-late-fill-{outcome.lower()}",
            logical_quantity="2",
            intent_id="intent-t005-unknown",
            client_order_id="client-t005-unknown",
            exchange_order_id=f"exchange-t005-{outcome.lower()}",
            exchange_trade_id=f"trade-t005-late-{outcome.lower()}",
            fill_quantity="0.5",
            order_cumulative_filled_quantity="0.5",
            leg_cumulative_filled_quantity="0.5",
            outcome="PARTIAL",
            created_at_utc="2026-07-17T14:01:08.000000Z",
        )
        current = _append_fact(repository, current, late_fill)
        assert current.etf_filled_quantity == Decimal("0.5")
        assert tuple(value.intent_id for value in repository.incomplete_intents(initial.executor_id)) == (
            "intent-t005-unknown",
        )
    finally:
        manager.engine.dispose()


def test_t005_proven_no_fill_rejects_sweeps_before_submit_uncertainty(tmp_path: Path):
    _require_reconciliation_contract()
    manager = _open_manager(tmp_path / "proof-before-uncertainty.sqlite")
    try:
        repository = LeveragedEtfJournalRepository(manager)
        initial = _initial()
        repository.create_executor(initial)
        current = _prepare_unknown(repository, initial)
        proof = _no_fill_proof()
        proof["sweeps"][0]["observed_at_utc"] = "2026-07-17T14:01:02.000000Z"
        proof["sweeps"][1]["observed_at_utc"] = "2026-07-17T14:01:03.000000Z"
        event = _reconciliation_event(
            initial,
            "event-t005-proof-before-uncertainty",
            outcome="CONSISTENT_NO_FILL",
            evidence_state="PROVEN_NO_FILL",
            proven_no_fill=proof,
        )

        _assert_fact_rejected_atomically(repository, initial.executor_id, event)
        assert repository.load_snapshot(initial.executor_id) == current
    finally:
        manager.engine.dispose()


def test_t005_proof_and_sweep_ids_cannot_be_reused_across_owner_scope(tmp_path: Path):
    _require_reconciliation_contract()
    manager = _open_manager(tmp_path / "proof-owner-isolation.sqlite")
    try:
        repository = LeveragedEtfJournalRepository(manager)
        first = _initial()
        second_payload = _snapshot_for_executor(first, "exec-t005-proof-second").model_dump(mode="json")
        second_payload.update({"pair_id": "intc_intw", "nav_cycle_id": "xnys-2026-07-18"})
        second = type(first).model_validate(second_payload)
        repository.create_executor(first)
        repository.create_executor(second)

        first_current = _prepare_unknown(
            repository,
            first,
            intent_id="intent-t005-proof-first",
            client_order_id="client-t005-proof-first",
            event_prefix="t005-proof-first",
        )
        shared_proof = _no_fill_proof(proof_id="proof-t005-shared")
        first_event = _reconciliation_event(
            first,
            "event-t005-proof-first-reconciled",
            outcome="CONSISTENT_NO_FILL",
            intent_id="intent-t005-proof-first",
            client_order_id="client-t005-proof-first",
            evidence_state="PROVEN_NO_FILL",
            proven_no_fill=shared_proof,
        )
        _append_fact(repository, first_current, first_event)
        assert repository.incomplete_intents(first.executor_id) == ()

        second_current = _prepare_unknown(
            repository,
            second,
            intent_id="intent-t005-proof-second",
            client_order_id="client-t005-proof-second",
            event_prefix="t005-proof-second",
        )
        reused = _reconciliation_event(
            second,
            "event-t005-proof-second-reused",
            outcome="CONSISTENT_NO_FILL",
            intent_id="intent-t005-proof-second",
            client_order_id="client-t005-proof-second",
            evidence_state="PROVEN_NO_FILL",
            proven_no_fill=shared_proof,
        )
        _assert_fact_rejected_atomically(repository, second.executor_id, reused)
        assert repository.load_snapshot(second.executor_id) == second_current
        assert tuple(value.intent_id for value in repository.incomplete_intents(second.executor_id)) == (
            "intent-t005-proof-second",
        )
    finally:
        manager.engine.dispose()


def test_t005_legacy_unproven_no_fill_replays_nonterminal_after_20260721_upgrade(tmp_path: Path):
    db_path = tmp_path / "legacy-unproven-no-fill.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial()
    repository.create_executor(initial)
    current = _prepare_unknown(repository, initial)
    current = _append_fact(
        repository,
        current,
        _reconciliation_event(
            initial,
            "event-t005-legacy-no-fill",
            outcome="CONSISTENT_NO_FILL",
        ),
    )
    assert current.state.value == "RECONCILING"
    legacy_snapshot_payload = current.model_dump(mode="json")
    legacy_snapshot_payload["state"] = "MAKER_WORKING"
    legacy_current = type(current).model_validate(legacy_snapshot_payload)
    manager.engine.dispose()

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TRIGGER lepf_journal_no_update")
        payload_json = connection.execute(
            "SELECT payload_json FROM LeveragedEtfJournalEvent WHERE event_id = ?",
            ("event-t005-legacy-no-fill",),
        ).fetchone()[0]
        mutation = json.loads(payload_json)
        mutation.pop("reducer_semantics_version", None)
        legacy_payload = mutation["event"]["payload"]
        legacy_payload.pop("evidence_state")
        legacy_payload.pop("proven_no_fill")
        mutation["snapshot_after"] = legacy_current.model_dump(mode="json")
        mutation["snapshot_after_hash"] = legacy_current.canonical_sha256()
        rewritten_json, rewritten_hash = _canonical_json_hash(mutation)
        connection.execute(
            "UPDATE LeveragedEtfJournalEvent SET payload_json = ?, payload_hash = ? WHERE event_id = ?",
            (rewritten_json, rewritten_hash, "event-t005-legacy-no-fill"),
        )
        legacy_snapshot_json = legacy_current.canonical_json()
        _, legacy_snapshot_hash = _canonical_json_hash(legacy_current.model_dump(mode="json"))
        connection.execute(
            """
            UPDATE LeveragedEtfExecutorSnapshot
            SET state = ?, snapshot_json = ?, snapshot_hash = ?
            WHERE executor_id = ?
            """,
            ("MAKER_WORKING", legacy_snapshot_json, legacy_snapshot_hash, initial.executor_id),
        )
        connection.executescript(SQLITE_GUARD_DDL["lepf_journal_no_update"])
        for trigger_name in (
            "lepf_snapshot_episode_referenced_no_delete",
            "lepf_episode_executor_fk_insert",
            "lepf_episode_identity_insert",
            "lepf_episode_no_update",
            "lepf_episode_no_delete",
        ):
            connection.execute(f'DROP TRIGGER IF EXISTS "{trigger_name}"')
        connection.execute('DROP TABLE IF EXISTS "LeveragedEtfExposureEpisodeAudit"')
        connection.execute(
            "UPDATE Metadata SET value = '20260721' WHERE key = 'local_db_version'"
        )

    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        assert reopened_repository.replay(initial.executor_id) == legacy_current
        incomplete = reopened_repository.incomplete_intents(initial.executor_id)
        assert tuple(value.intent_id for value in incomplete) == ("intent-t005-unknown",)
        assert incomplete[0].status is JournalEventType.RECONCILIATION

        late_fill = _fact_event(
            initial,
            JournalEventType.FILL,
            "event-t005-legacy-late-fill",
            logical_quantity="2",
            intent_id="intent-t005-unknown",
            client_order_id="client-t005-unknown",
            exchange_order_id="exchange-t005-legacy-late",
            exchange_trade_id="trade-t005-legacy-late",
            fill_quantity="0.5",
            order_cumulative_filled_quantity="0.5",
            leg_cumulative_filled_quantity="0.5",
            outcome="PARTIAL",
            created_at_utc="2026-07-17T14:01:08.000000Z",
        )
        repaired = _append_fact(reopened_repository, legacy_current, late_fill)
        assert repaired.etf_filled_quantity == Decimal("0.5")
    finally:
        reopened.engine.dispose()


def _episode_record(
    record_type,
    snapshot,
    *,
    episode_id: str = "episode-t005-one",
    revision: int = 1,
    status: str = "UNFINISHED",
    t0_utc: str = EPISODE_T0,
    process_boot_id: str = "boot-t005-a",
    hedge_phase_deadline_ms: int = 1000,
    rollback_phase_deadline_ms: int = 2000,
    unhedged_response_deadline_ms: int = 4000,
    latest_monotonic_elapsed_ms: int = 0,
    recorded_at_utc: str = EPISODE_T0,
    **overrides,
):
    return record_type.model_validate(
        {
            "schema_version": 1,
            "episode_id": episode_id,
            "executor_id": snapshot.executor_id,
            "pair_id": snapshot.pair_id,
            "nav_cycle_id": snapshot.nav_cycle_id,
            "revision": revision,
            "status": status,
            "t0_utc": t0_utc,
            "process_boot_id": process_boot_id,
            "hedge_phase_deadline_ms": hedge_phase_deadline_ms,
            "rollback_phase_deadline_ms": rollback_phase_deadline_ms,
            "unhedged_response_deadline_ms": unhedged_response_deadline_ms,
            "latest_monotonic_elapsed_ms": latest_monotonic_elapsed_ms,
            "recorded_at_utc": recorded_at_utc,
            **overrides,
        }
    )


def test_t005_exposure_episode_is_append_only_restart_visible_and_snapshot_neutral(tmp_path: Path):
    status_type = getattr(durable_contract, "ExposureEpisodeStatus", None)
    record_type = getattr(durable_contract, "ExposureEpisodeAuditV1", None)
    assert status_type is not None, "exposure episode status contract is missing"
    assert record_type is not None, "versioned exposure episode audit DTO is missing"
    assert hasattr(LeveragedEtfJournalRepository, "compare_and_append_exposure_episode")

    db_path = tmp_path / "episode-audit.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial()
    repository.create_executor(initial)
    snapshot_json = initial.canonical_json()
    snapshot_fields = tuple(type(initial).model_fields)

    started = _episode_record(record_type, initial)
    assert repository.compare_and_append_exposure_episode(started, expected_revision=0) == started
    assert repository.compare_and_append_exposure_episode(started, expected_revision=0) == started
    assert repository.active_exposure_episode(initial.executor_id) == started
    assert repository.unfinished_exposure_episodes(initial.executor_id) == (started,)

    progressed = _episode_record(
        record_type,
        initial,
        revision=2,
        latest_monotonic_elapsed_ms=10,
        recorded_at_utc=EPISODE_PROGRESS_AT,
    )
    assert repository.compare_and_append_exposure_episode(progressed, expected_revision=1) == progressed
    restarted = _episode_record(
        record_type,
        initial,
        revision=3,
        process_boot_id="boot-t005-b",
        latest_monotonic_elapsed_ms=0,
        recorded_at_utc=EPISODE_RESTART_AT,
    )
    assert repository.compare_and_append_exposure_episode(restarted, expected_revision=2) == restarted

    same_boot_regression = _episode_record(
        record_type,
        initial,
        revision=4,
        process_boot_id="boot-t005-a",
        latest_monotonic_elapsed_ms=9,
        recorded_at_utc=EPISODE_CLOSE_AT,
    )
    with pytest.raises((JournalConflictError, JournalIntegrityError), match="monotonic|elapsed|boot"):
        repository.compare_and_append_exposure_episode(same_boot_regression, expected_revision=3)

    closed = _episode_record(
        record_type,
        initial,
        revision=4,
        status="CLOSED",
        process_boot_id="boot-t005-b",
        latest_monotonic_elapsed_ms=2,
        recorded_at_utc=EPISODE_CLOSE_AT,
    )
    assert repository.compare_and_append_exposure_episode(closed, expected_revision=3) == closed
    assert repository.exposure_episode_records(initial.executor_id, started.episode_id) == (
        started,
        progressed,
        restarted,
        closed,
    )
    assert repository.replay_exposure_episode(initial.executor_id, started.episode_id) == closed
    assert repository.active_exposure_episode(initial.executor_id) is None
    assert repository.unfinished_exposure_episodes(initial.executor_id) == ()
    assert repository.load_snapshot(initial.executor_id).canonical_json() == snapshot_json
    assert tuple(type(repository.load_snapshot(initial.executor_id)).model_fields) == snapshot_fields
    assert repository.load_snapshot(initial.executor_id).last_journal_sequence == 0
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        assert reopened_repository.exposure_episode_records(initial.executor_id, started.episode_id) == (
            started,
            progressed,
            restarted,
            closed,
        )
        assert reopened_repository.replay_exposure_episode(initial.executor_id, started.episode_id) == closed
        assert reopened_repository.active_exposure_episode(initial.executor_id) is None

        next_episode = _episode_record(
            record_type,
            initial,
            episode_id="episode-t005-two",
            t0_utc="2026-07-17T14:01:12.000000Z",
            recorded_at_utc="2026-07-17T14:01:12.000000Z",
        )
        assert reopened_repository.compare_and_append_exposure_episode(next_episode, expected_revision=0) == next_episode
        assert reopened_repository.active_exposure_episode(initial.executor_id) == next_episode
    finally:
        reopened.engine.dispose()


def test_t005_exposure_episode_rejects_reset_wrong_owner_and_closed_reopen(tmp_path: Path):
    _, record_type = _require_episode_contract()
    manager = _open_manager(tmp_path / "episode-invariants.sqlite")
    try:
        repository = LeveragedEtfJournalRepository(manager)
        initial = _initial()
        repository.create_executor(initial)
        started = _episode_record(record_type, initial)
        repository.compare_and_append_exposure_episode(started, expected_revision=0)

        invalid_records = (
            _episode_record(
                record_type,
                initial,
                revision=2,
                t0_utc="2026-07-17T14:01:08.000001Z",
                recorded_at_utc=EPISODE_PROGRESS_AT,
            ),
            _episode_record(
                record_type,
                initial,
                revision=2,
                hedge_phase_deadline_ms=1001,
                recorded_at_utc=EPISODE_PROGRESS_AT,
            ),
            _episode_record(
                record_type,
                initial,
                episode_id="episode-t005-reset",
                recorded_at_utc=EPISODE_PROGRESS_AT,
            ),
            _episode_record(
                record_type,
                initial,
                revision=2,
                pair_id="foreign_pair",
                recorded_at_utc=EPISODE_PROGRESS_AT,
            ),
            _episode_record(
                record_type,
                initial,
                revision=2,
                nav_cycle_id="xnys-2026-07-18",
                recorded_at_utc=EPISODE_PROGRESS_AT,
            ),
        )
        for record in invalid_records:
            expected_revision = 0 if record.episode_id != started.episode_id else 1
            with pytest.raises((JournalConflictError, JournalIntegrityError)):
                repository.compare_and_append_exposure_episode(record, expected_revision=expected_revision)
            assert repository.exposure_episode_records(initial.executor_id, started.episode_id) == (started,)

        closed = _episode_record(
            record_type,
            initial,
            revision=2,
            status="CLOSED",
            latest_monotonic_elapsed_ms=1,
            recorded_at_utc=EPISODE_CLOSE_AT,
        )
        repository.compare_and_append_exposure_episode(closed, expected_revision=1)
        reopened_old = _episode_record(
            record_type,
            initial,
            revision=3,
            latest_monotonic_elapsed_ms=2,
            recorded_at_utc="2026-07-17T14:01:12.000000Z",
        )
        with pytest.raises((JournalConflictError, JournalIntegrityError), match="closed|terminal|status"):
            repository.compare_and_append_exposure_episode(reopened_old, expected_revision=2)
    finally:
        manager.engine.dispose()


def test_t005_exposure_episode_is_pair_cycle_executor_isolated(tmp_path: Path):
    _, record_type = _require_episode_contract()
    manager = _open_manager(tmp_path / "episode-isolation.sqlite")
    try:
        repository = LeveragedEtfJournalRepository(manager)
        first = _initial()
        second_payload = _snapshot_for_executor(first, "exec-t005-second").model_dump(mode="json")
        second_payload.update({"pair_id": "intc_intw", "nav_cycle_id": "xnys-2026-07-18"})
        second = type(first).model_validate(second_payload)
        repository.create_executor(first)
        repository.create_executor(second)

        first_record = _episode_record(record_type, first, episode_id="episode-t005-first")
        second_record = _episode_record(
            record_type,
            second,
            episode_id="episode-t005-second-owner",
            t0_utc="2026-07-18T14:01:08.000000Z",
            recorded_at_utc="2026-07-18T14:01:08.000000Z",
        )
        repository.compare_and_append_exposure_episode(first_record, expected_revision=0)
        repository.compare_and_append_exposure_episode(second_record, expected_revision=0)

        assert repository.unfinished_exposure_episodes(first.executor_id) == (first_record,)
        assert repository.unfinished_exposure_episodes(second.executor_id) == (second_record,)
        assert repository.unfinished_exposure_episodes() == (first_record, second_record)

        conflicting_owner = _episode_record(
            record_type,
            second,
            episode_id=first_record.episode_id,
            t0_utc="2026-07-18T14:01:09.000000Z",
            recorded_at_utc="2026-07-18T14:01:09.000000Z",
        )
        with pytest.raises((JournalConflictError, JournalIntegrityError), match="episode|owner|identity"):
            repository.compare_and_append_exposure_episode(conflicting_owner, expected_revision=0)
    finally:
        manager.engine.dispose()


def test_t005_exposure_episode_revalidates_untrusted_dto_before_write(tmp_path: Path):
    _, record_type = _require_episode_contract()
    manager = _open_manager(tmp_path / "episode-untrusted.sqlite")
    try:
        repository = LeveragedEtfJournalRepository(manager)
        initial = _initial()
        repository.create_executor(initial)
        valid = _episode_record(record_type, initial)
        malformed = (
            valid.model_copy(update={"latest_monotonic_elapsed_ms": -1}),
            valid.model_copy(update={"latest_monotonic_elapsed_ms": True}),
            valid.model_copy(update={"hedge_phase_deadline_ms": 0}),
            valid.model_copy(update={"revision": "1"}),
        )
        for record in malformed:
            with pytest.raises((JournalConflictError, JournalIntegrityError, ValidationError)):
                repository.compare_and_append_exposure_episode(record, expected_revision=0)
            assert repository.unfinished_exposure_episodes(initial.executor_id) == ()
        with pytest.raises((JournalConflictError, JournalIntegrityError), match="revision|integer"):
            repository.compare_and_append_exposure_episode(valid, expected_revision=True)
        assert repository.unfinished_exposure_episodes(initial.executor_id) == ()
    finally:
        manager.engine.dispose()


def test_t005_exposure_episode_reopen_detects_column_payload_corruption(tmp_path: Path):
    _, record_type = _require_episode_contract()
    db_path = tmp_path / "episode-corruption.sqlite"
    manager = _open_manager(db_path)
    repository = LeveragedEtfJournalRepository(manager)
    initial = _initial()
    repository.create_executor(initial)
    started = _episode_record(record_type, initial)
    repository.compare_and_append_exposure_episode(started, expected_revision=0)
    manager.engine.dispose()

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TRIGGER lepf_episode_no_update")
        connection.execute(
            "UPDATE LeveragedEtfExposureEpisodeAudit "
            "SET latest_monotonic_elapsed_ms = 99 WHERE episode_id = ? AND revision = 1",
            (started.episode_id,),
        )
        connection.executescript(SQLITE_GUARD_DDL["lepf_episode_no_update"])

    reopened = _open_manager(db_path)
    try:
        reopened_repository = LeveragedEtfJournalRepository(reopened)
        with pytest.raises(JournalIntegrityError, match="episode|column|payload|integrity"):
            reopened_repository.exposure_episode_records(initial.executor_id, started.episode_id)
    finally:
        reopened.engine.dispose()
