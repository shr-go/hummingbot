from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError
from sqlalchemy import inspect

from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.data_feed.yahoo_finance.acquisition import (
    AnchorCandidate,
    AnchorPollingCheckpoint,
    AnchorRepositoryKey,
    CheckpointIntegrityError,
    RevisionObservation,
)
from hummingbot.model import leveraged_etf_repository as durable_contract
from hummingbot.model.leveraged_etf_repository import AnchorIntegrityError, AnchorRevisionConflict
from hummingbot.model.sql_connection_manager import SQLConnectionManager, SQLConnectionType
from test.hummingbot.data_feed.yahoo_finance.test_pair_scoped_anchor import (
    OFFICIAL_CLOSE,
    finalized_pair,
    observation_pair,
)


def _open_manager(db_path: Path) -> SQLConnectionManager:
    return SQLConnectionManager(
        ClientConfigAdapter(ClientConfigMap()),
        SQLConnectionType.TRADE_FILLS,
        db_path=str(db_path),
    )


class _RevisionInt(int):
    pass


def _checkpoint_fields(revision: Any) -> dict[str, Any]:
    return {
        "integrity_version": 3,
        "pair_id": "sndk_snxx",
        "cycle_id": "xnys-2026-07-17",
        "target_session_date": "2026-07-17",
        "official_close_utc": "2026-07-17T20:00:00.000000Z",
        "deadline_utc": "2026-07-17T20:10:00.000000Z",
        "next_poll_utc": "2026-07-17T20:00:00.000000Z",
        "revision": revision,
    }


def _canonical_json_hash(value: dict[str, Any]) -> tuple[str, str]:
    payload_json = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return payload_json, hashlib.sha256(payload_json.encode("utf-8")).hexdigest()


def _checkpoint_payload_from_value(revision: Any):
    return durable_contract.CanonicalOpaqueAnchorPayloadV2.from_value(
        kind="ANCHOR_CHECKPOINT",
        contract_version_field="integrity_version",
        contract_version=3,
        value=_checkpoint_fields(revision),
    )


def _checkpoint_envelope(payload, revision: int = 1):
    fields = _checkpoint_fields(revision)
    return durable_contract.OpaqueAnchorCheckpointV2(
        key=durable_contract.AnchorStorageKeyV2(
            pair_id=fields["pair_id"],
            cycle_id=fields["cycle_id"],
        ),
        target_session_date=fields["target_session_date"],
        official_close_utc=fields["official_close_utc"],
        deadline_utc=fields["deadline_utc"],
        revision=revision,
        payload=payload,
    )


def _checkpoint_copied_with_payload_revision(revision: Any):
    valid_payload = _checkpoint_payload_from_value(1)
    payload_json, payload_hash = _canonical_json_hash(_checkpoint_fields(revision))
    copied_payload = valid_payload.model_copy(
        update={
            "payload_json": payload_json,
            "payload_hash": payload_hash,
        }
    )
    copied_checkpoint = _checkpoint_envelope(valid_payload).model_copy(
        update={"payload": copied_payload}
    )
    assert type(copied_payload) is durable_contract.CanonicalOpaqueAnchorPayloadV2
    assert type(copied_checkpoint) is durable_contract.OpaqueAnchorCheckpointV2
    return copied_payload, copied_checkpoint


class _CaptureMappings:
    @staticmethod
    def one_or_none():
        return None


class _CaptureResult:
    rowcount = 1

    @staticmethod
    def mappings():
        return _CaptureMappings()


class _CaptureTransaction:
    def __init__(self):
        self.is_active = True

    def commit(self):
        self.is_active = False

    def rollback(self):
        self.is_active = False


class _CaptureConnection:
    def __init__(self):
        self.begin_count = 0
        self.insert_parameters: list[dict[str, Any]] = []

    def begin(self):
        self.begin_count += 1
        return _CaptureTransaction()

    @staticmethod
    def exec_driver_sql(_statement: str):
        return None

    def execute(self, statement, parameters=None):
        sql = str(statement)
        if "SELECT pair_id, cycle_id" in sql:
            return _CaptureResult()
        if "INSERT INTO LeveragedEtfAnchorState" in sql:
            self.insert_parameters.append(dict(parameters))
            return _CaptureResult()
        raise AssertionError(f"unexpected capture SQL: {sql}")

    @staticmethod
    def close():
        return None


class _CaptureEngine:
    def __init__(self, connection: _CaptureConnection):
        self._connection = connection
        self.connect_count = 0

    def connect(self):
        self.connect_count += 1
        return self._connection


@pytest.fixture
def manager(tmp_path: Path):
    value = _open_manager(tmp_path / "pair-scoped-anchor.sqlite")
    try:
        yield value
    finally:
        value.engine.dispose()


@pytest.mark.parametrize(
    "invalid_revision",
    [True, 1.0, Decimal("1"), "1", _RevisionInt(1)],
    ids=("bool", "float", "decimal", "string", "int-subclass"),
)
def test_checkpoint_payload_from_value_requires_exact_builtin_revision(invalid_revision: Any):
    with pytest.raises((TypeError, ValueError, ValidationError), match="revision|JSON serializable"):
        _checkpoint_payload_from_value(invalid_revision)


@pytest.mark.parametrize(
    "invalid_revision",
    [True, 1.0, "1"],
    ids=("bool", "float", "string"),
)
def test_checkpoint_payload_from_canonical_json_requires_exact_builtin_revision(invalid_revision: Any):
    payload_json, payload_hash = _canonical_json_hash(_checkpoint_fields(invalid_revision))

    with pytest.raises((ValueError, ValidationError), match="revision"):
        durable_contract.CanonicalOpaqueAnchorPayloadV2.from_canonical_json(
            kind="ANCHOR_CHECKPOINT",
            contract_version_field="integrity_version",
            contract_version=3,
            payload_json=payload_json,
            payload_hash=payload_hash,
        )


@pytest.mark.parametrize(
    "invalid_revision",
    [True, 1.0, "1"],
    ids=("bool", "float", "string"),
)
def test_checkpoint_row_reload_rejects_noncanonical_payload_revision(invalid_revision: Any):
    fields = _checkpoint_fields(invalid_revision)
    payload_json, payload_hash = _canonical_json_hash(fields)
    row = {
        "pair_id": fields["pair_id"],
        "cycle_id": fields["cycle_id"],
        "schema_version": 2,
        "state_kind": "CHECKPOINT",
        "revision": 1,
        "target_session_date": fields["target_session_date"],
        "official_close_utc": fields["official_close_utc"],
        "deadline_utc": fields["deadline_utc"],
        "evidence_hash": None,
        "payload_version_field": "integrity_version",
        "payload_contract_version": 3,
        "payload_json": payload_json,
        "payload_hash": payload_hash,
        "created_at_utc": fields["next_poll_utc"],
        "updated_at_utc": fields["next_poll_utc"],
    }

    with pytest.raises(AnchorIntegrityError, match="revision|payload integrity"):
        durable_contract.PairScopedAnchorRepository._decode_anchor_row(row)


@pytest.mark.parametrize(
    "invalid_revision",
    [True, 1.0, Decimal("1"), "1", _RevisionInt(1)],
    ids=("bool", "float", "decimal", "string", "int-subclass"),
)
def test_repository_write_path_never_binds_noncanonical_payload_revision(invalid_revision: Any):
    connection = _CaptureConnection()
    repository = durable_contract.PairScopedAnchorRepository(
        SimpleNamespace(engine=_CaptureEngine(connection))
    )
    rejection = None

    try:
        payload = _checkpoint_payload_from_value(invalid_revision)
        checkpoint = _checkpoint_envelope(payload)
        repository.compare_and_set_opaque_checkpoint(
            checkpoint.key,
            checkpoint,
            expected_revision=0,
        )
    except (TypeError, ValueError, ValidationError, AnchorIntegrityError) as exception:
        rejection = exception

    assert rejection is not None
    assert connection.insert_parameters == []


@pytest.mark.parametrize(
    "invalid_revision",
    [True, 1.0, "1"],
    ids=("bool", "float", "string"),
)
def test_model_copy_payload_value_revalidates_checkpoint_revision(invalid_revision: Any):
    copied_payload, _ = _checkpoint_copied_with_payload_revision(invalid_revision)

    with pytest.raises((ValueError, ValidationError), match="revision"):
        copied_payload.value()


@pytest.mark.parametrize(
    "invalid_revision",
    [True, 1.0, "1"],
    ids=("bool", "float", "string"),
)
def test_model_copy_checkpoint_rejects_before_transaction_or_insert(invalid_revision: Any):
    _, copied_checkpoint = _checkpoint_copied_with_payload_revision(invalid_revision)
    connection = _CaptureConnection()
    engine = _CaptureEngine(connection)
    repository = durable_contract.PairScopedAnchorRepository(SimpleNamespace(engine=engine))

    with pytest.raises((ValueError, ValidationError, AnchorIntegrityError), match="revision|integrity"):
        repository.compare_and_set_opaque_checkpoint(
            copied_checkpoint.key,
            copied_checkpoint,
            expected_revision=0,
        )

    assert engine.connect_count == 0
    assert connection.begin_count == 0
    assert connection.insert_parameters == []


def test_exact_builtin_checkpoint_revision_round_trips_and_cas(
    manager: SQLConnectionManager,
):
    repository = durable_contract.PairScopedAnchorRepository(manager)
    first = _checkpoint_envelope(_checkpoint_payload_from_value(1), revision=1)

    assert repository.compare_and_set_opaque_checkpoint(first.key, first, expected_revision=0) == first
    loaded = repository.load_opaque(first.key)
    assert loaded == first
    assert type(loaded.payload.value()["revision"]) is int

    second = _checkpoint_envelope(_checkpoint_payload_from_value(2), revision=2)
    assert repository.compare_and_set_opaque_checkpoint(second.key, second, expected_revision=1) == second
    reloaded = repository.load_opaque(second.key)
    assert reloaded == second
    assert type(reloaded.payload.value()["revision"]) is int


class _F003OpaqueAdapter:
    """Test-only lossless bridge; F006 owns the sole production adapter."""

    def __init__(self, manager: SQLConnectionManager):
        repository_type = getattr(
            durable_contract,
            "PairScopedAnchorRepository",
            durable_contract.AnchorRepositoryV1,
        )
        self.repository = repository_type(manager)
        self.pair_scoped = hasattr(durable_contract, "AnchorStorageKeyV2")

    def _key(self, key: AnchorRepositoryKey) -> Any:
        if self.pair_scoped:
            return durable_contract.AnchorStorageKeyV2.model_validate(key.to_fields())
        return key.cycle_id

    def _payload(
        self,
        *,
        kind: str,
        version_field: str,
        contract_version: int,
        fields: dict[str, Any],
    ) -> Any:
        if self.pair_scoped:
            return durable_contract.CanonicalOpaqueAnchorPayloadV2.from_value(
                kind=kind,
                contract_version_field=version_field,
                contract_version=contract_version,
                value=fields,
            )
        legacy_fields = dict(fields)
        legacy_fields.setdefault("schema_version", contract_version)
        return durable_contract.CanonicalOpaquePayload.from_value(
            schema_version=legacy_fields["schema_version"],
            kind=kind,
            value=legacy_fields,
        )

    def compare_and_set_checkpoint(
        self,
        key: AnchorRepositoryKey,
        checkpoint: AnchorPollingCheckpoint,
        expected_revision: int,
    ) -> AnchorPollingCheckpoint:
        fields = checkpoint.to_recovery_fields()
        payload = self._payload(
            kind="ANCHOR_CHECKPOINT",
            version_field="integrity_version",
            contract_version=checkpoint.integrity_version,
            fields=fields,
        )
        if self.pair_scoped:
            storage_key = self._key(key)
            envelope = durable_contract.OpaqueAnchorCheckpointV2(
                key=storage_key,
                target_session_date=fields["target_session_date"],
                official_close_utc=fields["official_close_utc"],
                deadline_utc=fields["deadline_utc"],
                revision=checkpoint.revision,
                payload=payload,
            )
            self.repository.compare_and_set_opaque_checkpoint(
                storage_key,
                envelope,
                expected_revision=expected_revision,
            )
        else:
            envelope = durable_contract.OpaqueAnchorCheckpointV1(
                cycle_id=key.cycle_id,
                target_session_date=fields["target_session_date"],
                official_close_utc=fields["official_close_utc"],
                deadline_utc=fields["deadline_utc"],
                revision=checkpoint.revision,
                payload=payload,
            )
            self.repository.compare_and_set_opaque_checkpoint(envelope, expected_revision=expected_revision)
        return checkpoint

    def finalize_if_absent(
        self,
        key: AnchorRepositoryKey,
        candidate: AnchorCandidate,
        expected_revision: int,
    ) -> AnchorCandidate:
        fields = candidate.to_evidence_fields()
        payload = self._payload(
            kind="ANCHOR_RECORD",
            version_field="evidence_version",
            contract_version=candidate.evidence_version,
            fields=fields,
        )
        if self.pair_scoped:
            storage_key = self._key(key)
            envelope = durable_contract.OpaqueAnchorFinalizedV2(
                key=storage_key,
                target_session_date=fields["target_session_date"],
                official_close_utc=fields["official_close_utc"],
                deadline_utc=fields["deadline_utc"],
                revision=expected_revision,
                evidence_hash=candidate.evidence_hash,
                payload=payload,
            )
            finalized = self.repository.finalize_opaque_if_absent(
                storage_key,
                envelope,
                expected_revision=expected_revision,
            )
        else:
            envelope = durable_contract.OpaqueAnchorFinalizedV1(
                cycle_id=key.cycle_id,
                target_session_date=fields["target_session_date"],
                official_close_utc=fields["official_close_utc"],
                deadline_utc=fields["deadline_utc"],
                revision=expected_revision,
                evidence_hash=candidate.evidence_hash,
                payload=payload,
            )
            finalized = self.repository.finalize_opaque_if_absent(envelope, expected_revision=expected_revision)
        return AnchorCandidate.from_evidence_fields(finalized.payload.value())

    def load(self, key: AnchorRepositoryKey) -> AnchorPollingCheckpoint | AnchorCandidate | None:
        stored = self.repository.load_opaque(self._key(key))
        if stored is None:
            return None
        fields = stored.payload.value()
        if stored.payload.kind == "ANCHOR_CHECKPOINT":
            return AnchorPollingCheckpoint.from_recovery_fields(fields)
        return AnchorCandidate.from_evidence_fields(fields)

    def append_revision_observation(
        self,
        key: AnchorRepositoryKey,
        observation: RevisionObservation,
    ) -> None:
        fields = observation.to_fields()
        if self.pair_scoped:
            storage_key = self._key(key)
            envelope = durable_contract.OpaqueAnchorRevisionObservationV2(
                key=storage_key,
                evidence_hash=observation.evidence_hash,
                observed_at_utc=fields["observed_at_utc"],
                payload=self._payload(
                    kind="ANCHOR_REVISION_OBSERVATION",
                    version_field="schema_version",
                    contract_version=observation.schema_version,
                    fields=fields,
                ),
            )
            self.repository.append_opaque_revision_observation(storage_key, envelope)
        else:
            self.repository.append_revision_observation(
                key.cycle_id,
                observation.evidence_hash,
                fields["observed_at_utc"],
            )

    def revision_observations(self, key: AnchorRepositoryKey) -> tuple[RevisionObservation, ...]:
        if self.pair_scoped:
            values = self.repository.opaque_revision_observations(self._key(key))
            return tuple(RevisionObservation.from_fields(value.payload.value()) for value in values)
        values = self.repository.revision_observations(key.cycle_id)
        return tuple(RevisionObservation.from_fields(value.model_dump(mode="json")) for value in values)


@pytest.mark.asyncio
async def test_same_cycle_empty_reset_and_cas_are_independent_for_two_pairs(
    manager: SQLConnectionManager,
):
    _, sndk_empty, _ = await finalized_pair("sndk_snxx", "SNDK", "SNXX", 1)
    _, intc_empty, _ = await finalized_pair("intc_intw", "INTC", "INTW", 5)
    sndk_key = AnchorRepositoryKey.create("sndk_snxx", sndk_empty.cycle_id)
    intc_key = AnchorRepositoryKey.create("intc_intw", intc_empty.cycle_id)
    repository = _F003OpaqueAdapter(manager)

    repository.compare_and_set_checkpoint(sndk_key, sndk_empty, expected_revision=0)
    repository.compare_and_set_checkpoint(intc_key, intc_empty, expected_revision=0)
    sndk_reset = replace(
        sndk_empty,
        attempt=1,
        next_poll_utc=sndk_empty.official_close_utc + timedelta(seconds=1),
        revision=2,
        integrity_hash=None,
    )
    repository.compare_and_set_checkpoint(sndk_key, sndk_reset, expected_revision=1)

    assert repository.load(sndk_key) == sndk_reset
    assert repository.load(intc_key) == intc_empty


@pytest.mark.asyncio
async def test_finalized_evidence_and_revisions_round_trip_after_reopen_by_pair(tmp_path: Path):
    db_path = tmp_path / "pair-finalized-reopen.sqlite"
    acquisition, empty, final = await finalized_pair("sndk_snxx", "SNDK", "SNXX", 1)
    key = AnchorRepositoryKey.create("sndk_snxx", empty.cycle_id)
    manager = _open_manager(db_path)
    repository = _F003OpaqueAdapter(manager)
    repository.compare_and_set_checkpoint(key, empty, expected_revision=0)
    assert repository.finalize_if_absent(key, final.candidate, expected_revision=1) == final.candidate
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        restored = _F003OpaqueAdapter(reopened)
        assert restored.load(key) == final.candidate
        stock, etf = observation_pair("SNDK", "SNXX", OFFICIAL_CLOSE + timedelta(seconds=90), "9", "a", 5)
        etf = replace(etf, close=etf.close + 1)
        revision = acquisition.assess_finalized(final.candidate, stock, etf).revision_observation
        restored.append_revision_observation(key, revision)
        restored.append_revision_observation(key, revision)
        assert restored.revision_observations(key) == (revision,)
    finally:
        reopened.engine.dispose()


@pytest.mark.asyncio
async def test_wrong_pair_key_and_payload_are_rejected_before_mutation(
    manager: SQLConnectionManager,
):
    _, sndk_empty, _ = await finalized_pair("sndk_snxx", "SNDK", "SNXX", 1)
    _, intc_empty, _ = await finalized_pair("intc_intw", "INTC", "INTW", 5)
    sndk_key = AnchorRepositoryKey.create("sndk_snxx", sndk_empty.cycle_id)
    repository = _F003OpaqueAdapter(manager)

    with pytest.raises((AnchorIntegrityError, AnchorRevisionConflict, CheckpointIntegrityError), match="pair|key"):
        repository.compare_and_set_checkpoint(sndk_key, intc_empty, expected_revision=0)
    assert repository.load(sndk_key) is None


def test_schema_uses_composite_pair_cycle_identity_and_foreign_key(manager: SQLConnectionManager):
    inspector = inspect(manager.engine)
    state_pk = tuple(inspector.get_pk_constraint("LeveragedEtfAnchorState")["constrained_columns"])
    observation_pk = tuple(
        inspector.get_pk_constraint("LeveragedEtfAnchorRevisionObservation")["constrained_columns"]
    )
    observation_fks = inspector.get_foreign_keys("LeveragedEtfAnchorRevisionObservation")
    state_indexes = inspector.get_indexes("LeveragedEtfAnchorState")
    observation_indexes = inspector.get_indexes("LeveragedEtfAnchorRevisionObservation")

    assert state_pk == ("pair_id", "cycle_id")
    assert observation_pk[:2] == ("pair_id", "cycle_id")
    assert any(
        tuple(foreign_key["constrained_columns"]) == ("pair_id", "cycle_id")
        and tuple(foreign_key["referred_columns"]) == ("pair_id", "cycle_id")
        for foreign_key in observation_fks
    )
    assert any(tuple(index["column_names"])[:2] == ("pair_id", "cycle_id") for index in state_indexes)
    assert any(tuple(index["column_names"])[:2] == ("pair_id", "cycle_id") for index in observation_indexes)


def test_cycle_only_repository_surface_fails_closed(manager: SQLConnectionManager):
    legacy_repository = durable_contract.AnchorRepositoryV1(manager)

    with pytest.raises(AnchorIntegrityError, match="pair|cycle-only|legacy"):
        legacy_repository.load_opaque("xnys-2026-07-17")


def test_historical_v1_checkpoint_dto_still_deserializes():
    historical = durable_contract.AnchorPollingCheckpointV1(
        cycle_id="xnys-2026-07-17",
        target_session_date="2026-07-17",
        official_close_utc="2026-07-17T20:00:00.000000Z",
        deadline_utc="2026-07-17T20:10:00.000000Z",
        attempt=0,
        next_poll_utc="2026-07-17T20:00:00.000000Z",
        confirmation_count=0,
        candidate_stock_close=None,
        candidate_etf_close=None,
        candidate_stock_raw_response_hash=None,
        candidate_etf_raw_response_hash=None,
        stock_received_at_utc=None,
        etf_received_at_utc=None,
        revision=1,
    )

    assert historical.schema_version == 1
