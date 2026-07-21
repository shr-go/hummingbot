import json
import sqlite3
from pathlib import Path

import pytest
from pydantic import ValidationError
from sqlalchemy import text

from hummingbot.model.leveraged_etf_persistence import SQLITE_GUARD_DDL
from hummingbot.model.leveraged_etf_repository import (
    AnchorIntegrityError,
    CanonicalOpaqueAnchorPayloadV2,
    PairScopedAnchorRepository,
    _reject_opaque_anchor_secret_bearing_fields,
)

from .test_journal_repository import (
    _canonical_json_hash,
    _open_manager,
    _pair_scoped_checkpoint,
    _pair_scoped_final,
    _pair_scoped_revision,
)


SURFACE_CASES = (
    ("checkpoint", False, "api-key"),
    ("checkpoint", True, "credentials"),
    ("finalized", False, "access_token"),
    ("finalized", True, "clientSecret"),
    ("observation", False, "db_password"),
    ("observation", True, "private-key-pem"),
)


def _surface(surface: str):
    key, checkpoint = _pair_scoped_checkpoint("sndk_snxx", revision=1)
    if surface == "checkpoint":
        return key, checkpoint
    finalized = _pair_scoped_final(key, revision=1, evidence_hash="7" * 64)
    if surface == "finalized":
        return key, finalized
    return key, _pair_scoped_revision(
        key,
        evidence_hash="8" * 64,
        observed_at_utc="2026-07-18T20:03:00.000000Z",
    )


def _secret_value(envelope, secret_key: str, nested: bool) -> dict:
    value = envelope.payload.value()
    if nested:
        value["provider_state"] = {"history": [{secret_key: "must-not-persist"}]}
    else:
        value[secret_key] = "must-not-persist"
    return value


def _canonical_secret(envelope, secret_key: str, nested: bool) -> tuple[str, str]:
    return _canonical_json_hash(_secret_value(envelope, secret_key, nested))


def _forged_secret_payload(envelope, secret_key: str, nested: bool) -> CanonicalOpaqueAnchorPayloadV2:
    payload_json, payload_hash = _canonical_secret(envelope, secret_key, nested)
    return CanonicalOpaqueAnchorPayloadV2.model_construct(
        schema_version=2,
        kind=envelope.payload.kind,
        contract_version_field=envelope.payload.contract_version_field,
        contract_version=envelope.payload.contract_version,
        payload_json=payload_json,
        payload_hash=payload_hash,
    )


def _forged_secret_envelope(surface: str, secret_key: str, nested: bool):
    key, envelope = _surface(surface)
    return key, envelope.model_copy(
        update={"payload": _forged_secret_payload(envelope, secret_key, nested)},
    )


@pytest.mark.parametrize(("surface", "nested", "secret_key"), SURFACE_CASES)
def test_t007_from_value_rejects_secret_bearing_anchor_keys(surface: str, nested: bool, secret_key: str):
    _, envelope = _surface(surface)

    with pytest.raises((ValueError, ValidationError), match="secret"):
        CanonicalOpaqueAnchorPayloadV2.from_value(
            kind=envelope.payload.kind,
            contract_version_field=envelope.payload.contract_version_field,
            contract_version=envelope.payload.contract_version,
            value=_secret_value(envelope, secret_key, nested),
        )


@pytest.mark.parametrize(("surface", "nested", "secret_key"), SURFACE_CASES)
def test_t007_canonical_decode_rejects_hash_valid_secret_bearing_anchor_keys(
    surface: str,
    nested: bool,
    secret_key: str,
):
    _, envelope = _surface(surface)
    payload_json, payload_hash = _canonical_secret(envelope, secret_key, nested)

    with pytest.raises((ValueError, ValidationError), match="secret"):
        CanonicalOpaqueAnchorPayloadV2.from_canonical_json(
            kind=envelope.payload.kind,
            contract_version_field=envelope.payload.contract_version_field,
            contract_version=envelope.payload.contract_version,
            payload_json=payload_json,
            payload_hash=payload_hash,
        )


@pytest.mark.parametrize(("surface", "nested", "secret_key"), SURFACE_CASES)
def test_t007_value_revalidation_rejects_forged_secret_bearing_anchor_keys(
    surface: str,
    nested: bool,
    secret_key: str,
):
    _, envelope = _surface(surface)
    forged = _forged_secret_payload(envelope, secret_key, nested)

    with pytest.raises(ValueError, match="secret"):
        forged.value()


def test_t007_normalized_secret_key_vocabulary_is_rejected():
    _, checkpoint = _surface("checkpoint")
    secret_keys = (
        "exchangeApiKey",
        "exchange_api_keys",
        "credential",
        "user_credentials",
        "access-token",
        "refreshTokens",
        "client_secret",
        "signing_secret_keys",
        "sharedSecrets",
        "db.password",
        "backup_passwords",
        "walletPassphrase",
        "wallet_passphrases",
        "signing-private-key",
        "signing_private_keys",
        "private_key_pem",
        "pkcs8",
    )

    for secret_key in secret_keys:
        with pytest.raises((ValueError, ValidationError), match="secret"):
            CanonicalOpaqueAnchorPayloadV2.from_value(
                kind=checkpoint.payload.kind,
                contract_version_field=checkpoint.payload.contract_version_field,
                contract_version=checkpoint.payload.contract_version,
                value=_secret_value(checkpoint, secret_key, nested=True),
            )


def test_t007_secret_key_walk_is_cycle_safe():
    value = {"provider_state": {}}
    value["provider_state"]["self"] = value

    _reject_opaque_anchor_secret_bearing_fields(value)


def test_t007_secret_key_walk_rejects_secret_on_a_cycle_sibling():
    value = {"provider_state": {}}
    value["provider_state"]["self"] = value
    value["provider_state"]["nested"] = {"API.Keys": "must-not-persist"}

    with pytest.raises(ValueError, match="secret"):
        _reject_opaque_anchor_secret_bearing_fields(value)


@pytest.mark.parametrize("surface", ("checkpoint", "finalized", "observation"))
def test_t007_envelope_revalidation_rejects_forged_secret_payload(surface: str):
    _, envelope = _forged_secret_envelope(surface, "provider-auth-token", nested=True)

    with pytest.raises(AnchorIntegrityError, match="secret"):
        PairScopedAnchorRepository._revalidate_opaque_write_envelope(
            envelope,
            type(envelope),
            surface,
        )


def _row_count(manager, table_name: str) -> int:
    with manager.engine.connect() as connection:
        return connection.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar_one()


@pytest.mark.parametrize("surface", ("checkpoint", "finalized", "observation"))
def test_t007_rejected_secret_write_leaves_no_new_row(tmp_path: Path, surface: str):
    manager = _open_manager(tmp_path / f"secret-write-{surface}.sqlite")
    try:
        repository = PairScopedAnchorRepository(manager)
        key, checkpoint = _pair_scoped_checkpoint("sndk_snxx", revision=1)
        if surface == "checkpoint":
            _, candidate = _forged_secret_envelope(surface, "api_key", nested=False)
            with pytest.raises(AnchorIntegrityError, match="secret"):
                repository.compare_and_set_opaque_checkpoint(key, candidate, expected_revision=0)
            assert _row_count(manager, "LeveragedEtfAnchorState") == 0
            return

        repository.compare_and_set_opaque_checkpoint(key, checkpoint, expected_revision=0)
        finalized = _pair_scoped_final(key, revision=1, evidence_hash="7" * 64)
        if surface == "finalized":
            candidate = finalized.model_copy(
                update={"payload": _forged_secret_payload(finalized, "client_secret", nested=True)},
            )
            with pytest.raises(AnchorIntegrityError, match="secret"):
                repository.finalize_opaque_if_absent(key, candidate, expected_revision=1)
            assert _row_count(manager, "LeveragedEtfAnchorState") == 1
            assert repository.load_opaque(key) == checkpoint
            return

        repository.finalize_opaque_if_absent(key, finalized, expected_revision=1)
        observation = _pair_scoped_revision(
            key,
            evidence_hash="8" * 64,
            observed_at_utc="2026-07-18T20:03:00.000000Z",
        )
        candidate = observation.model_copy(
            update={"payload": _forged_secret_payload(observation, "private_key_pem", nested=True)},
        )
        with pytest.raises(AnchorIntegrityError, match="secret"):
            repository.append_opaque_revision_observation(key, candidate)
        assert _row_count(manager, "LeveragedEtfAnchorRevisionObservation") == 0
        assert repository.opaque_revision_observations(key) == ()
    finally:
        manager.engine.dispose()


def _seed_anchor_surface(db_path: Path, surface: str):
    manager = _open_manager(db_path)
    try:
        repository = PairScopedAnchorRepository(manager)
        key, checkpoint = _pair_scoped_checkpoint("sndk_snxx", revision=1)
        repository.compare_and_set_opaque_checkpoint(key, checkpoint, expected_revision=0)
        if surface == "checkpoint":
            return key
        finalized = _pair_scoped_final(key, revision=1, evidence_hash="7" * 64)
        repository.finalize_opaque_if_absent(key, finalized, expected_revision=1)
        if surface == "observation":
            repository.append_opaque_revision_observation(
                key,
                _pair_scoped_revision(
                    key,
                    evidence_hash="8" * 64,
                    observed_at_utc="2026-07-18T20:03:00.000000Z",
                ),
            )
        return key
    finally:
        manager.engine.dispose()


def _inject_hash_valid_secret(db_path: Path, surface: str, pair_id: str, cycle_id: str) -> None:
    table_name = (
        "LeveragedEtfAnchorRevisionObservation"
        if surface == "observation"
        else "LeveragedEtfAnchorState"
    )
    trigger_name = {
        "checkpoint": None,
        "finalized": "lepf_anchor_finalized_no_update",
        "observation": "lepf_anchor_observation_no_update",
    }[surface]
    with sqlite3.connect(db_path) as connection:
        if trigger_name is not None:
            connection.execute(f'DROP TRIGGER "{trigger_name}"')
        select_fields = (
            "payload_json, evidence_hash, observed_at_utc"
            if surface == "observation"
            else "payload_json, evidence_hash, NULL"
        )
        row = connection.execute(
            f'SELECT {select_fields} FROM "{table_name}" WHERE pair_id = ? AND cycle_id = ?',
            (pair_id, cycle_id),
        ).fetchone()
        payload_json = row[0]
        value = json.loads(payload_json)
        value["provider_state"] = {"nested": [{"legacy-private-key": "must-fail-closed"}]}
        rewritten_json, rewritten_hash = _canonical_json_hash(value)
        identity_sql = "pair_id = ? AND cycle_id = ?"
        identity_parameters = [pair_id, cycle_id]
        if surface == "observation":
            identity_sql += " AND evidence_hash = ? AND observed_at_utc = ?"
            identity_parameters.extend((row[1], row[2]))
        connection.execute(
            f'UPDATE "{table_name}" SET payload_json = ?, payload_hash = ? '
            f"WHERE {identity_sql}",
            (rewritten_json, rewritten_hash, *identity_parameters),
        )
        if trigger_name is not None:
            connection.execute(SQLITE_GUARD_DDL[trigger_name])


@pytest.mark.parametrize("surface", ("checkpoint", "finalized", "observation"))
def test_t007_hash_valid_legacy_secret_payload_fails_closed_after_reopen(tmp_path: Path, surface: str):
    db_path = tmp_path / f"legacy-secret-{surface}.sqlite"
    key = _seed_anchor_surface(db_path, surface)
    _inject_hash_valid_secret(db_path, surface, key.pair_id, key.cycle_id)

    reopened = _open_manager(db_path)
    try:
        repository = PairScopedAnchorRepository(reopened)
        with pytest.raises(AnchorIntegrityError, match="secret"):
            if surface == "observation":
                repository.opaque_revision_observations(key)
            else:
                repository.load_opaque(key)
    finally:
        reopened.engine.dispose()


def _with_compatible_opaque_fields(envelope):
    value = envelope.payload.value()
    value["provider_state"] = {
        "api_key_hint": "stored-elsewhere",
        "credential_status": "configured",
        "token_count": 0,
        "secretary": "operator",
        "password_policy": "external",
        "passphrase_format": "external",
        "public_key": "public-material",
        "pem_format": "PKCS8",
        "pkcs8_format": "text",
    }
    payload = CanonicalOpaqueAnchorPayloadV2.from_value(
        kind=envelope.payload.kind,
        contract_version_field=envelope.payload.contract_version_field,
        contract_version=envelope.payload.contract_version,
        value=value,
    )
    return type(envelope).model_validate(
        {**envelope.model_dump(mode="python"), "payload": payload},
    )


def test_t007_non_secret_opaque_payloads_round_trip_after_reopen(tmp_path: Path):
    db_path = tmp_path / "non-secret-compatible.sqlite"
    manager = _open_manager(db_path)
    key, checkpoint = _pair_scoped_checkpoint("sndk_snxx", revision=1)
    checkpoint = _with_compatible_opaque_fields(checkpoint)
    finalized = _with_compatible_opaque_fields(
        _pair_scoped_final(key, revision=1, evidence_hash="7" * 64)
    )
    observation = _with_compatible_opaque_fields(
        _pair_scoped_revision(
            key,
            evidence_hash="8" * 64,
            observed_at_utc="2026-07-18T20:03:00.000000Z",
        )
    )
    repository = PairScopedAnchorRepository(manager)
    repository.compare_and_set_opaque_checkpoint(key, checkpoint, expected_revision=0)
    repository.finalize_opaque_if_absent(key, finalized, expected_revision=1)
    repository.append_opaque_revision_observation(key, observation)
    manager.engine.dispose()

    reopened = _open_manager(db_path)
    try:
        repository = PairScopedAnchorRepository(reopened)
        assert repository.load_opaque(key) == finalized
        assert repository.opaque_revision_observations(key) == (observation,)
    finally:
        reopened.engine.dispose()


_T008_API_KEY_FIELD = "T008-API-KEY-FIELD-7D38-api_key"
_T008_API_KEY_VALUE = "T008-API-KEY-VALUE-9F271"
_T008_PKCS8_FIELD = "T008-PKCS8-FIELD-91AC-private_key_pem"
_T008_PKCS8_MARKER = "T008-PKCS8-VALUE-B31C2"
_T008_PKCS8_VALUE = (
    "-----BEGIN PRIVATE KEY-----\n"
    f"{_T008_PKCS8_MARKER}\n"
    "-----END PRIVATE KEY-----"
)
_T008_SECRET_SENTINELS = (
    _T008_API_KEY_FIELD,
    _T008_API_KEY_VALUE,
    _T008_PKCS8_FIELD,
    _T008_PKCS8_MARKER,
    _T008_PKCS8_VALUE,
    "-----BEGIN PRIVATE KEY-----",
    "-----END PRIVATE KEY-----",
)


def _t008_secret_value(envelope) -> dict:
    value = envelope.payload.value()
    value["provider_state"] = {
        "history": [
            {
                _T008_API_KEY_FIELD: _T008_API_KEY_VALUE,
                _T008_PKCS8_FIELD: _T008_PKCS8_VALUE,
            }
        ]
    }
    return value


def _t008_canonical_secret(envelope) -> tuple[str, str]:
    return _canonical_json_hash(_t008_secret_value(envelope))


def _t008_forged_payload(envelope) -> CanonicalOpaqueAnchorPayloadV2:
    payload_json, payload_hash = _t008_canonical_secret(envelope)
    return CanonicalOpaqueAnchorPayloadV2.model_construct(
        schema_version=2,
        kind=envelope.payload.kind,
        contract_version_field=envelope.payload.contract_version_field,
        contract_version=envelope.payload.contract_version,
        payload_json=payload_json,
        payload_hash=payload_hash,
    )


def _t008_forged_envelope(surface: str):
    key, envelope = _surface(surface)
    return key, envelope.model_copy(update={"payload": _t008_forged_payload(envelope)})


def _t008_capture_secret_rejection(operation):
    try:
        operation()
    except (ValueError, ValidationError, AnchorIntegrityError) as error:
        assert "secret" in str(error).lower()
        return error
    pytest.fail("secret-bearing opaque anchor payload was accepted")


def _t008_nested_renderings(value) -> tuple[str, ...]:
    renderings = []
    pending = [value]
    visited = set()
    while pending:
        candidate = pending.pop()
        if isinstance(candidate, str):
            renderings.append(candidate)
        elif isinstance(candidate, dict):
            if id(candidate) in visited:
                continue
            visited.add(id(candidate))
            pending.extend(candidate.keys())
            pending.extend(candidate.values())
        elif isinstance(candidate, (list, tuple, set)):
            if id(candidate) in visited:
                continue
            visited.add(id(candidate))
            pending.extend(candidate)
        else:
            renderings.extend((str(candidate), repr(candidate)))
    return tuple(renderings)


def _t008_assert_secret_safe(error: BaseException, payload_json: str) -> None:
    escaped_payload_json = json.dumps(payload_json, ensure_ascii=False)
    forbidden = (*_T008_SECRET_SENTINELS, payload_json, escaped_payload_json)
    pending = [error]
    visited = set()
    while pending:
        candidate = pending.pop()
        if candidate is None or id(candidate) in visited:
            continue
        visited.add(id(candidate))
        renderings = [
            str(candidate),
            repr(candidate),
            repr(candidate.args),
            json.dumps(candidate.args, default=repr, ensure_ascii=False),
        ]
        if isinstance(candidate, ValidationError):
            structured = candidate.errors(
                include_url=True,
                include_context=True,
                include_input=True,
            )
            renderings.extend(_t008_nested_renderings(structured))
            renderings.extend(
                (
                    repr(structured),
                    json.dumps(structured, default=repr, ensure_ascii=False),
                    candidate.json(
                        include_url=True,
                        include_context=True,
                        include_input=True,
                    ),
                )
            )
        if any(secret in rendering for secret in forbidden for rendering in renderings):
            pytest.fail("opaque anchor error surface exposed a T008 secret sentinel")
        pending.extend((candidate.__cause__, candidate.__context__))


def test_t008_direct_payload_construction_redacts_secret_input():
    _, checkpoint = _surface("checkpoint")
    payload_json, payload_hash = _t008_canonical_secret(checkpoint)

    error = _t008_capture_secret_rejection(
        lambda: CanonicalOpaqueAnchorPayloadV2(
            schema_version=2,
            kind=checkpoint.payload.kind,
            contract_version_field=checkpoint.payload.contract_version_field,
            contract_version=checkpoint.payload.contract_version,
            payload_json=payload_json,
            payload_hash=payload_hash,
        )
    )

    _t008_assert_secret_safe(error, payload_json)


def test_t008_from_value_redacts_secret_input():
    _, checkpoint = _surface("checkpoint")
    payload_json, _ = _t008_canonical_secret(checkpoint)

    error = _t008_capture_secret_rejection(
        lambda: CanonicalOpaqueAnchorPayloadV2.from_value(
            kind=checkpoint.payload.kind,
            contract_version_field=checkpoint.payload.contract_version_field,
            contract_version=checkpoint.payload.contract_version,
            value=_t008_secret_value(checkpoint),
        )
    )

    _t008_assert_secret_safe(error, payload_json)


def test_t008_canonical_decode_redacts_secret_input():
    _, checkpoint = _surface("checkpoint")
    payload_json, payload_hash = _t008_canonical_secret(checkpoint)

    error = _t008_capture_secret_rejection(
        lambda: CanonicalOpaqueAnchorPayloadV2.from_canonical_json(
            kind=checkpoint.payload.kind,
            contract_version_field=checkpoint.payload.contract_version_field,
            contract_version=checkpoint.payload.contract_version,
            payload_json=payload_json,
            payload_hash=payload_hash,
        )
    )

    _t008_assert_secret_safe(error, payload_json)


def test_t008_value_revalidation_redacts_secret_input():
    _, checkpoint = _surface("checkpoint")
    forged = _t008_forged_payload(checkpoint)

    error = _t008_capture_secret_rejection(forged.value)

    _t008_assert_secret_safe(error, forged.payload_json)


@pytest.mark.parametrize("surface", ("checkpoint", "finalized", "observation"))
def test_t008_envelope_revalidation_redacts_secret_input(surface: str):
    _, envelope = _t008_forged_envelope(surface)

    error = _t008_capture_secret_rejection(
        lambda: PairScopedAnchorRepository._revalidate_opaque_write_envelope(
            envelope,
            type(envelope),
            surface,
        )
    )

    _t008_assert_secret_safe(error, envelope.payload.payload_json)


@pytest.mark.parametrize("surface", ("checkpoint", "finalized", "observation"))
def test_t008_write_rejection_is_redacted_and_preserves_storage(tmp_path: Path, surface: str):
    manager = _open_manager(tmp_path / f"redacted-secret-write-{surface}.sqlite")
    try:
        repository = PairScopedAnchorRepository(manager)
        key, checkpoint = _surface("checkpoint")
        if surface == "checkpoint":
            _, candidate = _t008_forged_envelope(surface)
            error = _t008_capture_secret_rejection(
                lambda: repository.compare_and_set_opaque_checkpoint(key, candidate, expected_revision=0)
            )
            assert _row_count(manager, "LeveragedEtfAnchorState") == 0
        else:
            repository.compare_and_set_opaque_checkpoint(key, checkpoint, expected_revision=0)
            _, finalized = _surface("finalized")
            if surface == "finalized":
                candidate = finalized.model_copy(update={"payload": _t008_forged_payload(finalized)})
                error = _t008_capture_secret_rejection(
                    lambda: repository.finalize_opaque_if_absent(key, candidate, expected_revision=1)
                )
                assert repository.load_opaque(key) == checkpoint
            else:
                repository.finalize_opaque_if_absent(key, finalized, expected_revision=1)
                _, observation = _t008_forged_envelope(surface)
                candidate = observation
                error = _t008_capture_secret_rejection(
                    lambda: repository.append_opaque_revision_observation(key, candidate)
                )
                assert _row_count(manager, "LeveragedEtfAnchorRevisionObservation") == 0
                assert repository.opaque_revision_observations(key) == ()
        _t008_assert_secret_safe(error, candidate.payload.payload_json)
    finally:
        manager.engine.dispose()


def _t008_inject_hash_valid_secret(db_path: Path, surface: str, pair_id: str, cycle_id: str) -> str:
    table_name = (
        "LeveragedEtfAnchorRevisionObservation"
        if surface == "observation"
        else "LeveragedEtfAnchorState"
    )
    trigger_name = {
        "checkpoint": None,
        "finalized": "lepf_anchor_finalized_no_update",
        "observation": "lepf_anchor_observation_no_update",
    }[surface]
    with sqlite3.connect(db_path) as connection:
        if trigger_name is not None:
            connection.execute(f'DROP TRIGGER "{trigger_name}"')
        select_fields = (
            "payload_json, evidence_hash, observed_at_utc"
            if surface == "observation"
            else "payload_json, evidence_hash, NULL"
        )
        row = connection.execute(
            f'SELECT {select_fields} FROM "{table_name}" WHERE pair_id = ? AND cycle_id = ?',
            (pair_id, cycle_id),
        ).fetchone()
        value = json.loads(row[0])
        value["provider_state"] = {
            "history": [
                {
                    _T008_API_KEY_FIELD: _T008_API_KEY_VALUE,
                    _T008_PKCS8_FIELD: _T008_PKCS8_VALUE,
                }
            ]
        }
        payload_json, payload_hash = _canonical_json_hash(value)
        identity_sql = "pair_id = ? AND cycle_id = ?"
        identity_parameters = [pair_id, cycle_id]
        if surface == "observation":
            identity_sql += " AND evidence_hash = ? AND observed_at_utc = ?"
            identity_parameters.extend((row[1], row[2]))
        connection.execute(
            f'UPDATE "{table_name}" SET payload_json = ?, payload_hash = ? '
            f"WHERE {identity_sql}",
            (payload_json, payload_hash, *identity_parameters),
        )
        if trigger_name is not None:
            connection.execute(SQLITE_GUARD_DDL[trigger_name])
    return payload_json


@pytest.mark.parametrize("surface", ("checkpoint", "finalized", "observation"))
def test_t008_hash_valid_legacy_secret_reopen_error_is_redacted(tmp_path: Path, surface: str):
    db_path = tmp_path / f"redacted-legacy-secret-{surface}.sqlite"
    key = _seed_anchor_surface(db_path, surface)
    payload_json = _t008_inject_hash_valid_secret(db_path, surface, key.pair_id, key.cycle_id)

    reopened = _open_manager(db_path)
    try:
        repository = PairScopedAnchorRepository(reopened)
        if surface == "observation":
            operation = lambda: repository.opaque_revision_observations(key)
        else:
            operation = lambda: repository.load_opaque(key)
        error = _t008_capture_secret_rejection(operation)
        _t008_assert_secret_safe(error, payload_json)
    finally:
        reopened.engine.dispose()
