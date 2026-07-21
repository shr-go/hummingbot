"""The sole runtime bridge from F003 anchor values to F004 opaque storage.

F003 intentionally owns the typed Yahoo/NAV evidence and F004 intentionally
does not import it.  This adapter is the narrow composition boundary: it
accepts only the pair-scoped F003 port and only the pair-scoped F004 storage
repository.  In particular, it never attempts a cycle-only lookup or a V1
write.
"""

from __future__ import annotations

from typing import Any

from hummingbot.data_feed.yahoo_finance.acquisition import (
    AnchorCandidate,
    AnchorPollingCheckpoint,
    AnchorRepositoryKey,
    CheckpointIntegrityError,
    RevisionObservation,
)
from hummingbot.model.leveraged_etf_repository import (
    AnchorIntegrityError,
    AnchorStorageKeyV2,
    CanonicalOpaqueAnchorPayloadV2,
    OpaqueAnchorCheckpointV2,
    OpaqueAnchorFinalizedV2,
    OpaqueAnchorRevisionObservationV2,
    PairScopedAnchorRepository,
)


__all__ = ["F004AnchorRepositoryAdapter"]


class F004AnchorRepositoryAdapter:
    """Lossless, fail-closed implementation of F003's ``AnchorRepositoryV2``.

    The adapter does not translate F004 conflicts into a looser result.  CAS,
    immutable-finalization, and append-only failures retain the durable
    repository's exact failure semantics.
    """

    def __init__(self, repository: PairScopedAnchorRepository) -> None:
        if not isinstance(repository, PairScopedAnchorRepository):
            raise TypeError("repository must be an F004 PairScopedAnchorRepository")
        self._repository = repository

    @staticmethod
    def _require_key(key: AnchorRepositoryKey) -> AnchorRepositoryKey:
        if type(key) is not AnchorRepositoryKey:
            raise TypeError("anchor operations require an exact AnchorRepositoryKey v2")
        key.to_fields()
        return key

    @classmethod
    def _storage_key(cls, key: AnchorRepositoryKey) -> AnchorStorageKeyV2:
        fields = cls._require_key(key).to_fields()
        try:
            storage_key = AnchorStorageKeyV2.model_validate(fields)
        except Exception as exception:
            raise CheckpointIntegrityError(
                f"could not construct pair-scoped F004 storage key: {exception}"
            ) from exception
        if storage_key.model_dump(mode="python") != fields:
            raise CheckpointIntegrityError("F004 storage key changed F003 pair/cycle identity")
        return storage_key

    @staticmethod
    def _payload(
        *,
        kind: str,
        version_field: str,
        contract_version: int,
        fields: dict[str, Any],
    ) -> CanonicalOpaqueAnchorPayloadV2:
        try:
            return CanonicalOpaqueAnchorPayloadV2.from_value(
                kind=kind,
                contract_version_field=version_field,
                contract_version=contract_version,
                value=fields,
            )
        except Exception as exception:
            raise CheckpointIntegrityError(
                f"could not encode lossless {kind} opaque payload: {exception}"
            ) from exception

    @staticmethod
    def _assert_stored_key(key: AnchorRepositoryKey, stored_key: AnchorStorageKeyV2) -> None:
        if type(stored_key) is not AnchorStorageKeyV2:
            raise AnchorIntegrityError("stored anchor has an invalid pair-scoped storage key")
        fields = key.to_fields()
        if stored_key.model_dump(mode="python") != fields:
            raise AnchorIntegrityError("stored anchor pair/cycle identity does not match the requested key")

    @staticmethod
    def _checkpoint_from_fields(
        key: AnchorRepositoryKey,
        fields: dict[str, Any],
    ) -> AnchorPollingCheckpoint:
        checkpoint = AnchorPollingCheckpoint.from_recovery_fields(fields)
        key.validate_checkpoint(checkpoint)
        if checkpoint.to_recovery_fields() != fields:
            raise CheckpointIntegrityError("F004 checkpoint payload is not a lossless F003 recovery record")
        return checkpoint

    @staticmethod
    def _candidate_from_fields(key: AnchorRepositoryKey, fields: dict[str, Any]) -> AnchorCandidate:
        candidate = AnchorCandidate.from_evidence_fields(fields)
        key.validate_candidate(candidate)
        if candidate.to_evidence_fields() != fields:
            raise CheckpointIntegrityError("F004 finalized payload is not lossless F003 evidence")
        return candidate

    @staticmethod
    def _revision_from_fields(
        key: AnchorRepositoryKey,
        fields: dict[str, Any],
    ) -> RevisionObservation:
        revision = RevisionObservation.from_fields(fields)
        key.validate_revision_observation(revision)
        if revision.to_fields() != fields:
            raise CheckpointIntegrityError("F004 revision payload is not lossless F003 evidence")
        return revision

    def load(self, key: AnchorRepositoryKey) -> AnchorPollingCheckpoint | AnchorCandidate | None:
        storage_key = self._storage_key(key)
        stored = self._repository.load_opaque(storage_key)
        if stored is None:
            return None
        self._assert_stored_key(key, stored.key)
        fields = stored.payload.value()
        if type(stored) is OpaqueAnchorCheckpointV2:
            if stored.payload.kind != "ANCHOR_CHECKPOINT":
                raise AnchorIntegrityError("checkpoint envelope contains the wrong opaque payload kind")
            return self._checkpoint_from_fields(key, fields)
        if type(stored) is OpaqueAnchorFinalizedV2:
            if stored.payload.kind != "ANCHOR_RECORD":
                raise AnchorIntegrityError("finalized envelope contains the wrong opaque payload kind")
            candidate = self._candidate_from_fields(key, fields)
            if stored.evidence_hash != candidate.evidence_hash:
                raise AnchorIntegrityError("finalized envelope evidence hash disagrees with F003 evidence")
            return candidate
        raise AnchorIntegrityError("pair-scoped repository returned an unsupported anchor state")

    def compare_and_set_checkpoint(
        self,
        key: AnchorRepositoryKey,
        checkpoint: AnchorPollingCheckpoint,
        expected_revision: int,
    ) -> AnchorPollingCheckpoint:
        key = self._require_key(key)
        key.validate_checkpoint(checkpoint)
        fields = checkpoint.to_recovery_fields()
        lossless = self._checkpoint_from_fields(key, fields)
        storage_key = self._storage_key(key)
        envelope = OpaqueAnchorCheckpointV2(
            key=storage_key,
            target_session_date=fields["target_session_date"],
            official_close_utc=fields["official_close_utc"],
            deadline_utc=fields["deadline_utc"],
            revision=checkpoint.revision,
            payload=self._payload(
                kind="ANCHOR_CHECKPOINT",
                version_field="integrity_version",
                contract_version=checkpoint.integrity_version,
                fields=fields,
            ),
        )
        persisted = self._repository.compare_and_set_opaque_checkpoint(
            storage_key,
            envelope,
            expected_revision=expected_revision,
        )
        self._assert_stored_key(key, persisted.key)
        if persisted.payload.kind != "ANCHOR_CHECKPOINT":
            raise AnchorIntegrityError("F004 checkpoint write returned a non-checkpoint")
        returned = self._checkpoint_from_fields(key, persisted.payload.value())
        if returned != lossless:
            raise AnchorIntegrityError("F004 checkpoint write did not preserve every F003 recovery field")
        return returned

    def finalize_if_absent(
        self,
        key: AnchorRepositoryKey,
        candidate: AnchorCandidate,
        expected_revision: int,
    ) -> AnchorCandidate:
        key = self._require_key(key)
        key.validate_candidate(candidate)
        fields = candidate.to_evidence_fields()
        lossless = self._candidate_from_fields(key, fields)
        storage_key = self._storage_key(key)
        envelope = OpaqueAnchorFinalizedV2(
            key=storage_key,
            target_session_date=fields["target_session_date"],
            official_close_utc=fields["official_close_utc"],
            deadline_utc=fields["deadline_utc"],
            revision=expected_revision,
            evidence_hash=candidate.evidence_hash,
            payload=self._payload(
                kind="ANCHOR_RECORD",
                version_field="evidence_version",
                contract_version=candidate.evidence_version,
                fields=fields,
            ),
        )
        persisted = self._repository.finalize_opaque_if_absent(
            storage_key,
            envelope,
            expected_revision=expected_revision,
        )
        self._assert_stored_key(key, persisted.key)
        if persisted.payload.kind != "ANCHOR_RECORD":
            raise AnchorIntegrityError("F004 finalization returned a non-finalized anchor")
        returned = self._candidate_from_fields(key, persisted.payload.value())
        if persisted.evidence_hash != returned.evidence_hash:
            raise AnchorIntegrityError("F004 finalization evidence hash disagrees with F003 evidence")
        if returned != lossless:
            raise AnchorIntegrityError("F004 finalization did not preserve every F003 evidence field")
        return returned

    def append_revision_observation(
        self,
        key: AnchorRepositoryKey,
        revision_observation: RevisionObservation,
    ) -> None:
        key = self._require_key(key)
        key.validate_revision_observation(revision_observation)
        fields = revision_observation.to_fields()
        lossless = self._revision_from_fields(key, fields)
        storage_key = self._storage_key(key)
        envelope = OpaqueAnchorRevisionObservationV2(
            key=storage_key,
            evidence_hash=revision_observation.evidence_hash,
            observed_at_utc=fields["observed_at_utc"],
            payload=self._payload(
                kind="ANCHOR_REVISION_OBSERVATION",
                version_field="schema_version",
                contract_version=revision_observation.schema_version,
                fields=fields,
            ),
        )
        self._repository.append_opaque_revision_observation(storage_key, envelope)
        if lossless != revision_observation:
            raise AnchorIntegrityError("revision serialization changed F003 evidence")

    def revision_observations(self, key: AnchorRepositoryKey) -> tuple[RevisionObservation, ...]:
        storage_key = self._storage_key(key)
        observations = self._repository.opaque_revision_observations(storage_key)
        restored: list[RevisionObservation] = []
        for stored in observations:
            if type(stored) is not OpaqueAnchorRevisionObservationV2:
                raise AnchorIntegrityError("F004 returned an unsupported anchor revision envelope")
            self._assert_stored_key(key, stored.key)
            if stored.payload.kind != "ANCHOR_REVISION_OBSERVATION":
                raise AnchorIntegrityError("revision envelope contains the wrong opaque payload kind")
            revision = self._revision_from_fields(key, stored.payload.value())
            if stored.evidence_hash != revision.evidence_hash:
                raise AnchorIntegrityError("revision envelope evidence hash disagrees with F003 evidence")
            restored.append(revision)
        return tuple(restored)
