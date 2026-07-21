"""Pair-local NAV acquisition and close-stage coordination.

This module plans safety intents.  It intentionally does not submit an
exchange request: F005/F007 own execution and recovery of those intents.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from hummingbot.data_feed.yahoo_finance.acquisition import (
    AnchorAcquisitionStatus,
    AnchorCandidate,
    AnchorPollingCheckpoint,
    AnchorRepositoryKey,
    AnchorRepositoryV2,
    CheckpointIntegrityError,
    FinalizedAnchorStatus,
    RevisionObservation,
    YahooAnchorAcquisition,
)
from hummingbot.model.leveraged_etf_repository import AnchorRevisionConflict
from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import NavConfig, SessionName

if TYPE_CHECKING:
    from controllers.generic.equity_leveraged_etf_arbitrage.controller import AnchorFacts


__all__ = [
    "AnchorAlert",
    "AnchorCycleCoordinator",
    "AnchorCycleSnapshot",
    "AnchorRuntimeStatus",
    "NavStage",
    "SessionStageCoordinator",
    "SessionStageDecision",
    "StageIntent",
    "StageIntentKind",
]


_ZERO = Decimal("0")


class AnchorRuntimeStatus(str, Enum):
    """Controller-visible lifecycle of one pair/cycle anchor."""

    PENDING = "PENDING"
    ACQUIRING = "ACQUIRING"
    AVAILABLE = "AVAILABLE"
    ANCHOR_UNAVAILABLE = "ANCHOR_UNAVAILABLE"
    STALE = "STALE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class AnchorAlert(str, Enum):
    ANCHOR_UNAVAILABLE = "ANCHOR_UNAVAILABLE"
    DATA_REVISION = "DATA_REVISION"
    STALE_ANCHOR = "STALE_ANCHOR"
    ANCHOR_INTEGRITY_FAILURE = "ANCHOR_INTEGRITY_FAILURE"


@dataclass(frozen=True, slots=True)
class AnchorCycleSnapshot:
    key: AnchorRepositoryKey
    status: AnchorRuntimeStatus
    remaining_deadline_seconds: Decimal
    checkpoint: AnchorPollingCheckpoint | None = None
    candidate: AnchorCandidate | None = None
    alerts: tuple[AnchorAlert, ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if type(self.key) is not AnchorRepositoryKey:
            raise TypeError("anchor cycle state requires an exact AnchorRepositoryKey v2")
        if not isinstance(self.status, AnchorRuntimeStatus):
            raise TypeError("anchor cycle state has an invalid status")
        if not isinstance(self.remaining_deadline_seconds, Decimal) or not self.remaining_deadline_seconds.is_finite():
            raise ValueError("anchor remaining deadline must be a finite Decimal")
        if self.remaining_deadline_seconds < _ZERO:
            raise ValueError("anchor remaining deadline cannot be negative")
        if self.checkpoint is not None:
            self.key.validate_checkpoint(self.checkpoint)
        if self.candidate is not None:
            self.key.validate_candidate(self.candidate)
        if self.checkpoint is not None and self.candidate is not None:
            raise ValueError("anchor state cannot contain both a checkpoint and finalized evidence")
        if any(not isinstance(alert, AnchorAlert) for alert in self.alerts):
            raise TypeError("anchor alerts must be AnchorAlert values")
        if self.failure_reason is not None and not isinstance(self.failure_reason, str):
            raise TypeError("anchor failure reason must be a string")

    def to_controller_anchor_facts(
        self,
        *,
        raw_bp: Decimal,
        net_bp: Decimal,
    ) -> "AnchorFacts | None":
        """Build the T002 epoch anchor only from validated finalized evidence.

        The local import avoids making the Controller own or duplicate any F003
        evidence type while giving an epoch source one canonical conversion.
        """

        if self.status is not AnchorRuntimeStatus.AVAILABLE or self.candidate is None:
            return None
        from controllers.generic.equity_leveraged_etf_arbitrage.controller import AnchorFacts

        return AnchorFacts(
            nav_cycle_id=self.key.cycle_id,
            s0=self.candidate.stock_close,
            l0=self.candidate.etf_close,
            h=self.candidate.hedge_ratio,
            raw_bp=raw_bp,
            net_bp=net_bp,
            valid=True,
        )


class AnchorCycleCoordinator:
    """Persists F003's checkpoint-driven acquisition without extending its API.

    A restart reloads the original checkpoint and gives F003 the original
    absolute deadline.  It never creates another window for the same key.
    """

    def __init__(
        self,
        *,
        acquisition: YahooAnchorAcquisition,
        repository: AnchorRepositoryV2,
        utc_clock: Callable[[], datetime],
    ) -> None:
        if not isinstance(acquisition, YahooAnchorAcquisition):
            raise TypeError("acquisition must be an F003 YahooAnchorAcquisition")
        if not isinstance(repository, AnchorRepositoryV2):
            raise TypeError("repository must implement F003 AnchorRepositoryV2")
        if not callable(utc_clock):
            raise TypeError("utc_clock must be callable")
        self._acquisition = acquisition
        self._repository = repository
        self._utc_clock = utc_clock

    async def refresh(
        self,
        *,
        pair_id: str,
        stock_symbol: str,
        etf_symbol: str,
        cycle_id: str,
        target_session_date: date,
        official_close_utc: datetime,
    ) -> AnchorCycleSnapshot:
        """Advance at most one acquisition round for an exact pair/cycle key."""

        key = AnchorRepositoryKey.create(pair_id, cycle_id)
        try:
            now = self._utc_now()
            stored = self._repository.load(key)
            if isinstance(stored, AnchorCandidate):
                key.validate_candidate(stored)
                self._validate_candidate_context(stored, target_session_date, official_close_utc)
                return self._snapshot(key, AnchorRuntimeStatus.AVAILABLE, now, candidate=stored)
            if stored is not None:
                key.validate_checkpoint(stored)
                self._validate_checkpoint_context(stored, target_session_date, official_close_utc)
                return await self._advance(key, stored, stock_symbol, etf_symbol, now)

            checkpoint = self._acquisition.create_checkpoint(
                cycle_id,
                target_session_date,
                official_close_utc,
                pair_id=pair_id,
                stock_symbol=stock_symbol,
                etf_symbol=etf_symbol,
            )
            if now < official_close_utc:
                return self._snapshot(key, AnchorRuntimeStatus.PENDING, now, checkpoint=checkpoint)
            if now >= checkpoint.deadline_utc:
                return self._snapshot(
                    key,
                    AnchorRuntimeStatus.ANCHOR_UNAVAILABLE,
                    now,
                    alerts=(AnchorAlert.ANCHOR_UNAVAILABLE,),
                    failure_reason="original anchor deadline elapsed before checkpoint creation",
                )
            try:
                checkpoint = self._repository.compare_and_set_checkpoint(
                    key,
                    checkpoint,
                    expected_revision=0,
                )
            except AnchorRevisionConflict:
                return self._after_conflict(key, now, target_session_date, official_close_utc)
            return await self._advance(key, checkpoint, stock_symbol, etf_symbol, now)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError):
            raise
        except Exception as exception:
            return self._snapshot(
                key,
                AnchorRuntimeStatus.RECOVERY_REQUIRED,
                self._safe_now(),
                alerts=(AnchorAlert.ANCHOR_INTEGRITY_FAILURE,),
                failure_reason=self._safe_reason(exception),
            )

    def assess_revision(
        self,
        key: AnchorRepositoryKey,
        stock_observation: object,
        etf_observation: object,
    ) -> AnchorCycleSnapshot:
        """Append a finalized-anchor revision alert without changing the anchor."""

        key = self._require_key(key)
        try:
            now = self._utc_now()
            stored = self._repository.load(key)
            if not isinstance(stored, AnchorCandidate):
                raise CheckpointIntegrityError("revision assessment requires a finalized pair-scoped anchor")
            key.validate_candidate(stored)
            assessment = self._acquisition.assess_finalized(
                stored,
                stock_observation,
                etf_observation,
            )
            if assessment.status is FinalizedAnchorStatus.RECOVERY_REQUIRED:
                return self._snapshot(
                    key,
                    AnchorRuntimeStatus.RECOVERY_REQUIRED,
                    now,
                    candidate=stored,
                    alerts=(AnchorAlert.ANCHOR_INTEGRITY_FAILURE,),
                    failure_reason="finalized anchor integrity assessment requires recovery",
                )
            if assessment.status is FinalizedAnchorStatus.DATA_REVISION:
                revision = assessment.revision_observation
                if not isinstance(revision, RevisionObservation):
                    raise CheckpointIntegrityError("data revision assessment lacks revision evidence")
                key.validate_revision_observation(revision)
                self._repository.append_revision_observation(key, revision)
                return self._snapshot(
                    key,
                    AnchorRuntimeStatus.AVAILABLE,
                    now,
                    candidate=stored,
                    alerts=(AnchorAlert.DATA_REVISION,),
                )
            return self._snapshot(key, AnchorRuntimeStatus.AVAILABLE, now, candidate=stored)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exception:
            return self._snapshot(
                key,
                AnchorRuntimeStatus.RECOVERY_REQUIRED,
                self._safe_now(),
                alerts=(AnchorAlert.ANCHOR_INTEGRITY_FAILURE,),
                failure_reason=self._safe_reason(exception),
            )

    async def _advance(
        self,
        key: AnchorRepositoryKey,
        checkpoint: AnchorPollingCheckpoint,
        stock_symbol: str,
        etf_symbol: str,
        now: datetime,
    ) -> AnchorCycleSnapshot:
        if now >= checkpoint.deadline_utc:
            return self._snapshot(
                key,
                AnchorRuntimeStatus.ANCHOR_UNAVAILABLE,
                now,
                checkpoint=checkpoint,
                alerts=(AnchorAlert.ANCHOR_UNAVAILABLE,),
                failure_reason="original anchor deadline elapsed",
            )
        result = await self._acquisition.advance(checkpoint, stock_symbol, etf_symbol)
        updated = result.checkpoint
        key.validate_checkpoint(updated)
        if updated.revision != checkpoint.revision:
            try:
                updated = self._repository.compare_and_set_checkpoint(
                    key,
                    updated,
                    expected_revision=checkpoint.revision,
                )
            except AnchorRevisionConflict:
                return self._after_conflict(
                    key,
                    self._utc_now(),
                    checkpoint.target_session_date,
                    checkpoint.official_close_utc,
                )

        current_now = self._utc_now()
        if result.status is AnchorAcquisitionStatus.FINALIZABLE:
            if result.candidate is None:
                raise CheckpointIntegrityError("F003 finalizable result lacks a candidate")
            key.validate_candidate(result.candidate)
            try:
                candidate = self._repository.finalize_if_absent(
                    key,
                    result.candidate,
                    expected_revision=updated.revision,
                )
            except AnchorRevisionConflict:
                return self._after_conflict(
                    key,
                    current_now,
                    checkpoint.target_session_date,
                    checkpoint.official_close_utc,
                )
            return self._snapshot(key, AnchorRuntimeStatus.AVAILABLE, current_now, candidate=candidate)
        if result.status is AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE:
            return self._snapshot(
                key,
                AnchorRuntimeStatus.ANCHOR_UNAVAILABLE,
                current_now,
                checkpoint=updated,
                alerts=(AnchorAlert.ANCHOR_UNAVAILABLE,),
                failure_reason=result.failure_reason,
            )
        status = (
            AnchorRuntimeStatus.PENDING
            if result.status is AnchorAcquisitionStatus.WAITING
            else AnchorRuntimeStatus.ACQUIRING
        )
        return self._snapshot(key, status, current_now, checkpoint=updated)

    def _after_conflict(
        self,
        key: AnchorRepositoryKey,
        now: datetime,
        target_session_date: date,
        official_close_utc: datetime,
    ) -> AnchorCycleSnapshot:
        stored = self._repository.load(key)
        if isinstance(stored, AnchorCandidate):
            key.validate_candidate(stored)
            self._validate_candidate_context(stored, target_session_date, official_close_utc)
            return self._snapshot(key, AnchorRuntimeStatus.AVAILABLE, now, candidate=stored)
        if isinstance(stored, AnchorPollingCheckpoint):
            key.validate_checkpoint(stored)
            self._validate_checkpoint_context(stored, target_session_date, official_close_utc)
            status = (
                AnchorRuntimeStatus.ANCHOR_UNAVAILABLE
                if now >= stored.deadline_utc
                else AnchorRuntimeStatus.ACQUIRING
            )
            alerts = (AnchorAlert.ANCHOR_UNAVAILABLE,) if status is AnchorRuntimeStatus.ANCHOR_UNAVAILABLE else ()
            return self._snapshot(key, status, now, checkpoint=stored, alerts=alerts)
        return self._snapshot(
            key,
            AnchorRuntimeStatus.RECOVERY_REQUIRED,
            now,
            alerts=(AnchorAlert.ANCHOR_INTEGRITY_FAILURE,),
            failure_reason="checkpoint CAS conflicted but no pair-scoped state could be reloaded",
        )

    @staticmethod
    def _require_key(key: AnchorRepositoryKey) -> AnchorRepositoryKey:
        if type(key) is not AnchorRepositoryKey:
            raise TypeError("revision assessment requires an exact AnchorRepositoryKey v2")
        key.to_fields()
        return key

    @staticmethod
    def _validate_checkpoint_context(
        checkpoint: AnchorPollingCheckpoint,
        target_session_date: date,
        official_close_utc: datetime,
    ) -> None:
        if (
            checkpoint.target_session_date != target_session_date
            or checkpoint.official_close_utc != official_close_utc
        ):
            raise CheckpointIntegrityError("persisted checkpoint does not match the requested NAV session")

    @staticmethod
    def _validate_candidate_context(
        candidate: AnchorCandidate,
        target_session_date: date,
        official_close_utc: datetime,
    ) -> None:
        if (
            candidate.target_session_date != target_session_date
            or candidate.official_close_utc != official_close_utc
        ):
            raise CheckpointIntegrityError("persisted anchor does not match the requested NAV session")

    def _snapshot(
        self,
        key: AnchorRepositoryKey,
        status: AnchorRuntimeStatus,
        now: datetime,
        *,
        checkpoint: AnchorPollingCheckpoint | None = None,
        candidate: AnchorCandidate | None = None,
        alerts: tuple[AnchorAlert, ...] = (),
        failure_reason: str | None = None,
    ) -> AnchorCycleSnapshot:
        deadline = (
            candidate.deadline_utc
            if candidate is not None
            else checkpoint.deadline_utc
            if checkpoint is not None
            else now
        )
        return AnchorCycleSnapshot(
            key=key,
            status=status,
            remaining_deadline_seconds=self._remaining(deadline, now),
            checkpoint=checkpoint,
            candidate=candidate,
            alerts=alerts,
            failure_reason=failure_reason,
        )

    @staticmethod
    def _remaining(deadline: datetime, now: datetime) -> Decimal:
        seconds = Decimal(str((deadline - now).total_seconds()))
        return max(_ZERO, seconds)

    def _utc_now(self) -> datetime:
        value = self._utc_clock()
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("UTC clock must return an aware UTC datetime")
        return value.astimezone(timezone.utc)

    def _safe_now(self) -> datetime:
        try:
            return self._utc_now()
        except Exception:
            return datetime.now(timezone.utc)

    @staticmethod
    def _safe_reason(exception: Exception) -> str:
        # Contract exceptions contain field names and never configuration
        # objects.  Keep only the exception type when an arbitrary adapter
        # supplied the error text.
        if isinstance(exception, (CheckpointIntegrityError, AnchorRevisionConflict, ValueError, TypeError)):
            return str(exception)
        return type(exception).__name__


class NavStage(str, Enum):
    NORMAL = "NORMAL"
    CLOSE_30 = "CLOSE_30"
    CLOSE_1 = "CLOSE_1"
    POST_CLOSE = "POST_CLOSE"


class StageIntentKind(str, Enum):
    BLOCK_NEW_EXPOSURE = "BLOCK_NEW_EXPOSURE"
    CANCEL_ENTRY_MAKERS = "CANCEL_ENTRY_MAKERS"
    REQUEST_MAKER_EXIT = "REQUEST_MAKER_EXIT"
    CANCEL_EXIT_MAKERS = "CANCEL_EXIT_MAKERS"
    REQUEST_EMERGENCY_MARKET_FLATTEN = "REQUEST_EMERGENCY_MARKET_FLATTEN"
    PAUSE_NEW_EXPOSURE = "PAUSE_NEW_EXPOSURE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


@dataclass(frozen=True, slots=True)
class StageIntent:
    pair_id: str
    cycle_id: str
    kind: StageIntentKind
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("stage intent pair_id must be non-empty")
        if not isinstance(self.cycle_id, str) or not self.cycle_id:
            raise ValueError("stage intent cycle_id must be non-empty")
        if not isinstance(self.kind, StageIntentKind):
            raise TypeError("stage intent kind is invalid")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("stage intent reason must be non-empty")


@dataclass(frozen=True, slots=True)
class SessionStageDecision:
    pair_id: str
    cycle_id: str
    stage: NavStage
    operational_status: AnchorRuntimeStatus
    entry_allowed: bool
    pair_paused: bool
    emergency_market_requested: bool
    intents: tuple[StageIntent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("stage decision pair_id must be non-empty")
        if not isinstance(self.cycle_id, str) or not self.cycle_id:
            raise ValueError("stage decision cycle_id must be non-empty")
        if not isinstance(self.stage, NavStage):
            raise TypeError("stage decision stage is invalid")
        if not isinstance(self.operational_status, AnchorRuntimeStatus):
            raise TypeError("stage decision operational status is invalid")
        for value, name in (
            (self.entry_allowed, "entry_allowed"),
            (self.pair_paused, "pair_paused"),
            (self.emergency_market_requested, "emergency_market_requested"),
        ):
            if type(value) is not bool:
                raise TypeError(f"stage decision {name} must be a bool")
        if not isinstance(self.intents, tuple) or any(not isinstance(item, StageIntent) for item in self.intents):
            raise TypeError("stage decision intents must be an immutable tuple")
        if any(item.pair_id != self.pair_id or item.cycle_id != self.cycle_id for item in self.intents):
            raise ValueError("stage intent identity must match its pair decision")

    def for_pair(self, pair_id: str) -> "SessionStageDecision":
        """Reuse a session decision's timing for another independently scoped pair."""

        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("pair_id must be non-empty")
        return replace(
            self,
            pair_id=pair_id,
            intents=tuple(replace(intent, pair_id=pair_id) for intent in self.intents),
        )


class SessionStageCoordinator:
    """Convert F003 NAV timing and pair state into non-executable Controller intents."""

    def __init__(self, nav: NavConfig) -> None:
        if not isinstance(nav, NavConfig):
            raise TypeError("nav must be an F003 NavConfig")
        self._nav = nav

    def evaluate(
        self,
        *,
        pair_id: str,
        cycle_id: str,
        official_close_utc: datetime,
        now_utc: datetime,
        anchor_status: AnchorRuntimeStatus,
        spread_bp: Decimal,
        p99_bp: Decimal,
        active_holding_cycle_id: str | None = None,
        anchor_cycle_id: str | None = None,
    ) -> SessionStageDecision:
        if not isinstance(pair_id, str) or not pair_id:
            raise ValueError("pair_id must be non-empty")
        if not isinstance(cycle_id, str) or not cycle_id:
            raise ValueError("cycle_id must be non-empty")
        now_utc = self._utc(now_utc, "now")
        official_close_utc = self._utc(official_close_utc, "official close")
        if not isinstance(anchor_status, AnchorRuntimeStatus):
            raise TypeError("anchor_status must be an AnchorRuntimeStatus")
        self._finite_decimal(spread_bp, "spread_bp")
        self._finite_decimal(p99_bp, "p99_bp", nonnegative=True)
        if active_holding_cycle_id is not None and not isinstance(active_holding_cycle_id, str):
            raise TypeError("active_holding_cycle_id must be a string when supplied")
        if anchor_cycle_id is not None and not isinstance(anchor_cycle_id, str):
            raise TypeError("anchor_cycle_id must be a string when supplied")

        close_1_at = official_close_utc - timedelta(seconds=self._nav.force_market_close_lead_seconds)
        close_30_at = official_close_utc - timedelta(minutes=self._nav.maker_close_lead_minutes)
        if now_utc >= official_close_utc:
            stage = NavStage.POST_CLOSE
        elif now_utc >= close_1_at:
            stage = NavStage.CLOSE_1
        elif now_utc >= close_30_at:
            stage = NavStage.CLOSE_30
        else:
            stage = NavStage.NORMAL

        operational_status = anchor_status
        cross_cycle = active_holding_cycle_id is not None and active_holding_cycle_id != cycle_id
        stale = anchor_cycle_id is not None and anchor_cycle_id != cycle_id
        if cross_cycle:
            operational_status = AnchorRuntimeStatus.RECOVERY_REQUIRED
        elif stale:
            operational_status = AnchorRuntimeStatus.STALE

        anomalous = spread_bp > p99_bp + Decimal("100")
        entry_allowed = (
            stage is NavStage.NORMAL
            and operational_status is AnchorRuntimeStatus.AVAILABLE
            and not anomalous
            and not cross_cycle
        )
        intents: list[StageIntent] = []

        def add(kind: StageIntentKind, reason: str) -> None:
            if any(intent.kind is kind for intent in intents):
                return
            intents.append(StageIntent(pair_id, cycle_id, kind, reason))

        if not entry_allowed:
            add(StageIntentKind.BLOCK_NEW_EXPOSURE, "NAV/session/anchor state blocks new exposure")
        if stage is NavStage.CLOSE_30:
            add(StageIntentKind.CANCEL_ENTRY_MAKERS, "official close is within 30 minutes")
            add(StageIntentKind.REQUEST_MAKER_EXIT, "official close is within 30 minutes")
        elif stage is NavStage.CLOSE_1:
            add(StageIntentKind.CANCEL_ENTRY_MAKERS, "official close is within 60 seconds")
            add(StageIntentKind.CANCEL_EXIT_MAKERS, "official close is within 60 seconds")
            add(
                StageIntentKind.REQUEST_EMERGENCY_MARKET_FLATTEN,
                "official close is within 60 seconds",
            )
        if anomalous:
            add(StageIntentKind.PAUSE_NEW_EXPOSURE, "spread exceeds P99 plus 100 bp")
        if cross_cycle:
            add(StageIntentKind.RECOVERY_REQUIRED, "holding belongs to a prior NAV cycle")
        elif operational_status is AnchorRuntimeStatus.RECOVERY_REQUIRED:
            add(StageIntentKind.RECOVERY_REQUIRED, "anchor integrity requires recovery")
        elif operational_status is AnchorRuntimeStatus.STALE:
            add(StageIntentKind.PAUSE_NEW_EXPOSURE, "anchor cycle is stale")

        return SessionStageDecision(
            pair_id=pair_id,
            cycle_id=cycle_id,
            stage=stage,
            operational_status=operational_status,
            entry_allowed=entry_allowed,
            pair_paused=anomalous or stale,
            emergency_market_requested=stage is NavStage.CLOSE_1,
            intents=tuple(intents),
        )

    @staticmethod
    def select_session(
        requested: SessionName,
        sessions: Mapping[SessionName, object],
    ) -> SessionName:
        """Use F003's regular-session fallback without inventing another session."""

        if not isinstance(requested, SessionName):
            raise TypeError("requested session must be an F003 SessionName")
        if SessionName.REGULAR not in sessions:
            raise ValueError("session mapping must include F003 regular session")
        return requested if requested in sessions else SessionName.REGULAR

    @staticmethod
    def _utc(value: datetime, label: str) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError(f"{label} must be an aware UTC datetime")
        return value.astimezone(timezone.utc)

    @staticmethod
    def _finite_decimal(value: Decimal, label: str, *, nonnegative: bool = False) -> None:
        if not isinstance(value, Decimal) or not value.is_finite():
            raise ValueError(f"{label} must be a finite Decimal")
        if nonnegative and value < _ZERO:
            raise ValueError(f"{label} must be non-negative")
