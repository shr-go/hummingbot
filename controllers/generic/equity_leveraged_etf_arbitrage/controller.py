"""N-pair preflight, allocation, reservation, and action planning.

This Controller intentionally has no order-placement calls.  It freezes
connector/market facts through an injected epoch source, delegates all shared
margin math to the pure F006 allocator, validates any leverage mutation with a
fresh snapshot, then emits only native Strategy V2 Executor actions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

from pydantic import ConfigDict, Field, field_validator, model_validator

from controllers.generic.equity_leveraged_etf_arbitrage.reservations import (
    ReservationConflict,
    ReservationStore,
)
from controllers.generic.equity_leveraged_etf_arbitrage.nav import SessionStageDecision
from controllers.generic.equity_leveraged_etf_arbitrage.shadow import (
    ShadowPairInput,
    ShadowPlan,
    ShadowPlanner,
)
from controllers.generic.equity_leveraged_etf_arbitrage.status import ControllerOperationalStatus
from hummingbot.core.data_type.common import MarketDict
from hummingbot.strategy_v2.controllers.controller_base import (
    ControllerBase,
    ControllerConfigBase,
)
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import (
    LeveragedEtfPairDirection,
    LeveragedEtfPairExecutorConfig,
    LeveragedEtfPairOperation,
    LeverageReservationV1,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.allocator import (
    AccountRiskSnapshot,
    AllocationResult,
    AllocationStatus,
    FrozenAllocationSnapshot,
    FrozenExecutionCosts,
    FrozenPair,
    PairCandidate,
    PortfolioAllocator,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import (
    PairConfig,
    RiskConfig,
    StrategyConfig,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.domain import ArbitrageDirection
from hummingbot.strategy_v2.leveraged_etf_arbitrage.math import calculate_hedge_ratio
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    ExecutorAction,
    StopExecutorAction,
    StoreExecutorAction,
)
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


_ZERO = Decimal("0")
_RECOVERY_REQUIRED = "RECOVERY_REQUIRED"

__all__ = [
    "AnchorFacts",
    "ControllerEpoch",
    "EquityLeveragedEtfArbitrageController",
    "EquityLeveragedEtfArbitrageControllerConfig",
    "PairEpochFacts",
    "ReservationConflict",
]


def _finite_decimal(value: Decimal, name: str, *, positive: bool = False) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{name} must be a finite Decimal")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")


def _utc_from_timestamp(timestamp: float) -> str:
    return (
        datetime.fromtimestamp(timestamp, tz=timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True, slots=True)
class AnchorFacts:
    """Pair-bound anchor inputs already acquired by the F003/T003 layer."""

    nav_cycle_id: str
    s0: Decimal
    l0: Decimal
    h: Decimal
    raw_bp: Decimal
    net_bp: Decimal
    valid: bool

    def __post_init__(self) -> None:
        if not isinstance(self.nav_cycle_id, str) or not self.nav_cycle_id:
            raise ValueError("nav_cycle_id must be non-empty")
        for name, value in (("s0", self.s0), ("l0", self.l0), ("h", self.h)):
            _finite_decimal(value, name, positive=True)
        for name, value in (("raw_bp", self.raw_bp), ("net_bp", self.net_bp)):
            _finite_decimal(value, name)
        if type(self.valid) is not bool:
            raise ValueError("anchor validity must be a bool")


@dataclass(frozen=True, slots=True)
class PairEpochFacts:
    """One pair's immutable market/anchor facts for an allocation epoch."""

    pair_id: str
    frozen_pair: FrozenPair | None
    anchor: AnchorFacts | None
    market_data_fresh: bool
    bracket_data_fresh: bool
    current_etf_leverage: int | None = None
    current_stock_leverage: int | None = None
    entry_ready: bool = True
    failure_reason: str | None = None
    nav_decision: SessionStageDecision | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("pair_id must be non-empty")
        for name, value in (
            ("market_data_fresh", self.market_data_fresh),
            ("bracket_data_fresh", self.bracket_data_fresh),
            ("entry_ready", self.entry_ready),
        ):
            if type(value) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name, value in (
            ("current_etf_leverage", self.current_etf_leverage),
            ("current_stock_leverage", self.current_stock_leverage),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise ValueError(f"{name} must be a positive int when supplied")
        if self.failure_reason is not None and not isinstance(self.failure_reason, str):
            raise ValueError("failure_reason must be a string when supplied")
        if self.nav_decision is not None:
            if not isinstance(self.nav_decision, SessionStageDecision):
                raise ValueError("nav_decision must be a SessionStageDecision when supplied")
            if self.nav_decision.pair_id != self.pair_id:
                raise ValueError("nav_decision pair_id must match PairEpochFacts")
            if self.anchor is not None and self.nav_decision.cycle_id != self.anchor.nav_cycle_id:
                raise ValueError("nav_decision cycle_id must match the pair anchor cycle")


@dataclass(frozen=True, slots=True)
class ControllerEpoch:
    """The one immutable input accepted by a Controller allocation call."""

    account: AccountRiskSnapshot
    account_data_fresh: bool
    pairs: tuple[PairEpochFacts, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.account, AccountRiskSnapshot):
            raise ValueError("ControllerEpoch.account must be an AccountRiskSnapshot")
        if type(self.account_data_fresh) is not bool:
            raise ValueError("account_data_fresh must be a bool")
        if not 1 <= len(self.pairs) <= 4:
            raise ValueError("ControllerEpoch must contain one to four pair facts")
        if any(not isinstance(pair, PairEpochFacts) for pair in self.pairs):
            raise ValueError("ControllerEpoch pairs must be PairEpochFacts")
        pair_ids = tuple(pair.pair_id for pair in self.pairs)
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("ControllerEpoch cannot contain duplicate pair ids")


class EpochSource(Protocol):
    """Build F006-owned frozen facts without direct order side effects."""

    async def build_epoch(
        self,
        preflight: object,
        configured_pairs: Sequence[object],
        active_reservations: Sequence[object],
    ) -> ControllerEpoch:
        """Return a single immutable account/market/anchor epoch."""


class EquityLeveragedEtfArbitrageControllerConfig(ControllerConfigBase):
    """Native Controller configuration composed from the frozen F003 models."""

    controller_name: str = "equity_leveraged_etf_arbitrage.controller"
    connector_name: str = "binance_perpetual"
    strategy: StrategyConfig
    risk: RiskConfig
    pairs: tuple[PairConfig, ...] = Field(min_length=1, max_length=4)

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    @field_validator("pairs", mode="before")
    @classmethod
    def _freeze_pairs(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def _validate_enabled_pairs(self):
        enabled = tuple(pair for pair in self.pairs if pair.enabled)
        if not enabled:
            raise ValueError("Controller requires at least one enabled pair")
        identifiers = tuple(pair.id for pair in enabled)
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Controller cannot register duplicate enabled pair ids")
        return self

    def update_markets(self, markets: MarketDict) -> MarketDict:
        for pair in self.pairs:
            if pair.enabled:
                markets = markets.add_or_update(
                    self.connector_name, pair.etf_trading_pair
                )
                markets = markets.add_or_update(
                    self.connector_name, pair.stock_trading_pair
                )
        return markets


@dataclass(frozen=True, slots=True)
class _EpochPlan:
    preflight: object
    epoch: ControllerEpoch
    snapshot: FrozenAllocationSnapshot
    allocation: AllocationResult
    facts_by_pair_id: dict[str, PairEpochFacts]


@dataclass(frozen=True, slots=True)
class _PlannedCreate:
    pair_id: str
    config: LeveragedEtfPairExecutorConfig
    exposure_increasing: bool
    current_etf_leverage: int | None
    current_stock_leverage: int | None


class EquityLeveragedEtfArbitrageController(ControllerBase):
    """Fail-closed N-pair Controller that emits only Executor actions."""

    def __init__(
        self,
        config: EquityLeveragedEtfArbitrageControllerConfig | object,
        market_data_provider,
        actions_queue,
        *,
        connector: object,
        epoch_source: EpochSource,
        reservations: ReservationStore,
        allocator: PortfolioAllocator | None = None,
        update_interval: float = 1.0,
    ) -> None:
        super().__init__(
            config=config,
            market_data_provider=market_data_provider,
            actions_queue=actions_queue,
            update_interval=update_interval,
        )
        if connector is None:
            raise TypeError("connector is required")
        if not hasattr(connector, "strict_account_preflight") or not hasattr(
            connector, "set_leverage_with_result"
        ):
            raise TypeError(
                "connector must expose strict_account_preflight and set_leverage_with_result"
            )
        if not hasattr(epoch_source, "build_epoch"):
            raise TypeError("epoch_source must expose build_epoch")
        if not all(
            hasattr(reservations, attribute)
            for attribute in ("active", "reserve", "release")
        ):
            raise TypeError("reservations must expose active, reserve, and release")
        self.config = config
        self._connector = connector
        self._epoch_source = epoch_source
        self._reservations = reservations
        self._allocator = allocator or PortfolioAllocator()
        self._configured_pairs = self._normalize_pairs(getattr(config, "pairs", ()))
        self._pending_actions: list[ExecutorAction] = []
        self._inflight_pairs: dict[str, str] = {}
        self._stop_requested_executor_ids: set[str] = set()
        self._stored_executor_ids: set[str] = set()
        self.last_failure: str | None = None
        self.last_allocation_hash: str | None = None
        self.last_operational_status: ControllerOperationalStatus | None = None
        self.last_shadow_plan: ShadowPlan | None = None

    @property
    def registered_pair_ids(self) -> tuple[str, ...]:
        return tuple(pair.id for pair in self._configured_pairs)

    def replace_configured_pairs(self, pairs: Sequence[object]) -> None:
        """Apply an already-validated runtime pair set without reconstructing Executors."""

        self._configured_pairs = self._normalize_pairs(pairs)

    async def update_processed_data(self) -> None:
        self._pending_actions = []
        self.last_failure = None
        self._pending_actions.extend(self._terminal_and_removed_actions())

        try:
            active_reservations = tuple(self._reservations.active())
        except Exception as error:
            self._fail(f"could not read active reservations: {error}")
            self._set_processed_data()
            return

        plan = await self._freeze_plan(active_reservations)
        if plan is None:
            self._set_processed_data()
            return
        self.last_allocation_hash = plan.allocation.deterministic_hash

        planned_creates, stop_actions = self._actions_for_plan(
            plan, active_reservations
        )
        self._append_unique_stops(stop_actions)

        exposure_creates = tuple(
            create for create in planned_creates if create.exposure_increasing
        )
        refreshed = await self._validate_leverage_and_refresh(
            plan, active_reservations, exposure_creates
        )
        if refreshed is None:
            self._set_processed_data()
            return
        if refreshed is not plan:
            plan = refreshed
            self.last_allocation_hash = plan.allocation.deterministic_hash
            planned_creates, stop_actions = self._actions_for_plan(
                plan, active_reservations
            )
            self._append_unique_stops(stop_actions)

        for create in planned_creates:
            if (
                create.pair_id in self._inflight_pairs
                or self._pair_has_durable_reservation(
                    create.pair_id,
                    active_reservations,
                )
            ):
                continue
            if create.exposure_increasing:
                try:
                    self._reservations.reserve(create.config)
                except Exception as error:
                    self._fail(f"could not reserve {create.pair_id}: {error}")
                    continue
            self._inflight_pairs[create.pair_id] = create.config.id
            self._pending_actions.append(
                CreateExecutorAction(
                    controller_id=self.config.id, executor_config=create.config
                )
            )
        self._set_processed_data()

    def determine_executor_actions(self) -> list[ExecutorAction]:
        actions = self._pending_actions
        self._pending_actions = []
        return actions

    def to_format_status(self) -> list[str]:
        failure = self.last_failure or "none"
        allocation = self.last_allocation_hash or "unavailable"
        lines = [
            "Equity Leveraged ETF Arbitrage Controller:",
            f"  Registered pairs: {', '.join(self.registered_pair_ids)}",
            f"  Allocation hash: {allocation}",
            f"  Last preflight failure: {failure}",
        ]
        if self.last_operational_status is not None:
            lines.extend(self.last_operational_status.to_lines())
        return lines

    def build_shadow_plan(self, inputs: tuple[ShadowPairInput, ...]) -> ShadowPlan:
        """Build a report-only plan without connector mutation or action emission.

        This deliberately bypasses normal update processing: normal processing
        performs strict account reads and can set leverage, while a shadow
        caller needs a pure supplied snapshot only.
        """

        plan = ShadowPlanner().plan(inputs)
        decisions = tuple(sorted((item.nav_decision for item in inputs), key=lambda item: item.pair_id))
        self.last_shadow_plan = plan
        self.last_operational_status = ControllerOperationalStatus(
            decisions=decisions,
            shadow_plan=plan,
        )
        return plan

    @staticmethod
    def _normalize_pairs(pairs: Sequence[object]) -> tuple[object, ...]:
        enabled = tuple(pair for pair in pairs if getattr(pair, "enabled", True))
        if not 1 <= len(enabled) <= 4:
            raise ValueError("Controller must register one to four enabled pairs")
        identifiers = tuple(getattr(pair, "id", None) for pair in enabled)
        if any(
            not isinstance(identifier, str) or not identifier
            for identifier in identifiers
        ):
            raise ValueError("every configured pair needs a non-empty id")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("configured pair ids must be unique")
        for pair in enabled:
            for field_name in ("etf_trading_pair", "stock_trading_pair"):
                value = getattr(pair, field_name, None)
                if not isinstance(value, str) or not value:
                    raise ValueError(f"configured pair {pair.id} lacks {field_name}")
        return tuple(sorted(enabled, key=lambda pair: pair.id))

    async def _freeze_plan(
        self, active_reservations: Sequence[object]
    ) -> _EpochPlan | None:
        preflight = await self._strict_preflight(active_reservations)
        if preflight is None:
            return None
        try:
            epoch = await self._epoch_source.build_epoch(
                preflight, self._configured_pairs, active_reservations
            )
        except Exception as error:
            self._fail(f"could not freeze controller epoch: {error}")
            return None
        if not isinstance(epoch, ControllerEpoch):
            self._fail("epoch source returned an invalid ControllerEpoch")
            return None
        try:
            expected_account = self._account_from_preflight(preflight)
        except Exception as error:
            self._fail(f"preflight account snapshot could not be frozen: {error}")
            return None
        if epoch.account != expected_account:
            self._fail("epoch account snapshot does not exactly match strict preflight")
            return None
        if not epoch.account_data_fresh:
            self._fail("account facts are stale")
            return None

        self._record_nav_status(epoch)

        facts_by_pair_id = {facts.pair_id: facts for facts in epoch.pairs}
        usable: list[FrozenPair] = []
        for configured in self._configured_pairs:
            facts = facts_by_pair_id.get(configured.id)
            if facts is None:
                continue
            if not self._pair_facts_ready(configured, facts):
                continue
            assert facts.frozen_pair is not None
            usable.append(facts.frozen_pair)
        if not usable:
            self._fail("no configured pair has fresh anchor, book, and bracket facts")
            return None

        try:
            snapshot = FrozenAllocationSnapshot(
                account=epoch.account,
                pairs=tuple(sorted(usable, key=lambda pair: pair.pair_id)),
                max_total_notional_ratio=self.config.strategy.max_total_notional_ratio,
                execution_costs=FrozenExecutionCosts.from_strategy_config(
                    self.config.strategy
                ),
                stale=False,
            )
            first = self._allocator.allocate(snapshot)
            second = self._allocator.allocate(snapshot)
        except Exception as error:
            self._fail(f"could not allocate immutable epoch: {error}")
            return None
        if first != second or first.deterministic_hash != second.deterministic_hash:
            self._fail("allocator was not exact-equal for a repeated immutable epoch")
            return None
        if first.status is AllocationStatus.FAIL_CLOSED:
            self._fail(first.failure_reason or "allocator failed closed")
            return None
        return _EpochPlan(
            preflight=preflight,
            epoch=epoch,
            snapshot=snapshot,
            allocation=first,
            facts_by_pair_id=facts_by_pair_id,
        )

    def _account_from_preflight(self, preflight: object) -> AccountRiskSnapshot:
        source = getattr(preflight, "account", None)
        if isinstance(source, AccountRiskSnapshot):
            return source
        return AccountRiskSnapshot.from_binance(
            source,
            normal_initial_margin_budget_ratio=self.config.risk.normal_initial_margin_budget_ratio,
            p99_initial_margin_budget_ratio=self.config.risk.p99_initial_margin_budget_ratio,
            stress_extra_bp=self.config.risk.stress_extra_bp_above_p99,
            stress_equity_to_maintenance_margin_multiple=self.config.risk.stress_equity_to_maintenance_margin_multiple,
            reconciliation_tolerance=self.config.risk.account_margin_reconciliation_tolerance,
        )

    async def _strict_preflight(
        self, active_reservations: Sequence[object]
    ) -> object | None:
        active_pairs = tuple(sorted(self._configured_trading_pairs()))
        related_pairs = tuple(
            sorted(
                set(active_pairs).union(
                    self._executor_and_reservation_pairs(active_reservations)
                )
            )
        )
        known_pairs = tuple(sorted(self._known_position_pairs(active_reservations)))
        try:
            preflight = await self._connector.strict_account_preflight(
                trading_pairs=active_pairs,
                related_trading_pairs=related_pairs,
                known_position_trading_pairs=known_pairs,
                max_age_seconds=Decimal(self.config.strategy.account_data_max_age_ms)
                / Decimal("1000"),
                consistency_tolerance=getattr(
                    self.config.risk,
                    "account_margin_reconciliation_tolerance",
                    Decimal("0"),
                ),
            )
        except Exception as error:
            self._fail(f"strict account/symbol preflight failed: {error}")
            return None
        data_time = getattr(preflight, "data_time", None)
        if (
            not isinstance(data_time, (int, float))
            or isinstance(data_time, bool)
            or not math.isfinite(data_time)
        ):
            self._fail("preflight did not provide a usable data_time")
            return None
        max_age = self.config.strategy.account_data_max_age_ms / 1000
        if abs(self.market_data_provider.time() - data_time) > max_age:
            self._fail("preflight account snapshot is stale")
            return None
        return preflight

    def _record_nav_status(self, epoch: ControllerEpoch) -> None:
        decisions = tuple(
            sorted(
                (
                    facts.nav_decision
                    for facts in epoch.pairs
                    if facts.nav_decision is not None
                ),
                key=lambda decision: decision.pair_id,
            )
        )
        if not decisions:
            return
        shadow = self.last_shadow_plan
        if shadow is not None and shadow.pair_ids != tuple(decision.pair_id for decision in decisions):
            shadow = None
        self.last_operational_status = ControllerOperationalStatus(
            decisions=decisions,
            shadow_plan=shadow,
        )

    def _pair_facts_ready(self, configured: object, facts: PairEpochFacts) -> bool:
        frozen = facts.frozen_pair
        anchor = facts.anchor
        if (
            frozen is None
            or anchor is None
            or not anchor.valid
            or not facts.market_data_fresh
            or not facts.bracket_data_fresh
            or frozen.stale
            or (
                facts.nav_decision is not None
                and not facts.nav_decision.entry_allowed
            )
        ):
            return False
        try:
            if frozen.pair_id != configured.id:
                return False
            if (
                frozen.etf.symbol != configured.etf_trading_pair
                or frozen.stock.symbol != configured.stock_trading_pair
            ):
                return False
            if frozen.etf_daily_multiplier != configured.etf_daily_multiplier:
                return False
            if frozen.stock_anchor != anchor.s0 or frozen.etf_anchor != anchor.l0:
                return False
            if frozen.current_net_bp != anchor.net_bp:
                return False
            return anchor.h == calculate_hedge_ratio(
                anchor.s0, anchor.l0, frozen.etf_daily_multiplier
            )
        except Exception:
            return False

    def _actions_for_plan(
        self,
        plan: _EpochPlan,
        active_reservations: Sequence[object],
    ) -> tuple[list[_PlannedCreate], list[StopExecutorAction]]:
        candidates = {
            candidate.pair_id: candidate
            for candidate in plan.allocation.candidate.pairs
        }
        active_by_pair = self._active_executors_by_pair()
        creates: list[_PlannedCreate] = []
        stops: list[StopExecutorAction] = []
        for configured in self._configured_pairs:
            pair_id = configured.id
            facts = plan.facts_by_pair_id.get(pair_id)
            candidate = candidates.get(pair_id)
            active = active_by_pair.get(pair_id, ())
            if active:
                if (
                    facts is not None
                    and facts.frozen_pair is not None
                    and self._must_stop(
                        active,
                        candidate,
                        facts.frozen_pair.direction,
                    )
                ):
                    stops.extend(self._stop_action(executor.id) for executor in active)
                continue
            if facts is None or facts.frozen_pair is None or candidate is None:
                continue
            planned = self._build_executor_config(configured, facts, candidate)
            if planned is None:
                continue
            if planned.exposure_increasing:
                if not facts.entry_ready or candidate.canonical_etf_slice_quantity <= 0:
                    continue
                if self._pair_has_durable_reservation(pair_id, active_reservations):
                    continue
            creates.append(planned)
        return creates, stops

    def _build_executor_config(
        self,
        configured: object,
        facts: PairEpochFacts,
        candidate: PairCandidate,
    ) -> _PlannedCreate | None:
        frozen = facts.frozen_pair
        anchor = facts.anchor
        assert frozen is not None and anchor is not None
        target_etf = candidate.target_etf_quantity
        target_stock = candidate.target_stock_quantity
        current_direction = self._direction_for_quantities(
            frozen.etf.current_quantity, frozen.stock.current_quantity
        )
        if current_direction is not None and current_direction is not frozen.direction:
            # The allocator deliberately preserves the old target while an
            # opposite opportunity is observed.  The Controller converts that
            # preserved target into the required close-before-reopen action.
            target_etf = _ZERO
            target_stock = _ZERO
        delta_etf = target_etf - frozen.etf.current_quantity
        delta_stock = target_stock - frozen.stock.current_quantity
        if delta_etf == 0 and delta_stock == 0:
            return None
        if delta_etf == 0 or delta_stock == 0:
            self._fail(f"{configured.id} has an unpaired target delta")
            return None
        current_gross = frozen.etf.notional_for(
            frozen.etf.current_quantity
        ) + frozen.stock.notional_for(frozen.stock.current_quantity)
        target_gross = frozen.etf.notional_for(target_etf) + frozen.stock.notional_for(
            target_stock
        )
        direction = (
            self._direction_for_quantities(target_etf, target_stock) or frozen.direction
        )
        if current_gross == 0:
            operation = LeveragedEtfPairOperation.OPEN
        elif target_gross == 0 or (
            current_direction is not None and direction is not current_direction
        ):
            operation = LeveragedEtfPairOperation.CLOSE
            direction = current_direction or frozen.direction
        elif target_gross > current_gross:
            operation = LeveragedEtfPairOperation.ADD
        else:
            operation = LeveragedEtfPairOperation.REDUCE

        increasing = operation in {
            LeveragedEtfPairOperation.OPEN,
            LeveragedEtfPairOperation.ADD,
        }
        etf_leverage = candidate.etf_leverage or facts.current_etf_leverage
        stock_leverage = candidate.stock_leverage or facts.current_stock_leverage
        if etf_leverage is None:
            etf_leverage = frozen.etf.leverage_schedule.supported_leverages[-1]
        if stock_leverage is None:
            stock_leverage = frozen.stock.leverage_schedule.supported_leverages[-1]
        try:
            etf_cap = frozen.etf.leverage_schedule.max_notional_for_leverage(
                etf_leverage
            )
            stock_cap = frozen.stock.leverage_schedule.max_notional_for_leverage(
                stock_leverage
            )
        except Exception as error:
            self._fail(f"{configured.id} has invalid selected leverage: {error}")
            return None
        reservation = LeverageReservationV1(
            etf_quantity=abs(delta_etf) if increasing else _ZERO,
            stock_quantity=abs(delta_stock) if increasing else _ZERO,
            etf_leverage=etf_leverage,
            stock_leverage=stock_leverage,
            etf_notional_cap=etf_cap,
            stock_notional_cap=stock_cap,
        )
        timestamp = self.market_data_provider.time()
        payload = {
            "id": self._next_executor_id(configured.id),
            "timestamp": timestamp,
            "controller_id": self.config.id,
            "pair_id": configured.id,
            "nav_cycle_id": anchor.nav_cycle_id,
            "operation": operation,
            "direction": LeveragedEtfPairDirection(direction.value),
            "etf_connector_name": self.config.connector_name,
            "etf_trading_pair": configured.etf_trading_pair,
            "stock_connector_name": self.config.connector_name,
            "stock_trading_pair": configured.stock_trading_pair,
            "s0": anchor.s0,
            "l0": anchor.l0,
            "h": anchor.h,
            "created_raw_bp": anchor.raw_bp,
            "created_net_bp": anchor.net_bp,
            # F004 names the desired *final* gross position.  The two target
            # quantities below are the finite adjustment that this Executor
            # must make, so a CLOSE carries non-zero quantities but a zero
            # final gross target (as required by the wire contract).
            "target_gross_notional": target_gross,
            "etf_target_quantity": abs(delta_etf),
            "stock_target_quantity": abs(delta_stock),
            "leverage_reservation": reservation,
            "created_at_utc": _utc_from_timestamp(timestamp),
        }
        payload["config_hash"] = self._config_hash(payload)
        try:
            config = LeveragedEtfPairExecutorConfig(**payload)
        except Exception as error:
            self._fail(
                f"could not build F004 Executor config for {configured.id}: {error}"
            )
            return None
        return _PlannedCreate(
            pair_id=configured.id,
            config=config,
            exposure_increasing=increasing,
            current_etf_leverage=facts.current_etf_leverage,
            current_stock_leverage=facts.current_stock_leverage,
        )

    async def _validate_leverage_and_refresh(
        self,
        plan: _EpochPlan,
        active_reservations: Sequence[object],
        creates: Sequence[_PlannedCreate],
    ) -> _EpochPlan | None:
        targets: dict[str, tuple[int, Decimal, int | None]] = {}
        for create in creates:
            config = create.config
            for trading_pair, leverage, cap, current in (
                (
                    config.etf_trading_pair,
                    config.leverage_reservation.etf_leverage,
                    config.leverage_reservation.etf_notional_cap,
                    create.current_etf_leverage,
                ),
                (
                    config.stock_trading_pair,
                    config.leverage_reservation.stock_leverage,
                    config.leverage_reservation.stock_notional_cap,
                    create.current_stock_leverage,
                ),
            ):
                prior = targets.get(trading_pair)
                target = (leverage, cap, current)
                if prior is not None and prior[:2] != target[:2]:
                    self._fail(
                        f"inconsistent selected leverage for shared symbol {trading_pair}"
                    )
                    return None
                targets[trading_pair] = target

        changed = False
        for trading_pair in sorted(targets):
            leverage, expected_cap, current = targets[trading_pair]
            if current == leverage:
                continue
            try:
                result = await self._connector.set_leverage_with_result(
                    trading_pair, leverage
                )
            except Exception as error:
                self._fail(f"could not set leverage for {trading_pair}: {error}")
                return None
            actual_leverage = getattr(result, "leverage", None)
            actual_cap = getattr(result, "max_notional_value", None)
            if actual_leverage != leverage or actual_cap != expected_cap:
                self._fail(f"leverage response mismatch for {trading_pair}")
                return None
            changed = True
        if not changed:
            return plan

        refreshed = await self._freeze_plan(active_reservations)
        if refreshed is None:
            return None
        if (
            refreshed.snapshot != plan.snapshot
            or refreshed.allocation != plan.allocation
        ):
            self._fail(
                "fresh account/bracket snapshot invalidated the stable allocation plan"
            )
            return None
        return refreshed

    def _terminal_and_removed_actions(self) -> list[ExecutorAction]:
        actions: list[ExecutorAction] = []
        registered = set(self.registered_pair_ids)
        for executor in sorted(self._owned_executors(), key=lambda value: value.id):
            pair_id = executor.config.pair_id
            if self._executor_is_non_done(executor):
                if pair_id not in registered:
                    action = self._stop_action(executor.id)
                    if action is not None:
                        actions.append(action)
                continue
            self._inflight_pairs.pop(pair_id, None)
            self._stop_requested_executor_ids.discard(executor.id)
            if executor.id in self._stored_executor_ids:
                continue
            try:
                self._reservations.release(
                    executor.id, _utc_from_timestamp(self.market_data_provider.time())
                )
            except Exception as error:
                self._fail(
                    f"could not release terminal reservation for {executor.id}: {error}"
                )
                continue
            self._stored_executor_ids.add(executor.id)
            actions.append(
                StoreExecutorAction(
                    controller_id=self.config.id, executor_id=executor.id
                )
            )
        return actions

    def _active_executors_by_pair(self) -> dict[str, tuple[ExecutorInfo, ...]]:
        values: dict[str, list[ExecutorInfo]] = defaultdict(list)
        for executor in self._owned_executors():
            if self._executor_is_non_done(executor):
                values[executor.config.pair_id].append(executor)
        return {
            pair_id: tuple(sorted(executors, key=lambda value: value.id))
            for pair_id, executors in values.items()
        }

    def _owned_executors(self) -> tuple[ExecutorInfo, ...]:
        owned: list[ExecutorInfo] = []
        for executor in self.executors_info:
            config = getattr(executor, "config", None)
            if not isinstance(config, LeveragedEtfPairExecutorConfig):
                continue
            if config.controller_id == self.config.id:
                owned.append(executor)
        return tuple(owned)

    @staticmethod
    def _executor_is_non_done(executor: ExecutorInfo) -> bool:
        state = (
            executor.custom_info.get("state")
            if isinstance(executor.custom_info, dict)
            else None
        )
        return (
            executor.status is not RunnableStatus.TERMINATED
            or state == _RECOVERY_REQUIRED
        )

    def _must_stop(
        self,
        executors: Sequence[ExecutorInfo],
        candidate: PairCandidate | None,
        requested_direction: ArbitrageDirection,
    ) -> bool:
        if candidate is None or candidate.target_gross_notional == 0:
            return True
        desired = self._direction_for_quantities(
            candidate.target_etf_quantity, candidate.target_stock_quantity
        )
        for executor in executors:
            if (
                desired is None
                or executor.config.direction.value != desired.value
                or executor.config.direction.value != requested_direction.value
            ):
                return True
            if candidate.target_gross_notional < executor.config.target_gross_notional:
                return True
        return False

    def _append_unique_stops(self, actions: Iterable[StopExecutorAction]) -> None:
        for action in actions:
            if (
                action is not None
                and action.executor_id not in self._stop_requested_executor_ids
            ):
                self._stop_requested_executor_ids.add(action.executor_id)
                self._pending_actions.append(action)

    def _stop_action(self, executor_id: str) -> StopExecutorAction | None:
        if executor_id in self._stop_requested_executor_ids:
            return None
        return StopExecutorAction(
            controller_id=self.config.id, executor_id=executor_id, keep_position=True
        )

    def _pair_has_durable_reservation(
        self, pair_id: str, reservations: Sequence[object]
    ) -> bool:
        configured = next(
            (pair for pair in self._configured_pairs if pair.id == pair_id), None
        )
        if configured is None:
            return False
        symbols = {
            (self.config.connector_name, configured.etf_trading_pair),
            (self.config.connector_name, configured.stock_trading_pair),
        }
        return any(
            (
                getattr(reservation, "connector_name", None),
                getattr(reservation, "trading_pair", None),
            )
            in symbols
            for reservation in reservations
        )

    def _configured_trading_pairs(self) -> tuple[str, ...]:
        return tuple(
            pair_name
            for pair in self._configured_pairs
            for pair_name in (pair.etf_trading_pair, pair.stock_trading_pair)
        )

    def _executor_and_reservation_pairs(
        self, reservations: Sequence[object]
    ) -> set[str]:
        related = set(self._configured_trading_pairs())
        for executor in self._owned_executors():
            # A completed OPEN/ADD can leave a legitimate paired position
            # while its finite adjustment Executor is being stored.  Keep all
            # observed self-owned symbols related for strict preflight so that
            # a subsequent CLOSE is not mistaken for an external position.
            related.update(
                (executor.config.etf_trading_pair, executor.config.stock_trading_pair)
            )
        related.update(
            reservation.trading_pair
            for reservation in reservations
            if isinstance(getattr(reservation, "trading_pair", None), str)
        )
        return related

    def _known_position_pairs(self, reservations: Sequence[object]) -> set[str]:
        known: set[str] = set()
        for executor in self._owned_executors():
            known.update(
                (executor.config.etf_trading_pair, executor.config.stock_trading_pair)
            )
        known.update(
            reservation.trading_pair
            for reservation in reservations
            if isinstance(getattr(reservation, "trading_pair", None), str)
        )
        return known

    @staticmethod
    def _direction_for_quantities(
        etf_quantity: Decimal, stock_quantity: Decimal
    ) -> ArbitrageDirection | None:
        if (
            etf_quantity == 0
            or stock_quantity == 0
            or etf_quantity.is_signed() == stock_quantity.is_signed()
        ):
            return None
        return (
            ArbitrageDirection.SHORT_ETF_LONG_STOCK
            if etf_quantity < 0
            else ArbitrageDirection.LONG_ETF_SHORT_STOCK
        )

    def _next_executor_id(self, pair_id: str) -> str:
        return f"{self.config.id}:{pair_id}:{uuid4().hex}"

    @staticmethod
    def _config_hash(payload: dict[str, object]) -> str:
        def default(value: object):
            if isinstance(value, Decimal):
                return format(value, "f")
            if hasattr(value, "value"):
                return value.value
            if hasattr(value, "model_dump"):
                return value.model_dump(mode="json")
            raise TypeError(f"cannot serialize {type(value)!r} into a config hash")

        return hashlib.sha256(
            json.dumps(
                payload,
                default=default,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    def _set_processed_data(self) -> None:
        self.processed_data = {
            "registered_pair_ids": self.registered_pair_ids,
            "last_failure": self.last_failure,
            "allocation_hash": self.last_allocation_hash,
            "operational_status": (
                None
                if self.last_operational_status is None
                else self.last_operational_status.to_dict()
            ),
        }

    def _fail(self, reason: str) -> None:
        self.last_failure = reason
