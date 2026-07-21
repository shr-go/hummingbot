"""Exact, side-effect-free shared-account allocation for leveraged ETF pairs.

This module intentionally contains no connector, Controller, Executor, clock,
or persistence calls.  A caller first freezes all market/account/bracket input
into :class:`FrozenAllocationSnapshot`, then receives an immutable plan that is
safe to hash, compare, and reuse without recomputing a maker slice.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, fields, is_dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_DOWN, localcontext
from enum import Enum
from typing import Any

from hummingbot.strategy_v2.leveraged_etf_arbitrage.domain import (
    ArbitrageDirection,
    BookSide,
    DepthLevel,
    InsufficientDepthError,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.math import (
    calculate_hedge_ratio,
    calculate_opportunity,
    depth_vwap,
    stock_book_walk_bp,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.risk import (
    LeverageSchedule,
    LeverageSelection,
    RiskInputError,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.state import (
    ProjectedLegState,
    ProjectedPairState,
    ProjectedStateName,
)


LATTICE_DENOMINATOR = 1_000_000
_BASIS_POINTS = Decimal("10000")
_ZERO = Decimal("0")


class AllocationInputError(ValueError):
    """The frozen snapshot cannot safely produce an exposure-increasing plan."""


class AllocationStatus(str, Enum):
    ALLOCATED = "ALLOCATED"
    RISK_REDUCTION_ONLY = "RISK_REDUCTION_ONLY"
    FAIL_CLOSED = "FAIL_CLOSED"


def _decimal(value: Decimal, name: str, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise AllocationInputError(f"{name} must be a finite Decimal")
    if positive and value <= 0:
        raise AllocationInputError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise AllocationInputError(f"{name} must be non-negative")
    return value


def _sign_for_direction(direction: ArbitrageDirection) -> tuple[Decimal, Decimal]:
    if direction is ArbitrageDirection.SHORT_ETF_LONG_STOCK:
        return Decimal("-1"), Decimal("1")
    if direction is ArbitrageDirection.LONG_ETF_SHORT_STOCK:
        return Decimal("1"), Decimal("-1")
    raise AllocationInputError("direction must be an ArbitrageDirection")


def _floor_to_step(quantity: Decimal, step: Decimal) -> Decimal:
    if quantity <= 0:
        return _ZERO
    return (quantity / step).to_integral_value(rounding=ROUND_DOWN) * step


def _ceil_to_step(quantity: Decimal, step: Decimal) -> Decimal:
    if quantity <= 0:
        return _ZERO
    return (quantity / step).to_integral_value(rounding=ROUND_CEILING) * step


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        _decimal(self.price, "book price", positive=True)
        _decimal(self.quantity, "book quantity", positive=True)


@dataclass(frozen=True, slots=True)
class AllocationTier:
    minimum_net_bp: Decimal
    target_ratio: Decimal

    def __post_init__(self) -> None:
        _decimal(self.minimum_net_bp, "tier minimum net bp", nonnegative=True)
        _decimal(self.target_ratio, "tier target ratio", nonnegative=True)


@dataclass(frozen=True, slots=True)
class FrozenLeg:
    """One leg's complete allocation-relevant frozen input."""

    symbol: str
    mark_price: Decimal
    contract_multiplier: Decimal
    quantity_step: Decimal
    min_quantity: Decimal
    min_notional: Decimal
    leverage_schedule: LeverageSchedule
    current_quantity: Decimal = _ZERO
    owned_open_quantity: Decimal = _ZERO
    owned_reservation_quantity: Decimal = _ZERO
    exchange_position_initial_margin: Decimal = _ZERO
    exchange_open_order_initial_margin: Decimal = _ZERO
    exchange_maint_margin: Decimal = _ZERO
    new_open_order_initial_margin_rate: Decimal = _ZERO
    safety_notional_buffer: Decimal = _ZERO

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise AllocationInputError("leg symbol must be non-empty")
        _decimal(self.mark_price, f"{self.symbol} mark price", positive=True)
        _decimal(self.contract_multiplier, f"{self.symbol} contract multiplier", positive=True)
        _decimal(self.quantity_step, f"{self.symbol} quantity step", positive=True)
        _decimal(self.min_quantity, f"{self.symbol} minimum quantity", positive=True)
        _decimal(self.min_notional, f"{self.symbol} minimum notional", positive=True)
        for name, value in (
            ("current quantity", self.current_quantity),
            ("owned open quantity", self.owned_open_quantity),
            ("owned reservation quantity", self.owned_reservation_quantity),
        ):
            _decimal(value, f"{self.symbol} {name}")
        for name, value in (
            ("exchange position initial margin", self.exchange_position_initial_margin),
            ("exchange open-order initial margin", self.exchange_open_order_initial_margin),
            ("exchange maintenance margin", self.exchange_maint_margin),
            ("new open-order initial margin rate", self.new_open_order_initial_margin_rate),
            ("safety notional buffer", self.safety_notional_buffer),
        ):
            _decimal(value, f"{self.symbol} {name}", nonnegative=True)
        if not isinstance(self.leverage_schedule, LeverageSchedule):
            raise AllocationInputError(f"{self.symbol} requires a leverage schedule")
        if self.leverage_schedule.symbol != self.symbol:
            raise AllocationInputError("leg symbol and leverage schedule symbol must match")

    def notional_for(self, quantity: Decimal) -> Decimal:
        return abs(quantity) * self.contract_multiplier * self.mark_price


@dataclass(frozen=True, slots=True)
class FrozenPair:
    """A pair snapshot after all external values have been frozen."""

    pair_id: str
    direction: ArbitrageDirection
    stock_anchor: Decimal
    etf_anchor: Decimal
    etf_daily_multiplier: Decimal
    requested_target_ratio: Decimal
    tiers: tuple[AllocationTier, ...]
    p99_bp: Decimal
    current_net_bp: Decimal
    max_stock_taker_impact_bp: Decimal
    etf_best_bid: Decimal
    etf_best_ask: Decimal
    stock_bids: tuple[BookLevel, ...]
    stock_asks: tuple[BookLevel, ...]
    etf: FrozenLeg
    stock: FrozenLeg
    stale: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise AllocationInputError("pair id must be non-empty")
        if not isinstance(self.direction, ArbitrageDirection):
            raise AllocationInputError(f"{self.pair_id} direction must be ArbitrageDirection")
        for name, value in (
            ("stock anchor", self.stock_anchor),
            ("ETF anchor", self.etf_anchor),
            ("ETF daily multiplier", self.etf_daily_multiplier),
            ("ETF best bid", self.etf_best_bid),
            ("ETF best ask", self.etf_best_ask),
        ):
            _decimal(value, f"{self.pair_id} {name}", positive=True)
        for name, value in (
            ("requested target ratio", self.requested_target_ratio),
            ("P99 bp", self.p99_bp),
            ("maximum stock impact bp", self.max_stock_taker_impact_bp),
        ):
            _decimal(value, f"{self.pair_id} {name}", nonnegative=True)
        _decimal(self.current_net_bp, f"{self.pair_id} current net bp")
        if self.etf_best_ask < self.etf_best_bid:
            raise AllocationInputError(f"{self.pair_id} ETF best ask cannot be below best bid")
        if not isinstance(self.etf, FrozenLeg) or not isinstance(self.stock, FrozenLeg):
            raise AllocationInputError(f"{self.pair_id} legs must be frozen")
        if not self.tiers:
            raise AllocationInputError(f"{self.pair_id} requires at least one tier")
        previous: AllocationTier | None = None
        for tier in self.tiers:
            if not isinstance(tier, AllocationTier):
                raise AllocationInputError(f"{self.pair_id} tiers must be AllocationTier values")
            if previous is not None:
                if tier.minimum_net_bp <= previous.minimum_net_bp:
                    raise AllocationInputError(f"{self.pair_id} tier thresholds must be strictly increasing")
                if tier.target_ratio < previous.target_ratio:
                    raise AllocationInputError(f"{self.pair_id} tier targets must be non-decreasing")
            previous = tier
        if self.tiers[0].minimum_net_bp != 0:
            raise AllocationInputError(f"{self.pair_id} tiers must begin at 0 bp")
        if not self.stock_bids or not self.stock_asks:
            raise AllocationInputError(f"{self.pair_id} requires both stock book sides")
        if any(not isinstance(level, BookLevel) for level in self.stock_bids + self.stock_asks):
            raise AllocationInputError(f"{self.pair_id} books must contain BookLevel values")
        if type(self.stale) is not bool:
            raise AllocationInputError(f"{self.pair_id} stale must be a bool")


@dataclass(frozen=True, slots=True)
class AccountRiskSnapshot:
    total_margin_balance: Decimal
    total_initial_margin: Decimal
    total_maint_margin: Decimal
    available_balance: Decimal
    normal_initial_margin_budget_ratio: Decimal
    p99_initial_margin_budget_ratio: Decimal
    stress_extra_bp: Decimal
    stress_equity_to_maintenance_margin_multiple: Decimal
    reconciliation_tolerance: Decimal

    def __post_init__(self) -> None:
        _decimal(self.total_margin_balance, "total margin balance", positive=True)
        for name, value in (
            ("total initial margin", self.total_initial_margin),
            ("total maintenance margin", self.total_maint_margin),
            ("available balance", self.available_balance),
            ("stress extra bp", self.stress_extra_bp),
            ("reconciliation tolerance", self.reconciliation_tolerance),
        ):
            _decimal(value, name, nonnegative=True)
        for name, value in (
            ("normal initial margin budget ratio", self.normal_initial_margin_budget_ratio),
            ("P99 initial margin budget ratio", self.p99_initial_margin_budget_ratio),
            ("stress equity/MM multiple", self.stress_equity_to_maintenance_margin_multiple),
        ):
            _decimal(value, name, positive=True)
        if self.normal_initial_margin_budget_ratio > 1 or self.p99_initial_margin_budget_ratio > 1:
            raise AllocationInputError("initial margin budget ratios cannot exceed one")
        if self.p99_initial_margin_budget_ratio < self.normal_initial_margin_budget_ratio:
            raise AllocationInputError("P99 initial margin budget cannot be below normal budget")
        if self.reconciliation_tolerance > Decimal("0.01"):
            raise AllocationInputError("reconciliation tolerance cannot exceed 0.01 USDT")

    @classmethod
    def from_binance(
        cls,
        source: object,
        *,
        normal_initial_margin_budget_ratio: Decimal,
        p99_initial_margin_budget_ratio: Decimal,
        stress_extra_bp: Decimal,
        stress_equity_to_maintenance_margin_multiple: Decimal,
        reconciliation_tolerance: Decimal,
    ) -> "AccountRiskSnapshot":
        """Adapt the typed F002 Account Information V3 snapshot without I/O."""

        from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_risk_data import (
            BinancePerpetualAccountRiskSnapshot,
        )

        if not isinstance(source, BinancePerpetualAccountRiskSnapshot):
            raise AllocationInputError("source must be BinancePerpetualAccountRiskSnapshot")
        return cls(
            total_margin_balance=source.total_margin_balance,
            total_initial_margin=source.total_initial_margin,
            total_maint_margin=source.total_maint_margin,
            available_balance=source.available_balance,
            normal_initial_margin_budget_ratio=normal_initial_margin_budget_ratio,
            p99_initial_margin_budget_ratio=p99_initial_margin_budget_ratio,
            stress_extra_bp=stress_extra_bp,
            stress_equity_to_maintenance_margin_multiple=stress_equity_to_maintenance_margin_multiple,
            reconciliation_tolerance=reconciliation_tolerance,
        )


@dataclass(frozen=True, slots=True)
class FrozenAllocationSnapshot:
    account: AccountRiskSnapshot
    pairs: tuple[FrozenPair, ...]
    max_total_notional_ratio: Decimal
    stale: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.account, AccountRiskSnapshot):
            raise AllocationInputError("allocation snapshot account must be frozen")
        if not 1 <= len(self.pairs) <= 4:
            raise AllocationInputError("allocation snapshot must contain one to four pairs")
        if any(not isinstance(pair, FrozenPair) for pair in self.pairs):
            raise AllocationInputError("allocation snapshot pairs must be frozen")
        pair_ids = tuple(pair.pair_id for pair in self.pairs)
        if len(set(pair_ids)) != len(pair_ids):
            raise AllocationInputError("allocation snapshot contains duplicate pair ids")
        _decimal(self.max_total_notional_ratio, "maximum total notional ratio", positive=True)
        if type(self.stale) is not bool:
            raise AllocationInputError("allocation snapshot stale must be a bool")


@dataclass(frozen=True, slots=True)
class PairCandidate:
    pair_id: str
    target_etf_quantity: Decimal
    target_stock_quantity: Decimal
    target_gross_notional: Decimal
    canonical_etf_slice_quantity: Decimal
    canonical_stock_slice_quantity: Decimal
    stock_vwap: Decimal | None
    executable_net_bp: Decimal
    etf_leverage: int | None
    stock_leverage: int | None
    states: tuple[ProjectedPairState, ...]
    final_state_initial_margin: Decimal
    max_state_initial_margin: Decimal
    max_state_maint_margin: Decimal
    canonical_slice_reused: bool
    exposure_increasing: bool


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    n: int
    scale: Decimal
    pairs: tuple[PairCandidate, ...]
    im_unmodeled: Decimal
    mm_unmodeled: Decimal
    projected_initial_margin: Decimal
    delta_initial_margin_available: Decimal
    projected_maint_margin: Decimal
    stress_loss: Decimal
    initial_margin_budget: Decimal
    feasible: bool
    constraint_failures: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class AllocationResult:
    status: AllocationStatus
    nominal_scale: Decimal
    risk_scale_n: int
    risk_scale: Decimal
    candidate: CandidateEvaluation
    skipped_leverage_pair_ids: tuple[str, ...]
    cap_descent_count: int
    evaluation_count: int
    n_star_feasible: bool
    n_star_plus_one_feasible: bool | None
    failure_reason: str | None = None

    @property
    def deterministic_hash(self) -> str:
        payload = json.dumps(_canonical_value(self), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    return format(value.normalize(), "f")


def _canonical_value(value: Any) -> Any:
    if isinstance(value, Decimal):
        return _canonical_decimal(value)
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _canonical_value(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, tuple):
        return [_canonical_value(item) for item in value]
    if isinstance(value, list):
        return [_canonical_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    return value


class PortfolioAllocator:
    """Evaluate the fixed 10^-6 risk lattice for one immutable snapshot."""

    def allocate(self, snapshot: FrozenAllocationSnapshot) -> AllocationResult:
        try:
            with localcontext() as context:
                context.prec = 50
                return self._allocate(snapshot)
        except (AllocationInputError, RiskInputError, ArithmeticError, ValueError) as error:
            return self._failed_result(snapshot, str(error))

    def _allocate(self, snapshot: FrozenAllocationSnapshot) -> AllocationResult:
        if not isinstance(snapshot, FrozenAllocationSnapshot):
            raise AllocationInputError("allocator requires a FrozenAllocationSnapshot")
        if snapshot.stale or any(pair.stale for pair in snapshot.pairs):
            raise AllocationInputError("allocation snapshot is stale")
        pairs = tuple(sorted(snapshot.pairs, key=lambda pair: pair.pair_id))
        for pair in pairs:
            self._validate_book(pair.stock_asks, BookSide.BUY, pair.pair_id)
            self._validate_book(pair.stock_bids, BookSide.SELL, pair.pair_id)

        current = {pair.pair_id: self._current_gross(pair) for pair in pairs}
        caps = self._initial_caps(snapshot, pairs, current)
        skipped: set[str] = set()
        descents = 0
        limit = sum(len(pair.tiers) for pair in pairs) + len(pairs)

        for _ in range(limit + 1):
            changed = self._drop_leverage_ineligible(snapshot, pairs, caps, current, skipped)
            if changed:
                descents += changed
                continue
            nominal_scale = self._nominal_scale(snapshot, caps)
            solve = self._solve_lattice(snapshot, pairs, caps, current, nominal_scale)
            if solve[0] is None:
                return self._failed_result(
                    snapshot,
                    solve[1] or "risk lattice validation failed",
                    nominal_scale,
                    skipped,
                    descents,
                )
            candidate, n_star, count, neighbor = solve[0]
            changed = self._descend_signal_caps(snapshot, pairs, caps, current, candidate)
            if changed:
                descents += changed
                if descents > limit:
                    return self._failed_result(
                        snapshot,
                        "finite cap descent exceeded its limit",
                        nominal_scale,
                        skipped,
                        descents,
                    )
                continue
            status = AllocationStatus.ALLOCATED if candidate.feasible else AllocationStatus.RISK_REDUCTION_ONLY
            return AllocationResult(
                status=status,
                nominal_scale=nominal_scale,
                risk_scale_n=n_star,
                risk_scale=Decimal(n_star) / Decimal(LATTICE_DENOMINATOR),
                candidate=candidate,
                skipped_leverage_pair_ids=tuple(sorted(skipped)),
                cap_descent_count=descents,
                evaluation_count=count,
                n_star_feasible=candidate.feasible,
                n_star_plus_one_feasible=neighbor,
            )
        return self._failed_result(snapshot, "finite cap descent did not stabilize", skipped=skipped, descents=descents)

    @staticmethod
    def _validate_book(levels: tuple[BookLevel, ...], side: BookSide, pair_id: str) -> None:
        previous: Decimal | None = None
        for level in levels:
            if previous is not None:
                if side is BookSide.BUY and level.price < previous:
                    raise AllocationInputError(f"{pair_id} stock ask book is nonmonotonic")
                if side is BookSide.SELL and level.price > previous:
                    raise AllocationInputError(f"{pair_id} stock bid book is nonmonotonic")
            previous = level.price

    @staticmethod
    def _current_gross(pair: FrozenPair) -> Decimal:
        return pair.etf.notional_for(pair.etf.current_quantity) + pair.stock.notional_for(pair.stock.current_quantity)

    @staticmethod
    def _entry_etf_price(pair: FrozenPair) -> Decimal:
        return pair.etf_best_ask if pair.direction is ArbitrageDirection.SHORT_ETF_LONG_STOCK else pair.etf_best_bid

    @staticmethod
    def _stock_side(pair: FrozenPair) -> BookSide:
        return BookSide.BUY if pair.direction is ArbitrageDirection.SHORT_ETF_LONG_STOCK else BookSide.SELL

    @staticmethod
    def _stock_levels(pair: FrozenPair) -> tuple[DepthLevel, ...]:
        source = pair.stock_asks if pair.direction is ArbitrageDirection.SHORT_ETF_LONG_STOCK else pair.stock_bids
        return tuple(DepthLevel(level.price, level.quantity) for level in source)

    def _initial_caps(
        self,
        snapshot: FrozenAllocationSnapshot,
        pairs: tuple[FrozenPair, ...],
        current: dict[str, Decimal],
    ) -> dict[str, Decimal]:
        caps: dict[str, Decimal] = {}
        for pair in pairs:
            requested = pair.requested_target_ratio * snapshot.account.total_margin_balance
            if requested <= current[pair.pair_id]:
                caps[pair.pair_id] = requested
                continue
            probe = self._probe_net_bp(pair, requested, current[pair.pair_id])
            caps[pair.pair_id] = self._supported_cap(
                pair,
                probe,
                requested,
                current[pair.pair_id],
                snapshot.account.total_margin_balance,
            )
        return caps

    def _supported_cap(
        self,
        pair: FrozenPair,
        net_bp: Decimal | None,
        upper: Decimal,
        current: Decimal,
        equity: Decimal,
    ) -> Decimal:
        if net_bp is None or net_bp <= 0:
            return current
        candidates = tuple(
            tier.target_ratio
            for tier in pair.tiers
            if tier.target_ratio * equity <= upper and tier.minimum_net_bp <= net_bp
        )
        if not candidates:
            return current
        supported = max(candidates) * equity
        return max(current, min(upper, supported))

    def _probe_net_bp(self, pair: FrozenPair, requested: Decimal, current: Decimal) -> Decimal | None:
        target = self._target_for_gross(pair, requested)
        etf_delta = max(_ZERO, abs(target[0]) - abs(pair.etf.current_quantity))
        stock_delta = max(_ZERO, abs(target[1]) - abs(pair.stock.current_quantity))
        if requested <= current or etf_delta == 0 or stock_delta == 0:
            return None
        probe = self._minimum_slice(pair, etf_delta, stock_delta)
        if probe is None:
            return None
        return probe[2]

    def _drop_leverage_ineligible(
        self,
        snapshot: FrozenAllocationSnapshot,
        pairs: tuple[FrozenPair, ...],
        caps: dict[str, Decimal],
        current: dict[str, Decimal],
        skipped: set[str],
    ) -> int:
        """Remove whole pairs whose unscaled target does not fit either leg."""

        removed = 0
        for pair in pairs:
            pair_id = pair.pair_id
            if caps[pair_id] <= current[pair_id] or pair_id in skipped:
                continue
            target_etf, target_stock, _ = self._target_for_gross(pair, caps[pair_id])
            try:
                self._select_pair_leverage(pair, target_etf, target_stock)
            except RiskInputError:
                caps[pair_id] = current[pair_id]
                skipped.add(pair_id)
                removed += 1
        return removed

    @staticmethod
    def _nominal_scale(snapshot: FrozenAllocationSnapshot, caps: dict[str, Decimal]) -> Decimal:
        total = sum(caps.values(), _ZERO)
        cap = snapshot.max_total_notional_ratio * snapshot.account.total_margin_balance
        if total == 0 or total <= cap:
            return Decimal("1")
        return cap / total

    def _solve_lattice(
        self,
        snapshot: FrozenAllocationSnapshot,
        pairs: tuple[FrozenPair, ...],
        caps: dict[str, Decimal],
        current: dict[str, Decimal],
        nominal_scale: Decimal,
    ) -> tuple[tuple[CandidateEvaluation, int, int, bool | None] | None, str | None]:
        evaluations: dict[int, CandidateEvaluation] = {}
        evaluation_count = 0

        def evaluate(n: int, *, force: bool = False) -> CandidateEvaluation:
            nonlocal evaluation_count
            if not force and n in evaluations:
                return evaluations[n]
            evaluation_count += 1
            if evaluation_count > 24:
                raise AllocationInputError("fixed cap vector exceeded 24 candidate evaluations")
            value = self._evaluate_candidate(snapshot, pairs, caps, current, nominal_scale, n)
            if not force:
                evaluations[n] = value
                self._validate_observed_monotonicity(evaluations)
            return value

        try:
            full = evaluate(LATTICE_DENOMINATOR)
            if full.feasible:
                repeated = evaluate(LATTICE_DENOMINATOR, force=True)
                if repeated != full:
                    return None, "candidate evaluation was not exact-equal on repeated full-scale input"
                return (full, LATTICE_DENOMINATOR, evaluation_count, None), None

            zero = evaluate(0)
            if not zero.feasible:
                # F(0) is explicitly retained as the no-new-exposure proof.
                return (zero, 0, evaluation_count, None), None

            low = 0
            high = LATTICE_DENOMINATOR
            while low < high:
                mid = (low + high + 1) // 2
                if evaluate(mid).feasible:
                    low = mid
                else:
                    high = mid - 1

            best = evaluate(low)
            repeated = evaluate(low, force=True)
            if repeated != best:
                return None, "candidate evaluation was not exact-equal on repeated n_star"
            neighbor = evaluate(low + 1).feasible if low < LATTICE_DENOMINATOR else None
            if neighbor is not False:
                return None, "n_star neighbor validation did not prove a unique maximum"
            return (best, low, evaluation_count, neighbor), None
        except AllocationInputError as error:
            return None, str(error)

    @staticmethod
    def _validate_observed_monotonicity(evaluations: dict[int, CandidateEvaluation]) -> None:
        ordered = tuple(sorted(evaluations.items()))
        for (left_n, left), (right_n, right) in zip(ordered, ordered[1:]):
            if left_n >= right_n:
                raise AllocationInputError("risk lattice keys must be strictly ordered")
            if not left.feasible and right.feasible:
                raise AllocationInputError("risk feasibility is nonmonotonic on the fixed lattice")
            left_pairs = {pair.pair_id: pair for pair in left.pairs}
            for right_pair in right.pairs:
                left_pair = left_pairs[right_pair.pair_id]
                if right_pair.canonical_etf_slice_quantity < left_pair.canonical_etf_slice_quantity:
                    raise AllocationInputError("canonical ETF slice is nonmonotonic on the fixed lattice")
                if right_pair.canonical_stock_slice_quantity < left_pair.canonical_stock_slice_quantity:
                    raise AllocationInputError("canonical stock slice is nonmonotonic on the fixed lattice")

    def _evaluate_candidate(
        self,
        snapshot: FrozenAllocationSnapshot,
        pairs: tuple[FrozenPair, ...],
        caps: dict[str, Decimal],
        current: dict[str, Decimal],
        nominal_scale: Decimal,
        n: int,
    ) -> CandidateEvaluation:
        if not 0 <= n <= LATTICE_DENOMINATOR:
            raise AllocationInputError("risk lattice n must be within its fixed range")
        scale = Decimal(n) / Decimal(LATTICE_DENOMINATOR)
        pair_candidates: list[PairCandidate] = []
        for pair in pairs:
            cap = caps[pair.pair_id] * nominal_scale
            target = (
                current[pair.pair_id] + scale * (cap - current[pair.pair_id])
                if cap > current[pair.pair_id]
                else cap
            )
            pair_candidates.append(self._evaluate_pair(pair, target, current[pair.pair_id]))

        im_unmodeled, mm_unmodeled = self._unmodeled_residuals(snapshot, pairs)
        max_modeled_im = sum((pair.max_state_initial_margin for pair in pair_candidates), _ZERO)
        max_modeled_mm = sum((pair.max_state_maint_margin for pair in pair_candidates), _ZERO)
        account = snapshot.account
        projected_im = max(account.total_initial_margin, im_unmodeled + max_modeled_im)
        projected_mm = max(account.total_maint_margin, mm_unmodeled + max_modeled_mm)
        delta_available = max(_ZERO, im_unmodeled + max_modeled_im - account.total_initial_margin)
        source_pairs_by_id = {pair.pair_id: pair for pair in pairs}
        budget_ratio = (
            account.p99_initial_margin_budget_ratio
            if any(
                source_pairs_by_id[pair.pair_id].p99_bp > 0
                and pair.executable_net_bp >= source_pairs_by_id[pair.pair_id].p99_bp
                for pair in pair_candidates
            )
            else account.normal_initial_margin_budget_ratio
        )
        budget = account.total_margin_balance * budget_ratio
        stress_loss = sum(
            (
                pair.target_gross_notional
                * max(
                    _ZERO,
                    self._pair_by_id(pairs, pair.pair_id).p99_bp
                    + account.stress_extra_bp
                    - pair.executable_net_bp,
                )
                / _BASIS_POINTS
                for pair in pair_candidates
            ),
            _ZERO,
        )
        failures: list[str] = []
        if projected_im > budget:
            failures.append("initial-margin-budget")
        if delta_available > account.available_balance:
            failures.append("available-balance")
        if (
            account.total_margin_balance - stress_loss
            < account.stress_equity_to_maintenance_margin_multiple * projected_mm
        ):
            failures.append("stress-maintenance-margin")
        return CandidateEvaluation(
            n=n,
            scale=scale,
            pairs=tuple(pair_candidates),
            im_unmodeled=im_unmodeled,
            mm_unmodeled=mm_unmodeled,
            projected_initial_margin=projected_im,
            delta_initial_margin_available=delta_available,
            projected_maint_margin=projected_mm,
            stress_loss=stress_loss,
            initial_margin_budget=budget,
            feasible=not failures,
            constraint_failures=tuple(failures),
        )

    @staticmethod
    def _pair_by_id(pairs: tuple[FrozenPair, ...], pair_id: str) -> FrozenPair:
        return next(pair for pair in pairs if pair.pair_id == pair_id)

    def _unmodeled_residuals(
        self,
        snapshot: FrozenAllocationSnapshot,
        pairs: tuple[FrozenPair, ...],
    ) -> tuple[Decimal, Decimal]:
        known_im = sum(
            (
                leg.exchange_position_initial_margin + leg.exchange_open_order_initial_margin
                for pair in pairs
                for leg in (pair.etf, pair.stock)
            ),
            _ZERO,
        )
        known_mm = sum(
            (leg.exchange_maint_margin for pair in pairs for leg in (pair.etf, pair.stock)),
            _ZERO,
        )
        account = snapshot.account
        tolerance = account.reconciliation_tolerance
        if known_im > account.total_initial_margin + tolerance:
            raise AllocationInputError("modeled initial margin exceeds account total beyond tolerance")
        if known_mm > account.total_maint_margin + tolerance:
            raise AllocationInputError("modeled maintenance margin exceeds account total beyond tolerance")
        return max(_ZERO, account.total_initial_margin - known_im), max(_ZERO, account.total_maint_margin - known_mm)

    def _target_for_gross(self, pair: FrozenPair, gross: Decimal) -> tuple[Decimal, Decimal, Decimal]:
        """Split a gross target, then quantize ETF first and stock from h."""

        _decimal(gross, f"{pair.pair_id} gross target", nonnegative=True)
        if gross == 0:
            return _ZERO, _ZERO, _ZERO
        hedge_ratio = calculate_hedge_ratio(
            pair.stock_anchor,
            pair.etf_anchor,
            pair.etf_daily_multiplier,
        )
        notional_ratio = hedge_ratio * pair.stock.mark_price / pair.etf.mark_price
        etf_notional = gross / (Decimal("1") + notional_ratio)
        etf_quantity = _floor_to_step(
            etf_notional / (pair.etf.contract_multiplier * pair.etf.mark_price),
            pair.etf.quantity_step,
        )
        if etf_quantity == 0:
            return _ZERO, _ZERO, _ZERO
        stock_quantity = _floor_to_step(
            etf_quantity
            * pair.etf.contract_multiplier
            * hedge_ratio
            / pair.stock.contract_multiplier,
            pair.stock.quantity_step,
        )
        if stock_quantity == 0:
            return _ZERO, _ZERO, _ZERO
        etf_sign, stock_sign = _sign_for_direction(pair.direction)
        signed_etf = etf_sign * etf_quantity
        signed_stock = stock_sign * stock_quantity
        actual = pair.etf.notional_for(signed_etf) + pair.stock.notional_for(signed_stock)
        return signed_etf, signed_stock, actual

    def _minimum_slice(
        self,
        pair: FrozenPair,
        etf_delta: Decimal,
        stock_delta: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
        """Return the smallest legal probe as ETF qty, stock qty, g, and VWAP."""

        hedge = calculate_hedge_ratio(pair.stock_anchor, pair.etf_anchor, pair.etf_daily_multiplier)
        etf_price = self._entry_etf_price(pair)
        stock_best = pair.stock_asks[0].price if self._stock_side(pair) is BookSide.BUY else pair.stock_bids[0].price
        etf_needed = max(
            pair.etf.min_quantity,
            pair.etf.min_notional / (etf_price * pair.etf.contract_multiplier),
        )
        stock_needed = max(
            pair.stock.min_quantity,
            pair.stock.min_notional / (stock_best * pair.stock.contract_multiplier),
        )
        needed_for_stock = (
            stock_needed
            * pair.stock.contract_multiplier
            / (pair.etf.contract_multiplier * hedge)
        )
        quantity = _ceil_to_step(max(etf_needed, needed_for_stock), pair.etf.quantity_step)
        details = self._slice_details(pair, quantity, etf_delta, stock_delta)
        return details

    def _canonical_slice(
        self,
        pair: FrozenPair,
        etf_delta: Decimal,
        stock_delta: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
        """Find the unique greatest legal ETF quantity step for this candidate."""

        minimum = self._minimum_slice(pair, etf_delta, stock_delta)
        if minimum is None:
            return None
        minimum_etf = minimum[0]
        maximum_etf = _floor_to_step(etf_delta, pair.etf.quantity_step)
        if minimum_etf > maximum_etf:
            return None
        low = int((minimum_etf / pair.etf.quantity_step).to_integral_value(rounding=ROUND_DOWN))
        high = int((maximum_etf / pair.etf.quantity_step).to_integral_value(rounding=ROUND_DOWN))
        best = minimum
        while low < high:
            mid = (low + high + 1) // 2
            quantity = Decimal(mid) * pair.etf.quantity_step
            details = self._slice_details(pair, quantity, etf_delta, stock_delta)
            if details is None:
                high = mid - 1
            else:
                low = mid
                best = details
        if best[0] != Decimal(low) * pair.etf.quantity_step:
            details = self._slice_details(pair, Decimal(low) * pair.etf.quantity_step, etf_delta, stock_delta)
            if details is None:
                raise AllocationInputError("canonical slice binary search lost its legal lower bound")
            best = details
        return best

    def _slice_details(
        self,
        pair: FrozenPair,
        etf_quantity: Decimal,
        etf_delta: Decimal,
        stock_delta: Decimal,
    ) -> tuple[Decimal, Decimal, Decimal, Decimal] | None:
        """Return ``(ETF qty, stock qty, net bp, VWAP)`` when a slice is legal."""

        if etf_quantity <= 0 or etf_quantity > etf_delta:
            return None
        etf_price = self._entry_etf_price(pair)
        if etf_quantity < pair.etf.min_quantity:
            return None
        if pair.etf.notional_for(etf_quantity) < pair.etf.min_notional:
            return None
        hedge = calculate_hedge_ratio(pair.stock_anchor, pair.etf_anchor, pair.etf_daily_multiplier)
        stock_quantity = _floor_to_step(
            etf_quantity * pair.etf.contract_multiplier * hedge / pair.stock.contract_multiplier,
            pair.stock.quantity_step,
        )
        if stock_quantity <= 0 or stock_quantity > stock_delta:
            return None
        if stock_quantity < pair.stock.min_quantity:
            return None
        try:
            stock_vwap = depth_vwap(self._stock_levels(pair), stock_quantity, self._stock_side(pair))
        except InsufficientDepthError:
            return None
        if pair.stock.notional_for(stock_quantity) < pair.stock.min_notional:
            return None
        impact = stock_book_walk_bp(
            stock_vwap,
            pair.stock_asks[0].price if self._stock_side(pair) is BookSide.BUY else pair.stock_bids[0].price,
            self._stock_side(pair),
        )
        if impact > pair.max_stock_taker_impact_bp:
            return None
        try:
            opportunity = calculate_opportunity(
                stock_anchor=pair.stock_anchor,
                etf_anchor=pair.etf_anchor,
                etf_daily_multiplier=pair.etf_daily_multiplier,
                stock_entry_price=stock_vwap,
                etf_entry_price=etf_price,
                etf_quantity=etf_quantity,
                stock_quantity=stock_quantity,
                stock_contract_multiplier=pair.stock.contract_multiplier,
                etf_contract_multiplier=pair.etf.contract_multiplier,
                maker_fee_bp=_ZERO,
                taker_fee_bp=Decimal("4"),
                maker_slippage_bp_per_fill=Decimal("2"),
            )
        except ValueError:
            return etf_quantity, stock_quantity, Decimal("-1"), stock_vwap
        if opportunity.direction is not pair.direction:
            return etf_quantity, stock_quantity, Decimal("-1"), stock_vwap
        return etf_quantity, stock_quantity, opportunity.net_bp.display, stock_vwap

    def _capacity_notional(self, leg: FrozenLeg, target_quantity: Decimal) -> Decimal:
        target = leg.notional_for(target_quantity)
        transient = (
            leg.notional_for(leg.current_quantity)
            + leg.notional_for(leg.owned_open_quantity)
            + leg.notional_for(leg.owned_reservation_quantity)
        )
        return max(target, transient) + leg.safety_notional_buffer

    def _select_pair_leverage(
        self,
        pair: FrozenPair,
        target_etf: Decimal,
        target_stock: Decimal,
    ) -> tuple[LeverageSelection, LeverageSelection]:
        return (
            pair.etf.leverage_schedule.select_leverage(self._capacity_notional(pair.etf, target_etf)),
            pair.stock.leverage_schedule.select_leverage(self._capacity_notional(pair.stock, target_stock)),
        )

    def _evaluate_pair(self, pair: FrozenPair, requested_gross: Decimal, current_gross: Decimal) -> PairCandidate:
        target_etf, target_stock, actual_gross = self._target_for_gross(pair, requested_gross)
        current_etf = pair.etf.current_quantity
        current_stock = pair.stock.current_quantity
        if (
            (current_etf != 0 and target_etf != 0 and current_etf.is_signed() != target_etf.is_signed())
            or (current_stock != 0 and target_stock != 0 and current_stock.is_signed() != target_stock.is_signed())
        ):
            # Direction flips are a close-then-reopen lifecycle operation.  The
            # pure allocator must not reserve the opposite exposure in one epoch.
            target_etf, target_stock, actual_gross = current_etf, current_stock, current_gross
        increasing = actual_gross > current_gross
        etf_delta = max(_ZERO, abs(target_etf) - abs(current_etf)) if increasing else _ZERO
        stock_delta = max(_ZERO, abs(target_stock) - abs(current_stock)) if increasing else _ZERO

        canonical = None
        if increasing and etf_delta > 0 and stock_delta > 0:
            canonical = self._canonical_slice(pair, etf_delta, stock_delta)
        if increasing and canonical is None:
            # A candidate with no legal next maker slice is exactly the current
            # fully hedged target; it cannot reserve a fictitious small order.
            target_etf, target_stock, actual_gross = current_etf, current_stock, current_gross
            increasing = False
            etf_delta = stock_delta = _ZERO

        if canonical is None:
            etf_slice = stock_slice = _ZERO
            net_bp = pair.current_net_bp
            stock_vwap: Decimal | None = None
        else:
            etf_slice, stock_slice, net_bp, stock_vwap = canonical

        try:
            etf_selection, stock_selection = self._select_pair_leverage(pair, target_etf, target_stock)
        except RiskInputError as error:
            raise AllocationInputError(f"{pair.pair_id} candidate leverage selection failed: {error}") from error

        states = self._projected_states(
            pair,
            target_etf,
            target_stock,
            etf_slice,
            stock_slice,
            etf_selection,
            stock_selection,
            increasing,
        )
        final_state = next(state for state in states if state.name is ProjectedStateName.FINAL)
        max_im = max(state.initial_margin for state in states)
        max_mm = max(state.maintenance_margin for state in states)
        return PairCandidate(
            pair_id=pair.pair_id,
            target_etf_quantity=target_etf,
            target_stock_quantity=target_stock,
            target_gross_notional=actual_gross,
            canonical_etf_slice_quantity=etf_slice,
            canonical_stock_slice_quantity=stock_slice,
            stock_vwap=stock_vwap,
            executable_net_bp=net_bp,
            etf_leverage=etf_selection.leverage,
            stock_leverage=stock_selection.leverage,
            states=states,
            final_state_initial_margin=final_state.initial_margin,
            max_state_initial_margin=max_im,
            max_state_maint_margin=max_mm,
            canonical_slice_reused=True,
            exposure_increasing=increasing,
        )

    def _projected_states(
        self,
        pair: FrozenPair,
        target_etf: Decimal,
        target_stock: Decimal,
        etf_slice: Decimal,
        stock_slice: Decimal,
        etf_selection: LeverageSelection,
        stock_selection: LeverageSelection,
        increasing: bool,
    ) -> tuple[ProjectedPairState, ...]:
        current_etf = pair.etf.current_quantity
        current_stock = pair.stock.current_quantity
        etf_change = target_etf - current_etf
        stock_change = target_stock - current_stock
        etf_sign = Decimal("-1") if etf_change < 0 else Decimal("1")
        stock_sign = Decimal("-1") if stock_change < 0 else Decimal("1")
        signed_etf_slice = etf_sign * etf_slice
        signed_stock_slice = stock_sign * stock_slice

        def make_state(
            name: ProjectedStateName,
            etf_position: Decimal,
            etf_candidate_open: Decimal,
            etf_candidate_reservation: Decimal,
            stock_position: Decimal,
            stock_candidate_open: Decimal,
            stock_candidate_reservation: Decimal,
        ) -> ProjectedPairState:
            return ProjectedPairState(
                name=name,
                etf=self._leg_state(
                    pair.etf,
                    etf_position,
                    etf_candidate_open,
                    etf_candidate_reservation,
                    etf_selection,
                ),
                stock=self._leg_state(
                    pair.stock,
                    stock_position,
                    stock_candidate_open,
                    stock_candidate_reservation,
                    stock_selection,
                ),
            )

        if not increasing or etf_slice == 0 or stock_slice == 0:
            # REDUCE/CLOSE paths never receive advance credit for a release:
            # keep the current state and compare it with final only.
            return (
                make_state(
                    ProjectedStateName.CURRENT_RESERVATION,
                    current_etf,
                    _ZERO,
                    _ZERO,
                    current_stock,
                    _ZERO,
                    _ZERO,
                ),
                make_state(
                    ProjectedStateName.FINAL,
                    target_etf,
                    _ZERO,
                    _ZERO,
                    target_stock,
                    _ZERO,
                    _ZERO,
                ),
            )

        states: list[ProjectedPairState] = [
            make_state(
                ProjectedStateName.CURRENT_RESERVATION,
                current_etf,
                _ZERO,
                etf_change,
                current_stock,
                _ZERO,
                stock_change,
            ),
            make_state(
                ProjectedStateName.MAKER_OPEN,
                current_etf,
                signed_etf_slice,
                etf_change - signed_etf_slice,
                current_stock,
                _ZERO,
                stock_change,
            ),
            make_state(
                ProjectedStateName.ETF_ONE_LEG,
                current_etf + signed_etf_slice,
                _ZERO,
                etf_change - signed_etf_slice,
                current_stock,
                _ZERO,
                stock_change,
            ),
        ]
        for partial in self._stock_partial_points(pair, stock_slice):
            signed_partial = stock_sign * partial
            states.append(
                make_state(
                    ProjectedStateName.STOCK_PARTIAL,
                    current_etf + signed_etf_slice,
                    _ZERO,
                    etf_change - signed_etf_slice,
                    current_stock + signed_partial,
                    _ZERO,
                    stock_change - signed_partial,
                )
            )
        states.extend(
            (
                make_state(
                    ProjectedStateName.SLICE_HEDGED,
                    current_etf + signed_etf_slice,
                    _ZERO,
                    etf_change - signed_etf_slice,
                    current_stock + signed_stock_slice,
                    _ZERO,
                    stock_change - signed_stock_slice,
                ),
                make_state(
                    ProjectedStateName.FINAL,
                    target_etf,
                    _ZERO,
                    _ZERO,
                    target_stock,
                    _ZERO,
                    _ZERO,
                ),
            )
        )
        return tuple(states)

    def _stock_partial_points(self, pair: FrozenPair, stock_slice: Decimal) -> tuple[Decimal, ...]:
        """Enumerate finite margin-critical stock partial-fill quantities."""

        candidates: set[Decimal] = {_ZERO, stock_slice}
        step = pair.stock.quantity_step
        # A net-position zero crossing is a margin-critical state.
        current_abs = abs(pair.stock.current_quantity)
        if _ZERO < current_abs < stock_slice:
            candidates.add(_floor_to_step(current_abs, step))
            candidates.add(min(stock_slice, _ceil_to_step(current_abs, step)))
        # Maintenance bracket floors/caps can change the piecewise formula.
        for bracket in pair.stock.leverage_schedule.brackets:
            for boundary in (bracket.notional_floor, bracket.notional_cap):
                quantity = boundary / (pair.stock.mark_price * pair.stock.contract_multiplier)
                if _ZERO < quantity < stock_slice:
                    candidates.add(_floor_to_step(quantity, step))
                    candidates.add(min(stock_slice, _ceil_to_step(quantity, step)))
        return tuple(sorted(point for point in candidates if _ZERO <= point <= stock_slice))

    def _leg_state(
        self,
        leg: FrozenLeg,
        position_quantity: Decimal,
        candidate_open_quantity: Decimal,
        candidate_reservation_quantity: Decimal,
        selection: LeverageSelection,
    ) -> ProjectedLegState:
        existing_open_notional = leg.notional_for(leg.owned_open_quantity)
        candidate_open_notional = leg.notional_for(candidate_open_quantity)
        existing_reservation_notional = leg.notional_for(leg.owned_reservation_quantity)
        candidate_reservation_notional = leg.notional_for(candidate_reservation_quantity)
        position_notional = leg.notional_for(position_quantity)
        position_margin = max(position_notional / selection.leverage, leg.exchange_position_initial_margin)
        existing_open_margin = max(existing_open_notional / selection.leverage, leg.exchange_open_order_initial_margin)
        candidate_open_margin = max(
            candidate_open_notional / selection.leverage,
            candidate_open_notional * leg.new_open_order_initial_margin_rate,
        )
        reservation_margin = (existing_reservation_notional + candidate_reservation_notional) / selection.leverage
        total_notional = (
            position_notional
            + existing_open_notional
            + candidate_open_notional
            + existing_reservation_notional
            + candidate_reservation_notional
        )
        return ProjectedLegState(
            position_quantity=position_quantity,
            open_order_quantity=leg.owned_open_quantity + candidate_open_quantity,
            reservation_quantity=leg.owned_reservation_quantity + candidate_reservation_quantity,
            notional=total_notional,
            initial_margin=position_margin + existing_open_margin + candidate_open_margin + reservation_margin,
            maintenance_margin=leg.leverage_schedule.maintenance_margin(total_notional),
        )

    def _descend_signal_caps(
        self,
        snapshot: FrozenAllocationSnapshot,
        pairs: tuple[FrozenPair, ...],
        caps: dict[str, Decimal],
        current: dict[str, Decimal],
        candidate: CandidateEvaluation,
    ) -> int:
        changed = 0
        result_by_id = {pair.pair_id: pair for pair in candidate.pairs}
        for pair in pairs:
            pair_id = pair.pair_id
            if caps[pair_id] <= current[pair_id]:
                continue
            result = result_by_id[pair_id]
            if result.canonical_etf_slice_quantity == 0:
                caps[pair_id] = current[pair_id]
                changed += 1
                continue
            supported = self._supported_cap(
                pair,
                result.executable_net_bp,
                caps[pair_id],
                current[pair_id],
                snapshot.account.total_margin_balance,
            )
            if supported < caps[pair_id]:
                caps[pair_id] = supported
                changed += 1
        return changed

    def _failed_result(
        self,
        snapshot: FrozenAllocationSnapshot | object,
        reason: str,
        nominal_scale: Decimal = Decimal("1"),
        skipped: set[str] | None = None,
        descents: int = 0,
    ) -> AllocationResult:
        account = snapshot.account if isinstance(snapshot, FrozenAllocationSnapshot) else None
        empty = CandidateEvaluation(
            n=0,
            scale=_ZERO,
            pairs=(),
            im_unmodeled=_ZERO,
            mm_unmodeled=_ZERO,
            projected_initial_margin=account.total_initial_margin if account else _ZERO,
            delta_initial_margin_available=_ZERO,
            projected_maint_margin=account.total_maint_margin if account else _ZERO,
            stress_loss=_ZERO,
            initial_margin_budget=(
                account.total_margin_balance * account.normal_initial_margin_budget_ratio if account else _ZERO
            ),
            feasible=False,
            constraint_failures=("fail-closed",),
        )
        return AllocationResult(
            status=AllocationStatus.FAIL_CLOSED,
            nominal_scale=nominal_scale,
            risk_scale_n=0,
            risk_scale=_ZERO,
            candidate=empty,
            skipped_leverage_pair_ids=tuple(sorted(skipped or ())),
            cap_descent_count=descents,
            evaluation_count=0,
            n_star_feasible=False,
            n_star_plus_one_feasible=None,
            failure_reason=reason,
        )
