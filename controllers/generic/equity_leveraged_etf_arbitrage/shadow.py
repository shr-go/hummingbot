"""Deterministic, side-effect-free Controller shadow plans."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal

from controllers.generic.equity_leveraged_etf_arbitrage.nav import (
    AnchorRuntimeStatus,
    NavStage,
    SessionStageDecision,
    StageIntent,
)


__all__ = [
    "ShadowMetrics",
    "ShadowPairInput",
    "ShadowPairPlan",
    "ShadowPlan",
    "ShadowPlanner",
]


_ZERO = Decimal("0")


def _decimal(value: Decimal, label: str, *, nonnegative: bool = False) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise ValueError(f"{label} must be a finite Decimal")
    if nonnegative and value < _ZERO:
        raise ValueError(f"{label} must be non-negative")
    return value


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


@dataclass(frozen=True, slots=True)
class ShadowPairInput:
    """All facts required for one non-executable shadow recommendation."""

    pair_id: str
    cycle_id: str
    target_gross_notional: Decimal
    etf_leverage: int
    stock_leverage: int
    raw_bp: Decimal
    net_bp: Decimal
    nav_decision: SessionStageDecision

    def __post_init__(self) -> None:
        if not isinstance(self.pair_id, str) or not self.pair_id:
            raise ValueError("shadow pair_id must be non-empty")
        if not isinstance(self.cycle_id, str) or not self.cycle_id:
            raise ValueError("shadow cycle_id must be non-empty")
        _decimal(self.target_gross_notional, "shadow target_gross_notional", nonnegative=True)
        _decimal(self.raw_bp, "shadow raw_bp")
        _decimal(self.net_bp, "shadow net_bp")
        for value, label in ((self.etf_leverage, "ETF leverage"), (self.stock_leverage, "stock leverage")):
            if type(value) is not int or value <= 0:
                raise ValueError(f"shadow {label} must be a positive exact int")
        if not isinstance(self.nav_decision, SessionStageDecision):
            raise TypeError("shadow input requires a SessionStageDecision")
        if (
            self.nav_decision.pair_id != self.pair_id
            or self.nav_decision.cycle_id != self.cycle_id
        ):
            raise ValueError("shadow input must use the matching pair/cycle stage decision")


@dataclass(frozen=True, slots=True)
class ShadowPairPlan:
    pair_id: str
    cycle_id: str
    target_gross_notional: Decimal
    etf_leverage: int
    stock_leverage: int
    raw_bp: Decimal
    net_bp: Decimal
    stage: NavStage
    anchor_status: AnchorRuntimeStatus
    entry_allowed: bool
    intents: tuple[StageIntent, ...]


@dataclass(frozen=True, slots=True)
class ShadowMetrics:
    pair_count: int
    entry_allowed_pair_count: int
    blocked_pair_count: int
    emergency_market_request_count: int
    total_target_gross_notional: Decimal

    def __post_init__(self) -> None:
        for value, label in (
            (self.pair_count, "pair_count"),
            (self.entry_allowed_pair_count, "entry_allowed_pair_count"),
            (self.blocked_pair_count, "blocked_pair_count"),
            (self.emergency_market_request_count, "emergency_market_request_count"),
        ):
            if type(value) is not int or value < 0:
                raise ValueError(f"shadow {label} must be a non-negative exact int")
        if self.entry_allowed_pair_count + self.blocked_pair_count != self.pair_count:
            raise ValueError("shadow entry and blocked counts must partition pairs")
        _decimal(self.total_target_gross_notional, "shadow total target", nonnegative=True)


@dataclass(frozen=True, slots=True)
class ShadowPlan:
    """A report-only plan; exchange actions are structurally impossible here."""

    pairs: tuple[ShadowPairPlan, ...]
    deterministic_hash: str
    total_target_gross_notional: Decimal
    metrics: ShadowMetrics
    exchange_actions: tuple[object, ...] = ()
    has_exchange_side_effects: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.pairs, tuple) or not self.pairs:
            raise ValueError("shadow plan requires at least one immutable pair plan")
        pair_ids = tuple(pair.pair_id for pair in self.pairs)
        if pair_ids != tuple(sorted(pair_ids)) or len(set(pair_ids)) != len(pair_ids):
            raise ValueError("shadow plan pairs must be unique and sorted")
        if not isinstance(self.deterministic_hash, str) or len(self.deterministic_hash) != 64:
            raise ValueError("shadow plan hash must be a SHA-256 hex string")
        _decimal(self.total_target_gross_notional, "shadow total target", nonnegative=True)
        if not isinstance(self.metrics, ShadowMetrics):
            raise TypeError("shadow plan metrics must be ShadowMetrics")
        if (
            self.metrics.pair_count != len(self.pairs)
            or self.metrics.total_target_gross_notional != self.total_target_gross_notional
        ):
            raise ValueError("shadow plan metrics do not match pair plans")
        if self.exchange_actions != () or self.has_exchange_side_effects is not False:
            raise ValueError("shadow plans cannot contain exchange actions or side effects")

    @property
    def pair_ids(self) -> tuple[str, ...]:
        return tuple(pair.pair_id for pair in self.pairs)


class ShadowPlanner:
    """Pure projection of a supplied Controller allocation/stage snapshot."""

    def plan(self, inputs: tuple[ShadowPairInput, ...]) -> ShadowPlan:
        if not isinstance(inputs, tuple) or not inputs:
            raise ValueError("shadow planning requires a non-empty tuple of pair inputs")
        if any(not isinstance(value, ShadowPairInput) for value in inputs):
            raise TypeError("shadow planning requires ShadowPairInput values")
        ordered = tuple(sorted(inputs, key=lambda value: value.pair_id))
        if len({value.pair_id for value in ordered}) != len(ordered):
            raise ValueError("shadow planning cannot duplicate a pair")
        pairs = tuple(
            ShadowPairPlan(
                pair_id=value.pair_id,
                cycle_id=value.cycle_id,
                target_gross_notional=value.target_gross_notional,
                etf_leverage=value.etf_leverage,
                stock_leverage=value.stock_leverage,
                raw_bp=value.raw_bp,
                net_bp=value.net_bp,
                stage=value.nav_decision.stage,
                anchor_status=value.nav_decision.operational_status,
                entry_allowed=value.nav_decision.entry_allowed,
                intents=value.nav_decision.intents,
            )
            for value in ordered
        )
        total = sum((pair.target_gross_notional for pair in pairs), _ZERO)
        metrics = ShadowMetrics(
            pair_count=len(pairs),
            entry_allowed_pair_count=sum(pair.entry_allowed for pair in pairs),
            blocked_pair_count=sum(not pair.entry_allowed for pair in pairs),
            emergency_market_request_count=sum(
                any(intent.kind.value == "REQUEST_EMERGENCY_MARKET_FLATTEN" for intent in pair.intents)
                for pair in pairs
            ),
            total_target_gross_notional=total,
        )
        payload = {
            "pairs": [
                {
                    "pair_id": pair.pair_id,
                    "cycle_id": pair.cycle_id,
                    "target_gross_notional": _decimal_text(pair.target_gross_notional),
                    "etf_leverage": pair.etf_leverage,
                    "stock_leverage": pair.stock_leverage,
                    "raw_bp": _decimal_text(pair.raw_bp),
                    "net_bp": _decimal_text(pair.net_bp),
                    "stage": pair.stage.value,
                    "anchor_status": pair.anchor_status.value,
                    "entry_allowed": pair.entry_allowed,
                    "intents": [
                        {
                            "kind": intent.kind.value,
                            "reason": intent.reason,
                        }
                        for intent in pair.intents
                    ],
                }
                for pair in pairs
            ],
            "total_target_gross_notional": _decimal_text(total),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return ShadowPlan(
            pairs=pairs,
            deterministic_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            total_target_gross_notional=total,
            metrics=metrics,
        )
