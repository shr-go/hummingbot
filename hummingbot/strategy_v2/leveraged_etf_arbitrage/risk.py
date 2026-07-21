"""Pure leverage-bracket and margin helpers for the portfolio allocator.

The Binance connector owns payload parsing.  This module deliberately accepts
already-frozen Decimal values so allocation can be repeated byte-for-byte from
one immutable account/bracket snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable


class RiskInputError(ValueError):
    """Raised when a frozen margin or leverage input is not internally safe."""


def _decimal(value: Decimal, name: str, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise RiskInputError(f"{name} must be a finite Decimal")
    if positive and value <= 0:
        raise RiskInputError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise RiskInputError(f"{name} must be non-negative")
    return value


@dataclass(frozen=True, slots=True)
class LeverageBracket:
    """One adjusted Binance leverage-bracket row.

    ``notional_cap`` is exclusive, matching Binance's bracket semantics.  The
    connector's ``notionalCoef`` must already have been applied by the caller.
    """

    bracket: int
    initial_leverage: int
    notional_floor: Decimal
    notional_cap: Decimal
    maint_margin_ratio: Decimal
    cum: Decimal

    def __post_init__(self) -> None:
        if isinstance(self.bracket, bool) or not isinstance(self.bracket, int) or self.bracket <= 0:
            raise RiskInputError("bracket must be a positive integer")
        if (
            isinstance(self.initial_leverage, bool)
            or not isinstance(self.initial_leverage, int)
            or self.initial_leverage <= 0
        ):
            raise RiskInputError("initial leverage must be a positive integer")
        _decimal(self.notional_floor, "notional floor", nonnegative=True)
        _decimal(self.notional_cap, "notional cap", positive=True)
        _decimal(self.maint_margin_ratio, "maintenance margin ratio", nonnegative=True)
        _decimal(self.cum, "maintenance cumulative amount", nonnegative=True)
        if self.notional_cap <= self.notional_floor:
            raise RiskInputError("notional cap must be greater than its floor")

    def contains(self, notional: Decimal) -> bool:
        notional = _decimal(notional, "notional")
        absolute = abs(notional)
        return self.notional_floor <= absolute < self.notional_cap


@dataclass(frozen=True, slots=True)
class LeverageSelection:
    leverage: int
    bracket: LeverageBracket
    max_notional_cap: Decimal


@dataclass(frozen=True, slots=True)
class LeverageSchedule:
    """The immutable bracket schedule for one Binance symbol."""

    symbol: str
    brackets: tuple[LeverageBracket, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise RiskInputError("leverage schedule symbol must be non-empty")
        if not self.brackets:
            raise RiskInputError("leverage schedule must contain at least one bracket")
        previous: LeverageBracket | None = None
        for bracket in self.brackets:
            if not isinstance(bracket, LeverageBracket):
                raise RiskInputError("all schedule entries must be LeverageBracket values")
            if previous is not None:
                if bracket.bracket <= previous.bracket:
                    raise RiskInputError("bracket identifiers must be strictly increasing")
                if bracket.notional_floor != previous.notional_cap:
                    raise RiskInputError("bracket bounds must be contiguous")
                if bracket.initial_leverage > previous.initial_leverage:
                    raise RiskInputError("initial leverage must be non-increasing by bracket")
            elif bracket.notional_floor != 0:
                raise RiskInputError("the first leverage bracket must start at zero")
            previous = bracket

    @classmethod
    def from_binance(cls, source: object) -> "LeverageSchedule":
        """Losslessly adapt the F002 typed bracket snapshot.

        The conversion is deliberately one-way: payload parsing, timestamps,
        and ``notionalCoef`` semantics remain owned by the connector.  Its
        adjusted Decimal values become the allocator's immutable input.
        """

        from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_risk_data import (
            BinancePerpetualLeverageBrackets,
        )

        if not isinstance(source, BinancePerpetualLeverageBrackets):
            raise RiskInputError("source must be BinancePerpetualLeverageBrackets")
        return cls(
            symbol=source.symbol,
            brackets=tuple(
                LeverageBracket(
                    bracket=bracket.bracket,
                    initial_leverage=bracket.initial_leverage,
                    notional_floor=bracket.adjusted_notional_floor,
                    notional_cap=bracket.adjusted_notional_cap,
                    maint_margin_ratio=bracket.maint_margin_ratio,
                    cum=bracket.adjusted_cum,
                )
                for bracket in source.brackets
            ),
        )

    @property
    def supported_leverages(self) -> tuple[int, ...]:
        return tuple(sorted({bracket.initial_leverage for bracket in self.brackets}, reverse=True))

    def bracket_for_notional(self, notional: Decimal) -> LeverageBracket:
        notional = _decimal(notional, "notional")
        for bracket in self.brackets:
            if bracket.contains(notional):
                return bracket
        raise RiskInputError(
            f"absolute notional {abs(notional)} is outside leverage brackets for {self.symbol}"
        )

    def max_notional_for_leverage(self, leverage: int) -> Decimal:
        if isinstance(leverage, bool) or not isinstance(leverage, int) or leverage <= 0:
            raise RiskInputError("leverage must be a positive integer")
        eligible = tuple(bracket for bracket in self.brackets if leverage <= bracket.initial_leverage)
        if not eligible:
            raise RiskInputError(f"leverage {leverage} is not supported for {self.symbol}")
        return eligible[-1].notional_cap

    def select_leverage(self, required_notional: Decimal) -> LeverageSelection:
        """Return the highest leverage whose *exclusive* cap contains capacity.

        Equality intentionally does not fit.  It either belongs to the next
        maintenance bracket or requires a lower leverage with a larger cap.
        """

        required_notional = abs(_decimal(required_notional, "required notional", nonnegative=True))
        for leverage in self.supported_leverages:
            cap = self.max_notional_for_leverage(leverage)
            if required_notional < cap:
                return LeverageSelection(
                    leverage=leverage,
                    bracket=self.bracket_for_notional(required_notional),
                    max_notional_cap=cap,
                )
        raise RiskInputError(
            f"no leverage bracket can contain {required_notional} notional for {self.symbol}"
        )

    def maintenance_margin(self, notional: Decimal) -> Decimal:
        notional = abs(_decimal(notional, "maintenance notional", nonnegative=True))
        bracket = self.bracket_for_notional(notional)
        return max(Decimal("0"), notional * bracket.maint_margin_ratio - bracket.cum)


def maximum_margin(values: Iterable[Decimal]) -> Decimal:
    """Return a deterministic non-negative maximum for a finite collection."""

    parsed = tuple(_decimal(value, "margin", nonnegative=True) for value in values)
    return max(parsed, default=Decimal("0"))
