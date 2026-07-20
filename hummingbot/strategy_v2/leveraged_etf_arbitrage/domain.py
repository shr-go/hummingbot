from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

from hummingbot.strategy_v2.leveraged_etf_arbitrage.decimal_policy import (
    decision_decimal_context,
    validate_bounded_decimal,
)


def _validate_decimal(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    return validate_bounded_decimal(
        value,
        name,
        positive=positive,
        nonnegative=nonnegative,
    )


class ArbitrageDirection(str, Enum):
    SHORT_ETF_LONG_STOCK = "SHORT_ETF_LONG_STOCK"
    LONG_ETF_SHORT_STOCK = "LONG_ETF_SHORT_STOCK"


class BookSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class InsufficientDepthError(ValueError):
    """Raised when an order book cannot execute the complete requested quantity."""


@dataclass(frozen=True, slots=True)
class DepthLevel:
    price: Decimal
    quantity: Decimal

    def __post_init__(self) -> None:
        _validate_decimal(self.price, "depth price", positive=True)
        _validate_decimal(self.quantity, "depth quantity", positive=True)


@dataclass(frozen=True, slots=True)
class LegQuantities:
    etf_quantity: Decimal
    stock_quantity: Decimal

    def __post_init__(self) -> None:
        _validate_decimal(self.etf_quantity, "ETF quantity")
        _validate_decimal(self.stock_quantity, "stock quantity")
        if self.etf_quantity == 0 or self.stock_quantity == 0:
            raise ValueError("leg quantities must be nonzero")
        if self.etf_quantity.is_signed() == self.stock_quantity.is_signed():
            raise ValueError("ETF and stock quantities must have opposing signs")


@dataclass(frozen=True, slots=True)
class LegNotionals:
    etf: Decimal
    stock: Decimal
    gross: Decimal

    def __post_init__(self) -> None:
        _validate_decimal(self.etf, "ETF notional", positive=True)
        _validate_decimal(self.stock, "stock notional", positive=True)
        _validate_decimal(self.gross, "gross notional", positive=True)
        with decision_decimal_context():
            if self.gross != self.etf + self.stock:
                raise ValueError("gross notional must equal the sum of both absolute leg notionals")


@dataclass(frozen=True, slots=True)
class RoundTripCosts:
    etf_entry_maker_fee_quote: Decimal
    etf_exit_maker_fee_quote: Decimal
    stock_entry_taker_fee_quote: Decimal
    stock_exit_taker_fee_quote: Decimal
    etf_entry_slippage_budget_quote: Decimal
    etf_exit_slippage_budget_quote: Decimal
    total_quote: Decimal
    total_bp: Decimal

    def __post_init__(self) -> None:
        components = (
            self.etf_entry_maker_fee_quote,
            self.etf_exit_maker_fee_quote,
            self.stock_entry_taker_fee_quote,
            self.stock_exit_taker_fee_quote,
            self.etf_entry_slippage_budget_quote,
            self.etf_exit_slippage_budget_quote,
        )
        for component in components:
            _validate_decimal(component, "cost component", nonnegative=True)
        _validate_decimal(self.total_quote, "total cost quote", nonnegative=True)
        _validate_decimal(self.total_bp, "total cost bp", nonnegative=True)
        with decision_decimal_context():
            if self.total_quote != sum(components, start=Decimal("0")):
                raise ValueError("total quote cost must equal the six one-time cost entries")


@dataclass(frozen=True, slots=True)
class Opportunity:
    direction: ArbitrageDirection
    hedge_ratio: Decimal
    theoretical_etf_price: Decimal
    quantities: LegQuantities
    notionals: LegNotionals
    gross_profit_quote: Decimal
    raw_bp: Decimal
    costs: RoundTripCosts
    net_bp: Decimal

    def __post_init__(self) -> None:
        if not isinstance(self.direction, ArbitrageDirection):
            raise TypeError("direction must be an ArbitrageDirection")
        _validate_decimal(self.hedge_ratio, "hedge ratio", positive=True)
        _validate_decimal(self.theoretical_etf_price, "theoretical ETF price", positive=True)
        _validate_decimal(self.gross_profit_quote, "gross profit quote", positive=True)
        _validate_decimal(self.raw_bp, "raw bp", positive=True)
        _validate_decimal(self.net_bp, "net bp")
        with decision_decimal_context():
            if self.net_bp != self.raw_bp - self.costs.total_bp:
                raise ValueError("net bp must subtract the fixed cost ledger exactly once")


@dataclass(frozen=True, slots=True)
class EntryConfirmationState:
    direction: Optional[ArbitrageDirection] = None
    target: Decimal = Decimal("0")
    consecutive_count: int = 0

    def __post_init__(self) -> None:
        _validate_decimal(self.target, "entry target", nonnegative=True)
        if isinstance(self.consecutive_count, bool) or not isinstance(self.consecutive_count, int):
            raise TypeError("consecutive count must be an int")
        if self.direction is None:
            if self.target != 0 or self.consecutive_count != 0:
                raise ValueError("an empty confirmation state must have zero target and count")
        else:
            if not isinstance(self.direction, ArbitrageDirection):
                raise TypeError("confirmation direction must be an ArbitrageDirection")
            if self.target <= 0 or self.consecutive_count <= 0:
                raise ValueError("an active confirmation state must have positive target and count")

    def confirmed_target(self, required_confirmations: int) -> Decimal:
        if isinstance(required_confirmations, bool) or not isinstance(required_confirmations, int):
            raise TypeError("required confirmations must be an int")
        if required_confirmations <= 0:
            raise ValueError("required confirmations must be positive")
        return self.target if self.consecutive_count >= required_confirmations else Decimal("0")
