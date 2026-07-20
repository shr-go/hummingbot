from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction
from typing import Optional

from hummingbot.strategy_v2.leveraged_etf_arbitrage.decimal_policy import (
    DecisionCertainty,
    DecisionSemanticKind,
    DecisionValue,
    decision_decimal_context,
    validate_bounded_decimal,
)


_BASIS_POINTS = Decimal("10000")


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
    theoretical_etf_price: DecisionValue
    quantities: LegQuantities
    notionals: LegNotionals
    gross_profit_quote: Decimal
    raw_bp: Decimal
    costs: RoundTripCosts
    net_bp: DecisionValue

    def __post_init__(self) -> None:
        self.validate_integrity()

    def validate_integrity(self) -> None:
        """Recompute the complete opportunity from its bound raw semantic operands."""

        if not isinstance(self.direction, ArbitrageDirection):
            raise TypeError("direction must be an ArbitrageDirection")
        _validate_decimal(self.hedge_ratio, "hedge ratio", positive=True)
        for decision_value, field_name, semantic_kind in (
            (
                self.theoretical_etf_price,
                "theoretical ETF price",
                DecisionSemanticKind.THEORETICAL_ETF_PRICE,
            ),
            (self.net_bp, "net bp", DecisionSemanticKind.OPPORTUNITY_NET_BP),
        ):
            if not isinstance(decision_value, DecisionValue):
                raise TypeError(f"{field_name} must be a DecisionValue")
            decision_value.validate_integrity()
            if decision_value.certainty is not DecisionCertainty.EXACT_DERIVED:
                raise ValueError(f"{field_name} must retain exact decision authority")
            if decision_value.semantic_kind is not semantic_kind:
                raise ValueError(f"{field_name} has the wrong decision semantic kind")
        if not isinstance(self.quantities, LegQuantities):
            raise TypeError("opportunity quantities must be LegQuantities")
        if not isinstance(self.notionals, LegNotionals):
            raise TypeError("opportunity notionals must be LegNotionals")
        if not isinstance(self.costs, RoundTripCosts):
            raise TypeError("opportunity costs must be RoundTripCosts")
        _validate_decimal(self.theoretical_etf_price.display, "theoretical ETF price", positive=True)
        _validate_decimal(self.gross_profit_quote, "gross profit quote", positive=True)
        _validate_decimal(self.raw_bp, "raw bp", positive=True)
        _validate_decimal(self.net_bp.display, "net bp")

        theoretical_values = self.theoretical_etf_price.verified_operand_values(
            DecisionSemanticKind.THEORETICAL_ETF_PRICE
        )
        net_values = self.net_bp.verified_operand_values(
            DecisionSemanticKind.OPPORTUNITY_NET_BP
        )
        theoretical_bindings = {
            "stock_price": net_values["stock_entry_price"],
            "stock_anchor": net_values["stock_anchor"],
            "etf_anchor": net_values["etf_anchor"],
            "etf_daily_multiplier": net_values["etf_daily_multiplier"],
        }
        if theoretical_values != theoretical_bindings:
            raise ValueError(
                "opportunity theoretical operands do not match the net-bp raw inputs"
            )

        stock_anchor = Fraction(net_values["stock_anchor"])
        etf_anchor = Fraction(net_values["etf_anchor"])
        multiplier = Fraction(net_values["etf_daily_multiplier"])
        stock_price = Fraction(net_values["stock_entry_price"])
        etf_price = Fraction(net_values["etf_entry_price"])
        etf_quantity = Fraction(net_values["etf_quantity"])
        stock_contract = Fraction(net_values["stock_contract_multiplier"])
        etf_contract = Fraction(net_values["etf_contract_multiplier"])
        exact_hedge_ratio = multiplier * etf_anchor / stock_anchor
        exact_theoretical = etf_anchor * (
            1 + multiplier * (stock_price / stock_anchor - 1)
        )
        if exact_theoretical <= 0:
            raise ValueError("opportunity theoretical ETF price must be positive")
        if etf_price > exact_theoretical:
            expected_direction = ArbitrageDirection.SHORT_ETF_LONG_STOCK
        elif etf_price < exact_theoretical:
            expected_direction = ArbitrageDirection.LONG_ETF_SHORT_STOCK
        else:
            raise ValueError("opportunity raw inputs contain no directional spread")

        with decision_decimal_context():
            expected_hedge_ratio = (
                Decimal(exact_hedge_ratio.numerator)
                / Decimal(exact_hedge_ratio.denominator)
            )
            expected_theoretical_display = (
                Decimal(exact_theoretical.numerator)
                / Decimal(exact_theoretical.denominator)
            )
            expected_stock_quantity = (
                net_values["etf_quantity"]
                * (
                    net_values["etf_contract_multiplier"]
                    / net_values["stock_contract_multiplier"]
                )
                * expected_hedge_ratio
            )
            if expected_direction is ArbitrageDirection.SHORT_ETF_LONG_STOCK:
                expected_quantities = LegQuantities(
                    etf_quantity=-net_values["etf_quantity"],
                    stock_quantity=expected_stock_quantity,
                )
            else:
                expected_quantities = LegQuantities(
                    etf_quantity=net_values["etf_quantity"],
                    stock_quantity=-expected_stock_quantity,
                )
            expected_etf_notional = (
                abs(expected_quantities.etf_quantity)
                * net_values["etf_contract_multiplier"]
                * net_values["etf_entry_price"]
            )
            expected_stock_notional = (
                abs(expected_quantities.stock_quantity)
                * net_values["stock_contract_multiplier"]
                * net_values["stock_entry_price"]
            )
            expected_notionals = LegNotionals(
                etf=expected_etf_notional,
                stock=expected_stock_notional,
                gross=expected_etf_notional + expected_stock_notional,
            )
            exact_gross_profit = (
                abs(etf_price - exact_theoretical) * etf_quantity * etf_contract
            )
            expected_gross_profit = (
                Decimal(exact_gross_profit.numerator)
                / Decimal(exact_gross_profit.denominator)
            )
            expected_raw_bp = (
                _BASIS_POINTS
                * expected_gross_profit
                / expected_notionals.gross
            )
            etf_maker_fee = (
                expected_notionals.etf
                * net_values["maker_fee_bp"]
                / _BASIS_POINTS
            )
            stock_taker_fee = (
                expected_notionals.stock
                * net_values["taker_fee_bp"]
                / _BASIS_POINTS
            )
            etf_slippage = (
                expected_notionals.etf
                * net_values["maker_slippage_bp_per_fill"]
                / _BASIS_POINTS
            )
            expected_total_quote = Decimal("2") * (
                etf_maker_fee + stock_taker_fee + etf_slippage
            )
            expected_costs = RoundTripCosts(
                etf_entry_maker_fee_quote=etf_maker_fee,
                etf_exit_maker_fee_quote=etf_maker_fee,
                stock_entry_taker_fee_quote=stock_taker_fee,
                stock_exit_taker_fee_quote=stock_taker_fee,
                etf_entry_slippage_budget_quote=etf_slippage,
                etf_exit_slippage_budget_quote=etf_slippage,
                total_quote=expected_total_quote,
                total_bp=(
                    _BASIS_POINTS
                    * expected_total_quote
                    / expected_notionals.gross
                ),
            )
            expected_net_bp_display = expected_raw_bp - expected_costs.total_bp

        exact_stock_quantity = (
            etf_quantity * etf_contract / stock_contract * exact_hedge_ratio
        )
        exact_etf_notional = etf_quantity * etf_contract * etf_price
        exact_stock_notional = exact_stock_quantity * stock_contract * stock_price
        exact_gross_notional = exact_etf_notional + exact_stock_notional
        exact_raw_bp = (
            Fraction(_BASIS_POINTS)
            * exact_gross_profit
            / exact_gross_notional
        )
        exact_total_quote = 2 * (
            exact_etf_notional
            * Fraction(net_values["maker_fee_bp"])
            / Fraction(_BASIS_POINTS)
            + exact_stock_notional
            * Fraction(net_values["taker_fee_bp"])
            / Fraction(_BASIS_POINTS)
            + exact_etf_notional
            * Fraction(net_values["maker_slippage_bp_per_fill"])
            / Fraction(_BASIS_POINTS)
        )
        exact_net_bp = exact_raw_bp - (
            Fraction(_BASIS_POINTS) * exact_total_quote / exact_gross_notional
        )

        if self.direction is not expected_direction:
            raise ValueError("opportunity direction does not match its raw inputs")
        if self.hedge_ratio != expected_hedge_ratio:
            raise ValueError("opportunity hedge ratio does not match its raw inputs")
        if self.theoretical_etf_price.exact_fraction != exact_theoretical:
            raise ValueError("opportunity theoretical exact value does not match its raw inputs")
        if self.theoretical_etf_price.display != expected_theoretical_display:
            raise ValueError("opportunity theoretical display does not match its raw inputs")
        if self.quantities != expected_quantities:
            raise ValueError("opportunity quantities do not match its raw inputs")
        if self.notionals != expected_notionals:
            raise ValueError("opportunity notionals do not match its raw inputs")
        if self.gross_profit_quote != expected_gross_profit:
            raise ValueError("opportunity gross profit does not match its raw inputs")
        if self.raw_bp != expected_raw_bp:
            raise ValueError("opportunity raw bp does not match its raw inputs")
        if self.costs != expected_costs:
            raise ValueError("opportunity cost ledger does not match its raw inputs")
        if self.net_bp.exact_fraction != exact_net_bp:
            raise ValueError("opportunity net bp exact value does not match its raw inputs")
        if self.net_bp.display != expected_net_bp_display:
            raise ValueError("opportunity net bp display does not match its raw inputs")

    def __reduce__(self):
        self.validate_integrity()
        return (
            type(self),
            (
                self.direction,
                self.hedge_ratio,
                self.theoretical_etf_price,
                self.quantities,
                self.notionals,
                self.gross_profit_quote,
                self.raw_bp,
                self.costs,
                self.net_bp,
            ),
        )


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
