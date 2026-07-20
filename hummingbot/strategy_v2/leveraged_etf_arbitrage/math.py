from collections.abc import Mapping, Sequence
from decimal import ROUND_DOWN, Decimal
from fractions import Fraction
from functools import wraps
from typing import Optional

from hummingbot.strategy_v2.leveraged_etf_arbitrage.domain import (
    ArbitrageDirection,
    BookSide,
    DepthLevel,
    EntryConfirmationState,
    InsufficientDepthError,
    LegNotionals,
    LegQuantities,
    Opportunity,
    RoundTripCosts,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.decimal_policy import (
    DecisionSemanticKind,
    DecisionValue,
    RawDecisionKind,
    RawExactDecision,
    decision_decimal_context,
    validate_bounded_decimal,
)


BASIS_POINTS = Decimal("10000")


def _decimal(value: object, name: str, *, positive: bool = False, nonnegative: bool = False) -> Decimal:
    return validate_bounded_decimal(
        value,
        name,
        positive=positive,
        nonnegative=nonnegative,
    )


def _decision_parts(
    value: object,
    name: str,
    *,
    semantic_kind: DecisionSemanticKind | None,
    raw_kind: RawDecisionKind | None,
    positive: bool = False,
    nonnegative: bool = False,
) -> tuple[Decimal, Fraction | None]:
    """Authorize only a recomputable derived value or a kind-bound raw value."""

    if isinstance(value, DecisionValue):
        value.validate_integrity()
        display = _decimal(
            value.display,
            f"{name} display",
            positive=positive,
            nonnegative=nonnegative,
        )
        exact = value.exact_fraction
        if exact is not None and (
            semantic_kind is None or value.semantic_kind is not semantic_kind
        ):
            return display, None
        if exact is not None:
            if positive and exact <= 0:
                raise ValueError(f"{name} exact value must be positive")
            if nonnegative and exact < 0:
                raise ValueError(f"{name} exact value must be nonnegative")
        return display, exact
    if isinstance(value, RawExactDecision):
        value.validate_integrity()
        display = _decimal(
            value.value,
            f"{name} raw value",
            positive=positive,
            nonnegative=nonnegative,
        )
        if raw_kind is None or value.kind is not raw_kind:
            return display, None
        return display, value.exact_fraction
    display = _decimal(value, name, positive=positive, nonnegative=nonnegative)
    return display, None


def _threshold_display(
    value: object,
    name: str,
    *,
    nonnegative: bool = False,
) -> tuple[Decimal, bool]:
    """Parse a configured threshold and report whether derived input is trusted."""

    if isinstance(value, DecisionValue):
        value.validate_integrity()
        return (
            _decimal(value.display, f"{name} display", nonnegative=nonnegative),
            value.exact_fraction is not None
            and value.semantic_kind is DecisionSemanticKind.OPPORTUNITY_NET_BP,
        )
    return _decimal(value, name, nonnegative=nonnegative), True


def _fixed_decimal_context(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        with decision_decimal_context():
            return function(*args, **kwargs)

    return wrapped


def _exact_theoretical_price(
    stock_price: Decimal,
    stock_anchor: Decimal,
    etf_anchor: Decimal,
    etf_daily_multiplier: Decimal,
) -> Fraction:
    stock_price_fraction = Fraction(stock_price)
    stock_anchor_fraction = Fraction(stock_anchor)
    return Fraction(etf_anchor) * (
        1
        + Fraction(etf_daily_multiplier)
        * (stock_price_fraction / stock_anchor_fraction - 1)
    )


def _display_exact_fraction(
    exact_value: Fraction,
    field_name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> Decimal:
    display_value = Decimal(exact_value.numerator) / Decimal(exact_value.denominator)
    display_value = validate_bounded_decimal(
        display_value,
        field_name,
        positive=positive,
        nonnegative=nonnegative,
    )
    return display_value


def _exact_opportunity_bp(
    *,
    stock_anchor: Decimal,
    etf_anchor: Decimal,
    etf_daily_multiplier: Decimal,
    stock_entry_price: Decimal,
    etf_entry_price: Decimal,
    etf_quantity: Decimal,
    stock_contract_multiplier: Decimal,
    etf_contract_multiplier: Decimal,
    maker_fee_bp: Decimal,
    taker_fee_bp: Decimal,
    maker_slippage_bp_per_fill: Decimal,
) -> tuple[Fraction, Fraction, Fraction]:
    stock_anchor_fraction = Fraction(stock_anchor)
    etf_anchor_fraction = Fraction(etf_anchor)
    multiplier_fraction = Fraction(etf_daily_multiplier)
    stock_price_fraction = Fraction(stock_entry_price)
    etf_price_fraction = Fraction(etf_entry_price)
    etf_quantity_fraction = Fraction(etf_quantity)
    stock_contract_fraction = Fraction(stock_contract_multiplier)
    etf_contract_fraction = Fraction(etf_contract_multiplier)
    exact_hedge_ratio = multiplier_fraction * etf_anchor_fraction / stock_anchor_fraction
    exact_theoretical = _exact_theoretical_price(
        stock_entry_price,
        stock_anchor,
        etf_anchor,
        etf_daily_multiplier,
    )
    exact_stock_quantity = (
        etf_quantity_fraction
        * etf_contract_fraction
        / stock_contract_fraction
        * exact_hedge_ratio
    )
    exact_etf_notional = etf_quantity_fraction * etf_contract_fraction * etf_price_fraction
    exact_stock_notional = exact_stock_quantity * stock_contract_fraction * stock_price_fraction
    exact_gross_notional = exact_etf_notional + exact_stock_notional
    exact_gross_profit = (
        abs(etf_price_fraction - exact_theoretical)
        * etf_quantity_fraction
        * etf_contract_fraction
    )
    exact_raw_bp = Fraction(BASIS_POINTS) * exact_gross_profit / exact_gross_notional
    exact_total_quote = 2 * (
        exact_etf_notional * Fraction(maker_fee_bp) / Fraction(BASIS_POINTS)
        + exact_stock_notional * Fraction(taker_fee_bp) / Fraction(BASIS_POINTS)
        + exact_etf_notional
        * Fraction(maker_slippage_bp_per_fill)
        / Fraction(BASIS_POINTS)
    )
    exact_cost_bp = Fraction(BASIS_POINTS) * exact_total_quote / exact_gross_notional
    return exact_gross_profit, exact_raw_bp, exact_raw_bp - exact_cost_bp


def _direction(value: object) -> ArbitrageDirection:
    if not isinstance(value, ArbitrageDirection):
        raise TypeError("direction must be an ArbitrageDirection")
    return value


def _book_side(value: object) -> BookSide:
    if not isinstance(value, BookSide):
        raise TypeError("book side must be a BookSide")
    return value


@_fixed_decimal_context
def calculate_hedge_ratio(
    stock_anchor: Decimal,
    etf_anchor: Decimal,
    etf_daily_multiplier: Decimal,
) -> Decimal:
    stock_anchor = _decimal(stock_anchor, "stock anchor", positive=True)
    etf_anchor = _decimal(etf_anchor, "ETF anchor", positive=True)
    etf_daily_multiplier = _decimal(etf_daily_multiplier, "ETF daily multiplier", positive=True)
    exact_ratio = Fraction(etf_daily_multiplier) * Fraction(etf_anchor) / Fraction(stock_anchor)
    return _display_exact_fraction(
        exact_ratio,
        "hedge ratio result",
        positive=True,
    )


@_fixed_decimal_context
def calculate_theoretical_etf_price(
    stock_price: Decimal,
    stock_anchor: Decimal,
    etf_anchor: Decimal,
    etf_daily_multiplier: Decimal,
) -> DecisionValue:
    stock_price = _decimal(stock_price, "stock price", positive=True)
    stock_anchor = _decimal(stock_anchor, "stock anchor", positive=True)
    etf_anchor = _decimal(etf_anchor, "ETF anchor", positive=True)
    etf_daily_multiplier = _decimal(etf_daily_multiplier, "ETF daily multiplier", positive=True)
    exact_price = _exact_theoretical_price(
        stock_price,
        stock_anchor,
        etf_anchor,
        etf_daily_multiplier,
    )
    if exact_price <= 0:
        raise ValueError("theoretical ETF price must be positive")
    return DecisionValue.from_theoretical_price_operands(
        stock_price=stock_price,
        stock_anchor=stock_anchor,
        etf_anchor=etf_anchor,
        etf_daily_multiplier=etf_daily_multiplier,
    )


@_fixed_decimal_context
def determine_arbitrage_direction(
    etf_price: Decimal | RawExactDecision | DecisionValue,
    theoretical_etf_price: Decimal | RawExactDecision | DecisionValue,
) -> Optional[ArbitrageDirection]:
    _, exact_etf_price = _decision_parts(
        etf_price,
        "ETF price",
        semantic_kind=None,
        raw_kind=RawDecisionKind.ETF_PRICE,
        positive=True,
    )
    _, exact_theoretical_price = _decision_parts(
        theoretical_etf_price,
        "theoretical ETF price",
        semantic_kind=DecisionSemanticKind.THEORETICAL_ETF_PRICE,
        raw_kind=None,
        positive=True,
    )
    if exact_etf_price is None or exact_theoretical_price is None:
        return None
    if exact_etf_price > exact_theoretical_price:
        return ArbitrageDirection.SHORT_ETF_LONG_STOCK
    if exact_etf_price < exact_theoretical_price:
        return ArbitrageDirection.LONG_ETF_SHORT_STOCK
    return None


@_fixed_decimal_context
def calculate_leg_quantities(
    etf_quantity: Decimal,
    direction: ArbitrageDirection,
    hedge_ratio: Decimal,
    etf_contract_multiplier: Decimal,
    stock_contract_multiplier: Decimal,
) -> LegQuantities:
    etf_quantity = _decimal(etf_quantity, "ETF quantity", positive=True)
    direction = _direction(direction)
    hedge_ratio = _decimal(hedge_ratio, "hedge ratio", positive=True)
    etf_contract_multiplier = _decimal(etf_contract_multiplier, "ETF contract multiplier", positive=True)
    stock_contract_multiplier = _decimal(stock_contract_multiplier, "stock contract multiplier", positive=True)
    stock_quantity = etf_quantity * (etf_contract_multiplier / stock_contract_multiplier) * hedge_ratio
    if direction is ArbitrageDirection.SHORT_ETF_LONG_STOCK:
        return LegQuantities(etf_quantity=-etf_quantity, stock_quantity=stock_quantity)
    return LegQuantities(etf_quantity=etf_quantity, stock_quantity=-stock_quantity)


@_fixed_decimal_context
def calculate_leg_notionals(
    quantities: LegQuantities,
    etf_price: Decimal,
    stock_price: Decimal,
    etf_contract_multiplier: Decimal,
    stock_contract_multiplier: Decimal,
) -> LegNotionals:
    if not isinstance(quantities, LegQuantities):
        raise TypeError("quantities must be LegQuantities")
    etf_price = _decimal(etf_price, "ETF price", positive=True)
    stock_price = _decimal(stock_price, "stock price", positive=True)
    etf_contract_multiplier = _decimal(etf_contract_multiplier, "ETF contract multiplier", positive=True)
    stock_contract_multiplier = _decimal(stock_contract_multiplier, "stock contract multiplier", positive=True)
    etf_notional = abs(quantities.etf_quantity) * etf_contract_multiplier * etf_price
    stock_notional = abs(quantities.stock_quantity) * stock_contract_multiplier * stock_price
    return LegNotionals(etf=etf_notional, stock=stock_notional, gross=etf_notional + stock_notional)


@_fixed_decimal_context
def calculate_round_trip_costs(
    notionals: LegNotionals,
    maker_fee_bp: Decimal,
    taker_fee_bp: Decimal,
    maker_slippage_bp_per_fill: Decimal,
) -> RoundTripCosts:
    if not isinstance(notionals, LegNotionals):
        raise TypeError("notionals must be LegNotionals")
    maker_fee_bp = _decimal(maker_fee_bp, "maker fee bp", nonnegative=True)
    taker_fee_bp = _decimal(taker_fee_bp, "taker fee bp", nonnegative=True)
    maker_slippage_bp_per_fill = _decimal(
        maker_slippage_bp_per_fill,
        "maker slippage bp per fill",
        nonnegative=True,
    )
    etf_maker_fee = notionals.etf * maker_fee_bp / BASIS_POINTS
    stock_taker_fee = notionals.stock * taker_fee_bp / BASIS_POINTS
    etf_slippage = notionals.etf * maker_slippage_bp_per_fill / BASIS_POINTS
    total_quote = Decimal("2") * (etf_maker_fee + stock_taker_fee + etf_slippage)
    return RoundTripCosts(
        etf_entry_maker_fee_quote=etf_maker_fee,
        etf_exit_maker_fee_quote=etf_maker_fee,
        stock_entry_taker_fee_quote=stock_taker_fee,
        stock_exit_taker_fee_quote=stock_taker_fee,
        etf_entry_slippage_budget_quote=etf_slippage,
        etf_exit_slippage_budget_quote=etf_slippage,
        total_quote=total_quote,
        total_bp=BASIS_POINTS * total_quote / notionals.gross,
    )


@_fixed_decimal_context
def calculate_opportunity(
    *,
    stock_anchor: Decimal,
    etf_anchor: Decimal,
    etf_daily_multiplier: Decimal,
    stock_entry_price: Decimal,
    etf_entry_price: Decimal,
    etf_quantity: Decimal,
    stock_contract_multiplier: Decimal,
    etf_contract_multiplier: Decimal,
    maker_fee_bp: Decimal,
    taker_fee_bp: Decimal,
    maker_slippage_bp_per_fill: Decimal,
) -> Opportunity:
    stock_anchor = _decimal(stock_anchor, "stock anchor", positive=True)
    etf_anchor = _decimal(etf_anchor, "ETF anchor", positive=True)
    etf_daily_multiplier = _decimal(etf_daily_multiplier, "ETF daily multiplier", positive=True)
    stock_entry_price = _decimal(stock_entry_price, "stock entry price", positive=True)
    etf_entry_price = _decimal(etf_entry_price, "ETF entry price", positive=True)
    etf_quantity = _decimal(etf_quantity, "ETF quantity", positive=True)
    stock_contract_multiplier = _decimal(
        stock_contract_multiplier,
        "stock contract multiplier",
        positive=True,
    )
    etf_contract_multiplier = _decimal(
        etf_contract_multiplier,
        "ETF contract multiplier",
        positive=True,
    )
    maker_fee_bp = _decimal(maker_fee_bp, "maker fee bp", nonnegative=True)
    taker_fee_bp = _decimal(taker_fee_bp, "taker fee bp", nonnegative=True)
    maker_slippage_bp_per_fill = _decimal(
        maker_slippage_bp_per_fill,
        "maker slippage bp per fill",
        nonnegative=True,
    )
    theoretical_price = calculate_theoretical_etf_price(
        stock_entry_price,
        stock_anchor,
        etf_anchor,
        etf_daily_multiplier,
    )
    exact_theoretical_price = _exact_theoretical_price(
        stock_entry_price,
        stock_anchor,
        etf_anchor,
        etf_daily_multiplier,
    )
    etf_entry_fraction = Fraction(etf_entry_price)
    if etf_entry_fraction > exact_theoretical_price:
        direction = ArbitrageDirection.SHORT_ETF_LONG_STOCK
    elif etf_entry_fraction < exact_theoretical_price:
        direction = ArbitrageDirection.LONG_ETF_SHORT_STOCK
    else:
        direction = None
    if direction is None:
        raise ValueError("entry prices contain no directional spread")
    hedge_ratio = calculate_hedge_ratio(stock_anchor, etf_anchor, etf_daily_multiplier)
    quantities = calculate_leg_quantities(
        etf_quantity,
        direction,
        hedge_ratio,
        etf_contract_multiplier,
        stock_contract_multiplier,
    )
    notionals = calculate_leg_notionals(
        quantities,
        etf_entry_price,
        stock_entry_price,
        etf_contract_multiplier,
        stock_contract_multiplier,
    )
    exact_gross_profit, _exact_raw_bp, _exact_net_bp = _exact_opportunity_bp(
        stock_anchor=stock_anchor,
        etf_anchor=etf_anchor,
        etf_daily_multiplier=etf_daily_multiplier,
        stock_entry_price=stock_entry_price,
        etf_entry_price=etf_entry_price,
        etf_quantity=etf_quantity,
        stock_contract_multiplier=stock_contract_multiplier,
        etf_contract_multiplier=etf_contract_multiplier,
        maker_fee_bp=maker_fee_bp,
        taker_fee_bp=taker_fee_bp,
        maker_slippage_bp_per_fill=maker_slippage_bp_per_fill,
    )
    gross_profit_quote = _display_exact_fraction(
        exact_gross_profit,
        "gross profit quote result",
        positive=True,
    )
    raw_bp_display = BASIS_POINTS * gross_profit_quote / notionals.gross
    costs = calculate_round_trip_costs(
        notionals,
        maker_fee_bp,
        taker_fee_bp,
        maker_slippage_bp_per_fill,
    )
    net_bp_display = raw_bp_display - costs.total_bp
    net_bp = DecisionValue.from_opportunity_net_bp_operands(
        stock_anchor=stock_anchor,
        etf_anchor=etf_anchor,
        etf_daily_multiplier=etf_daily_multiplier,
        stock_entry_price=stock_entry_price,
        etf_entry_price=etf_entry_price,
        etf_quantity=etf_quantity,
        stock_contract_multiplier=stock_contract_multiplier,
        etf_contract_multiplier=etf_contract_multiplier,
        maker_fee_bp=maker_fee_bp,
        taker_fee_bp=taker_fee_bp,
        maker_slippage_bp_per_fill=maker_slippage_bp_per_fill,
    )
    if net_bp.display != net_bp_display:
        raise ArithmeticError("semantic net bp display diverges from the opportunity ledger")
    return Opportunity(
        direction=direction,
        hedge_ratio=hedge_ratio,
        theoretical_etf_price=theoretical_price,
        quantities=quantities,
        notionals=notionals,
        gross_profit_quote=gross_profit_quote,
        raw_bp=raw_bp_display,
        costs=costs,
        net_bp=net_bp,
    )


@_fixed_decimal_context
def depth_vwap(
    levels: Sequence[DepthLevel],
    quantity: Decimal,
    side: BookSide,
) -> Decimal:
    quantity = _decimal(quantity, "depth quantity", positive=True)
    side = _book_side(side)
    remaining = quantity
    quote_value = Decimal("0")
    previous_price: Optional[Decimal] = None
    for level in levels:
        if not isinstance(level, DepthLevel):
            raise TypeError("every order-book level must be a DepthLevel")
        if previous_price is not None:
            if side is BookSide.BUY and level.price < previous_price:
                raise ValueError("buy-side depth prices must be monotonic nondecreasing")
            if side is BookSide.SELL and level.price > previous_price:
                raise ValueError("sell-side depth prices must be monotonic nonincreasing")
        previous_price = level.price
        if remaining > 0:
            filled = min(remaining, level.quantity)
            quote_value += filled * level.price
            remaining -= filled
    if remaining > 0:
        raise InsufficientDepthError("order book cannot execute the full stock quantity")
    return validate_bounded_decimal(
        quote_value / quantity,
        "stock VWAP result",
        positive=True,
    )


@_fixed_decimal_context
def stock_book_walk_bp(vwap: Decimal, best_quote: Decimal, side: BookSide) -> Decimal:
    vwap = _decimal(vwap, "stock VWAP", positive=True)
    best_quote = _decimal(best_quote, "best stock quote", positive=True)
    side = _book_side(side)
    if side is BookSide.BUY:
        impact = (vwap / best_quote - Decimal("1")) * BASIS_POINTS
    else:
        impact = (Decimal("1") - vwap / best_quote) * BASIS_POINTS
    if impact < 0:
        raise ValueError("stock VWAP cannot improve beyond the same-side best quote")
    return validate_bounded_decimal(impact, "stock book walk result", nonnegative=True)


def _ordered_tiers(
    position_tiers: Mapping[Decimal | DecisionValue, Decimal],
) -> tuple[tuple[tuple[Decimal, Decimal], ...], bool]:
    if not isinstance(position_tiers, Mapping):
        raise TypeError("position tiers must be a mapping")
    parsed: list[tuple[Decimal, Decimal]] = []
    all_thresholds_trusted = True
    for threshold, target in position_tiers.items():
        threshold_display, threshold_is_trusted = _threshold_display(
            threshold,
            "tier threshold",
            nonnegative=True,
        )
        all_thresholds_trusted = all_thresholds_trusted and threshold_is_trusted
        parsed.append(
            (
                threshold_display,
                _decimal(target, "tier target", positive=True),
            )
        )
    ordered = tuple(sorted(parsed, key=lambda tier: tier[0]))
    if not ordered or ordered[0][0] != 0:
        raise ValueError("position tiers must contain the base threshold 0")
    if any(current[1] < previous[1] for previous, current in zip(ordered, ordered[1:])):
        raise ValueError("position tier targets must be monotonic nondecreasing")
    return ordered, all_thresholds_trusted


@_fixed_decimal_context
def select_entry_target(
    net_bp: Decimal | RawExactDecision | DecisionValue,
    position_tiers: Mapping[Decimal | DecisionValue, Decimal],
) -> Decimal:
    _, exact_net_bp = _decision_parts(
        net_bp,
        "net bp",
        semantic_kind=DecisionSemanticKind.OPPORTUNITY_NET_BP,
        raw_kind=RawDecisionKind.NET_BP,
    )
    ordered_tiers, all_thresholds_trusted = _ordered_tiers(position_tiers)
    if exact_net_bp is None or not all_thresholds_trusted:
        return Decimal("0")
    if exact_net_bp <= 0:
        return Decimal("0")
    return max(
        target
        for threshold, target in ordered_tiers
        if Fraction(threshold) <= exact_net_bp
    )


@_fixed_decimal_context
def advance_entry_confirmation(
    state: EntryConfirmationState,
    direction: Optional[ArbitrageDirection],
    target: Decimal,
) -> EntryConfirmationState:
    if not isinstance(state, EntryConfirmationState):
        raise TypeError("state must be an EntryConfirmationState")
    target = _decimal(target, "entry target", nonnegative=True)
    if direction is None or target == 0:
        return EntryConfirmationState()
    direction = _direction(direction)
    if state.direction is direction and state.target == target:
        return EntryConfirmationState(
            direction=direction,
            target=target,
            consecutive_count=state.consecutive_count + 1,
        )
    return EntryConfirmationState(direction=direction, target=target, consecutive_count=1)


@_fixed_decimal_context
def select_reduce_target(
    net_bp: Decimal | RawExactDecision | DecisionValue,
    current_target: Decimal,
    position_tiers: Mapping[Decimal | DecisionValue, Decimal],
    reduce_bp_by_current_target: Mapping[Decimal, Decimal | DecisionValue],
) -> Decimal:
    _, exact_net_bp = _decision_parts(
        net_bp,
        "net bp",
        semantic_kind=DecisionSemanticKind.OPPORTUNITY_NET_BP,
        raw_kind=RawDecisionKind.NET_BP,
    )
    current_target = _decimal(current_target, "current target", nonnegative=True)
    ordered_tiers, all_tier_thresholds_trusted = _ordered_tiers(position_tiers)
    if current_target == 0:
        return Decimal("0")
    ordered_targets: list[Decimal] = []
    for _, target in ordered_tiers:
        if not ordered_targets or target != ordered_targets[-1]:
            ordered_targets.append(target)
    if current_target not in ordered_targets:
        raise ValueError("current target is not a configured position tier")
    if not isinstance(reduce_bp_by_current_target, Mapping):
        raise TypeError("reduce thresholds must be a mapping")
    parsed_reductions: dict[Decimal, Decimal] = {}
    all_reduce_thresholds_trusted = True
    for target, threshold in reduce_bp_by_current_target.items():
        parsed_target = _decimal(target, "reduce target", positive=True)
        parsed_threshold, threshold_is_trusted = _threshold_display(
            threshold,
            "reduce threshold",
            nonnegative=True,
        )
        parsed_reductions[parsed_target] = parsed_threshold
        all_reduce_thresholds_trusted = all_reduce_thresholds_trusted and threshold_is_trusted
    if current_target not in parsed_reductions:
        raise ValueError("current target has no reduce threshold")
    current_index = ordered_targets.index(current_target)
    previous_target = Decimal("0") if current_index == 0 else ordered_targets[current_index - 1]
    if (
        exact_net_bp is None
        or not all_tier_thresholds_trusted
        or not all_reduce_thresholds_trusted
    ):
        return previous_target
    if exact_net_bp <= 0:
        return Decimal("0")
    if exact_net_bp < Fraction(parsed_reductions[current_target]):
        return previous_target
    return current_target


@_fixed_decimal_context
def quantize_quantity(quantity: Decimal, step: Decimal) -> Decimal:
    quantity = _decimal(quantity, "quantity")
    step = _decimal(step, "quantity step", positive=True)
    quantized_abs = (abs(quantity) / step).to_integral_value(rounding=ROUND_DOWN) * step
    if quantized_abs.is_zero():
        result = quantized_abs.copy_abs()
    else:
        result = -quantized_abs if quantity < 0 else quantized_abs
    return validate_bounded_decimal(result, "quantized quantity result")
