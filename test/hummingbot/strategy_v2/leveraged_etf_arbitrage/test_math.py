from dataclasses import FrozenInstanceError, fields, is_dataclass
from decimal import Decimal
from enum import Enum

import pytest

from hummingbot.strategy_v2.leveraged_etf_arbitrage.domain import (
    ArbitrageDirection,
    BookSide,
    DepthLevel,
    EntryConfirmationState,
    InsufficientDepthError,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.math import (
    advance_entry_confirmation,
    calculate_hedge_ratio,
    calculate_leg_notionals,
    calculate_leg_quantities,
    calculate_opportunity,
    calculate_round_trip_costs,
    calculate_theoretical_etf_price,
    depth_vwap,
    determine_arbitrage_direction,
    quantize_quantity,
    select_entry_target,
    select_reduce_target,
    stock_book_walk_bp,
)


D = Decimal


@pytest.mark.parametrize(
    ("stock_anchor", "etf_anchor", "multiplier", "stock_price", "expected_h", "expected_theoretical"),
    [
        (D("100"), D("50"), D("2"), D("110"), D("1"), D("60")),
        (D("80"), D("40"), D("3"), D("90"), D("1.5"), D("55")),
        (D("37.5"), D("12.25"), D("2"), D("36"), D("0.6533333333333333333333333333"), D("11.27")),
    ],
)
def test_anchor_ratio_and_theoretical_price_vectors(
    stock_anchor: Decimal,
    etf_anchor: Decimal,
    multiplier: Decimal,
    stock_price: Decimal,
    expected_h: Decimal,
    expected_theoretical: Decimal,
):
    hedge_ratio = calculate_hedge_ratio(stock_anchor, etf_anchor, multiplier)

    assert hedge_ratio == expected_h
    assert calculate_theoretical_etf_price(
        stock_price=stock_price,
        stock_anchor=stock_anchor,
        etf_anchor=etf_anchor,
        etf_daily_multiplier=multiplier,
    ) == expected_theoretical


@pytest.mark.parametrize(
    ("etf_price", "expected_direction", "expected_etf_quantity", "expected_stock_quantity"),
    [
        (D("62"), ArbitrageDirection.SHORT_ETF_LONG_STOCK, D("-10"), D("2.5")),
        (D("58"), ArbitrageDirection.LONG_ETF_SHORT_STOCK, D("10"), D("-2.5")),
    ],
)
def test_both_directions_use_contract_multiplier_aware_quantities(
    etf_price: Decimal,
    expected_direction: ArbitrageDirection,
    expected_etf_quantity: Decimal,
    expected_stock_quantity: Decimal,
):
    theoretical = calculate_theoretical_etf_price(D("110"), D("100"), D("50"), D("2"))
    direction = determine_arbitrage_direction(etf_price, theoretical)
    quantities = calculate_leg_quantities(
        etf_quantity=D("10"),
        direction=direction,
        hedge_ratio=D("1"),
        etf_contract_multiplier=D("0.5"),
        stock_contract_multiplier=D("2"),
    )

    assert direction is expected_direction
    assert quantities.etf_quantity == expected_etf_quantity
    assert quantities.stock_quantity == expected_stock_quantity


def test_quantity_ratio_and_notional_ratio_handle_non_unit_contract_multipliers():
    quantities = calculate_leg_quantities(
        etf_quantity=D("7"),
        direction=ArbitrageDirection.SHORT_ETF_LONG_STOCK,
        hedge_ratio=D("1.25"),
        etf_contract_multiplier=D("0.2"),
        stock_contract_multiplier=D("5"),
    )
    notionals = calculate_leg_notionals(
        quantities=quantities,
        etf_price=D("25"),
        stock_price=D("40"),
        etf_contract_multiplier=D("0.2"),
        stock_contract_multiplier=D("5"),
    )

    assert abs(quantities.stock_quantity / quantities.etf_quantity) == D("0.05")
    assert notionals.stock / notionals.etf == D("2")
    assert notionals.stock / notionals.etf == D("1.25") * D("40") / D("25")
    assert notionals.gross == notionals.etf + notionals.stock


@pytest.mark.parametrize(
    ("etf_price", "expected_direction", "expected_etf_notional"),
    [
        (D("62"), ArbitrageDirection.SHORT_ETF_LONG_STOCK, D("310")),
        (D("58"), ArbitrageDirection.LONG_ETF_SHORT_STOCK, D("290")),
    ],
)
def test_raw_cost_and_net_bp_vectors_for_both_directions(
    etf_price: Decimal,
    expected_direction: ArbitrageDirection,
    expected_etf_notional: Decimal,
):
    opportunity = calculate_opportunity(
        stock_anchor=D("100"),
        etf_anchor=D("50"),
        etf_daily_multiplier=D("2"),
        stock_entry_price=D("110"),
        etf_entry_price=etf_price,
        etf_quantity=D("10"),
        stock_contract_multiplier=D("2"),
        etf_contract_multiplier=D("0.5"),
        maker_fee_bp=D("1"),
        taker_fee_bp=D("4"),
        maker_slippage_bp_per_fill=D("2"),
    )
    expected_stock_notional = D("550")
    expected_gross_notional = expected_etf_notional + expected_stock_notional
    expected_gross_profit = D("10")
    expected_cost_quote = (
        D("2") * D("1") * expected_etf_notional / D("10000")
        + D("2") * D("2") * expected_etf_notional / D("10000")
        + D("2") * D("4") * expected_stock_notional / D("10000")
    )

    assert opportunity.direction is expected_direction
    assert opportunity.theoretical_etf_price == D("60")
    assert opportunity.notionals.etf == expected_etf_notional
    assert opportunity.notionals.stock == expected_stock_notional
    assert opportunity.gross_profit_quote == expected_gross_profit
    assert opportunity.raw_bp == D("10000") * expected_gross_profit / expected_gross_notional
    assert opportunity.costs.total_quote == expected_cost_quote
    assert opportunity.costs.total_bp == D("10000") * expected_cost_quote / expected_gross_notional
    assert opportunity.net_bp == opportunity.raw_bp - opportunity.costs.total_bp


def test_cost_ledger_charges_each_fixed_round_trip_item_once():
    quantities = calculate_leg_quantities(
        etf_quantity=D("10"),
        direction=ArbitrageDirection.SHORT_ETF_LONG_STOCK,
        hedge_ratio=D("1"),
        etf_contract_multiplier=D("0.5"),
        stock_contract_multiplier=D("2"),
    )
    notionals = calculate_leg_notionals(quantities, D("62"), D("110"), D("0.5"), D("2"))
    costs = calculate_round_trip_costs(
        notionals=notionals,
        maker_fee_bp=D("1"),
        taker_fee_bp=D("4"),
        maker_slippage_bp_per_fill=D("2"),
    )

    assert costs.etf_entry_maker_fee_quote == D("0.031")
    assert costs.etf_exit_maker_fee_quote == D("0.031")
    assert costs.stock_entry_taker_fee_quote == D("0.22")
    assert costs.stock_exit_taker_fee_quote == D("0.22")
    assert costs.etf_entry_slippage_budget_quote == D("0.062")
    assert costs.etf_exit_slippage_budget_quote == D("0.062")
    assert costs.total_quote == D("0.626")
    assert {field.name for field in fields(costs)} == {
        "etf_entry_maker_fee_quote",
        "etf_exit_maker_fee_quote",
        "stock_entry_taker_fee_quote",
        "stock_exit_taker_fee_quote",
        "etf_entry_slippage_budget_quote",
        "etf_exit_slippage_budget_quote",
        "total_quote",
        "total_bp",
    }


def test_entry_depth_impact_is_in_g_once_and_future_exit_impact_is_not_reserved():
    opportunity = calculate_opportunity(
        stock_anchor=D("100"),
        etf_anchor=D("50"),
        etf_daily_multiplier=D("2"),
        stock_entry_price=D("110"),
        etf_entry_price=D("62"),
        etf_quantity=D("10"),
        stock_contract_multiplier=D("2"),
        etf_contract_multiplier=D("0.5"),
        maker_fee_bp=D("0"),
        taker_fee_bp=D("4"),
        maker_slippage_bp_per_fill=D("2"),
    )
    visible_entry_walk = stock_book_walk_bp(D("110"), D("108"), BookSide.BUY)

    assert visible_entry_walk > 0
    assert opportunity.gross_profit_quote == D("10")
    assert opportunity.net_bp == opportunity.raw_bp - opportunity.costs.total_bp
    assert opportunity.costs.total_quote == D("0.564")


@pytest.mark.parametrize(
    ("side", "levels", "quantity", "expected"),
    [
        (
            BookSide.BUY,
            (DepthLevel(D("10"), D("2")), DepthLevel(D("11"), D("3"))),
            D("4"),
            D("10.5"),
        ),
        (
            BookSide.SELL,
            (DepthLevel(D("10"), D("2")), DepthLevel(D("9"), D("3"))),
            D("4"),
            D("9.5"),
        ),
        (
            BookSide.BUY,
            (DepthLevel(D("10"), D("2")), DepthLevel(D("10"), D("2"))),
            D("4"),
            D("10"),
        ),
    ],
)
def test_executable_depth_vwap(side: BookSide, levels: tuple[DepthLevel, ...], quantity: Decimal, expected: Decimal):
    assert depth_vwap(levels=levels, quantity=quantity, side=side) == expected


@pytest.mark.parametrize("side", [BookSide.BUY, BookSide.SELL])
def test_depth_vwap_fails_when_full_quantity_is_not_executable(side: BookSide):
    second_price = D("11") if side is BookSide.BUY else D("9")
    levels = (DepthLevel(D("10"), D("2")), DepthLevel(second_price, D("3")))

    with pytest.raises(InsufficientDepthError):
        depth_vwap(levels, D("6"), side)


@pytest.mark.parametrize(
    ("side", "levels"),
    [
        (BookSide.BUY, (DepthLevel(D("11"), D("1")), DepthLevel(D("10"), D("1")))),
        (BookSide.SELL, (DepthLevel(D("10"), D("1")), DepthLevel(D("11"), D("1")))),
    ],
)
def test_depth_vwap_rejects_non_monotonic_book_sides(side: BookSide, levels: tuple[DepthLevel, ...]):
    with pytest.raises(ValueError, match="monotonic"):
        depth_vwap(levels, D("2"), side)


@pytest.mark.parametrize(
    ("vwap", "best_quote", "side", "expected"),
    [
        (D("101"), D("100"), BookSide.BUY, D("100")),
        (D("99"), D("100"), BookSide.SELL, D("100")),
        (D("100"), D("100"), BookSide.BUY, D("0")),
    ],
)
def test_stock_book_walk_bp(vwap: Decimal, best_quote: Decimal, side: BookSide, expected: Decimal):
    assert stock_book_walk_bp(vwap, best_quote, side) == expected


@pytest.mark.parametrize(
    ("net_bp", "expected_target"),
    [
        (D("-1"), D("0")),
        (D("0"), D("0")),
        (D("0.0001"), D("1")),
        (D("22.28"), D("1")),
        (D("22.29"), D("3")),
        (D("44.58"), D("8")),
        (D("999"), D("8")),
    ],
)
def test_entry_tier_lookup_has_nonpositive_zero_exception(net_bp: Decimal, expected_target: Decimal):
    tiers = {D("0"): D("1"), D("22.29"): D("3"), D("44.58"): D("8")}

    assert select_entry_target(net_bp, tiers) == expected_target


def test_entry_confirmation_requires_consecutive_stable_direction_and_target():
    state = EntryConfirmationState()
    state = advance_entry_confirmation(state, ArbitrageDirection.SHORT_ETF_LONG_STOCK, D("3"))
    assert state.consecutive_count == 1
    assert state.confirmed_target(required_confirmations=3) == D("0")

    state = advance_entry_confirmation(state, ArbitrageDirection.SHORT_ETF_LONG_STOCK, D("3"))
    state = advance_entry_confirmation(state, ArbitrageDirection.SHORT_ETF_LONG_STOCK, D("3"))
    assert state.confirmed_target(required_confirmations=3) == D("3")

    state = advance_entry_confirmation(state, ArbitrageDirection.SHORT_ETF_LONG_STOCK, D("8"))
    assert state.consecutive_count == 1
    state = advance_entry_confirmation(state, ArbitrageDirection.LONG_ETF_SHORT_STOCK, D("8"))
    assert state.consecutive_count == 1
    assert state.direction is ArbitrageDirection.LONG_ETF_SHORT_STOCK

    state = advance_entry_confirmation(state, None, D("0"))
    assert state == EntryConfirmationState()


@pytest.mark.parametrize(
    ("net_bp", "current_target", "expected_target"),
    [
        (D("17"), D("3"), D("3")),
        (D("16.999"), D("3"), D("1")),
        (D("0.001"), D("1"), D("1")),
        (D("0"), D("1"), D("0")),
        (D("-0.001"), D("8"), D("0")),
        (D("100"), D("0"), D("0")),
    ],
)
def test_reduce_hysteresis_uses_strict_threshold_and_one_previous_tier(
    net_bp: Decimal,
    current_target: Decimal,
    expected_target: Decimal,
):
    tiers = {D("0"): D("1"), D("22.29"): D("3"), D("44.58"): D("8")}
    reductions = {D("1"): D("0"), D("3"): D("17"), D("8"): D("39")}

    assert select_reduce_target(net_bp, current_target, tiers, reductions) == expected_target


def test_reduce_hysteresis_rejects_unknown_current_target():
    with pytest.raises(ValueError, match="current target"):
        select_reduce_target(D("1"), D("2"), {D("0"): D("1")}, {D("1"): D("0")})


@pytest.mark.parametrize(
    ("quantity", "step", "expected"),
    [
        (D("1.29"), D("0.1"), D("1.2")),
        (D("-1.29"), D("0.1"), D("-1.2")),
        (D("1.2"), D("0.1"), D("1.2")),
        (D("0.009"), D("0.01"), D("0.00")),
    ],
)
def test_quantity_quantization_is_toward_zero(quantity: Decimal, step: Decimal, expected: Decimal):
    assert quantize_quantity(quantity, step) == expected


@pytest.mark.parametrize(
    "call",
    [
        lambda: calculate_hedge_ratio(D("0"), D("50"), D("2")),
        lambda: calculate_hedge_ratio(D("100"), D("-1"), D("2")),
        lambda: calculate_theoretical_etf_price(D("0"), D("100"), D("50"), D("2")),
        lambda: calculate_theoretical_etf_price(D("1"), D("100"), D("50"), D("2")),
        lambda: calculate_leg_quantities(
            D("0"), ArbitrageDirection.SHORT_ETF_LONG_STOCK, D("1"), D("1"), D("1")
        ),
        lambda: DepthLevel(D("0"), D("1")),
        lambda: DepthLevel(D("1"), D("-1")),
        lambda: stock_book_walk_bp(D("0"), D("1"), BookSide.BUY),
        lambda: quantize_quantity(D("1"), D("0")),
    ],
)
def test_zero_and_negative_prices_quantities_and_steps_are_rejected(call):
    with pytest.raises(ValueError):
        call()


@pytest.mark.parametrize(
    "call",
    [
        lambda: calculate_hedge_ratio(100.0, D("50"), D("2")),
        lambda: calculate_theoretical_etf_price(D("110"), D("100"), D("50"), 2.0),
        lambda: calculate_leg_quantities(
            D("1"), ArbitrageDirection.SHORT_ETF_LONG_STOCK, D("1"), 1.0, D("1")
        ),
        lambda: DepthLevel(1.0, D("1")),
        lambda: depth_vwap((DepthLevel(D("1"), D("1")),), 1.0, BookSide.BUY),
        lambda: select_entry_target(1.0, {D("0"): D("1")}),
        lambda: quantize_quantity(1.0, D("0.1")),
    ],
)
def test_math_rejects_binary_float_inputs(call):
    with pytest.raises(TypeError, match="Decimal"):
        call()


class FloatSentinelDecimal(Decimal):
    def __float__(self):
        raise AssertionError("trading decision converted Decimal through binary float")


def _assert_no_float(value: object) -> None:
    assert not isinstance(value, float)
    if is_dataclass(value):
        for field in fields(value):
            _assert_no_float(getattr(value, field.name))
    elif isinstance(value, dict):
        for key, nested in value.items():
            _assert_no_float(key)
            _assert_no_float(nested)
    elif isinstance(value, (tuple, list)):
        for nested in value:
            _assert_no_float(nested)
    elif isinstance(value, Enum):
        _assert_no_float(value.value)


def test_float_sentinel_guards_every_decision_output():
    sentinel = FloatSentinelDecimal
    opportunity = calculate_opportunity(
        stock_anchor=sentinel("100"),
        etf_anchor=sentinel("50"),
        etf_daily_multiplier=sentinel("2"),
        stock_entry_price=sentinel("110"),
        etf_entry_price=sentinel("62"),
        etf_quantity=sentinel("10"),
        stock_contract_multiplier=sentinel("2"),
        etf_contract_multiplier=sentinel("0.5"),
        maker_fee_bp=sentinel("0"),
        taker_fee_bp=sentinel("4"),
        maker_slippage_bp_per_fill=sentinel("2"),
    )
    vwap = depth_vwap(
        (DepthLevel(sentinel("10"), sentinel("2")), DepthLevel(sentinel("11"), sentinel("3"))),
        sentinel("4"),
        BookSide.BUY,
    )
    confirmation = advance_entry_confirmation(
        EntryConfirmationState(),
        ArbitrageDirection.SHORT_ETF_LONG_STOCK,
        sentinel("3"),
    )

    _assert_no_float(opportunity)
    _assert_no_float(vwap)
    _assert_no_float(confirmation)


def test_domain_and_math_results_are_frozen_values():
    level = DepthLevel(D("10"), D("1"))
    state = EntryConfirmationState()

    with pytest.raises(FrozenInstanceError):
        level.price = D("11")
    with pytest.raises(FrozenInstanceError):
        state.consecutive_count = 1
