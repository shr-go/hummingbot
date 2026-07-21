import copy
import hashlib
import json
from dataclasses import replace
from decimal import Decimal

import pytest

from hummingbot.strategy_v2.leveraged_etf_arbitrage import decimal_policy
from hummingbot.strategy_v2.leveraged_etf_arbitrage.domain import (
    ArbitrageDirection,
    LegQuantities,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.math import (
    calculate_opportunity,
    quantize_quantity,
    select_entry_target,
    select_reduce_target,
)


D = Decimal


def _rehash_decision_fields(fields_payload: dict[str, object]) -> None:
    payload = {key: value for key, value in fields_payload.items() if key != "integrity_hash"}
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    fields_payload["integrity_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _replace_operand(fields_payload: dict[str, object], name: str, value: str) -> None:
    fields_payload["operands"] = [
        [operand_name, value if operand_name == name else operand_value]
        for operand_name, operand_value in fields_payload["operands"]
    ]


def _canonical_slice(etf_entry_price: Decimal = D("44.1096")):
    desired_etf_quantity = D("7.34")
    etf_quantity = quantize_quantity(desired_etf_quantity, D("0.1"))
    ideal_stock_quantity = (
        etf_quantity
        * (D("0.6") / D("1.5"))
        * (D("2.5") * D("40") / D("125"))
    )
    stock_quantity = quantize_quantity(ideal_stock_quantity, D("0.25"))
    opportunity = calculate_opportunity(
        stock_anchor=D("125"),
        etf_anchor=D("40"),
        etf_daily_multiplier=D("2.5"),
        stock_entry_price=D("130"),
        etf_entry_price=etf_entry_price,
        etf_quantity=etf_quantity,
        stock_quantity=stock_quantity,
        stock_contract_multiplier=D("1.5"),
        etf_contract_multiplier=D("0.6"),
        maker_fee_bp=D("1"),
        taker_fee_bp=D("4"),
        maker_slippage_bp_per_fill=D("2"),
    )
    return opportunity, ideal_stock_quantity


def test_v3_opportunity_binds_actual_quantized_slice_and_changes_boundary_authority():
    actual, ideal_stock_quantity = _canonical_slice()
    ideal = calculate_opportunity(
        stock_anchor=D("125"),
        etf_anchor=D("40"),
        etf_daily_multiplier=D("2.5"),
        stock_entry_price=D("130"),
        etf_entry_price=D("44.1096"),
        etf_quantity=D("7.3"),
        stock_quantity=ideal_stock_quantity,
        stock_contract_multiplier=D("1.5"),
        etf_contract_multiplier=D("0.6"),
        maker_fee_bp=D("1"),
        taker_fee_bp=D("4"),
        maker_slippage_bp_per_fill=D("2"),
    )
    operands = actual.net_bp.verified_operand_values(
        decimal_policy.DecisionSemanticKind.OPPORTUNITY_NET_BP
    )

    assert actual.schema_version == 3
    assert actual.net_bp.schema_version == 3
    assert operands["etf_quantity"] == D("7.3")
    assert operands["stock_quantity"] == D("2.25")
    assert actual.quantities == LegQuantities(etf_quantity=D("-7.3"), stock_quantity=D("2.25"))
    assert actual.hedge_ratio == D("0.8")
    assert actual.executable_hedge_ratio == D("3.375") / D("4.38")
    assert actual.notionals.etf == D("193.200048")
    assert actual.notionals.stock == D("438.750")
    assert actual.notionals.gross == D("631.950048")
    assert actual.gross_profit_quote == D("0.480048")
    assert actual.raw_bp == D("10000") * actual.gross_profit_quote / actual.notionals.gross
    assert actual.costs.total_quote == sum(
        (
            actual.costs.etf_entry_maker_fee_quote,
            actual.costs.etf_exit_maker_fee_quote,
            actual.costs.stock_entry_taker_fee_quote,
            actual.costs.stock_exit_taker_fee_quote,
            actual.costs.etf_entry_slippage_budget_quote,
            actual.costs.etf_exit_slippage_budget_quote,
        ),
        start=D("0"),
    )
    assert actual.net_bp > D("0.2")
    assert ideal.net_bp < D("0")

    tiers = {D("0"): D("1"), D("0.2"): D("3")}
    reductions = {D("1"): D("0"), D("3"): D("0.2")}
    assert select_entry_target(actual.net_bp, tiers) == D("3")
    assert select_entry_target(ideal.net_bp, tiers) == D("0")
    assert select_reduce_target(actual.net_bp, D("3"), tiers, reductions) == D("3")
    assert select_reduce_target(ideal.net_bp, D("3"), tiers, reductions) == D("0")


@pytest.mark.parametrize(
    ("etf_entry_price", "expected_direction", "expected_quantities"),
    [
        (
            D("44.1096"),
            ArbitrageDirection.SHORT_ETF_LONG_STOCK,
            LegQuantities(etf_quantity=D("-7.3"), stock_quantity=D("2.25")),
        ),
        (
            D("43.8904"),
            ArbitrageDirection.LONG_ETF_SHORT_STOCK,
            LegQuantities(etf_quantity=D("7.3"), stock_quantity=D("-2.25")),
        ),
    ],
)
def test_executable_quantities_cover_both_directions_and_every_cost_entry(
    etf_entry_price,
    expected_direction,
    expected_quantities,
):
    opportunity, _ = _canonical_slice(etf_entry_price)

    assert opportunity.direction is expected_direction
    assert opportunity.quantities == expected_quantities
    assert all(
        component > 0
        for component in (
            opportunity.costs.etf_entry_maker_fee_quote,
            opportunity.costs.etf_exit_maker_fee_quote,
            opportunity.costs.stock_entry_taker_fee_quote,
            opportunity.costs.stock_exit_taker_fee_quote,
            opportunity.costs.etf_entry_slippage_budget_quote,
            opportunity.costs.etf_exit_slippage_budget_quote,
        )
    )


@pytest.mark.parametrize(
    ("quantity_name", "replacement"),
    [("etf_quantity", "7.2"), ("stock_quantity", "2")],
)
def test_serialized_authority_and_composite_integrity_reject_altered_actual_quantities(
    quantity_name,
    replacement,
):
    opportunity, _ = _canonical_slice()
    fields = copy.deepcopy(opportunity.net_bp.to_fields())
    _replace_operand(fields, quantity_name, replacement)
    _rehash_decision_fields(fields)

    with pytest.raises(decimal_policy.DecisionValueIntegrityError, match="recomputed|fraction|display"):
        decimal_policy.DecisionValue.from_fields(fields)

    if quantity_name == "etf_quantity":
        changed = LegQuantities(etf_quantity=D("-7.2"), stock_quantity=D("2.25"))
    else:
        changed = LegQuantities(etf_quantity=D("-7.3"), stock_quantity=D("2"))
    with pytest.raises(ValueError, match="opportunity|quantit|notional|bp|cost"):
        replace(opportunity, quantities=changed)


def test_zero_or_out_of_domain_actual_hedge_fails_closed_after_step_quantization():
    too_small = quantize_quantity(D("0.24"), D("0.25"))
    assert too_small == 0

    for invalid_stock_quantity in (too_small, D("-2.25"), D("1e13")):
        with pytest.raises((TypeError, ValueError), match="stock|quantity|positive|domain|exponent"):
            calculate_opportunity(
                stock_anchor=D("125"),
                etf_anchor=D("40"),
                etf_daily_multiplier=D("2.5"),
                stock_entry_price=D("130"),
                etf_entry_price=D("44.1096"),
                etf_quantity=D("7.3"),
                stock_quantity=invalid_stock_quantity,
                stock_contract_multiplier=D("1.5"),
                etf_contract_multiplier=D("0.6"),
                maker_fee_bp=D("1"),
                taker_fee_bp=D("4"),
                maker_slippage_bp_per_fill=D("2"),
            )
