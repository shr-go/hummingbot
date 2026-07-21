from dataclasses import replace
from decimal import Decimal

import pytest

from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_risk_data import (
    BinancePerpetualAccountRiskSnapshot,
    BinancePerpetualLeverageBracket,
    BinancePerpetualLeverageBrackets,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.allocator import (
    AccountRiskSnapshot,
    AllocationStatus,
    AllocationTier,
    BookLevel,
    FrozenAllocationSnapshot,
    FrozenLeg,
    FrozenPair,
    PortfolioAllocator,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.domain import ArbitrageDirection
from hummingbot.strategy_v2.leveraged_etf_arbitrage.risk import LeverageBracket, LeverageSchedule


D = Decimal


def _schedule(symbol: str, leverage: int = 100, cap: str = "1000000") -> LeverageSchedule:
    return LeverageSchedule(
        symbol=symbol,
        brackets=(
            LeverageBracket(
                bracket=1,
                initial_leverage=leverage,
                notional_floor=D("0"),
                notional_cap=D(cap),
                maint_margin_ratio=D("0.005"),
                cum=D("0"),
            ),
        ),
    )


def _leg(
    symbol: str,
    mark: str,
    schedule: LeverageSchedule | None = None,
    **overrides,
) -> FrozenLeg:
    values = {
        "symbol": symbol,
        "mark_price": D(mark),
        "contract_multiplier": D("1"),
        "quantity_step": D("1"),
        "min_quantity": D("1"),
        "min_notional": D("1"),
        "current_quantity": D("0"),
        "owned_open_quantity": D("0"),
        "owned_reservation_quantity": D("0"),
        "exchange_position_initial_margin": D("0"),
        "exchange_open_order_initial_margin": D("0"),
        "exchange_maint_margin": D("0"),
        "leverage_schedule": schedule or _schedule(symbol),
    }
    values.update(overrides)
    return FrozenLeg(**values)


def _pair(
    pair_id: str,
    requested_ratio: str = "10",
    etf_schedule: LeverageSchedule | None = None,
    stock_schedule: LeverageSchedule | None = None,
    **overrides,
) -> FrozenPair:
    etf = _leg(f"{pair_id}-ETF", "60", etf_schedule)
    stock = _leg(f"{pair_id}-STOCK", "100", stock_schedule)
    values = {
        "pair_id": pair_id,
        "direction": ArbitrageDirection.SHORT_ETF_LONG_STOCK,
        "stock_anchor": D("100"),
        "etf_anchor": D("50"),
        "etf_daily_multiplier": D("2"),
        "requested_target_ratio": D(requested_ratio),
        "tiers": (AllocationTier(minimum_net_bp=D("0"), target_ratio=D(requested_ratio)),),
        "p99_bp": D("0"),
        "current_net_bp": D("1000"),
        "max_stock_taker_impact_bp": D("1000"),
        "etf_best_bid": D("59"),
        "etf_best_ask": D("60"),
        "stock_bids": (BookLevel(price=D("100"), quantity=D("100000")),),
        "stock_asks": (BookLevel(price=D("100"), quantity=D("100000")),),
        "etf": etf,
        "stock": stock,
    }
    values.update(overrides)
    return FrozenPair(**values)


def _account(
    equity: str = "1000",
    total_initial_margin: str = "0",
    total_maint_margin: str = "0",
    available_balance: str | None = None,
) -> AccountRiskSnapshot:
    return AccountRiskSnapshot(
        total_margin_balance=D(equity),
        total_initial_margin=D(total_initial_margin),
        total_maint_margin=D(total_maint_margin),
        available_balance=D(available_balance if available_balance is not None else equity),
        normal_initial_margin_budget_ratio=D("0.80"),
        p99_initial_margin_budget_ratio=D("0.85"),
        stress_extra_bp=D("100"),
        stress_equity_to_maintenance_margin_multiple=D("2"),
        reconciliation_tolerance=D("0.01"),
    )


def _snapshot(
    *pairs: FrozenPair,
    account: AccountRiskSnapshot | None = None,
    max_total_notional_ratio: str = "32",
) -> FrozenAllocationSnapshot:
    return FrozenAllocationSnapshot(
        account=account or _account(),
        pairs=tuple(pairs),
        max_total_notional_ratio=D(max_total_notional_ratio),
    )


def _pair_result(result, pair_id: str):
    return next(pair for pair in result.candidate.pairs if pair.pair_id == pair_id)


def test_four_pairs_share_global_cap_and_repeat_exactly() -> None:
    snapshot = _snapshot(*(_pair(f"pair-{index}") for index in range(4)))

    first = PortfolioAllocator().allocate(snapshot)
    second = PortfolioAllocator().allocate(snapshot)

    assert first.status is AllocationStatus.ALLOCATED
    assert first.nominal_scale == D("0.8")
    assert first == second
    assert first.deterministic_hash == second.deterministic_hash
    assert tuple(pair.pair_id for pair in first.candidate.pairs) == ("pair-0", "pair-1", "pair-2", "pair-3")
    assert first.evaluation_count <= 24
    assert sum(pair.target_gross_notional for pair in first.candidate.pairs) <= D("32000")


def test_lattice_returns_unique_maximum_and_exact_neighbor_validation() -> None:
    pair = _pair(
        "margin-boundary",
        requested_ratio="32",
        etf_schedule=_schedule("margin-boundary-ETF", leverage=32),
        stock_schedule=_schedule("margin-boundary-STOCK", leverage=32),
    )

    result = PortfolioAllocator().allocate(_snapshot(pair))

    assert result.status is AllocationStatus.ALLOCATED
    assert 0 < result.risk_scale_n < 1_000_000
    assert result.risk_scale == D(result.risk_scale_n) / D("1000000")
    assert result.n_star_feasible is True
    assert result.n_star_plus_one_feasible is False
    assert result.evaluation_count <= 24


def test_leverage_ineligible_pair_is_skipped_whole_and_remaining_pair_reallocates() -> None:
    ineligible = _pair(
        "ineligible",
        requested_ratio="20",
        etf_schedule=_schedule("ineligible-ETF", cap="1000"),
        stock_schedule=_schedule("ineligible-STOCK", cap="1000"),
    )
    eligible = _pair("eligible", requested_ratio="20")

    result = PortfolioAllocator().allocate(_snapshot(ineligible, eligible))

    assert result.status is AllocationStatus.ALLOCATED
    assert result.skipped_leverage_pair_ids == ("ineligible",)
    assert _pair_result(result, "ineligible").target_gross_notional == D("0")
    assert _pair_result(result, "eligible").target_gross_notional > D("16000")


def test_candidate_slice_is_maximal_and_is_reused_by_the_selected_result() -> None:
    pair = _pair(
        "slice",
        requested_ratio="5",
        max_stock_taker_impact_bp=D("10"),
        stock_asks=(
            BookLevel(price=D("100"), quantity=D("1")),
            BookLevel(price=D("110"), quantity=D("1000")),
        ),
    )

    result = PortfolioAllocator().allocate(_snapshot(pair))
    candidate = _pair_result(result, "slice")

    assert result.status is AllocationStatus.ALLOCATED
    assert candidate.canonical_etf_slice_quantity == D("1")
    assert candidate.canonical_stock_slice_quantity == D("1")
    assert candidate.stock_vwap == D("100")
    assert candidate.canonical_slice_reused is True
    assert {state.name for state in candidate.states} >= {
        "CURRENT_RESERVATION",
        "MAKER_OPEN",
        "ETF_ONE_LEG",
        "STOCK_PARTIAL",
        "SLICE_HEDGED",
        "FINAL",
    }


def test_open_order_slice_margin_can_reduce_n_star_without_post_search_substitution() -> None:
    expensive_open_order = _leg(
        "slice-margin-ETF",
        "60",
        _schedule("slice-margin-ETF", leverage=100),
        new_open_order_initial_margin_rate=D("0.50"),
    )
    pair = _pair(
        "slice-margin",
        requested_ratio="16",
        etf=expensive_open_order,
        stock=_leg("slice-margin-STOCK", "100", _schedule("slice-margin-STOCK", leverage=100)),
    )

    result = PortfolioAllocator().allocate(
        _snapshot(pair, account=_account(equity="1000", available_balance="300"))
    )
    candidate = _pair_result(result, "slice-margin")

    assert result.status is AllocationStatus.ALLOCATED
    assert result.risk_scale_n < 1_000_000
    assert candidate.canonical_slice_reused is True
    assert candidate.max_state_initial_margin > candidate.final_state_initial_margin


def test_unmodeled_margin_residuals_and_available_balance_are_preserved() -> None:
    etf = _leg(
        "residual-ETF",
        "60",
        exchange_position_initial_margin=D("2"),
        exchange_maint_margin=D("1"),
    )
    stock = _leg(
        "residual-STOCK",
        "100",
        exchange_position_initial_margin=D("3"),
        exchange_maint_margin=D("2"),
    )
    pair = _pair("residual", requested_ratio="1", etf=etf, stock=stock)
    account = _account(
        total_initial_margin="10",
        total_maint_margin="8",
        available_balance="10",
    )

    result = PortfolioAllocator().allocate(_snapshot(pair, account=account))

    assert result.status is AllocationStatus.ALLOCATED
    assert result.candidate.im_unmodeled == D("5")
    assert result.candidate.mm_unmodeled == D("5")
    assert result.candidate.projected_initial_margin >= D("10")
    assert result.candidate.delta_initial_margin_available <= D("10")


def test_f_zero_infeasible_only_allows_reduction_and_bad_books_fail_closed() -> None:
    current = _leg(
        "reduction-ETF",
        "60",
        current_quantity=D("-1"),
        exchange_position_initial_margin=D("1"),
        exchange_maint_margin=D("1"),
    )
    stock = _leg(
        "reduction-STOCK",
        "100",
        current_quantity=D("1"),
        exchange_position_initial_margin=D("1"),
        exchange_maint_margin=D("1"),
    )
    reduction_pair = _pair("reduction", requested_ratio="0", etf=current, stock=stock)
    reduction_only = PortfolioAllocator().allocate(
        _snapshot(reduction_pair, account=_account(equity="2", total_initial_margin="2", total_maint_margin="2"))
    )
    bad_book = _pair(
        "bad-book",
        stock_asks=(
            BookLevel(price=D("101"), quantity=D("1")),
            BookLevel(price=D("100"), quantity=D("1")),
        ),
    )

    assert reduction_only.status is AllocationStatus.RISK_REDUCTION_ONLY
    assert reduction_only.risk_scale_n == 0
    assert PortfolioAllocator().allocate(_snapshot(bad_book)).status is AllocationStatus.FAIL_CLOSED


def test_bracket_cap_equality_uses_the_next_lower_leverage() -> None:
    schedule = LeverageSchedule(
        symbol="cap-equality",
        brackets=(
            LeverageBracket(1, 32, D("0"), D("1000"), D("0.005"), D("0")),
            LeverageBracket(2, 20, D("1000"), D("2000"), D("0.01"), D("5")),
        ),
    )

    selection = schedule.select_leverage(D("1000"))

    assert selection.leverage == 20
    assert selection.max_notional_cap == D("2000")


def test_stale_snapshot_and_no_legal_slice_do_not_create_a_small_order() -> None:
    stale = _pair("stale", stale=True)
    no_slice = _pair(
        "no-slice",
        requested_ratio="0.01",
        etf=_leg("no-slice-ETF", "60", min_quantity=D("100")),
    )

    stale_result = PortfolioAllocator().allocate(_snapshot(stale))
    no_slice_result = PortfolioAllocator().allocate(_snapshot(no_slice))

    assert stale_result.status is AllocationStatus.FAIL_CLOSED
    assert no_slice_result.status is AllocationStatus.ALLOCATED
    assert _pair_result(no_slice_result, "no-slice").canonical_etf_slice_quantity == D("0")
    assert _pair_result(no_slice_result, "no-slice").target_gross_notional == D("0")


@pytest.mark.parametrize("quantity_step", [D("1"), D("0.5")])
def test_quantization_plateaus_are_deterministic(quantity_step: Decimal) -> None:
    pair = _pair(
        "plateau",
        requested_ratio="0.11",
        etf=_leg("plateau-ETF", "60", quantity_step=quantity_step),
    )
    snapshot = _snapshot(pair)

    first = PortfolioAllocator().allocate(snapshot)
    second = PortfolioAllocator().allocate(snapshot)

    assert first == second
    assert first.deterministic_hash == second.deterministic_hash


def test_signal_cap_descent_is_finite_and_does_not_restore_a_higher_tier() -> None:
    pair = _pair(
        "cap-descent",
        requested_ratio="10",
        tiers=(
            AllocationTier(minimum_net_bp=D("0"), target_ratio=D("5")),
            AllocationTier(minimum_net_bp=D("600"), target_ratio=D("10")),
        ),
        max_stock_taker_impact_bp=D("1000"),
        stock_asks=(
            BookLevel(price=D("100"), quantity=D("1")),
            BookLevel(price=D("110"), quantity=D("100000")),
        ),
    )

    result = PortfolioAllocator().allocate(_snapshot(pair))

    assert result.status is AllocationStatus.ALLOCATED
    assert result.cap_descent_count == 1
    assert D("0") < _pair_result(result, "cap-descent").target_gross_notional <= D("5000")


def test_exact_available_balance_boundary_passes_and_lower_value_reduces_the_lattice() -> None:
    pair = _pair("available", requested_ratio="4")
    base_snapshot = _snapshot(pair)
    base = PortfolioAllocator().allocate(base_snapshot)
    exact_account = replace(base_snapshot.account, available_balance=base.candidate.delta_initial_margin_available)
    just_below_account = replace(
        exact_account,
        available_balance=base.candidate.delta_initial_margin_available - D("0.000001"),
    )

    exact = PortfolioAllocator().allocate(replace(base_snapshot, account=exact_account))
    below = PortfolioAllocator().allocate(replace(base_snapshot, account=just_below_account))

    assert exact.status is AllocationStatus.ALLOCATED
    assert exact.risk_scale_n == base.risk_scale_n
    assert below.status is AllocationStatus.ALLOCATED
    assert below.risk_scale_n < exact.risk_scale_n


def test_p99_budget_stress_and_negative_margin_residuals_are_accounted_for() -> None:
    p99_pair = _pair("p99", requested_ratio="4", p99_bp=D("100"))
    p99 = PortfolioAllocator().allocate(_snapshot(p99_pair))
    invalid = _pair(
        "negative-residual",
        etf=_leg("negative-residual-ETF", "60", exchange_position_initial_margin=D("2")),
        stock=_leg("negative-residual-STOCK", "100", exchange_position_initial_margin=D("2")),
    )
    invalid_account = _account(total_initial_margin="3")

    assert p99.status is AllocationStatus.ALLOCATED
    assert p99.candidate.initial_margin_budget == D("850")
    assert p99.candidate.stress_loss >= D("0")
    assert (
        PortfolioAllocator().allocate(_snapshot(invalid, account=invalid_account)).status
        is AllocationStatus.FAIL_CLOSED
    )


def test_typed_f002_account_and_bracket_adapters_remain_pure() -> None:
    source_brackets = BinancePerpetualLeverageBrackets(
        symbol="adapter",
        notional_coef=D("1"),
        brackets=(
            BinancePerpetualLeverageBracket(
                bracket=1,
                initial_leverage=32,
                notional_cap=D("1000"),
                notional_floor=D("0"),
                maint_margin_ratio=D("0.005"),
                cum=D("0"),
                notional_coef=D("1"),
            ),
        ),
        data_time=1.0,
        cache_time=1.0,
    )
    source_account = BinancePerpetualAccountRiskSnapshot(
        total_initial_margin=D("1"),
        total_maint_margin=D("1"),
        total_wallet_balance=D("100"),
        total_unrealized_profit=D("0"),
        total_margin_balance=D("100"),
        total_position_initial_margin=D("1"),
        total_open_order_initial_margin=D("0"),
        total_cross_wallet_balance=D("100"),
        total_cross_unrealized_profit=D("0"),
        available_balance=D("99"),
        max_withdraw_amount=D("99"),
        assets=(),
        positions=(),
        data_time=1.0,
    )

    schedule = LeverageSchedule.from_binance(source_brackets)
    account = AccountRiskSnapshot.from_binance(
        source_account,
        normal_initial_margin_budget_ratio=D("0.80"),
        p99_initial_margin_budget_ratio=D("0.85"),
        stress_extra_bp=D("100"),
        stress_equity_to_maintenance_margin_multiple=D("2"),
        reconciliation_tolerance=D("0.01"),
    )

    assert schedule.select_leverage(D("999")).leverage == 32
    assert account.total_margin_balance == D("100")
    assert account.available_balance == D("99")
