from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

import pytest

from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, TradeType
from hummingbot.core.event.events import MarketEvent, OrderCancelledEvent
from hummingbot.model.leveraged_etf_repository import JournalEventType
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import LeveragedEtfPairState
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.leveraged_etf_pair_executor import (
    LeveragedEtfPairExecutor,
    LeveragedEtfPairSafetyPolicy,
)
from test.hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.test_partial_fill_hedging import (
    FakeConnector,
    FakeJournalRepository,
    _config,
    _created_event,
    _fill_event,
)


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance_ms(self, milliseconds: int) -> None:
        self.value += milliseconds / 1000


class SafetyFakeConnector(FakeConnector):
    def __init__(self, timeline):
        super().__init__(timeline)
        self.cancel_calls = []
        self.cancel_error = None
        self.statuses = {}
        self.trades = {}
        self.positions = {}

    def cancel(self, trading_pair, client_order_id):
        self.timeline.append(("cancel", trading_pair, client_order_id))
        self.cancel_calls.append((trading_pair, client_order_id))
        if self.cancel_error is not None:
            error = self.cancel_error
            self.cancel_error = None
            raise error
        return client_order_id

    async def get_order_status_by_client_order_id(self, trading_pair, client_order_id):
        return self.statuses.get(
            client_order_id,
            SimpleNamespace(
                status="NEW",
                executed_quantity=Decimal("0"),
                exchange_order_id=None,
            ),
        )

    async def get_account_trades(self, trading_pair, exchange_order_id=None):
        return tuple(self.trades.get((trading_pair, exchange_order_id), ()))

    async def get_position_risk_snapshots(self, trading_pair=None):
        return tuple(self.positions.get(trading_pair, ()))


def _policy(**overrides) -> LeveragedEtfPairSafetyPolicy:
    values = {
        "executor_safety_interval_ms": 1,
        "divergence_cancel_bp": Decimal("3"),
        "divergence_confirmations": 2,
        "maker_max_age_ms": 1_000,
        "unhedged_response_deadline_ms": 20,
        "hedge_phase_deadline_ms": 10,
        "hedge_submit_timeout_ms": 1,
        "hedge_reconcile_timeout_ms": 5,
        "hedge_max_attempts": 2,
        "hedge_retry_backoff_ms": (1,),
        "rollback_submit_timeout_ms": 1,
        "rollback_reconcile_timeout_ms": 5,
        "rollback_phase_deadline_ms": 10,
        "rollback_max_attempts": 2,
        "rollback_retry_backoff_ms": (1,),
        "order_eventual_consistency_grace_ms": 1,
    }
    values.update(overrides)
    return LeveragedEtfPairSafetyPolicy(**values)


def _executor(*, policy=None):
    timeline = []
    clock = FakeClock()
    connector = SafetyFakeConnector(timeline)
    repository = FakeJournalRepository(timeline)
    strategy = SimpleNamespace(connectors={"binance_perpetual": connector}, current_timestamp=1721224862.0)
    executor = LeveragedEtfPairExecutor(
        strategy=strategy,
        config=_config(),
        journal_repository=repository,
        update_interval=0.01,
        safety_policy=policy or _policy(),
        monotonic_clock=clock,
    )
    return executor, connector, repository, clock


async def _start_maker(executor, connector):
    await executor.control_task()
    executor.process_order_created_event(
        MarketEvent.SellOrderCreated.value,
        connector,
        _created_event(
            executor.maker_client_order_id,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
            exchange_order_id="10001",
        ),
    )


def _set_stock_minimum(connector, minimum: Decimal):
    connector.trading_rules["SNDK-USDT"] = TradingRule(
        trading_pair="SNDK-USDT",
        min_order_size=minimum,
        min_price_increment=Decimal("0.01"),
        min_base_amount_increment=minimum,
        min_notional_size=Decimal("1"),
    )


def _set_etf_minimum(connector, minimum: Decimal):
    connector.trading_rules["SNXX-USDT"] = TradingRule(
        trading_pair="SNXX-USDT",
        min_order_size=minimum,
        min_price_increment=Decimal("0.01"),
        min_base_amount_increment=minimum,
        min_notional_size=Decimal("1"),
    )


@pytest.mark.asyncio
async def test_cancel_then_late_fill_rolls_back_only_the_unhedged_increment_after_cancel_confirmation():
    executor, connector, repository, _ = _executor()
    await _start_maker(executor, connector)

    executor.update_maker_safety(net_bp=Decimal("0"))
    await executor.control_task()

    assert executor.state is LeveragedEtfPairState.MAKER_CANCEL_PENDING
    assert connector.cancel_calls == [("SNXX-USDT", executor.maker_client_order_id)]
    assert [event.event_type for event in repository.events][-2:] == [
        JournalEventType.PREPARED,
        JournalEventType.CANCEL_REQUESTED,
    ]

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            executor.maker_client_order_id,
            "late-etf-fill",
            "10",
            timestamp=1.0,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
        ),
    )
    assert not [order for order in connector.orders if order["trading_pair"] == "SNDK-USDT"]

    executor.process_order_canceled_event(
        MarketEvent.OrderCancelled.value,
        connector,
        OrderCancelledEvent(timestamp=1.1, order_id=executor.maker_client_order_id, exchange_order_id="10001"),
    )
    await executor.control_task()

    rollback = connector.orders[-1]
    assert rollback["trading_pair"] == "SNXX-USDT"
    assert rollback["side"] == "BUY"
    assert rollback["order_type"] is OrderType.MARKET
    assert rollback["amount"] == Decimal("10")
    assert rollback["position_action"] is PositionAction.CLOSE


@pytest.mark.asyncio
async def test_cancel_unknown_stays_reconciling_and_escalates_at_absolute_deadline_without_replacement():
    executor, connector, _, clock = _executor()
    await _start_maker(executor, connector)
    connector.cancel_error = RuntimeError("cancel transport ambiguity")

    executor.update_maker_safety(direction_stable=False)
    await executor.control_task()

    assert executor.state is LeveragedEtfPairState.RECONCILING
    assert len(connector.orders) == 1
    clock.advance_ms(20)
    await executor.control_task()
    assert executor.state is LeveragedEtfPairState.RECOVERY_REQUIRED
    assert len(connector.orders) == 1


@pytest.mark.asyncio
async def test_divergence_requires_post_improvement_consecutive_confirmations_and_resets_when_interrupted():
    executor, connector, _, _ = _executor()
    await _start_maker(executor, connector)

    for residual in (Decimal("10"), Decimal("5"), Decimal("9"), Decimal("6"), Decimal("9")):
        executor.update_maker_safety(residual_bp=residual)
        await executor.control_task()
        assert not connector.cancel_calls

    executor.update_maker_safety(residual_bp=Decimal("9"))
    await executor.control_task()
    assert connector.cancel_calls == [("SNXX-USDT", executor.maker_client_order_id)]


@pytest.mark.asyncio
async def test_dust_uses_first_unhedged_t0_and_cancels_at_hedge_deadline_without_extending_it():
    executor, connector, _, clock = _executor()
    await _start_maker(executor, connector)
    _set_stock_minimum(connector, Decimal("20"))

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(executor.maker_client_order_id, "dust-1", "10", timestamp=1.0, trading_pair="SNXX-USDT", side=TradeType.SELL),
    )
    assert executor.exposure_started_at == 0
    assert executor.hedge_deadline_at == pytest.approx(0.01)
    assert executor.hedge_dust_quantity > 0

    clock.advance_ms(5)
    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(executor.maker_client_order_id, "dust-2", "10", timestamp=2.0, trading_pair="SNXX-USDT", side=TradeType.SELL),
    )
    assert executor.exposure_started_at == 0
    assert executor.hedge_deadline_at == pytest.approx(0.01)

    clock.advance_ms(5)
    await executor.control_task()
    assert connector.cancel_calls == [("SNXX-USDT", executor.maker_client_order_id)]
    assert executor.state is LeveragedEtfPairState.MAKER_CANCEL_PENDING


@pytest.mark.asyncio
async def test_only_legal_exact_rollback_is_reduce_only_and_never_oversizes_dust():
    executor, connector, _, clock = _executor()
    await _start_maker(executor, connector)
    _set_stock_minimum(connector, Decimal("20"))

    for trade_id in ("rollback-dust-1", "rollback-dust-2"):
        executor.process_order_filled_event(
            MarketEvent.OrderFilled.value,
            connector,
            _fill_event(executor.maker_client_order_id, trade_id, "10", timestamp=1.0, trading_pair="SNXX-USDT", side=TradeType.SELL),
        )
    clock.advance_ms(10)
    await executor.control_task()
    executor.process_order_canceled_event(
        MarketEvent.OrderCancelled.value,
        connector,
        OrderCancelledEvent(timestamp=1.1, order_id=executor.maker_client_order_id, exchange_order_id="10001"),
    )
    await executor.control_task()

    rollback = connector.orders[-1]
    assert rollback["amount"] == Decimal("20")
    assert rollback["position_action"] is PositionAction.CLOSE


@pytest.mark.asyncio
async def test_when_stock_and_exact_rollback_are_both_illegal_deadline_escalates_without_orders():
    executor, connector, _, clock = _executor()
    await _start_maker(executor, connector)
    _set_stock_minimum(connector, Decimal("20"))
    _set_etf_minimum(connector, Decimal("30"))

    for trade_id in ("illegal-dust-1", "illegal-dust-2"):
        executor.process_order_filled_event(
            MarketEvent.OrderFilled.value,
            connector,
            _fill_event(executor.maker_client_order_id, trade_id, "10", timestamp=1.0, trading_pair="SNXX-USDT", side=TradeType.SELL),
        )
    clock.advance_ms(10)
    await executor.control_task()
    executor.process_order_canceled_event(
        MarketEvent.OrderCancelled.value,
        connector,
        OrderCancelledEvent(timestamp=1.1, order_id=executor.maker_client_order_id, exchange_order_id="10001"),
    )
    await executor.control_task()
    assert len(connector.orders) == 1

    clock.advance_ms(10)
    await executor.control_task()
    assert executor.state is LeveragedEtfPairState.RECOVERY_REQUIRED
    assert len(connector.orders) == 1


@pytest.mark.asyncio
async def test_attempt_budget_exhaustion_cancels_then_rolls_back_after_known_stock_rejection():
    executor, connector, _, _ = _executor(policy=_policy(hedge_max_attempts=1))
    await _start_maker(executor, connector)
    connector.next_submission_error = type(
        "RejectedSubmission",
        (RuntimeError,),
        {"failure_kind": SimpleNamespace(value="AUTHORITATIVE_REJECTION")},
    )("rejected")

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(executor.maker_client_order_id, "reject-hedge", "10", timestamp=1.0, trading_pair="SNXX-USDT", side=TradeType.SELL),
    )
    await executor.control_task()
    assert connector.cancel_calls == [("SNXX-USDT", executor.maker_client_order_id)]

    executor.process_order_canceled_event(
        MarketEvent.OrderCancelled.value,
        connector,
        OrderCancelledEvent(timestamp=1.1, order_id=executor.maker_client_order_id, exchange_order_id="10001"),
    )
    await executor.control_task()
    assert connector.orders[-1]["position_action"] is PositionAction.CLOSE


@pytest.mark.asyncio
async def test_contradictory_reconciliation_facts_fail_closed_without_guessing_a_reverse_order():
    executor, connector, _, clock = _executor()
    await _start_maker(executor, connector)
    maker = executor.maker_client_order_id
    connector.statuses[maker] = SimpleNamespace(
        status="CANCELED",
        executed_quantity=Decimal("10"),
        exchange_order_id="10001",
    )

    executor.mark_intent_submission_unknown(maker, "test transport ambiguity")
    clock.advance_ms(2)
    await executor.control_task()
    clock.advance_ms(1)
    await executor.control_task()

    assert executor.state is LeveragedEtfPairState.RECOVERY_REQUIRED
    assert len(connector.orders) == 1
