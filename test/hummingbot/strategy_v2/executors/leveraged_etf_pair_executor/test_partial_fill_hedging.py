from __future__ import annotations

import asyncio
from collections import defaultdict
from decimal import Decimal
from types import SimpleNamespace

import pytest

from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.data_type.trade_fee import AddedToCostTradeFee
from hummingbot.core.event.events import (
    BuyOrderCreatedEvent,
    MarketEvent,
    MarketOrderFailureEvent,
    OrderFilledEvent,
    SellOrderCreatedEvent,
)
from hummingbot.model.leveraged_etf_repository import JournalEventType, LeveragedEtfJournalRepository
from hummingbot.model.sql_connection_manager import SQLConnectionManager, SQLConnectionType
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import (
    LeverageReservationV1,
    LeveragedEtfPairDirection,
    LeveragedEtfPairExecutorConfig,
    LeveragedEtfPairOperation,
    LeveragedEtfPairState,
)
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.leveraged_etf_pair_executor import (
    LeveragedEtfPairExecutor,
)


class FakeJournalRepository:
    """Records durable calls so order submission order is observable in tests."""

    def __init__(self, timeline: list[tuple]):
        self.timeline = timeline
        self.created_snapshots = []
        self.events = []

    def create_executor(self, snapshot):
        self.created_snapshots.append(snapshot)
        self.timeline.append(("journal-create", snapshot.executor_id))
        return snapshot

    def append_and_reduce(self, executor_id, event):
        if hasattr(event.payload, "identity"):
            assert executor_id == event.payload.identity.executor_id
        self.events.append(event)
        self.timeline.append(("journal", event.event_type.value, event))
        return SimpleNamespace(event=event)


class FakeConnector:
    def __init__(self, timeline: list[tuple], *, stock_step: Decimal = Decimal("0.1")):
        self.timeline = timeline
        self.listeners = defaultdict(list)
        self.orders = []
        self.preflight_calls = []
        self.preflight_error = None
        self.next_submission_error = None
        self.async_submission_unknown_on_next_submit = False
        self._unknown_submission_order_ids = set()
        self.trading_rules = {
            "SNXX-USDT": TradingRule(
                trading_pair="SNXX-USDT",
                min_order_size=Decimal("0.1"),
                min_price_increment=Decimal("0.01"),
                min_base_amount_increment=Decimal("0.1"),
                min_notional_size=Decimal("1"),
            ),
            "SNDK-USDT": TradingRule(
                trading_pair="SNDK-USDT",
                min_order_size=stock_step,
                min_price_increment=Decimal("0.01"),
                min_base_amount_increment=stock_step,
                min_notional_size=Decimal("1"),
            ),
        }

    def add_listener(self, event_tag, listener):
        self.listeners[event_tag].append(listener)

    def remove_listener(self, event_tag, listener):
        self.listeners[event_tag].remove(listener)

    async def strict_account_preflight(
        self,
        trading_pairs,
        related_trading_pairs=None,
        known_position_trading_pairs=None,
        **_,
    ):
        self.preflight_calls.append(tuple(trading_pairs))
        if self.preflight_error is not None:
            raise self.preflight_error
        return SimpleNamespace(
            instruments=(
                SimpleNamespace(trading_pair="SNXX-USDT", contract_multiplier=Decimal("2")),
                SimpleNamespace(trading_pair="SNDK-USDT", contract_multiplier=Decimal("0.5")),
            )
        )

    def get_price_by_type(self, trading_pair, price_type):
        prices = {
            ("SNXX-USDT", PriceType.BestAsk): Decimal("30"),
            ("SNXX-USDT", PriceType.BestBid): Decimal("29.9"),
            ("SNDK-USDT", PriceType.BestAsk): Decimal("250"),
            ("SNDK-USDT", PriceType.BestBid): Decimal("249.9"),
        }
        return prices[(trading_pair, price_type)]

    def buy(self, trading_pair, amount, order_type, price, client_order_id=None, **kwargs):
        return self._submit("BUY", trading_pair, amount, order_type, price, client_order_id, kwargs)

    def sell(self, trading_pair, amount, order_type, price, client_order_id=None, **kwargs):
        return self._submit("SELL", trading_pair, amount, order_type, price, client_order_id, kwargs)

    def _submit(self, side, trading_pair, amount, order_type, price, client_order_id, kwargs):
        self.timeline.append(("connector", side, client_order_id))
        if self.next_submission_error is not None:
            error = self.next_submission_error
            self.next_submission_error = None
            raise error
        self.orders.append(
            {
                "side": side,
                "trading_pair": trading_pair,
                "amount": amount,
                "order_type": order_type,
                "price": price,
                "client_order_id": client_order_id,
                "position_action": kwargs.get("position_action"),
            }
        )
        if self.async_submission_unknown_on_next_submit:
            self.async_submission_unknown_on_next_submit = False
            asyncio.get_running_loop().call_soon(self._record_submission_unknown, client_order_id)
        return client_order_id

    def _record_submission_unknown(self, client_order_id):
        self.timeline.append(("connector-submission-unknown", client_order_id))
        self._unknown_submission_order_ids.add(client_order_id)

    def is_order_submission_unknown(self, client_order_id):
        return client_order_id in self._unknown_submission_order_ids


def _config(
    *,
    operation: LeveragedEtfPairOperation = LeveragedEtfPairOperation.OPEN,
) -> LeveragedEtfPairExecutorConfig:
    return LeveragedEtfPairExecutorConfig(
        id="f005-exec-1",
        timestamp=1721224862.0,
        controller_id="controller-from-config",
        schema_version=1,
        pair_id="sndk_snxx",
        nav_cycle_id="xnys-2026-07-17",
        operation=operation,
        direction=LeveragedEtfPairDirection.SHORT_ETF_LONG_STOCK,
        etf_connector_name="binance_perpetual",
        etf_trading_pair="SNXX-USDT",
        stock_connector_name="binance_perpetual",
        stock_trading_pair="SNDK-USDT",
        s0="250",
        l0="30",
        h="0.24",
        created_raw_bp="48.75",
        created_net_bp="42.08",
        target_gross_notional="10000",
        etf_target_quantity="100",
        # 100 ETF * (2 ETF multiplier / .5 stock multiplier) * h(.24) = 96 stock.
        stock_target_quantity="96",
        leverage_reservation=LeverageReservationV1(
            etf_quantity="100",
            stock_quantity="96",
            etf_leverage=20,
            stock_leverage=20,
            etf_notional_cap="1000000",
            stock_notional_cap="1000000",
        ),
        config_hash="a" * 64,
        created_at_utc="2026-07-17T14:01:02.000000Z",
    )


def _executor(
    *,
    stock_step: Decimal = Decimal("0.1"),
    operation: LeveragedEtfPairOperation = LeveragedEtfPairOperation.OPEN,
):
    timeline = []
    connector = FakeConnector(timeline, stock_step=stock_step)
    repository = FakeJournalRepository(timeline)
    strategy = SimpleNamespace(connectors={"binance_perpetual": connector}, current_timestamp=1721224862.0)
    executor = LeveragedEtfPairExecutor(
        strategy=strategy,
        config=_config(operation=operation),
        journal_repository=repository,
        update_interval=0.01,
    )
    return executor, connector, repository, timeline


def _fill_event(order_id: str, trade_id: str, amount: str, *, timestamp: float, trading_pair: str, side: TradeType):
    return OrderFilledEvent(
        timestamp=timestamp,
        order_id=order_id,
        exchange_order_id="10001" if trading_pair == "SNXX-USDT" else "20001",
        exchange_trade_id=trade_id,
        trading_pair=trading_pair,
        trade_type=side,
        order_type=OrderType.LIMIT_MAKER if trading_pair == "SNXX-USDT" else OrderType.MARKET,
        price=Decimal("30") if trading_pair == "SNXX-USDT" else Decimal("250"),
        amount=Decimal(amount),
        trade_fee=AddedToCostTradeFee(percent=Decimal("0")),
        position=PositionAction.OPEN.value,
    )


def _created_event(order_id: str, *, trading_pair: str, side: TradeType, exchange_order_id: str):
    event_class = BuyOrderCreatedEvent if side is TradeType.BUY else SellOrderCreatedEvent
    return event_class(
        timestamp=1.0,
        type=OrderType.LIMIT_MAKER if trading_pair == "SNXX-USDT" else OrderType.MARKET,
        trading_pair=trading_pair,
        amount=Decimal("100") if trading_pair == "SNXX-USDT" else Decimal("9.6"),
        price=Decimal("30") if trading_pair == "SNXX-USDT" else Decimal("250"),
        order_id=order_id,
        creation_timestamp=1.0,
        exchange_order_id=exchange_order_id,
        position=PositionAction.OPEN.value,
    )


@pytest.mark.asyncio
async def test_maker_preflight_and_submission_are_durable_before_the_native_connector_call():
    executor, connector, repository, timeline = _executor()

    await executor.control_task()

    assert connector.preflight_calls == [("SNXX-USDT", "SNDK-USDT")]
    assert executor.state is LeveragedEtfPairState.MAKER_SUBMITTING
    assert len(repository.created_snapshots) == 1
    assert len(connector.orders) == 1
    maker = connector.orders[0]
    assert maker == {
        "side": "SELL",
        "trading_pair": "SNXX-USDT",
        "amount": Decimal("100"),
        "order_type": OrderType.LIMIT_MAKER,
        "price": Decimal("30"),
        "client_order_id": executor.maker_client_order_id,
        "position_action": PositionAction.OPEN,
    }
    assert len(maker["client_order_id"]) <= 32
    prepared_index = next(
        index
        for index, item in enumerate(timeline)
        if item[:2] == ("journal", JournalEventType.PREPARED.value)
        and item[2].payload.identity.action.value == "ETF_MAKER"
    )
    connector_index = next(index for index, item in enumerate(timeline) if item[0] == "connector")
    assert prepared_index < connector_index
    assert [event.event_type for event in repository.events][-1] is JournalEventType.PREPARED

    executor.process_order_created_event(
        MarketEvent.SellOrderCreated.value,
        connector,
        _created_event(
            maker["client_order_id"],
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
            exchange_order_id="maker-exchange-1",
        ),
    )
    assert executor.state is LeveragedEtfPairState.MAKER_WORKING
    assert [event.event_type for event in repository.events][-2:] == [
        JournalEventType.ORDER_CREATED,
        JournalEventType.ACKNOWLEDGED,
    ]


@pytest.mark.asyncio
async def test_native_maker_submission_waits_for_authoritative_confirmation_and_reconciles_async_unknown():
    executor, connector, repository, _ = _executor()
    connector.async_submission_unknown_on_next_submit = True

    await executor.control_task()

    assert len(connector.orders) == 1
    assert [event.event_type for event in repository.events] == [
        JournalEventType.STATE_TRANSITION,
        JournalEventType.PREPARED,
    ]
    assert executor.state is LeveragedEtfPairState.MAKER_SUBMITTING

    await asyncio.sleep(0)
    await executor.control_task()

    assert executor.state is LeveragedEtfPairState.RECONCILING
    assert [event.event_type for event in repository.events][-1] is JournalEventType.SUBMIT_UNKNOWN
    assert JournalEventType.ACKNOWLEDGED not in [event.event_type for event in repository.events]
    assert len(connector.orders) == 1


@pytest.mark.asyncio
async def test_native_stock_submission_reconciles_async_unknown_before_another_hedge_submission():
    executor, connector, repository, _ = _executor()
    await executor.control_task()
    executor.process_order_created_event(
        MarketEvent.SellOrderCreated.value,
        connector,
        _created_event(
            executor.maker_client_order_id,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
            exchange_order_id="maker-exchange-1",
        ),
    )
    connector.async_submission_unknown_on_next_submit = True

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            executor.maker_client_order_id,
            "etf-native-stock-unknown-1",
            "10",
            timestamp=1.0,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
        ),
    )
    stock_orders = [order for order in connector.orders if order["trading_pair"] == "SNDK-USDT"]
    assert len(stock_orders) == 1
    stock_events = [
        event.event_type
        for event in repository.events
        if getattr(event.payload, "identity", None) is not None
        and event.payload.identity.action.value == "STOCK_HEDGE"
    ]
    assert stock_events == [JournalEventType.PREPARED, JournalEventType.HEDGE_REQUESTED]

    await asyncio.sleep(0)
    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            executor.maker_client_order_id,
            "etf-native-stock-unknown-2",
            "5",
            timestamp=2.0,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
        ),
    )

    assert executor.state is LeveragedEtfPairState.RECONCILING
    assert [order for order in connector.orders if order["trading_pair"] == "SNDK-USDT"] == stock_orders
    stock_events = [
        event.event_type
        for event in repository.events
        if getattr(event.payload, "identity", None) is not None
        and event.payload.identity.action.value == "STOCK_HEDGE"
    ]
    assert stock_events == [
        JournalEventType.PREPARED,
        JournalEventType.HEDGE_REQUESTED,
        JournalEventType.SUBMIT_UNKNOWN,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        LeveragedEtfPairOperation.REDUCE,
        LeveragedEtfPairOperation.CLOSE,
        LeveragedEtfPairOperation.EMERGENCY_FLATTEN,
    ],
)
async def test_entry_executor_rejects_non_entry_operations_without_creating_an_intent_or_submitting(
    operation: LeveragedEtfPairOperation,
):
    executor, connector, repository, _ = _executor(operation=operation)

    await executor.control_task()

    assert executor.state is LeveragedEtfPairState.RECOVERY_REQUIRED
    assert connector.preflight_calls == []
    assert connector.orders == []
    assert JournalEventType.PREPARED not in [event.event_type for event in repository.events]


@pytest.mark.asyncio
async def test_maker_below_minimum_notional_never_creates_a_prepared_intent_or_submits():
    executor, connector, repository, _ = _executor()
    connector.trading_rules["SNXX-USDT"].min_notional_size = Decimal("3000.01")

    await executor.control_task()

    assert executor.state is LeveragedEtfPairState.RECOVERY_REQUIRED
    assert connector.orders == []
    assert JournalEventType.PREPARED not in [event.event_type for event in repository.events]


@pytest.mark.asyncio
async def test_stock_partial_fill_below_minimum_notional_remains_dust_without_a_prepared_intent_or_submission():
    executor, connector, repository, _ = _executor()
    connector.trading_rules["SNDK-USDT"].min_notional_size = Decimal("2400.01")
    await executor.control_task()
    executor.process_order_created_event(
        MarketEvent.SellOrderCreated.value,
        connector,
        _created_event(
            executor.maker_client_order_id,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
            exchange_order_id="maker-exchange-1",
        ),
    )

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            executor.maker_client_order_id,
            "etf-notional-dust-1",
            "10",
            timestamp=1.0,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
        ),
    )

    assert executor.state is LeveragedEtfPairState.STOCK_HEDGE_PENDING
    assert executor.hedge_dust_quantity == Decimal("9.6")
    assert [order for order in connector.orders if order["trading_pair"] == "SNDK-USDT"] == []
    assert not [
        event
        for event in repository.events
        if getattr(event.payload, "identity", None) is not None
        and event.payload.identity.action.value == "STOCK_HEDGE"
    ]


@pytest.mark.asyncio
async def test_strict_preflight_failure_never_submits_an_order_and_reports_a_safe_non_terminal_state():
    executor, connector, repository, _ = _executor()
    connector.preflight_error = RuntimeError("account facts are stale")

    await executor.control_task()

    assert connector.orders == []
    assert executor.state is LeveragedEtfPairState.RECOVERY_REQUIRED
    assert [event.event_type for event in repository.events] == [
        JournalEventType.STATE_TRANSITION,
        JournalEventType.STATE_TRANSITION,
    ]
    await executor.control_task()
    assert connector.orders == []


@pytest.mark.asyncio
async def test_partial_etf_fills_are_deduplicated_and_submit_only_the_incremental_multiplier_aware_stock_hedge():
    executor, connector, repository, timeline = _executor()
    await executor.control_task()
    maker_order_id = executor.maker_client_order_id

    first_fill = _fill_event(
        maker_order_id,
        "etf-trade-1",
        "10",
        timestamp=20.0,
        trading_pair="SNXX-USDT",
        side=TradeType.SELL,
    )
    # A fill can be delivered before the created event and must still hedge immediately.
    executor.process_order_filled_event(MarketEvent.OrderFilled.value, connector, first_fill)

    stock_orders = [order for order in connector.orders if order["trading_pair"] == "SNDK-USDT"]
    assert [order["amount"] for order in stock_orders] == [Decimal("9.6")]
    assert all(order["side"] == "BUY" and order["order_type"] is OrderType.MARKET for order in stock_orders)
    assert executor.etf_filled_quantity == Decimal("10")
    assert executor.stock_hedge_target_quantity == Decimal("9.6")
    assert executor.stock_pending_quantity == Decimal("9.6")

    executor.process_order_filled_event(MarketEvent.OrderFilled.value, connector, first_fill)
    assert [order["amount"] for order in connector.orders if order["trading_pair"] == "SNDK-USDT"] == [Decimal("9.6")]

    first_stock_order_id = stock_orders[0]["client_order_id"]
    executor.process_order_created_event(
        MarketEvent.BuyOrderCreated.value,
        connector,
        _created_event(
            first_stock_order_id,
            trading_pair="SNDK-USDT",
            side=TradeType.BUY,
            exchange_order_id="stock-exchange-1",
        ),
    )

    # A distinct trade with an earlier timestamp is still a new fact; trade IDs, not arrival time, deduplicate.
    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            maker_order_id,
            "etf-trade-2",
            "5",
            timestamp=10.0,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
        ),
    )
    stock_orders = [order for order in connector.orders if order["trading_pair"] == "SNDK-USDT"]
    assert [order["amount"] for order in stock_orders] == [Decimal("9.6"), Decimal("4.8")]
    assert executor.etf_filled_quantity == Decimal("15")
    assert executor.stock_hedge_target_quantity == Decimal("14.4")
    assert executor.stock_pending_quantity == Decimal("14.4")

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            first_stock_order_id,
            "stock-trade-1",
            "4.8",
            timestamp=21.0,
            trading_pair="SNDK-USDT",
            side=TradeType.BUY,
        ),
    )
    assert executor.stock_filled_quantity == Decimal("4.8")
    assert executor.stock_pending_quantity == Decimal("9.6")
    assert [order["amount"] for order in connector.orders if order["trading_pair"] == "SNDK-USDT"] == [
        Decimal("9.6"),
        Decimal("4.8"),
    ]

    stock_prepared_index = next(
        index
        for index, item in enumerate(timeline)
        if item[:2] == ("journal", JournalEventType.PREPARED.value)
        and item[2].payload.identity.action.value == "STOCK_HEDGE"
    )
    first_stock_connector_index = next(
        index
        for index, item in enumerate(timeline)
        if item[:3] == ("connector", "BUY", stock_orders[0]["client_order_id"])
    )
    assert stock_prepared_index < first_stock_connector_index
    assert any(event.event_type is JournalEventType.FILL for event in repository.events)


@pytest.mark.asyncio
async def test_small_fills_accumulate_to_a_legal_quantized_stock_market_order_without_oversizing():
    executor, connector, _, _ = _executor(stock_step=Decimal("0.1"))
    await executor.control_task()
    maker_order_id = executor.maker_client_order_id

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            maker_order_id,
            "etf-dust-1",
            "0.06",
            timestamp=1.0,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
        ),
    )
    assert executor.hedge_dust_quantity == Decimal("0.0576")
    assert not [order for order in connector.orders if order["trading_pair"] == "SNDK-USDT"]

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            maker_order_id,
            "etf-dust-2",
            "0.05",
            timestamp=2.0,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
        ),
    )
    stock_orders = [order for order in connector.orders if order["trading_pair"] == "SNDK-USDT"]
    assert [order["amount"] for order in stock_orders] == [Decimal("0.1")]
    assert executor.hedge_dust_quantity == Decimal("0.0056")


@pytest.mark.asyncio
async def test_full_maker_fill_hedges_the_complete_multiplier_aware_stock_target_and_completes():
    executor, connector, _, _ = _executor()
    await executor.control_task()

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            executor.maker_client_order_id,
            "etf-full-fill",
            "100",
            timestamp=1.0,
            trading_pair="SNXX-USDT",
            side=TradeType.SELL,
        ),
    )
    stock_order = next(order for order in connector.orders if order["trading_pair"] == "SNDK-USDT")
    assert stock_order["amount"] == Decimal("96")
    assert executor.state is LeveragedEtfPairState.STOCK_HEDGE_PENDING

    executor.process_order_filled_event(
        MarketEvent.OrderFilled.value,
        connector,
        _fill_event(
            stock_order["client_order_id"],
            "stock-full-fill",
            "96",
            timestamp=2.0,
            trading_pair="SNDK-USDT",
            side=TradeType.BUY,
        ),
    )
    assert executor.stock_filled_quantity == Decimal("96")
    assert executor.stock_pending_quantity == Decimal("0")
    assert executor.state is LeveragedEtfPairState.COMPLETED


@pytest.mark.asyncio
async def test_maker_timeout_and_post_only_rejection_are_recorded_without_a_duplicate_submission():
    executor, connector, repository, _ = _executor()
    connector.next_submission_error = TimeoutError("submission timed out")

    await executor.control_task()

    assert connector.orders == []
    assert executor.state is LeveragedEtfPairState.RECONCILING
    assert [event.event_type for event in repository.events] == [
        JournalEventType.STATE_TRANSITION,
        JournalEventType.PREPARED,
        JournalEventType.SUBMIT_UNKNOWN,
    ]
    await executor.control_task()
    assert len([event for event in repository.events if event.event_type is JournalEventType.PREPARED]) == 1

    executor, connector, repository, _ = _executor()
    await executor.control_task()
    executor.process_order_failed_event(
        MarketEvent.OrderFailure.value,
        connector,
        MarketOrderFailureEvent(
            timestamp=3.0,
            order_id=executor.maker_client_order_id,
            order_type=OrderType.LIMIT_MAKER,
        ),
    )
    # The local connector had already allocated a stable client ID, so F004
    # records a terminal reconciliation fact and enters safe reconciliation;
    # it must not manufacture a replacement maker order.
    assert executor.state is LeveragedEtfPairState.RECONCILING
    assert [event.event_type for event in repository.events][-2:] == [
        JournalEventType.RECONCILIATION,
        JournalEventType.STATE_TRANSITION,
    ]
    await executor.control_task()
    assert len(connector.orders) == 1


def test_event_registration_uses_the_native_executor_base_listener_lifecycle():
    executor, connector, _, _ = _executor()

    executor.register_events()

    assert set(connector.listeners) == {
        MarketEvent.OrderCancelled,
        MarketEvent.BuyOrderCreated,
        MarketEvent.SellOrderCreated,
        MarketEvent.OrderFilled,
        MarketEvent.BuyOrderCompleted,
        MarketEvent.SellOrderCompleted,
        MarketEvent.OrderFailure,
    }
    executor.unregister_events()
    assert all(not listeners for listeners in connector.listeners.values())


@pytest.mark.asyncio
async def test_entry_events_reduce_against_the_real_f004_journal_repository(tmp_path):
    manager = SQLConnectionManager(
        ClientConfigAdapter(ClientConfigMap()),
        SQLConnectionType.TRADE_FILLS,
        db_path=str(tmp_path / "pair-entry.sqlite"),
    )
    try:
        timeline = []
        connector = FakeConnector(timeline)
        strategy = SimpleNamespace(connectors={"binance_perpetual": connector}, current_timestamp=1721224862.0)
        executor = LeveragedEtfPairExecutor(
            strategy=strategy,
            config=_config(),
            journal_repository=LeveragedEtfJournalRepository(manager),
        )

        await executor.control_task()
        executor.process_order_filled_event(
            MarketEvent.OrderFilled.value,
            connector,
            _fill_event(
                executor.maker_client_order_id,
                "f004-etf-fill-1",
                "10",
                timestamp=1.0,
                trading_pair="SNXX-USDT",
                side=TradeType.SELL,
            ),
        )
        stock_order = next(order for order in connector.orders if order["trading_pair"] == "SNDK-USDT")
        executor.process_order_filled_event(
            MarketEvent.OrderFilled.value,
            connector,
            _fill_event(
                stock_order["client_order_id"],
                "f004-stock-fill-1",
                "9.6",
                timestamp=2.0,
                trading_pair="SNDK-USDT",
                side=TradeType.BUY,
            ),
        )

        repository = LeveragedEtfJournalRepository(manager)
        snapshot = repository.replay(executor.config.id)
        assert snapshot.state is LeveragedEtfPairState.MAKER_WORKING
        assert executor.state is LeveragedEtfPairState.MAKER_WORKING
        assert snapshot.etf_filled_quantity == Decimal("10")
        assert snapshot.stock_submitted_quantity == Decimal("9.6")
        assert snapshot.stock_filled_quantity == Decimal("9.6")
        assert [event.event.event_type for event in repository.events(executor.config.id)] == [
            JournalEventType.STATE_TRANSITION,
            JournalEventType.PREPARED,
            JournalEventType.ACKNOWLEDGED,
            JournalEventType.FILL,
            JournalEventType.PREPARED,
            JournalEventType.HEDGE_REQUESTED,
            JournalEventType.ACKNOWLEDGED,
            JournalEventType.FILL,
            JournalEventType.HEDGE_CONFIRMED,
        ]
    finally:
        manager.engine.dispose()
