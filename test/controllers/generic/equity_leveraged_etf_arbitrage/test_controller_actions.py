import asyncio
from dataclasses import dataclass, replace
from decimal import Decimal
from types import SimpleNamespace
from typing import Iterable

import pytest

from controllers.generic.equity_leveraged_etf_arbitrage.controller import (
    AnchorFacts,
    ControllerEpoch,
    EquityLeveragedEtfArbitrageController,
    PairEpochFacts,
    ReservationConflict,
)
from controllers.generic.equity_leveraged_etf_arbitrage.reservations import (
    F004ReservationStore,
)
from controllers.generic.equity_leveraged_etf_arbitrage.nav import (
    AnchorRuntimeStatus,
    NavStage,
    SessionStageDecision,
)
from hummingbot.model.leveraged_etf_repository import LeveragedEtfJournalRepository
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import (
    LeveragedEtfPairExecutorConfig,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.allocator import (
    AccountRiskSnapshot,
    AllocationTier,
    BookLevel,
    FrozenLeg,
    FrozenPair,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import StrategyConfig
from hummingbot.strategy_v2.leveraged_etf_arbitrage.domain import ArbitrageDirection
from hummingbot.strategy_v2.leveraged_etf_arbitrage.risk import (
    LeverageBracket,
    LeverageSchedule,
)
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executor_actions import (
    CreateExecutorAction,
    StopExecutorAction,
    StoreExecutorAction,
)
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


D = Decimal
_NOW = 1780000000.0


@dataclass(frozen=True)
class _ConfiguredPair:
    id: str
    stock_trading_pair: str
    etf_trading_pair: str
    etf_daily_multiplier: Decimal = D("2")
    enabled: bool = True


@dataclass(frozen=True)
class _ControllerConfig:
    id: str
    connector_name: str
    strategy: StrategyConfig
    risk: object
    pairs: tuple[_ConfiguredPair, ...]
    controller_name: str = "equity_leveraged_etf_arbitrage.controller"


@dataclass(frozen=True)
class _ActiveReservation:
    executor_id: str
    connector_name: str
    trading_pair: str
    leg: str = "ETF"


class _ReservationStore:
    def __init__(
        self, active: Iterable[_ActiveReservation] = (), *, reject: bool = False
    ):
        self.active_records = list(active)
        self.reject = reject
        self.reserve_calls: list[LeveragedEtfPairExecutorConfig] = []
        self.released_executor_ids: list[str] = []

    def active(self):
        return tuple(self.active_records)

    def reserve(self, executor_config: LeveragedEtfPairExecutorConfig):
        if self.reject:
            raise ReservationConflict("duplicate reservation")
        for current in self.active_records:
            if current.executor_id != executor_config.id and (
                current.connector_name,
                current.trading_pair,
            ) in {
                (executor_config.etf_connector_name, executor_config.etf_trading_pair),
                (
                    executor_config.stock_connector_name,
                    executor_config.stock_trading_pair,
                ),
            }:
                raise ReservationConflict("symbol is already reserved")
        self.reserve_calls.append(executor_config)
        self.active_records.extend(
            (
                _ActiveReservation(
                    executor_id=executor_config.id,
                    connector_name=executor_config.etf_connector_name,
                    trading_pair=executor_config.etf_trading_pair,
                    leg="ETF",
                ),
                _ActiveReservation(
                    executor_id=executor_config.id,
                    connector_name=executor_config.stock_connector_name,
                    trading_pair=executor_config.stock_trading_pair,
                    leg="STOCK",
                ),
            )
        )

    def release(self, executor_id: str, released_at_utc: str):
        self.released_executor_ids.append(executor_id)
        self.active_records = [
            record
            for record in self.active_records
            if record.executor_id != executor_id
        ]


class _F004Repository(LeveragedEtfJournalRepository):
    """Narrow in-memory F004 boundary double that preserves the real type."""

    def __init__(self, events: list[str]):
        self.events = events
        self.records = []

    def active_reservations(self, executor_id=None):
        records = self.records
        if executor_id is not None:
            records = [
                record for record in records if record.executor_id == executor_id
            ]
        return tuple(records)

    def reserve(self, reservation):
        self.events.append(f"reserve:{reservation.leg}")
        self.records.append(reservation)
        return reservation

    def release_reservation(self, reservation_id: str, released_at_utc: str):
        self.events.append(f"release:{reservation_id}")
        return next(
            record for record in self.records if record.reservation_id == reservation_id
        )


class _MarketDataProvider:
    ready = True

    def time(self) -> float:
        return _NOW


class _Connector:
    def __init__(self, preflights, *, leverage_result=None):
        self.preflights = list(preflights)
        self.leverage_result = leverage_result
        self.preflight_calls: list[dict] = []
        self.leverage_calls: list[tuple[str, int]] = []

    async def strict_account_preflight(self, **kwargs):
        self.preflight_calls.append(kwargs)
        value = self.preflights[
            min(len(self.preflight_calls) - 1, len(self.preflights) - 1)
        ]
        if isinstance(value, Exception):
            raise value
        return value

    async def set_leverage_with_result(self, trading_pair: str, leverage: int):
        self.leverage_calls.append((trading_pair, leverage))
        if self.leverage_result is not None:
            return self.leverage_result(trading_pair, leverage)
        return SimpleNamespace(
            symbol=trading_pair, leverage=leverage, max_notional_value=D("1000000")
        )


class _EpochSource:
    def __init__(self, epochs):
        self.epochs = list(epochs)
        self.calls = []

    async def build_epoch(self, preflight, configured_pairs, active_reservations):
        self.calls.append(
            (preflight, tuple(configured_pairs), tuple(active_reservations))
        )
        return self.epochs[min(len(self.calls) - 1, len(self.epochs) - 1)]


def _strategy() -> StrategyConfig:
    return StrategyConfig(
        strategy_id="equity_leveraged_etf_arbitrage",
        max_total_notional_ratio="32",
        controller_interval_ms=1000,
        executor_safety_interval_ms=250,
        entry_confirmations=3,
        entry_confirmation_interval_ms=1000,
        divergence_cancel_bp="3",
        divergence_confirmations=3,
        market_data_max_age_ms=1500,
        account_data_max_age_ms=5000,
        maker_fee_bp="0",
        taker_fee_bp="4",
        maker_slippage_bp_per_fill="2",
        include_funding_cost=False,
        hedge_submit_timeout_ms=1000,
        hedge_reconcile_timeout_ms=5000,
        unhedged_response_deadline_ms=20000,
        hedge_phase_deadline_ms=10000,
        hedge_max_attempts=3,
        hedge_retry_backoff_ms=(100, 250, 500),
        rollback_submit_timeout_ms=1000,
        rollback_reconcile_timeout_ms=5000,
        rollback_phase_deadline_ms=10000,
        rollback_max_attempts=3,
        rollback_retry_backoff_ms=(100, 250, 500),
        order_eventual_consistency_grace_ms=2000,
    )


def _account() -> AccountRiskSnapshot:
    return AccountRiskSnapshot(
        total_margin_balance=D("1000"),
        total_initial_margin=D("0"),
        total_maint_margin=D("0"),
        available_balance=D("1000"),
        normal_initial_margin_budget_ratio=D("0.8"),
        p99_initial_margin_budget_ratio=D("0.85"),
        stress_extra_bp=D("100"),
        stress_equity_to_maintenance_margin_multiple=D("2"),
        reconciliation_tolerance=D("0.01"),
    )


def _schedule(symbol: str) -> LeverageSchedule:
    return LeverageSchedule(
        symbol=symbol,
        brackets=(
            LeverageBracket(
                bracket=1,
                initial_leverage=20,
                notional_floor=D("0"),
                notional_cap=D("1000000"),
                maint_margin_ratio=D("0.005"),
                cum=D("0"),
            ),
        ),
    )


def _configured_pair(number: int) -> _ConfiguredPair:
    return _ConfiguredPair(
        id=f"pair{number}",
        etf_trading_pair=f"ETF{number}-USDT",
        stock_trading_pair=f"STK{number}-USDT",
    )


def _frozen_pair(
    configured: _ConfiguredPair,
    *,
    target_ratio: Decimal = D("1"),
    direction: ArbitrageDirection = ArbitrageDirection.SHORT_ETF_LONG_STOCK,
    current_etf: Decimal = D("0"),
    current_stock: Decimal = D("0"),
) -> FrozenPair:
    etf = FrozenLeg(
        symbol=configured.etf_trading_pair,
        mark_price=D("60"),
        contract_multiplier=D("1"),
        quantity_step=D("1"),
        min_quantity=D("1"),
        min_notional=D("1"),
        leverage_schedule=_schedule(configured.etf_trading_pair),
        current_quantity=current_etf,
    )
    stock = FrozenLeg(
        symbol=configured.stock_trading_pair,
        mark_price=D("100"),
        contract_multiplier=D("1"),
        quantity_step=D("1"),
        min_quantity=D("1"),
        min_notional=D("1"),
        leverage_schedule=_schedule(configured.stock_trading_pair),
        current_quantity=current_stock,
    )
    return FrozenPair(
        pair_id=configured.id,
        direction=direction,
        stock_anchor=D("100"),
        etf_anchor=D("50"),
        etf_daily_multiplier=D("2"),
        requested_target_ratio=target_ratio,
        tiers=(AllocationTier(minimum_net_bp=D("0"), target_ratio=target_ratio),),
        p99_bp=D("0"),
        current_net_bp=D("1000"),
        max_stock_taker_impact_bp=D("1000"),
        etf_best_bid=D("59"),
        etf_best_ask=D("60"),
        stock_bids=(BookLevel(price=D("100"), quantity=D("100000")),),
        stock_asks=(BookLevel(price=D("100"), quantity=D("100000")),),
        etf=etf,
        stock=stock,
    )


def _anchor(pair: FrozenPair) -> AnchorFacts:
    return AnchorFacts(
        nav_cycle_id="xnys-2026-05-28",
        s0=pair.stock_anchor,
        l0=pair.etf_anchor,
        h=D("1"),
        raw_bp=D("1000"),
        net_bp=pair.current_net_bp,
        valid=True,
    )


def _entry_allowed_nav_decision(pair_id: str, cycle_id: str) -> SessionStageDecision:
    return SessionStageDecision(
        pair_id=pair_id,
        cycle_id=cycle_id,
        stage=NavStage.NORMAL,
        operational_status=AnchorRuntimeStatus.AVAILABLE,
        entry_allowed=True,
        pair_paused=False,
        emergency_market_requested=False,
        intents=(),
    )


def _epoch(
    *pairs: FrozenPair,
    account: AccountRiskSnapshot | None = None,
    stale_account: bool = False,
) -> ControllerEpoch:
    return ControllerEpoch(
        account=account or _account(),
        account_data_fresh=not stale_account,
        pairs=tuple(
            PairEpochFacts(
                pair_id=pair.pair_id,
                frozen_pair=pair,
                anchor=(anchor := _anchor(pair)),
                market_data_fresh=True,
                bracket_data_fresh=True,
                current_etf_leverage=1,
                current_stock_leverage=1,
                nav_decision=_entry_allowed_nav_decision(
                    pair.pair_id, anchor.nav_cycle_id
                ),
            )
            for pair in pairs
        ),
    )


def _preflight(*, data_time: float = _NOW, account: AccountRiskSnapshot | None = None):
    return SimpleNamespace(account=account or _account(), data_time=data_time)


def _controller(configured_pairs, epochs, *, connector=None, reservations=None):
    config = _ControllerConfig(
        id="portfolio-controller",
        connector_name="binance_perpetual",
        strategy=_strategy(),
        risk=SimpleNamespace(),
        pairs=tuple(configured_pairs),
    )
    connector = connector or _Connector((_preflight(), _preflight()))
    controller = EquityLeveragedEtfArbitrageController(
        config=config,
        market_data_provider=_MarketDataProvider(),
        actions_queue=asyncio.Queue(),
        connector=connector,
        epoch_source=_EpochSource(epochs),
        reservations=reservations or _ReservationStore(),
    )
    return controller, connector


def _actions(controller):
    async def run():
        await controller.update_processed_data()
        return controller.determine_executor_actions()

    return asyncio.run(run())


def _executor_info(
    config, status: RunnableStatus, *, custom_state: str | None = None
) -> ExecutorInfo:
    custom_info = {} if custom_state is None else {"state": custom_state}
    return ExecutorInfo(
        id=config.id,
        timestamp=_NOW,
        type=config.type,
        status=status,
        config=config,
        net_pnl_pct=D("0"),
        net_pnl_quote=D("0"),
        cum_fees_quote=D("0"),
        filled_amount_quote=D("0"),
        is_active=status is not RunnableStatus.TERMINATED,
        is_trading=False,
        custom_info=custom_info,
        controller_id=config.controller_id,
    )


def _created_config(configured: _ConfiguredPair) -> LeveragedEtfPairExecutorConfig:
    controller, _ = _controller((configured,), (_epoch(_frozen_pair(configured)),))
    actions = _actions(controller)
    return next(
        action.executor_config
        for action in actions
        if isinstance(action, CreateExecutorAction)
    )


def test_registers_one_to_four_pairs_and_isolates_pair_local_market_failure():
    configured = tuple(_configured_pair(index) for index in range(1, 5))
    frozen = tuple(_frozen_pair(pair) for pair in configured)
    epoch = _epoch(*frozen)
    unavailable = replace(
        epoch.pairs[2],
        frozen_pair=None,
        anchor=None,
        market_data_fresh=False,
        failure_reason="stock order book unavailable",
    )
    controller, connector = _controller(
        configured,
        (replace(epoch, pairs=epoch.pairs[:2] + (unavailable,) + epoch.pairs[3:]),),
    )

    actions = _actions(controller)

    creates = [action for action in actions if isinstance(action, CreateExecutorAction)]
    assert {action.executor_config.pair_id for action in creates} == {
        "pair1",
        "pair2",
        "pair4",
    }
    assert connector.preflight_calls[0]["trading_pairs"] == (
        "ETF1-USDT",
        "ETF2-USDT",
        "ETF3-USDT",
        "ETF4-USDT",
        "STK1-USDT",
        "STK2-USDT",
        "STK3-USDT",
        "STK4-USDT",
    )


@pytest.mark.parametrize("stale_account, stale_book", ((True, False), (False, True)))
def test_stale_account_or_book_fails_closed(stale_account: bool, stale_book: bool):
    configured = _configured_pair(1)
    epoch = _epoch(_frozen_pair(configured), stale_account=stale_account)
    if stale_book:
        epoch = replace(
            epoch, pairs=(replace(epoch.pairs[0], market_data_fresh=False),)
        )
    controller, connector = _controller((configured,), (epoch,))

    assert _actions(controller) == []
    assert connector.leverage_calls == []


@pytest.mark.parametrize("data_time", (float("nan"), _NOW - 6))
def test_invalid_or_stale_preflight_timestamp_fails_closed(data_time: float):
    configured = _configured_pair(1)
    connector = _Connector((_preflight(data_time=data_time),))
    controller, _ = _controller(
        (configured,), (_epoch(_frozen_pair(configured)),), connector=connector
    )

    assert _actions(controller) == []


def test_unknown_related_position_preflight_error_fails_closed():
    configured = _configured_pair(1)
    connector = _Connector((RuntimeError("unknown related position"),))
    controller, _ = _controller(
        (configured,), (_epoch(_frozen_pair(configured)),), connector=connector
    )

    assert _actions(controller) == []


def test_restart_with_durable_symbol_reservation_blocks_duplicate_executor():
    configured = _configured_pair(1)
    reservations = _ReservationStore(
        (
            _ActiveReservation(
                "prior-executor", "binance_perpetual", configured.etf_trading_pair
            ),
        )
    )
    controller, connector = _controller(
        (configured,),
        (_epoch(_frozen_pair(configured)),),
        reservations=reservations,
    )

    assert _actions(controller) == []
    assert connector.preflight_calls[0]["known_position_trading_pairs"] == (
        configured.etf_trading_pair,
    )
    assert reservations.reserve_calls == []


def test_f004_adapter_bootstraps_then_locks_both_symbols_and_rejects_duplicate():
    configured = _configured_pair(1)
    config = _created_config(configured)
    events: list[str] = []
    repository = _F004Repository(events)
    store = F004ReservationStore(
        repository, lambda created: events.append(f"ensure:{created.id}")
    )

    persisted = store.reserve(config)

    assert [reservation.leg for reservation in persisted] == ["ETF", "STOCK"]
    assert events == [f"ensure:{config.id}", "reserve:ETF", "reserve:STOCK"]
    duplicate = config.model_copy(update={"id": "portfolio-controller:pair1:duplicate"})
    with pytest.raises(ReservationConflict, match="already reserved"):
        store.reserve(duplicate)
    assert events == [f"ensure:{config.id}", "reserve:ETF", "reserve:STOCK"]


@pytest.mark.parametrize(
    ("status", "custom_state"),
    (
        (RunnableStatus.SHUTTING_DOWN, None),
        (RunnableStatus.TERMINATED, "RECOVERY_REQUIRED"),
    ),
)
def test_every_non_done_executor_blocks_pair_replacement(status, custom_state):
    configured = _configured_pair(1)
    config = _created_config(configured)
    controller, _ = _controller((configured,), (_epoch(_frozen_pair(configured)),))
    controller.executors_info = [
        _executor_info(config, status, custom_state=custom_state)
    ]

    actions = _actions(controller)

    assert not any(isinstance(action, CreateExecutorAction) for action in actions)
    assert not any(isinstance(action, StoreExecutorAction) for action in actions)


@pytest.mark.parametrize(
    ("direction", "target_ratio"),
    (
        (ArbitrageDirection.SHORT_ETF_LONG_STOCK, D("0")),
        (ArbitrageDirection.LONG_ETF_SHORT_STOCK, D("1")),
    ),
)
def test_target_decrease_or_direction_flip_stops_existing_executor(
    direction, target_ratio
):
    configured = _configured_pair(1)
    config = _created_config(configured)
    current_pair = _frozen_pair(
        configured,
        direction=direction,
        target_ratio=target_ratio,
        current_etf=-config.etf_target_quantity,
        current_stock=config.stock_target_quantity,
    )
    controller, _ = _controller((configured,), (_epoch(current_pair),))
    controller.executors_info = [_executor_info(config, RunnableStatus.RUNNING)]

    actions = _actions(controller)

    assert actions == [
        StopExecutorAction(
            controller_id="portfolio-controller",
            executor_id=config.id,
            keep_position=True,
        )
    ]


def test_direction_flip_creates_only_a_close_after_the_old_executor_is_done():
    configured = _configured_pair(1)
    config = _created_config(configured)
    current_pair = _frozen_pair(
        configured,
        direction=ArbitrageDirection.LONG_ETF_SHORT_STOCK,
        target_ratio=D("1"),
        current_etf=-config.etf_target_quantity,
        current_stock=config.stock_target_quantity,
    )
    reservations = _ReservationStore()
    controller, _ = _controller(
        (configured,), (_epoch(current_pair),), reservations=reservations
    )

    actions = _actions(controller)

    assert len(actions) == 1
    assert isinstance(actions[0], CreateExecutorAction)
    assert actions[0].executor_config.operation.value == "CLOSE"
    assert actions[0].executor_config.direction.value == "SHORT_ETF_LONG_STOCK"
    assert actions[0].executor_config.target_gross_notional == D("0")
    assert reservations.reserve_calls == []


def test_leverage_response_mismatch_never_reserves_or_creates():
    configured = _configured_pair(1)
    connector = _Connector(
        (_preflight(),),
        leverage_result=lambda pair, leverage: SimpleNamespace(
            symbol=pair,
            leverage=leverage + 1,
            max_notional_value=D("1000000"),
        ),
    )
    reservations = _ReservationStore()
    controller, _ = _controller(
        (configured,),
        (_epoch(_frozen_pair(configured)),),
        connector=connector,
        reservations=reservations,
    )

    assert _actions(controller) == []
    assert reservations.reserve_calls == []


def test_refreshed_snapshot_invalidation_discards_old_plan_before_create():
    configured = _configured_pair(1)
    initial = _epoch(_frozen_pair(configured, target_ratio=D("1")))
    invalidated = _epoch(_frozen_pair(configured, target_ratio=D("0")))
    controller, connector = _controller((configured,), (initial, invalidated))

    assert _actions(controller) == []
    assert connector.preflight_calls and len(connector.preflight_calls) == 2


def test_terminal_executor_is_stored_and_its_reservation_is_released():
    configured = _configured_pair(1)
    config = _created_config(configured)
    reservations = _ReservationStore(
        (
            _ActiveReservation(
                config.id, config.etf_connector_name, config.etf_trading_pair
            ),
            _ActiveReservation(
                config.id,
                config.stock_connector_name,
                config.stock_trading_pair,
                "STOCK",
            ),
        )
    )
    unavailable = PairEpochFacts(
        pair_id=configured.id,
        frozen_pair=None,
        anchor=None,
        market_data_fresh=False,
        bracket_data_fresh=False,
        failure_reason="book unavailable",
    )
    controller, connector = _controller(
        (configured,),
        (
            ControllerEpoch(
                account=_account(), account_data_fresh=True, pairs=(unavailable,)
            ),
        ),
        reservations=reservations,
    )
    controller.executors_info = [_executor_info(config, RunnableStatus.TERMINATED)]

    actions = _actions(controller)

    assert actions == [
        StoreExecutorAction(controller_id="portfolio-controller", executor_id=config.id)
    ]
    assert reservations.released_executor_ids == [config.id]
    assert connector.preflight_calls[0]["known_position_trading_pairs"] == (
        config.etf_trading_pair,
        config.stock_trading_pair,
    )


def test_removed_pair_requests_stop_without_reconstructing_an_active_executor():
    removed = _configured_pair(2)
    retained = _configured_pair(1)
    removed_config = _created_config(removed)
    unavailable = PairEpochFacts(
        pair_id=retained.id,
        frozen_pair=None,
        anchor=None,
        market_data_fresh=False,
        bracket_data_fresh=False,
        failure_reason="intentionally no new plan",
    )
    controller, _ = _controller(
        (retained,),
        (
            ControllerEpoch(
                account=_account(), account_data_fresh=True, pairs=(unavailable,)
            ),
        ),
    )
    controller.executors_info = [_executor_info(removed_config, RunnableStatus.RUNNING)]

    actions = _actions(controller)

    assert actions == [
        StopExecutorAction(
            controller_id="portfolio-controller",
            executor_id=removed_config.id,
            keep_position=True,
        )
    ]


def test_standard_action_consumer_receives_only_create_actions_after_preflight():
    configured = _configured_pair(1)
    controller, _ = _controller((configured,), (_epoch(_frozen_pair(configured)),))

    async def consume():
        await controller.update_processed_data()
        actions = controller.determine_executor_actions()
        await controller.send_actions(actions)
        return await controller.actions_queue.get()

    actions = asyncio.run(consume())

    assert len(actions) == 1
    assert isinstance(actions[0], CreateExecutorAction)
