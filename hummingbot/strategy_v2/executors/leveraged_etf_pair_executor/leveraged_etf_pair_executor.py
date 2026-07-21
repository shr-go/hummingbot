"""Native Strategy V2 executor for a leveraged ETF / stock perpetual pair.

The executor deliberately owns only the entry path.  It persists every order
intent before calling a connector, keeps the maker leg singular, and converts
each accepted ETF fill into the smallest legal incremental stock hedge.  Exit,
cancel, recovery, and restart orchestration are separate feature work.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Dict, Optional, Tuple, Union

from hummingbot.connector.markets_recorder import MarketsRecorder
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.event.events import (
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderFilledEvent,
    SellOrderCreatedEvent,
)
from hummingbot.model.leveraged_etf_repository import (
    AcknowledgedJournalPayloadV1,
    FillJournalPayloadV1,
    HedgeJournalPayloadV1,
    JournalEventType,
    JournalEventV1,
    JournalSideEffect,
    OrderCreatedJournalPayloadV1,
    PreparedJournalPayloadV1,
    ReconciliationJournalPayloadV1,
    ReconciliationOutcome,
    RejectedJournalPayloadV1,
    SideEffectIdentityV1,
    StateTransitionJournalPayloadV1,
    SubmitUnknownJournalPayloadV1,
)
from hummingbot.strategy.strategy_v2_base import StrategyV2Base
from hummingbot.strategy_v2.executors.executor_base import ExecutorBase
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import (
    LeveragedEtfPairDirection,
    LeveragedEtfPairExecutorConfig,
    LeveragedEtfPairExecutorCustomInfoV1,
    LeveragedEtfPairExecutorReportV1,
    LeveragedEtfPairExecutorSnapshotV1,
    LeveragedEtfPairExecutorStateV1,
    LeveragedEtfPairState,
    OrderReferenceV1,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.domain import ArbitrageDirection
from hummingbot.strategy_v2.leveraged_etf_arbitrage.math import calculate_leg_quantities


_ZERO = Decimal("0")
_NAN = Decimal("NaN")


def _canonical_decimal(value: Decimal) -> str:
    """Return the canonical decimal spelling required by F004 identities."""
    if value == 0:
        return "0"
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


@dataclass
class _OrderIntent:
    identity: SideEffectIdentityV1
    filled_quantity: Decimal = _ZERO
    exchange_order_id: Optional[str] = None

    @property
    def remaining_quantity(self) -> Decimal:
        return max(_ZERO, self.identity.order_quantity - self.filled_quantity)


class LeveragedEtfPairExecutor(ExecutorBase):
    """Submit one ETF maker order and hedge each accepted ETF fill immediately.

    Native connector order methods are used so the normal Hummingbot order
    tracker and event bus retain ownership of exchange I/O.  Preallocated client
    IDs make those calls replay-safe from the durable side-effect identity.
    """

    def __init__(
        self,
        strategy: StrategyV2Base,
        config: LeveragedEtfPairExecutorConfig,
        update_interval: float = 1.0,
        max_retries: int = 10,
        journal_repository=None,
    ):
        connector_names = tuple(dict.fromkeys(config.connector_names))
        super().__init__(
            strategy=strategy,
            connectors=list(connector_names),
            config=config,
            update_interval=update_interval,
            max_retries=max_retries,
        )
        self.config: LeveragedEtfPairExecutorConfig = config
        self._journal_repository = (
            journal_repository
            if journal_repository is not None
            else MarketsRecorder.get_instance().leveraged_etf_journal_repository
        )
        self._state = LeveragedEtfPairState.CREATED
        self._journal_initialized = False
        self._journal_event_count = 0
        self._event_clock: datetime = config.created_at_utc
        self._preflight_completed = False
        self._submission_halted = False
        self._maker_submission_started = False

        self._contract_multipliers: Dict[Tuple[str, str], Decimal] = {}
        self._maker_intent: Optional[_OrderIntent] = None
        self._stock_intents: Dict[str, _OrderIntent] = {}
        self._seen_trades: set[Tuple[str, str]] = set()

        self._etf_filled_quantity = _ZERO
        self._stock_filled_quantity = _ZERO
        self._stock_hedge_target_quantity = _ZERO
        self._hedge_dust_quantity = _ZERO
        self._close_reason: Optional[str] = None

    @property
    def state(self) -> LeveragedEtfPairState:
        return self._state

    @property
    def maker_client_order_id(self) -> Optional[str]:
        return None if self._maker_intent is None else self._maker_intent.identity.client_order_id

    @property
    def etf_filled_quantity(self) -> Decimal:
        return self._etf_filled_quantity

    @property
    def stock_filled_quantity(self) -> Decimal:
        return self._stock_filled_quantity

    @property
    def stock_hedge_target_quantity(self) -> Decimal:
        return self._stock_hedge_target_quantity

    @property
    def stock_pending_quantity(self) -> Decimal:
        return sum((intent.remaining_quantity for intent in self._stock_intents.values()), _ZERO)

    @property
    def hedge_dust_quantity(self) -> Decimal:
        return self._hedge_dust_quantity

    @property
    def filled_amount_quote(self) -> Decimal:
        return self._etf_filled_quantity * self.config.l0 + self._stock_filled_quantity * self.config.s0

    async def validate_sufficient_balance(self):
        """F002 strict account preflight is the authoritative entry precondition."""
        return None

    def get_net_pnl_quote(self) -> Decimal:
        return _ZERO

    def get_net_pnl_pct(self) -> Decimal:
        return _ZERO

    def get_cum_fees_quote(self) -> Decimal:
        return _ZERO

    def early_stop(self, keep_position: bool = False):
        """Do not issue unscoped cancels; preserve the journal for recovery work."""
        self._submission_halted = True
        self._close_reason = "early stop requested; cancellation is outside the entry executor scope"

    async def control_task(self):
        if self._submission_halted or self._maker_submission_started:
            return
        self._ensure_journal_snapshot()
        try:
            await self._run_strict_preflight()
        except Exception:
            self._submission_halted = True
            self._close_reason = "strict account preflight failed"
            self._transition_to_recovery_required("strict account preflight failed")
            return
        try:
            self._submit_maker_order()
        except Exception:
            self._submission_halted = True
            self._close_reason = "maker setup or submission failed"
            self._transition_to_recovery_required("maker setup or submission failed")

    def process_order_created_event(
        self,
        _: int,
        market,
        event: Union[BuyOrderCreatedEvent, SellOrderCreatedEvent],
    ):
        intent = self._intent_for_client_order_id(event.order_id)
        if intent is None:
            return
        exchange_order_id = event.exchange_order_id
        if not exchange_order_id:
            self._transition_to_recovery_required("order created event omitted exchange order id")
            return
        if intent.exchange_order_id == exchange_order_id:
            return
        if intent.exchange_order_id is not None and intent.exchange_order_id != exchange_order_id:
            self._transition_to_recovery_required("client order id mapped to conflicting exchange order ids")
            return
        self._append_identity_event(
            event_type=JournalEventType.ORDER_CREATED,
            payload=OrderCreatedJournalPayloadV1(identity=intent.identity, exchange_order_id=exchange_order_id),
            identity=intent.identity,
            exchange_order_id=exchange_order_id,
        )
        intent.exchange_order_id = exchange_order_id

    def process_order_filled_event(self, _: int, market, event: OrderFilledEvent):
        intent = self._intent_for_client_order_id(event.order_id)
        if intent is None:
            return
        leg = "ETF" if intent.identity.leg == "ETF" else "STOCK"
        trade_id = str(event.exchange_trade_id or "")
        exchange_order_id = str(event.exchange_order_id or intent.exchange_order_id or "")
        quantity = Decimal(event.amount)
        if not trade_id or not exchange_order_id or quantity <= _ZERO:
            self._transition_to_recovery_required("fill event lacked a positive exchange trade/order identity")
            return
        trade_key = (leg, trade_id)
        if trade_key in self._seen_trades:
            return
        if intent.exchange_order_id is not None and intent.exchange_order_id != exchange_order_id:
            self._transition_to_recovery_required("fill event exchange order id disagreed with order identity")
            return
        if quantity > intent.remaining_quantity:
            self._transition_to_recovery_required("fill exceeded prepared order quantity")
            return

        order_cumulative = intent.filled_quantity + quantity
        leg_cumulative = (
            self._etf_filled_quantity + quantity
            if leg == "ETF"
            else self._stock_filled_quantity + quantity
        )
        outcome = "FILLED" if order_cumulative == intent.identity.order_quantity else "PARTIAL"
        self._append_identity_event(
            event_type=JournalEventType.FILL,
            payload=FillJournalPayloadV1(
                identity=intent.identity,
                exchange_order_id=exchange_order_id,
                exchange_trade_id=trade_id,
                price=Decimal(event.price),
                fill_quantity=quantity,
                order_cumulative_filled_quantity=order_cumulative,
                leg_cumulative_filled_quantity=leg_cumulative,
                outcome=outcome,
            ),
            identity=intent.identity,
            exchange_order_id=exchange_order_id,
            exchange_trade_id=trade_id,
        )
        intent.exchange_order_id = exchange_order_id
        intent.filled_quantity = order_cumulative
        self._seen_trades.add(trade_key)

        if leg == "ETF":
            self._etf_filled_quantity = leg_cumulative
            self._state = LeveragedEtfPairState.STOCK_HEDGE_PENDING
            try:
                self._submit_incremental_stock_hedge()
            except Exception:
                self._submission_halted = True
                self._close_reason = "stock hedge setup or submission failed"
                self._transition_to_recovery_required("stock hedge setup or submission failed")
        else:
            self._stock_filled_quantity = leg_cumulative
            self._state = LeveragedEtfPairState.STOCK_HEDGE_PENDING
            if outcome == "FILLED":
                self._append_identity_event(
                    event_type=JournalEventType.HEDGE_CONFIRMED,
                    payload=HedgeJournalPayloadV1(identity=intent.identity, phase="CONFIRMED"),
                    identity=intent.identity,
                )
                if self.stock_pending_quantity == _ZERO and self._exposure_is_balanced():
                    self._state = (
                        LeveragedEtfPairState.COMPLETED
                        if (
                            self._etf_filled_quantity == self.config.etf_target_quantity
                            and self._stock_filled_quantity == self.config.stock_target_quantity
                        )
                        else LeveragedEtfPairState.MAKER_WORKING
                    )

    def process_order_failed_event(self, _: int, market, event: MarketOrderFailureEvent):
        if self._submission_halted:
            return
        intent = self._intent_for_client_order_id(event.order_id)
        if intent is None:
            return
        # The connector had already allocated the stable client id, so preserve the
        # ambiguity/rejection fact and stop automatic submission rather than creating
        # another maker order.  The F004 reducer permits reconciliation after ACK.
        self._append_identity_event(
            event_type=JournalEventType.RECONCILIATION,
            payload=ReconciliationJournalPayloadV1(
                identity=intent.identity,
                outcome=ReconciliationOutcome.REJECTED,
                exchange_order_id=intent.exchange_order_id,
                order_cumulative_filled_quantity=intent.filled_quantity,
            ),
            identity=intent.identity,
            exchange_order_id=intent.exchange_order_id,
        )
        self._submission_halted = True
        self._transition_to_reconciling("connector reported order failure after client id allocation")

    def get_custom_info(self) -> Dict:
        state = LeveragedEtfPairExecutorStateV1(
            executor_id=self.config.id,
            state=self._state,
            etf_submitted_quantity=(
                _ZERO if self._maker_intent is None else self._maker_intent.identity.order_quantity
            ),
            etf_filled_quantity=self._etf_filled_quantity,
            etf_remaining_quantity=max(_ZERO, self.config.etf_target_quantity - self._etf_filled_quantity),
            stock_submitted_quantity=sum(
                (intent.identity.order_quantity for intent in self._stock_intents.values()),
                _ZERO,
            ),
            stock_filled_quantity=self._stock_filled_quantity,
            stock_remaining_quantity=max(_ZERO, self.config.stock_target_quantity - self._stock_filled_quantity),
            hedge_dust_quantity=self._hedge_dust_quantity,
            maker_order_ids=self._order_references((self._maker_intent,) if self._maker_intent else ()),
            stock_order_ids=self._order_references(tuple(self._stock_intents.values())),
            leverage_reservation=self.config.leverage_reservation,
            updated_at_utc=self._event_clock,
            close_reason=self._close_reason,
            last_journal_sequence=self._journal_event_count,
        )
        return LeveragedEtfPairExecutorCustomInfoV1(
            state=state,
            report=LeveragedEtfPairExecutorReportV1(
                net_pnl_pct=_ZERO,
                net_pnl_quote=_ZERO,
                realized_pnl_quote=_ZERO,
                unrealized_pnl_quote=_ZERO,
                cum_fees_quote=_ZERO,
                filled_amount_quote=self.filled_amount_quote,
            ),
        ).model_dump(mode="json")

    def _ensure_journal_snapshot(self):
        if self._journal_initialized:
            return
        initial_state = LeveragedEtfPairExecutorStateV1(
            executor_id=self.config.id,
            state=LeveragedEtfPairState.CREATED,
            etf_submitted_quantity=_ZERO,
            etf_filled_quantity=_ZERO,
            etf_remaining_quantity=self.config.etf_target_quantity,
            stock_submitted_quantity=_ZERO,
            stock_filled_quantity=_ZERO,
            stock_remaining_quantity=self.config.stock_target_quantity,
            hedge_dust_quantity=_ZERO,
            maker_order_ids=(),
            stock_order_ids=(),
            leverage_reservation=self.config.leverage_reservation,
            updated_at_utc=self.config.created_at_utc,
            close_reason=None,
            last_journal_sequence=0,
        )
        snapshot = LeveragedEtfPairExecutorSnapshotV1.from_config_and_state(self.config, initial_state)
        self._journal_repository.create_executor(snapshot)
        self._journal_initialized = True
        self._append_state_transition(LeveragedEtfPairState.PREFLIGHT, "entry preflight started")

    async def _run_strict_preflight(self):
        if self._preflight_completed:
            return
        pairs_by_connector: Dict[str, list[str]] = {}
        for connector_name, trading_pair in (
            (self.config.etf_connector_name, self.config.etf_trading_pair),
            (self.config.stock_connector_name, self.config.stock_trading_pair),
        ):
            pairs_by_connector.setdefault(connector_name, []).append(trading_pair)

        for connector_name, trading_pairs in pairs_by_connector.items():
            connector = self.connectors[connector_name]
            preflight = getattr(connector, "strict_account_preflight", None)
            if preflight is None:
                raise RuntimeError(f"connector {connector_name} does not implement strict_account_preflight")
            snapshot = await preflight(
                tuple(trading_pairs),
                related_trading_pairs=tuple(trading_pairs),
                # T001 opens new pair exposure. Existing activity must remain an
                # authoritative preflight failure rather than being silently marked
                # known by this entry-only executor.
                known_position_trading_pairs=(),
            )
            for instrument in snapshot.instruments:
                pair = instrument.trading_pair
                if pair in trading_pairs:
                    multiplier = Decimal(instrument.contract_multiplier)
                    if multiplier <= _ZERO:
                        raise RuntimeError(f"non-positive contract multiplier for {pair}")
                    self._contract_multipliers[(connector_name, pair)] = multiplier

        etf_multiplier = self._contract_multiplier(self.config.etf_connector_name, self.config.etf_trading_pair)
        stock_multiplier = self._contract_multiplier(self.config.stock_connector_name, self.config.stock_trading_pair)
        if self.config.etf_target_quantity <= _ZERO or self.config.stock_target_quantity <= _ZERO:
            raise RuntimeError("entry executor requires positive ETF and stock target quantities")
        expected_stock_quantity = abs(
            calculate_leg_quantities(
                etf_quantity=self.config.etf_target_quantity,
                direction=ArbitrageDirection(self.config.direction.value),
                hedge_ratio=self.config.h,
                etf_contract_multiplier=etf_multiplier,
                stock_contract_multiplier=stock_multiplier,
            ).stock_quantity
        )
        quantized_stock_quantity = self._quantize_order_quantity(
            self.config.stock_connector_name,
            self.config.stock_trading_pair,
            expected_stock_quantity,
        )
        if quantized_stock_quantity != self.config.stock_target_quantity:
            raise RuntimeError("preflight contract multipliers do not reproduce the frozen stock target")
        if (
            self._quantize_order_quantity(
                self.config.etf_connector_name,
                self.config.etf_trading_pair,
                self.config.etf_target_quantity,
            )
            != self.config.etf_target_quantity
        ):
            raise RuntimeError("frozen ETF target is not a legal order quantity")
        self._preflight_completed = True

    def _submit_maker_order(self):
        if self._maker_submission_started:
            return
        self._maker_submission_started = True
        maker_side = self._etf_trade_side()
        price_type = PriceType.BestBid if maker_side is TradeType.BUY else PriceType.BestAsk
        price = self._quantize_price(
            self.config.etf_connector_name,
            self.config.etf_trading_pair,
            Decimal(self.get_price(self.config.etf_connector_name, self.config.etf_trading_pair, price_type)),
        )
        identity = self._new_identity(
            action=JournalSideEffect.ETF_MAKER,
            leg="ETF",
            logical_quantity=self.config.etf_target_quantity,
            order_quantity=self.config.etf_target_quantity,
        )
        self._maker_intent = _OrderIntent(identity=identity)
        self._append_identity_event(
            event_type=JournalEventType.PREPARED,
            payload=PreparedJournalPayloadV1(identity=identity),
            identity=identity,
        )
        self._state = LeveragedEtfPairState.MAKER_SUBMITTING
        try:
            order_id = self._native_submit(
                connector_name=self.config.etf_connector_name,
                trading_pair=self.config.etf_trading_pair,
                side=maker_side,
                amount=self.config.etf_target_quantity,
                order_type=OrderType.LIMIT_MAKER,
                price=price,
                client_order_id=identity.client_order_id,
            )
        except Exception as exc:
            self._record_submission_exception(identity, exc, is_maker=True)
            return
        if order_id != identity.client_order_id:
            self._transition_to_recovery_required("connector did not retain the preallocated maker client id")
            return
        self._append_identity_event(
            event_type=JournalEventType.ACKNOWLEDGED,
            payload=AcknowledgedJournalPayloadV1(identity=identity),
            identity=identity,
        )
        # A connector can deliver a fill before this synchronous call returns.
        # F004 accepts the late ACK, but local state must not regress from the
        # fill-driven hedge state.
        self._state = (
            LeveragedEtfPairState.MAKER_WORKING
            if self._etf_filled_quantity == _ZERO
            else LeveragedEtfPairState.STOCK_HEDGE_PENDING
        )

    def _submit_incremental_stock_hedge(self):
        if self._submission_halted:
            return
        etf_multiplier = self._contract_multiplier(self.config.etf_connector_name, self.config.etf_trading_pair)
        stock_multiplier = self._contract_multiplier(self.config.stock_connector_name, self.config.stock_trading_pair)
        raw_target = abs(
            calculate_leg_quantities(
                etf_quantity=self._etf_filled_quantity,
                direction=ArbitrageDirection(self.config.direction.value),
                hedge_ratio=self.config.h,
                etf_contract_multiplier=etf_multiplier,
                stock_contract_multiplier=stock_multiplier,
            ).stock_quantity
        )
        quantized_target = self._quantize_order_quantity(
            self.config.stock_connector_name,
            self.config.stock_trading_pair,
            raw_target,
        )
        if quantized_target > self.config.stock_target_quantity:
            self._transition_to_recovery_required("incremental hedge target exceeded frozen stock target")
            return
        self._stock_hedge_target_quantity = quantized_target
        self._hedge_dust_quantity = raw_target - quantized_target
        incremental_quantity = quantized_target - self._stock_filled_quantity - self.stock_pending_quantity
        if incremental_quantity <= _ZERO:
            return
        legal_increment = self._quantize_order_quantity(
            self.config.stock_connector_name,
            self.config.stock_trading_pair,
            incremental_quantity,
        )
        if legal_increment != incremental_quantity or legal_increment <= _ZERO:
            self._transition_to_recovery_required("incremental hedge quantity is not legally quantized")
            return
        identity = self._new_identity(
            action=JournalSideEffect.STOCK_HEDGE,
            leg="STOCK",
            logical_quantity=quantized_target,
            order_quantity=legal_increment,
        )
        intent = _OrderIntent(identity=identity)
        self._stock_intents[identity.client_order_id] = intent
        self._append_identity_event(
            event_type=JournalEventType.PREPARED,
            payload=PreparedJournalPayloadV1(identity=identity),
            identity=identity,
        )
        self._append_identity_event(
            event_type=JournalEventType.HEDGE_REQUESTED,
            payload=HedgeJournalPayloadV1(identity=identity, phase="REQUESTED"),
            identity=identity,
        )
        try:
            order_id = self._native_submit(
                connector_name=self.config.stock_connector_name,
                trading_pair=self.config.stock_trading_pair,
                side=self._stock_trade_side(),
                amount=legal_increment,
                order_type=OrderType.MARKET,
                price=_NAN,
                client_order_id=identity.client_order_id,
            )
        except Exception as exc:
            self._record_submission_exception(identity, exc, is_maker=False)
            return
        if order_id != identity.client_order_id:
            self._transition_to_recovery_required("connector did not retain the preallocated stock client id")
            return
        self._append_identity_event(
            event_type=JournalEventType.ACKNOWLEDGED,
            payload=AcknowledgedJournalPayloadV1(identity=identity),
            identity=identity,
        )

    def _record_submission_exception(self, identity: SideEffectIdentityV1, exc: Exception, *, is_maker: bool):
        failure_kind = getattr(exc, "failure_kind", None)
        if getattr(failure_kind, "value", None) == "AUTHORITATIVE_REJECTION":
            self._append_identity_event(
                event_type=JournalEventType.REJECTED,
                payload=RejectedJournalPayloadV1(identity=identity, reason="authoritative submission rejection"),
                identity=identity,
            )
            self._state = LeveragedEtfPairState.PREFLIGHT if is_maker else LeveragedEtfPairState.STOCK_HEDGE_PENDING
        else:
            self._append_identity_event(
                event_type=JournalEventType.SUBMIT_UNKNOWN,
                payload=SubmitUnknownJournalPayloadV1(
                    identity=identity,
                    uncertainty_started_at_utc=self._event_clock,
                ),
                identity=identity,
            )
            self._state = LeveragedEtfPairState.RECONCILING
        self._submission_halted = True
        self._close_reason = "order submission failed"

    def _native_submit(
        self,
        *,
        connector_name: str,
        trading_pair: str,
        side: TradeType,
        amount: Decimal,
        order_type: OrderType,
        price: Decimal,
        client_order_id: str,
    ) -> str:
        connector = self.connectors[connector_name]
        submit = connector.buy if side is TradeType.BUY else connector.sell
        return submit(
            trading_pair=trading_pair,
            amount=amount,
            order_type=order_type,
            price=price,
            client_order_id=client_order_id,
            position_action=PositionAction.OPEN,
        )

    def _new_identity(
        self,
        *,
        action: JournalSideEffect,
        leg: str,
        logical_quantity: Decimal,
        order_quantity: Decimal,
    ) -> SideEffectIdentityV1:
        logical_text = _canonical_decimal(logical_quantity)
        idempotency_key = f"{self.config.id}:{self.config.operation.value}:{leg}:{logical_text}:1"
        digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
        connector_name, trading_pair = (
            (self.config.etf_connector_name, self.config.etf_trading_pair)
            if leg == "ETF"
            else (self.config.stock_connector_name, self.config.stock_trading_pair)
        )
        return SideEffectIdentityV1(
            executor_id=self.config.id,
            operation=self.config.operation,
            action=action,
            leg=leg,
            logical_quantity=logical_quantity,
            order_quantity=order_quantity,
            attempt=1,
            intent_id=f"lep-int-{digest[:24]}",
            idempotency_key=idempotency_key,
            connector_name=connector_name,
            trading_pair=trading_pair,
            client_order_id=f"lepf-{digest[:27]}",
            deadline_utc=self.config.created_at_utc + timedelta(days=1),
        )

    def _append_state_transition(self, state: LeveragedEtfPairState, reason: str):
        event = JournalEventV1(
            event_id=self._new_event_id(JournalEventType.STATE_TRANSITION, None),
            event_type=JournalEventType.STATE_TRANSITION,
            payload=StateTransitionJournalPayloadV1(reason=reason, target_state=state),
            created_at_utc=self._next_event_time(),
        )
        self._journal_repository.append_and_reduce(self.config.id, event)
        self._journal_event_count += 1
        self._state = state

    def _append_identity_event(
        self,
        *,
        event_type: JournalEventType,
        payload,
        identity: SideEffectIdentityV1,
        exchange_order_id: Optional[str] = None,
        exchange_trade_id: Optional[str] = None,
    ):
        event = JournalEventV1(
            event_id=self._new_event_id(event_type, identity),
            event_type=event_type,
            intent_id=identity.intent_id,
            idempotency_key=identity.idempotency_key,
            connector_name=identity.connector_name,
            trading_pair=identity.trading_pair,
            client_order_id=identity.client_order_id,
            exchange_order_id=exchange_order_id,
            exchange_trade_id=exchange_trade_id,
            payload=payload,
            created_at_utc=self._next_event_time(),
        )
        self._journal_repository.append_and_reduce(self.config.id, event)
        self._journal_event_count += 1

    def _transition_to_reconciling(self, reason: str):
        if not self._journal_initialized or self._state is LeveragedEtfPairState.RECONCILING:
            self._state = LeveragedEtfPairState.RECONCILING
            return
        self._append_state_transition(LeveragedEtfPairState.RECONCILING, reason)

    def _transition_to_recovery_required(self, reason: str):
        if not self._journal_initialized or self._state is LeveragedEtfPairState.RECOVERY_REQUIRED:
            self._state = LeveragedEtfPairState.RECOVERY_REQUIRED
            return
        self._append_state_transition(LeveragedEtfPairState.RECOVERY_REQUIRED, reason)

    def _next_event_time(self) -> datetime:
        self._event_clock += timedelta(microseconds=1)
        return self._event_clock

    def _new_event_id(
        self,
        event_type: JournalEventType,
        identity: Optional[SideEffectIdentityV1],
    ) -> str:
        sequence = self._journal_event_count + 1
        seed = f"{self.config.id}:{event_type.value}:{sequence}:{identity.intent_id if identity else 'state'}"
        return f"lep-evt-{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:32]}"

    def _intent_for_client_order_id(self, client_order_id: str) -> Optional[_OrderIntent]:
        if self._maker_intent is not None and client_order_id == self._maker_intent.identity.client_order_id:
            return self._maker_intent
        return self._stock_intents.get(client_order_id)

    def _contract_multiplier(self, connector_name: str, trading_pair: str) -> Decimal:
        try:
            return self._contract_multipliers[(connector_name, trading_pair)]
        except KeyError as exc:
            raise RuntimeError(f"strict preflight omitted multiplier for {connector_name}:{trading_pair}") from exc

    def _trading_rule(self, connector_name: str, trading_pair: str) -> TradingRule:
        try:
            return self.get_trading_rules(connector_name, trading_pair)
        except (KeyError, AttributeError) as exc:
            raise RuntimeError(f"trading rule unavailable for {connector_name}:{trading_pair}") from exc

    def _quantize_order_quantity(self, connector_name: str, trading_pair: str, quantity: Decimal) -> Decimal:
        if quantity <= _ZERO:
            return _ZERO
        rule = self._trading_rule(connector_name, trading_pair)
        increment = Decimal(rule.min_base_amount_increment)
        min_order_size = Decimal(rule.min_order_size)
        if increment <= _ZERO or min_order_size <= _ZERO:
            raise RuntimeError(f"invalid quantity rule for {connector_name}:{trading_pair}")
        quantized = (quantity / increment).to_integral_value(rounding=ROUND_DOWN) * increment
        return quantized if quantized >= min_order_size else _ZERO

    def _quantize_price(self, connector_name: str, trading_pair: str, price: Decimal) -> Decimal:
        rule = self._trading_rule(connector_name, trading_pair)
        increment = Decimal(rule.min_price_increment)
        if price <= _ZERO or increment <= _ZERO:
            raise RuntimeError(f"invalid price rule for {connector_name}:{trading_pair}")
        return (price / increment).to_integral_value(rounding=ROUND_DOWN) * increment

    def _etf_trade_side(self) -> TradeType:
        return (
            TradeType.SELL
            if self.config.direction is LeveragedEtfPairDirection.SHORT_ETF_LONG_STOCK
            else TradeType.BUY
        )

    def _stock_trade_side(self) -> TradeType:
        return (
            TradeType.BUY
            if self.config.direction is LeveragedEtfPairDirection.SHORT_ETF_LONG_STOCK
            else TradeType.SELL
        )

    def _exposure_is_balanced(self) -> bool:
        return (
            self._etf_filled_quantity * self.config.stock_target_quantity
            == self._stock_filled_quantity * self.config.etf_target_quantity
        )

    @staticmethod
    def _order_references(intents: Tuple[_OrderIntent, ...]) -> Tuple[OrderReferenceV1, ...]:
        references = []
        for sequence, intent in enumerate(intents):
            if intent.exchange_order_id is not None:
                references.append(
                    OrderReferenceV1(
                        sequence=sequence,
                        client_order_id=intent.identity.client_order_id,
                        exchange_order_id=intent.exchange_order_id,
                    )
                )
        return tuple(references)
