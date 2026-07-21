"""Native Strategy V2 executor for a leveraged ETF / stock perpetual pair.

The executor persists every side effect before calling a connector, keeps the
maker leg singular, and converts each accepted ETF fill into the smallest legal
cumulative stock hedge.  It also owns the entry-safety cancellation, deadline,
and exact rollback path; normal close/stop behavior remains separate work.
"""

from __future__ import annotations

import hashlib
import inspect
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import ROUND_DOWN, Decimal
from typing import Callable, Dict, Optional, Tuple, Union

from hummingbot.connector.markets_recorder import MarketsRecorder
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.core.data_type.common import OrderType, PositionAction, PriceType, TradeType
from hummingbot.core.event.events import (
    BuyOrderCreatedEvent,
    MarketOrderFailureEvent,
    OrderCancelledEvent,
    OrderFilledEvent,
    SellOrderCreatedEvent,
)
from hummingbot.model.leveraged_etf_repository import (
    AcknowledgedJournalPayloadV1,
    CancelJournalPayloadV1,
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
    RollbackJournalPayloadV1,
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
    LeveragedEtfPairOperation,
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
    submitted: bool = False
    acknowledged: bool = False
    submission_unknown: bool = False
    terminal: bool = False
    unknown_since_monotonic: Optional[float] = None
    reconciliation_sweeps: int = 0
    last_sweep_monotonic: Optional[float] = None
    last_sweep_signature: Optional[Tuple] = None

    @property
    def remaining_quantity(self) -> Decimal:
        return max(_ZERO, self.identity.order_quantity - self.filled_quantity)

    @property
    def pending_quantity(self) -> Decimal:
        return _ZERO if self.terminal else self.remaining_quantity


@dataclass
class _CancelIntent:
    identity: SideEffectIdentityV1
    target: _OrderIntent
    requested_at_monotonic: float
    submission_unknown: bool = False
    unknown_since_monotonic: Optional[float] = None
    confirmed: bool = False
    reconciliation_sweeps: int = 0
    last_sweep_monotonic: Optional[float] = None
    last_sweep_signature: Optional[Tuple] = None


@dataclass
class _MakerSafetyObservation:
    net_bp: Decimal
    direction_stable: bool
    model_valid: bool
    target_etf_quantity: Decimal
    entry_permitted: bool
    residual_bp: Optional[Decimal] = None
    reprice_requested: bool = False
    revision: int = 0


@dataclass(frozen=True)
class LeveragedEtfPairSafetyPolicy:
    """Runtime-only safety inputs kept outside the frozen F004 executor DTO.

    F004 deliberately persists only immutable execution facts.  These values are
    supplied by the Strategy/Controller runtime and can therefore be exercised
    with a deterministic monotonic clock without widening the wire contract.
    """

    executor_safety_interval_ms: int = 250
    divergence_cancel_bp: Decimal = Decimal("3")
    divergence_confirmations: int = 3
    maker_max_age_ms: Optional[int] = None
    unhedged_response_deadline_ms: int = 20_000
    hedge_phase_deadline_ms: int = 10_000
    hedge_submit_timeout_ms: int = 1_000
    hedge_reconcile_timeout_ms: int = 5_000
    hedge_max_attempts: int = 3
    hedge_retry_backoff_ms: Tuple[int, ...] = (100, 250, 500)
    rollback_submit_timeout_ms: int = 1_000
    rollback_reconcile_timeout_ms: int = 5_000
    rollback_phase_deadline_ms: int = 10_000
    rollback_max_attempts: int = 3
    rollback_retry_backoff_ms: Tuple[int, ...] = (100, 250, 500)
    order_eventual_consistency_grace_ms: int = 2_000

    def __post_init__(self):
        positive_values = {
            "executor_safety_interval_ms": self.executor_safety_interval_ms,
            "unhedged_response_deadline_ms": self.unhedged_response_deadline_ms,
            "hedge_phase_deadline_ms": self.hedge_phase_deadline_ms,
            "hedge_submit_timeout_ms": self.hedge_submit_timeout_ms,
            "hedge_reconcile_timeout_ms": self.hedge_reconcile_timeout_ms,
            "hedge_max_attempts": self.hedge_max_attempts,
            "rollback_submit_timeout_ms": self.rollback_submit_timeout_ms,
            "rollback_reconcile_timeout_ms": self.rollback_reconcile_timeout_ms,
            "rollback_phase_deadline_ms": self.rollback_phase_deadline_ms,
            "rollback_max_attempts": self.rollback_max_attempts,
            "order_eventual_consistency_grace_ms": self.order_eventual_consistency_grace_ms,
        }
        if self.maker_max_age_ms is not None:
            positive_values["maker_max_age_ms"] = self.maker_max_age_ms
        if any(not isinstance(value, int) or value <= 0 for value in positive_values.values()):
            raise ValueError("executor safety durations and attempt limits must be positive integers")
        if not isinstance(self.divergence_confirmations, int) or self.divergence_confirmations <= 0:
            raise ValueError("divergence_confirmations must be a positive integer")
        if not isinstance(self.divergence_cancel_bp, Decimal) or not self.divergence_cancel_bp.is_finite():
            raise ValueError("divergence_cancel_bp must be a finite Decimal")
        if self.divergence_cancel_bp < _ZERO:
            raise ValueError("divergence_cancel_bp must be non-negative")
        if self.hedge_phase_deadline_ms + self.rollback_phase_deadline_ms > self.unhedged_response_deadline_ms:
            raise ValueError("hedge and rollback phases exceed the absolute unhedged deadline")
        if self.hedge_phase_deadline_ms < (
            self.hedge_submit_timeout_ms + self.order_eventual_consistency_grace_ms
        ):
            raise ValueError("hedge phase cannot contain one submit timeout and consistency grace")
        if self.rollback_phase_deadline_ms < (
            self.rollback_submit_timeout_ms + self.order_eventual_consistency_grace_ms
        ):
            raise ValueError("rollback phase cannot contain one submit timeout and consistency grace")
        if len(self.hedge_retry_backoff_ms) < self.hedge_max_attempts - 1:
            raise ValueError("hedge retry backoff does not cover all attempts")
        if len(self.rollback_retry_backoff_ms) < self.rollback_max_attempts - 1:
            raise ValueError("rollback retry backoff does not cover all attempts")
        if any(not isinstance(value, int) or value <= 0 for value in self.hedge_retry_backoff_ms):
            raise ValueError("hedge retry backoff values must be positive integers")
        if any(not isinstance(value, int) or value <= 0 for value in self.rollback_retry_backoff_ms):
            raise ValueError("rollback retry backoff values must be positive integers")


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
        safety_policy: Optional[LeveragedEtfPairSafetyPolicy] = None,
        monotonic_clock: Optional[Callable[[], float]] = None,
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
        self._safety_policy = safety_policy or LeveragedEtfPairSafetyPolicy()
        self._monotonic_clock = monotonic_clock or time.monotonic

        self._contract_multipliers: Dict[Tuple[str, str], Decimal] = {}
        self._maker_intent: Optional[_OrderIntent] = None
        self._stock_intents: Dict[str, _OrderIntent] = {}
        self._rollback_intents: Dict[str, _OrderIntent] = {}
        self._cancel_intent: Optional[_CancelIntent] = None
        self._seen_trades: set[Tuple[str, str]] = set()

        self._etf_filled_quantity = _ZERO
        self._stock_filled_quantity = _ZERO
        self._rollback_filled_quantity = _ZERO
        self._stock_hedge_target_quantity = _ZERO
        self._hedge_dust_quantity = _ZERO
        self._close_reason: Optional[str] = None
        self._maker_terminal = False
        self._maker_started_at_monotonic: Optional[float] = None
        self._exposure_started_at_monotonic: Optional[float] = None
        self._hedge_deadline_at_monotonic: Optional[float] = None
        self._absolute_deadline_at_monotonic: Optional[float] = None
        self._rollback_started_at_monotonic: Optional[float] = None
        self._rollback_deadline_at_monotonic: Optional[float] = None
        self._rollback_required = False
        self._next_hedge_attempt_not_before = 0.0
        self._next_rollback_attempt_not_before = 0.0
        self._maker_safety = _MakerSafetyObservation(
            net_bp=config.created_net_bp,
            direction_stable=True,
            model_valid=True,
            target_etf_quantity=config.etf_target_quantity,
            entry_permitted=True,
        )
        self._last_monitored_safety_revision = -1
        self._initial_abs_residual_bp: Optional[Decimal] = None
        self._best_abs_residual_bp: Optional[Decimal] = None
        self._current_abs_residual_bp: Optional[Decimal] = None
        self._divergence_consecutive_count = 0

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
        return sum((intent.pending_quantity for intent in self._stock_intents.values()), _ZERO)

    @property
    def hedge_dust_quantity(self) -> Decimal:
        return self._hedge_dust_quantity

    @property
    def exposure_started_at(self) -> Optional[float]:
        """The monotonic t0 for the current unhedged ETF exposure episode."""
        return self._exposure_started_at_monotonic

    @property
    def hedge_deadline_at(self) -> Optional[float]:
        return self._hedge_deadline_at_monotonic

    @property
    def absolute_deadline_at(self) -> Optional[float]:
        return self._absolute_deadline_at_monotonic

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

    def update_maker_safety(
        self,
        *,
        net_bp: Optional[Decimal] = None,
        direction_stable: Optional[bool] = None,
        model_valid: Optional[bool] = None,
        target_etf_quantity: Optional[Decimal] = None,
        entry_permitted: Optional[bool] = None,
        residual_bp: Optional[Decimal] = None,
        reprice_requested: Optional[bool] = None,
    ) -> None:
        """Accept the latest Controller-derived maker safety facts.

        The Controller remains the source of opportunity, target, model, and
        session facts.  This executor merely consumes one immutable observation
        at a time and never invents a replacement maker after cancellation.
        """
        current = self._maker_safety
        if net_bp is not None:
            net_bp = Decimal(net_bp)
            if not net_bp.is_finite():
                raise ValueError("net_bp must be finite")
        if target_etf_quantity is not None:
            target_etf_quantity = Decimal(target_etf_quantity)
            if not target_etf_quantity.is_finite() or target_etf_quantity < _ZERO:
                raise ValueError("target_etf_quantity must be a finite non-negative Decimal")
        if residual_bp is not None:
            residual_bp = Decimal(residual_bp)
            if not residual_bp.is_finite():
                raise ValueError("residual_bp must be finite")
        self._maker_safety = _MakerSafetyObservation(
            net_bp=current.net_bp if net_bp is None else net_bp,
            direction_stable=current.direction_stable if direction_stable is None else bool(direction_stable),
            model_valid=current.model_valid if model_valid is None else bool(model_valid),
            target_etf_quantity=(
                current.target_etf_quantity if target_etf_quantity is None else target_etf_quantity
            ),
            entry_permitted=current.entry_permitted if entry_permitted is None else bool(entry_permitted),
            residual_bp=current.residual_bp if residual_bp is None else residual_bp,
            reprice_requested=current.reprice_requested if reprice_requested is None else bool(reprice_requested),
            revision=current.revision + 1,
        )

    def mark_intent_submission_unknown(self, client_order_id: str, reason: str) -> None:
        """Expose an explicit fail-closed test/runtime hook for F002 ambiguity."""
        intent = self._intent_for_client_order_id(client_order_id)
        if intent is None:
            raise ValueError("cannot mark an unknown client order id as ambiguous")
        self._mark_submission_unknown(intent=intent, reason=reason)

    async def control_task(self):
        self._consume_submission_unknowns()
        if self._submission_halted:
            await self._reconcile_submission_unknowns()
            self._enforce_absolute_deadline()
            return
        if self._maker_submission_started:
            await self._advance_safety_state_machine()
            return
        self._ensure_journal_snapshot()
        if not self._entry_operation_is_supported():
            self._halt_unsupported_operation()
            return
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
        if intent.submission_unknown:
            return
        exchange_order_id = event.exchange_order_id
        if not exchange_order_id:
            self._transition_to_recovery_required("order created event omitted exchange order id")
            return
        if intent.exchange_order_id is not None and intent.exchange_order_id != exchange_order_id:
            self._transition_to_recovery_required("client order id mapped to conflicting exchange order ids")
            return
        if intent.exchange_order_id is None:
            self._append_identity_event(
                event_type=JournalEventType.ORDER_CREATED,
                payload=OrderCreatedJournalPayloadV1(identity=intent.identity, exchange_order_id=exchange_order_id),
                identity=intent.identity,
                exchange_order_id=exchange_order_id,
            )
            intent.exchange_order_id = exchange_order_id
        was_acknowledged = intent.acknowledged
        self._acknowledge_intent(intent)
        if not was_acknowledged:
            self._apply_acknowledged_state(intent)
            if intent.identity.action is JournalSideEffect.STOCK_HEDGE and not self._submission_halted:
                try:
                    # ETF fills that arrived while the native stock submission
                    # was unconfirmed were deliberately held.  An authoritative
                    # creation event is the safe point to submit that backlog.
                    self._submit_incremental_stock_hedge()
                except Exception:
                    self._submission_halted = True
                    self._close_reason = "stock hedge setup or submission failed"
                    self._transition_to_recovery_required("stock hedge setup or submission failed")

    def process_order_canceled_event(self, _: int, market, event: OrderCancelledEvent):
        """Treat a maker cancel event as terminal only after persisting it.

        A late ETF fill remains admissible until this event or a terminal REST
        reconciliation is durably recorded.  We deliberately retain the maker
        intent and its exchange ID for later F007 recovery/audit work.
        """
        if self._maker_intent is None or event.order_id != self._maker_intent.identity.client_order_id:
            return
        if self._cancel_intent is None or self._cancel_intent.confirmed:
            return
        maker = self._maker_intent
        exchange_order_id = str(event.exchange_order_id or maker.exchange_order_id or "") or None
        if maker.exchange_order_id is not None and exchange_order_id != maker.exchange_order_id:
            self._transition_to_recovery_required("maker cancel event exchange order id disagreed with maker identity")
            return
        if exchange_order_id is not None:
            maker.exchange_order_id = exchange_order_id
        try:
            self._confirm_maker_cancel()
        except Exception:
            # Never infer terminal cancellation if its durable fact cannot be
            # accepted.  F007 can reconcile the retained maker/cancel IDs.
            self._submission_halted = True
            self._close_reason = "unable to persist maker cancellation confirmation"
            self._transition_to_recovery_required(self._close_reason)

    def process_order_filled_event(self, _: int, market, event: OrderFilledEvent):
        intent = self._intent_for_client_order_id(event.order_id)
        if intent is None:
            return
        if intent.submission_unknown:
            return
        trade_id = str(event.exchange_trade_id or "")
        exchange_order_id = str(event.exchange_order_id or intent.exchange_order_id or "")
        quantity = Decimal(event.amount)
        self._record_authoritative_fill(
            intent=intent,
            exchange_order_id=exchange_order_id,
            exchange_trade_id=trade_id,
            price=Decimal(event.price),
            quantity=quantity,
        )

    def process_order_failed_event(self, _: int, market, event: MarketOrderFailureEvent):
        if self._submission_halted:
            return
        intent = self._intent_for_client_order_id(event.order_id)
        if intent is None:
            return
        # The connector had already allocated the stable client id, so preserve the
        # ambiguity/rejection fact and stop automatic submission rather than creating
        # another maker order.  The F004 reducer permits reconciliation from a
        # prepared or acknowledged native submission.
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
        intent.terminal = True
        if intent.identity.action is JournalSideEffect.ETF_MAKER:
            self._submission_halted = True
            self._transition_to_reconciling("connector reported order failure after client id allocation")
            return
        if intent.identity.action is JournalSideEffect.STOCK_HEDGE:
            self._rollback_required = True
            self._schedule_next_hedge_attempt()
            return
        self._submission_halted = True
        self._transition_to_reconciling("connector reported rollback order failure after client id allocation")

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
        if not self._entry_operation_is_supported():
            self._halt_unsupported_operation()
            return
        maker_side = self._etf_trade_side()
        price_type = PriceType.BestBid if maker_side is TradeType.BUY else PriceType.BestAsk
        price = self._quantize_price(
            self.config.etf_connector_name,
            self.config.etf_trading_pair,
            Decimal(self.get_price(self.config.etf_connector_name, self.config.etf_trading_pair, price_type)),
        )
        legal_quantity = self._quantize_order_quantity_at_price(
            self.config.etf_connector_name,
            self.config.etf_trading_pair,
            self.config.etf_target_quantity,
            price,
        )
        if legal_quantity != self.config.etf_target_quantity:
            self._submission_halted = True
            self._close_reason = "maker quantity is below the actual-price trading-rule minimum"
            self._transition_to_recovery_required(self._close_reason)
            return
        self._maker_submission_started = True
        self._maker_started_at_monotonic = self._now_monotonic()
        identity = self._new_identity(
            action=JournalSideEffect.ETF_MAKER,
            leg="ETF",
            logical_quantity=self.config.etf_target_quantity,
            order_quantity=self.config.etf_target_quantity,
        )
        intent = _OrderIntent(identity=identity)
        self._maker_intent = intent
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
            self._mark_submission_unknown(intent=intent, reason="connector did not retain maker client id")
            return
        intent.submitted = True
        self._consume_submission_unknowns()

    def _submit_incremental_stock_hedge(self):
        if self._submission_halted or self._rollback_required:
            return
        if not self._entry_operation_is_supported():
            self._halt_unsupported_operation()
            return
        if self._consume_submission_unknowns() or self._has_unacknowledged_stock_submission():
            return
        if not self._post_fill_hedge_is_permitted():
            self._rollback_required = True
            self._request_maker_cancel("post-fill opportunity or direction check failed")
            return
        if self._hedge_backoff_is_active():
            return
        if not self._can_start_hedge_attempt():
            self._rollback_required = True
            self._request_maker_cancel("hedge phase has insufficient remaining time")
            return
        if self._hedge_attempt_count() >= self._safety_policy.hedge_max_attempts:
            self._rollback_required = True
            self._request_maker_cancel("hedge attempt budget exhausted")
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
        covered_quantity = self._stock_filled_quantity + self.stock_pending_quantity
        incremental_quantity = quantized_target - covered_quantity
        if incremental_quantity <= _ZERO:
            return
        try:
            stock_price = self._current_executable_stock_price()
        except Exception:
            self._rollback_required = True
            self._request_maker_cancel("stock executable depth or price became unavailable")
            return
        legal_increment = self._quantize_order_quantity_at_price(
            self.config.stock_connector_name,
            self.config.stock_trading_pair,
            incremental_quantity,
            stock_price,
        )
        if legal_increment <= _ZERO:
            self._hedge_dust_quantity = max(_ZERO, raw_target - covered_quantity)
            return
        self._hedge_dust_quantity = max(_ZERO, raw_target - covered_quantity - legal_increment)
        identity = self._new_identity(
            action=JournalSideEffect.STOCK_HEDGE,
            leg="STOCK",
            logical_quantity=quantized_target,
            order_quantity=legal_increment,
            attempt=self._hedge_attempt_count() + 1,
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
            self._mark_submission_unknown(intent=intent, reason="connector did not retain stock client id")
            return
        intent.submitted = True
        self._consume_submission_unknowns()

    def _record_authoritative_fill(
        self,
        *,
        intent: _OrderIntent,
        exchange_order_id: str,
        exchange_trade_id: str,
        price: Decimal,
        quantity: Decimal,
    ) -> None:
        """Persist and apply one idempotent fill for maker, hedge, or rollback."""
        action = intent.identity.action
        if action is JournalSideEffect.ETF_MAKER:
            trade_key = ("ETF_MAKER", exchange_trade_id)
            leg_cumulative = self._etf_filled_quantity + quantity
        elif action is JournalSideEffect.STOCK_HEDGE:
            trade_key = ("STOCK_HEDGE", exchange_trade_id)
            leg_cumulative = self._stock_filled_quantity + quantity
        elif action is JournalSideEffect.ETF_ROLLBACK:
            trade_key = ("ETF_ROLLBACK", exchange_trade_id)
            leg_cumulative = self._rollback_filled_quantity + quantity
        else:
            self._transition_to_recovery_required("fill event referenced a non-order side effect")
            return
        if not exchange_trade_id or not exchange_order_id or quantity <= _ZERO or not price.is_finite() or price <= _ZERO:
            self._transition_to_recovery_required("fill event lacked a positive exchange trade/order identity")
            return
        if trade_key in self._seen_trades:
            return
        if intent.exchange_order_id is not None and intent.exchange_order_id != exchange_order_id:
            self._transition_to_recovery_required("fill event exchange order id disagreed with order identity")
            return
        if quantity > intent.remaining_quantity:
            self._transition_to_recovery_required("fill exceeded prepared order quantity")
            return

        # A fill is an authoritative confirmation even when it races ahead of
        # OrderCreated.  It must be durably acknowledged before its exposure is
        # allowed to affect hedge or rollback decisions.
        self._acknowledge_intent(intent)
        order_cumulative = intent.filled_quantity + quantity
        outcome = "FILLED" if order_cumulative == intent.identity.order_quantity else "PARTIAL"
        self._append_identity_event(
            event_type=JournalEventType.FILL,
            payload=FillJournalPayloadV1(
                identity=intent.identity,
                exchange_order_id=exchange_order_id,
                exchange_trade_id=exchange_trade_id,
                price=price,
                fill_quantity=quantity,
                order_cumulative_filled_quantity=order_cumulative,
                leg_cumulative_filled_quantity=leg_cumulative,
                outcome=outcome,
            ),
            identity=intent.identity,
            exchange_order_id=exchange_order_id,
            exchange_trade_id=exchange_trade_id,
        )
        intent.exchange_order_id = exchange_order_id
        intent.filled_quantity = order_cumulative
        intent.terminal = outcome == "FILLED"
        self._seen_trades.add(trade_key)

        if action is JournalSideEffect.ETF_MAKER:
            self._etf_filled_quantity = leg_cumulative
            if intent.terminal:
                self._maker_terminal = True
            self._refresh_exposure_episode()
            self._state = LeveragedEtfPairState.STOCK_HEDGE_PENDING
            if not self._post_fill_hedge_is_permitted():
                self._rollback_required = True
                self._request_maker_cancel("post-fill opportunity or direction check failed")
                return
            try:
                self._submit_incremental_stock_hedge()
            except Exception:
                self._submission_halted = True
                self._close_reason = "stock hedge setup or submission failed"
                self._transition_to_recovery_required("stock hedge setup or submission failed")
            return

        if action is JournalSideEffect.STOCK_HEDGE:
            self._stock_filled_quantity = leg_cumulative
            self._state = LeveragedEtfPairState.STOCK_HEDGE_PENDING
            if intent.terminal:
                self._append_identity_event(
                    event_type=JournalEventType.HEDGE_CONFIRMED,
                    payload=HedgeJournalPayloadV1(identity=intent.identity, phase="CONFIRMED"),
                    identity=intent.identity,
                )
            self._refresh_exposure_episode()
            if self._exposure_is_balanced() and self.stock_pending_quantity == _ZERO:
                self._state = (
                    LeveragedEtfPairState.COMPLETED
                    if (
                        self._etf_filled_quantity == self.config.etf_target_quantity
                        and self._stock_filled_quantity == self.config.stock_target_quantity
                        and self._rollback_filled_quantity == _ZERO
                    )
                    else LeveragedEtfPairState.MAKER_WORKING
                )
            if not self._submission_halted and not self._rollback_required:
                try:
                    # A native stock fill can arrive before OrderCreated.  It
                    # authoritatively releases exactly the deferred cumulative
                    # hedge delta, never a duplicate order.
                    self._submit_incremental_stock_hedge()
                except Exception:
                    self._submission_halted = True
                    self._close_reason = "stock hedge setup or submission failed"
                    self._transition_to_recovery_required("stock hedge setup or submission failed")
            return

        self._rollback_filled_quantity = leg_cumulative
        self._state = LeveragedEtfPairState.ETF_ROLLBACK_PENDING
        self._refresh_exposure_episode()
        if intent.terminal:
            self._append_identity_event(
                event_type=JournalEventType.ROLLBACK_CONFIRMED,
                payload=RollbackJournalPayloadV1(identity=intent.identity, phase="CONFIRMED"),
                identity=intent.identity,
            )
            if self._unhedged_etf_quantity() == _ZERO:
                self._state = LeveragedEtfPairState.FAILED_SAFE
            else:
                self._transition_to_recovery_required("completed rollback did not clear exact unhedged ETF exposure")

    def _now_monotonic(self) -> float:
        now = float(self._monotonic_clock())
        if now < 0:
            raise RuntimeError("monotonic clock returned a negative value")
        return now

    def _milliseconds(self, value: int) -> float:
        return value / 1000

    def _hedged_etf_equivalent_quantity(self) -> Decimal:
        if self.config.stock_target_quantity <= _ZERO:
            return _ZERO
        return self._stock_filled_quantity * self.config.etf_target_quantity / self.config.stock_target_quantity

    def _unhedged_etf_quantity(self) -> Decimal:
        return max(
            _ZERO,
            self._etf_filled_quantity - self._rollback_filled_quantity - self._hedged_etf_equivalent_quantity(),
        )

    def _refresh_exposure_episode(self) -> None:
        outstanding = self._unhedged_etf_quantity()
        if outstanding > _ZERO and self._exposure_started_at_monotonic is None:
            start = self._now_monotonic()
            self._exposure_started_at_monotonic = start
            self._hedge_deadline_at_monotonic = start + self._milliseconds(
                self._safety_policy.hedge_phase_deadline_ms
            )
            self._absolute_deadline_at_monotonic = start + self._milliseconds(
                self._safety_policy.unhedged_response_deadline_ms
            )
            return
        if outstanding == _ZERO and self._exposure_started_at_monotonic is not None:
            self._exposure_started_at_monotonic = None
            self._hedge_deadline_at_monotonic = None
            self._absolute_deadline_at_monotonic = None
            self._rollback_started_at_monotonic = None
            self._rollback_deadline_at_monotonic = None
            self._hedge_dust_quantity = _ZERO

    def _post_fill_hedge_is_permitted(self) -> bool:
        observation = self._maker_safety
        return (
            observation.net_bp > _ZERO
            and observation.direction_stable
            and observation.model_valid
            and observation.entry_permitted
            and observation.target_etf_quantity > self._hedged_etf_equivalent_quantity()
        )

    def _hedge_attempt_count(self) -> int:
        return len(self._stock_intents)

    def _rollback_attempt_count(self) -> int:
        return len(self._rollback_intents)

    def _can_start_hedge_attempt(self) -> bool:
        if self._exposure_started_at_monotonic is None:
            return True
        now = self._now_monotonic()
        if now < self._next_hedge_attempt_not_before:
            return False
        deadline = min(
            self._hedge_deadline_at_monotonic or now,
            self._absolute_deadline_at_monotonic or now,
        )
        return now + self._milliseconds(
            self._safety_policy.hedge_submit_timeout_ms + self._safety_policy.order_eventual_consistency_grace_ms
        ) <= deadline

    def _hedge_backoff_is_active(self) -> bool:
        return self._now_monotonic() < self._next_hedge_attempt_not_before

    def _can_start_rollback_attempt(self) -> bool:
        now = self._now_monotonic()
        if now < self._next_rollback_attempt_not_before:
            return False
        deadline = min(
            self._rollback_deadline_at_monotonic or now,
            self._absolute_deadline_at_monotonic or now,
        )
        return now + self._milliseconds(
            self._safety_policy.rollback_submit_timeout_ms + self._safety_policy.order_eventual_consistency_grace_ms
        ) <= deadline

    def _schedule_next_hedge_attempt(self) -> None:
        attempt = self._hedge_attempt_count()
        if attempt < self._safety_policy.hedge_max_attempts:
            self._next_hedge_attempt_not_before = self._now_monotonic() + self._milliseconds(
                self._safety_policy.hedge_retry_backoff_ms[attempt - 1]
            )

    def _schedule_next_rollback_attempt(self) -> None:
        attempt = self._rollback_attempt_count()
        if attempt < self._safety_policy.rollback_max_attempts:
            self._next_rollback_attempt_not_before = self._now_monotonic() + self._milliseconds(
                self._safety_policy.rollback_retry_backoff_ms[attempt - 1]
            )

    async def _advance_safety_state_machine(self) -> None:
        """Run one non-blocking safety tick; all elapsed checks use monotonic time."""
        self._enforce_absolute_deadline()
        if self._submission_halted:
            await self._reconcile_submission_unknowns()
            self._enforce_absolute_deadline()
            return

        self._monitor_maker_safety()
        self._enforce_absolute_deadline()
        if self._submission_halted or self._exposure_started_at_monotonic is None:
            return

        now = self._now_monotonic()
        if (
            self._hedge_deadline_at_monotonic is not None
            and now >= self._hedge_deadline_at_monotonic
        ):
            self._rollback_required = True
            self._request_maker_cancel("hedge phase deadline reached")
        if self._hedge_attempt_count() >= self._safety_policy.hedge_max_attempts:
            self._rollback_required = True
            self._request_maker_cancel("hedge attempt budget exhausted")

        if self._rollback_required:
            if not self._maker_terminal:
                self._request_maker_cancel("unhedged exposure requires exact rollback")
                return
            self._start_or_continue_rollback()
            return

        if self.stock_pending_quantity == _ZERO:
            try:
                self._submit_incremental_stock_hedge()
            except Exception:
                self._submission_halted = True
                self._close_reason = "stock hedge setup or submission failed"
                self._transition_to_recovery_required("stock hedge setup or submission failed")

    def _monitor_maker_safety(self) -> None:
        maker = self._maker_intent
        if maker is None or self._maker_terminal or self._cancel_intent is not None:
            return
        now = self._now_monotonic()
        if (
            self._safety_policy.maker_max_age_ms is not None
            and self._maker_started_at_monotonic is not None
            and now - self._maker_started_at_monotonic >= self._milliseconds(self._safety_policy.maker_max_age_ms)
        ):
            self._request_maker_cancel("maker maximum age reached")
            return

        observation = self._maker_safety
        reason = None
        if observation.net_bp <= _ZERO:
            reason = "maker opportunity is nonpositive"
        elif not observation.direction_stable:
            reason = "maker direction flipped"
        elif not observation.model_valid:
            reason = "maker model or anchor became invalid"
        elif not observation.entry_permitted:
            reason = "maker entry time is no longer permitted"
        elif observation.target_etf_quantity <= self._hedged_etf_equivalent_quantity():
            reason = "maker target was removed or already hedged"
        elif observation.reprice_requested:
            reason = "maker reprice requested"
        if reason is not None:
            self._request_maker_cancel(reason)
            return

        if observation.revision == self._last_monitored_safety_revision:
            return
        self._last_monitored_safety_revision = observation.revision
        if observation.residual_bp is None:
            return
        current = abs(observation.residual_bp)
        if self._initial_abs_residual_bp is None:
            self._initial_abs_residual_bp = current
            self._best_abs_residual_bp = current
            self._current_abs_residual_bp = current
            self._divergence_consecutive_count = 0
            return
        self._current_abs_residual_bp = current
        assert self._best_abs_residual_bp is not None
        assert self._initial_abs_residual_bp is not None
        if current < self._best_abs_residual_bp:
            self._best_abs_residual_bp = current
            self._divergence_consecutive_count = 0
            return
        has_improved = self._best_abs_residual_bp < self._initial_abs_residual_bp
        is_diverging = (
            has_improved
            and current - self._best_abs_residual_bp >= self._safety_policy.divergence_cancel_bp
        )
        self._divergence_consecutive_count = (
            self._divergence_consecutive_count + 1 if is_diverging else 0
        )
        if self._divergence_consecutive_count >= self._safety_policy.divergence_confirmations:
            self._request_maker_cancel("maker residual diverged after convergence")

    def _request_maker_cancel(self, reason: str) -> None:
        maker = self._maker_intent
        if maker is None or self._maker_terminal or self._cancel_intent is not None:
            return
        remaining = maker.remaining_quantity
        if remaining <= _ZERO:
            self._maker_terminal = True
            maker.terminal = True
            return
        identity = self._new_identity(
            action=JournalSideEffect.CANCEL,
            leg="ETF",
            logical_quantity=maker.identity.order_quantity,
            order_quantity=remaining,
            attempt=1,
        )
        cancel = _CancelIntent(
            identity=identity,
            target=maker,
            requested_at_monotonic=self._now_monotonic(),
        )
        self._cancel_intent = cancel
        self._append_identity_event(
            event_type=JournalEventType.PREPARED,
            payload=PreparedJournalPayloadV1(identity=identity),
            identity=identity,
        )
        self._append_identity_event(
            event_type=JournalEventType.CANCEL_REQUESTED,
            payload=CancelJournalPayloadV1(
                identity=identity,
                phase="REQUESTED",
                target_intent_id=maker.identity.intent_id,
                target_client_order_id=maker.identity.client_order_id,
                target_exchange_order_id=maker.exchange_order_id,
                final_order_cumulative_filled_quantity=maker.filled_quantity,
            ),
            identity=identity,
        )
        self._state = LeveragedEtfPairState.MAKER_CANCEL_PENDING
        try:
            order_id = self.connectors[maker.identity.connector_name].cancel(
                maker.identity.trading_pair,
                maker.identity.client_order_id,
            )
        except Exception:
            self._mark_cancel_unknown(reason)
            return
        if order_id != maker.identity.client_order_id:
            self._mark_cancel_unknown("connector did not retain maker cancellation client id")

    def _mark_cancel_unknown(self, reason: str) -> None:
        cancel = self._cancel_intent
        if cancel is None or cancel.submission_unknown:
            return
        cancel.submission_unknown = True
        cancel.unknown_since_monotonic = self._now_monotonic()
        self._append_identity_event(
            event_type=JournalEventType.SUBMIT_UNKNOWN,
            payload=SubmitUnknownJournalPayloadV1(
                identity=cancel.identity,
                uncertainty_started_at_utc=self._event_clock,
            ),
            identity=cancel.identity,
        )
        self._submission_halted = True
        self._close_reason = reason
        self._state = LeveragedEtfPairState.RECONCILING

    def _confirm_maker_cancel(self) -> None:
        cancel = self._cancel_intent
        if cancel is None or cancel.confirmed:
            return
        maker = cancel.target
        self._bridge_late_fill_cancel_to_reconciliation()
        self._append_identity_event(
            event_type=JournalEventType.CANCEL_CONFIRMED,
            payload=CancelJournalPayloadV1(
                identity=cancel.identity,
                phase="CONFIRMED",
                target_intent_id=maker.identity.intent_id,
                target_client_order_id=maker.identity.client_order_id,
                target_exchange_order_id=maker.exchange_order_id,
                final_order_cumulative_filled_quantity=maker.filled_quantity,
            ),
            identity=cancel.identity,
        )
        cancel.confirmed = True
        maker.terminal = True
        self._maker_terminal = True
        self._append_identity_event(
            event_type=JournalEventType.RECONCILIATION,
            payload=ReconciliationJournalPayloadV1(
                identity=maker.identity,
                outcome=ReconciliationOutcome.CANCELED,
                exchange_order_id=maker.exchange_order_id,
                order_cumulative_filled_quantity=maker.filled_quantity,
            ),
            identity=maker.identity,
            exchange_order_id=maker.exchange_order_id,
        )
        if self._unhedged_etf_quantity() == _ZERO:
            self._state = LeveragedEtfPairState.ABORTED_NO_FILL
        else:
            self._state = LeveragedEtfPairState.STOCK_HEDGE_PENDING

    def _bridge_late_fill_cancel_to_reconciliation(self) -> None:
        """Use F004's legal reconciliation bridge for cancel/fill event races.

        A maker fill received while cancellation is pending correctly changes the
        durable snapshot to STOCK_HEDGE_PENDING.  F004 permits an explicit move
        from that state to RECONCILING, from which the already-prepared cancel
        side effect can be confirmed without discarding either factual event.
        """
        if self._state is LeveragedEtfPairState.STOCK_HEDGE_PENDING:
            self._append_state_transition(
                LeveragedEtfPairState.RECONCILING,
                "reconciling maker cancellation after a late ETF fill",
            )

    def _all_stock_intents_known_terminal(self) -> bool:
        return all(intent.terminal and not intent.submission_unknown for intent in self._stock_intents.values())

    def _start_or_continue_rollback(self) -> None:
        outstanding = self._unhedged_etf_quantity()
        if outstanding == _ZERO:
            self._refresh_exposure_episode()
            return
        if not self._all_stock_intents_known_terminal():
            if self._absolute_deadline_reached():
                self._submission_halted = True
                self._transition_to_recovery_required("stock intent remains non-terminal; rollback could create reverse exposure")
            return
        now = self._now_monotonic()
        if self._rollback_started_at_monotonic is None:
            self._rollback_started_at_monotonic = now
            self._rollback_deadline_at_monotonic = min(
                now + self._milliseconds(self._safety_policy.rollback_phase_deadline_ms),
                self._absolute_deadline_at_monotonic or now,
            )
        if any(not intent.terminal for intent in self._rollback_intents.values()):
            return
        if self._rollback_attempt_count() >= self._safety_policy.rollback_max_attempts:
            self._submission_halted = True
            self._transition_to_recovery_required("rollback attempt budget exhausted")
            return
        if not self._can_start_rollback_attempt():
            if self._rollback_deadline_reached() or self._absolute_deadline_reached():
                self._submission_halted = True
                self._transition_to_recovery_required("rollback phase has insufficient remaining time")
            return
        rollback_side = TradeType.BUY if self._etf_trade_side() is TradeType.SELL else TradeType.SELL
        rollback_price_type = PriceType.BestAsk if rollback_side is TradeType.BUY else PriceType.BestBid
        try:
            rollback_price = Decimal(
                self.get_price(self.config.etf_connector_name, self.config.etf_trading_pair, rollback_price_type)
            )
            legal_quantity = self._quantize_order_quantity_at_price(
                self.config.etf_connector_name,
                self.config.etf_trading_pair,
                outstanding,
                rollback_price,
            )
        except Exception:
            legal_quantity = _ZERO
        # Rollback is exact reduce-only.  Quantizing down, oversizing, or joining
        # another Executor's dust would turn a safety action into new exposure.
        if legal_quantity != outstanding:
            if self._rollback_deadline_reached() or self._absolute_deadline_reached():
                self._submission_halted = True
                self._transition_to_recovery_required("exact ETF rollback is below exchange minimum")
            return
        identity = self._new_identity(
            action=JournalSideEffect.ETF_ROLLBACK,
            leg="ETF",
            logical_quantity=outstanding,
            order_quantity=outstanding,
            attempt=self._rollback_attempt_count() + 1,
        )
        intent = _OrderIntent(identity=identity)
        self._rollback_intents[identity.client_order_id] = intent
        self._append_identity_event(
            event_type=JournalEventType.PREPARED,
            payload=PreparedJournalPayloadV1(identity=identity),
            identity=identity,
        )
        self._append_identity_event(
            event_type=JournalEventType.ROLLBACK_REQUESTED,
            payload=RollbackJournalPayloadV1(identity=identity, phase="REQUESTED"),
            identity=identity,
        )
        self._state = LeveragedEtfPairState.ETF_ROLLBACK_PENDING
        try:
            order_id = self._native_submit(
                connector_name=self.config.etf_connector_name,
                trading_pair=self.config.etf_trading_pair,
                side=rollback_side,
                amount=outstanding,
                order_type=OrderType.MARKET,
                price=_NAN,
                client_order_id=identity.client_order_id,
                position_action=PositionAction.CLOSE,
            )
        except Exception as exc:
            self._record_submission_exception(identity, exc, is_maker=False)
            self._schedule_next_rollback_attempt()
            return
        if order_id != identity.client_order_id:
            self._mark_submission_unknown(intent=intent, reason="connector did not retain rollback client id")
            return
        intent.submitted = True
        self._consume_submission_unknowns()

    def _absolute_deadline_reached(self) -> bool:
        return (
            self._absolute_deadline_at_monotonic is not None
            and self._now_monotonic() >= self._absolute_deadline_at_monotonic
        )

    def _rollback_deadline_reached(self) -> bool:
        return (
            self._rollback_deadline_at_monotonic is not None
            and self._now_monotonic() >= self._rollback_deadline_at_monotonic
        )

    def _enforce_absolute_deadline(self) -> None:
        if self._absolute_deadline_reached() and self._unhedged_etf_quantity() > _ZERO:
            self._submission_halted = True
            self._close_reason = "unhedged response deadline exceeded"
            self._transition_to_recovery_required(self._close_reason)
            return
        cancel = self._cancel_intent
        if (
            cancel is not None
            and cancel.submission_unknown
            and cancel.unknown_since_monotonic is not None
            and self._now_monotonic() >= cancel.unknown_since_monotonic + self._milliseconds(
                self._safety_policy.unhedged_response_deadline_ms
            )
        ):
            self._submission_halted = True
            self._close_reason = "maker cancellation remained unknown beyond reconciliation deadline"
            self._transition_to_recovery_required(self._close_reason)

    async def _reconcile_submission_unknowns(self) -> None:
        unknown_intents = tuple(
            intent
            for intent in (self._maker_intent, *self._stock_intents.values(), *self._rollback_intents.values())
            if intent is not None and intent.submission_unknown
        )
        cancel_unknown = self._cancel_intent is not None and self._cancel_intent.submission_unknown
        if not unknown_intents and not cancel_unknown:
            return
        for intent in unknown_intents:
            await self._reconcile_unknown_intent(intent)
            if self._state is LeveragedEtfPairState.RECOVERY_REQUIRED:
                return
        if cancel_unknown:
            await self._reconcile_unknown_cancel()
            if self._state is LeveragedEtfPairState.RECOVERY_REQUIRED:
                return
        unresolved = any(intent.submission_unknown for intent in unknown_intents)
        unresolved = unresolved or bool(self._cancel_intent is not None and self._cancel_intent.submission_unknown)
        if not unresolved:
            self._submission_halted = False

    async def _reconcile_unknown_intent(self, intent: _OrderIntent) -> None:
        if not self._unknown_sweep_is_due(intent.unknown_since_monotonic, intent.last_sweep_monotonic):
            return
        connector = self.connectors[intent.identity.connector_name]
        status_getter = getattr(connector, "get_order_status_by_client_order_id", None)
        trades_getter = getattr(connector, "get_account_trades", None)
        positions_getter = getattr(connector, "get_position_risk_snapshots", None)
        if not all(callable(method) for method in (status_getter, trades_getter, positions_getter)):
            return
        try:
            status = await self._maybe_await(
                status_getter(intent.identity.trading_pair, intent.identity.client_order_id)
            )
            exchange_order_id = self._reconciliation_exchange_order_id(status, intent)
            trades = await self._maybe_await(
                trades_getter(intent.identity.trading_pair, exchange_order_id)
            )
            positions = await self._maybe_await(positions_getter(intent.identity.trading_pair))
        except Exception:
            return
        status_name = self._reconciliation_status_name(status)
        if status_name is None:
            self._transition_to_recovery_required("order reconciliation returned an unsupported status")
            return
        try:
            executed_quantity = Decimal(getattr(status, "executed_quantity", _ZERO) or _ZERO)
        except Exception:
            self._transition_to_recovery_required("order reconciliation returned an invalid cumulative quantity")
            return
        if not executed_quantity.is_finite() or executed_quantity < _ZERO:
            self._transition_to_recovery_required("order reconciliation returned an invalid cumulative quantity")
            return
        if executed_quantity < intent.filled_quantity:
            self._transition_to_recovery_required("order status cumulative fill disagreed with durable trade facts")
            return
        if executed_quantity > intent.filled_quantity:
            if not self._record_missing_reconciled_fills(intent, exchange_order_id, executed_quantity, trades):
                # The journal cannot synthesize an un-attributed trade.  A
                # status total higher than its known trade facts remains a
                # contradiction until F002 supplies stable trade identities.
                self._transition_to_recovery_required("order status cumulative fill disagreed with durable trade facts")
                return
            if intent.terminal:
                return
        signature = self._reconciliation_signature(status_name, executed_quantity, exchange_order_id, trades, positions)
        intent.last_sweep_monotonic = self._now_monotonic()
        if status_name == "NOT_FOUND":
            if intent.filled_quantity != _ZERO or not self._positions_show_no_evidence(positions):
                self._transition_to_recovery_required("not-found reconciliation contradicts known order or position facts")
                return
            if intent.last_sweep_signature == signature:
                intent.reconciliation_sweeps += 1
            else:
                intent.last_sweep_signature = signature
                intent.reconciliation_sweeps = 1
            if intent.reconciliation_sweeps < 2:
                return
            self._append_reconciliation(
                intent,
                ReconciliationOutcome.CONSISTENT_NO_FILL,
                exchange_order_id=None,
            )
            intent.submission_unknown = False
            intent.terminal = True
            return
        if status_name in {"NEW", "PARTIALLY_FILLED"}:
            outcome = (
                ReconciliationOutcome.NEW
                if status_name == "NEW"
                else ReconciliationOutcome.PARTIALLY_FILLED
            )
            self._append_reconciliation(intent, outcome, exchange_order_id=exchange_order_id)
            intent.submission_unknown = False
            intent.acknowledged = True
            return
        outcomes = {
            "FILLED": ReconciliationOutcome.FILLED,
            "CANCELED": ReconciliationOutcome.CANCELED,
            "EXPIRED": ReconciliationOutcome.EXPIRED,
            "REJECTED": ReconciliationOutcome.REJECTED,
        }
        outcome = outcomes.get(status_name)
        if outcome is None:
            self._transition_to_recovery_required("order reconciliation returned an unsupported terminal status")
            return
        self._append_reconciliation(intent, outcome, exchange_order_id=exchange_order_id)
        intent.submission_unknown = False
        intent.terminal = True
        if intent.identity.action is JournalSideEffect.STOCK_HEDGE:
            self._schedule_next_hedge_attempt()
            self._rollback_required = self._hedge_attempt_count() >= self._safety_policy.hedge_max_attempts
        elif intent.identity.action is JournalSideEffect.ETF_ROLLBACK:
            self._schedule_next_rollback_attempt()
        elif intent.identity.action is JournalSideEffect.ETF_MAKER:
            self._maker_terminal = True

    def _record_missing_reconciled_fills(
        self,
        intent: _OrderIntent,
        exchange_order_id: Optional[str],
        executed_quantity: Decimal,
        trades,
    ) -> bool:
        """Apply only trade-ID-backed fills discovered during F002 reconciliation."""
        if exchange_order_id is None:
            return False
        action_key = intent.identity.action.value
        missing = []
        try:
            for trade in trades:
                trade_id = str(getattr(trade, "trade_id", "") or "")
                trade_order_id = str(getattr(trade, "exchange_order_id", "") or "")
                quantity = Decimal(getattr(trade, "quantity", _ZERO))
                price = Decimal(getattr(trade, "price", _ZERO))
                if trade_order_id != exchange_order_id or not trade_id or quantity <= _ZERO or price <= _ZERO:
                    return False
                if (action_key, trade_id) not in self._seen_trades:
                    missing.append((trade_id, quantity, price))
        except Exception:
            return False
        if intent.filled_quantity + sum((quantity for _, quantity, _ in missing), _ZERO) != executed_quantity:
            return False
        # F004 requires a reconciliation fact after SUBMIT_UNKNOWN before any
        # subsequent fill.  UNKNOWN records the prior zero/partial cumulative
        # fact without claiming terminality; the following trade IDs establish
        # the actual fills and may restore the ordinary hedge/rollback path.
        self._append_reconciliation(intent, ReconciliationOutcome.UNKNOWN, exchange_order_id=exchange_order_id)
        intent.submission_unknown = False
        intent.acknowledged = True
        if not any(
            candidate.submission_unknown
            for candidate in (self._maker_intent, *self._stock_intents.values(), *self._rollback_intents.values())
            if candidate is not None
        ):
            self._submission_halted = False
        for trade_id, quantity, price in missing:
            self._record_authoritative_fill(
                intent=intent,
                exchange_order_id=exchange_order_id,
                exchange_trade_id=trade_id,
                price=price,
                quantity=quantity,
            )
        return intent.filled_quantity == executed_quantity

    async def _reconcile_unknown_cancel(self) -> None:
        cancel = self._cancel_intent
        if cancel is None or not cancel.submission_unknown:
            return
        if not self._unknown_sweep_is_due(cancel.unknown_since_monotonic, cancel.last_sweep_monotonic):
            return
        connector = self.connectors[cancel.target.identity.connector_name]
        status_getter = getattr(connector, "get_order_status_by_client_order_id", None)
        trades_getter = getattr(connector, "get_account_trades", None)
        positions_getter = getattr(connector, "get_position_risk_snapshots", None)
        if not all(callable(method) for method in (status_getter, trades_getter, positions_getter)):
            return
        try:
            status = await self._maybe_await(
                status_getter(cancel.target.identity.trading_pair, cancel.target.identity.client_order_id)
            )
            exchange_order_id = self._reconciliation_exchange_order_id(status, cancel.target)
            trades = await self._maybe_await(
                trades_getter(cancel.target.identity.trading_pair, exchange_order_id)
            )
            positions = await self._maybe_await(positions_getter(cancel.target.identity.trading_pair))
        except Exception:
            return
        status_name = self._reconciliation_status_name(status)
        if status_name is None:
            self._transition_to_recovery_required("maker cancel reconciliation returned an unsupported status")
            return
        try:
            executed_quantity = Decimal(getattr(status, "executed_quantity", _ZERO) or _ZERO)
        except Exception:
            self._transition_to_recovery_required("maker cancel reconciliation returned an invalid cumulative quantity")
            return
        if executed_quantity != cancel.target.filled_quantity:
            self._transition_to_recovery_required("maker cancel status cumulative fill disagreed with durable trade facts")
            return
        if status_name == "FILLED":
            # A filled maker order proves that the cancel request did not
            # become a cancellation confirmation.  Do not write a misleading
            # CANCEL_CONFIRMED fact; the additive F004 three-state contract
            # will provide the authoritative terminal-resolution surface.
            self._transition_to_recovery_required(
                "maker cancellation reconciliation reported a filled maker order"
            )
            return
        if status_name in {"CANCELED", "EXPIRED", "REJECTED"}:
            self._bridge_late_fill_cancel_to_reconciliation()
            self._append_identity_event(
                event_type=JournalEventType.RECONCILIATION,
                payload=ReconciliationJournalPayloadV1(
                    identity=cancel.identity,
                    outcome=ReconciliationOutcome.CANCELED,
                    exchange_order_id=None,
                    order_cumulative_filled_quantity=_ZERO,
                ),
                identity=cancel.identity,
            )
            cancel.submission_unknown = False
            self._confirm_maker_cancel()
            return
        if status_name != "NOT_FOUND":
            cancel.last_sweep_monotonic = self._now_monotonic()
            return
        if cancel.target.filled_quantity != _ZERO or not self._positions_show_no_evidence(positions):
            self._transition_to_recovery_required("maker cancel not-found result contradicts known trade or position facts")
            return
        signature = self._reconciliation_signature(status_name, executed_quantity, exchange_order_id, trades, positions)
        cancel.last_sweep_monotonic = self._now_monotonic()
        if cancel.last_sweep_signature == signature:
            cancel.reconciliation_sweeps += 1
        else:
            cancel.last_sweep_signature = signature
            cancel.reconciliation_sweeps = 1
        if cancel.reconciliation_sweeps >= 2:
            self._bridge_late_fill_cancel_to_reconciliation()
            self._append_identity_event(
                event_type=JournalEventType.RECONCILIATION,
                payload=ReconciliationJournalPayloadV1(
                    identity=cancel.identity,
                    outcome=ReconciliationOutcome.CANCELED,
                    exchange_order_id=None,
                    order_cumulative_filled_quantity=_ZERO,
                ),
                identity=cancel.identity,
            )
            cancel.submission_unknown = False
            self._confirm_maker_cancel()

    def _unknown_sweep_is_due(self, unknown_since: Optional[float], last_sweep: Optional[float]) -> bool:
        if unknown_since is None:
            return False
        now = self._now_monotonic()
        if now < unknown_since + self._milliseconds(self._safety_policy.order_eventual_consistency_grace_ms):
            return False
        return last_sweep is None or now - last_sweep >= self._milliseconds(
            self._safety_policy.executor_safety_interval_ms
        )

    @staticmethod
    async def _maybe_await(value):
        return await value if inspect.isawaitable(value) else value

    @staticmethod
    def _reconciliation_status_name(status) -> Optional[str]:
        value = getattr(status, "status", None)
        value = getattr(value, "value", value)
        if not isinstance(value, str):
            return None
        normalized = "EXPIRED" if value == "EXPIRED_IN_MATCH" else value
        return normalized if normalized in {
            "NEW", "PARTIALLY_FILLED", "FILLED", "CANCELED", "EXPIRED", "REJECTED", "NOT_FOUND"
        } else None

    def _reconciliation_exchange_order_id(self, status, intent: _OrderIntent) -> Optional[str]:
        observed = getattr(status, "exchange_order_id", None)
        observed = None if observed is None else str(observed)
        if intent.exchange_order_id is not None and observed is not None and observed != intent.exchange_order_id:
            raise RuntimeError("reconciliation exchange order id disagreed with local identity")
        return observed or intent.exchange_order_id

    @staticmethod
    def _reconciliation_signature(status_name: str, executed: Decimal, exchange_order_id, trades, positions) -> Tuple:
        trade_signature = tuple(
            sorted(
                (
                    str(getattr(trade, "trade_id", "")),
                    str(getattr(trade, "exchange_order_id", "")),
                    str(getattr(trade, "quantity", "")),
                )
                for trade in trades
            )
        )
        position_signature = tuple(sorted(repr(position) for position in positions))
        return status_name, _canonical_decimal(executed), exchange_order_id, trade_signature, position_signature

    @staticmethod
    def _positions_show_no_evidence(positions) -> bool:
        # Position snapshots are a consistency check, never a mechanism for
        # assigning a fill to this Executor.  Any non-zero position fact keeps a
        # NOT_FOUND order unknown instead of licensing a guessed reverse order.
        for position in positions:
            for field_name in ("position_amount", "position_amt", "amount"):
                value = getattr(position, field_name, None)
                if value is not None:
                    try:
                        if Decimal(value) != _ZERO:
                            return False
                    except Exception:
                        return False
        return True

    def _append_reconciliation(
        self,
        intent: _OrderIntent,
        outcome: ReconciliationOutcome,
        *,
        exchange_order_id: Optional[str],
    ) -> None:
        self._append_identity_event(
            event_type=JournalEventType.RECONCILIATION,
            payload=ReconciliationJournalPayloadV1(
                identity=intent.identity,
                outcome=outcome,
                exchange_order_id=exchange_order_id,
                order_cumulative_filled_quantity=intent.filled_quantity,
            ),
            identity=intent.identity,
            exchange_order_id=exchange_order_id,
        )

    def _record_submission_exception(self, identity: SideEffectIdentityV1, exc: Exception, *, is_maker: bool):
        failure_kind = getattr(exc, "failure_kind", None)
        if getattr(failure_kind, "value", None) == "AUTHORITATIVE_REJECTION":
            self._append_identity_event(
                event_type=JournalEventType.REJECTED,
                payload=RejectedJournalPayloadV1(identity=identity, reason="authoritative submission rejection"),
                identity=identity,
            )
            intent = self._intent_for_client_order_id(identity.client_order_id)
            if intent is not None:
                intent.terminal = True
            if is_maker:
                self._state = LeveragedEtfPairState.PREFLIGHT
                self._submission_halted = True
            elif identity.action is JournalSideEffect.STOCK_HEDGE:
                self._state = LeveragedEtfPairState.STOCK_HEDGE_PENDING
                self._rollback_required = self._hedge_attempt_count() >= self._safety_policy.hedge_max_attempts
                self._schedule_next_hedge_attempt()
            else:
                self._state = LeveragedEtfPairState.ETF_ROLLBACK_PENDING
                self._submission_halted = True
        else:
            intent = self._intent_for_client_order_id(identity.client_order_id)
            if intent is None:
                raise RuntimeError("submission failure referenced an unknown local intent")
            self._mark_submission_unknown(intent=intent, reason="order submission failed")
            return
        self._close_reason = "order submission failed"

    def _acknowledge_intent(self, intent: _OrderIntent):
        if intent.acknowledged:
            return
        self._append_identity_event(
            event_type=JournalEventType.ACKNOWLEDGED,
            payload=AcknowledgedJournalPayloadV1(identity=intent.identity),
            identity=intent.identity,
        )
        intent.acknowledged = True

    def _apply_acknowledged_state(self, intent: _OrderIntent):
        if intent.identity.action is JournalSideEffect.ETF_MAKER:
            self._state = (
                LeveragedEtfPairState.MAKER_WORKING
                if self._etf_filled_quantity == _ZERO
                else LeveragedEtfPairState.STOCK_HEDGE_PENDING
            )
        elif intent.identity.action is JournalSideEffect.STOCK_HEDGE:
            self._state = LeveragedEtfPairState.STOCK_HEDGE_PENDING
        else:
            self._state = LeveragedEtfPairState.ETF_ROLLBACK_PENDING

    def _consume_submission_unknowns(self) -> bool:
        intents = tuple(
            intent
            for intent in (self._maker_intent, *self._stock_intents.values(), *self._rollback_intents.values())
            if intent is not None and intent.submitted and not intent.acknowledged and not intent.submission_unknown
        )
        for intent in intents:
            checker = getattr(self.connectors[intent.identity.connector_name], "is_order_submission_unknown", None)
            if not callable(checker):
                continue
            try:
                submission_unknown = bool(checker(intent.identity.client_order_id))
            except Exception:
                self._mark_submission_unknown(
                    intent=intent,
                    reason="unable to determine native order submission status",
                )
                return True
            if submission_unknown:
                self._mark_submission_unknown(
                    intent=intent,
                    reason="connector retained an unknown native order submission",
                )
                return True
        return self._submission_halted

    def _mark_submission_unknown(self, *, intent: _OrderIntent, reason: str):
        if intent.submission_unknown:
            return
        intent.submission_unknown = True
        intent.unknown_since_monotonic = self._now_monotonic()
        self._append_identity_event(
            event_type=JournalEventType.SUBMIT_UNKNOWN,
            payload=SubmitUnknownJournalPayloadV1(
                identity=intent.identity,
                uncertainty_started_at_utc=self._event_clock,
            ),
            identity=intent.identity,
        )
        self._submission_halted = True
        self._close_reason = reason
        self._state = LeveragedEtfPairState.RECONCILING

    def _has_unacknowledged_stock_submission(self) -> bool:
        return any(
            intent.submitted and not intent.acknowledged and not intent.submission_unknown and not intent.terminal
            for intent in self._stock_intents.values()
        )

    def _entry_operation_is_supported(self) -> bool:
        return self.config.operation in {
            LeveragedEtfPairOperation.OPEN,
            LeveragedEtfPairOperation.ADD,
        }

    def _halt_unsupported_operation(self):
        self._submission_halted = True
        self._close_reason = f"{self.config.operation.value} is outside the entry executor scope"
        self._transition_to_recovery_required(self._close_reason)

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
        position_action: PositionAction = PositionAction.OPEN,
    ) -> str:
        connector = self.connectors[connector_name]
        submit = connector.buy if side is TradeType.BUY else connector.sell
        return submit(
            trading_pair=trading_pair,
            amount=amount,
            order_type=order_type,
            price=price,
            client_order_id=client_order_id,
            position_action=position_action,
        )

    def _new_identity(
        self,
        *,
        action: JournalSideEffect,
        leg: str,
        logical_quantity: Decimal,
        order_quantity: Decimal,
        attempt: int = 1,
    ) -> SideEffectIdentityV1:
        logical_text = _canonical_decimal(logical_quantity)
        idempotency_key = f"{self.config.id}:{self.config.operation.value}:{leg}:{logical_text}:{attempt}"
        # The F004 idempotency-key grammar intentionally omits action.  Keep the
        # key canonical while deriving distinct intent/client identities for a
        # maker, its cancellation, and a rollback at the same logical amount.
        identity_seed = f"{idempotency_key}:{action.value}"
        digest = hashlib.sha256(identity_seed.encode("utf-8")).hexdigest()
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
            attempt=attempt,
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
        return self._stock_intents.get(client_order_id) or self._rollback_intents.get(client_order_id)

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

    def _quantize_order_quantity_at_price(
        self,
        connector_name: str,
        trading_pair: str,
        quantity: Decimal,
        price: Decimal,
    ) -> Decimal:
        if not price.is_finite() or price <= _ZERO:
            raise RuntimeError(f"invalid executable price for {connector_name}:{trading_pair}")
        rule = self._trading_rule(connector_name, trading_pair)
        minimum_notional = Decimal(rule.min_notional_size)
        if not minimum_notional.is_finite() or minimum_notional < _ZERO:
            raise RuntimeError(f"invalid notional rule for {connector_name}:{trading_pair}")
        quantized = self._quantize_order_quantity(connector_name, trading_pair, quantity)
        if quantized <= _ZERO:
            return _ZERO
        return quantized if quantized * price >= minimum_notional else _ZERO

    def _quantize_price(self, connector_name: str, trading_pair: str, price: Decimal) -> Decimal:
        rule = self._trading_rule(connector_name, trading_pair)
        increment = Decimal(rule.min_price_increment)
        if price <= _ZERO or increment <= _ZERO:
            raise RuntimeError(f"invalid price rule for {connector_name}:{trading_pair}")
        return (price / increment).to_integral_value(rounding=ROUND_DOWN) * increment

    def _current_executable_stock_price(self) -> Decimal:
        price_type = PriceType.BestAsk if self._stock_trade_side() is TradeType.BUY else PriceType.BestBid
        return Decimal(
            self.get_price(
                self.config.stock_connector_name,
                self.config.stock_trading_pair,
                price_type,
            )
        )

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
