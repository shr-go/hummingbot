"""F004-backed per-symbol leverage-reservation boundary.

The controller owns pair/symbol exclusivity.  F004 owns the canonical,
append-only reservation records and verifies that each record agrees with the
Executor snapshot it references.  The injected ``ensure_executor`` callback
is intentionally the only bridge to the Executor lifecycle: it must make the
F004 snapshot durable before this adapter writes the two leg reservations.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Protocol

from hummingbot.model.leveraged_etf_repository import (
    LeveragedEtfJournalRepository,
    ReservationIdentityPayloadV1,
    StrategyReservationV1,
)
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import (
    LeveragedEtfPairExecutorConfig,
)


class ReservationConflict(RuntimeError):
    """A reservation cannot safely be created or released."""


class ReservationStore(Protocol):
    """Small lifecycle interface used by the Controller and its tests."""

    def active(self) -> Sequence[object]:
        """Return every not-yet-released reservation, deterministically."""

    def reserve(self, executor_config: LeveragedEtfPairExecutorConfig) -> object:
        """Persist both non-zero legs of an exposure-increasing Executor."""

    def release(self, executor_id: str, released_at_utc: str) -> object:
        """Release every active reservation belonging to a terminal Executor."""


def _canonical_decimal(value: Decimal) -> str:
    if not isinstance(value, Decimal) or not value.is_finite() or value <= 0:
        raise ReservationConflict(
            "reservation quantity must be a finite positive Decimal"
        )
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


class F004ReservationStore:
    """Adapter over F004's journal repository with strategy-wide symbol locking.

    F004 enforces the durable row identity and its relationship to the
    Executor snapshot.  Its table intentionally permits distinct Executors to
    reference the same symbol, because it is generic persistence.  This
    Controller-owned adapter adds the portfolio invariant: a live symbol may
    be reserved by only one pair Executor.
    """

    def __init__(
        self,
        repository: LeveragedEtfJournalRepository,
        ensure_executor: Callable[[LeveragedEtfPairExecutorConfig], object],
    ) -> None:
        if not isinstance(repository, LeveragedEtfJournalRepository):
            raise TypeError("repository must be an F004 LeveragedEtfJournalRepository")
        if not callable(ensure_executor):
            raise TypeError("ensure_executor must be callable")
        self._repository = repository
        self._ensure_executor = ensure_executor

    def active(self) -> tuple[StrategyReservationV1, ...]:
        try:
            return self._repository.active_reservations()
        except Exception as error:
            raise ReservationConflict(
                f"could not load durable reservations: {error}"
            ) from error

    def reserve(
        self, executor_config: LeveragedEtfPairExecutorConfig
    ) -> tuple[StrategyReservationV1, ...]:
        if not isinstance(executor_config, LeveragedEtfPairExecutorConfig):
            raise TypeError(
                "executor_config must be an F004 LeveragedEtfPairExecutorConfig"
            )
        proposals = self._proposals(executor_config)
        if not proposals:
            return tuple()
        live_symbols = {
            (
                reservation.connector_name,
                reservation.trading_pair,
            ): reservation.executor_id
            for reservation in self.active()
        }
        for proposal in proposals:
            owner = live_symbols.get((proposal.connector_name, proposal.trading_pair))
            if owner is not None and owner != executor_config.id:
                raise ReservationConflict(
                    f"{proposal.connector_name}/{proposal.trading_pair} is already reserved by {owner}"
                )

        # F004 deliberately requires a snapshot first.  The caller supplies
        # that durable bootstrap from the native Executor creation boundary;
        # F006 never constructs or imports the F005 runtime.
        try:
            self._ensure_executor(executor_config)
        except Exception as error:
            raise ReservationConflict(
                f"could not durably bootstrap executor reservation: {error}"
            ) from error

        persisted: list[StrategyReservationV1] = []
        try:
            for proposal in proposals:
                persisted.append(self._repository.reserve(proposal))
        except Exception as error:
            # The journal has no multi-row reservation API.  Compensate the
            # first leg rather than leaving a create action with one-sided
            # capacity.  A compensation failure is still surfaced as fatal.
            timestamp = executor_config.model_dump(mode="json")["created_at_utc"]
            for reservation in persisted:
                try:
                    self._repository.release_reservation(
                        reservation.reservation_id, timestamp
                    )
                except Exception:
                    pass
            raise ReservationConflict(
                f"could not durably reserve both legs: {error}"
            ) from error
        return tuple(persisted)

    def release(
        self, executor_id: str, released_at_utc: str
    ) -> tuple[StrategyReservationV1, ...]:
        if not isinstance(executor_id, str) or not executor_id:
            raise TypeError("executor_id must be a non-empty string")
        released: list[StrategyReservationV1] = []
        try:
            for reservation in self._repository.active_reservations(executor_id):
                released.append(
                    self._repository.release_reservation(
                        reservation.reservation_id, released_at_utc
                    )
                )
        except Exception as error:
            raise ReservationConflict(
                f"could not release durable reservations for {executor_id}: {error}"
            ) from error
        return tuple(released)

    @staticmethod
    def _proposals(
        executor_config: LeveragedEtfPairExecutorConfig,
    ) -> tuple[StrategyReservationV1, ...]:
        reservation = executor_config.leverage_reservation
        created_at_utc = executor_config.model_dump(mode="json")["created_at_utc"]
        values = (
            (
                "ETF",
                reservation.etf_quantity,
                reservation.etf_leverage,
                reservation.etf_notional_cap,
                executor_config.etf_connector_name,
                executor_config.etf_trading_pair,
            ),
            (
                "STOCK",
                reservation.stock_quantity,
                reservation.stock_leverage,
                reservation.stock_notional_cap,
                executor_config.stock_connector_name,
                executor_config.stock_trading_pair,
            ),
        )
        proposals: list[StrategyReservationV1] = []
        for (
            leg,
            quantity,
            leverage,
            notional_cap,
            connector_name,
            trading_pair,
        ) in values:
            if quantity == 0:
                continue
            quantity_text = _canonical_decimal(quantity)
            reservation_key = f"{executor_config.id}:{leg}:{quantity_text}"
            proposals.append(
                StrategyReservationV1(
                    reservation_id=f"{executor_config.id}:{leg}:reservation",
                    executor_id=executor_config.id,
                    reservation_key=reservation_key,
                    connector_name=connector_name,
                    trading_pair=trading_pair,
                    leg=leg,
                    quantity=quantity,
                    leverage=leverage,
                    notional_cap=notional_cap,
                    payload=ReservationIdentityPayloadV1(
                        reservation_key=reservation_key,
                        executor_id=executor_config.id,
                        connector_name=connector_name,
                        trading_pair=trading_pair,
                        leg=leg,
                        logical_quantity=quantity,
                    ),
                    created_at_utc=created_at_utc,
                    updated_at_utc=created_at_utc,
                    released_at_utc=None,
                )
            )
        return tuple(proposals)
