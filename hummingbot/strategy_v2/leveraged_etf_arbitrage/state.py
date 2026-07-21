"""Immutable projected execution states used by the exact allocator."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class ProjectedStateName(str, Enum):
    CURRENT_RESERVATION = "CURRENT_RESERVATION"
    MAKER_OPEN = "MAKER_OPEN"
    ETF_ONE_LEG = "ETF_ONE_LEG"
    STOCK_PARTIAL = "STOCK_PARTIAL"
    SLICE_HEDGED = "SLICE_HEDGED"
    FINAL = "FINAL"


@dataclass(frozen=True, slots=True)
class ProjectedLegState:
    position_quantity: Decimal
    open_order_quantity: Decimal
    reservation_quantity: Decimal
    notional: Decimal
    initial_margin: Decimal
    maintenance_margin: Decimal


@dataclass(frozen=True, slots=True)
class ProjectedPairState:
    name: ProjectedStateName
    etf: ProjectedLegState
    stock: ProjectedLegState

    @property
    def initial_margin(self) -> Decimal:
        return self.etf.initial_margin + self.stock.initial_margin

    @property
    def maintenance_margin(self) -> Decimal:
        return self.etf.maintenance_margin + self.stock.maintenance_margin
