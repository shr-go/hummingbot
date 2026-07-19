from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field

from hummingbot.core.data_type.common import TradeType
from hummingbot.strategy_v2.executors.arbitrage_executor.data_types import ArbitrageExecutorConfig
from hummingbot.strategy_v2.executors.dca_executor.data_types import DCAExecutorConfig
from hummingbot.strategy_v2.executors.grid_executor.data_types import GridExecutorConfig
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import LeveragedEtfPairExecutorConfig
from hummingbot.strategy_v2.executors.lp_executor.data_types import LPExecutorConfig
from hummingbot.strategy_v2.executors.order_executor.data_types import OrderExecutorConfig
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.executors.twap_executor.data_types import TWAPExecutorConfig
from hummingbot.strategy_v2.executors.xemm_executor.data_types import XEMMExecutorConfig
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors import CloseType

AnyExecutorConfig = Union[
    PositionExecutorConfig,
    DCAExecutorConfig,
    GridExecutorConfig,
    XEMMExecutorConfig,
    ArbitrageExecutorConfig,
    OrderExecutorConfig,
    TWAPExecutorConfig,
    LPExecutorConfig,
    LeveragedEtfPairExecutorConfig,
]


class ExecutorInfo(BaseModel):
    id: str
    timestamp: float
    type: str
    status: RunnableStatus
    config: AnyExecutorConfig = Field(..., discriminator="type")
    net_pnl_pct: Decimal
    net_pnl_quote: Decimal
    cum_fees_quote: Decimal
    filled_amount_quote: Decimal
    is_active: bool
    is_trading: bool
    custom_info: Dict
    close_timestamp: Optional[float] = None
    close_type: Optional[CloseType] = None
    controller_id: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def is_done(self):
        return self.status == RunnableStatus.TERMINATED

    @property
    def side(self) -> Optional[TradeType]:
        return self.custom_info.get("side", None)

    @property
    def trading_pair(self) -> Optional[str]:
        return self.trading_pairs[0] if self.trading_pairs else None

    @property
    def connector_name(self) -> Optional[str]:
        return self.connector_names[0] if self.connector_names else None

    @property
    def trading_pairs(self) -> Tuple[str, ...]:
        configured_pairs = getattr(self.config, "trading_pairs", None)
        if configured_pairs is not None:
            return tuple(configured_pairs)
        trading_pair = getattr(self.config, "trading_pair", None)
        if trading_pair is not None:
            return (trading_pair,)
        return tuple(
            market.trading_pair
            for market in (
                getattr(self.config, "buying_market", None),
                getattr(self.config, "selling_market", None),
            )
            if market is not None
        )

    @property
    def connector_names(self) -> Tuple[str, ...]:
        configured_names = getattr(self.config, "connector_names", None)
        if configured_names is not None:
            return tuple(configured_names)
        connector_name = getattr(self.config, "connector_name", None)
        if connector_name is not None:
            return (connector_name,)
        return tuple(
            market.connector_name
            for market in (
                getattr(self.config, "buying_market", None),
                getattr(self.config, "selling_market", None),
            )
            if market is not None
        )

    def to_dict(self):
        base_dict = self.model_dump()
        base_dict["side"] = self.side
        return base_dict


class PerformanceReport(BaseModel):
    realized_pnl_quote: Decimal = Decimal("0")
    unrealized_pnl_quote: Decimal = Decimal("0")
    unrealized_pnl_pct: Decimal = Decimal("0")
    realized_pnl_pct: Decimal = Decimal("0")
    global_pnl_quote: Decimal = Decimal("0")
    global_pnl_pct: Decimal = Decimal("0")
    volume_traded: Decimal = Decimal("0")
    positions_summary: List = []
    close_type_counts: Dict[CloseType, int] = {}
