import json
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from hummingbot.core.data_type.common import TradeType
from hummingbot.model.executors import Executors
from hummingbot.strategy_v2.executors.arbitrage_executor.data_types import ArbitrageExecutorConfig
from hummingbot.strategy_v2.executors.data_types import ConnectorPair
from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import (
    LeveragedEtfPairExecutorConfig,
    LeveragedEtfPairExecutorCustomInfoV1,
    LeveragedEtfPairExecutorReportV1,
    LeveragedEtfPairExecutorStateV1,
)
from hummingbot.strategy_v2.executors.position_executor.data_types import PositionExecutorConfig
from hummingbot.strategy_v2.models.base import RunnableStatus
from hummingbot.strategy_v2.models.executors_info import ExecutorInfo


def _contract_fixture(name: str) -> dict:
    relative_path = Path("contracts/equity_leveraged_etf/v1/round_trip_vectors.json")
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))["fixtures"][name]
    raise AssertionError(f"F001 contract vectors not found: {relative_path}")


def _pair_config(fixture: dict) -> LeveragedEtfPairExecutorConfig:
    return LeveragedEtfPairExecutorConfig(
        schema_version=fixture["schema_version"],
        id=fixture["executor_id"],
        timestamp=1721224862.0,
        controller_id=fixture["controller_id"],
        pair_id=fixture["pair_id"],
        nav_cycle_id=fixture["nav_cycle_id"],
        operation=fixture["operation"],
        direction=fixture["direction"],
        etf_connector_name=fixture["etf_connector_name"],
        etf_trading_pair=fixture["etf_trading_pair"],
        stock_connector_name=fixture["stock_connector_name"],
        stock_trading_pair=fixture["stock_trading_pair"],
        s0=fixture["s0"],
        l0=fixture["l0"],
        h=fixture["h"],
        created_raw_bp=fixture["created_raw_bp"],
        created_net_bp=fixture["created_net_bp"],
        target_gross_notional=fixture["target_gross_notional"],
        etf_target_quantity=fixture["etf_target_quantity"],
        stock_target_quantity=fixture["stock_target_quantity"],
        leverage_reservation=fixture["leverage_reservation"],
        config_hash=fixture["config_hash"],
        created_at_utc=fixture["created_at_utc"],
    )


def _pair_custom_info(fixture: dict) -> dict:
    state = LeveragedEtfPairExecutorStateV1(
        schema_version=fixture["schema_version"],
        executor_id=fixture["executor_id"],
        state=fixture["state"],
        etf_submitted_quantity=fixture["etf_submitted_quantity"],
        etf_filled_quantity=fixture["etf_filled_quantity"],
        etf_remaining_quantity=fixture["etf_remaining_quantity"],
        stock_submitted_quantity=fixture["stock_submitted_quantity"],
        stock_filled_quantity=fixture["stock_filled_quantity"],
        stock_remaining_quantity=fixture["stock_remaining_quantity"],
        hedge_dust_quantity=fixture["hedge_dust_quantity"],
        maker_order_ids=fixture["maker_order_ids"],
        stock_order_ids=fixture["stock_order_ids"],
        leverage_reservation=fixture["leverage_reservation"],
        updated_at_utc=fixture["updated_at_utc"],
        close_reason=fixture["close_reason"],
        last_journal_sequence=fixture["last_journal_sequence"],
    )
    return LeveragedEtfPairExecutorCustomInfoV1(
        state=state,
        report=LeveragedEtfPairExecutorReportV1(),
    ).model_dump(mode="json")


def _executor_info_payload(config, custom_info=None) -> dict:
    return {
        "id": config.id,
        "timestamp": config.timestamp,
        "type": config.type,
        "status": RunnableStatus.RUNNING,
        "config": config.model_dump(mode="json"),
        "net_pnl_pct": "0",
        "net_pnl_quote": "0",
        "cum_fees_quote": "0",
        "filled_amount_quote": "0",
        "is_active": True,
        "is_trading": True,
        "custom_info": custom_info or {},
        "controller_id": config.controller_id,
    }


def test_pair_executor_info_deserializes_through_static_union_and_json_round_trip():
    fixture = _contract_fixture("executor_active")
    config = _pair_config(fixture)

    info = ExecutorInfo.model_validate(_executor_info_payload(config, _pair_custom_info(fixture)))
    restored = ExecutorInfo.model_validate_json(info.model_dump_json())

    assert isinstance(info.config, LeveragedEtfPairExecutorConfig)
    assert isinstance(restored.config, LeveragedEtfPairExecutorConfig)
    assert restored.model_dump(mode="json") == info.model_dump(mode="json")
    assert info.connector_name == fixture["etf_connector_name"]
    assert info.trading_pair == fixture["etf_trading_pair"]
    assert info.connector_names == (
        fixture["etf_connector_name"],
        fixture["stock_connector_name"],
    )
    assert info.trading_pairs == (
        fixture["etf_trading_pair"],
        fixture["stock_trading_pair"],
    )


def test_pair_executor_survives_markets_recorder_row_shape():
    fixture = _contract_fixture("executor_active")
    config = _pair_config(fixture)
    info = ExecutorInfo.model_validate(_executor_info_payload(config, _pair_custom_info(fixture)))
    recorder_payload = json.loads(info.model_dump_json())

    restored = Executors(**recorder_payload).to_executor_info()

    assert isinstance(restored.config, LeveragedEtfPairExecutorConfig)
    assert restored.config == config
    assert restored.custom_info == info.custom_info
    assert restored.connector_names == info.connector_names
    assert restored.trading_pairs == info.trading_pairs


def test_single_leg_executor_info_accessors_remain_compatible():
    config = PositionExecutorConfig(
        id="position-1",
        timestamp=1.0,
        controller_id="controller-1",
        connector_name="binance_perpetual",
        trading_pair="ETH-USDT",
        side=TradeType.BUY,
        amount=Decimal("1"),
    )
    info = ExecutorInfo.model_validate(_executor_info_payload(config))

    assert info.connector_name == "binance_perpetual"
    assert info.trading_pair == "ETH-USDT"
    assert info.connector_names == ("binance_perpetual",)
    assert info.trading_pairs == ("ETH-USDT",)


def test_existing_multi_leg_executor_info_is_pair_aware():
    config = ArbitrageExecutorConfig(
        id="arbitrage-1",
        timestamp=1.0,
        controller_id="controller-1",
        buying_market=ConnectorPair(connector_name="binance", trading_pair="ETH-USDT"),
        selling_market=ConnectorPair(connector_name="kraken", trading_pair="ETH-USD"),
        order_amount=Decimal("1"),
        min_profitability=Decimal("0.001"),
    )
    info = ExecutorInfo.model_validate(_executor_info_payload(config))

    assert info.connector_name == "binance"
    assert info.trading_pair == "ETH-USDT"
    assert info.connector_names == ("binance", "kraken")
    assert info.trading_pairs == ("ETH-USDT", "ETH-USD")


def test_historical_position_executor_row_still_deserializes():
    payload = {
        "id": "historical-position-1",
        "timestamp": 1.0,
        "type": "position_executor",
        "status": RunnableStatus.TERMINATED,
        "config": {
            "id": "historical-position-1",
            "type": "position_executor",
            "timestamp": 1.0,
            "controller_id": "main",
            "trading_pair": "ETH-USDT",
            "connector_name": "binance_perpetual",
            "side": 1,
            "entry_price": "1000",
            "amount": "1",
            "triple_barrier_config": {
                "stop_loss": "0.02",
                "take_profit": "0.03",
                "time_limit": 60,
                "trailing_stop": None,
                "open_order_type": 2,
                "take_profit_order_type": 1,
                "stop_loss_order_type": 1,
                "time_limit_order_type": 1,
            },
            "leverage": 1,
            "activation_bounds": None,
            "level_id": None,
        },
        "net_pnl_pct": "0.01",
        "net_pnl_quote": "1",
        "cum_fees_quote": "0.1",
        "filled_amount_quote": "100",
        "is_active": False,
        "is_trading": False,
        "custom_info": {},
        "close_timestamp": 2.0,
        "close_type": None,
        "controller_id": "main",
    }

    info = ExecutorInfo.model_validate(payload)

    assert isinstance(info.config, PositionExecutorConfig)
    assert info.connector_names == ("binance_perpetual",)
    assert info.trading_pairs == ("ETH-USDT",)


def test_unknown_executor_discriminator_remains_rejected():
    fixture = _contract_fixture("executor_active")
    config = _pair_config(fixture)
    payload = _executor_info_payload(config)
    payload["config"]["type"] = "future_pair_executor"

    with pytest.raises(ValidationError, match="union_tag_invalid"):
        ExecutorInfo.model_validate(payload)
