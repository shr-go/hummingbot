import asyncio
import functools
import json
import re
from dataclasses import FrozenInstanceError
from decimal import Decimal
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase
from typing import Any, Callable, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
from aioresponses.core import aioresponses
from bidict import bidict

import hummingbot.connector.derivative.binance_perpetual.binance_perpetual_constants as CONSTANTS
import hummingbot.connector.derivative.binance_perpetual.binance_perpetual_web_utils as web_utils
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_api_order_book_data_source import (
    BinancePerpetualAPIOrderBookDataSource,
)
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_derivative import BinancePerpetualDerivative
from hummingbot.connector.test_support.network_mocking_assistant import NetworkMockingAssistant
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.connector.utils import get_new_client_order_id
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderState
from hummingbot.core.data_type.limit_order import LimitOrder
from hummingbot.core.data_type.trade_fee import TokenAmount
from hummingbot.core.event.event_logger import EventLogger
from hummingbot.core.event.events import MarketEvent, OrderFilledEvent


class BinancePerpetualDerivativeUnitTest(IsolatedAsyncioWrapperTestCase):
    # the level is required to receive logs from the data source logger
    level = 0

    start_timestamp: float = pd.Timestamp("2021-01-01", tz="UTC").timestamp()

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.base_asset = "COINALPHA"
        cls.quote_asset = "HBOT"
        cls.trading_pair = f"{cls.base_asset}-{cls.quote_asset}"
        cls.symbol = f"{cls.base_asset}{cls.quote_asset}"
        cls.domain = CONSTANTS.TESTNET_DOMAIN
        cls.listen_key = "TEST_LISTEN_KEY"

    def setUp(self) -> None:
        super().setUp()
        self.log_records = []

        self.ws_sent_messages = []
        self.ws_incoming_messages = asyncio.Queue()
        self.resume_test_event = asyncio.Event()

        self.exchange = BinancePerpetualDerivative(
            binance_perpetual_api_key="testAPIKey",
            binance_perpetual_api_secret="testSecret",
            trading_pairs=[self.trading_pair],
            domain=self.domain,
        )

        if hasattr(self.exchange, "_time_synchronizer"):
            self.exchange._time_synchronizer.add_time_offset_ms_sample(0)
            self.exchange._time_synchronizer.logger().setLevel(1)
            self.exchange._time_synchronizer.logger().addHandler(self)

        BinancePerpetualAPIOrderBookDataSource._trading_pair_symbol_map = {
            self.domain: bidict({self.symbol: self.trading_pair})
        }

        self.exchange._set_current_timestamp(1640780000)
        self.exchange.logger().setLevel(1)
        self.exchange.logger().addHandler(self)
        self.exchange._order_tracker.logger().setLevel(1)
        self.exchange._order_tracker.logger().addHandler(self)
        self.mocking_assistant = NetworkMockingAssistant(self.local_event_loop)
        self.test_task: Optional[asyncio.Task] = None
        self.resume_test_event = asyncio.Event()
        self._initialize_event_loggers()

    @property
    def all_symbols_url(self):
        url = web_utils.public_rest_url(path_url=CONSTANTS.EXCHANGE_INFO_URL)
        return url

    @property
    def latest_prices_url(self):
        url = web_utils.public_rest_url(
            path_url=CONSTANTS.TICKER_PRICE_CHANGE_URL
        )
        url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?") + ".*")
        return url

    @property
    def network_status_url(self):
        url = web_utils.public_rest_url(path_url=CONSTANTS.PING_URL)
        return url

    @property
    def trading_rules_url(self):
        url = web_utils.public_rest_url(path_url=CONSTANTS.EXCHANGE_INFO_URL)
        return url

    @property
    def balance_url(self):
        url = web_utils.private_rest_url(path_url=CONSTANTS.ACCOUNT_INFO_URL)
        return url

    @property
    def funding_info_url(self):
        url = web_utils.public_rest_url(
            path_url=CONSTANTS.TICKER_PRICE_CHANGE_URL
        )
        url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?") + ".*")
        return url

    @property
    def funding_payment_url(self):
        url = web_utils.private_rest_url(
            path_url=CONSTANTS.GET_INCOME_HISTORY_URL
        )
        url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?") + ".*")
        return url

    def tearDown(self) -> None:
        self.test_task and self.test_task.cancel()
        BinancePerpetualAPIOrderBookDataSource._trading_pair_symbol_map = {}
        super().tearDown()

    def _initialize_event_loggers(self):
        self.buy_order_completed_logger = EventLogger()
        self.sell_order_completed_logger = EventLogger()
        self.order_cancelled_logger = EventLogger()
        self.order_filled_logger = EventLogger()
        self.funding_payment_completed_logger = EventLogger()

        events_and_loggers = [
            (MarketEvent.BuyOrderCompleted, self.buy_order_completed_logger),
            (MarketEvent.SellOrderCompleted, self.sell_order_completed_logger),
            (MarketEvent.OrderCancelled, self.order_cancelled_logger),
            (MarketEvent.OrderFilled, self.order_filled_logger),
            (MarketEvent.FundingPaymentCompleted, self.funding_payment_completed_logger)]

        for event, logger in events_and_loggers:
            self.exchange.add_listener(event, logger)

    def handle(self, record):
        self.log_records.append(record)

    def _is_logged(self, log_level: str, message: str) -> bool:
        return any(record.levelname == log_level and record.getMessage() == message for record in self.log_records)

    def _create_exception_and_unlock_test_with_event(self, exception):
        self.resume_test_event.set()
        raise exception

    def _return_calculation_and_set_done_event(self, calculation: Callable, *args, **kwargs):
        if self.resume_test_event.is_set():
            raise asyncio.CancelledError
        self.resume_test_event.set()
        return calculation(*args, **kwargs)

    def _get_position_risk_api_endpoint_single_position_list(self) -> List[Dict[str, Any]]:
        positions = [
            {
                "symbol": self.symbol,
                "positionAmt": "1",
                "entryPrice": "10",
                "markPrice": "11",
                "unRealizedProfit": "1",
                "liquidationPrice": "100",
                "leverage": "1",
                "maxNotionalValue": "9",
                "marginType": "cross",
                "isolatedMargin": "0",
                "isAutoAddMargin": "false",
                "positionSide": "BOTH",
                "notional": "11",
                "isolatedWallet": "0",
                "updateTime": int(self.start_timestamp),
            }
        ]
        return positions

    def _get_wrong_symbol_position_risk_api_endpoint_single_position_list(self) -> List[Dict[str, Any]]:
        positions = [
            {
                "symbol": f"{self.symbol}_230331",
                "positionAmt": "1",
                "entryPrice": "10",
                "markPrice": "11",
                "unRealizedProfit": "1",
                "liquidationPrice": "100",
                "leverage": "1",
                "maxNotionalValue": "9",
                "marginType": "cross",
                "isolatedMargin": "0",
                "isAutoAddMargin": "false",
                "positionSide": "BOTH",
                "notional": "11",
                "isolatedWallet": "0",
                "updateTime": int(self.start_timestamp),
            }
        ]
        return positions

    def _get_account_update_ws_event_single_position_dict(self) -> Dict[str, Any]:
        account_update = {
            "e": "ACCOUNT_UPDATE",
            "E": 1564745798939,
            "T": 1564745798938,
            "a": {
                "m": "POSITION",
                "B": [
                    {"a": "USDT", "wb": "122624.12345678", "cw": "100.12345678", "bc": "50.12345678"},
                ],
                "P": [
                    {
                        "s": self.symbol,
                        "pa": "1",
                        "ep": "10",
                        "cr": "200",
                        "up": "1",
                        "mt": "cross",
                        "iw": "0.00000000",
                        "ps": "BOTH",
                    },
                ],
            },
        }
        return account_update

    def _get_wrong_symbol_account_update_ws_event_single_position_dict(self) -> Dict[str, Any]:
        account_update = {
            "e": "ACCOUNT_UPDATE",
            "E": 1564745798939,
            "T": 1564745798938,
            "a": {
                "m": "POSITION",
                "B": [
                    {"a": "USDT", "wb": "122624.12345678", "cw": "100.12345678", "bc": "50.12345678"},
                ],
                "P": [
                    {
                        "s": f"{self.symbol}_230331",
                        "pa": "1",
                        "ep": "10",
                        "cr": "200",
                        "up": "1",
                        "mt": "cross",
                        "iw": "0.00000000",
                        "ps": "BOTH",
                    },
                ],
            },
        }
        return account_update

    def _get_income_history_dict(self) -> List:
        income_history = [{
            "income": 1,
            "symbol": self.symbol,
            "time": self.start_timestamp,
        }]
        return income_history

    def _get_funding_info_dict(self) -> Dict[str, Any]:
        funding_info = {
            "indexPrice": 1000,
            "markPrice": 1001,
            "nextFundingTime": self.start_timestamp + 8 * 60 * 60,
            "lastFundingRate": 1010
        }
        return funding_info

    def _get_reconciliation_order(
            self,
            client_order_id: str = "exec-sndk-snxx-0001-stock-0",
            exchange_order_id: int = 8886774,
            status: str = "PARTIALLY_FILLED",
            executed_quantity: str = "0.125",
    ) -> Dict[str, Any]:
        execution_price = Decimal("10000.125")
        executed = Decimal(executed_quantity)
        return {
            "avgPrice": "0" if executed == 0 else f"{execution_price:f}",
            "clientOrderId": client_order_id,
            "cumQuote": f"{execution_price * executed:f}",
            "executedQty": executed_quantity,
            "orderId": exchange_order_id,
            "origQty": "1.250",
            "origType": "LIMIT",
            "price": "10000.125",
            "reduceOnly": False,
            "side": "SELL",
            "positionSide": "BOTH",
            "status": status,
            "closePosition": False,
            "symbol": self.symbol,
            "time": 1700000000000,
            "timeInForce": "GTC",
            "type": "LIMIT",
            "updateTime": 1700000000123,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": False,
        }

    def _track_submission_unknown_order(
            self,
            client_order_id: str,
            exchange_order_id: Optional[str] = None,
            trade_type: TradeType = TradeType.SELL,
            order_type: OrderType = OrderType.LIMIT,
    ) -> InFlightOrder:
        self.exchange.start_tracking_order(
            order_id=client_order_id,
            exchange_order_id=exchange_order_id,
            trading_pair=self.trading_pair,
            trade_type=trade_type,
            price=Decimal("0") if order_type is OrderType.MARKET else Decimal("10000.125"),
            amount=Decimal("1.250"),
            order_type=order_type,
            leverage=1,
            position_action=PositionAction.OPEN,
        )
        self.exchange._unknown_submission_order_ids.add(client_order_id)
        return self.exchange.in_flight_orders[client_order_id]

    def _submission_unknown_user_event(
            self,
            client_order_id: str,
            exchange_order_id: int = 8886774,
            status: str = "NEW",
            side: str = "SELL",
            order_type: str = "LIMIT",
    ) -> Dict[str, Any]:
        return {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1700000000124,
            "T": 1700000000123,
            "o": {
                "s": self.symbol,
                "c": client_order_id,
                "S": side,
                "o": order_type,
                "f": "GTC",
                "q": "1.250",
                "p": "0" if order_type == "MARKET" else "10000.125",
                "ap": "0",
                "x": "NEW",
                "X": status,
                "i": exchange_order_id,
                "l": "0",
                "z": "0",
                "L": "0",
                "N": self.quote_asset,
                "n": "0",
                "T": 1700000000123,
                "t": 0,
                "m": False,
                "R": False,
                "ps": "BOTH",
                "cp": False,
                "rp": "0",
            },
        }

    def _submission_unknown_fill_event(
            self,
            client_order_id: str,
            status: str,
            last_fill_quantity: str,
            cumulative_quantity: str,
            trade_id: int,
            cumulative_quote: Optional[str] = None,
            average_price: Optional[str] = None,
            fill_price: str = "10000.125",
            order_type: str = "LIMIT",
    ) -> Dict[str, Any]:
        parsed_fill_price = Decimal(fill_price)
        event = self._submission_unknown_user_event(
            client_order_id=client_order_id,
            order_type=order_type,
        )
        event["o"].update({
            "ap": average_price or f"{parsed_fill_price:f}",
            "x": "TRADE",
            "X": status,
            "l": last_fill_quantity,
            "z": cumulative_quantity,
            "L": f"{parsed_fill_price:f}",
            "t": trade_id,
        })
        if cumulative_quote is not None:
            event["o"]["Z"] = cumulative_quote
        return event

    async def _assert_submission_unknown_native_multi_price_lifecycle(
            self,
            client_order_id: str,
            order_type: OrderType,
    ) -> None:
        order_type_value = "MARKET" if order_type is OrderType.MARKET else "LIMIT"
        tracked_order = self._track_submission_unknown_order(
            client_order_id=client_order_id,
            order_type=order_type,
        )
        first_partial = self._submission_unknown_fill_event(
            client_order_id=client_order_id,
            status="PARTIALLY_FILLED",
            last_fill_quantity="0.100",
            cumulative_quantity="0.100",
            trade_id=1,
            fill_price="10000.125",
            average_price="10000.125",
            order_type=order_type_value,
        )
        self.assertNotIn("Z", first_partial["o"])
        await self.exchange._process_user_stream_event(first_partial)

        self.exchange._unknown_submission_order_ids.add(client_order_id)
        second_partial = self._submission_unknown_fill_event(
            client_order_id=client_order_id,
            status="PARTIALLY_FILLED",
            last_fill_quantity="0.200",
            cumulative_quantity="0.300",
            trade_id=2,
            fill_price="10000.126",
            average_price="10000.12566667",
            order_type=order_type_value,
        )
        self.assertNotIn("Z", second_partial["o"])
        await self.exchange._process_user_stream_event(second_partial)
        before_duplicate = (
            tracked_order.current_state,
            tracked_order.executed_amount_base,
            tracked_order.executed_amount_quote,
            tuple(tracked_order.order_fills),
        )

        self.exchange._unknown_submission_order_ids.add(client_order_id)
        await self.exchange._process_user_stream_event(second_partial)
        self.assertEqual(
            before_duplicate,
            (
                tracked_order.current_state,
                tracked_order.executed_amount_base,
                tracked_order.executed_amount_quote,
                tuple(tracked_order.order_fills),
            ),
        )

        self.exchange._unknown_submission_order_ids.add(client_order_id)
        filled = self._submission_unknown_fill_event(
            client_order_id=client_order_id,
            status="FILLED",
            last_fill_quantity="0.950",
            cumulative_quantity="1.250",
            trade_id=3,
            fill_price="10000.127",
            average_price="10000.12668",
            order_type=order_type_value,
        )
        self.assertNotIn("Z", filled["o"])
        await self.exchange._process_user_stream_event(filled)

        self.assertEqual(OrderState.FILLED, tracked_order.current_state)
        self.assertEqual(Decimal("1.250"), tracked_order.executed_amount_base)
        self.assertEqual(Decimal("12500.158350"), tracked_order.executed_amount_quote)
        self.assertEqual(3, len(tracked_order.order_fills))
        self.assertFalse(self.exchange.is_order_submission_unknown(client_order_id))
        self.assertEqual("0" if order_type is OrderType.MARKET else "10000.125", filled["o"]["p"])

    def _get_reconciliation_trade(
            self,
            trade_id: int = 698759,
            exchange_order_id: int = 8886774,
            price: str = "10000.125",
    ) -> Dict[str, Any]:
        return {
            "buyer": False,
            "commission": "0.00001234",
            "commissionAsset": self.quote_asset,
            "id": trade_id,
            "maker": False,
            "orderId": exchange_order_id,
            "price": price,
            "qty": "0.125",
            "quoteQty": "1250.015625",
            "realizedPnl": "-0.00000001",
            "side": "SELL",
            "positionSide": "BOTH",
            "symbol": self.symbol,
            "time": 1700000000456,
        }

    def _get_reconciliation_position(self, position_amount: str = "-0.125") -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "positionSide": "BOTH",
            "positionAmt": position_amount,
            "entryPrice": "10000.125",
            "breakEvenPrice": "10000.12509872",
            "markPrice": "10001.25",
            "unRealizedProfit": "-0.140625",
            "liquidationPrice": "12000.5",
            "isolatedMargin": "0",
            "notional": "-1250.15625",
            "marginAsset": self.quote_asset,
            "isolatedWallet": "0",
            "initialMargin": "62.5078125",
            "maintMargin": "5.000625",
            "positionInitialMargin": "62.5078125",
            "openOrderInitialMargin": "0",
            "adl": 1,
            "bidNotional": "0",
            "askNotional": "0",
            "updateTime": 1700000000789,
        }

    def _get_trading_pair_symbol_map(self) -> Dict[str, str]:
        trading_pair_symbol_map = {self.symbol: f"{self.base_asset}-{self.quote_asset}"}
        return trading_pair_symbol_map

    def _get_exchange_info_mock_response(
            self,
            margin_asset: str = "HBOT",
            min_order_size: float = 1,
            min_price_increment: float = 2,
            min_base_amount_increment: float = 3,
            min_notional_size: float = 4,
    ) -> Dict[str, Any]:
        mocked_exchange_info = {  # irrelevant fields removed
            "symbols": [
                {
                    "symbol": self.symbol,
                    "pair": self.symbol,
                    "contractType": "PERPETUAL",
                    "baseAsset": self.base_asset,
                    "quoteAsset": self.quote_asset,
                    "marginAsset": margin_asset,
                    "status": "TRADING",
                    "filters": [
                        {
                            "filterType": "PRICE_FILTER",
                            "maxPrice": "300",
                            "minPrice": "0.0001",
                            "tickSize": str(min_price_increment),
                        },
                        {
                            "filterType": "LOT_SIZE",
                            "maxQty": "10000000",
                            "minQty": str(min_order_size),
                            "stepSize": str(min_base_amount_increment),
                        },
                        {
                            "filterType": "MIN_NOTIONAL",
                            "notional": str(min_notional_size),
                        },
                    ],
                }
            ],
        }

        return mocked_exchange_info

    def _get_exchange_info_error_mock_response(
            self,
            margin_asset: str = "HBOT",
            min_order_size: float = 1,
            min_price_increment: float = 2,
            min_base_amount_increment: float = 3,
            min_notional_size: float = 4,
    ) -> Dict[str, Any]:
        mocked_exchange_info = {  # irrelevant fields removed
            "symbols": [
                {
                    "symbol": self.symbol,
                    "pair": self.symbol,
                    "contractType": "PERPETUAL",
                    "baseAsset": self.base_asset,
                    "quoteAsset": self.quote_asset,
                    "marginAsset": margin_asset,
                    "status": "TRADING",
                }
            ],
        }

        return mocked_exchange_info

    @aioresponses()
    async def test_existing_account_position_detected_on_positions_update(self, req_mock):
        self._simulate_trading_rules_initialized()

        url = web_utils.private_rest_url(
            CONSTANTS.POSITION_INFORMATION_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        positions = self._get_position_risk_api_endpoint_single_position_list()
        req_mock.get(regex_url, body=json.dumps(positions))

        await self.exchange._update_positions()

        self.assertEqual(len(self.exchange.account_positions), 1)
        pos = list(self.exchange.account_positions.values())[0]
        self.assertEqual(pos.trading_pair.replace("-", ""), self.symbol)

    @aioresponses()
    async def test_wrong_symbol_position_detected_on_positions_update(self, req_mock):
        self._simulate_trading_rules_initialized()

        url = web_utils.private_rest_url(
            CONSTANTS.POSITION_INFORMATION_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        positions = self._get_wrong_symbol_position_risk_api_endpoint_single_position_list()
        req_mock.get(regex_url, body=json.dumps(positions))

        await self.exchange._update_positions()

        self.assertEqual(len(self.exchange.account_positions), 0)

    @aioresponses()
    async def test_account_position_updated_on_positions_update(self, req_mock):
        self._simulate_trading_rules_initialized()
        url = web_utils.private_rest_url(
            CONSTANTS.POSITION_INFORMATION_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        positions = self._get_position_risk_api_endpoint_single_position_list()
        req_mock.get(regex_url, body=json.dumps(positions))

        await self.exchange._update_positions()

        self.assertEqual(len(self.exchange.account_positions), 1)
        pos = list(self.exchange.account_positions.values())[0]
        self.assertEqual(pos.amount, 1)

        positions[0]["positionAmt"] = "2"
        req_mock.get(regex_url, body=json.dumps(positions))
        await self.exchange._update_positions()

        pos = list(self.exchange.account_positions.values())[0]
        self.assertEqual(pos.amount, 2)

    @aioresponses()
    async def test_new_account_position_detected_on_positions_update(self, req_mock):
        self._simulate_trading_rules_initialized()
        url = web_utils.private_rest_url(
            CONSTANTS.POSITION_INFORMATION_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.get(regex_url, body=json.dumps([]))

        await self.exchange._update_positions()

        self.assertEqual(len(self.exchange.account_positions), 0)

        positions = self._get_position_risk_api_endpoint_single_position_list()
        req_mock.get(regex_url, body=json.dumps(positions))
        await self.exchange._update_positions()

        self.assertEqual(len(self.exchange.account_positions), 1)

    @aioresponses()
    async def test_closed_account_position_removed_on_positions_update(self, req_mock):
        self._simulate_trading_rules_initialized()
        url = web_utils.private_rest_url(
            CONSTANTS.POSITION_INFORMATION_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        positions = self._get_position_risk_api_endpoint_single_position_list()
        req_mock.get(regex_url, body=json.dumps(positions))

        await self.exchange._update_positions()

        self.assertEqual(len(self.exchange.account_positions), 1)

        positions[0]["positionAmt"] = "0"
        req_mock.get(regex_url, body=json.dumps(positions))
        await self.exchange._update_positions()

        self.assertEqual(len(self.exchange.account_positions), 0)

    async def test_supported_position_modes(self):
        linear_connector = self.exchange
        expected_result = [PositionMode.ONEWAY, PositionMode.HEDGE]
        self.assertEqual(expected_result, linear_connector.supported_position_modes())

    @aioresponses()
    async def test_set_position_mode_change_successful(self, mock_api):
        self._simulate_trading_rules_initialized()

        url = web_utils.private_rest_url(CONSTANTS.CHANGE_POSITION_MODE_URL, domain=self.domain)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))
        get_position_mode_response = {"dualSidePosition": False}  # One-way Mode
        post_position_mode_response = {"code": 200, "msg": "success"}
        mock_api.get(regex_url, body=json.dumps(get_position_mode_response))
        mock_api.post(regex_url, body=json.dumps(post_position_mode_response))

        success, msg = await self.exchange._trading_pair_position_mode_set(PositionMode.HEDGE, self.trading_pair)

        self.assertTrue(success)

    @aioresponses()
    async def test_set_position_initial_mode_unchanged(self, mock_api):
        url = web_utils.private_rest_url(CONSTANTS.CHANGE_POSITION_MODE_URL, domain=self.domain)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))
        get_position_mode_response = {"dualSidePosition": False}  # One-way Mode
        mock_api.get(regex_url, body=json.dumps(get_position_mode_response))

        success, msg = await self.exchange._trading_pair_position_mode_set(PositionMode.ONEWAY, self.trading_pair)

        self.assertTrue(success)

    @aioresponses()
    async def test_set_position_mode_diff_initial_mode_change_successful(self, mock_api):
        url = web_utils.private_rest_url(CONSTANTS.CHANGE_POSITION_MODE_URL, domain=self.domain)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))
        get_position_mode_response = {"dualSidePosition": False}  # One-way Mode
        post_position_mode_response = {"code": 200, "msg": "success"}

        mock_api.get(regex_url, body=json.dumps(get_position_mode_response))
        mock_api.post(regex_url, body=json.dumps(post_position_mode_response))

        success, msg = await self.exchange._trading_pair_position_mode_set(PositionMode.HEDGE, self.trading_pair)

        self.assertTrue(success)

    @aioresponses()
    async def test_set_position_mode_diff_initial_mode_change_fail(self, mock_api):
        url = web_utils.private_rest_url(CONSTANTS.CHANGE_POSITION_MODE_URL, domain=self.domain)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))
        get_position_mode_response = {"dualSidePosition": False}  # One-way Mode
        post_position_mode_response = {"code": -4059, "msg": "No need to change position side."}

        mock_api.get(regex_url, body=json.dumps(get_position_mode_response))
        mock_api.post(regex_url, body=json.dumps(post_position_mode_response))

        success, msg = await self.exchange._trading_pair_position_mode_set(PositionMode.HEDGE, self.trading_pair)

        self.assertFalse(success)

    @aioresponses()
    async def test_initialize_position_mode_success(self, mock_api):
        url = web_utils.private_rest_url(CONSTANTS.CHANGE_POSITION_MODE_URL, domain=self.domain)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))
        mock_api.get(regex_url, body=json.dumps({"dualSidePosition": True}))

        await self.exchange._initialize_position_mode()

        self.assertEqual(PositionMode.HEDGE, self.exchange.position_mode)

    @aioresponses()
    async def test_initialize_position_mode_exception(self, mock_api):
        url = web_utils.private_rest_url(CONSTANTS.CHANGE_POSITION_MODE_URL, domain=self.domain)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))
        mock_api.get(regex_url, exception=Exception("API error"))

        await self.exchange._initialize_position_mode()

        self.assertEqual(PositionMode.ONEWAY, self.exchange.position_mode)
        self.assertTrue(
            self._is_logged("WARNING", "Could not fetch position mode from exchange. Using default.")
        )

    async def test_format_trading_rules(self):
        margin_asset = self.quote_asset
        min_order_size = 1
        min_price_increment = 2
        min_base_amount_increment = 3
        min_notional_size = 4
        mocked_response = self._get_exchange_info_mock_response(
            margin_asset, min_order_size, min_price_increment, min_base_amount_increment, min_notional_size
        )
        self._simulate_trading_rules_initialized()
        trading_rules = await self.exchange._format_trading_rules(mocked_response)

        self.assertEqual(1, len(trading_rules))

        trading_rule = trading_rules[0]

        self.assertEqual(min_order_size, trading_rule.min_order_size)
        self.assertEqual(min_price_increment, trading_rule.min_price_increment)
        self.assertEqual(min_base_amount_increment, trading_rule.min_base_amount_increment)
        self.assertEqual(min_notional_size, trading_rule.min_notional_size)
        self.assertEqual(margin_asset, trading_rule.buy_order_collateral_token)
        self.assertEqual(margin_asset, trading_rule.sell_order_collateral_token)

    async def test_format_trading_rules_exception(self):
        margin_asset = self.quote_asset
        min_order_size = 1
        min_price_increment = 2
        min_base_amount_increment = 3
        min_notional_size = 4
        mocked_response = self._get_exchange_info_error_mock_response(
            margin_asset, min_order_size, min_price_increment, min_base_amount_increment, min_notional_size
        )
        self._simulate_trading_rules_initialized()

        await self.exchange._format_trading_rules(mocked_response)
        self.assertTrue(self._is_logged(
            "ERROR",
            f"Error parsing the trading pair rule {mocked_response['symbols'][0]}. Error: 'filters'. Skipping..."
        ))

    async def test_get_collateral_token(self):
        margin_asset = self.quote_asset
        self._simulate_trading_rules_initialized()

        self.assertEqual(margin_asset, self.exchange.get_buy_collateral_token(self.trading_pair))
        self.assertEqual(margin_asset, self.exchange.get_sell_collateral_token(self.trading_pair))

    async def test_buy_order_fill_event_takes_fee_from_update_event(self):
        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        order = self.exchange.in_flight_orders.get("OID1")

        partial_fill = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1568879465651,
            "T": 1568879465650,
            "o": {
                "s": self.trading_pair,
                "c": order.client_order_id,
                "S": "BUY",
                "o": "TRAILING_STOP_MARKET",
                "f": "GTC",
                "q": "1",
                "p": "10000",
                "ap": "0",
                "sp": "7103.04",
                "x": "TRADE",
                "X": "PARTIALLY_FILLED",
                "i": 8886774,
                "l": "0.1",
                "z": "0.1",
                "L": "10000",
                "N": "HBOT",
                "n": "20",
                "T": 1568879465651,
                "t": 1,
                "b": "0",
                "a": "9.91",
                "m": False,
                "R": False,
                "wt": "CONTRACT_PRICE",
                "ot": "TRAILING_STOP_MARKET",
                "ps": "LONG",
                "cp": False,
                "AP": "7476.89",
                "cr": "5.0",
                "rp": "0"
            }

        }

        mock_user_stream = AsyncMock()
        mock_user_stream.get.side_effect = functools.partial(self._return_calculation_and_set_done_event,
                                                             lambda: partial_fill)

        self.exchange._user_stream_tracker._user_stream = mock_user_stream

        self.test_task = self.local_event_loop.create_task(self.exchange._user_stream_event_listener())
        await self.resume_test_event.wait()

        self.assertEqual(1, len(self.order_filled_logger.event_log))
        fill_event: OrderFilledEvent = self.order_filled_logger.event_log[0]
        self.assertEqual(Decimal("0"), fill_event.trade_fee.percent)
        self.assertEqual(
            [TokenAmount(partial_fill["o"]["N"], Decimal(partial_fill["o"]["n"]))], fill_event.trade_fee.flat_fees
        )

        complete_fill = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1568879465651,
            "T": 1568879465650,
            "o": {
                "s": self.trading_pair,
                "c": order.client_order_id,
                "S": "BUY",
                "o": "TRAILING_STOP_MARKET",
                "f": "GTC",
                "q": "1",
                "p": "10000",
                "ap": "0",
                "sp": "7103.04",
                "x": "TRADE",
                "X": "FILLED",
                "i": 8886774,
                "l": "0.9",
                "z": "1",
                "L": "10000",
                "N": "HBOT",
                "n": "30",
                "T": 1568879465651,
                "t": 2,
                "b": "0",
                "a": "9.91",
                "m": False,
                "R": False,
                "wt": "CONTRACT_PRICE",
                "ot": "TRAILING_STOP_MARKET",
                "ps": "LONG",
                "cp": False,
                "AP": "7476.89",
                "cr": "5.0",
                "rp": "0"
            }

        }

        self.resume_test_event = asyncio.Event()
        mock_user_stream.get.side_effect = functools.partial(self._return_calculation_and_set_done_event,
                                                             lambda: complete_fill)

        self.test_task = self.local_event_loop.create_task(self.exchange._user_stream_event_listener())
        await self.resume_test_event.wait()

        self.assertEqual(2, len(self.order_filled_logger.event_log))
        fill_event: OrderFilledEvent = self.order_filled_logger.event_log[1]
        self.assertEqual(Decimal("0"), fill_event.trade_fee.percent)
        self.assertEqual([TokenAmount(complete_fill["o"]["N"], Decimal(complete_fill["o"]["n"]))],
                         fill_event.trade_fee.flat_fees)

    async def test_sell_order_fill_event_takes_fee_from_update_event(self):
        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.SELL,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        order = self.exchange.in_flight_orders.get("OID1")

        partial_fill = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1568879465651,
            "T": 1568879465650,
            "o": {
                "s": self.trading_pair,
                "c": order.client_order_id,
                "S": "SELL",
                "o": "TRAILING_STOP_MARKET",
                "f": "GTC",
                "q": "1",
                "p": "10000",
                "ap": "0",
                "sp": "7103.04",
                "x": "TRADE",
                "X": "PARTIALLY_FILLED",
                "i": 8886774,
                "l": "0.1",
                "z": "0.1",
                "L": "10000",
                "N": self.quote_asset,
                "n": "20",
                "T": 1568879465651,
                "t": 1,
                "b": "0",
                "a": "9.91",
                "m": False,
                "R": False,
                "wt": "CONTRACT_PRICE",
                "ot": "TRAILING_STOP_MARKET",
                "ps": "LONG",
                "cp": False,
                "AP": "7476.89",
                "cr": "5.0",
                "rp": "0"
            }
        }

        mock_user_stream = AsyncMock()
        mock_user_stream.get.side_effect = functools.partial(self._return_calculation_and_set_done_event,
                                                             lambda: partial_fill)

        self.exchange._user_stream_tracker._user_stream = mock_user_stream

        self.test_task = self.local_event_loop.create_task(self.exchange._user_stream_event_listener())
        await self.resume_test_event.wait()

        self.assertEqual(1, len(self.order_filled_logger.event_log))
        fill_event: OrderFilledEvent = self.order_filled_logger.event_log[0]
        self.assertEqual(Decimal("0"), fill_event.trade_fee.percent)
        self.assertEqual(
            [TokenAmount(partial_fill["o"]["N"], Decimal(partial_fill["o"]["n"]))], fill_event.trade_fee.flat_fees
        )

        complete_fill = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1568879465651,
            "T": 1568879465650,
            "o": {
                "s": self.trading_pair,
                "c": order.client_order_id,
                "S": "SELL",
                "o": "TRAILING_STOP_MARKET",
                "f": "GTC",
                "q": "1",
                "p": "10000",
                "ap": "0",
                "sp": "7103.04",
                "x": "TRADE",
                "X": "FILLED",
                "i": 8886774,
                "l": "0.9",
                "z": "1",
                "L": "10000",
                "N": self.quote_asset,
                "n": "30",
                "T": 1568879465651,
                "t": 2,
                "b": "0",
                "a": "9.91",
                "m": False,
                "R": False,
                "wt": "CONTRACT_PRICE",
                "ot": "TRAILING_STOP_MARKET",
                "ps": "LONG",
                "cp": False,
                "AP": "7476.89",
                "cr": "5.0",
                "rp": "0"
            }

        }

        self.resume_test_event = asyncio.Event()
        mock_user_stream.get.side_effect = functools.partial(self._return_calculation_and_set_done_event,
                                                             lambda: complete_fill)

        self.test_task = self.local_event_loop.create_task(self.exchange._user_stream_event_listener())
        await self.resume_test_event.wait()

        self.assertEqual(2, len(self.order_filled_logger.event_log))
        fill_event: OrderFilledEvent = self.order_filled_logger.event_log[1]
        self.assertEqual(Decimal("0"), fill_event.trade_fee.percent)
        self.assertEqual([TokenAmount(complete_fill["o"]["N"], Decimal(complete_fill["o"]["n"]))],
                         fill_event.trade_fee.flat_fees)

    async def test_order_fill_event_ignored_for_repeated_trade_id(self):
        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        order = self.exchange.in_flight_orders.get("OID1")

        partial_fill = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1568879465651,
            "T": 1568879465650,
            "o": {
                "s": self.trading_pair,
                "c": order.client_order_id,
                "S": "BUY",
                "o": "TRAILING_STOP_MARKET",
                "f": "GTC",
                "q": "1",
                "p": "10000",
                "ap": "0",
                "sp": "7103.04",
                "x": "TRADE",
                "X": "PARTIALLY_FILLED",
                "i": 8886774,
                "l": "0.1",
                "z": "0.1",
                "L": "10000",
                "N": self.quote_asset,
                "n": "20",
                "T": 1568879465651,
                "t": 1,
                "b": "0",
                "a": "9.91",
                "m": False,
                "R": False,
                "wt": "CONTRACT_PRICE",
                "ot": "TRAILING_STOP_MARKET",
                "ps": "LONG",
                "cp": False,
                "AP": "7476.89",
                "cr": "5.0",
                "rp": "0"
            }
        }

        mock_user_stream = AsyncMock()
        mock_user_stream.get.side_effect = functools.partial(self._return_calculation_and_set_done_event,
                                                             lambda: partial_fill)

        self.exchange._user_stream_tracker._user_stream = mock_user_stream

        self.test_task = self.local_event_loop.create_task(self.exchange._user_stream_event_listener())
        await self.resume_test_event.wait()

        self.assertEqual(1, len(self.order_filled_logger.event_log))
        fill_event: OrderFilledEvent = self.order_filled_logger.event_log[0]
        self.assertEqual(Decimal("0"), fill_event.trade_fee.percent)
        self.assertEqual(
            [TokenAmount(partial_fill["o"]["N"], Decimal(partial_fill["o"]["n"]))], fill_event.trade_fee.flat_fees
        )

        repeated_partial_fill = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1568879465651,
            "T": 1568879465650,
            "o": {
                "s": self.trading_pair,
                "c": order.client_order_id,
                "S": "BUY",
                "o": "TRAILING_STOP_MARKET",
                "f": "GTC",
                "q": "1",
                "p": "10000",
                "ap": "0",
                "sp": "7103.04",
                "x": "TRADE",
                "X": "PARTIALLY_FILLED",
                "i": 8886774,
                "l": "0.1",
                "z": "0.1",
                "L": "10000",
                "N": self.quote_asset,
                "n": "20",
                "T": 1568879465651,
                "t": 1,
                "b": "0",
                "a": "9.91",
                "m": False,
                "R": False,
                "wt": "CONTRACT_PRICE",
                "ot": "TRAILING_STOP_MARKET",
                "ps": "LONG",
                "cp": False,
                "AP": "7476.89",
                "cr": "5.0",
                "rp": "0"
            }
        }

        self.resume_test_event = asyncio.Event()
        mock_user_stream.get.side_effect = functools.partial(self._return_calculation_and_set_done_event,
                                                             lambda: repeated_partial_fill)

        self.test_task = self.local_event_loop.create_task(self.exchange._user_stream_event_listener())
        await self.resume_test_event.wait()

        self.assertEqual(1, len(self.order_filled_logger.event_log))

        self.assertEqual(0, len(self.buy_order_completed_logger.event_log))

    async def test_fee_is_zero_when_not_included_in_fill_event(self):
        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        order = self.exchange.in_flight_orders.get("OID1")

        partial_fill = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1568879465651,
            "T": 1568879465650,
            "o": {
                "s": self.trading_pair,
                "c": order.client_order_id,
                "S": "BUY",
                "o": "TRAILING_STOP_MARKET",
                "f": "GTC",
                "q": "1",
                "p": "10000",
                "ap": "0",
                "sp": "7103.04",
                "x": "TRADE",
                "X": "PARTIALLY_FILLED",
                "i": 8886774,
                "l": "0.1",
                "z": "0.1",
                "L": "10000",
                # "N": "USDT", //Do not include fee asset
                # "n": "20", //Do not include fee amount
                "T": 1568879465651,
                "t": 1,
                "b": "0",
                "a": "9.91",
                "m": False,
                "R": False,
                "wt": "CONTRACT_PRICE",
                "ot": "TRAILING_STOP_MARKET",
                "ps": "LONG",
                "cp": False,
                "AP": "7476.89",
                "cr": "5.0",
                "rp": "0"
            }

        }

        await self.exchange._process_user_stream_event(event_message=partial_fill)

        self.assertEqual(1, len(self.order_filled_logger.event_log))
        fill_event: OrderFilledEvent = self.order_filled_logger.event_log[0]
        self.assertEqual(Decimal("0"), fill_event.trade_fee.percent)
        self.assertEqual(0, len(fill_event.trade_fee.flat_fees))

    async def test_order_event_with_cancelled_status_marks_order_as_cancelled(self):
        self._simulate_trading_rules_initialized()
        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        order = self.exchange.in_flight_orders.get("OID1")

        partial_fill = {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1568879465651,
            "T": 1568879465650,
            "o": {
                "s": self.trading_pair,
                "c": order.client_order_id,
                "S": "BUY",
                "o": "TRAILING_STOP_MARKET",
                "f": "GTC",
                "q": "1",
                "p": "10000",
                "ap": "0",
                "sp": "7103.04",
                "x": "TRADE",
                "X": "CANCELED",
                "i": 8886774,
                "l": "0.1",
                "z": "0.1",
                "L": "10000",
                "N": self.quote_asset,
                "n": "20",
                "T": 1568879465651,
                "t": 1,
                "b": "0",
                "a": "9.91",
                "m": False,
                "R": False,
                "wt": "CONTRACT_PRICE",
                "ot": "TRAILING_STOP_MARKET",
                "ps": "LONG",
                "cp": False,
                "AP": "7476.89",
                "cr": "5.0",
                "rp": "0"
            }

        }

        mock_user_stream = AsyncMock()
        mock_user_stream.get.side_effect = functools.partial(self._return_calculation_and_set_done_event,
                                                             lambda: partial_fill)

        self.exchange._user_stream_tracker._user_stream = mock_user_stream

        self.test_task = self.local_event_loop.create_task(self.exchange._user_stream_event_listener())
        await self.resume_test_event.wait()
        await asyncio.sleep(0.001)

        self.assertEqual(1, len(self.order_cancelled_logger.event_log))

        self.assertTrue(self._is_logged(
            "INFO",
            f"Successfully canceled order {order.client_order_id}."
        ))

    async def test_user_stream_event_listener_raises_cancelled_error(self):
        mock_user_stream = AsyncMock()
        mock_user_stream.get.side_effect = asyncio.CancelledError

        self.exchange._user_stream_tracker._user_stream = mock_user_stream
        with self.assertRaises(asyncio.CancelledError):
            await self.exchange._user_stream_event_listener()

    async def test_margin_call_event(self):
        self._simulate_trading_rules_initialized()
        margin_call = {
            "e": "MARGIN_CALL",
            "E": 1587727187525,
            "cw": "3.16812045",
            "p": [
                {
                    "s": self.symbol,
                    "ps": "LONG",
                    "pa": "1.327",
                    "mt": "CROSSED",
                    "iw": "0",
                    "mp": "187.17127",
                    "up": "-1.166074",
                    "mm": "1.614445"
                }
            ]
        }

        mock_user_stream = AsyncMock()
        mock_user_stream.get.side_effect = functools.partial(self._return_calculation_and_set_done_event,
                                                             lambda: margin_call)

        self.exchange._user_stream_tracker._user_stream = mock_user_stream

        self.test_task = self.local_event_loop.create_task(self.exchange._user_stream_event_listener())
        await self.resume_test_event.wait()

        self.assertTrue(self._is_logged(
            "WARNING",
            "Margin Call: Your position risk is too high, and you are at risk of liquidation. "
            "Close your positions or add additional margin to your wallet."
        ))
        self.assertTrue(self._is_logged(
            "INFO",
            f"Margin Required: 1.614445. Negative PnL assets: {self.trading_pair}: -1.166074, ."
        ))

    async def test_wrong_symbol_margin_call_event(self):
        self._simulate_trading_rules_initialized()
        margin_call = {
            "e": "MARGIN_CALL",
            "E": 1587727187525,
            "cw": "3.16812045",
            "p": [
                {
                    "s": f"{self.symbol}_230331",
                    "ps": "LONG",
                    "pa": "1.327",
                    "mt": "CROSSED",
                    "iw": "0",
                    "mp": "187.17127",
                    "up": "-1.166074",
                    "mm": "1.614445"
                }
            ]
        }

        mock_user_stream = AsyncMock()
        mock_user_stream.get.side_effect = functools.partial(self._return_calculation_and_set_done_event,
                                                             lambda: margin_call)

        self.exchange._user_stream_tracker._user_stream = mock_user_stream

        self.test_task = self.local_event_loop.create_task(self.exchange._user_stream_event_listener())
        await self.resume_test_event.wait()

        self.assertTrue(self._is_logged(
            "WARNING",
            "Margin Call: Your position risk is too high, and you are at risk of liquidation. "
            "Close your positions or add additional margin to your wallet."
        ))
        self.assertTrue(self._is_logged(
            "INFO",
            "Margin Required: 0. Negative PnL assets: ."
        ))

    @aioresponses()
    @patch("hummingbot.connector.derivative.binance_perpetual.binance_perpetual_derivative."
           "BinancePerpetualDerivative.current_timestamp")
    async def test_update_order_fills_from_trades_successful(self, req_mock, mock_timestamp):
        self._simulate_trading_rules_initialized()
        self.exchange._last_poll_timestamp = 0
        mock_timestamp.return_value = 1

        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.SELL,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        trades = [{"buyer": False,
                   "commission": "0",
                   "commissionAsset": self.quote_asset,
                   "id": 698759,
                   "maker": False,
                   "orderId": "8886774",
                   "price": "10000",
                   "qty": "0.5",
                   "quoteQty": "5000",
                   "realizedPnl": "0",
                   "side": "SELL",
                   "positionSide": "SHORT",
                   "symbol": "COINALPHAHBOT",
                   "time": 1000}]

        url = web_utils.private_rest_url(
            CONSTANTS.ACCOUNT_TRADE_LIST_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.get(regex_url, body=json.dumps(trades))

        await self.exchange._update_order_fills_from_trades()

        in_flight_orders = self.exchange._order_tracker.active_orders

        self.assertTrue("OID1" in in_flight_orders)

        self.assertEqual("OID1", in_flight_orders["OID1"].client_order_id)
        self.assertEqual(f"{self.base_asset}-{self.quote_asset}", in_flight_orders["OID1"].trading_pair)
        self.assertEqual(OrderType.LIMIT, in_flight_orders["OID1"].order_type)
        self.assertEqual(TradeType.SELL, in_flight_orders["OID1"].trade_type)
        self.assertEqual(10000, in_flight_orders["OID1"].price)
        self.assertEqual(1, in_flight_orders["OID1"].amount)
        self.assertEqual("8886774", in_flight_orders["OID1"].exchange_order_id)
        self.assertEqual(OrderState.PENDING_CREATE, in_flight_orders["OID1"].current_state)
        self.assertEqual(1, in_flight_orders["OID1"].leverage)
        self.assertEqual(PositionAction.OPEN, in_flight_orders["OID1"].position)

        self.assertEqual(0.5, in_flight_orders["OID1"].executed_amount_base)
        self.assertEqual(5000, in_flight_orders["OID1"].executed_amount_quote)
        self.assertEqual(1, in_flight_orders["OID1"].last_update_timestamp)

        self.assertTrue("698759" in in_flight_orders["OID1"].order_fills.keys())

    @aioresponses()
    async def test_update_order_fills_from_trades_failed(self, req_mock):
        self.exchange._set_current_timestamp(1640001112.0)
        self.exchange._last_poll_timestamp = 0
        self._simulate_trading_rules_initialized()
        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.SELL,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        url = web_utils.private_rest_url(
            CONSTANTS.ACCOUNT_TRADE_LIST_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.get(regex_url, exception=Exception())

        await self.exchange._update_order_fills_from_trades()

        in_flight_orders = self.exchange._order_tracker.active_orders

        # Nothing has changed
        self.assertTrue("OID1" in in_flight_orders)

        self.assertEqual("OID1", in_flight_orders["OID1"].client_order_id)
        self.assertEqual(f"{self.base_asset}-{self.quote_asset}", in_flight_orders["OID1"].trading_pair)
        self.assertEqual(OrderType.LIMIT, in_flight_orders["OID1"].order_type)
        self.assertEqual(TradeType.SELL, in_flight_orders["OID1"].trade_type)
        self.assertEqual(10000, in_flight_orders["OID1"].price)
        self.assertEqual(1, in_flight_orders["OID1"].amount)
        self.assertEqual("8886774", in_flight_orders["OID1"].exchange_order_id)
        self.assertEqual(OrderState.PENDING_CREATE, in_flight_orders["OID1"].current_state)
        self.assertEqual(1, in_flight_orders["OID1"].leverage)
        self.assertEqual(PositionAction.OPEN, in_flight_orders["OID1"].position)

        self.assertEqual(0, in_flight_orders["OID1"].executed_amount_base)
        self.assertEqual(0, in_flight_orders["OID1"].executed_amount_quote)
        self.assertEqual(1640001112.0, in_flight_orders["OID1"].last_update_timestamp)

        # Error was logged
        self.assertTrue(self._is_logged("NETWORK",
                                        f"Error fetching trades update for the order {self.trading_pair}: ."))

    @aioresponses()
    @patch("hummingbot.connector.derivative.binance_perpetual.binance_perpetual_derivative."
           "BinancePerpetualDerivative.current_timestamp")
    async def test_update_order_status_successful(self, req_mock, mock_timestamp):
        self._simulate_trading_rules_initialized()
        self.exchange._last_poll_timestamp = 0
        mock_timestamp.return_value = 1

        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.SELL,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        order = {"avgPrice": "0.00000",
                 "clientOrderId": "OID1",
                 "cumQuote": "5000",
                 "executedQty": "0.5",
                 "orderId": 8886774,
                 "origQty": "1",
                 "origType": "LIMIT",
                 "price": "10000",
                 "reduceOnly": False,
                 "side": "SELL",
                 "positionSide": "LONG",
                 "status": "PARTIALLY_FILLED",
                 "closePosition": False,
                 "symbol": f"{self.base_asset}{self.quote_asset}",
                 "time": 1000,
                 "timeInForce": "GTC",
                 "type": "LIMIT",
                 "priceRate": "0.3",
                 "updateTime": 2000,
                 "workingType": "CONTRACT_PRICE",
                 "priceProtect": False}

        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.get(regex_url, body=json.dumps(order))

        await self.exchange._update_order_status()
        await asyncio.sleep(0.001)

        in_flight_orders = self.exchange._order_tracker.active_orders

        self.assertTrue("OID1" in in_flight_orders)

        self.assertEqual("OID1", in_flight_orders["OID1"].client_order_id)
        self.assertEqual(f"{self.base_asset}-{self.quote_asset}", in_flight_orders["OID1"].trading_pair)
        self.assertEqual(OrderType.LIMIT, in_flight_orders["OID1"].order_type)
        self.assertEqual(TradeType.SELL, in_flight_orders["OID1"].trade_type)
        self.assertEqual(10000, in_flight_orders["OID1"].price)
        self.assertEqual(1, in_flight_orders["OID1"].amount)
        self.assertEqual("8886774", in_flight_orders["OID1"].exchange_order_id)
        self.assertEqual(OrderState.PARTIALLY_FILLED, in_flight_orders["OID1"].current_state)
        self.assertEqual(1, in_flight_orders["OID1"].leverage)
        self.assertEqual(PositionAction.OPEN, in_flight_orders["OID1"].position)

        # Processing an order update should not impact trade fill information
        self.assertEqual(Decimal("0"), in_flight_orders["OID1"].executed_amount_base)
        self.assertEqual(Decimal("0"), in_flight_orders["OID1"].executed_amount_quote)

        self.assertEqual(2, in_flight_orders["OID1"].last_update_timestamp)

        self.assertEqual(0, len(in_flight_orders["OID1"].order_fills))

    @aioresponses()
    @patch("hummingbot.connector.derivative.binance_perpetual.binance_perpetual_derivative."
           "BinancePerpetualDerivative.current_timestamp")
    async def test_request_order_status_successful(self, req_mock, mock_timestamp):
        self._simulate_trading_rules_initialized()
        self.exchange._last_poll_timestamp = 0
        mock_timestamp.return_value = 1

        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.SELL,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )
        tracked_order = self.exchange._order_tracker.fetch_order("OID1")

        order = {"avgPrice": "0.00000",
                 "clientOrderId": "OID1",
                 "cumQuote": "5000",
                 "executedQty": "0.5",
                 "orderId": 8886774,
                 "origQty": "1",
                 "origType": "LIMIT",
                 "price": "10000",
                 "reduceOnly": False,
                 "side": "SELL",
                 "positionSide": "LONG",
                 "status": "PARTIALLY_FILLED",
                 "closePosition": False,
                 "symbol": f"{self.base_asset}{self.quote_asset}",
                 "time": 1000,
                 "timeInForce": "GTC",
                 "type": "LIMIT",
                 "priceRate": "0.3",
                 "updateTime": 2000,
                 "workingType": "CONTRACT_PRICE",
                 "priceProtect": False}

        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.get(regex_url, body=json.dumps(order))

        order_update = await self.exchange._request_order_status(tracked_order)

        in_flight_orders = self.exchange._order_tracker.active_orders
        self.assertTrue("OID1" in in_flight_orders)

        self.assertEqual(order_update.client_order_id, in_flight_orders["OID1"].client_order_id)
        self.assertEqual(OrderState.PARTIALLY_FILLED, order_update.new_state)
        self.assertEqual(0, len(in_flight_orders["OID1"].order_fills))

    @aioresponses()
    async def test_set_leverage_successful(self, req_mock):
        self._simulate_trading_rules_initialized()
        trading_pair = f"{self.base_asset}-{self.quote_asset}"
        symbol = f"{self.base_asset}{self.quote_asset}"
        leverage = 21

        response = {
            "leverage": leverage,
            "maxNotionalValue": "1000000",
            "symbol": symbol
        }

        url = web_utils.private_rest_url(
            CONSTANTS.SET_LEVERAGE_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.post(regex_url, body=json.dumps(response))

        success, msg = await self.exchange._set_trading_pair_leverage(trading_pair, leverage)
        self.assertEqual(success, True)
        self.assertEqual(msg, '')

    @aioresponses()
    async def test_set_leverage_failed(self, req_mock):
        self._simulate_trading_rules_initialized()
        trading_pair = f"{self.base_asset}-{self.quote_asset}"
        symbol = f"{self.base_asset}{self.quote_asset}"
        leverage = 21

        response = {"leverage": 0,
                    "maxNotionalValue": "1000000",
                    "symbol": symbol}

        url = web_utils.private_rest_url(
            CONSTANTS.SET_LEVERAGE_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.post(regex_url, body=json.dumps(response))

        success, message = await self.exchange._set_trading_pair_leverage(trading_pair, leverage)
        self.assertEqual(success, False)
        self.assertEqual(message, 'Unable to set leverage')

    @aioresponses()
    async def test_fetch_funding_payment_successful(self, req_mock):
        self._simulate_trading_rules_initialized()
        income_history = self._get_income_history_dict()

        url = web_utils.private_rest_url(
            CONSTANTS.GET_INCOME_HISTORY_URL, domain=self.domain
        )
        regex_url_income_history = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.get(regex_url_income_history, body=json.dumps(income_history))

        funding_info = self._get_funding_info_dict()

        url = web_utils.public_rest_url(
            CONSTANTS.MARK_PRICE_URL, domain=self.domain
        )
        regex_url_funding_info = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.get(regex_url_funding_info, body=json.dumps(funding_info))

        # Fetch from exchange with REST API - safe_ensure_future, not immediately
        await self.exchange._update_funding_payment(self.trading_pair, True)

        req_mock.get(regex_url_income_history, body=json.dumps(income_history))

        # Fetch once received
        await self.exchange._update_funding_payment(self.trading_pair, True)

        self.assertTrue(len(self.funding_payment_completed_logger.event_log) == 1)

        funding_info_logged = self.funding_payment_completed_logger.event_log[0]

        self.assertTrue(funding_info_logged.trading_pair == f"{self.base_asset}-{self.quote_asset}")

        self.assertEqual(funding_info_logged.funding_rate, funding_info["lastFundingRate"])
        self.assertEqual(funding_info_logged.amount, income_history[0]["income"])

    @aioresponses()
    async def test_fetch_funding_payment_failed(self, req_mock):
        self._simulate_trading_rules_initialized()
        url = web_utils.private_rest_url(
            CONSTANTS.GET_INCOME_HISTORY_URL, domain=self.domain
        )
        regex_url_income_history = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.get(regex_url_income_history, exception=Exception)

        await self.exchange._update_funding_payment(self.trading_pair, False)

        self.assertTrue(self._is_logged(
            "NETWORK",
            f"Unexpected error while fetching last fee payment for {self.trading_pair}.",
        ))

    @aioresponses()
    async def test_cancel_all_successful(self, mocked_api):
        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        cancel_response = {"code": 200, "msg": "success", "status": "CANCELED"}
        mocked_api.delete(regex_url, body=json.dumps(cancel_response))

        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        self.exchange.start_tracking_order(
            order_id="OID2",
            exchange_order_id="8886775",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10101"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        self.assertTrue("OID1" in self.exchange._order_tracker._in_flight_orders)
        self.assertTrue("OID2" in self.exchange._order_tracker._in_flight_orders)

        cancellation_results = await self.exchange.cancel_all(timeout_seconds=1)

        order_cancelled_events = self.order_cancelled_logger.event_log

        self.assertEqual(0, len(order_cancelled_events))
        self.assertEqual(2, len(cancellation_results))

    @aioresponses()
    async def test_cancel_all_unknown_order(self, req_mock):
        self._simulate_trading_rules_initialized()
        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        cancel_response = {"code": -2011, "msg": "Unknown order sent."}
        req_mock.delete(regex_url, body=json.dumps(cancel_response))

        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        tracked_order = self.exchange._order_tracker.fetch_order("OID1")
        tracked_order.current_state = OrderState.OPEN

        self.assertTrue("OID1" in self.exchange._order_tracker._in_flight_orders)

        cancellation_results = await self.exchange.cancel_all(timeout_seconds=1)

        self.assertEqual(1, len(cancellation_results))
        self.assertEqual("OID1", cancellation_results[0].order_id)

        self.assertTrue(self._is_logged(
            "DEBUG",
            "The order OID1 does not exist on Binance Perpetuals. "
            "No cancelation needed."
        ))

        self.assertTrue("OID1" in self.exchange._order_tracker._order_not_found_records)

    @aioresponses()
    async def test_cancel_all_exception(self, req_mock):
        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        req_mock.delete(regex_url, exception=Exception())

        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        tracked_order = self.exchange._order_tracker.fetch_order("OID1")
        tracked_order.current_state = OrderState.OPEN

        self.assertTrue("OID1" in self.exchange._order_tracker._in_flight_orders)

        cancellation_results = await self.exchange.cancel_all(timeout_seconds=1)

        self.assertEqual(1, len(cancellation_results))
        self.assertEqual("OID1", cancellation_results[0].order_id)

        self.assertTrue(self._is_logged(
            "ERROR",
            "Failed to cancel order OID1",
        ))

        self.assertTrue("OID1" in self.exchange._order_tracker._in_flight_orders)

    @aioresponses()
    async def test_cancel_order_successful(self, mock_api):
        self._simulate_trading_rules_initialized()
        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        cancel_response = {
            "clientOrderId": "ODI1",
            "cumQty": "0",
            "cumQuote": "0",
            "executedQty": "0",
            "orderId": 283194212,
            "origQty": "11",
            "origType": "TRAILING_STOP_MARKET",
            "price": "0",
            "reduceOnly": False,
            "side": "BUY",
            "positionSide": "SHORT",
            "status": "CANCELED",
            "stopPrice": "9300",
            "closePosition": False,
            "symbol": "BTCUSDT",
            "timeInForce": "GTC",
            "type": "TRAILING_STOP_MARKET",
            "activatePrice": "9020",
            "priceRate": "0.3",
            "updateTime": 1571110484038,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": False
        }
        mock_api.delete(regex_url, body=json.dumps(cancel_response))

        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )
        tracked_order = self.exchange._order_tracker.fetch_order("OID1")
        tracked_order.current_state = OrderState.OPEN

        self.assertTrue("OID1" in self.exchange._order_tracker._in_flight_orders)

        canceled_order_id = await self.exchange._execute_cancel(trading_pair=self.trading_pair, order_id="OID1")
        await asyncio.sleep(0.01)

        order_cancelled_events = self.order_cancelled_logger.event_log

        self.assertEqual(1, len(order_cancelled_events))
        self.assertEqual("OID1", canceled_order_id)

    @aioresponses()
    async def test_cancel_order_failed(self, mock_api):
        self._simulate_trading_rules_initialized()
        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        cancel_response = {
            "clientOrderId": "ODI1",
            "cumQty": "0",
            "cumQuote": "0",
            "executedQty": "0",
            "orderId": 283194212,
            "origQty": "11",
            "origType": "TRAILING_STOP_MARKET",
            "price": "0",
            "reduceOnly": False,
            "side": "BUY",
            "positionSide": "SHORT",
            "status": "FILLED",
            "stopPrice": "9300",
            "closePosition": False,
            "symbol": "BTCUSDT",
            "timeInForce": "GTC",
            "type": "TRAILING_STOP_MARKET",
            "activatePrice": "9020",
            "priceRate": "0.3",
            "updateTime": 1571110484038,
            "workingType": "CONTRACT_PRICE",
            "priceProtect": False
        }
        mock_api.delete(regex_url, body=json.dumps(cancel_response))

        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )
        tracked_order = self.exchange._order_tracker.fetch_order("OID1")
        tracked_order.current_state = OrderState.OPEN

        self.assertTrue("OID1" in self.exchange._order_tracker._in_flight_orders)

        await self.exchange._execute_cancel(trading_pair=self.trading_pair, order_id="OID1")

        order_cancelled_events = self.order_cancelled_logger.event_log

        self.assertEqual(0, len(order_cancelled_events))

    @aioresponses()
    async def test_create_order_successful(self, req_mock):
        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        create_response = {"updateTime": int(self.start_timestamp),
                           "status": "NEW",
                           "orderId": "8886774"}
        req_mock.post(regex_url, body=json.dumps(create_response))
        self._simulate_trading_rules_initialized()

        await self.exchange._create_order(
            trade_type=TradeType.BUY,
            order_id="OID1",
            trading_pair=self.trading_pair,
            amount=Decimal("10000"),
            order_type=OrderType.LIMIT,
            position_action=PositionAction.OPEN,
            price=Decimal("10000"))

        self.assertTrue("OID1" in self.exchange._order_tracker._in_flight_orders)

    @aioresponses()
    @patch("hummingbot.connector.derivative.binance_perpetual.binance_perpetual_web_utils.get_current_server_time")
    async def test_place_order_manage_server_overloaded_error_unkown_order(self, mock_api, mock_seconds_counter: MagicMock):
        mock_seconds_counter.return_value = 1640780000
        self.exchange._set_current_timestamp(1640780000)
        self.exchange._last_poll_timestamp = (self.exchange.current_timestamp -
                                              self.exchange.UPDATE_ORDER_STATUS_MIN_INTERVAL - 1)
        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        mock_response = {"code": -1003, "msg": "Unknown error, please check your request or try again later."}

        mock_api.post(regex_url, body=json.dumps(mock_response), status=503)
        self._simulate_trading_rules_initialized()

        with self.assertRaisesRegex(
            IOError,
            "submission outcome is unknown for client order ID OID1",
        ):
            await self.exchange._place_order(
                trade_type=TradeType.BUY,
                order_id="OID1",
                trading_pair=self.trading_pair,
                amount=Decimal("10000"),
                order_type=OrderType.LIMIT,
                position_action=PositionAction.OPEN,
                price=Decimal("10000"))

    @aioresponses()
    async def test_create_limit_maker_successful(self, req_mock):
        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        create_response = {"updateTime": int(self.start_timestamp),
                           "status": "NEW",
                           "orderId": "8886774"}
        req_mock.post(regex_url, body=json.dumps(create_response))
        self._simulate_trading_rules_initialized()

        await self.exchange._create_order(
            trade_type=TradeType.BUY,
            order_id="OID1",
            trading_pair=self.trading_pair,
            amount=Decimal("10000"),
            order_type=OrderType.LIMIT_MAKER,
            position_action=PositionAction.OPEN,
            price=Decimal("10000"))

        self.assertTrue("OID1" in self.exchange._order_tracker._in_flight_orders)

    @aioresponses()
    async def test_create_order_exception(self, req_mock):
        url = web_utils.private_rest_url(
            CONSTANTS.ORDER_URL, domain=self.domain
        )
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))
        req_mock.post(regex_url, exception=Exception())
        self._simulate_trading_rules_initialized()
        await self.exchange._create_order(
            trade_type=TradeType.BUY,
            order_id="OID1",
            trading_pair=self.trading_pair,
            amount=Decimal("10000"),
            order_type=OrderType.LIMIT,
            position_action=PositionAction.OPEN,
            price=Decimal("1010"))
        await asyncio.sleep(0.001)

        self.assertTrue("OID1" not in self.exchange._order_tracker._in_flight_orders)

        # The order amount is quantizied
        # "Error submitting buy LIMIT order to Binance_perpetual for 9999 COINALPHA-HBOT 1010."
        self.assertTrue(self._is_logged(
            "NETWORK",
            f"Error submitting {TradeType.BUY.name.lower()} {OrderType.LIMIT.name.upper()} order to {self.exchange.name_cap} for "
            f"{Decimal('9999')} {self.trading_pair} {Decimal('1010')}.",
        ))

    async def test_create_order_min_order_size_failure(self):
        self._simulate_trading_rules_initialized()
        margin_asset = self.quote_asset
        min_order_size = 3
        mocked_response = self._get_exchange_info_mock_response(margin_asset, min_order_size=min_order_size)
        trading_rules = await self.exchange._format_trading_rules(mocked_response)
        self.exchange._trading_rules[self.trading_pair] = trading_rules[0]
        trade_type = TradeType.BUY
        amount = Decimal("2")

        await self.exchange._create_order(
            trade_type=trade_type,
            order_id="OID1",
            trading_pair=self.trading_pair,
            amount=amount,
            order_type=OrderType.LIMIT,
            position_action=PositionAction.OPEN,
            price=Decimal("1010"))

        await asyncio.sleep(0.001)

        self.assertTrue("OID1" not in self.exchange._order_tracker._in_flight_orders)

        self.assertTrue(self._is_logged(
            "INFO",
            "Order OID1 has failed. Order Update: OrderUpdate(trading_pair='COINALPHA-HBOT', "
            "update_timestamp=1640780000.0, new_state=<OrderState.FAILED: 6>, client_order_id='OID1', "
            "exchange_order_id=None, misc_updates={'error_message': 'Order amount 2 is lower than minimum order size 3 "
            "for the pair COINALPHA-HBOT. The order will not be created.', 'error_type': 'ValueError'})"
        ))

    async def test_create_order_min_notional_size_failure(self):
        margin_asset = self.quote_asset
        min_notional_size = 10
        self._simulate_trading_rules_initialized()
        mocked_response = self._get_exchange_info_mock_response(margin_asset,
                                                                min_notional_size=min_notional_size,
                                                                min_base_amount_increment=0.5)
        trading_rules = await self.exchange._format_trading_rules(mocked_response)
        self.exchange._trading_rules[self.trading_pair] = trading_rules[0]
        trade_type = TradeType.BUY
        amount = Decimal("2")
        price = Decimal("4")

        await self.exchange._create_order(
            trade_type=trade_type,
            order_id="OID1",
            trading_pair=self.trading_pair,
            amount=amount,
            order_type=OrderType.LIMIT,
            position_action=PositionAction.OPEN,
            price=price)
        await asyncio.sleep(0.001)

        self.assertTrue("OID1" not in self.exchange._order_tracker._in_flight_orders)

    async def test_restore_tracking_states_only_registers_open_orders(self):
        orders = []
        orders.append(InFlightOrder(
            client_order_id="OID1",
            exchange_order_id="EOID1",
            trading_pair=self.trading_pair,
            order_type=OrderType.LIMIT,
            trade_type=TradeType.BUY,
            amount=Decimal("1000.0"),
            price=Decimal("1.0"),
            creation_timestamp=1640001112.223,
        ))
        orders.append(InFlightOrder(
            client_order_id="OID2",
            exchange_order_id="EOID2",
            trading_pair=self.trading_pair,
            order_type=OrderType.LIMIT,
            trade_type=TradeType.BUY,
            amount=Decimal("1000.0"),
            price=Decimal("1.0"),
            creation_timestamp=1640001112.223,
            initial_state=OrderState.CANCELED
        ))
        orders.append(InFlightOrder(
            client_order_id="OID3",
            exchange_order_id="EOID3",
            trading_pair=self.trading_pair,
            order_type=OrderType.LIMIT,
            trade_type=TradeType.BUY,
            amount=Decimal("1000.0"),
            price=Decimal("1.0"),
            creation_timestamp=1640001112.223,
            initial_state=OrderState.FILLED
        ))
        orders.append(InFlightOrder(
            client_order_id="OID4",
            exchange_order_id="EOID4",
            trading_pair=self.trading_pair,
            order_type=OrderType.LIMIT,
            trade_type=TradeType.BUY,
            amount=Decimal("1000.0"),
            price=Decimal("1.0"),
            creation_timestamp=1640001112.223,
            initial_state=OrderState.FAILED
        ))

        tracking_states = {order.client_order_id: order.to_json() for order in orders}

        self.exchange.restore_tracking_states(tracking_states)

        self.assertIn("OID1", self.exchange.in_flight_orders)
        self.assertNotIn("OID2", self.exchange.in_flight_orders)
        self.assertNotIn("OID3", self.exchange.in_flight_orders)
        self.assertNotIn("OID4", self.exchange.in_flight_orders)

    async def test_caller_id_reaches_native_order_tracker_created_event_and_signed_order_request(self):
        client_order_id = "exec-sndk-snxx-0005-maker-0"
        self._simulate_trading_rules_initialized()
        self.exchange._api_post = AsyncMock(return_value={
            "updateTime": 1700000000000,
            "status": "NEW",
            "orderId": 8886774,
        })
        created_logger = EventLogger()
        self.exchange.add_listener(MarketEvent.BuyOrderCreated, created_logger)

        await self.exchange._create_order(
            trade_type=TradeType.BUY,
            order_id=client_order_id,
            trading_pair=self.trading_pair,
            amount=Decimal("3"),
            order_type=OrderType.LIMIT_MAKER,
            position_action=PositionAction.OPEN,
            price=Decimal("10000"),
        )
        await asyncio.sleep(0.001)

        tracked_order = self.exchange.in_flight_orders[client_order_id]
        self.assertEqual(client_order_id, tracked_order.client_order_id)
        self.assertEqual("8886774", tracked_order.exchange_order_id)
        self.assertEqual(OrderState.OPEN, tracked_order.current_state)
        self.assertEqual(1, len(created_logger.event_log))
        self.assertEqual(client_order_id, created_logger.event_log[0].order_id)
        request = self.exchange._api_post.await_args.kwargs
        self.assertEqual(CONSTANTS.ORDER_URL, request["path_url"])
        self.assertTrue(request["is_auth_required"])
        self.assertEqual(client_order_id, request["data"]["newClientOrderId"])
        self.assertEqual("GTX", request["data"]["timeInForce"])

    async def test_preallocated_market_close_preserves_native_market_and_reduce_only_payload(self):
        client_order_id = "exec-sndk-snxx-0005-stock-0"
        self._simulate_trading_rules_initialized()
        self.exchange._position_mode = PositionMode.ONEWAY
        self.exchange._api_post = AsyncMock(return_value={
            "updateTime": 1700000000000,
            "status": "NEW",
            "orderId": 8886775,
        })

        exchange_order_id, update_timestamp = await self.exchange._place_order(
            trade_type=TradeType.SELL,
            order_id=client_order_id,
            trading_pair=self.trading_pair,
            amount=Decimal("1.250"),
            order_type=OrderType.MARKET,
            position_action=PositionAction.CLOSE,
            price=Decimal("NaN"),
        )

        self.assertEqual("8886775", exchange_order_id)
        self.assertEqual(1700000000, update_timestamp)
        request = self.exchange._api_post.await_args.kwargs
        self.assertEqual(
            {
                "symbol": self.symbol,
                "side": "SELL",
                "quantity": "1.250",
                "type": "MARKET",
                "newClientOrderId": client_order_id,
                "reduceOnly": "true",
            },
            request["data"],
        )

    async def test_post_only_market_is_rejected_before_network_and_limit_maker_keeps_gtx(self):
        self._simulate_trading_rules_initialized()
        self.exchange._api_post = AsyncMock(return_value={
            "updateTime": 1700000000000,
            "status": "NEW",
            "orderId": 8886776,
        })

        with self.assertRaisesRegex(ValueError, "post-only.*MARKET"):
            await self.exchange._place_order(
                trade_type=TradeType.BUY,
                order_id="exec-sndk-snxx-0006-stock-0",
                trading_pair=self.trading_pair,
                amount=Decimal("1"),
                order_type=OrderType.MARKET,
                position_action=PositionAction.OPEN,
                price=Decimal("NaN"),
                post_only=True,
            )

        self.exchange._api_post.assert_not_awaited()

        await self.exchange._place_order(
            trade_type=TradeType.BUY,
            order_id="exec-sndk-snxx-0006-maker-0",
            trading_pair=self.trading_pair,
            amount=Decimal("1"),
            order_type=OrderType.LIMIT_MAKER,
            position_action=PositionAction.OPEN,
            price=Decimal("10000"),
            post_only=True,
        )

        request = self.exchange._api_post.await_args.kwargs
        self.assertEqual("LIMIT", request["data"]["type"])
        self.assertEqual("GTX", request["data"]["timeInForce"])

    async def test_timeout_after_send_keeps_caller_id_as_unknown_pending_submission(self):
        client_order_id = "exec-sndk-snxx-0007-stock-0"
        self._simulate_trading_rules_initialized()
        self.exchange._api_post = AsyncMock(side_effect=asyncio.TimeoutError)

        await self.exchange._create_order(
            trade_type=TradeType.BUY,
            order_id=client_order_id,
            trading_pair=self.trading_pair,
            amount=Decimal("3"),
            order_type=OrderType.MARKET,
            position_action=PositionAction.OPEN,
            price=Decimal("10000"),
        )
        await asyncio.sleep(0.001)

        tracked_order = self.exchange.in_flight_orders[client_order_id]
        self.assertEqual(client_order_id, tracked_order.client_order_id)
        self.assertIsNone(tracked_order.exchange_order_id)
        self.assertEqual(OrderState.PENDING_CREATE, tracked_order.current_state)
        self.assertTrue(self.exchange.is_order_submission_unknown(client_order_id))

    async def test_typed_order_query_covers_late_partial_and_terminal_statuses(self):
        client_order_id = "exec-sndk-snxx-0008-stock-0"
        self._simulate_trading_rules_initialized()
        expected_terminal = {
            "NEW": False,
            "PARTIALLY_FILLED": False,
            "FILLED": True,
            "CANCELED": True,
            "EXPIRED": True,
            "REJECTED": True,
        }

        for status, is_terminal in expected_terminal.items():
            with self.subTest(status=status):
                executed_quantity = {
                    "NEW": "0",
                    "FILLED": "1.250",
                    "REJECTED": "0",
                }.get(status, "0.125")
                self.exchange._api_get = AsyncMock(return_value=self._get_reconciliation_order(
                    client_order_id=client_order_id,
                    status=status,
                    executed_quantity=executed_quantity,
                ))

                fact = await self.exchange.get_order_status_by_client_order_id(
                    trading_pair=self.trading_pair,
                    client_order_id=client_order_id,
                )

                self.assertEqual("BinancePerpetualOrderSnapshot", type(fact).__name__)
                self.assertEqual(client_order_id, fact.client_order_id)
                self.assertEqual("8886774", fact.exchange_order_id)
                self.assertEqual(status, fact.status.value)
                self.assertEqual(is_terminal, fact.is_terminal)
                self.assertFalse(fact.is_not_found)
                self.assertEqual(Decimal("10000.125"), fact.price)
                self.assertEqual(Decimal("1.250"), fact.original_quantity)
                self.assertEqual(Decimal(executed_quantity), fact.executed_quantity)
                self.assertEqual(
                    Decimal("10000.125") * Decimal(executed_quantity),
                    fact.cumulative_quote_quantity,
                )
                self.assertEqual(1700000000123, fact.update_time_ms)
                with self.assertRaises(FrozenInstanceError):
                    fact.client_order_id = "changed"
                self.exchange._api_get.assert_awaited_once_with(
                    path_url=CONSTANTS.ORDER_URL,
                    params={
                        "symbol": self.symbol,
                        "origClientOrderId": client_order_id,
                    },
                    is_auth_required=True,
                    return_err=True,
                    limit_id=CONSTANTS.GET_ORDER_LIMIT_ID,
                )

    async def test_order_query_returns_one_non_terminal_not_found_fact(self):
        client_order_id = "exec-sndk-snxx-0009-stock-0"
        self._simulate_trading_rules_initialized()
        self.exchange._api_get = AsyncMock(return_value={
            "code": CONSTANTS.ORDER_NOT_EXIST_ERROR_CODE,
            "msg": "Order does not exist.",
        })

        fact = await self.exchange.get_order_status_by_client_order_id(
            trading_pair=self.trading_pair,
            client_order_id=client_order_id,
        )

        self.assertEqual(client_order_id, fact.client_order_id)
        self.assertEqual("NOT_FOUND", fact.status.value)
        self.assertTrue(fact.is_not_found)
        self.assertFalse(fact.is_terminal)
        self.assertIsNone(fact.exchange_order_id)
        self.assertIsNone(fact.executed_quantity)

    async def test_order_query_rejects_conflicting_id_and_redacts_transport_payload(self):
        client_order_id = "exec-sndk-snxx-0010-stock-0"
        self._simulate_trading_rules_initialized()
        conflicting = self._get_reconciliation_order(client_order_id="different-order-id")
        self.exchange._api_get = AsyncMock(return_value=conflicting)

        with self.assertRaisesRegex(ValueError, "client order ID does not match"):
            await self.exchange.get_order_status_by_client_order_id(
                trading_pair=self.trading_pair,
                client_order_id=client_order_id,
            )

        secret_payload = "private-key-and-response-payload-sentinel"
        self.exchange._api_get = AsyncMock(side_effect=IOError(secret_payload))
        with self.assertRaises(ValueError) as raised:
            await self.exchange.get_order_status_by_client_order_id(
                trading_pair=self.trading_pair,
                client_order_id=client_order_id,
            )
        self.assertNotIn(secret_payload, str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)

    async def test_typed_open_orders_are_immutable_deduplicated_and_conflicts_fail_closed(self):
        self._simulate_trading_rules_initialized()
        first = self._get_reconciliation_order(
            client_order_id="exec-sndk-snxx-0011-maker-0",
            exchange_order_id=8886780,
            status="NEW",
            executed_quantity="0",
        )
        second = self._get_reconciliation_order(
            client_order_id="exec-sndk-snxx-0011-stock-0",
            exchange_order_id=8886781,
            status="PARTIALLY_FILLED",
        )
        self.exchange._api_get = AsyncMock(return_value=[first, first.copy(), second])

        facts = await self.exchange.get_open_orders(self.trading_pair)

        self.assertIsInstance(facts, tuple)
        self.assertEqual(2, len(facts))
        self.assertEqual(
            {"exec-sndk-snxx-0011-maker-0", "exec-sndk-snxx-0011-stock-0"},
            {fact.client_order_id for fact in facts},
        )
        self.assertEqual({"NEW", "PARTIALLY_FILLED"}, {fact.status.value for fact in facts})
        self.exchange._api_get.assert_awaited_once_with(
            path_url=CONSTANTS.OPEN_ORDERS_URL,
            params={"symbol": self.symbol},
            is_auth_required=True,
        )

        conflict = first.copy()
        conflict["orderId"] = 9999999
        self.exchange._api_get = AsyncMock(return_value=[first, conflict])
        with self.assertRaisesRegex(ValueError, "conflicting client order ID"):
            await self.exchange.get_open_orders(self.trading_pair)

    async def test_typed_account_trades_preserve_exact_decimals_and_deduplicate_trade_id(self):
        self._simulate_trading_rules_initialized()
        trade = self._get_reconciliation_trade()
        self.exchange._api_get = AsyncMock(return_value=[trade, trade.copy()])

        facts = await self.exchange.get_account_trades(
            trading_pair=self.trading_pair,
            exchange_order_id="8886774",
        )

        self.assertIsInstance(facts, tuple)
        self.assertEqual(1, len(facts))
        fact = facts[0]
        self.assertEqual("698759", fact.trade_id)
        self.assertEqual("8886774", fact.exchange_order_id)
        self.assertEqual(Decimal("10000.125"), fact.price)
        self.assertEqual(Decimal("0.125"), fact.quantity)
        self.assertEqual(Decimal("1250.015625"), fact.quote_quantity)
        self.assertEqual(Decimal("0.00001234"), fact.commission)
        self.assertEqual(Decimal("-0.00000001"), fact.realized_pnl)
        self.assertEqual(TradeType.SELL, fact.side)
        self.assertEqual(1700000000456, fact.timestamp_ms)
        with self.assertRaises(FrozenInstanceError):
            fact.trade_id = "changed"
        self.exchange._api_get.assert_awaited_once_with(
            path_url=CONSTANTS.ACCOUNT_TRADE_LIST_URL,
            params={"symbol": self.symbol, "orderId": "8886774"},
            is_auth_required=True,
        )

        conflict = trade.copy()
        conflict["price"] = "10000.126"
        self.exchange._api_get = AsyncMock(return_value=[trade, conflict])
        with self.assertRaisesRegex(ValueError, "conflicting trade ID"):
            await self.exchange.get_account_trades(
                trading_pair=self.trading_pair,
                exchange_order_id="8886774",
            )

    async def test_reconciliation_reads_reject_malformed_identifiers_and_non_finite_fills(self):
        self._simulate_trading_rules_initialized()
        client_order_id = "exec-sndk-snxx-0012-stock-0"
        malformed_order = self._get_reconciliation_order(client_order_id=client_order_id)
        malformed_order["orderId"] = True
        self.exchange._api_get = AsyncMock(return_value=malformed_order)

        with self.assertRaisesRegex(ValueError, "orderId"):
            await self.exchange.get_order_status_by_client_order_id(
                self.trading_pair, client_order_id
            )

        malformed_trade = self._get_reconciliation_trade()
        malformed_trade["id"] = ""
        self.exchange._api_get = AsyncMock(return_value=[malformed_trade])
        with self.assertRaisesRegex(ValueError, "trade.*id"):
            await self.exchange.get_account_trades(self.trading_pair, "8886774")

        malformed_trade = self._get_reconciliation_trade()
        malformed_trade["qty"] = "NaN"
        self.exchange._api_get = AsyncMock(return_value=[malformed_trade])
        with self.assertRaisesRegex(ValueError, "qty"):
            await self.exchange.get_account_trades(self.trading_pair, "8886774")

    async def test_position_v3_facts_deduplicate_identical_rows_and_reject_conflicts(self):
        self._simulate_trading_rules_initialized()
        position = self._get_reconciliation_position()
        self.exchange._api_get = AsyncMock(return_value=[position, position.copy()])

        facts = await self.exchange.get_position_risk_snapshots(self.trading_pair)

        self.assertIsInstance(facts, tuple)
        self.assertEqual(1, len(facts))
        self.assertEqual(self.symbol, facts[0].symbol)
        self.assertEqual(Decimal("-0.125"), facts[0].position_amount)
        self.assertEqual(Decimal("-1250.15625"), facts[0].notional)

        conflict = position.copy()
        conflict["positionAmt"] = "-0.250"
        self.exchange._api_get = AsyncMock(return_value=[position, conflict])
        with self.assertRaisesRegex(ValueError, "conflicting position"):
            await self.exchange.get_position_risk_snapshots(self.trading_pair)

    async def test_reconciliation_reads_propagate_cancellation(self):
        client_order_id = "exec-sndk-snxx-0013-stock-0"
        self._simulate_trading_rules_initialized()
        calls = (
            lambda: self.exchange.get_order_status_by_client_order_id(
                self.trading_pair, client_order_id
            ),
            lambda: self.exchange.get_open_orders(self.trading_pair),
            lambda: self.exchange.get_account_trades(self.trading_pair, "8886774"),
            lambda: self.exchange.get_position_risk_snapshots(self.trading_pair),
        )

        for call in calls:
            with self.subTest(call=call):
                self.exchange._api_get = AsyncMock(side_effect=asyncio.CancelledError)
                with self.assertRaises(asyncio.CancelledError):
                    await call()

    async def test_submission_unknown_native_polling_keeps_four_not_found_observations_non_terminal(self):
        client_order_id = "exec-sndk-snxx-0014-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        self.exchange._last_poll_timestamp = 0
        self.exchange._api_get = AsyncMock(return_value={
            "code": CONSTANTS.ORDER_NOT_EXIST_ERROR_CODE,
            "msg": CONSTANTS.ORDER_NOT_EXIST_MESSAGE,
        })
        self.exchange._update_order_fills_from_trades = AsyncMock()
        self.exchange._update_balances = AsyncMock()
        self.exchange._update_positions = AsyncMock()

        observed_states = []
        for _ in range(4):
            await self.exchange._status_polling_loop_fetch_updates()
            observed_states.append((
                tracked_order.current_state,
                tracked_order.exchange_order_id,
                self.exchange.is_order_submission_unknown(client_order_id),
                self.exchange._order_tracker._order_not_found_records.get(client_order_id, 0),
                client_order_id in self.exchange.in_flight_orders,
            ))

        self.assertEqual(
            [(OrderState.PENDING_CREATE, None, True, 0, True)] * 4,
            observed_states,
        )
        self.assertNotIn(client_order_id, self.exchange._order_tracker.lost_orders)
        self.assertEqual(4, self.exchange._api_get.await_count)
        for request in self.exchange._api_get.await_args_list:
            self.assertEqual(CONSTANTS.ORDER_URL, request.kwargs["path_url"])
            self.assertEqual(CONSTANTS.GET_ORDER_LIMIT_ID, request.kwargs["limit_id"])
            self.assertTrue(request.kwargs["return_err"])

    async def test_submission_unknown_cancel_not_found_never_counts_or_fails_order(self):
        client_order_id = "exec-sndk-snxx-0015-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        self.exchange._api_delete = AsyncMock(return_value={
            "code": CONSTANTS.UNKNOWN_ORDER_ERROR_CODE,
            "msg": f"{CONSTANTS.UNKNOWN_ORDER_MESSAGE}.",
        })

        with patch.object(
            self.exchange._order_tracker,
            "process_order_not_found",
            wraps=self.exchange._order_tracker.process_order_not_found,
        ) as process_not_found:
            result = await self.exchange._execute_order_cancel(tracked_order)

        self.assertIsNone(result)
        process_not_found.assert_not_awaited()
        self.assertEqual(OrderState.PENDING_CREATE, tracked_order.current_state)
        self.assertTrue(self.exchange.is_order_submission_unknown(client_order_id))
        self.assertEqual(
            0,
            self.exchange._order_tracker._order_not_found_records.get(client_order_id, 0),
        )
        request = self.exchange._api_delete.await_args.kwargs
        self.assertEqual(CONSTANTS.ORDER_URL, request["path_url"])
        self.assertNotIn("limit_id", request)

    async def test_submission_unknown_cancel_minus_2013_is_also_non_terminal(self):
        client_order_id = "exec-sndk-snxx-0016-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        self.exchange._api_delete = AsyncMock(return_value={
            "code": CONSTANTS.ORDER_NOT_EXIST_ERROR_CODE,
            "msg": CONSTANTS.ORDER_NOT_EXIST_MESSAGE,
        })

        result = await self.exchange._execute_order_cancel(tracked_order)

        self.assertIsNone(result)
        self.assertEqual(OrderState.PENDING_CREATE, tracked_order.current_state)
        self.assertTrue(self.exchange.is_order_submission_unknown(client_order_id))
        self.assertEqual(
            0,
            self.exchange._order_tracker._order_not_found_records.get(client_order_id, 0),
        )

    async def test_submission_unknown_native_positive_status_validates_updates_and_clears(self):
        client_order_id = "exec-sndk-snxx-0017-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        self.exchange._last_poll_timestamp = 0
        self.exchange._api_get = AsyncMock(return_value=self._get_reconciliation_order(
            client_order_id=client_order_id,
            status="NEW",
            executed_quantity="0",
        ))

        await self.exchange._update_order_status()
        await asyncio.sleep(0.001)

        self.assertEqual(OrderState.OPEN, tracked_order.current_state)
        self.assertEqual("8886774", tracked_order.exchange_order_id)
        self.assertFalse(self.exchange.is_order_submission_unknown(client_order_id))
        request = self.exchange._api_get.await_args.kwargs
        self.assertEqual(CONSTANTS.GET_ORDER_LIMIT_ID, request["limit_id"])
        self.assertNotIn(CONSTANTS.ORDERS_1MIN, request.values())
        self.assertNotIn(CONSTANTS.ORDERS_1SEC, request.values())

    async def test_submission_unknown_native_malformed_positive_status_cannot_mutate_or_clear(self):
        client_order_id = "exec-sndk-snxx-0018-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        self.exchange._last_poll_timestamp = 0
        malformed = self._get_reconciliation_order(
            client_order_id=client_order_id,
            status="NEW",
            executed_quantity="0",
        )
        del malformed["avgPrice"]
        self.exchange._api_get = AsyncMock(return_value=malformed)

        await self.exchange._update_order_status()
        await asyncio.sleep(0.001)

        self.assertEqual(OrderState.PENDING_CREATE, tracked_order.current_state)
        self.assertIsNone(tracked_order.exchange_order_id)
        self.assertTrue(self.exchange.is_order_submission_unknown(client_order_id))

    async def test_submission_unknown_typed_not_found_then_positive_updates_exact_order_idempotently(self):
        client_order_id = "exec-sndk-snxx-0019-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        created_logger = EventLogger()
        self.exchange.add_listener(MarketEvent.SellOrderCreated, created_logger)
        not_found = {
            "code": CONSTANTS.ORDER_NOT_EXIST_ERROR_CODE,
            "msg": CONSTANTS.ORDER_NOT_EXIST_MESSAGE,
        }
        positive = self._get_reconciliation_order(
            client_order_id=client_order_id,
            status="NEW",
            executed_quantity="0",
        )
        self.exchange._api_get = AsyncMock(side_effect=[not_found, positive, positive.copy()])

        not_found_fact = await self.exchange.get_order_status_by_client_order_id(
            self.trading_pair,
            client_order_id,
        )

        self.assertTrue(not_found_fact.is_not_found)
        self.assertEqual(OrderState.PENDING_CREATE, tracked_order.current_state)
        self.assertTrue(self.exchange.is_order_submission_unknown(client_order_id))
        self.assertEqual(
            0,
            self.exchange._order_tracker._order_not_found_records.get(client_order_id, 0),
        )

        self.exchange._order_tracker._order_not_found_records[client_order_id] = 2
        positive_fact = await self.exchange.get_order_status_by_client_order_id(
            self.trading_pair,
            client_order_id,
        )
        repeated_fact = await self.exchange.get_order_status_by_client_order_id(
            self.trading_pair,
            client_order_id,
        )

        self.assertFalse(positive_fact.is_not_found)
        self.assertEqual(positive_fact, repeated_fact)
        self.assertEqual(OrderState.OPEN, tracked_order.current_state)
        self.assertEqual("8886774", tracked_order.exchange_order_id)
        self.assertFalse(self.exchange.is_order_submission_unknown(client_order_id))
        self.assertNotIn(client_order_id, self.exchange._order_tracker._order_not_found_records)
        self.assertEqual(1, len(created_logger.event_log))

    async def test_submission_unknown_user_stream_malformed_and_wrong_id_events_do_not_clear(self):
        malformed_id = "exec-sndk-snxx-0020-stock-0"
        self._simulate_trading_rules_initialized()
        malformed_order = self._track_submission_unknown_order(malformed_id)
        secret_status = "private-response-payload-sentinel"
        malformed_event = self._submission_unknown_user_event(
            client_order_id=malformed_id,
            status=secret_status,
        )

        with self.assertRaises(Exception) as malformed_error:
            await self.exchange._process_user_stream_event(malformed_event)

        self.assertNotIn(secret_status, str(malformed_error.exception))
        self.assertIsNone(malformed_error.exception.__cause__)
        self.assertIsNone(malformed_error.exception.__context__)
        self.assertEqual(OrderState.PENDING_CREATE, malformed_order.current_state)
        self.assertTrue(self.exchange.is_order_submission_unknown(malformed_id))

        wrong_id = "exec-sndk-snxx-0021-stock-0"
        wrong_id_order = self._track_submission_unknown_order(
            wrong_id,
            exchange_order_id="8886774",
        )
        wrong_id_event = self._submission_unknown_user_event(
            client_order_id=wrong_id,
            exchange_order_id=9999999,
        )

        with self.assertRaises(ValueError):
            await self.exchange._process_user_stream_event(wrong_id_event)

        self.assertEqual(OrderState.PENDING_CREATE, wrong_id_order.current_state)
        self.assertEqual("8886774", wrong_id_order.exchange_order_id)
        self.assertTrue(self.exchange.is_order_submission_unknown(wrong_id))

    async def test_submission_unknown_valid_user_stream_event_updates_once_and_clears(self):
        client_order_id = "exec-sndk-snxx-0022-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        created_logger = EventLogger()
        self.exchange.add_listener(MarketEvent.SellOrderCreated, created_logger)
        event = self._submission_unknown_user_event(client_order_id=client_order_id)

        await self.exchange._process_user_stream_event(event)
        await asyncio.sleep(0.001)
        await self.exchange._process_user_stream_event(event)
        await asyncio.sleep(0.001)

        self.assertEqual(OrderState.OPEN, tracked_order.current_state)
        self.assertEqual("8886774", tracked_order.exchange_order_id)
        self.assertFalse(self.exchange.is_order_submission_unknown(client_order_id))
        self.assertEqual(1, len(created_logger.event_log))

    async def test_submission_unknown_stream_atomic_rejects_zero_fill_partial_before_mutation(self):
        client_order_id = "exec-sndk-snxx-0030-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        event = self._submission_unknown_user_event(client_order_id=client_order_id)
        event["o"]["X"] = "PARTIALLY_FILLED"

        with self.assertRaises(ValueError):
            await self.exchange._process_user_stream_event(event)

        self.assertEqual(OrderState.PENDING_CREATE, tracked_order.current_state)
        self.assertIsNone(tracked_order.exchange_order_id)
        self.assertEqual(Decimal("0"), tracked_order.executed_amount_base)
        self.assertEqual({}, tracked_order.order_fills)
        self.assertTrue(self.exchange.is_order_submission_unknown(client_order_id))

    async def test_submission_unknown_stream_atomic_rejects_overfill_before_mutation(self):
        client_order_id = "exec-sndk-snxx-0031-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        event = self._submission_unknown_fill_event(
            client_order_id=client_order_id,
            status="PARTIALLY_FILLED",
            last_fill_quantity="2.000",
            cumulative_quantity="2.000",
            trade_id=1,
        )

        with self.assertRaises(ValueError):
            await self.exchange._process_user_stream_event(event)

        self.assertEqual(OrderState.PENDING_CREATE, tracked_order.current_state)
        self.assertIsNone(tracked_order.exchange_order_id)
        self.assertEqual(Decimal("0"), tracked_order.executed_amount_base)
        self.assertEqual({}, tracked_order.order_fills)
        self.assertTrue(self.exchange.is_order_submission_unknown(client_order_id))

    async def test_submission_unknown_stream_atomic_rejects_identity_contradictions(self):
        self._simulate_trading_rules_initialized()
        contradictions = (
            ("side", "S", "BUY"),
            ("type", "o", "MARKET"),
            ("quantity", "q", "1.251"),
            ("price", "p", "10000.126"),
        )
        observations = []

        for index, (label, field, value) in enumerate(contradictions, start=2):
            client_order_id = f"exec-sndk-snxx-003{index}-stock-0"
            tracked_order = self._track_submission_unknown_order(client_order_id)
            event = self._submission_unknown_user_event(client_order_id=client_order_id)
            event["o"][field] = value
            rejected = False
            try:
                await self.exchange._process_user_stream_event(event)
            except ValueError:
                rejected = True
            observations.append((
                label,
                rejected,
                tracked_order.current_state,
                tracked_order.exchange_order_id,
                tracked_order.executed_amount_base,
                len(tracked_order.order_fills),
                self.exchange.is_order_submission_unknown(client_order_id),
            ))

        self.assertEqual(
            [
                (label, True, OrderState.PENDING_CREATE, None, Decimal("0"), 0, True)
                for label, _, _ in contradictions
            ],
            observations,
        )

    async def test_submission_unknown_stream_atomic_rejects_missing_and_invalid_fill_facts(self):
        self._simulate_trading_rules_initialized()
        invalid_cases = []
        for field in ("l", "z", "ap", "L", "t"):
            invalid_cases.append((f"missing {field}", "missing", field, None))
        invalid_cases.extend((
            ("non-finite last quantity", "replace", "l", "NaN"),
            ("non-finite cumulative quantity", "replace", "z", "Infinity"),
            ("non-finite average price", "replace", "ap", "NaN"),
            ("non-finite last price", "replace", "L", "Infinity"),
            ("non-finite cumulative quote", "replace", "Z", "NaN"),
            ("negative cumulative quantity", "replace", "z", "-0.125"),
            ("negative average price", "replace", "ap", "-10000.125"),
            ("negative cumulative quote", "replace", "Z", "-1250.015625"),
            ("negative trade ID", "replace", "t", -1),
            ("zero trade ID with fill", "replace", "t", 0),
        ))
        observations = []

        for index, (label, operation, field, value) in enumerate(invalid_cases, start=40):
            client_order_id = f"exec-sndk-snxx-{index:04d}-stock-0"
            tracked_order = self._track_submission_unknown_order(client_order_id)
            event = self._submission_unknown_fill_event(
                client_order_id=client_order_id,
                status="PARTIALLY_FILLED",
                last_fill_quantity="0.125",
                cumulative_quantity="0.125",
                trade_id=1,
            )
            if operation == "missing":
                del event["o"][field]
            else:
                event["o"][field] = value
            rejected = False
            try:
                await self.exchange._process_user_stream_event(event)
            except ValueError:
                rejected = True
            observations.append((
                label,
                rejected,
                tracked_order.current_state,
                tracked_order.exchange_order_id,
                tracked_order.executed_amount_base,
                len(tracked_order.order_fills),
                self.exchange.is_order_submission_unknown(client_order_id),
            ))

        self.assertEqual(
            [
                (label, True, OrderState.PENDING_CREATE, None, Decimal("0"), 0, True)
                for label, _, _, _ in invalid_cases
            ],
            observations,
        )

    async def test_submission_unknown_stream_atomic_rejects_execution_status_contradictions(self):
        self._simulate_trading_rules_initialized()
        private_sentinel = "private-response-payload-sentinel"
        cases = (
            ("NEW", "NEW", "0.125", "0.125", 1),
            ("TRADE", "FILLED", "0.500", "0.500", 2),
            ("CANCELED", "CANCELED", "0.125", "0.125", 3),
            ("EXPIRED", "EXPIRED", "0.125", "0.125", 4),
            (private_sentinel, "NEW", "0", "0", 0),
        )
        observations = []

        for index, (execution, status, last_fill, cumulative, trade_id) in enumerate(cases, start=60):
            client_order_id = f"exec-sndk-snxx-{index:04d}-stock-0"
            tracked_order = self._track_submission_unknown_order(client_order_id)
            if last_fill == "0":
                event = self._submission_unknown_user_event(client_order_id=client_order_id)
            else:
                event = self._submission_unknown_fill_event(
                    client_order_id=client_order_id,
                    status=status,
                    last_fill_quantity=last_fill,
                    cumulative_quantity=cumulative,
                    trade_id=trade_id,
                )
            event["o"]["x"] = execution
            event["o"]["X"] = status
            rejected = False
            sanitized = False
            with patch.object(self.exchange._order_tracker, "process_trade_update") as process_trade_update:
                with patch.object(
                    self.exchange._order_tracker,
                    "process_order_update",
                    new=AsyncMock(),
                ) as process_order_update:
                    try:
                        await self.exchange._process_user_stream_event(event)
                    except ValueError as error:
                        rejected = True
                        sanitized = (
                            private_sentinel not in str(error)
                            and error.__cause__ is None
                            and error.__context__ is None
                        )
            observations.append((
                execution,
                rejected,
                sanitized,
                process_trade_update.call_count,
                process_order_update.await_count,
                tracked_order.current_state,
                self.exchange.is_order_submission_unknown(client_order_id),
            ))

        self.assertEqual(
            [
                (execution, True, True, 0, 0, OrderState.PENDING_CREATE, True)
                for execution, _, _, _, _ in cases
            ],
            observations,
        )

    async def test_submission_unknown_stream_atomic_rejects_cumulative_rollback_and_jumps(self):
        self._simulate_trading_rules_initialized()
        cumulative_quote = Decimal("10000.125") * Decimal("0.250")
        cases = (
            ("rollback", "0.050", "0.100", None, None),
            ("base jump", "0.125", "0.500", None, None),
            (
                "quote jump",
                "0.125",
                "0.250",
                f"{cumulative_quote + Decimal('1'):f}",
                f"{(cumulative_quote + Decimal('1')) / Decimal('0.250'):f}",
            ),
        )
        observations = []

        for index, (label, last_fill, cumulative, quote, average) in enumerate(cases, start=70):
            client_order_id = f"exec-sndk-snxx-{index:04d}-stock-0"
            tracked_order = self._track_submission_unknown_order(client_order_id)
            first_partial = self._submission_unknown_fill_event(
                client_order_id=client_order_id,
                status="PARTIALLY_FILLED",
                last_fill_quantity="0.125",
                cumulative_quantity="0.125",
                trade_id=1,
            )
            await self.exchange._process_user_stream_event(first_partial)
            self.exchange._unknown_submission_order_ids.add(client_order_id)
            invalid = self._submission_unknown_fill_event(
                client_order_id=client_order_id,
                status="PARTIALLY_FILLED",
                last_fill_quantity=last_fill,
                cumulative_quantity=cumulative,
                trade_id=2,
                cumulative_quote=quote,
                average_price=average,
            )
            rejected = False
            try:
                await self.exchange._process_user_stream_event(invalid)
            except ValueError:
                rejected = True
            observations.append((
                label,
                rejected,
                tracked_order.current_state,
                tracked_order.executed_amount_base,
                tracked_order.executed_amount_quote,
                len(tracked_order.order_fills),
                self.exchange.is_order_submission_unknown(client_order_id),
            ))

        first_quote = Decimal("10000.125") * Decimal("0.125")
        self.assertEqual(
            [
                (label, True, OrderState.PARTIALLY_FILLED, Decimal("0.125"), first_quote, 1, True)
                for label, _, _, _, _ in cases
            ],
            observations,
        )

    async def test_submission_unknown_stream_atomic_accepts_valid_lifecycle_and_duplicate_trade(self):
        self._simulate_trading_rules_initialized()

        new_id = "exec-sndk-snxx-0080-stock-0"
        new_order = self._track_submission_unknown_order(new_id)
        await self.exchange._process_user_stream_event(
            self._submission_unknown_user_event(client_order_id=new_id)
        )

        fill_id = "exec-sndk-snxx-0081-stock-0"
        fill_order = self._track_submission_unknown_order(fill_id)
        partial = self._submission_unknown_fill_event(
            client_order_id=fill_id,
            status="PARTIALLY_FILLED",
            last_fill_quantity="0.125",
            cumulative_quantity="0.125",
            trade_id=1,
        )
        await self.exchange._process_user_stream_event(partial)
        self.exchange._unknown_submission_order_ids.add(fill_id)
        await self.exchange._process_user_stream_event(partial)
        duplicate_snapshot = (
            fill_order.executed_amount_base,
            fill_order.executed_amount_quote,
            len(fill_order.order_fills),
        )
        self.exchange._unknown_submission_order_ids.add(fill_id)
        filled = self._submission_unknown_fill_event(
            client_order_id=fill_id,
            status="FILLED",
            last_fill_quantity="1.125",
            cumulative_quantity="1.250",
            trade_id=2,
        )
        await self.exchange._process_user_stream_event(filled)

        canceled_id = "exec-sndk-snxx-0082-stock-0"
        canceled_order = self._track_submission_unknown_order(canceled_id)
        canceled = self._submission_unknown_user_event(client_order_id=canceled_id)
        canceled["o"].update({"x": "CANCELED", "X": "CANCELED"})
        await self.exchange._process_user_stream_event(canceled)

        expired_id = "exec-sndk-snxx-0083-stock-0"
        expired_order = self._track_submission_unknown_order(expired_id)
        expired = self._submission_unknown_user_event(client_order_id=expired_id)
        expired["o"].update({"x": "EXPIRED", "X": "EXPIRED"})
        await self.exchange._process_user_stream_event(expired)

        self.assertEqual(OrderState.OPEN, new_order.current_state)
        self.assertEqual("8886774", new_order.exchange_order_id)
        self.assertFalse(self.exchange.is_order_submission_unknown(new_id))
        self.assertEqual(
            (Decimal("0.125"), Decimal("1250.015625"), 1),
            duplicate_snapshot,
        )
        self.assertEqual(OrderState.FILLED, fill_order.current_state)
        self.assertEqual(Decimal("1.250"), fill_order.executed_amount_base)
        self.assertEqual(2, len(fill_order.order_fills))
        self.assertFalse(self.exchange.is_order_submission_unknown(fill_id))
        self.assertEqual(OrderState.CANCELED, canceled_order.current_state)
        self.assertFalse(self.exchange.is_order_submission_unknown(canceled_id))
        self.assertEqual(OrderState.CANCELED, expired_order.current_state)
        self.assertFalse(self.exchange.is_order_submission_unknown(expired_id))

    async def test_submission_unknown_stream_atomic_clears_only_after_tracker_success_and_propagates_cancel(self):
        self._simulate_trading_rules_initialized()

        failure_id = "exec-sndk-snxx-0084-stock-0"
        failure_order = self._track_submission_unknown_order(failure_id)
        failure_event = self._submission_unknown_user_event(client_order_id=failure_id)
        with patch.object(
            self.exchange._order_tracker,
            "process_order_update",
            new=AsyncMock(side_effect=RuntimeError("tracker update failed")),
        ):
            with self.assertRaisesRegex(RuntimeError, "tracker update failed"):
                await self.exchange._process_user_stream_event(failure_event)

        cancel_id = "exec-sndk-snxx-0085-stock-0"
        cancel_order = self._track_submission_unknown_order(cancel_id)
        cancel_event = self._submission_unknown_user_event(client_order_id=cancel_id)
        with patch.object(
            self.exchange._order_tracker,
            "process_order_update",
            new=AsyncMock(side_effect=asyncio.CancelledError),
        ):
            with self.assertRaises(asyncio.CancelledError):
                await self.exchange._process_user_stream_event(cancel_event)

        self.assertEqual(OrderState.PENDING_CREATE, failure_order.current_state)
        self.assertIsNone(failure_order.exchange_order_id)
        self.assertTrue(self.exchange.is_order_submission_unknown(failure_id))
        self.assertEqual(OrderState.PENDING_CREATE, cancel_order.current_state)
        self.assertIsNone(cancel_order.exchange_order_id)
        self.assertTrue(self.exchange.is_order_submission_unknown(cancel_id))

    async def test_submission_unknown_stream_native_accepts_new_without_cumulative_quote(self):
        self._simulate_trading_rules_initialized()
        observations = []

        for index, order_type in enumerate((OrderType.LIMIT, OrderType.MARKET), start=90):
            client_order_id = f"exec-sndk-snxx-{index:04d}-stock-0"
            tracked_order = self._track_submission_unknown_order(
                client_order_id=client_order_id,
                order_type=order_type,
            )
            raw_order_type = "MARKET" if order_type is OrderType.MARKET else "LIMIT"
            event = self._submission_unknown_user_event(
                client_order_id=client_order_id,
                order_type=raw_order_type,
            )
            self.assertNotIn("Z", event["o"])

            error = None
            try:
                await self.exchange._process_user_stream_event(event)
            except ValueError as exception:
                error = exception
            observations.append((
                order_type,
                error,
                tracked_order.current_state,
                tracked_order.exchange_order_id,
                tracked_order.executed_amount_base,
                len(tracked_order.order_fills),
                self.exchange.is_order_submission_unknown(client_order_id),
                event["o"]["p"],
            ))

        self.assertEqual(
            [
                (OrderType.LIMIT, None, OrderState.OPEN, "8886774", Decimal("0"), 0, False, "10000.125"),
                (OrderType.MARKET, None, OrderState.OPEN, "8886774", Decimal("0"), 0, False, "0"),
            ],
            observations,
        )

    async def test_submission_unknown_stream_native_accepts_limit_multi_price_lifecycle(self):
        self._simulate_trading_rules_initialized()

        await self._assert_submission_unknown_native_multi_price_lifecycle(
            client_order_id="exec-sndk-snxx-0092-stock-0",
            order_type=OrderType.LIMIT,
        )

    async def test_submission_unknown_stream_native_accepts_market_multi_price_lifecycle(self):
        self._simulate_trading_rules_initialized()

        await self._assert_submission_unknown_native_multi_price_lifecycle(
            client_order_id="exec-sndk-snxx-0093-stock-0",
            order_type=OrderType.MARKET,
        )

    async def test_submission_unknown_stream_native_accepts_rounded_average_with_optional_quote(self):
        self._simulate_trading_rules_initialized()
        client_order_id = "exec-sndk-snxx-0101-stock-0"
        tracked_order = self._track_submission_unknown_order(client_order_id)
        first_partial = self._submission_unknown_fill_event(
            client_order_id=client_order_id,
            status="PARTIALLY_FILLED",
            last_fill_quantity="0.100",
            cumulative_quantity="0.100",
            trade_id=1,
            fill_price="10000.125",
            average_price="10000.125",
            cumulative_quote="1000.012500",
        )
        await self.exchange._process_user_stream_event(first_partial)
        self.exchange._unknown_submission_order_ids.add(client_order_id)
        rounded_partial = self._submission_unknown_fill_event(
            client_order_id=client_order_id,
            status="PARTIALLY_FILLED",
            last_fill_quantity="0.200",
            cumulative_quantity="0.300",
            trade_id=2,
            fill_price="10000.126",
            average_price="10000.12566667",
            cumulative_quote="3000.037700",
        )

        await self.exchange._process_user_stream_event(rounded_partial)

        self.assertEqual(OrderState.PARTIALLY_FILLED, tracked_order.current_state)
        self.assertEqual(Decimal("0.300"), tracked_order.executed_amount_base)
        self.assertEqual(Decimal("3000.037700"), tracked_order.executed_amount_quote)
        self.assertEqual(2, len(tracked_order.order_fills))
        self.assertFalse(self.exchange.is_order_submission_unknown(client_order_id))

    async def test_submission_unknown_stream_native_accepts_terminal_after_fill_without_cumulative_quote(self):
        self._simulate_trading_rules_initialized()
        cases = (
            ("CANCELED", OrderType.LIMIT),
            ("EXPIRED", OrderType.MARKET),
        )
        observations = []

        for index, (status, order_type) in enumerate(cases, start=94):
            client_order_id = f"exec-sndk-snxx-{index:04d}-stock-0"
            raw_order_type = "MARKET" if order_type is OrderType.MARKET else "LIMIT"
            tracked_order = self._track_submission_unknown_order(
                client_order_id=client_order_id,
                order_type=order_type,
            )
            partial = self._submission_unknown_fill_event(
                client_order_id=client_order_id,
                status="PARTIALLY_FILLED",
                last_fill_quantity="0.100",
                cumulative_quantity="0.100",
                trade_id=1,
                fill_price="10000.125",
                average_price="10000.125",
                order_type=raw_order_type,
            )
            await self.exchange._process_user_stream_event(partial)
            self.exchange._unknown_submission_order_ids.add(client_order_id)
            terminal = self._submission_unknown_user_event(
                client_order_id=client_order_id,
                order_type=raw_order_type,
            )
            terminal["o"].update({
                "ap": "10000.125",
                "x": status,
                "X": status,
                "z": "0.100",
            })
            self.assertNotIn("Z", terminal["o"])

            await self.exchange._process_user_stream_event(terminal)
            observations.append((
                status,
                tracked_order.current_state,
                tracked_order.executed_amount_base,
                tracked_order.executed_amount_quote,
                len(tracked_order.order_fills),
                self.exchange.is_order_submission_unknown(client_order_id),
            ))

        self.assertEqual(
            [
                (status, OrderState.CANCELED, Decimal("0.100"), Decimal("1000.012500"), 1, False)
                for status, _ in cases
            ],
            observations,
        )

    async def test_submission_unknown_stream_native_applies_reported_average_rounding_bound(self):
        self._simulate_trading_rules_initialized()
        cases = (
            ("average boundary", "10000.12500001", None, True),
            ("average over boundary", "10000.12500002", None, False),
            ("clearly wrong average", "10001.125", None, False),
            ("matching optional quote", "10000.125", "3000.037500", True),
            ("contradictory optional quote", "10000.125", "3000.037501", False),
        )
        observations = []

        for index, (label, average_price, cumulative_quote, should_accept) in enumerate(cases, start=96):
            client_order_id = f"exec-sndk-snxx-{index:04d}-stock-0"
            tracked_order = self._track_submission_unknown_order(client_order_id)
            event = self._submission_unknown_fill_event(
                client_order_id=client_order_id,
                status="PARTIALLY_FILLED",
                last_fill_quantity="0.300",
                cumulative_quantity="0.300",
                trade_id=1,
                fill_price="10000.125",
                average_price=average_price,
                cumulative_quote=cumulative_quote,
            )
            rejected = False
            try:
                await self.exchange._process_user_stream_event(event)
            except ValueError:
                rejected = True
            observations.append((
                label,
                rejected,
                tracked_order.current_state,
                tracked_order.exchange_order_id,
                tracked_order.executed_amount_base,
                tracked_order.executed_amount_quote,
                len(tracked_order.order_fills),
                self.exchange.is_order_submission_unknown(client_order_id),
                "Z" in event["o"],
                should_accept,
            ))

        self.assertEqual(
            [
                (
                    label,
                    not should_accept,
                    OrderState.PARTIALLY_FILLED if should_accept else OrderState.PENDING_CREATE,
                    "8886774" if should_accept else None,
                    Decimal("0.300") if should_accept else Decimal("0"),
                    Decimal("3000.037500") if should_accept else Decimal("0"),
                    1 if should_accept else 0,
                    not should_accept,
                    cumulative_quote is not None,
                    should_accept,
                )
                for label, _, cumulative_quote, should_accept in cases
            ],
            observations,
        )

    async def test_submission_unknown_restore_pending_without_exchange_id_is_conservative(self):
        client_order_id = "exec-sndk-snxx-0023-stock-0"
        restored_order = InFlightOrder(
            client_order_id=client_order_id,
            exchange_order_id=None,
            trading_pair=self.trading_pair,
            order_type=OrderType.LIMIT,
            trade_type=TradeType.SELL,
            amount=Decimal("1.250"),
            price=Decimal("10000.125"),
            creation_timestamp=1700000000,
            initial_state=OrderState.PENDING_CREATE,
        )

        self.exchange.restore_tracking_states({client_order_id: restored_order.to_json()})

        tracked_order = self.exchange.in_flight_orders[client_order_id]
        self.assertEqual(OrderState.PENDING_CREATE, tracked_order.current_state)
        self.assertIsNone(tracked_order.exchange_order_id)
        self.assertTrue(self.exchange.is_order_submission_unknown(client_order_id))
        self.assertIn(client_order_id, self.exchange._reserved_client_order_ids)

    async def test_submission_unknown_native_polling_redacts_errors_and_propagates_cancellation(self):
        client_order_id = "exec-sndk-snxx-0024-stock-0"
        self._simulate_trading_rules_initialized()
        tracked_order = self._track_submission_unknown_order(client_order_id)
        self.exchange._last_poll_timestamp = 0
        secret_payload = "private-key-and-response-payload-sentinel"
        self.exchange._api_get = AsyncMock(side_effect=IOError(secret_payload))

        await self.exchange._update_order_status()

        self.assertFalse(any(secret_payload in record.getMessage() for record in self.log_records))
        self.assertEqual(OrderState.PENDING_CREATE, tracked_order.current_state)
        self.assertTrue(self.exchange.is_order_submission_unknown(client_order_id))
        self.assertEqual(
            0,
            self.exchange._order_tracker._order_not_found_records.get(client_order_id, 0),
        )

        self.exchange._api_get = AsyncMock(side_effect=asyncio.CancelledError)
        with self.assertRaises(asyncio.CancelledError):
            await self.exchange._update_order_status()

    async def test_submission_unknown_get_and_write_rate_limit_identities_remain_separate(self):
        get_order_limit = next(
            rate_limit
            for rate_limit in CONSTANTS.RATE_LIMITS
            if rate_limit.limit_id == CONSTANTS.GET_ORDER_LIMIT_ID
        )
        write_order_limit = next(
            rate_limit
            for rate_limit in CONSTANTS.RATE_LIMITS
            if rate_limit.limit_id == CONSTANTS.ORDER_URL
        )

        self.assertEqual(
            [(CONSTANTS.REQUEST_WEIGHT, 1)],
            [(link.limit_id, link.weight) for link in get_order_limit.linked_limits],
        )
        self.assertEqual(
            {
                (CONSTANTS.REQUEST_WEIGHT, 1),
                (CONSTANTS.ORDERS_1MIN, 1),
                (CONSTANTS.ORDERS_1SEC, 1),
            },
            {(link.limit_id, link.weight) for link in write_order_limit.linked_limits},
        )

        self._simulate_trading_rules_initialized()
        self.exchange._api_post = AsyncMock(return_value={
            "updateTime": 1700000000000,
            "status": "NEW",
            "orderId": 8886774,
        })
        await self.exchange._place_order(
            trade_type=TradeType.SELL,
            order_id="exec-sndk-snxx-0025-stock-0",
            trading_pair=self.trading_pair,
            amount=Decimal("1.250"),
            order_type=OrderType.LIMIT,
            position_action=PositionAction.OPEN,
            price=Decimal("10000.125"),
        )

        request = self.exchange._api_post.await_args.kwargs
        self.assertEqual(CONSTANTS.ORDER_URL, request["path_url"])
        self.assertNotIn("limit_id", request)

    @patch("hummingbot.connector.utils.get_tracking_nonce")
    async def test_client_order_id_on_order(self, mocked_nonce):
        mocked_nonce.return_value = 4

        result = self.exchange.buy(
            trading_pair=self.trading_pair,
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            price=Decimal("2"),
            position_action="OPEN",
        )
        expected_client_order_id = get_new_client_order_id(
            is_buy=True,
            trading_pair=self.trading_pair,
            hbot_order_id_prefix=CONSTANTS.BROKER_ID,
            max_id_len=CONSTANTS.MAX_ORDER_ID_LEN,
        )

        self.assertEqual(result, expected_client_order_id)

        result = self.exchange.sell(
            trading_pair=self.trading_pair,
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            price=Decimal("2"),
            position_action="OPEN",
        )
        expected_client_order_id = get_new_client_order_id(
            is_buy=False,
            trading_pair=self.trading_pair,
            hbot_order_id_prefix=CONSTANTS.BROKER_ID,
            max_id_len=CONSTANTS.MAX_ORDER_ID_LEN,
        )

        self.assertEqual(result, expected_client_order_id)

    @aioresponses()
    async def test_update_balances(self, mock_api):
        url = web_utils.public_rest_url(CONSTANTS.SERVER_TIME_PATH_URL)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        response = {"serverTime": 1640000003000}

        mock_api.get(regex_url,
                     body=json.dumps(response))

        url = web_utils.private_rest_url(CONSTANTS.ACCOUNT_INFO_URL, domain=self.domain)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        response = {
            "feeTier": 0,
            "canTrade": True,
            "canDeposit": True,
            "canWithdraw": True,
            "updateTime": 0,
            "totalInitialMargin": "0.00000000",
            "totalMaintMargin": "0.00000000",
            "totalWalletBalance": "23.72469206",
            "totalUnrealizedProfit": "0.00000000",
            "totalMarginBalance": "23.72469206",
            "totalPositionInitialMargin": "0.00000000",
            "totalOpenOrderInitialMargin": "0.00000000",
            "totalCrossWalletBalance": "23.72469206",
            "totalCrossUnPnl": "0.00000000",
            "availableBalance": "23.72469206",
            "maxWithdrawAmount": "23.72469206",
            "assets": [
                {
                    "asset": "USDT",
                    "walletBalance": "23.72469206",
                    "unrealizedProfit": "0.00000000",
                    "marginBalance": "23.72469206",
                    "maintMargin": "0.00000000",
                    "initialMargin": "0.00000000",
                    "positionInitialMargin": "0.00000000",
                    "openOrderInitialMargin": "0.00000000",
                    "crossWalletBalance": "23.72469206",
                    "crossUnPnl": "0.00000000",
                    "availableBalance": "23.72469206",
                    "maxWithdrawAmount": "23.72469206",
                    "marginAvailable": True,
                    "updateTime": 1625474304765,
                },
                {
                    "asset": "BUSD",
                    "walletBalance": "103.12345678",
                    "unrealizedProfit": "0.00000000",
                    "marginBalance": "103.12345678",
                    "maintMargin": "0.00000000",
                    "initialMargin": "0.00000000",
                    "positionInitialMargin": "0.00000000",
                    "openOrderInitialMargin": "0.00000000",
                    "crossWalletBalance": "103.12345678",
                    "crossUnPnl": "0.00000000",
                    "availableBalance": "100.12345678",
                    "maxWithdrawAmount": "103.12345678",
                    "marginAvailable": True,
                    "updateTime": 1625474304765,
                }
            ],
            "positions": [{
                "symbol": "BTCUSDT",
                "initialMargin": "0",
                "maintMargin": "0",
                "unrealizedProfit": "0.00000000",
                "positionInitialMargin": "0",
                "openOrderInitialMargin": "0",
                "leverage": "100",
                "isolated": True,
                "entryPrice": "0.00000",
                "maxNotional": "250000",
                "bidNotional": "0",
                "askNotional": "0",
                "positionSide": "BOTH",
                "positionAmt": "0",
                "updateTime": 0,
            }
            ]
        }

        mock_api.get(regex_url, body=json.dumps(response))
        await self.exchange._update_balances()

        available_balances = self.exchange.available_balances
        total_balances = self.exchange.get_all_balances()

        self.assertEqual(Decimal("23.72469206"), available_balances["USDT"])
        self.assertEqual(Decimal("100.12345678"), available_balances["BUSD"])
        self.assertEqual(Decimal("23.72469206"), total_balances["USDT"])
        self.assertEqual(Decimal("103.12345678"), total_balances["BUSD"])

    @aioresponses()
    @patch("hummingbot.connector.time_synchronizer.TimeSynchronizer._current_seconds_counter")
    async def test_account_info_request_includes_timestamp(self, mock_api, mock_seconds_counter):
        mock_seconds_counter.return_value = 1000

        url = web_utils.public_rest_url(CONSTANTS.SERVER_TIME_PATH_URL)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        response = {"serverTime": 1640000003000}

        mock_api.get(regex_url,
                     body=json.dumps(response))

        url = web_utils.private_rest_url(CONSTANTS.ACCOUNT_INFO_URL, domain=self.domain)
        regex_url = re.compile(f"^{url}".replace(".", r"\.").replace("?", r"\?"))

        response = {
            "feeTier": 0,
            "canTrade": True,
            "canDeposit": True,
            "canWithdraw": True,
            "updateTime": 0,
            "totalInitialMargin": "0.00000000",
            "totalMaintMargin": "0.00000000",
            "totalWalletBalance": "23.72469206",
            "totalUnrealizedProfit": "0.00000000",
            "totalMarginBalance": "23.72469206",
            "totalPositionInitialMargin": "0.00000000",
            "totalOpenOrderInitialMargin": "0.00000000",
            "totalCrossWalletBalance": "23.72469206",
            "totalCrossUnPnl": "0.00000000",
            "availableBalance": "23.72469206",
            "maxWithdrawAmount": "23.72469206",
            "assets": [
                {
                    "asset": "USDT",
                    "walletBalance": "23.72469206",
                    "unrealizedProfit": "0.00000000",
                    "marginBalance": "23.72469206",
                    "maintMargin": "0.00000000",
                    "initialMargin": "0.00000000",
                    "positionInitialMargin": "0.00000000",
                    "openOrderInitialMargin": "0.00000000",
                    "crossWalletBalance": "23.72469206",
                    "crossUnPnl": "0.00000000",
                    "availableBalance": "23.72469206",
                    "maxWithdrawAmount": "23.72469206",
                    "marginAvailable": True,
                    "updateTime": 1625474304765,
                },
                {
                    "asset": "BUSD",
                    "walletBalance": "103.12345678",
                    "unrealizedProfit": "0.00000000",
                    "marginBalance": "103.12345678",
                    "maintMargin": "0.00000000",
                    "initialMargin": "0.00000000",
                    "positionInitialMargin": "0.00000000",
                    "openOrderInitialMargin": "0.00000000",
                    "crossWalletBalance": "103.12345678",
                    "crossUnPnl": "0.00000000",
                    "availableBalance": "100.12345678",
                    "maxWithdrawAmount": "103.12345678",
                    "marginAvailable": True,
                    "updateTime": 1625474304765,
                }
            ],
            "positions": [{
                "symbol": "BTCUSDT",
                "initialMargin": "0",
                "maintMargin": "0",
                "unrealizedProfit": "0.00000000",
                "positionInitialMargin": "0",
                "openOrderInitialMargin": "0",
                "leverage": "100",
                "isolated": True,
                "entryPrice": "0.00000",
                "maxNotional": "250000",
                "bidNotional": "0",
                "askNotional": "0",
                "positionSide": "BOTH",
                "positionAmt": "0",
                "updateTime": 0,
            }
            ]
        }

        mock_api.get(regex_url, body=json.dumps(response))
        await self.exchange._update_balances()

        account_request = next(((key, value) for key, value in mock_api.requests.items()
                                if key[1].human_repr().startswith(url)))
        request_params = account_request[1][0].kwargs["params"]
        self.assertIsInstance(request_params["timestamp"], int)

    async def test_limit_orders(self):
        self.exchange.start_tracking_order(
            order_id="OID1",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )
        self.exchange.start_tracking_order(
            order_id="OID2",
            exchange_order_id="8886774",
            trading_pair=self.trading_pair,
            trade_type=TradeType.BUY,
            price=Decimal("10000"),
            amount=Decimal("1"),
            order_type=OrderType.LIMIT,
            leverage=1,
            position_action=PositionAction.OPEN,
        )

        limit_orders = self.exchange.limit_orders

        self.assertEqual(len(limit_orders), 2)
        self.assertIsInstance(limit_orders, list)
        self.assertIsInstance(limit_orders[0], LimitOrder)

    def _simulate_trading_rules_initialized(self):
        margin_asset = self.quote_asset
        mocked_response = self._get_exchange_info_mock_response(margin_asset)
        self.exchange._initialize_trading_pair_symbols_from_exchange_info(mocked_response)
        self.exchange._trading_rules = {
            self.trading_pair: TradingRule(
                trading_pair=self.trading_pair,
                min_order_size=Decimal(str(1)),
                min_price_increment=Decimal(str(2)),
                min_base_amount_increment=Decimal(str(3)),
                min_notional_size=Decimal(str(4)),
            )
        }
        return self.exchange._trading_rules
