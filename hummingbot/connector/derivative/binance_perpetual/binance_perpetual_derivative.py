import asyncio
import math
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from typing import Any, AsyncIterable, Collection, Dict, List, Mapping, Optional, Tuple

from bidict import bidict

from hummingbot.connector.constants import s_decimal_NaN
from hummingbot.connector.derivative.binance_perpetual import (
    binance_perpetual_constants as CONSTANTS,
    binance_perpetual_web_utils as web_utils,
)
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_api_order_book_data_source import (
    BinancePerpetualAPIOrderBookDataSource,
)
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_auth import BinancePerpetualAuth
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_order_data import (
    BinancePerpetualOrderDataError,
    BinancePerpetualOrderSnapshot,
    BinancePerpetualOrderSubmissionUnknown,
    BinancePerpetualOrderStatus,
    BinancePerpetualTrade,
    validate_binance_client_order_id,
    validate_binance_exchange_order_id,
)
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_risk_data import (
    BinancePerpetualAccountConfig,
    BinancePerpetualAccountRiskSnapshot,
    BinancePerpetualInstrumentInfo,
    BinancePerpetualLeverageBrackets,
    BinancePerpetualLeverageChangeResult,
    BinancePerpetualMarginType,
    BinancePerpetualMultiAssetsMode,
    BinancePerpetualPositionMode,
    BinancePerpetualPositionRiskSnapshot,
    BinancePerpetualPreflightError,
    BinancePerpetualPreflightSnapshot,
    BinancePerpetualRiskDataError,
    BinancePerpetualSymbolConfig,
)
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_user_stream_data_source import (
    BinancePerpetualUserStreamDataSource,
)
from hummingbot.connector.derivative.position import Position
from hummingbot.connector.perpetual_derivative_py_base import PerpetualDerivativePyBase
from hummingbot.connector.trading_rule import TradingRule
from hummingbot.connector.utils import combine_to_hb_trading_pair
from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.data_type.common import OrderType, PositionAction, PositionMode, PositionSide, TradeType
from hummingbot.core.data_type.in_flight_order import InFlightOrder, OrderUpdate, TradeUpdate
from hummingbot.core.data_type.order_book_tracker_data_source import OrderBookTrackerDataSource
from hummingbot.core.data_type.trade_fee import TokenAmount, TradeFeeBase
from hummingbot.core.data_type.user_stream_tracker_data_source import UserStreamTrackerDataSource
from hummingbot.core.utils.async_utils import safe_gather
from hummingbot.core.utils.estimate_fee import build_trade_fee
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory

bpm_logger = None


class BinancePerpetualDerivative(PerpetualDerivativePyBase):
    web_utils = web_utils
    SHORT_POLL_INTERVAL = 5.0
    UPDATE_ORDER_STATUS_MIN_INTERVAL = 10.0
    LONG_POLL_INTERVAL = 120.0
    MAX_ACCOUNT_DATA_AGE_SECONDS = 5

    def __init__(
            self,
            balance_asset_limit: Optional[Dict[str, Dict[str, Decimal]]] = None,
            rate_limits_share_pct: Decimal = Decimal("100"),
            binance_perpetual_api_key: str = None,
            binance_perpetual_api_secret: str = None,
            trading_pairs: Optional[List[str]] = None,
            trading_required: bool = True,
            domain: str = CONSTANTS.DOMAIN,
    ):
        self.binance_perpetual_api_key = binance_perpetual_api_key
        self.binance_perpetual_secret_key = binance_perpetual_api_secret
        self._trading_required = trading_required
        self._trading_pairs = trading_pairs
        self._domain = domain
        self._position_mode = None
        self._leverage_bracket_cache: Dict[str, BinancePerpetualLeverageBrackets] = {}
        self._last_trade_history_timestamp = None
        self._unknown_submission_order_ids = set()
        super().__init__(balance_asset_limit, rate_limits_share_pct)

    @property
    def name(self) -> str:
        return CONSTANTS.EXCHANGE_NAME

    @property
    def authenticator(self) -> BinancePerpetualAuth:
        return BinancePerpetualAuth(self.binance_perpetual_api_key, self.binance_perpetual_secret_key,
                                    self._time_synchronizer)

    @property
    def rate_limits_rules(self) -> List[RateLimit]:
        return CONSTANTS.RATE_LIMITS

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def client_order_id_max_length(self) -> int:
        return CONSTANTS.MAX_ORDER_ID_LEN

    @property
    def client_order_id_prefix(self) -> str:
        return CONSTANTS.BROKER_ID

    @property
    def trading_rules_request_path(self) -> str:
        return CONSTANTS.EXCHANGE_INFO_URL

    @property
    def trading_pairs_request_path(self) -> str:
        return CONSTANTS.EXCHANGE_INFO_URL

    @property
    def check_network_request_path(self) -> str:
        return CONSTANTS.PING_URL

    @property
    def trading_pairs(self):
        return self._trading_pairs

    @property
    def is_cancel_request_in_exchange_synchronous(self) -> bool:
        return True

    @property
    def is_trading_required(self) -> bool:
        return self._trading_required

    @property
    def funding_fee_poll_interval(self) -> int:
        return 600

    def supported_order_types(self) -> List[OrderType]:
        """
        :return a list of OrderType supported by this connector
        """
        return [OrderType.LIMIT, OrderType.MARKET, OrderType.LIMIT_MAKER]

    def _validate_preallocated_client_order_id_format(self, client_order_id: str) -> None:
        validate_binance_client_order_id(
            client_order_id=client_order_id,
            max_length=self.client_order_id_max_length,
        )

    def is_order_submission_unknown(self, client_order_id: str) -> bool:
        return client_order_id in self._unknown_submission_order_ids

    def _resolve_order_submission_unknown(self, client_order_id: str) -> None:
        self._unknown_submission_order_ids.discard(client_order_id)
        self._order_tracker._order_not_found_records.pop(client_order_id, None)

    @staticmethod
    def _validated_non_negative_integer_string(value: Any, field: str) -> str:
        if isinstance(value, bool):
            raise BinancePerpetualOrderDataError(f"{field} must be a non-negative integer")
        if isinstance(value, int):
            if value < 0:
                raise BinancePerpetualOrderDataError(f"{field} must be a non-negative integer")
            return str(value)
        if (
            isinstance(value, str)
            and value.isascii()
            and value.isdigit()
            and (value == "0" or not value.startswith("0"))
        ):
            return value
        raise BinancePerpetualOrderDataError(f"{field} must be a non-negative integer")

    @staticmethod
    def _validated_finite_decimal(value: Any, field: str) -> Decimal:
        parsed = None
        if isinstance(value, str) and value != "" and value.strip() == value:
            try:
                parsed = Decimal(value)
            except (InvalidOperation, TypeError, ValueError):
                pass
        if parsed is None or not parsed.is_finite():
            raise BinancePerpetualOrderDataError(f"{field} must be a finite decimal")
        return parsed

    def _validate_snapshot_matches_tracked_order(
            self,
            snapshot: BinancePerpetualOrderSnapshot,
            tracked_order: InFlightOrder,
    ) -> None:
        if snapshot.client_order_id != tracked_order.client_order_id:
            raise BinancePerpetualOrderDataError("order snapshot client order ID is contradictory")
        if snapshot.trading_pair != tracked_order.trading_pair:
            raise BinancePerpetualOrderDataError("order snapshot trading pair is contradictory")
        if (
            tracked_order.exchange_order_id is not None
            and snapshot.exchange_order_id != tracked_order.exchange_order_id
        ):
            raise BinancePerpetualOrderDataError("order snapshot exchange order ID is contradictory")
        if snapshot.side is not tracked_order.trade_type:
            raise BinancePerpetualOrderDataError("order snapshot side is contradictory")
        expected_order_type = (
            OrderType.LIMIT
            if tracked_order.order_type is OrderType.LIMIT_MAKER
            else tracked_order.order_type
        )
        if snapshot.order_type is not expected_order_type:
            raise BinancePerpetualOrderDataError("order snapshot type is contradictory")
        if snapshot.original_quantity != tracked_order.amount:
            raise BinancePerpetualOrderDataError("order snapshot quantity is contradictory")
        if tracked_order.order_type.is_limit_type() and snapshot.price != tracked_order.price:
            raise BinancePerpetualOrderDataError("order snapshot price is contradictory")

    async def _apply_authoritative_order_snapshot(
            self,
            snapshot: BinancePerpetualOrderSnapshot,
    ) -> bool:
        if snapshot.is_not_found or not self.is_order_submission_unknown(snapshot.client_order_id):
            return False
        tracked_order = self._order_tracker.all_updatable_orders.get(snapshot.client_order_id)
        if tracked_order is None:
            return False
        self._validate_snapshot_matches_tracked_order(snapshot, tracked_order)
        order_update = OrderUpdate(
            trading_pair=tracked_order.trading_pair,
            update_timestamp=snapshot.update_time_ms * 1e-3,
            new_state=CONSTANTS.ORDER_STATE[snapshot.status.value],
            client_order_id=tracked_order.client_order_id,
            exchange_order_id=snapshot.exchange_order_id,
        )
        await self._order_tracker.process_order_update(order_update)
        self._resolve_order_submission_unknown(snapshot.client_order_id)
        return True

    def restore_tracking_states(self, saved_states: Dict[str, Any]):
        super().restore_tracking_states(saved_states)
        for tracked_order in self.in_flight_orders.values():
            if tracked_order.is_pending_create and tracked_order.exchange_order_id is None:
                self._unknown_submission_order_ids.add(tracked_order.client_order_id)

    def supported_position_modes(self):
        """
        This method needs to be overridden to provide the accurate information depending on the exchange.
        """
        return [PositionMode.ONEWAY, PositionMode.HEDGE]

    def get_buy_collateral_token(self, trading_pair: str) -> str:
        trading_rule: TradingRule = self._trading_rules[trading_pair]
        return trading_rule.buy_order_collateral_token

    def get_sell_collateral_token(self, trading_pair: str) -> str:
        trading_rule: TradingRule = self._trading_rules[trading_pair]
        return trading_rule.sell_order_collateral_token

    async def get_instrument_info(self, trading_pair: str) -> BinancePerpetualInstrumentInfo:
        response = await self._api_get(path_url=CONSTANTS.EXCHANGE_INFO_URL)
        return BinancePerpetualInstrumentInfo.from_exchange_info(
            payload=response,
            trading_pair=trading_pair,
            data_time=self.current_timestamp,
        )

    async def get_account_risk_snapshot(self) -> BinancePerpetualAccountRiskSnapshot:
        response = await self._api_get(
            path_url=CONSTANTS.ACCOUNT_INFO_V3_URL,
            is_auth_required=True,
        )
        return BinancePerpetualAccountRiskSnapshot.from_payload(response, self.current_timestamp)

    async def _reconciliation_api_get(self, context: str, **request_kwargs) -> Any:
        try:
            return await self._api_get(**request_kwargs)
        except asyncio.CancelledError:
            raise
        except Exception:
            raise BinancePerpetualOrderDataError(f"{context} request failed") from None

    async def get_order_status_by_client_order_id(
            self,
            trading_pair: str,
            client_order_id: str,
    ) -> BinancePerpetualOrderSnapshot:
        validate_binance_client_order_id(client_order_id)
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        response = await self._reconciliation_api_get(
            context="order status",
            path_url=CONSTANTS.ORDER_URL,
            params={"symbol": symbol, "origClientOrderId": client_order_id},
            is_auth_required=True,
            return_err=True,
            limit_id=CONSTANTS.GET_ORDER_LIMIT_ID,
        )
        data_time = self.current_timestamp
        if isinstance(response, dict) and "code" in response:
            if response.get("code") == CONSTANTS.ORDER_NOT_EXIST_ERROR_CODE:
                return BinancePerpetualOrderSnapshot.not_found(
                    client_order_id=client_order_id,
                    symbol=symbol,
                    trading_pair=trading_pair,
                    data_time=data_time,
                )
            raise BinancePerpetualOrderDataError("order status response contains an exchange error")

        fact = BinancePerpetualOrderSnapshot.from_payload(
            payload=response,
            expected_symbol=symbol,
            trading_pair=trading_pair,
            expected_client_order_id=client_order_id,
            data_time=data_time,
        )
        await self._apply_authoritative_order_snapshot(fact)
        return fact

    async def get_open_orders(
            self,
            trading_pair: str,
    ) -> Tuple[BinancePerpetualOrderSnapshot, ...]:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        response = await self._reconciliation_api_get(
            context="open orders",
            path_url=CONSTANTS.OPEN_ORDERS_URL,
            params={"symbol": symbol},
            is_auth_required=True,
        )
        if not isinstance(response, list):
            raise BinancePerpetualOrderDataError("open orders response must be an array")

        data_time = self.current_timestamp
        facts_by_client_order_id: Dict[str, BinancePerpetualOrderSnapshot] = {}
        client_order_id_by_exchange_order_id: Dict[str, str] = {}
        for payload in response:
            fact = BinancePerpetualOrderSnapshot.from_payload(
                payload=payload,
                expected_symbol=symbol,
                trading_pair=trading_pair,
                data_time=data_time,
            )
            if fact.status not in {
                BinancePerpetualOrderStatus.NEW,
                BinancePerpetualOrderStatus.PARTIALLY_FILLED,
            }:
                raise BinancePerpetualOrderDataError("open orders response contains a non-open order")

            existing_fact = facts_by_client_order_id.get(fact.client_order_id)
            if existing_fact is not None:
                if existing_fact != fact:
                    raise BinancePerpetualOrderDataError("conflicting client order ID in open orders response")
                continue

            existing_client_order_id = client_order_id_by_exchange_order_id.get(fact.exchange_order_id)
            if existing_client_order_id is not None and existing_client_order_id != fact.client_order_id:
                raise BinancePerpetualOrderDataError("conflicting exchange order ID in open orders response")
            facts_by_client_order_id[fact.client_order_id] = fact
            client_order_id_by_exchange_order_id[fact.exchange_order_id] = fact.client_order_id

        return tuple(
            facts_by_client_order_id[client_order_id]
            for client_order_id in sorted(facts_by_client_order_id)
        )

    async def get_account_trades(
            self,
            trading_pair: str,
            exchange_order_id: Optional[str] = None,
    ) -> Tuple[BinancePerpetualTrade, ...]:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        params = {"symbol": symbol}
        if exchange_order_id is not None:
            exchange_order_id = validate_binance_exchange_order_id(exchange_order_id)
            params["orderId"] = exchange_order_id
        response = await self._reconciliation_api_get(
            context="account trades",
            path_url=CONSTANTS.ACCOUNT_TRADE_LIST_URL,
            params=params,
            is_auth_required=True,
        )
        if not isinstance(response, list):
            raise BinancePerpetualOrderDataError("account trades response must be an array")

        data_time = self.current_timestamp
        facts_by_trade_id: Dict[str, BinancePerpetualTrade] = {}
        for payload in response:
            fact = BinancePerpetualTrade.from_payload(
                payload=payload,
                expected_symbol=symbol,
                trading_pair=trading_pair,
                expected_exchange_order_id=exchange_order_id,
                data_time=data_time,
            )
            existing_fact = facts_by_trade_id.get(fact.trade_id)
            if existing_fact is not None:
                if existing_fact != fact:
                    raise BinancePerpetualOrderDataError("conflicting trade ID in account trades response")
                continue
            facts_by_trade_id[fact.trade_id] = fact

        return tuple(sorted(
            facts_by_trade_id.values(),
            key=lambda fact: (fact.timestamp_ms, len(fact.trade_id), fact.trade_id),
        ))

    async def get_position_risk_snapshots(
            self,
            trading_pair: Optional[str] = None,
    ) -> Tuple[BinancePerpetualPositionRiskSnapshot, ...]:
        params = None
        expected_symbol = None
        if trading_pair is not None:
            expected_symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
            params = {"symbol": expected_symbol}
        try:
            response = await self._api_get(
                path_url=CONSTANTS.POSITION_INFORMATION_V3_URL,
                params=params,
                is_auth_required=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise BinancePerpetualRiskDataError("Position Information V3 request failed") from None
        if not isinstance(response, list):
            raise BinancePerpetualRiskDataError("Position Information V3 response must be an array")
        data_time = self.current_timestamp
        facts_by_key: Dict[Tuple[str, str], BinancePerpetualPositionRiskSnapshot] = {}
        for payload in response:
            fact = BinancePerpetualPositionRiskSnapshot.from_payload(payload, data_time)
            if expected_symbol is not None and fact.symbol != expected_symbol:
                raise BinancePerpetualRiskDataError("Position Information V3 symbol does not match request")
            key = (fact.symbol, fact.position_side)
            existing_fact = facts_by_key.get(key)
            if existing_fact is not None:
                if existing_fact != fact:
                    raise BinancePerpetualRiskDataError("conflicting position in Position Information V3 response")
                continue
            facts_by_key[key] = fact
        return tuple(
            facts_by_key[key]
            for key in sorted(facts_by_key)
        )

    async def get_account_config(self) -> BinancePerpetualAccountConfig:
        response = await self._api_get(
            path_url=CONSTANTS.ACCOUNT_CONFIG_URL,
            is_auth_required=True,
        )
        return BinancePerpetualAccountConfig.from_payload(response, self.current_timestamp)

    async def get_symbol_config(self, trading_pair: str) -> BinancePerpetualSymbolConfig:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        response = await self._api_get(
            path_url=CONSTANTS.SYMBOL_CONFIG_URL,
            params={"symbol": symbol},
            is_auth_required=True,
        )
        records = response if isinstance(response, list) else [response]
        matches = [
            record for record in records
            if isinstance(record, dict) and record.get("symbol") == symbol
        ]
        if len(matches) != 1:
            raise BinancePerpetualRiskDataError(
                f"symbol configuration must contain exactly one record for symbol {symbol}"
            )
        return BinancePerpetualSymbolConfig.from_payload(matches[0], self.current_timestamp)

    async def get_multi_assets_mode(self) -> BinancePerpetualMultiAssetsMode:
        response = await self._api_get(
            path_url=CONSTANTS.MULTI_ASSETS_MODE_URL,
            is_auth_required=True,
        )
        return BinancePerpetualMultiAssetsMode.from_payload(response, self.current_timestamp)

    async def get_position_mode_snapshot(self) -> BinancePerpetualPositionMode:
        response = await self._api_get(
            path_url=CONSTANTS.CHANGE_POSITION_MODE_URL,
            is_auth_required=True,
            limit_id=CONSTANTS.GET_POSITION_MODE_LIMIT_ID,
        )
        return BinancePerpetualPositionMode.from_payload(response, self.current_timestamp)

    async def get_leverage_brackets(
            self,
            trading_pair: str,
            refresh: bool = False,
    ) -> BinancePerpetualLeverageBrackets:
        cached = self._leverage_bracket_cache.get(trading_pair)
        if cached is not None and not refresh:
            return cached
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        response = await self._api_get(
            path_url=CONSTANTS.LEVERAGE_BRACKET_URL,
            params={"symbol": symbol},
            is_auth_required=True,
        )
        data_time = self.current_timestamp
        brackets = BinancePerpetualLeverageBrackets.from_payload(
            payload=response,
            expected_symbol=symbol,
            data_time=data_time,
            cache_time=data_time,
        )
        self._leverage_bracket_cache[trading_pair] = brackets
        return brackets

    async def set_leverage_with_result(
            self,
            trading_pair: str,
            leverage: int,
    ) -> BinancePerpetualLeverageChangeResult:
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        response = await self._api_post(
            path_url=CONSTANTS.SET_LEVERAGE_URL,
            data={"symbol": symbol, "leverage": leverage},
            is_auth_required=True,
        )
        result = BinancePerpetualLeverageChangeResult.from_payload(response, self.current_timestamp)
        if result.symbol != symbol:
            raise BinancePerpetualRiskDataError(
                f"change leverage response symbol {result.symbol} does not match requested symbol {symbol}"
            )
        return result

    async def strict_account_preflight(
            self,
            trading_pairs: Collection[str],
            related_trading_pairs: Optional[Collection[str]] = None,
            known_position_trading_pairs: Optional[Collection[str]] = None,
            max_age_seconds: float = 5.0,
            consistency_tolerance: Decimal = Decimal("0"),
    ) -> BinancePerpetualPreflightSnapshot:
        """Fetches and validates account state without changing any exchange account setting."""
        active_values = tuple(trading_pairs)
        if not active_values:
            raise BinancePerpetualPreflightError("preflight trading_pairs must be non-empty and unique")
        if any(not isinstance(pair, str) or pair == "" for pair in active_values):
            raise BinancePerpetualPreflightError("preflight trading_pairs must contain non-empty strings")
        if len(set(active_values)) != len(active_values):
            raise BinancePerpetualPreflightError("preflight trading_pairs must be non-empty and unique")
        active_pairs = tuple(sorted(active_values))

        related_values = tuple(active_pairs if related_trading_pairs is None else related_trading_pairs)
        known_values = tuple(known_position_trading_pairs or ())
        if any(not isinstance(pair, str) or pair == "" for pair in related_values):
            raise BinancePerpetualPreflightError(
                "related_trading_pairs must contain non-empty strings"
            )
        if any(not isinstance(pair, str) or pair == "" for pair in known_values):
            raise BinancePerpetualPreflightError(
                "known_position_trading_pairs must contain non-empty strings"
            )
        related_pairs = tuple(sorted(set(related_values)))
        known_pairs = tuple(sorted(set(known_values)))
        if not set(active_pairs).issubset(related_pairs):
            raise BinancePerpetualPreflightError("related_trading_pairs must include every active trading pair")
        if not set(known_pairs).issubset(related_pairs):
            raise BinancePerpetualPreflightError("known positions must be a subset of related trading pairs")
        if isinstance(max_age_seconds, bool) or not isinstance(max_age_seconds, (int, float, Decimal)):
            raise BinancePerpetualPreflightError(
                "max_age_seconds must be a finite number between 0 and 5 seconds"
            )
        if isinstance(max_age_seconds, Decimal):
            is_finite_max_age = max_age_seconds.is_finite()
        elif isinstance(max_age_seconds, float):
            is_finite_max_age = math.isfinite(max_age_seconds)
        else:
            is_finite_max_age = True
        if (
                not is_finite_max_age
                or max_age_seconds < 0
                or max_age_seconds > self.MAX_ACCOUNT_DATA_AGE_SECONDS
        ):
            raise BinancePerpetualPreflightError(
                "max_age_seconds must be a finite number between 0 and 5 seconds"
            )
        try:
            max_age = float(max_age_seconds)
        except (OverflowError, TypeError, ValueError) as exc:
            raise BinancePerpetualPreflightError(
                "max_age_seconds must be a finite number between 0 and 5 seconds"
            ) from exc
        if not math.isfinite(max_age):
            raise BinancePerpetualPreflightError(
                "max_age_seconds must be a finite number between 0 and 5 seconds"
            )
        if (
                not isinstance(consistency_tolerance, Decimal)
                or not consistency_tolerance.is_finite()
                or consistency_tolerance < 0
                or consistency_tolerance > Decimal("0.01")
        ):
            raise BinancePerpetualPreflightError(
                "consistency_tolerance must be a finite Decimal between 0 and 0.01"
            )

        async def authoritative_fetch(source: str, awaitable):
            try:
                return await awaitable
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise BinancePerpetualPreflightError(f"authoritative {source} fetch failed") from exc

        account = await authoritative_fetch(
            "Account Information V3", self.get_account_risk_snapshot()
        )
        positions = await authoritative_fetch(
            "Position Information V3", self.get_position_risk_snapshots()
        )
        account_config = await authoritative_fetch(
            "account configuration", self.get_account_config()
        )
        multi_assets_mode = await authoritative_fetch(
            "Multi-Assets mode", self.get_multi_assets_mode()
        )
        position_mode = await authoritative_fetch(
            "position mode", self.get_position_mode_snapshot()
        )

        instruments = []
        for trading_pair in related_pairs:
            instrument = await authoritative_fetch(
                f"exchange metadata for {trading_pair}",
                self.get_instrument_info(trading_pair),
            )
            instruments.append(instrument)
        instrument_by_pair = {instrument.trading_pair: instrument for instrument in instruments}
        if len(instrument_by_pair) != len(instruments):
            raise BinancePerpetualPreflightError("exchange metadata contains duplicate trading pairs")

        symbol_configs = []
        for trading_pair in related_pairs:
            symbol_config = await authoritative_fetch(
                f"symbol configuration for {trading_pair}",
                self.get_symbol_config(trading_pair),
            )
            symbol_configs.append(symbol_config)
        symbol_config_by_symbol = {config.symbol: config for config in symbol_configs}
        if len(symbol_config_by_symbol) != len(symbol_configs):
            raise BinancePerpetualPreflightError("symbol configuration contains duplicate symbols")

        leverage_brackets = []
        for trading_pair in active_pairs:
            brackets = await authoritative_fetch(
                f"leverage brackets for {trading_pair}",
                self.get_leverage_brackets(trading_pair, refresh=True),
            )
            leverage_brackets.append(brackets)

        now = self.current_timestamp
        freshness_sources = [
            ("Account Information V3", account.data_time),
            ("account configuration", account_config.data_time),
            ("Multi-Assets mode", multi_assets_mode.data_time),
            ("position mode", position_mode.data_time),
        ]
        freshness_sources.extend(
            (f"Position Information V3 {position.symbol}", position.data_time)
            for position in positions
        )
        freshness_sources.extend(
            (f"exchange metadata {instrument.symbol}", instrument.data_time)
            for instrument in instruments
        )
        freshness_sources.extend(
            (f"symbol configuration {config.symbol}", config.data_time)
            for config in symbol_configs
        )
        for brackets in leverage_brackets:
            freshness_sources.append((f"leverage brackets {brackets.symbol}", brackets.data_time))
            freshness_sources.append((f"leverage bracket cache {brackets.symbol}", brackets.cache_time))

        # Binance documents exchangeInfo.serverTime and accountConfig.updateTime as fields to ignore;
        # their local receive times above are therefore the freshness authorities.
        def add_source_timestamp(
                source: str,
                source_time_ms: int,
                allow_zero_when_inactive: bool = False,
                is_active: bool = True,
        ) -> None:
            if source_time_ms == 0:
                if allow_zero_when_inactive and not is_active:
                    return
                raise BinancePerpetualPreflightError(
                    f"{source} source timestamp is zero for active data"
                )
            freshness_sources.append((source, source_time_ms / 1e3))

        for asset in account.assets:
            add_source_timestamp(
                f"Account Information V3 asset {asset.asset} updateTime",
                asset.update_time_ms,
                allow_zero_when_inactive=True,
                is_active=asset.has_activity,
            )
        for position in account.positions:
            add_source_timestamp(
                f"Account Information V3 position {position.symbol} updateTime",
                position.update_time_ms,
                allow_zero_when_inactive=True,
                is_active=position.has_activity,
            )
        for position in positions:
            add_source_timestamp(
                f"Position Information V3 {position.symbol} updateTime",
                position.update_time_ms,
                allow_zero_when_inactive=True,
                is_active=position.has_activity,
            )
        for source, source_time in freshness_sources:
            age = now - source_time
            if not math.isfinite(age) or age < 0 or age > max_age:
                raise BinancePerpetualPreflightError(f"{source} snapshot is stale or future-dated")

        if not account_config.can_trade:
            raise BinancePerpetualPreflightError("account canTrade is false")
        if not multi_assets_mode.enabled:
            raise BinancePerpetualPreflightError("account is not in Multi-Assets mode")
        if not position_mode.is_one_way:
            raise BinancePerpetualPreflightError("account is not in authoritative One-way mode")
        if account_config.multi_assets_margin != multi_assets_mode.enabled:
            raise BinancePerpetualPreflightError("account configuration disagrees with Multi-Assets mode")
        if account_config.dual_side_position != position_mode.dual_side_position:
            raise BinancePerpetualPreflightError("account configuration disagrees with position mode")

        active_symbols = set()
        for trading_pair in active_pairs:
            instrument = instrument_by_pair[trading_pair]
            active_symbols.add(instrument.symbol)
            if instrument.contract_type != "TRADIFI_PERPETUAL":
                raise BinancePerpetualPreflightError(
                    f"{trading_pair} contract type is not TRADIFI_PERPETUAL"
                )
            if instrument.status != "TRADING":
                raise BinancePerpetualPreflightError(f"{trading_pair} status is not TRADING")
            if instrument.quote_asset != "USDT" or instrument.margin_asset != "USDT":
                raise BinancePerpetualPreflightError(f"{trading_pair} quote and margin assets must be USDT")
            if instrument.contract_multiplier <= 0:
                raise BinancePerpetualPreflightError(f"{trading_pair} contract multiplier must be positive")
            rule = self._trading_rules.get(trading_pair)
            if rule is None:
                raise BinancePerpetualPreflightError(f"{trading_pair} trading rule is not initialized")
            rule_fields = (
                ("min_order_size", rule.min_order_size, instrument.min_order_size),
                ("min_base_amount_increment", rule.min_base_amount_increment, instrument.step_size),
                ("min_price_increment", rule.min_price_increment, instrument.tick_size),
                ("min_notional_size", rule.min_notional_size, instrument.min_notional),
            )
            for field, actual, expected in rule_fields:
                if (
                        not isinstance(actual, Decimal)
                        or not actual.is_finite()
                        or actual <= 0
                        or actual != expected
                ):
                    raise BinancePerpetualPreflightError(
                        f"{trading_pair} trading rule {field} is invalid or inconsistent"
                    )

        related_symbols = {instrument.symbol for instrument in instruments}
        known_symbols = {instrument_by_pair[pair].symbol for pair in known_pairs}
        for instrument in instruments:
            config = symbol_config_by_symbol.get(instrument.symbol)
            if config is None:
                raise BinancePerpetualPreflightError(
                    f"missing symbol configuration for {instrument.symbol}"
                )
            if config.margin_type != BinancePerpetualMarginType.CROSSED:
                raise BinancePerpetualPreflightError(
                    f"{instrument.trading_pair} is not in Cross margin mode"
                )

        position_by_key = {}
        for position in positions:
            key = (position.symbol, position.position_side)
            if key in position_by_key:
                raise BinancePerpetualPreflightError(f"duplicate position snapshot for {key}")
            position_by_key[key] = position
            if position.symbol in related_symbols:
                if position.position_side != "BOTH":
                    raise BinancePerpetualPreflightError(
                        f"related position {position.symbol} is not in BOTH/One-way state"
                    )
                if position.margin_asset != "USDT":
                    raise BinancePerpetualPreflightError(
                        f"related position {position.symbol} margin asset is not USDT"
                    )
                if position.has_activity and position.symbol not in known_symbols:
                    raise BinancePerpetualPreflightError(
                        f"related unknown position or open order exists for {position.symbol}"
                    )

        account_position_by_key = {}
        for position in account.positions:
            key = (position.symbol, position.position_side)
            if key in account_position_by_key:
                raise BinancePerpetualPreflightError(f"duplicate Account V3 position for {key}")
            account_position_by_key[key] = position
            if position.symbol in related_symbols:
                if position.position_side != "BOTH":
                    raise BinancePerpetualPreflightError(
                        f"related Account V3 position {position.symbol} is not in BOTH/One-way state"
                    )
                if position.has_activity and position.symbol not in known_symbols:
                    raise BinancePerpetualPreflightError(
                        f"related unknown position or open order exists for {position.symbol}"
                    )

        def reconciles(left: Decimal, right: Decimal) -> bool:
            return abs(left - right) <= consistency_tolerance

        if set(account_position_by_key) != set(position_by_key):
            raise BinancePerpetualPreflightError(
                "Account V3 and Position V3 position sets do not reconcile"
            )
        for key, account_position in account_position_by_key.items():
            position = position_by_key[key]
            comparisons = (
                (account_position.position_amount, position.position_amount),
                (account_position.unrealized_profit, position.unrealized_profit),
                (account_position.initial_margin, position.initial_margin),
                (account_position.maint_margin, position.maint_margin),
                (position.initial_margin,
                 position.position_initial_margin + position.open_order_initial_margin),
            )
            if any(not reconciles(left, right) for left, right in comparisons):
                raise BinancePerpetualPreflightError(
                    f"Account V3 and Position V3 values do not reconcile for {key}"
                )

        total_position_initial_margin = sum(
            (position.position_initial_margin for position in positions),
            Decimal("0"),
        )
        total_open_order_initial_margin = sum(
            (position.open_order_initial_margin for position in positions),
            Decimal("0"),
        )
        total_initial_margin = sum(
            (position.initial_margin for position in positions),
            Decimal("0"),
        )
        total_maint_margin = sum(
            (position.maint_margin for position in positions),
            Decimal("0"),
        )
        total_unrealized_profit = sum(
            (position.unrealized_profit for position in positions),
            Decimal("0"),
        )
        account_reconciliations = (
            (account.total_position_initial_margin, total_position_initial_margin),
            (account.total_open_order_initial_margin, total_open_order_initial_margin),
            (account.total_initial_margin, total_initial_margin),
            (account.total_maint_margin, total_maint_margin),
            (account.total_unrealized_profit, total_unrealized_profit),
            (account.total_initial_margin,
             account.total_position_initial_margin + account.total_open_order_initial_margin),
            (account.total_margin_balance,
             account.total_wallet_balance + account.total_unrealized_profit),
        )
        if any(not reconciles(left, right) for left, right in account_reconciliations):
            raise BinancePerpetualPreflightError("account and per-symbol risk totals do not reconcile")
        if account.available_balance > account.total_margin_balance + consistency_tolerance:
            raise BinancePerpetualPreflightError("availableBalance exceeds totalMarginBalance")

        bracket_by_symbol = {brackets.symbol: brackets for brackets in leverage_brackets}
        if len(bracket_by_symbol) != len(leverage_brackets):
            raise BinancePerpetualPreflightError("duplicate leverage bracket symbols")
        for symbol in active_symbols:
            if symbol not in bracket_by_symbol:
                raise BinancePerpetualPreflightError(f"missing leverage brackets for {symbol}")
            config = symbol_config_by_symbol[symbol]
            brackets = bracket_by_symbol[symbol]
            current_notional = max(
                (
                    abs(position.notional)
                    for position in positions
                    if position.symbol == symbol
                ),
                default=Decimal("0"),
            )
            try:
                current_bracket = brackets.bracket_for_notional(current_notional)
            except BinancePerpetualRiskDataError as exc:
                raise BinancePerpetualPreflightError(
                    f"current notional is outside leverage brackets for {symbol}"
                ) from exc
            if config.leverage > current_bracket.initial_leverage:
                raise BinancePerpetualPreflightError(
                    f"configured leverage is not allowed by the current notional bracket for {symbol}"
                )
            try:
                authoritative_cap = brackets.max_notional_for_leverage(config.leverage)
            except BinancePerpetualRiskDataError as exc:
                raise BinancePerpetualPreflightError(
                    f"configured leverage is not allowed by leverage brackets for {symbol}"
                ) from exc
            if config.max_notional_value != authoritative_cap:
                raise BinancePerpetualPreflightError(
                    f"maxNotionalValue disagrees with leverage brackets for {symbol}"
                )

        return BinancePerpetualPreflightSnapshot(
            account=account,
            positions=tuple(positions),
            account_config=account_config,
            multi_assets_mode=multi_assets_mode,
            position_mode=position_mode,
            instruments=tuple(instruments),
            symbol_configs=tuple(symbol_configs),
            leverage_brackets=tuple(leverage_brackets),
            data_time=min(source_time for _, source_time in freshness_sources),
        )

    def _is_request_exception_related_to_time_synchronizer(self, request_exception: Exception):
        error_description = str(request_exception)
        is_time_synchronizer_related = ("-1021" in error_description
                                        and "Timestamp for this request" in error_description)
        return is_time_synchronizer_related

    def _is_order_not_found_during_status_update_error(self, status_update_exception: Exception) -> bool:
        return str(CONSTANTS.ORDER_NOT_EXIST_ERROR_CODE) in str(
            status_update_exception
        ) and CONSTANTS.ORDER_NOT_EXIST_MESSAGE in str(status_update_exception)

    def _is_order_not_found_during_cancelation_error(self, cancelation_exception: Exception) -> bool:
        return str(CONSTANTS.UNKNOWN_ORDER_ERROR_CODE) in str(
            cancelation_exception
        ) and CONSTANTS.UNKNOWN_ORDER_MESSAGE in str(cancelation_exception)

    def _create_web_assistants_factory(self) -> WebAssistantsFactory:
        return web_utils.build_api_factory(
            throttler=self._throttler,
            time_synchronizer=self._time_synchronizer,
            domain=self._domain,
            auth=self._auth)

    def _create_order_book_data_source(self) -> OrderBookTrackerDataSource:
        return BinancePerpetualAPIOrderBookDataSource(
            trading_pairs=self._trading_pairs,
            connector=self,
            api_factory=self._web_assistants_factory,
            domain=self.domain,
        )

    def _create_user_stream_data_source(self) -> UserStreamTrackerDataSource:
        return BinancePerpetualUserStreamDataSource(
            auth=self._auth,
            connector=self,
            api_factory=self._web_assistants_factory,
            domain=self.domain,
        )

    def _get_fee(self,
                 base_currency: str,
                 quote_currency: str,
                 order_type: OrderType,
                 order_side: TradeType,
                 position_action: PositionAction,
                 amount: Decimal,
                 price: Decimal = s_decimal_NaN,
                 is_maker: Optional[bool] = None) -> TradeFeeBase:
        is_maker = is_maker or False
        fee = build_trade_fee(
            self.name,
            is_maker,
            base_currency=base_currency,
            quote_currency=quote_currency,
            order_type=order_type,
            order_side=order_side,
            amount=amount,
            price=price,
        )
        return fee

    async def _update_trading_fees(self):
        """
        Update fees information from the exchange
        """
        pass

    async def _status_polling_loop_fetch_updates(self):
        await safe_gather(
            self._update_order_fills_from_trades(),
            self._update_order_status(),
            self._update_balances(),
            self._update_positions(),
        )

    async def _place_cancel(self, order_id: str, tracked_order: InFlightOrder):
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair=tracked_order.trading_pair)
        api_params = {
            "origClientOrderId": order_id,
            "symbol": symbol,
        }
        cancel_result = await self._api_delete(
            path_url=CONSTANTS.ORDER_URL,
            params=api_params,
            is_auth_required=True)
        is_cancel_not_found = (
            cancel_result.get("code") == CONSTANTS.UNKNOWN_ORDER_ERROR_CODE
            and CONSTANTS.UNKNOWN_ORDER_MESSAGE in str(cancel_result.get("msg", ""))
        )
        is_order_not_found = (
            cancel_result.get("code") == CONSTANTS.ORDER_NOT_EXIST_ERROR_CODE
            and CONSTANTS.ORDER_NOT_EXIST_MESSAGE in str(cancel_result.get("msg", ""))
        )
        if is_cancel_not_found or is_order_not_found:
            if self.is_order_submission_unknown(order_id):
                self.logger().debug(
                    f"Cancel found no authoritative order for submission-unknown order {order_id}."
                )
                return False
            if is_cancel_not_found:
                self.logger().debug(
                    f"The order {order_id} does not exist on Binance Perpetuals. "
                    f"No cancelation needed."
                )
                raise IOError(
                    f"{CONSTANTS.UNKNOWN_ORDER_ERROR_CODE} - {CONSTANTS.UNKNOWN_ORDER_MESSAGE}"
                )
            return False
        if cancel_result.get("status") == "CANCELED":
            if self.is_order_submission_unknown(order_id):
                if cancel_result.get("clientOrderId") != order_id:
                    raise BinancePerpetualOrderDataError(
                        "cancel confirmation client order ID is contradictory"
                    )
                exchange_order_id = self._validated_non_negative_integer_string(
                    cancel_result.get("orderId"),
                    "cancel confirmation exchange order ID",
                )
                if (
                    tracked_order.exchange_order_id is not None
                    and exchange_order_id != tracked_order.exchange_order_id
                ):
                    raise BinancePerpetualOrderDataError(
                        "cancel confirmation exchange order ID is contradictory"
                    )
            return True
        return False

    async def _execute_order_cancel(self, order: InFlightOrder) -> Optional[str]:
        if not self.is_order_submission_unknown(order.client_order_id):
            return await super()._execute_order_cancel(order)
        try:
            cancelled = await self._place_cancel(order.client_order_id, order)
            if not cancelled:
                return None
            update_timestamp = self.current_timestamp
            if update_timestamp is None or math.isnan(update_timestamp):
                update_timestamp = self._time()
            await self._order_tracker.process_order_update(OrderUpdate(
                client_order_id=order.client_order_id,
                trading_pair=order.trading_pair,
                update_timestamp=update_timestamp,
                new_state=CONSTANTS.ORDER_STATE["CANCELED"],
            ))
            self._resolve_order_submission_unknown(order.client_order_id)
            return order.client_order_id
        except asyncio.CancelledError:
            raise
        except Exception:
            self.logger().warning(
                f"Cancel did not resolve submission-unknown order {order.client_order_id}."
            )
            return None

    async def _place_order(
            self,
            order_id: str,
            trading_pair: str,
            amount: Decimal,
            trade_type: TradeType,
            order_type: OrderType,
            price: Decimal,
            position_action: PositionAction = PositionAction.NIL,
            **kwargs,
    ) -> Tuple[str, float]:

        post_only = kwargs.get("post_only", False)
        if type(post_only) is not bool:
            raise ValueError("post-only flag must be a boolean")
        if post_only and order_type is OrderType.MARKET:
            raise ValueError("post-only order cannot use MARKET order type")
        if post_only and order_type is not OrderType.LIMIT_MAKER:
            raise ValueError("post-only order must use LIMIT_MAKER order type")

        amount_str = f"{amount:f}"
        price_str = f"{price:f}"
        symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        api_params = {"symbol": symbol,
                      "side": "BUY" if trade_type is TradeType.BUY else "SELL",
                      "quantity": amount_str,
                      "type": "MARKET" if order_type is OrderType.MARKET else "LIMIT",
                      "newClientOrderId": order_id
                      }
        if order_type.is_limit_type():
            api_params["price"] = price_str
        if order_type == OrderType.LIMIT:
            api_params["timeInForce"] = CONSTANTS.TIME_IN_FORCE_GTC
        if order_type == OrderType.LIMIT_MAKER:
            api_params["timeInForce"] = CONSTANTS.TIME_IN_FORCE_GTX
        if self.position_mode == PositionMode.HEDGE:
            if position_action == PositionAction.OPEN:
                api_params["positionSide"] = "LONG" if trade_type is TradeType.BUY else "SHORT"
            else:
                api_params["positionSide"] = "SHORT" if trade_type is TradeType.BUY else "LONG"
        elif position_action == PositionAction.CLOSE:
            # In ONEWAY mode, reduceOnly ensures the order can only reduce the position,
            # never open a new one or flip direction. This prevents over-selling.
            api_params["reduceOnly"] = "true"
        try:
            order_result = await self._api_post(
                path_url=CONSTANTS.ORDER_URL,
                data=api_params,
                is_auth_required=True)
            o_id = str(order_result["orderId"])
            transact_time = order_result["updateTime"] * 1e-3
            self._unknown_submission_order_ids.discard(order_id)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            self._unknown_submission_order_ids.add(order_id)
            raise BinancePerpetualOrderSubmissionUnknown(order_id) from None
        except IOError as e:
            error_description = str(e)
            is_server_overloaded = ("status is 503" in error_description
                                    and "Unknown error, please check your request or try again later." in error_description)
            if is_server_overloaded:
                self._unknown_submission_order_ids.add(order_id)
                raise BinancePerpetualOrderSubmissionUnknown(order_id) from None
            raise
        return o_id, transact_time

    def _on_order_failure(
            self,
            order_id: str,
            trading_pair: str,
            amount: Decimal,
            trade_type: TradeType,
            order_type: OrderType,
            price: Optional[Decimal],
            exception: Exception,
            **kwargs,
    ):
        if isinstance(exception, BinancePerpetualOrderSubmissionUnknown):
            self._unknown_submission_order_ids.add(order_id)
            self.logger().warning(
                f"Submission outcome is unknown for order {order_id}; reconcile it before any retry."
            )
            return
        super()._on_order_failure(
            order_id=order_id,
            trading_pair=trading_pair,
            amount=amount,
            trade_type=trade_type,
            order_type=order_type,
            price=price,
            exception=exception,
            **kwargs,
        )

    async def _handle_update_error_for_active_order(self, order: InFlightOrder, error: Exception):
        if self.is_order_submission_unknown(order.client_order_id):
            self.logger().debug(
                f"Order {order.client_order_id} remains submission-unknown after a status polling error."
            )
            return
        await super()._handle_update_error_for_active_order(order=order, error=error)

    async def _all_trade_updates_for_order(self, order: InFlightOrder) -> List[TradeUpdate]:
        trade_updates = []
        try:
            exchange_order_id = await order.get_exchange_order_id()
            trading_pair = await self.exchange_symbol_associated_to_pair(trading_pair=order.trading_pair)
            all_fills_response = await self._api_get(
                path_url=CONSTANTS.ACCOUNT_TRADE_LIST_URL,
                params={
                    "symbol": trading_pair,
                },
                is_auth_required=True)

            for trade in all_fills_response:
                order_id = str(trade.get("orderId"))
                if order_id == exchange_order_id:
                    position_side = trade["positionSide"]
                    position_action = (PositionAction.OPEN
                                       if (order.trade_type is TradeType.BUY and position_side == "LONG"
                                           or order.trade_type is TradeType.SELL and position_side == "SHORT")
                                       else PositionAction.CLOSE)
                    fee = TradeFeeBase.new_perpetual_fee(
                        fee_schema=self.trade_fee_schema(),
                        position_action=position_action,
                        percent_token=trade["commissionAsset"],
                        flat_fees=[TokenAmount(amount=Decimal(trade["commission"]), token=trade["commissionAsset"])]
                    )
                    trade_update: TradeUpdate = TradeUpdate(
                        trade_id=str(trade["id"]),
                        client_order_id=order.client_order_id,
                        exchange_order_id=trade["orderId"],
                        trading_pair=order.trading_pair,
                        fill_timestamp=trade["time"] * 1e-3,
                        fill_price=Decimal(trade["price"]),
                        fill_base_amount=Decimal(trade["qty"]),
                        fill_quote_amount=Decimal(trade["quoteQty"]),
                        fee=fee,
                    )
                    trade_updates.append(trade_update)

        except asyncio.TimeoutError:
            raise IOError(f"Skipped order update with order fills for {order.client_order_id} "
                          "- waiting for exchange order id.")

        return trade_updates

    async def _request_order_status(self, tracked_order: InFlightOrder) -> OrderUpdate:
        trading_pair = await self.exchange_symbol_associated_to_pair(trading_pair=tracked_order.trading_pair)
        order_update = await self._api_get(
            path_url=CONSTANTS.ORDER_URL,
            params={
                "symbol": trading_pair,
                "origClientOrderId": tracked_order.client_order_id
            },
            is_auth_required=True,
            limit_id=CONSTANTS.GET_ORDER_LIMIT_ID)
        if "code" in order_update:
            if self._is_request_exception_related_to_time_synchronizer(request_exception=order_update):
                _order_update = OrderUpdate(
                    trading_pair=tracked_order.trading_pair,
                    update_timestamp=self.current_timestamp,
                    new_state=tracked_order.current_state,
                    client_order_id=tracked_order.client_order_id,
                )
                return _order_update
        _order_update: OrderUpdate = OrderUpdate(
            trading_pair=tracked_order.trading_pair,

            update_timestamp=order_update["updateTime"] * 1e-3,
            new_state=CONSTANTS.ORDER_STATE[order_update["status"]],
            client_order_id=order_update["clientOrderId"],
            exchange_order_id=order_update["orderId"],
        )
        return _order_update

    async def _iter_user_event_queue(self) -> AsyncIterable[Dict[str, any]]:
        while True:
            try:
                yield await self._user_stream_tracker.user_stream.get()
            except asyncio.CancelledError:
                raise
            except Exception:
                self.logger().network(
                    "Unknown error. Retrying after 1 seconds.",
                    exc_info=True,
                    app_warning_msg="Could not fetch user events from Binance. Check API key and network connection.",
                )
                await self._sleep(1.0)

    async def _user_stream_event_listener(self):
        """
        Wait for new messages from _user_stream_tracker.user_stream queue and processes them according to their
        message channels. The respective UserStreamDataSource queues these messages.
        """
        async for event_message in self._iter_user_event_queue():
            try:
                await self._process_user_stream_event(event_message)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self.logger().error(f"Unexpected error in user stream listener loop: {e}", exc_info=True)
                await self._sleep(5.0)

    async def _validated_unknown_user_stream_updates(
            self,
            event_message: Mapping[str, Any],
            order_message: Mapping[str, Any],
            tracked_order: InFlightOrder,
    ) -> Tuple[Optional[TradeUpdate], OrderUpdate]:
        client_order_id = validate_binance_client_order_id(order_message.get("c"))
        if client_order_id != tracked_order.client_order_id:
            raise BinancePerpetualOrderDataError("user stream client order ID is contradictory")

        exchange_order_id = self._validated_non_negative_integer_string(
            order_message.get("i"),
            "user stream exchange order ID",
        )
        if (
            tracked_order.exchange_order_id is not None
            and exchange_order_id != tracked_order.exchange_order_id
        ):
            raise BinancePerpetualOrderDataError("user stream exchange order ID is contradictory")

        expected_symbol = await self.exchange_symbol_associated_to_pair(tracked_order.trading_pair)
        if order_message.get("s") not in {expected_symbol, tracked_order.trading_pair}:
            raise BinancePerpetualOrderDataError("user stream symbol is contradictory")
        expected_side = "BUY" if tracked_order.trade_type is TradeType.BUY else "SELL"
        if order_message.get("S") != expected_side:
            raise BinancePerpetualOrderDataError("user stream side is contradictory")
        expected_order_type = "MARKET" if tracked_order.order_type is OrderType.MARKET else "LIMIT"
        if order_message.get("o") != expected_order_type:
            raise BinancePerpetualOrderDataError("user stream order type is contradictory")

        original_quantity = self._validated_finite_decimal(
            order_message.get("q"),
            "user stream order quantity",
        )
        if original_quantity <= 0 or original_quantity != tracked_order.amount:
            raise BinancePerpetualOrderDataError("user stream order quantity is contradictory")
        order_price = self._validated_finite_decimal(
            order_message.get("p"),
            "user stream order price",
        )
        if tracked_order.order_type.is_limit_type():
            if order_price <= 0 or order_price != tracked_order.price:
                raise BinancePerpetualOrderDataError("user stream order price is contradictory")
        elif order_price != 0:
            raise BinancePerpetualOrderDataError("user stream order price is contradictory")

        raw_status = order_message.get("X")
        execution_type = order_message.get("x")
        execution_statuses = {
            "NEW": {"NEW"},
            "TRADE": {"PARTIALLY_FILLED", "FILLED"},
            "CANCELED": {"CANCELED"},
            "EXPIRED": {"EXPIRED", "EXPIRED_IN_MATCH"},
        }
        if (
            not isinstance(execution_type, str)
            or not isinstance(raw_status, str)
            or raw_status not in execution_statuses.get(execution_type, set())
        ):
            raise BinancePerpetualOrderDataError(
                "user stream execution type and order status are contradictory"
            )
        event_timestamp_ms = int(self._validated_non_negative_integer_string(
            event_message.get("T"),
            "user stream event timestamp",
        ))
        fill_timestamp_ms = int(self._validated_non_negative_integer_string(
            order_message.get("T"),
            "user stream fill timestamp",
        ))
        trade_id = self._validated_non_negative_integer_string(
            order_message.get("t"),
            "user stream trade ID",
        )
        last_fill_base_amount = self._validated_finite_decimal(
            order_message.get("l"),
            "user stream last fill quantity",
        )
        cumulative_fill_base_amount = self._validated_finite_decimal(
            order_message.get("z"),
            "user stream cumulative fill quantity",
        )
        average_fill_price = self._validated_finite_decimal(
            order_message.get("ap"),
            "user stream average fill price",
        )
        last_fill_price = self._validated_finite_decimal(
            order_message.get("L"),
            "user stream last fill price",
        )
        cumulative_fill_quote_amount = self._validated_finite_decimal(
            order_message.get("Z"),
            "user stream cumulative fill quote amount",
        )
        if any(
            value < 0
            for value in (
                last_fill_base_amount,
                cumulative_fill_base_amount,
                average_fill_price,
                last_fill_price,
                cumulative_fill_quote_amount,
            )
        ):
            raise BinancePerpetualOrderDataError(
                "user stream fill quantities and prices must be non-negative"
            )
        if cumulative_fill_base_amount > original_quantity:
            raise BinancePerpetualOrderDataError(
                "user stream cumulative fill quantity exceeds the order quantity"
            )
        if (
            cumulative_fill_base_amount < tracked_order.executed_amount_base
            or cumulative_fill_quote_amount < tracked_order.executed_amount_quote
        ):
            raise BinancePerpetualOrderDataError("user stream cumulative fill rolled back")
        if cumulative_fill_base_amount == 0:
            if average_fill_price != 0 or cumulative_fill_quote_amount != 0:
                raise BinancePerpetualOrderDataError(
                    "user stream zero cumulative fill has contradictory price facts"
                )
        elif (
            average_fill_price <= 0
            or cumulative_fill_quote_amount <= 0
            or average_fill_price * cumulative_fill_base_amount != cumulative_fill_quote_amount
        ):
            raise BinancePerpetualOrderDataError(
                "user stream cumulative fill price facts are contradictory"
            )

        trade_update = None
        if execution_type == "NEW":
            if (
                trade_id != "0"
                or last_fill_base_amount != 0
                or last_fill_price != 0
                or cumulative_fill_base_amount != 0
                or tracked_order.executed_amount_base != 0
                or tracked_order.executed_amount_quote != 0
            ):
                raise BinancePerpetualOrderDataError(
                    "user stream new order contains contradictory fill facts"
                )
        elif execution_type == "TRADE":
            if trade_id == "0" or last_fill_price <= 0 or last_fill_base_amount <= 0:
                raise BinancePerpetualOrderDataError(
                    "user stream fill price and quantity must be positive"
                )
            if (
                raw_status == "PARTIALLY_FILLED"
                and not Decimal("0") < cumulative_fill_base_amount < original_quantity
            ):
                raise BinancePerpetualOrderDataError(
                    "user stream partial fill quantity is contradictory"
                )
            if raw_status == "FILLED" and cumulative_fill_base_amount != original_quantity:
                raise BinancePerpetualOrderDataError(
                    "user stream filled quantity is contradictory"
                )
            fee_amount = self._validated_finite_decimal(
                order_message.get("n", "0"),
                "user stream fee amount",
            )
            if fee_amount < 0:
                raise BinancePerpetualOrderDataError("user stream fee amount must be non-negative")
            fee_asset = order_message.get("N") or tracked_order.quote_asset
            if not isinstance(fee_asset, str) or fee_asset == "":
                raise BinancePerpetualOrderDataError("user stream fee asset must be a string")
            position_side = order_message.get("ps", "LONG")
            if position_side not in {"BOTH", "LONG", "SHORT"}:
                raise BinancePerpetualOrderDataError("user stream position side is unsupported")
            position_action = (
                PositionAction.OPEN
                if (
                    tracked_order.trade_type is TradeType.BUY and position_side == "LONG"
                    or tracked_order.trade_type is TradeType.SELL and position_side == "SHORT"
                )
                else PositionAction.CLOSE
            )
            flat_fees = [] if fee_amount == Decimal("0") else [
                TokenAmount(amount=fee_amount, token=fee_asset)
            ]
            fee = TradeFeeBase.new_perpetual_fee(
                fee_schema=self.trade_fee_schema(),
                position_action=position_action,
                percent_token=fee_asset,
                flat_fees=flat_fees,
            )
            candidate_trade_update = TradeUpdate(
                trade_id=trade_id,
                client_order_id=client_order_id,
                exchange_order_id=exchange_order_id,
                trading_pair=tracked_order.trading_pair,
                fill_timestamp=fill_timestamp_ms * 1e-3,
                fill_price=last_fill_price,
                fill_base_amount=last_fill_base_amount,
                fill_quote_amount=last_fill_price * last_fill_base_amount,
                fee=fee,
            )
            existing_fill = tracked_order.order_fills.get(trade_id)
            if existing_fill is not None:
                if (
                    candidate_trade_update != existing_fill
                    or cumulative_fill_base_amount != tracked_order.executed_amount_base
                    or cumulative_fill_quote_amount != tracked_order.executed_amount_quote
                ):
                    raise BinancePerpetualOrderDataError(
                        "user stream duplicate trade facts are contradictory"
                    )
            else:
                if (
                    cumulative_fill_base_amount
                    != tracked_order.executed_amount_base + last_fill_base_amount
                    or cumulative_fill_quote_amount
                    != tracked_order.executed_amount_quote + candidate_trade_update.fill_quote_amount
                ):
                    raise BinancePerpetualOrderDataError(
                        "user stream cumulative and last fill facts are contradictory"
                    )
                trade_update = candidate_trade_update
        else:
            if (
                trade_id != "0"
                or last_fill_base_amount != 0
                or last_fill_price != 0
                or cumulative_fill_base_amount != tracked_order.executed_amount_base
                or cumulative_fill_quote_amount != tracked_order.executed_amount_quote
            ):
                raise BinancePerpetualOrderDataError(
                    "user stream terminal order contains contradictory fill facts"
                )

        order_update = OrderUpdate(
            trading_pair=tracked_order.trading_pair,
            update_timestamp=event_timestamp_ms * 1e-3,
            new_state=CONSTANTS.ORDER_STATE[raw_status],
            client_order_id=client_order_id,
            exchange_order_id=exchange_order_id,
        )
        return trade_update, order_update

    async def _process_user_stream_event(self, event_message: Dict[str, Any]):
        event_type = event_message.get("e")
        if event_type == "ORDER_TRADE_UPDATE":
            order_message = event_message.get("o")
            if not isinstance(order_message, Mapping):
                raise BinancePerpetualOrderDataError("user stream order update must be an object")
            client_order_id = order_message.get("c", None)
            if self.is_order_submission_unknown(client_order_id):
                tracked_order = self._order_tracker.all_updatable_orders.get(client_order_id)
                if tracked_order is None:
                    return
                trade_update, order_update = await self._validated_unknown_user_stream_updates(
                    event_message=event_message,
                    order_message=order_message,
                    tracked_order=tracked_order,
                )
                if trade_update is not None:
                    self._order_tracker.process_trade_update(trade_update)
                    if tracked_order.order_fills.get(trade_update.trade_id) != trade_update:
                        raise BinancePerpetualOrderDataError(
                            "user stream trade update was not applied"
                        )
                await self._order_tracker.process_order_update(order_update)
                if (
                    tracked_order.exchange_order_id != order_update.exchange_order_id
                    or tracked_order.current_state != order_update.new_state
                ):
                    raise BinancePerpetualOrderDataError(
                        "user stream order update was not applied"
                    )
                self._resolve_order_submission_unknown(client_order_id)
                return
            tracked_order = self._order_tracker.all_fillable_orders.get(client_order_id)
            if tracked_order is not None:
                trade_id: str = str(order_message["t"])

                if trade_id != "0":  # Indicates that there has been a trade

                    fee_asset = order_message.get("N", tracked_order.quote_asset)
                    fee_amount = Decimal(order_message.get("n", "0"))
                    position_side = order_message.get("ps", "LONG")
                    position_action = (PositionAction.OPEN
                                       if (tracked_order.trade_type is TradeType.BUY and position_side == "LONG"
                                           or tracked_order.trade_type is TradeType.SELL and position_side == "SHORT")
                                       else PositionAction.CLOSE)
                    flat_fees = [] if fee_amount == Decimal("0") else [TokenAmount(amount=fee_amount, token=fee_asset)]

                    fee = TradeFeeBase.new_perpetual_fee(
                        fee_schema=self.trade_fee_schema(),
                        position_action=position_action,
                        percent_token=fee_asset,
                        flat_fees=flat_fees,
                    )

                    trade_update: TradeUpdate = TradeUpdate(
                        trade_id=trade_id,
                        client_order_id=client_order_id,
                        exchange_order_id=str(order_message["i"]),
                        trading_pair=tracked_order.trading_pair,
                        fill_timestamp=order_message["T"] * 1e-3,
                        fill_price=Decimal(order_message["L"]),
                        fill_base_amount=Decimal(order_message["l"]),
                        fill_quote_amount=Decimal(order_message["L"]) * Decimal(order_message["l"]),
                        fee=fee,
                    )
                    self._order_tracker.process_trade_update(trade_update)

            tracked_order = self._order_tracker.all_updatable_orders.get(client_order_id)
            if tracked_order is not None:
                order_update: OrderUpdate = OrderUpdate(
                    trading_pair=tracked_order.trading_pair,
                    update_timestamp=event_message["T"] * 1e-3,
                    new_state=CONSTANTS.ORDER_STATE[order_message["X"]],
                    client_order_id=client_order_id,
                    exchange_order_id=str(order_message["i"]),
                )

                self._order_tracker.process_order_update(order_update)

        elif event_type == "ACCOUNT_UPDATE":
            update_data = event_message.get("a", {})
            # update balances
            for asset in update_data.get("B", []):
                asset_name = asset["a"]
                self._account_balances[asset_name] = Decimal(asset["wb"])
                self._account_available_balances[asset_name] = Decimal(asset["cw"])

            # update position
            for asset in update_data.get("P", []):
                trading_pair = asset["s"]
                try:
                    hb_trading_pair = await self.trading_pair_associated_to_exchange_symbol(trading_pair)
                except KeyError:
                    # Ignore results for which their symbols is not tracked by the connector
                    continue

                side = PositionSide[asset['ps']]
                position = self._perpetual_trading.get_position(hb_trading_pair, side)
                if position is not None:
                    amount = Decimal(asset["pa"])
                    if amount == Decimal("0"):
                        pos_key = self._perpetual_trading.position_key(hb_trading_pair, side)
                        self._perpetual_trading.remove_position(pos_key)
                    else:
                        position.update_position(position_side=PositionSide[asset["ps"]],
                                                 unrealized_pnl=Decimal(asset["up"]),
                                                 entry_price=Decimal(asset["ep"]),
                                                 amount=Decimal(asset["pa"]))
                else:
                    await self._update_positions()
        elif event_type == "MARGIN_CALL":
            positions = event_message.get("p", [])
            total_maint_margin_required = Decimal(0)
            # total_pnl = 0
            negative_pnls_msg = ""
            for position in positions:
                trading_pair = position["s"]
                try:
                    hb_trading_pair = await self.trading_pair_associated_to_exchange_symbol(trading_pair)
                except KeyError:
                    # Ignore results for which their symbols is not tracked by the connector
                    continue
                existing_position = self._perpetual_trading.get_position(hb_trading_pair, PositionSide[position['ps']])
                if existing_position is not None:
                    existing_position.update_position(position_side=PositionSide[position["ps"]],
                                                      unrealized_pnl=Decimal(position["up"]),
                                                      amount=Decimal(position["pa"]))
                total_maint_margin_required += Decimal(position.get("mm", "0"))
                if float(position.get("up", 0)) < 1:
                    negative_pnls_msg += f"{hb_trading_pair}: {position.get('up')}, "
            self.logger().warning("Margin Call: Your position risk is too high, and you are at risk of "
                                  "liquidation. Close your positions or add additional margin to your wallet.")
            self.logger().info(f"Margin Required: {total_maint_margin_required}. "
                               f"Negative PnL assets: {negative_pnls_msg}.")

    async def _format_trading_rules(self, exchange_info_dict: Dict[str, Any]) -> List[TradingRule]:
        """
        Queries the necessary API endpoint and initialize the TradingRule object for each trading pair being traded.

        Parameters
        ----------
        exchange_info_dict:
            Trading rules dictionary response from the exchange
        """
        rules: list = exchange_info_dict.get("symbols", [])
        return_val: list = []
        for rule in rules:
            try:
                if web_utils.is_exchange_information_valid(rule):
                    trading_pair = await self.trading_pair_associated_to_exchange_symbol(symbol=rule["symbol"])
                    filters = rule["filters"]
                    filt_dict = {fil["filterType"]: fil for fil in filters}

                    min_order_size = Decimal(filt_dict.get("LOT_SIZE").get("minQty"))
                    step_size = Decimal(filt_dict.get("LOT_SIZE").get("stepSize"))
                    tick_size = Decimal(filt_dict.get("PRICE_FILTER").get("tickSize"))
                    min_notional = Decimal(filt_dict.get("MIN_NOTIONAL").get("notional"))
                    collateral_token = rule["marginAsset"]

                    return_val.append(
                        TradingRule(
                            trading_pair,
                            min_order_size=min_order_size,
                            min_price_increment=Decimal(tick_size),
                            min_base_amount_increment=Decimal(step_size),
                            min_notional_size=Decimal(min_notional),
                            buy_order_collateral_token=collateral_token,
                            sell_order_collateral_token=collateral_token,
                        )
                    )
            except Exception as e:
                self.logger().error(
                    f"Error parsing the trading pair rule {rule}. Error: {e}. Skipping...", exc_info=True
                )
        return return_val

    def _initialize_trading_pair_symbols_from_exchange_info(self, exchange_info: Dict[str, Any]):
        mapping = bidict()
        for symbol_data in filter(web_utils.is_exchange_information_valid, exchange_info.get("symbols", [])):
            exchange_symbol = symbol_data["pair"]
            base = symbol_data["baseAsset"]
            quote = symbol_data["quoteAsset"]
            trading_pair = combine_to_hb_trading_pair(base, quote)
            if trading_pair in mapping.inverse:
                self._resolve_trading_pair_symbols_duplicate(mapping, exchange_symbol, base, quote)
            else:
                mapping[exchange_symbol] = trading_pair
        self._set_trading_pair_symbol_map(mapping)

    async def _get_last_traded_price(self, trading_pair: str) -> float:
        exchange_symbol = await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)
        params = {"symbol": exchange_symbol}
        response = await self._api_get(
            path_url=CONSTANTS.TICKER_PRICE_CHANGE_URL,
            params=params)
        price = float(response["lastPrice"])
        return price

    def _resolve_trading_pair_symbols_duplicate(self, mapping: bidict, new_exchange_symbol: str, base: str, quote: str):
        """Resolves name conflicts provoked by futures contracts.

        If the expected BASEQUOTE combination matches one of the exchange symbols, it is the one taken, otherwise,
        the trading pair is removed from the map and an error is logged.
        """
        expected_exchange_symbol = f"{base}{quote}"
        trading_pair = combine_to_hb_trading_pair(base, quote)
        current_exchange_symbol = mapping.inverse[trading_pair]
        if current_exchange_symbol == expected_exchange_symbol:
            pass
        elif new_exchange_symbol == expected_exchange_symbol:
            mapping.pop(current_exchange_symbol)
            mapping[new_exchange_symbol] = trading_pair
        else:
            self.logger().error(
                f"Could not resolve the exchange symbols {new_exchange_symbol} and {current_exchange_symbol}")
            mapping.pop(current_exchange_symbol)

    async def _update_balances(self):
        """
        Calls the REST API to update total and available balances.
        """
        local_asset_names = set(self._account_balances.keys())
        remote_asset_names = set()

        account_info = await self._api_get(path_url=CONSTANTS.ACCOUNT_INFO_URL,
                                           is_auth_required=True)
        assets = account_info.get("assets")
        for asset in assets:
            asset_name = asset.get("asset")
            available_balance = Decimal(asset.get("availableBalance"))
            wallet_balance = Decimal(asset.get("walletBalance"))
            self._account_available_balances[asset_name] = available_balance
            self._account_balances[asset_name] = wallet_balance
            remote_asset_names.add(asset_name)

        asset_names_to_remove = local_asset_names.difference(remote_asset_names)
        for asset_name in asset_names_to_remove:
            del self._account_available_balances[asset_name]
            del self._account_balances[asset_name]

    async def _update_positions(self):
        positions = await self._api_get(path_url=CONSTANTS.POSITION_INFORMATION_URL,
                                        is_auth_required=True)
        for position in positions:
            trading_pair = position.get("symbol")
            try:
                hb_trading_pair = await self.trading_pair_associated_to_exchange_symbol(trading_pair)
            except KeyError:
                # Ignore results for which their symbols is not tracked by the connector
                continue
            position_side = PositionSide[position.get("positionSide")]
            unrealized_pnl = Decimal(position.get("unRealizedProfit"))
            entry_price = Decimal(position.get("entryPrice"))
            amount = Decimal(position.get("positionAmt"))
            leverage = Decimal(position.get("leverage"))
            pos_key = self._perpetual_trading.position_key(hb_trading_pair, position_side)
            if amount != 0:
                _position = Position(
                    trading_pair=await self.trading_pair_associated_to_exchange_symbol(trading_pair),
                    position_side=position_side,
                    unrealized_pnl=unrealized_pnl,
                    entry_price=entry_price,
                    amount=amount,
                    leverage=leverage
                )
                self._perpetual_trading.set_position(pos_key, _position)
            else:
                self._perpetual_trading.remove_position(pos_key)

    async def _update_order_fills_from_trades(self):
        last_tick = int(self._last_poll_timestamp / self.UPDATE_ORDER_STATUS_MIN_INTERVAL)
        current_tick = int(self.current_timestamp / self.UPDATE_ORDER_STATUS_MIN_INTERVAL)
        if current_tick > last_tick and len(self._order_tracker.active_orders) > 0:
            trading_pairs_to_order_map: Dict[str, Dict[str, Any]] = defaultdict(lambda: {})
            for order in self._order_tracker.active_orders.values():
                trading_pairs_to_order_map[order.trading_pair][order.exchange_order_id] = order
            trading_pairs = list(trading_pairs_to_order_map.keys())
            tasks = [
                self._api_get(
                    path_url=CONSTANTS.ACCOUNT_TRADE_LIST_URL,
                    params={"symbol": await self.exchange_symbol_associated_to_pair(trading_pair=trading_pair)},
                    is_auth_required=True,
                )
                for trading_pair in trading_pairs
            ]
            self.logger().debug(f"Polling for order fills of {len(tasks)} trading_pairs.")
            results = await safe_gather(*tasks, return_exceptions=True)
            for trades, trading_pair in zip(results, trading_pairs):
                order_map = trading_pairs_to_order_map.get(trading_pair)
                if isinstance(trades, Exception):
                    self.logger().network(
                        f"Error fetching trades update for the order {trading_pair}: {trades}.",
                        app_warning_msg=f"Failed to fetch trade update for {trading_pair}."
                    )
                    continue
                for trade in trades:
                    order_id = str(trade.get("orderId"))
                    if order_id in order_map:
                        tracked_order: InFlightOrder = order_map.get(order_id)
                        position_side = trade["positionSide"]
                        position_action = (PositionAction.OPEN
                                           if (tracked_order.trade_type is TradeType.BUY and position_side == "LONG"
                                               or tracked_order.trade_type is TradeType.SELL and position_side == "SHORT")
                                           else PositionAction.CLOSE)
                        fee = TradeFeeBase.new_perpetual_fee(
                            fee_schema=self.trade_fee_schema(),
                            position_action=position_action,
                            percent_token=trade["commissionAsset"],
                            flat_fees=[TokenAmount(amount=Decimal(trade["commission"]), token=trade["commissionAsset"])]
                        )
                        trade_update: TradeUpdate = TradeUpdate(
                            trade_id=str(trade["id"]),
                            client_order_id=tracked_order.client_order_id,
                            exchange_order_id=trade["orderId"],
                            trading_pair=tracked_order.trading_pair,
                            fill_timestamp=trade["time"] * 1e-3,
                            fill_price=Decimal(trade["price"]),
                            fill_base_amount=Decimal(trade["qty"]),
                            fill_quote_amount=Decimal(trade["quoteQty"]),
                            fee=fee,
                        )
                        self._order_tracker.process_trade_update(trade_update)

    async def _update_order_status(self):
        """
        Calls the REST API to get order/trade updates for each in-flight order.
        """
        last_tick = int(self._last_poll_timestamp / self.UPDATE_ORDER_STATUS_MIN_INTERVAL)
        current_tick = int(self.current_timestamp / self.UPDATE_ORDER_STATUS_MIN_INTERVAL)
        if current_tick > last_tick and len(self._order_tracker.active_orders) > 0:
            tracked_orders = list(self._order_tracker.active_orders.values())
            exchange_symbols = [
                await self.exchange_symbol_associated_to_pair(trading_pair=order.trading_pair)
                for order in tracked_orders
            ]
            tasks = [
                self._api_get(
                    path_url=CONSTANTS.ORDER_URL,
                    params={
                        "symbol": exchange_symbol,
                        "origClientOrderId": order.client_order_id
                    },
                    is_auth_required=True,
                    return_err=True,
                    limit_id=CONSTANTS.GET_ORDER_LIMIT_ID,
                )
                for order, exchange_symbol in zip(tracked_orders, exchange_symbols)
            ]
            self.logger().debug(f"Polling for order status updates of {len(tasks)} orders.")
            results = await safe_gather(*tasks, return_exceptions=True)

            for order_update, tracked_order, exchange_symbol in zip(
                    results,
                    tracked_orders,
                    exchange_symbols,
            ):
                client_order_id = tracked_order.client_order_id
                if client_order_id not in self._order_tracker.all_orders:
                    continue
                if isinstance(order_update, asyncio.CancelledError):
                    raise order_update
                if isinstance(order_update, BaseException):
                    if not isinstance(order_update, Exception):
                        raise order_update
                    self.logger().network(
                        f"Error fetching status update for order {client_order_id}."
                    )
                    continue
                if not isinstance(order_update, Mapping):
                    self.logger().network(
                        f"Malformed status update for order {client_order_id}."
                    )
                    continue
                if "code" in order_update:
                    is_not_found = (
                        order_update.get("code") == CONSTANTS.ORDER_NOT_EXIST_ERROR_CODE
                        or CONSTANTS.ORDER_NOT_EXIST_MESSAGE in str(order_update.get("msg", ""))
                    )
                    if is_not_found and self.is_order_submission_unknown(client_order_id):
                        self.logger().debug(
                            f"Order {client_order_id} remains submission-unknown after status NOT_FOUND."
                        )
                    elif is_not_found:
                        await self._order_tracker.process_order_not_found(client_order_id)
                    else:
                        self.logger().network(
                            f"Exchange rejected the status request for order {client_order_id}."
                        )
                    continue

                if self.is_order_submission_unknown(client_order_id):
                    try:
                        snapshot = BinancePerpetualOrderSnapshot.from_payload(
                            payload=order_update,
                            expected_symbol=exchange_symbol,
                            trading_pair=tracked_order.trading_pair,
                            expected_client_order_id=client_order_id,
                            data_time=self.current_timestamp,
                        )
                        await self._apply_authoritative_order_snapshot(snapshot)
                    except asyncio.CancelledError:
                        raise
                    except BinancePerpetualOrderDataError:
                        self.logger().network(
                            f"Malformed authoritative status update for order {client_order_id}."
                        )
                    continue

                try:
                    new_order_update = OrderUpdate(
                        trading_pair=await self.trading_pair_associated_to_exchange_symbol(order_update["symbol"]),
                        update_timestamp=order_update["updateTime"] * 1e-3,
                        new_state=CONSTANTS.ORDER_STATE[order_update["status"]],
                        client_order_id=order_update["clientOrderId"],
                        exchange_order_id=order_update["orderId"],
                    )
                except asyncio.CancelledError:
                    raise
                except (KeyError, TypeError, ValueError):
                    self.logger().network(
                        f"Malformed status update for order {client_order_id}."
                    )
                    continue
                self._order_tracker.process_order_update(new_order_update)

    async def _fetch_account_position_mode(self) -> Optional[PositionMode]:
        mode_snapshot = await self.get_position_mode_snapshot()
        self._position_mode = PositionMode.HEDGE if mode_snapshot.dual_side_position else PositionMode.ONEWAY
        return self._position_mode

    async def _get_position_mode(self) -> Optional[PositionMode]:
        # To-do: ensure there's no active order or contract before changing position mode
        if self._position_mode is None:
            await self._fetch_account_position_mode()
        return self._position_mode

    async def _trading_pair_position_mode_set(self, mode: PositionMode, trading_pair: str) -> Tuple[bool, str]:
        msg = ""
        success = True
        initial_mode = await self._get_position_mode()
        if initial_mode != mode:
            params = {
                "dualSidePosition": True if mode == PositionMode.HEDGE else False,
            }
            response = await self._api_post(
                path_url=CONSTANTS.CHANGE_POSITION_MODE_URL,
                data=params,
                is_auth_required=True,
                limit_id=CONSTANTS.POST_POSITION_MODE_LIMIT_ID,
                return_err=True
            )
            if not (response["msg"] == "success" and response["code"] == 200):
                success = False
                return success, str(response)
            self._position_mode = mode
        return success, msg

    async def _set_trading_pair_leverage(self, trading_pair: str, leverage: int) -> Tuple[bool, str]:
        try:
            result = await self.set_leverage_with_result(trading_pair, leverage)
        except BinancePerpetualRiskDataError:
            return False, "Unable to set leverage"
        if result.leverage == leverage:
            return True, ""
        return False, "Unable to set leverage"

    async def _fetch_last_fee_payment(self, trading_pair: str) -> Tuple[int, Decimal, Decimal]:
        exchange_symbol = await self.exchange_symbol_associated_to_pair(trading_pair)
        payment_response = await self._api_get(
            path_url=CONSTANTS.GET_INCOME_HISTORY_URL,
            params={
                "symbol": exchange_symbol,
                "incomeType": "FUNDING_FEE",
            },
            is_auth_required=True,
        )
        funding_info_response = await self._api_get(
            path_url=CONSTANTS.MARK_PRICE_URL,
            params={
                "symbol": exchange_symbol,
            },
        )
        sorted_payment_response = sorted(payment_response, key=lambda a: a.get('time', 0), reverse=True)
        if len(sorted_payment_response) < 1:
            timestamp, funding_rate, payment = 0, Decimal("-1"), Decimal("-1")
            return timestamp, funding_rate, payment
        funding_payment = sorted_payment_response[0]
        _payment = Decimal(funding_payment["income"])
        funding_rate = Decimal(funding_info_response["lastFundingRate"])
        timestamp = funding_payment["time"]
        if _payment != Decimal("0"):
            payment = _payment
        else:
            timestamp, funding_rate, payment = 0, Decimal("-1"), Decimal("-1")
        return timestamp, funding_rate, payment
