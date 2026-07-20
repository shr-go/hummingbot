from dataclasses import FrozenInstanceError, replace
from decimal import Decimal
from typing import Any, Dict, Tuple
from unittest.mock import AsyncMock

from bidict import bidict

import hummingbot.connector.derivative.binance_perpetual.binance_perpetual_constants as CONSTANTS
from hummingbot.connector.derivative.binance_perpetual.binance_perpetual_derivative import BinancePerpetualDerivative
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
    BinancePerpetualRiskDataError,
    BinancePerpetualSymbolConfig,
)
from hummingbot.connector.trading_rule import TradingRule
from test.isolated_asyncio_wrapper_test_case import IsolatedAsyncioWrapperTestCase


class BinancePerpetualRiskDataTest(IsolatedAsyncioWrapperTestCase):
    data_time = 1_640_780_000.0
    symbol = "SNDKUSDT"
    trading_pair = "SNDK-USDT"
    domain = CONSTANTS.TESTNET_DOMAIN

    def setUp(self) -> None:
        super().setUp()
        self.exchange = self._new_exchange()

    def _new_exchange(self) -> BinancePerpetualDerivative:
        exchange = BinancePerpetualDerivative(
            binance_perpetual_api_key="testAPIKey",
            binance_perpetual_api_secret="testSecret",
            trading_pairs=[self.trading_pair],
            domain=self.domain,
        )
        exchange._set_current_timestamp(self.data_time)
        exchange._set_trading_pair_symbol_map(bidict({self.symbol: self.trading_pair}))
        exchange._trading_rules = {
            self.trading_pair: TradingRule(
                trading_pair=self.trading_pair,
                min_order_size=Decimal("1"),
                min_price_increment=Decimal("0.01"),
                min_base_amount_increment=Decimal("1"),
                min_notional_size=Decimal("5"),
            )
        }
        return exchange

    def _exchange_info(self, **updates: Any) -> Dict[str, Any]:
        symbol = {
            "symbol": self.symbol,
            "pair": self.symbol,
            "contractType": "TRADIFI_PERPETUAL",
            "contractSize": "1E-2",
            "baseAsset": "SNDK",
            "quoteAsset": "USDT",
            "marginAsset": "USDT",
            "status": "TRADING",
            "filters": [
                {"filterType": "PRICE_FILTER", "minPrice": "0.01", "maxPrice": "10000", "tickSize": "0.01"},
                {"filterType": "LOT_SIZE", "minQty": "1", "maxQty": "100000", "stepSize": "1"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ],
        }
        symbol.update(updates)
        return {"serverTime": int(self.data_time * 1e3), "symbols": [symbol], "extra": "ignored"}

    def _account_v3(self, **updates: Any) -> Dict[str, Any]:
        payload = {
            "totalInitialMargin": "1.3E+1",
            "totalMaintMargin": "1.2",
            "totalWalletBalance": "100",
            "totalUnrealizedProfit": "2",
            "totalMarginBalance": "102",
            "totalPositionInitialMargin": "10",
            "totalOpenOrderInitialMargin": "3",
            "totalCrossWalletBalance": "100",
            "totalCrossUnPnl": "2",
            "availableBalance": "89",
            "maxWithdrawAmount": "89",
            "assets": [
                {
                    "asset": "USDT",
                    "walletBalance": "100",
                    "unrealizedProfit": "2",
                    "marginBalance": "102",
                    "maintMargin": "1.2",
                    "initialMargin": "13",
                    "positionInitialMargin": "10",
                    "openOrderInitialMargin": "3",
                    "crossWalletBalance": "100",
                    "crossUnPnl": "2",
                    "availableBalance": "89",
                    "maxWithdrawAmount": "89",
                    "updateTime": int(self.data_time * 1e3),
                    "extra": "ignored",
                }
            ],
            "positions": [
                {
                    "symbol": self.symbol,
                    "positionSide": "BOTH",
                    "positionAmt": "2",
                    "unrealizedProfit": "2",
                    "isolatedMargin": "0",
                    "notional": "50",
                    "isolatedWallet": "0",
                    "initialMargin": "13",
                    "maintMargin": "1.2",
                    "updateTime": int(self.data_time * 1e3),
                }
            ],
            "extra": {"newBinanceField": True},
        }
        payload.update(updates)
        return payload

    def _position_v3(self, symbol: str = None, **updates: Any) -> Dict[str, Any]:
        payload = {
            "symbol": symbol or self.symbol,
            "positionSide": "BOTH",
            "positionAmt": "2",
            "entryPrice": "2.5E+1",
            "breakEvenPrice": "25.01",
            "markPrice": "25",
            "unRealizedProfit": "2",
            "liquidationPrice": "0",
            "isolatedMargin": "0",
            "notional": "50",
            "marginAsset": "USDT",
            "isolatedWallet": "0",
            "initialMargin": "13",
            "maintMargin": "1.2",
            "positionInitialMargin": "10",
            "openOrderInitialMargin": "3",
            "adl": 2,
            "bidNotional": "0",
            "askNotional": "0",
            "updateTime": int(self.data_time * 1e3),
            "extra": "ignored",
        }
        payload.update(updates)
        return payload

    def _account_config(self, **updates: Any) -> Dict[str, Any]:
        payload = {
            "canTrade": True,
            "canDeposit": True,
            "canWithdraw": True,
            "dualSidePosition": False,
            "multiAssetsMargin": True,
            "updateTime": 0,
        }
        payload.update(updates)
        return payload

    def _symbol_config(self, symbol: str = None, **updates: Any) -> Dict[str, Any]:
        payload = {
            "symbol": symbol or self.symbol,
            "marginType": "CROSSED",
            "isAutoAddMargin": False,
            "leverage": 20,
            "maxNotionalValue": "1500",
        }
        payload.update(updates)
        return payload

    def _brackets(self, symbol: str = None, **updates: Any) -> Dict[str, Any]:
        payload = {
            "symbol": symbol or self.symbol,
            "notionalCoef": 1.5,
            "brackets": [
                {
                    "bracket": 1,
                    "initialLeverage": 20,
                    "notionalCap": "1000",
                    "notionalFloor": "0",
                    "maintMarginRatio": "0.01",
                    "cum": "0",
                },
                {
                    "bracket": 2,
                    "initialLeverage": 10,
                    "notionalCap": "5000",
                    "notionalFloor": "1000",
                    "maintMarginRatio": "0.02",
                    "cum": "10",
                },
            ],
        }
        payload.update(updates)
        return payload

    def _typed_bundle(self) -> Tuple[Any, ...]:
        instrument = BinancePerpetualInstrumentInfo.from_exchange_info(
            self._exchange_info(), self.trading_pair, self.data_time
        )
        account = BinancePerpetualAccountRiskSnapshot.from_payload(self._account_v3(), self.data_time)
        position = BinancePerpetualPositionRiskSnapshot.from_payload(self._position_v3(), self.data_time)
        account_config = BinancePerpetualAccountConfig.from_payload(self._account_config(), self.data_time)
        symbol_config = BinancePerpetualSymbolConfig.from_payload(self._symbol_config(), self.data_time)
        multi_assets = BinancePerpetualMultiAssetsMode.from_payload(
            {"multiAssetsMargin": True}, self.data_time
        )
        position_mode = BinancePerpetualPositionMode.from_payload(
            {"dualSidePosition": False}, self.data_time
        )
        brackets = BinancePerpetualLeverageBrackets.from_payload(
            self._brackets(), self.symbol, self.data_time, self.data_time
        )
        return instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets

    def _inactive_bundle(self) -> Tuple[Any, ...]:
        instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets = (
            self._typed_bundle()
        )
        zero = Decimal("0")
        account_position = replace(
            account.positions[0],
            position_amount=zero,
            unrealized_profit=zero,
            isolated_margin=zero,
            notional=zero,
            isolated_wallet=zero,
            initial_margin=zero,
            maint_margin=zero,
            update_time_ms=0,
        )
        asset = replace(
            account.assets[0],
            unrealized_profit=zero,
            margin_balance=Decimal("100"),
            maint_margin=zero,
            initial_margin=zero,
            position_initial_margin=zero,
            open_order_initial_margin=zero,
            cross_unrealized_profit=zero,
            available_balance=Decimal("100"),
            max_withdraw_amount=Decimal("100"),
        )
        account = replace(
            account,
            total_initial_margin=zero,
            total_maint_margin=zero,
            total_unrealized_profit=zero,
            total_margin_balance=Decimal("100"),
            total_position_initial_margin=zero,
            total_open_order_initial_margin=zero,
            total_cross_unrealized_profit=zero,
            available_balance=Decimal("100"),
            max_withdraw_amount=Decimal("100"),
            assets=(asset,),
            positions=(account_position,),
        )
        position = replace(
            position,
            position_amount=zero,
            unrealized_profit=zero,
            isolated_margin=zero,
            notional=zero,
            isolated_wallet=zero,
            initial_margin=zero,
            maint_margin=zero,
            position_initial_margin=zero,
            open_order_initial_margin=zero,
            bid_notional=zero,
            ask_notional=zero,
            update_time_ms=0,
        )
        return instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets

    @staticmethod
    def _bundle_with_notional(bundle: Tuple[Any, ...], notional: Decimal) -> Tuple[Any, ...]:
        instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets = bundle
        account = replace(
            account,
            positions=(replace(account.positions[0], notional=notional),),
        )
        position = replace(position, notional=notional)
        return instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets

    def _configure_preflight_sources(
            self,
            exchange: BinancePerpetualDerivative,
            bundle: Tuple[Any, ...],
    ) -> None:
        instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets = bundle
        exchange.get_instrument_info = AsyncMock(return_value=instrument)
        exchange.get_account_risk_snapshot = AsyncMock(return_value=account)
        exchange.get_position_risk_snapshots = AsyncMock(return_value=(position,))
        exchange.get_account_config = AsyncMock(return_value=account_config)
        exchange.get_symbol_config = AsyncMock(return_value=symbol_config)
        exchange.get_multi_assets_mode = AsyncMock(return_value=multi_assets)
        exchange.get_position_mode_snapshot = AsyncMock(return_value=position_mode)
        exchange.get_leverage_brackets = AsyncMock(return_value=brackets)
        exchange._api_post = AsyncMock()

    def test_account_v3_is_immutable_and_preserves_decimal_and_source_times(self):
        snapshot = BinancePerpetualAccountRiskSnapshot.from_payload(self._account_v3(), self.data_time)

        self.assertEqual(Decimal("13"), snapshot.total_initial_margin)
        self.assertEqual(Decimal("102"), snapshot.total_margin_balance)
        self.assertEqual(self.data_time, snapshot.data_time)
        self.assertEqual(int(self.data_time * 1e3), snapshot.assets[0].update_time_ms)
        self.assertEqual(int(self.data_time * 1e3), snapshot.positions[0].update_time_ms)
        with self.assertRaises(FrozenInstanceError):
            snapshot.total_margin_balance = Decimal("0")

    def test_account_v3_rejects_missing_required_fields_but_ignores_extra_fields(self):
        for field in ("totalMarginBalance", "totalOpenOrderInitialMargin", "assets", "positions"):
            with self.subTest(field=field):
                payload = self._account_v3()
                payload.pop(field)
                with self.assertRaisesRegex(BinancePerpetualRiskDataError, field):
                    BinancePerpetualAccountRiskSnapshot.from_payload(payload, self.data_time)

        snapshot = BinancePerpetualAccountRiskSnapshot.from_payload(self._account_v3(), self.data_time)
        self.assertEqual(Decimal("100"), snapshot.assets[0].wallet_balance)

    def test_position_v3_preserves_scientific_notation_and_all_margin_fields(self):
        position = BinancePerpetualPositionRiskSnapshot.from_payload(self._position_v3(), self.data_time)

        self.assertEqual(Decimal("25"), position.entry_price)
        self.assertEqual(Decimal("10"), position.position_initial_margin)
        self.assertEqual(Decimal("3"), position.open_order_initial_margin)
        self.assertEqual(Decimal("1.2"), position.maint_margin)
        self.assertEqual(int(self.data_time * 1e3), position.update_time_ms)

    def test_instrument_requires_explicit_tradfi_multiplier_and_preserves_non_trading_status(self):
        instrument = BinancePerpetualInstrumentInfo.from_exchange_info(
            self._exchange_info(status="BREAK"), self.trading_pair, self.data_time
        )

        self.assertEqual("BREAK", instrument.status)
        self.assertEqual(Decimal("0.01"), instrument.contract_multiplier)
        self.assertEqual(Decimal("0.01"), instrument.tick_size)

        exchange_info = self._exchange_info()
        exchange_info["symbols"][0].pop("contractSize")
        with self.assertRaisesRegex(BinancePerpetualRiskDataError, "contract multiplier"):
            BinancePerpetualInstrumentInfo.from_exchange_info(exchange_info, self.trading_pair, self.data_time)

    def test_instrument_reconciles_all_supported_multiplier_representations(self):
        instrument = BinancePerpetualInstrumentInfo.from_exchange_info(
            self._exchange_info(contractMultiplier="0.01"), self.trading_pair, self.data_time
        )

        self.assertEqual(Decimal("0.01"), instrument.contract_multiplier)
        with self.assertRaisesRegex(BinancePerpetualRiskDataError, "contract multiplier"):
            BinancePerpetualInstrumentInfo.from_exchange_info(
                self._exchange_info(contractMultiplier="0.02"), self.trading_pair, self.data_time
            )

    def test_leverage_brackets_apply_notional_coef_and_exact_cap_enters_next_bracket(self):
        brackets = BinancePerpetualLeverageBrackets.from_payload(
            [self._brackets()], self.symbol, self.data_time, self.data_time
        )

        self.assertEqual(Decimal("1.5"), brackets.notional_coef)
        self.assertEqual(Decimal("1500.0"), brackets.brackets[0].adjusted_notional_cap)
        self.assertEqual(1, brackets.bracket_for_notional(Decimal("1499.999")).bracket)
        self.assertEqual(2, brackets.bracket_for_notional(Decimal("1500")).bracket)
        self.assertEqual(Decimal("15.0"), brackets.brackets[1].adjusted_cum)
        self.assertEqual(Decimal("15.00"), brackets.maintenance_margin(Decimal("1500")))

    def test_leverage_brackets_reject_missing_or_inconsistent_data(self):
        payload = self._brackets()
        payload["brackets"][0].pop("cum")
        with self.assertRaisesRegex(BinancePerpetualRiskDataError, "cum"):
            BinancePerpetualLeverageBrackets.from_payload(
                payload, self.symbol, self.data_time, self.data_time
            )

        payload = self._brackets()
        payload["brackets"][1]["notionalFloor"] = "1001"
        with self.assertRaisesRegex(BinancePerpetualRiskDataError, "contiguous"):
            BinancePerpetualLeverageBrackets.from_payload(
                payload, self.symbol, self.data_time, self.data_time
            )

    def test_leverage_change_result_preserves_max_notional_value(self):
        result = BinancePerpetualLeverageChangeResult.from_payload(
            {"symbol": self.symbol, "leverage": 20, "maxNotionalValue": "1E+6"}, self.data_time
        )

        self.assertEqual(20, result.leverage)
        self.assertEqual(Decimal("1E+6"), result.max_notional_value)
        with self.assertRaisesRegex(BinancePerpetualRiskDataError, "maxNotionalValue"):
            BinancePerpetualLeverageChangeResult.from_payload(
                {"symbol": self.symbol, "leverage": 20}, self.data_time
            )

    def test_mode_payloads_require_authoritative_boolean_fields(self):
        with self.assertRaisesRegex(BinancePerpetualRiskDataError, "dualSidePosition"):
            BinancePerpetualPositionMode.from_payload({}, self.data_time)
        with self.assertRaisesRegex(BinancePerpetualRiskDataError, "boolean"):
            BinancePerpetualPositionMode.from_payload({"dualSidePosition": "false"}, self.data_time)
        with self.assertRaisesRegex(BinancePerpetualRiskDataError, "multiAssetsMargin"):
            BinancePerpetualMultiAssetsMode.from_payload({}, self.data_time)

    def test_symbol_config_uses_documented_json_boolean(self):
        config = BinancePerpetualSymbolConfig.from_payload(self._symbol_config(), self.data_time)

        self.assertFalse(config.is_auto_add_margin)
        for invalid in ("false", 0, None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(BinancePerpetualRiskDataError, "boolean"):
                    BinancePerpetualSymbolConfig.from_payload(
                        self._symbol_config(isAutoAddMargin=invalid), self.data_time
                    )

    async def test_connector_fetches_all_typed_risk_sources_and_leverage_result(self):
        get_responses = {
            CONSTANTS.EXCHANGE_INFO_URL: self._exchange_info(),
            CONSTANTS.ACCOUNT_INFO_V3_URL: self._account_v3(),
            CONSTANTS.POSITION_INFORMATION_V3_URL: [self._position_v3()],
            CONSTANTS.ACCOUNT_CONFIG_URL: self._account_config(),
            CONSTANTS.SYMBOL_CONFIG_URL: [self._symbol_config()],
            CONSTANTS.MULTI_ASSETS_MODE_URL: {"multiAssetsMargin": True},
            CONSTANTS.CHANGE_POSITION_MODE_URL: {"dualSidePosition": False},
            CONSTANTS.LEVERAGE_BRACKET_URL: [self._brackets()],
        }

        async def get_response(path_url: str, **_: Any) -> Any:
            return get_responses[path_url]

        self.exchange._api_get = AsyncMock(side_effect=get_response)
        self.exchange._api_post = AsyncMock(return_value={
            "symbol": self.symbol,
            "leverage": 20,
            "maxNotionalValue": "100000",
        })

        instrument = await self.exchange.get_instrument_info(self.trading_pair)
        account = await self.exchange.get_account_risk_snapshot()
        positions = await self.exchange.get_position_risk_snapshots()
        account_config = await self.exchange.get_account_config()
        symbol_config = await self.exchange.get_symbol_config(self.trading_pair)
        multi_assets = await self.exchange.get_multi_assets_mode()
        position_mode = await self.exchange.get_position_mode_snapshot()
        brackets = await self.exchange.get_leverage_brackets(self.trading_pair, refresh=True)
        leverage = await self.exchange.set_leverage_with_result(self.trading_pair, 20)

        self.assertIsInstance(instrument, BinancePerpetualInstrumentInfo)
        self.assertIsInstance(account, BinancePerpetualAccountRiskSnapshot)
        self.assertIsInstance(positions[0], BinancePerpetualPositionRiskSnapshot)
        self.assertIsInstance(account_config, BinancePerpetualAccountConfig)
        self.assertIsInstance(symbol_config, BinancePerpetualSymbolConfig)
        self.assertIsInstance(multi_assets, BinancePerpetualMultiAssetsMode)
        self.assertIsInstance(position_mode, BinancePerpetualPositionMode)
        self.assertIsInstance(brackets, BinancePerpetualLeverageBrackets)
        self.assertEqual(Decimal("100000"), leverage.max_notional_value)

    async def test_position_mode_fetch_failure_is_not_masked_by_default_one_way(self):
        self.exchange._api_get = AsyncMock(side_effect=RuntimeError("position mode unavailable"))

        with self.assertRaisesRegex(RuntimeError, "position mode unavailable"):
            await self.exchange.get_position_mode_snapshot()
        self.assertIsNone(self.exchange._position_mode)

        self.exchange._api_get = AsyncMock(return_value={})
        with self.assertRaisesRegex(BinancePerpetualRiskDataError, "dualSidePosition"):
            await self.exchange.get_position_mode_snapshot()
        self.assertIsNone(self.exchange._position_mode)

    async def test_leverage_bracket_cache_is_immutable_until_explicit_refresh(self):
        self.exchange._api_get = AsyncMock(side_effect=[
            [self._brackets()],
            [self._brackets(notionalCoef="2")],
        ])

        first = await self.exchange.get_leverage_brackets(self.trading_pair)
        cached = await self.exchange.get_leverage_brackets(self.trading_pair)
        self.exchange._set_current_timestamp(self.data_time + 1)
        refreshed = await self.exchange.get_leverage_brackets(self.trading_pair, refresh=True)

        self.assertIs(first, cached)
        self.assertEqual(Decimal("1.5"), cached.notional_coef)
        self.assertEqual(Decimal("2"), refreshed.notional_coef)
        self.assertEqual(self.data_time + 1, refreshed.data_time)
        self.assertEqual(2, self.exchange._api_get.await_count)

    async def test_strict_preflight_succeeds_without_mutating_account_modes(self):
        bundle = self._typed_bundle()
        self._configure_preflight_sources(self.exchange, bundle)

        snapshot = await self.exchange.strict_account_preflight(
            trading_pairs=[self.trading_pair],
            related_trading_pairs=[self.trading_pair],
            known_position_trading_pairs=[self.trading_pair],
            max_age_seconds=5,
        )

        self.assertEqual(self.data_time, snapshot.data_time)
        self.assertEqual(Decimal("102"), snapshot.account.total_margin_balance)
        self.assertEqual(self.symbol, snapshot.instruments[0].symbol)
        self.assertIsNone(self.exchange._position_mode)
        self.exchange._api_post.assert_not_awaited()

    async def test_strict_preflight_rejects_unsafe_freshness_parameters(self):
        max_age_cases = (
            5.000001,
            float("nan"),
            float("inf"),
            -0.001,
            True,
            "5",
            Decimal("5.0000000000000000000000000000001"),
            Decimal("-1E-10000"),
        )
        for value in max_age_cases:
            with self.subTest(parameter="max_age_seconds", value=value):
                exchange = self._new_exchange()
                self._configure_preflight_sources(exchange, self._typed_bundle())
                with self.assertRaisesRegex(BinancePerpetualPreflightError, "max_age_seconds"):
                    await exchange.strict_account_preflight(
                        trading_pairs=[self.trading_pair],
                        known_position_trading_pairs=[self.trading_pair],
                        max_age_seconds=value,
                    )

        tolerance_cases = (
            Decimal("0.0100001"),
            Decimal("-0.0001"),
            Decimal("NaN"),
            Decimal("Infinity"),
            0.01,
            "0.01",
            True,
        )
        for value in tolerance_cases:
            with self.subTest(parameter="consistency_tolerance", value=value):
                exchange = self._new_exchange()
                self._configure_preflight_sources(exchange, self._typed_bundle())
                with self.assertRaisesRegex(BinancePerpetualPreflightError, "consistency_tolerance"):
                    await exchange.strict_account_preflight(
                        trading_pairs=[self.trading_pair],
                        known_position_trading_pairs=[self.trading_pair],
                        max_age_seconds=4.5,
                        consistency_tolerance=value,
                    )

        for value in (Decimal("0"), Decimal("5")):
            with self.subTest(parameter="max_age_seconds boundary", value=value):
                exchange = self._new_exchange()
                self._configure_preflight_sources(exchange, self._typed_bundle())
                await exchange.strict_account_preflight(
                    trading_pairs=[self.trading_pair],
                    known_position_trading_pairs=[self.trading_pair],
                    max_age_seconds=value,
                    consistency_tolerance=Decimal("0.01"),
                )

    async def test_strict_preflight_validates_meaningful_source_timestamps(self):
        base = self._typed_bundle()
        instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets = base
        stale_time_ms = int((self.data_time - 6) * 1e3)
        future_time_ms = int((self.data_time + 1) * 1e3)
        cases = (
            (
                "active asset source time stale",
                (instrument, replace(account, assets=(replace(account.assets[0], update_time_ms=stale_time_ms),)),
                 position, account_config, symbol_config, multi_assets, position_mode, brackets),
            ),
            (
                "active Account V3 position source time stale",
                (instrument,
                 replace(account, positions=(replace(account.positions[0], update_time_ms=stale_time_ms),)),
                 position, account_config, symbol_config, multi_assets, position_mode, brackets),
            ),
            (
                "active Position V3 source time stale",
                (instrument, account, replace(position, update_time_ms=stale_time_ms), account_config,
                 symbol_config, multi_assets, position_mode, brackets),
            ),
            (
                "source time future",
                (instrument, replace(account, assets=(replace(account.assets[0], update_time_ms=future_time_ms),)),
                 position, account_config, symbol_config, multi_assets, position_mode, brackets),
            ),
            (
                "local receive time future",
                (instrument, replace(account, data_time=self.data_time + 1), position, account_config,
                 symbol_config, multi_assets, position_mode, brackets),
            ),
        )

        for name, bundle in cases:
            with self.subTest(name=name):
                exchange = self._new_exchange()
                self._configure_preflight_sources(exchange, bundle)
                with self.assertRaisesRegex(BinancePerpetualPreflightError, "stale|future"):
                    await exchange.strict_account_preflight(
                        trading_pairs=[self.trading_pair],
                        known_position_trading_pairs=[self.trading_pair],
                        max_age_seconds=5,
                    )

    async def test_strict_preflight_rejects_zero_timestamp_for_active_rows(self):
        base = self._typed_bundle()
        instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets = base
        cases = (
            (
                "active asset",
                (instrument, replace(account, assets=(replace(account.assets[0], update_time_ms=0),)),
                 position, account_config, symbol_config, multi_assets, position_mode, brackets),
            ),
            (
                "active Account V3 position",
                (instrument, replace(account, positions=(replace(account.positions[0], update_time_ms=0),)),
                 position, account_config, symbol_config, multi_assets, position_mode, brackets),
            ),
            (
                "active Position V3 row",
                (instrument, account, replace(position, update_time_ms=0), account_config,
                 symbol_config, multi_assets, position_mode, brackets),
            ),
        )
        for name, bundle in cases:
            with self.subTest(name=name):
                exchange = self._new_exchange()
                self._configure_preflight_sources(exchange, bundle)
                with self.assertRaisesRegex(BinancePerpetualPreflightError, "timestamp is zero"):
                    await exchange.strict_account_preflight(
                        trading_pairs=[self.trading_pair],
                        known_position_trading_pairs=[self.trading_pair],
                        max_age_seconds=5,
                    )

    async def test_strict_preflight_allows_documented_zero_timestamp_for_inactive_rows(self):
        bundle = self._inactive_bundle()
        bundle = (replace(bundle[0], source_time_ms=None), *bundle[1:])
        exchange = self._new_exchange()
        self._configure_preflight_sources(exchange, bundle)

        snapshot = await exchange.strict_account_preflight(
            trading_pairs=[self.trading_pair],
            known_position_trading_pairs=[],
            max_age_seconds=5,
        )

        self.assertEqual(0, snapshot.account.positions[0].update_time_ms)
        self.assertEqual(0, snapshot.positions[0].update_time_ms)
        self.assertIsNone(snapshot.instruments[0].source_time_ms)

    async def test_strict_preflight_reconciles_leverage_cap_and_exact_boundaries(self):
        base = self._typed_bundle()
        instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets = base
        rejection_cases = (
            (
                "max notional mismatch",
                (instrument, account, position, account_config,
                 replace(symbol_config, max_notional_value=Decimal("1500.01")),
                 multi_assets, position_mode, brackets),
                "maxNotionalValue",
            ),
            (
                "configured leverage unsupported",
                (instrument, account, position, account_config,
                 replace(symbol_config, leverage=21, max_notional_value=Decimal("1500")),
                 multi_assets, position_mode, brackets),
                "leverage",
            ),
            (
                "exact cap enters next bracket",
                self._bundle_with_notional(base, Decimal("1500")),
                "leverage",
            ),
        )
        for name, bundle, expected in rejection_cases:
            with self.subTest(name=name):
                exchange = self._new_exchange()
                self._configure_preflight_sources(exchange, bundle)
                with self.assertRaisesRegex(BinancePerpetualPreflightError, expected):
                    await exchange.strict_account_preflight(
                        trading_pairs=[self.trading_pair],
                        known_position_trading_pairs=[self.trading_pair],
                        max_age_seconds=5,
                    )

        allowed_boundary = self._bundle_with_notional(base, Decimal("1500"))
        allowed_boundary = (
            allowed_boundary[0],
            allowed_boundary[1],
            allowed_boundary[2],
            allowed_boundary[3],
            replace(allowed_boundary[4], leverage=10, max_notional_value=Decimal("7500")),
            allowed_boundary[5],
            allowed_boundary[6],
            allowed_boundary[7],
        )
        exchange = self._new_exchange()
        self._configure_preflight_sources(exchange, allowed_boundary)
        await exchange.strict_account_preflight(
            trading_pairs=[self.trading_pair],
            known_position_trading_pairs=[self.trading_pair],
            max_age_seconds=5,
        )

        last_cap = self._bundle_with_notional(allowed_boundary, Decimal("7500"))
        exchange = self._new_exchange()
        self._configure_preflight_sources(exchange, last_cap)
        with self.assertRaisesRegex(BinancePerpetualPreflightError, "outside leverage brackets"):
            await exchange.strict_account_preflight(
                trading_pairs=[self.trading_pair],
                known_position_trading_pairs=[self.trading_pair],
                max_age_seconds=5,
            )

    async def test_strict_preflight_rejects_unknown_bid_or_ask_orders_with_zero_margin(self):
        base = self._inactive_bundle()
        for field in ("bid_notional", "ask_notional"):
            with self.subTest(field=field):
                (instrument, account, position, account_config, symbol_config,
                 multi_assets, position_mode, brackets) = base
                position = replace(
                    position,
                    **{field: Decimal("0.00000001"), "update_time_ms": int(self.data_time * 1e3)},
                )
                exchange = self._new_exchange()
                self._configure_preflight_sources(
                    exchange,
                    (instrument, account, position, account_config, symbol_config,
                     multi_assets, position_mode, brackets),
                )
                with self.assertRaisesRegex(BinancePerpetualPreflightError, "unknown position or open order"):
                    await exchange.strict_account_preflight(
                        trading_pairs=[self.trading_pair],
                        known_position_trading_pairs=[],
                        max_age_seconds=5,
                    )

    async def test_strict_preflight_rejects_every_unsafe_or_inconsistent_state(self):
        base = self._typed_bundle()
        instrument, account, position, account_config, symbol_config, multi_assets, position_mode, brackets = base
        cases = [
            ("trading disabled", replace(account_config, can_trade=False), account, position, instrument,
             symbol_config, multi_assets, position_mode, brackets, [self.trading_pair], "canTrade"),
            ("single asset", account_config, account, position, instrument, symbol_config,
             replace(multi_assets, enabled=False), position_mode, brackets, [self.trading_pair], "Multi-Assets"),
            ("mode disagreement", replace(account_config, multi_assets_margin=False), account, position, instrument,
             symbol_config, multi_assets, position_mode, brackets, [self.trading_pair], "disagrees"),
            ("hedge mode", account_config, account, position, instrument, symbol_config, multi_assets,
             replace(position_mode, dual_side_position=True), brackets, [self.trading_pair], "One-way"),
            ("isolated", account_config, account, position, instrument,
             replace(symbol_config, margin_type=BinancePerpetualMarginType.ISOLATED), multi_assets,
             position_mode, brackets, [self.trading_pair], "Cross margin"),
            ("stale", account_config, replace(account, data_time=self.data_time - 6), position, instrument,
             symbol_config, multi_assets, position_mode, brackets, [self.trading_pair], "stale"),
            ("stopped", account_config, account, position, replace(instrument, status="BREAK"), symbol_config,
             multi_assets, position_mode, brackets, [self.trading_pair], "TRADING"),
            ("wrong contract", account_config, account, position, replace(instrument, contract_type="PERPETUAL"),
             symbol_config, multi_assets, position_mode, brackets, [self.trading_pair], "TRADIFI_PERPETUAL"),
            ("wrong quote", account_config, account, position, replace(instrument, quote_asset="USDC"),
             symbol_config, multi_assets, position_mode, brackets, [self.trading_pair], "USDT"),
            ("hedged position", account_config, account, replace(position, position_side="LONG"), instrument,
             symbol_config, multi_assets, position_mode, brackets, [self.trading_pair], "BOTH"),
            ("unknown related position", account_config, account, position, instrument, symbol_config,
             multi_assets, position_mode, brackets, [], "unknown position"),
            ("inconsistent totals", account_config, replace(account, total_initial_margin=Decimal("14")), position,
             instrument, symbol_config, multi_assets, position_mode, brackets, [self.trading_pair], "reconcile"),
        ]

        for (name, cfg, acct, pos, inst, sym_cfg, multi, mode, lev, known, expected) in cases:
            with self.subTest(name=name):
                exchange = self._new_exchange()
                bundle = (inst, acct, pos, cfg, sym_cfg, multi, mode, lev)
                self._configure_preflight_sources(exchange, bundle)
                with self.assertRaisesRegex(BinancePerpetualPreflightError, expected):
                    await exchange.strict_account_preflight(
                        trading_pairs=[self.trading_pair],
                        related_trading_pairs=[self.trading_pair],
                        known_position_trading_pairs=known,
                        max_age_seconds=5,
                    )
                exchange._api_post.assert_not_awaited()

        exchange = self._new_exchange()
        exchange._trading_rules = {}
        self._configure_preflight_sources(exchange, base)
        with self.assertRaisesRegex(BinancePerpetualPreflightError, "trading rule"):
            await exchange.strict_account_preflight(
                trading_pairs=[self.trading_pair],
                related_trading_pairs=[self.trading_pair],
                known_position_trading_pairs=[self.trading_pair],
                max_age_seconds=5,
            )

    async def test_strict_preflight_identifies_each_authoritative_fetch_failure(self):
        fetches = {
            "get_account_risk_snapshot": "Account Information V3",
            "get_position_risk_snapshots": "Position Information V3",
            "get_account_config": "account configuration",
            "get_multi_assets_mode": "Multi-Assets mode",
            "get_position_mode_snapshot": "position mode",
            "get_instrument_info": "exchange metadata",
            "get_symbol_config": "symbol configuration",
            "get_leverage_brackets": "leverage brackets",
        }

        for method_name, source_name in fetches.items():
            with self.subTest(source=source_name):
                exchange = self._new_exchange()
                self._configure_preflight_sources(exchange, self._typed_bundle())
                setattr(exchange, method_name, AsyncMock(side_effect=RuntimeError("unavailable")))
                with self.assertRaisesRegex(BinancePerpetualPreflightError, source_name):
                    await exchange.strict_account_preflight(
                        trading_pairs=[self.trading_pair],
                        related_trading_pairs=[self.trading_pair],
                        known_position_trading_pairs=[self.trading_pair],
                        max_age_seconds=5,
                    )
                self.assertIsNone(exchange._position_mode)
                exchange._api_post.assert_not_awaited()
