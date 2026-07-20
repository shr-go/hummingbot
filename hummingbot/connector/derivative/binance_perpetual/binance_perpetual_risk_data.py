from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping, Optional, Sequence, Tuple


class BinancePerpetualRiskDataError(ValueError):
    """Raised when Binance returns incomplete or internally invalid risk data."""


class BinancePerpetualPreflightError(BinancePerpetualRiskDataError):
    """Raised when authoritative account preflight cannot prove a safe state."""


_MAX_NOTIONAL_EXACT_DECIMAL_DIGITS = 128


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BinancePerpetualRiskDataError(f"{context} must be an object")
    return value


def _sequence(value: Any, context: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BinancePerpetualRiskDataError(f"{context} must be an array")
    return value


def _required(payload: Mapping[str, Any], field: str, context: str) -> Any:
    if field not in payload:
        raise BinancePerpetualRiskDataError(f"{context} is missing required field {field}")
    return payload[field]


def _string(payload: Mapping[str, Any], field: str, context: str) -> str:
    value = _required(payload, field, context)
    if not isinstance(value, str) or value == "":
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be a non-empty string")
    return value


def _decimal(payload: Mapping[str, Any], field: str, context: str) -> Decimal:
    value = _required(payload, field, context)
    if isinstance(value, bool):
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be a finite decimal") from exc
    if not parsed.is_finite():
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be a finite decimal")
    return parsed


def _integer(payload: Mapping[str, Any], field: str, context: str) -> int:
    value = _required(payload, field, context)
    if isinstance(value, bool):
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be an integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be an integer") from exc
    if isinstance(value, float) and not value.is_integer():
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be an integer")
    if isinstance(value, str) and str(parsed) != value.strip():
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be an integer")
    return parsed


def _boolean(payload: Mapping[str, Any], field: str, context: str) -> bool:
    value = _required(payload, field, context)
    if type(value) is not bool:
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be a boolean")
    return value


def _timestamp_ms(payload: Mapping[str, Any], field: str, context: str) -> int:
    timestamp = _integer(payload, field, context)
    if timestamp < 0:
        raise BinancePerpetualRiskDataError(f"{context}.{field} must be non-negative")
    return timestamp


def _data_time(value: Any) -> float:
    if isinstance(value, bool):
        raise BinancePerpetualRiskDataError("data_time must be a finite non-negative timestamp")
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise BinancePerpetualRiskDataError("data_time must be a finite non-negative timestamp") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise BinancePerpetualRiskDataError("data_time must be a finite non-negative timestamp")
    return parsed


def _validate_decimal_value(
        value: Any,
        field: str,
        *,
        non_negative: bool = False,
        positive: bool = False,
) -> None:
    if not isinstance(value, Decimal) or not value.is_finite():
        raise BinancePerpetualRiskDataError(f"{field} must be a finite Decimal")
    if positive and value <= 0:
        raise BinancePerpetualRiskDataError(f"{field} must be positive")
    if non_negative and value < 0:
        raise BinancePerpetualRiskDataError(f"{field} must be non-negative")


def _validate_maintenance_margin_relationship(
        maintenance_margin: Decimal,
        initial_margin: Decimal,
        context: str,
) -> None:
    if maintenance_margin > initial_margin:
        raise BinancePerpetualRiskDataError(
            f"{context}.maintMargin must not exceed initialMargin"
        )


def _exact_decimal_components(value: Decimal, field: str) -> Tuple[int, int]:
    _validate_decimal_value(value, field)
    sign, digits, exponent = value.as_tuple()
    if (
            not isinstance(exponent, int)
            or len(digits) > _MAX_NOTIONAL_EXACT_DECIMAL_DIGITS
            or abs(exponent) > _MAX_NOTIONAL_EXACT_DECIMAL_DIGITS
    ):
        raise BinancePerpetualRiskDataError(f"{field} exceeds supported exact decimal precision")
    coefficient = 0
    for digit in digits:
        coefficient = coefficient * 10 + digit
    if coefficient == 0:
        return 0, 0
    if sign:
        coefficient = -coefficient
    while coefficient % 10 == 0:
        coefficient //= 10
        exponent += 1
    return coefficient, exponent


def _exact_decimal_product_components(
        values: Sequence[Tuple[Decimal, str]],
) -> Tuple[int, int]:
    components = tuple(
        _exact_decimal_components(value, field)
        for value, field in values
    )
    if any(coefficient == 0 for coefficient, _ in components):
        return 0, 0
    coefficient = 1
    exponent = 0
    for value_coefficient, value_exponent in components:
        coefficient *= value_coefficient
        exponent += value_exponent
    return coefficient, exponent


def _decimal_distance_within_tolerance(
        left: Decimal,
        right_components: Tuple[int, int],
        tolerance: Decimal,
        context: str,
) -> bool:
    left_coefficient, left_exponent = _exact_decimal_components(left, f"{context}.notional")
    tolerance_coefficient, tolerance_exponent = _exact_decimal_components(
        tolerance,
        f"{context}.notional tolerance",
    )
    right_coefficient, right_exponent = right_components
    common_exponent = min(left_exponent, right_exponent, tolerance_exponent)
    left_scaled = left_coefficient * 10 ** (left_exponent - common_exponent)
    right_scaled = right_coefficient * 10 ** (right_exponent - common_exponent)
    tolerance_scaled = tolerance_coefficient * 10 ** (tolerance_exponent - common_exponent)
    return abs(left_scaled - right_scaled) <= tolerance_scaled


def _validate_position_identity_and_notional(
        position_side: Any,
        position_amount: Any,
        notional: Any,
        context: str,
) -> None:
    if not isinstance(position_side, str) or position_side not in {"BOTH", "LONG", "SHORT"}:
        raise BinancePerpetualRiskDataError(f"{context}.positionSide is unsupported")
    _validate_decimal_value(position_amount, f"{context}.positionAmt")
    _validate_decimal_value(notional, f"{context}.notional")
    if (position_amount == 0) != (notional == 0):
        raise BinancePerpetualRiskDataError(
            f"{context} position amount and signed notional zero state is contradictory"
        )
    if position_amount != 0 and position_amount.is_signed() != notional.is_signed():
        raise BinancePerpetualRiskDataError(
            f"{context} signed notional direction contradicts position amount"
        )
    if position_side == "LONG" and position_amount < 0:
        raise BinancePerpetualRiskDataError(f"{context} LONG position amount must be non-negative")
    if position_side == "SHORT" and position_amount > 0:
        raise BinancePerpetualRiskDataError(f"{context} SHORT position amount must be non-positive")


def _validate_non_negative_integer_value(value: Any, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BinancePerpetualRiskDataError(f"{field} must be a non-negative integer")


@dataclass(frozen=True)
class BinancePerpetualInstrumentInfo:
    symbol: str
    trading_pair: str
    contract_type: str
    status: str
    base_asset: str
    quote_asset: str
    margin_asset: str
    contract_multiplier: Decimal
    min_order_size: Decimal
    step_size: Decimal
    tick_size: Decimal
    min_notional: Decimal
    source_time_ms: Optional[int]
    data_time: float

    @classmethod
    def from_exchange_info(
            cls,
            payload: Mapping[str, Any],
            trading_pair: str,
            data_time: float,
    ) -> "BinancePerpetualInstrumentInfo":
        context = "exchangeInfo"
        payload = _mapping(payload, context)
        symbols = _sequence(_required(payload, "symbols", context), f"{context}.symbols")
        matches = []
        for index, raw_symbol in enumerate(symbols):
            symbol = _mapping(raw_symbol, f"{context}.symbols[{index}]")
            base_asset = _string(symbol, "baseAsset", f"{context}.symbols[{index}]")
            quote_asset = _string(symbol, "quoteAsset", f"{context}.symbols[{index}]")
            if f"{base_asset}-{quote_asset}" == trading_pair:
                matches.append(symbol)
        if len(matches) != 1:
            raise BinancePerpetualRiskDataError(
                f"exchangeInfo must contain exactly one record for trading pair {trading_pair}"
            )

        symbol = matches[0]
        symbol_context = f"exchangeInfo[{trading_pair}]"
        contract_type = _string(symbol, "contractType", symbol_context)
        multiplier_values = {
            field: _decimal(symbol, field, symbol_context)
            for field in ("contractSize", "contractMultiplier")
            if field in symbol
        }
        if not multiplier_values:
            if contract_type == "PERPETUAL":
                contract_multiplier = Decimal("1")
            else:
                raise BinancePerpetualRiskDataError(
                    f"{symbol_context} is missing an explicit contract multiplier"
                )
        else:
            if len(set(multiplier_values.values())) != 1:
                raise BinancePerpetualRiskDataError(
                    f"{symbol_context} has conflicting contract multiplier fields"
                )
            contract_multiplier = next(iter(multiplier_values.values()))
        if contract_multiplier <= 0:
            raise BinancePerpetualRiskDataError(f"{symbol_context} contract multiplier must be positive")

        filters_raw = _sequence(_required(symbol, "filters", symbol_context), f"{symbol_context}.filters")
        filters = {}
        for index, raw_filter in enumerate(filters_raw):
            filter_payload = _mapping(raw_filter, f"{symbol_context}.filters[{index}]")
            filter_type = _string(filter_payload, "filterType", f"{symbol_context}.filters[{index}]")
            if filter_type in filters:
                raise BinancePerpetualRiskDataError(f"{symbol_context} contains duplicate {filter_type} filters")
            filters[filter_type] = filter_payload

        try:
            lot_filter = filters["LOT_SIZE"]
            price_filter = filters["PRICE_FILTER"]
            notional_filter = filters.get("MIN_NOTIONAL") or filters["NOTIONAL"]
        except KeyError as exc:
            raise BinancePerpetualRiskDataError(
                f"{symbol_context} must contain PRICE_FILTER, LOT_SIZE, and MIN_NOTIONAL/NOTIONAL filters"
            ) from exc

        min_order_size = _decimal(lot_filter, "minQty", f"{symbol_context}.LOT_SIZE")
        step_size = _decimal(lot_filter, "stepSize", f"{symbol_context}.LOT_SIZE")
        tick_size = _decimal(price_filter, "tickSize", f"{symbol_context}.PRICE_FILTER")
        if "notional" in notional_filter:
            min_notional = _decimal(notional_filter, "notional", f"{symbol_context}.MIN_NOTIONAL")
        else:
            min_notional = _decimal(notional_filter, "minNotional", f"{symbol_context}.NOTIONAL")
        for field, value in (
            ("minQty", min_order_size),
            ("stepSize", step_size),
            ("tickSize", tick_size),
            ("minNotional", min_notional),
        ):
            if value <= 0:
                raise BinancePerpetualRiskDataError(f"{symbol_context}.{field} must be positive")

        source_time_ms = None
        if "serverTime" in payload:
            source_time_ms = _timestamp_ms(payload, "serverTime", context)
        return cls(
            symbol=_string(symbol, "symbol", symbol_context),
            trading_pair=trading_pair,
            contract_type=contract_type,
            status=_string(symbol, "status", symbol_context),
            base_asset=_string(symbol, "baseAsset", symbol_context),
            quote_asset=_string(symbol, "quoteAsset", symbol_context),
            margin_asset=_string(symbol, "marginAsset", symbol_context),
            contract_multiplier=contract_multiplier,
            min_order_size=min_order_size,
            step_size=step_size,
            tick_size=tick_size,
            min_notional=min_notional,
            source_time_ms=source_time_ms,
            data_time=_data_time(data_time),
        )


@dataclass(frozen=True)
class BinancePerpetualAccountAsset:
    asset: str
    wallet_balance: Decimal
    unrealized_profit: Decimal
    margin_balance: Decimal
    maint_margin: Decimal
    initial_margin: Decimal
    position_initial_margin: Decimal
    open_order_initial_margin: Decimal
    cross_wallet_balance: Decimal
    cross_unrealized_profit: Decimal
    available_balance: Decimal
    max_withdraw_amount: Decimal
    update_time_ms: int

    @property
    def has_activity(self) -> bool:
        return any(value != 0 for value in (
            self.wallet_balance,
            self.unrealized_profit,
            self.margin_balance,
            self.maint_margin,
            self.initial_margin,
            self.position_initial_margin,
            self.open_order_initial_margin,
            self.cross_wallet_balance,
            self.cross_unrealized_profit,
            self.available_balance,
            self.max_withdraw_amount,
        ))

    def validate(self, context: str = "Account Information V3 asset") -> None:
        if not isinstance(self.asset, str) or self.asset == "":
            raise BinancePerpetualRiskDataError(f"{context}.asset must be a non-empty string")
        decimal_fields = (
            ("walletBalance", self.wallet_balance),
            ("unrealizedProfit", self.unrealized_profit),
            ("marginBalance", self.margin_balance),
            ("maintMargin", self.maint_margin),
            ("initialMargin", self.initial_margin),
            ("positionInitialMargin", self.position_initial_margin),
            ("openOrderInitialMargin", self.open_order_initial_margin),
            ("crossWalletBalance", self.cross_wallet_balance),
            ("crossUnPnl", self.cross_unrealized_profit),
            ("availableBalance", self.available_balance),
            ("maxWithdrawAmount", self.max_withdraw_amount),
        )
        non_negative_fields = {
            "maintMargin",
            "initialMargin",
            "positionInitialMargin",
            "openOrderInitialMargin",
        }
        for field, value in decimal_fields:
            _validate_decimal_value(
                value,
                f"{context}.{field}",
                non_negative=field in non_negative_fields,
            )
        _validate_maintenance_margin_relationship(
            maintenance_margin=self.maint_margin,
            initial_margin=self.initial_margin,
            context=context,
        )
        _validate_non_negative_integer_value(self.update_time_ms, f"{context}.updateTime")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], context: str) -> "BinancePerpetualAccountAsset":
        payload = _mapping(payload, context)
        fact = cls(
            asset=_string(payload, "asset", context),
            wallet_balance=_decimal(payload, "walletBalance", context),
            unrealized_profit=_decimal(payload, "unrealizedProfit", context),
            margin_balance=_decimal(payload, "marginBalance", context),
            maint_margin=_decimal(payload, "maintMargin", context),
            initial_margin=_decimal(payload, "initialMargin", context),
            position_initial_margin=_decimal(payload, "positionInitialMargin", context),
            open_order_initial_margin=_decimal(payload, "openOrderInitialMargin", context),
            cross_wallet_balance=_decimal(payload, "crossWalletBalance", context),
            cross_unrealized_profit=_decimal(payload, "crossUnPnl", context),
            available_balance=_decimal(payload, "availableBalance", context),
            max_withdraw_amount=_decimal(payload, "maxWithdrawAmount", context),
            update_time_ms=_timestamp_ms(payload, "updateTime", context),
        )
        fact.validate(context)
        return fact


@dataclass(frozen=True)
class BinancePerpetualAccountPosition:
    symbol: str
    position_side: str
    position_amount: Decimal
    unrealized_profit: Decimal
    isolated_margin: Decimal
    notional: Decimal
    isolated_wallet: Decimal
    initial_margin: Decimal
    maint_margin: Decimal
    update_time_ms: int

    @property
    def has_activity(self) -> bool:
        return any(value != 0 for value in (
            self.position_amount,
            self.unrealized_profit,
            self.isolated_margin,
            self.notional,
            self.isolated_wallet,
            self.initial_margin,
            self.maint_margin,
        ))

    def validate(self, context: str = "Account Information V3 position") -> None:
        if not isinstance(self.symbol, str) or self.symbol == "":
            raise BinancePerpetualRiskDataError(f"{context}.symbol must be a non-empty string")
        _validate_position_identity_and_notional(
            position_side=self.position_side,
            position_amount=self.position_amount,
            notional=self.notional,
            context=context,
        )
        decimal_fields = (
            ("unrealizedProfit", self.unrealized_profit, False),
            ("isolatedMargin", self.isolated_margin, True),
            ("isolatedWallet", self.isolated_wallet, True),
            ("initialMargin", self.initial_margin, True),
            ("maintMargin", self.maint_margin, True),
        )
        for field, value, non_negative in decimal_fields:
            _validate_decimal_value(
                value,
                f"{context}.{field}",
                non_negative=non_negative,
            )
        _validate_maintenance_margin_relationship(
            maintenance_margin=self.maint_margin,
            initial_margin=self.initial_margin,
            context=context,
        )
        _validate_non_negative_integer_value(self.update_time_ms, f"{context}.updateTime")

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], context: str) -> "BinancePerpetualAccountPosition":
        payload = _mapping(payload, context)
        fact = cls(
            symbol=_string(payload, "symbol", context),
            position_side=_string(payload, "positionSide", context),
            position_amount=_decimal(payload, "positionAmt", context),
            unrealized_profit=_decimal(payload, "unrealizedProfit", context),
            isolated_margin=_decimal(payload, "isolatedMargin", context),
            notional=_decimal(payload, "notional", context),
            isolated_wallet=_decimal(payload, "isolatedWallet", context),
            initial_margin=_decimal(payload, "initialMargin", context),
            maint_margin=_decimal(payload, "maintMargin", context),
            update_time_ms=_timestamp_ms(payload, "updateTime", context),
        )
        fact.validate(context)
        return fact


@dataclass(frozen=True)
class BinancePerpetualAccountRiskSnapshot:
    total_initial_margin: Decimal
    total_maint_margin: Decimal
    total_wallet_balance: Decimal
    total_unrealized_profit: Decimal
    total_margin_balance: Decimal
    total_position_initial_margin: Decimal
    total_open_order_initial_margin: Decimal
    total_cross_wallet_balance: Decimal
    total_cross_unrealized_profit: Decimal
    available_balance: Decimal
    max_withdraw_amount: Decimal
    assets: Tuple[BinancePerpetualAccountAsset, ...]
    positions: Tuple[BinancePerpetualAccountPosition, ...]
    data_time: float

    def validate(self, context: str = "Account Information V3") -> None:
        decimal_fields = (
            ("totalInitialMargin", self.total_initial_margin),
            ("totalMaintMargin", self.total_maint_margin),
            ("totalWalletBalance", self.total_wallet_balance),
            ("totalUnrealizedProfit", self.total_unrealized_profit),
            ("totalMarginBalance", self.total_margin_balance),
            ("totalPositionInitialMargin", self.total_position_initial_margin),
            ("totalOpenOrderInitialMargin", self.total_open_order_initial_margin),
            ("totalCrossWalletBalance", self.total_cross_wallet_balance),
            ("totalCrossUnPnl", self.total_cross_unrealized_profit),
            ("availableBalance", self.available_balance),
            ("maxWithdrawAmount", self.max_withdraw_amount),
        )
        non_negative_fields = {
            "totalInitialMargin",
            "totalMaintMargin",
            "totalPositionInitialMargin",
            "totalOpenOrderInitialMargin",
        }
        for field, value in decimal_fields:
            _validate_decimal_value(
                value,
                f"{context}.{field}",
                non_negative=field in non_negative_fields,
            )
        _validate_maintenance_margin_relationship(
            maintenance_margin=self.total_maint_margin,
            initial_margin=self.total_initial_margin,
            context=context,
        )

        asset_identities = set()
        for index, asset in enumerate(self.assets):
            if not isinstance(asset, BinancePerpetualAccountAsset):
                raise BinancePerpetualRiskDataError(f"{context}.assets[{index}] has an invalid type")
            asset.validate(f"{context}.assets[{index}]")
            if asset.asset in asset_identities:
                raise BinancePerpetualRiskDataError(
                    f"{context} contains duplicate asset identity {asset.asset}"
                )
            asset_identities.add(asset.asset)

        position_identities = set()
        for index, position in enumerate(self.positions):
            if not isinstance(position, BinancePerpetualAccountPosition):
                raise BinancePerpetualRiskDataError(f"{context}.positions[{index}] has an invalid type")
            position.validate(f"{context}.positions[{index}]")
            identity = (position.symbol, position.position_side)
            if identity in position_identities:
                raise BinancePerpetualRiskDataError(
                    f"{context} contains duplicate position identity {identity}"
                )
            position_identities.add(identity)
        _data_time(self.data_time)

    @classmethod
    def from_payload(
            cls,
            payload: Mapping[str, Any],
            data_time: float,
    ) -> "BinancePerpetualAccountRiskSnapshot":
        context = "Account Information V3"
        payload = _mapping(payload, context)
        assets = tuple(
            BinancePerpetualAccountAsset.from_payload(item, f"{context}.assets[{index}]")
            for index, item in enumerate(
                _sequence(_required(payload, "assets", context), f"{context}.assets")
            )
        )
        positions = tuple(
            BinancePerpetualAccountPosition.from_payload(item, f"{context}.positions[{index}]")
            for index, item in enumerate(
                _sequence(_required(payload, "positions", context), f"{context}.positions")
            )
        )
        fact = cls(
            total_initial_margin=_decimal(payload, "totalInitialMargin", context),
            total_maint_margin=_decimal(payload, "totalMaintMargin", context),
            total_wallet_balance=_decimal(payload, "totalWalletBalance", context),
            total_unrealized_profit=_decimal(payload, "totalUnrealizedProfit", context),
            total_margin_balance=_decimal(payload, "totalMarginBalance", context),
            total_position_initial_margin=_decimal(payload, "totalPositionInitialMargin", context),
            total_open_order_initial_margin=_decimal(payload, "totalOpenOrderInitialMargin", context),
            total_cross_wallet_balance=_decimal(payload, "totalCrossWalletBalance", context),
            total_cross_unrealized_profit=_decimal(payload, "totalCrossUnPnl", context),
            available_balance=_decimal(payload, "availableBalance", context),
            max_withdraw_amount=_decimal(payload, "maxWithdrawAmount", context),
            assets=assets,
            positions=positions,
            data_time=_data_time(data_time),
        )
        fact.validate(context)
        return fact


@dataclass(frozen=True)
class BinancePerpetualPositionRiskSnapshot:
    symbol: str
    position_side: str
    position_amount: Decimal
    entry_price: Decimal
    break_even_price: Decimal
    mark_price: Decimal
    unrealized_profit: Decimal
    liquidation_price: Decimal
    isolated_margin: Decimal
    notional: Decimal
    margin_asset: str
    isolated_wallet: Decimal
    initial_margin: Decimal
    maint_margin: Decimal
    position_initial_margin: Decimal
    open_order_initial_margin: Decimal
    adl: int
    bid_notional: Decimal
    ask_notional: Decimal
    update_time_ms: int
    data_time: float

    @property
    def has_activity(self) -> bool:
        return any(value != 0 for value in (
            self.position_amount,
            self.unrealized_profit,
            self.isolated_margin,
            self.notional,
            self.isolated_wallet,
            self.initial_margin,
            self.maint_margin,
            self.position_initial_margin,
            self.open_order_initial_margin,
            self.bid_notional,
            self.ask_notional,
        ))

    def validate(self, context: str = "Position Information V3") -> None:
        if not isinstance(self.symbol, str) or self.symbol == "":
            raise BinancePerpetualRiskDataError(f"{context}.symbol must be a non-empty string")
        if not isinstance(self.margin_asset, str) or self.margin_asset == "":
            raise BinancePerpetualRiskDataError(f"{context}.marginAsset must be a non-empty string")
        _validate_position_identity_and_notional(
            position_side=self.position_side,
            position_amount=self.position_amount,
            notional=self.notional,
            context=context,
        )
        decimal_fields = (
            ("entryPrice", self.entry_price, True),
            ("breakEvenPrice", self.break_even_price, True),
            ("markPrice", self.mark_price, True),
            ("unRealizedProfit", self.unrealized_profit, False),
            ("liquidationPrice", self.liquidation_price, True),
            ("isolatedMargin", self.isolated_margin, True),
            ("isolatedWallet", self.isolated_wallet, True),
            ("initialMargin", self.initial_margin, True),
            ("maintMargin", self.maint_margin, True),
            ("positionInitialMargin", self.position_initial_margin, True),
            ("openOrderInitialMargin", self.open_order_initial_margin, True),
            ("bidNotional", self.bid_notional, True),
            ("askNotional", self.ask_notional, True),
        )
        for field, value, non_negative in decimal_fields:
            _validate_decimal_value(
                value,
                f"{context}.{field}",
                non_negative=non_negative,
            )
        if self.position_amount != 0:
            for field, value in (
                ("entryPrice", self.entry_price),
                ("breakEvenPrice", self.break_even_price),
            ):
                if value <= 0:
                    raise BinancePerpetualRiskDataError(
                        f"{context}.{field} must be positive for an active position"
                    )
        if self.has_activity and self.mark_price <= 0:
            raise BinancePerpetualRiskDataError(
                f"{context}.markPrice must be positive for active risk data"
            )
        _validate_maintenance_margin_relationship(
            maintenance_margin=self.maint_margin,
            initial_margin=self.initial_margin,
            context=context,
        )
        _validate_non_negative_integer_value(self.adl, f"{context}.adl")
        if self.adl > 4:
            raise BinancePerpetualRiskDataError(f"{context}.adl must be between 0 and 4")
        _validate_non_negative_integer_value(self.update_time_ms, f"{context}.updateTime")
        _data_time(self.data_time)

    def validate_notional_magnitude(
            self,
            *,
            position_amount_multiplier: Decimal,
            tolerance: Decimal,
            context: str = "Position Information V3",
    ) -> None:
        self.validate(context)
        _validate_decimal_value(
            position_amount_multiplier,
            f"{context}.position amount multiplier",
            positive=True,
        )
        _validate_decimal_value(
            tolerance,
            f"{context}.notional tolerance",
            non_negative=True,
        )
        expected_notional = _exact_decimal_product_components((
            (self.position_amount, f"{context}.positionAmt"),
            (self.mark_price, f"{context}.markPrice"),
            (position_amount_multiplier, f"{context}.position amount multiplier"),
        ))
        if not _decimal_distance_within_tolerance(
                left=self.notional,
                right_components=expected_notional,
                tolerance=tolerance,
                context=context,
        ):
            raise BinancePerpetualRiskDataError(
                f"{context} signed notional magnitude contradicts position amount and mark price"
            )

    @classmethod
    def from_payload(
            cls,
            payload: Mapping[str, Any],
            data_time: float,
    ) -> "BinancePerpetualPositionRiskSnapshot":
        context = "Position Information V3"
        payload = _mapping(payload, context)
        fact = cls(
            symbol=_string(payload, "symbol", context),
            position_side=_string(payload, "positionSide", context),
            position_amount=_decimal(payload, "positionAmt", context),
            entry_price=_decimal(payload, "entryPrice", context),
            break_even_price=_decimal(payload, "breakEvenPrice", context),
            mark_price=_decimal(payload, "markPrice", context),
            unrealized_profit=_decimal(payload, "unRealizedProfit", context),
            liquidation_price=_decimal(payload, "liquidationPrice", context),
            isolated_margin=_decimal(payload, "isolatedMargin", context),
            notional=_decimal(payload, "notional", context),
            margin_asset=_string(payload, "marginAsset", context),
            isolated_wallet=_decimal(payload, "isolatedWallet", context),
            initial_margin=_decimal(payload, "initialMargin", context),
            maint_margin=_decimal(payload, "maintMargin", context),
            position_initial_margin=_decimal(payload, "positionInitialMargin", context),
            open_order_initial_margin=_decimal(payload, "openOrderInitialMargin", context),
            adl=_integer(payload, "adl", context),
            bid_notional=_decimal(payload, "bidNotional", context),
            ask_notional=_decimal(payload, "askNotional", context),
            update_time_ms=_timestamp_ms(payload, "updateTime", context),
            data_time=_data_time(data_time),
        )
        fact.validate(context)
        return fact


@dataclass(frozen=True)
class BinancePerpetualAccountConfig:
    can_trade: bool
    can_deposit: bool
    can_withdraw: bool
    dual_side_position: bool
    multi_assets_margin: bool
    update_time_ms: int
    data_time: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], data_time: float) -> "BinancePerpetualAccountConfig":
        context = "account configuration"
        payload = _mapping(payload, context)
        return cls(
            can_trade=_boolean(payload, "canTrade", context),
            can_deposit=_boolean(payload, "canDeposit", context),
            can_withdraw=_boolean(payload, "canWithdraw", context),
            dual_side_position=_boolean(payload, "dualSidePosition", context),
            multi_assets_margin=_boolean(payload, "multiAssetsMargin", context),
            update_time_ms=_timestamp_ms(payload, "updateTime", context),
            data_time=_data_time(data_time),
        )


class BinancePerpetualMarginType(str, Enum):
    CROSSED = "CROSSED"
    ISOLATED = "ISOLATED"


@dataclass(frozen=True)
class BinancePerpetualSymbolConfig:
    symbol: str
    margin_type: BinancePerpetualMarginType
    is_auto_add_margin: bool
    leverage: int
    max_notional_value: Decimal
    data_time: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], data_time: float) -> "BinancePerpetualSymbolConfig":
        context = "symbol configuration"
        payload = _mapping(payload, context)
        raw_margin_type = _string(payload, "marginType", context).upper()
        if raw_margin_type in ("CROSS", "CROSSED"):
            margin_type = BinancePerpetualMarginType.CROSSED
        elif raw_margin_type == "ISOLATED":
            margin_type = BinancePerpetualMarginType.ISOLATED
        else:
            raise BinancePerpetualRiskDataError(
                f"{context}.marginType has unsupported value {raw_margin_type}"
            )
        leverage = _integer(payload, "leverage", context)
        max_notional_value = _decimal(payload, "maxNotionalValue", context)
        if leverage <= 0:
            raise BinancePerpetualRiskDataError(f"{context}.leverage must be positive")
        if max_notional_value <= 0:
            raise BinancePerpetualRiskDataError(f"{context}.maxNotionalValue must be positive")
        return cls(
            symbol=_string(payload, "symbol", context),
            margin_type=margin_type,
            is_auto_add_margin=_boolean(payload, "isAutoAddMargin", context),
            leverage=leverage,
            max_notional_value=max_notional_value,
            data_time=_data_time(data_time),
        )


@dataclass(frozen=True)
class BinancePerpetualMultiAssetsMode:
    enabled: bool
    data_time: float

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], data_time: float) -> "BinancePerpetualMultiAssetsMode":
        context = "Multi-Assets mode"
        payload = _mapping(payload, context)
        return cls(
            enabled=_boolean(payload, "multiAssetsMargin", context),
            data_time=_data_time(data_time),
        )


@dataclass(frozen=True)
class BinancePerpetualPositionMode:
    dual_side_position: bool
    data_time: float

    @property
    def is_one_way(self) -> bool:
        return not self.dual_side_position

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any], data_time: float) -> "BinancePerpetualPositionMode":
        context = "position mode"
        payload = _mapping(payload, context)
        return cls(
            dual_side_position=_boolean(payload, "dualSidePosition", context),
            data_time=_data_time(data_time),
        )


@dataclass(frozen=True)
class BinancePerpetualLeverageBracket:
    bracket: int
    initial_leverage: int
    notional_cap: Decimal
    notional_floor: Decimal
    maint_margin_ratio: Decimal
    cum: Decimal
    notional_coef: Decimal

    @property
    def adjusted_notional_cap(self) -> Decimal:
        return self.notional_cap * self.notional_coef

    @property
    def adjusted_notional_floor(self) -> Decimal:
        return self.notional_floor * self.notional_coef

    @property
    def adjusted_cum(self) -> Decimal:
        return self.cum * self.notional_coef

    def contains(self, notional: Decimal) -> bool:
        absolute_notional = abs(notional)
        return self.adjusted_notional_floor <= absolute_notional < self.adjusted_notional_cap


@dataclass(frozen=True)
class BinancePerpetualLeverageBrackets:
    symbol: str
    notional_coef: Decimal
    brackets: Tuple[BinancePerpetualLeverageBracket, ...]
    data_time: float
    cache_time: float

    @classmethod
    def from_payload(
            cls,
            payload: Any,
            expected_symbol: str,
            data_time: float,
            cache_time: float,
    ) -> "BinancePerpetualLeverageBrackets":
        context = "leverage brackets"
        if isinstance(payload, Mapping):
            candidates = [payload]
        else:
            candidates = list(_sequence(payload, context))
        matches = []
        for index, raw_candidate in enumerate(candidates):
            candidate = _mapping(raw_candidate, f"{context}[{index}]")
            if _string(candidate, "symbol", f"{context}[{index}]") == expected_symbol:
                matches.append(candidate)
        if len(matches) != 1:
            raise BinancePerpetualRiskDataError(
                f"{context} must contain exactly one record for symbol {expected_symbol}"
            )
        record = matches[0]
        notional_coef = _decimal(record, "notionalCoef", context) if "notionalCoef" in record else Decimal("1")
        if notional_coef <= 0:
            raise BinancePerpetualRiskDataError(f"{context}.notionalCoef must be positive")
        raw_brackets = _sequence(_required(record, "brackets", context), f"{context}.brackets")
        if len(raw_brackets) == 0:
            raise BinancePerpetualRiskDataError(f"{context}.brackets must not be empty")

        parsed_brackets = []
        previous = None
        for index, raw_bracket in enumerate(raw_brackets):
            bracket_context = f"{context}.brackets[{index}]"
            raw_bracket = _mapping(raw_bracket, bracket_context)
            bracket_number = _integer(raw_bracket, "bracket", bracket_context)
            initial_leverage = _integer(raw_bracket, "initialLeverage", bracket_context)
            notional_cap = _decimal(raw_bracket, "notionalCap", bracket_context)
            notional_floor = _decimal(raw_bracket, "notionalFloor", bracket_context)
            maint_margin_ratio = _decimal(raw_bracket, "maintMarginRatio", bracket_context)
            cum = _decimal(raw_bracket, "cum", bracket_context)
            if bracket_number <= 0 or initial_leverage <= 0:
                raise BinancePerpetualRiskDataError(f"{bracket_context} bracket and leverage must be positive")
            if notional_floor < 0 or notional_cap <= notional_floor:
                raise BinancePerpetualRiskDataError(f"{bracket_context} has invalid notional bounds")
            if maint_margin_ratio < 0 or cum < 0:
                raise BinancePerpetualRiskDataError(f"{bracket_context} margin values must be non-negative")
            if previous is None and notional_floor != 0:
                raise BinancePerpetualRiskDataError(f"{context} first bracket must start at zero")
            if previous is not None:
                if bracket_number <= previous.bracket:
                    raise BinancePerpetualRiskDataError(f"{context} bracket identifiers must be increasing")
                if notional_floor != previous.notional_cap:
                    raise BinancePerpetualRiskDataError(f"{context} notional bounds must be contiguous")
                if initial_leverage > previous.initial_leverage:
                    raise BinancePerpetualRiskDataError(
                        f"{context} initial leverage limits must be non-increasing"
                    )
            parsed = BinancePerpetualLeverageBracket(
                bracket=bracket_number,
                initial_leverage=initial_leverage,
                notional_cap=notional_cap,
                notional_floor=notional_floor,
                maint_margin_ratio=maint_margin_ratio,
                cum=cum,
                notional_coef=notional_coef,
            )
            parsed_brackets.append(parsed)
            previous = parsed

        return cls(
            symbol=expected_symbol,
            notional_coef=notional_coef,
            brackets=tuple(parsed_brackets),
            data_time=_data_time(data_time),
            cache_time=_data_time(cache_time),
        )

    def bracket_for_notional(self, notional: Decimal) -> BinancePerpetualLeverageBracket:
        if not isinstance(notional, Decimal) or not notional.is_finite():
            raise BinancePerpetualRiskDataError("notional must be a finite Decimal")
        for bracket in self.brackets:
            if bracket.contains(notional):
                return bracket
        raise BinancePerpetualRiskDataError(
            f"absolute notional {abs(notional)} is outside the returned leverage brackets for {self.symbol}"
        )

    def max_notional_for_leverage(self, leverage: int) -> Decimal:
        if isinstance(leverage, bool) or not isinstance(leverage, int) or leverage <= 0:
            raise BinancePerpetualRiskDataError("leverage must be a positive integer")
        eligible = tuple(
            bracket for bracket in self.brackets
            if leverage <= bracket.initial_leverage
        )
        if not eligible:
            raise BinancePerpetualRiskDataError(
                f"leverage {leverage} is not allowed by the returned brackets for {self.symbol}"
            )
        return eligible[-1].adjusted_notional_cap

    def maintenance_margin(self, notional: Decimal) -> Decimal:
        absolute_notional = abs(notional)
        bracket = self.bracket_for_notional(absolute_notional)
        return max(Decimal("0"), absolute_notional * bracket.maint_margin_ratio - bracket.adjusted_cum)


@dataclass(frozen=True)
class BinancePerpetualLeverageChangeResult:
    symbol: str
    leverage: int
    max_notional_value: Decimal
    data_time: float

    @classmethod
    def from_payload(
            cls,
            payload: Mapping[str, Any],
            data_time: float,
    ) -> "BinancePerpetualLeverageChangeResult":
        context = "change leverage response"
        payload = _mapping(payload, context)
        leverage = _integer(payload, "leverage", context)
        max_notional_value = _decimal(payload, "maxNotionalValue", context)
        if leverage <= 0:
            raise BinancePerpetualRiskDataError(f"{context}.leverage must be positive")
        if max_notional_value <= 0:
            raise BinancePerpetualRiskDataError(f"{context}.maxNotionalValue must be positive")
        return cls(
            symbol=_string(payload, "symbol", context),
            leverage=leverage,
            max_notional_value=max_notional_value,
            data_time=_data_time(data_time),
        )


@dataclass(frozen=True)
class BinancePerpetualPreflightSnapshot:
    account: BinancePerpetualAccountRiskSnapshot
    positions: Tuple[BinancePerpetualPositionRiskSnapshot, ...]
    account_config: BinancePerpetualAccountConfig
    multi_assets_mode: BinancePerpetualMultiAssetsMode
    position_mode: BinancePerpetualPositionMode
    instruments: Tuple[BinancePerpetualInstrumentInfo, ...]
    symbol_configs: Tuple[BinancePerpetualSymbolConfig, ...]
    leverage_brackets: Tuple[BinancePerpetualLeverageBrackets, ...]
    data_time: float
