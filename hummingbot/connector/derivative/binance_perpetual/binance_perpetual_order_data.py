from __future__ import annotations

import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Mapping, Optional

from hummingbot.core.data_type.common import OrderType, TradeType


BINANCE_CLIENT_ORDER_ID_MAX_LENGTH = 36
_BINANCE_CLIENT_ORDER_ID_PATTERN = re.compile(r"^[.A-Za-z0-9_:/-]+$")


class BinancePerpetualOrderDataError(ValueError):
    """Raised when a signed Binance reconciliation response is malformed or contradictory."""


class BinancePerpetualOrderSubmissionUnknown(IOError):
    """Signals that an order request may have reached Binance but did not receive an outcome."""

    def __init__(self, client_order_id: str):
        self.client_order_id = client_order_id
        super().__init__(f"submission outcome is unknown for client order ID {client_order_id}")


def validate_binance_client_order_id(
        client_order_id: Any,
        max_length: int = BINANCE_CLIENT_ORDER_ID_MAX_LENGTH,
) -> str:
    if not isinstance(client_order_id, str):
        raise BinancePerpetualOrderDataError("client order ID must be a string")
    if not 1 <= len(client_order_id) <= max_length:
        raise BinancePerpetualOrderDataError("client order ID has an invalid length")
    if _BINANCE_CLIENT_ORDER_ID_PATTERN.fullmatch(client_order_id) is None:
        raise BinancePerpetualOrderDataError("client order ID has an invalid format")
    return client_order_id


def validate_binance_exchange_order_id(exchange_order_id: Any) -> str:
    if (
        not isinstance(exchange_order_id, str)
        or not exchange_order_id.isascii()
        or not exchange_order_id.isdigit()
        or (exchange_order_id != "0" and exchange_order_id.startswith("0"))
    ):
        raise BinancePerpetualOrderDataError("exchange order ID must be a non-negative integer string")
    return exchange_order_id


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise BinancePerpetualOrderDataError(f"{context} must be an object")
    return value


def _required(payload: Mapping[str, Any], field: str, context: str) -> Any:
    if field not in payload:
        raise BinancePerpetualOrderDataError(f"{context} is missing required field {field}")
    return payload[field]


def _string(payload: Mapping[str, Any], field: str, context: str) -> str:
    value = _required(payload, field, context)
    if not isinstance(value, str) or value == "":
        raise BinancePerpetualOrderDataError(f"{context}.{field} must be a non-empty string")
    return value


def _numeric_identifier(payload: Mapping[str, Any], field: str, context: str) -> str:
    value = _required(payload, field, context)
    if isinstance(value, bool):
        raise BinancePerpetualOrderDataError(f"{context}.{field} must be a non-negative integer ID")
    if isinstance(value, int):
        if value < 0:
            raise BinancePerpetualOrderDataError(f"{context}.{field} must be a non-negative integer ID")
        return str(value)
    if (
        isinstance(value, str)
        and value.isascii()
        and value.isdigit()
        and (value == "0" or not value.startswith("0"))
    ):
        return value
    raise BinancePerpetualOrderDataError(f"{context}.{field} must be a non-negative integer ID")


def _decimal(payload: Mapping[str, Any], field: str, context: str) -> Decimal:
    value = _required(payload, field, context)
    if not isinstance(value, str) or value == "" or value.strip() != value:
        raise BinancePerpetualOrderDataError(f"{context}.{field} must be a finite decimal")
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError, ValueError):
        raise BinancePerpetualOrderDataError(f"{context}.{field} must be a finite decimal") from None
    if not parsed.is_finite():
        raise BinancePerpetualOrderDataError(f"{context}.{field} must be a finite decimal")
    return parsed


def _boolean(payload: Mapping[str, Any], field: str, context: str) -> bool:
    value = _required(payload, field, context)
    if type(value) is not bool:
        raise BinancePerpetualOrderDataError(f"{context}.{field} must be a boolean")
    return value


def _position_side(payload: Mapping[str, Any], context: str) -> str:
    value = _string(payload, "positionSide", context)
    if value not in {"BOTH", "LONG", "SHORT"}:
        raise BinancePerpetualOrderDataError(f"{context}.positionSide is unsupported")
    return value


def _timestamp_ms(payload: Mapping[str, Any], field: str, context: str) -> int:
    value = _required(payload, field, context)
    if isinstance(value, bool):
        raise BinancePerpetualOrderDataError(f"{context}.{field} must be a non-negative integer timestamp")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str) and value.isascii() and value.isdigit():
        parsed = int(value)
    else:
        raise BinancePerpetualOrderDataError(f"{context}.{field} must be a non-negative integer timestamp")
    if parsed < 0:
        raise BinancePerpetualOrderDataError(f"{context}.{field} must be a non-negative integer timestamp")
    return parsed


def _data_time(value: Any) -> float:
    if isinstance(value, bool):
        raise BinancePerpetualOrderDataError("data_time must be a finite non-negative timestamp")
    try:
        parsed = float(value)
    except (OverflowError, TypeError, ValueError):
        raise BinancePerpetualOrderDataError("data_time must be a finite non-negative timestamp") from None
    if not math.isfinite(parsed) or parsed < 0:
        raise BinancePerpetualOrderDataError("data_time must be a finite non-negative timestamp")
    return parsed


class BinancePerpetualOrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    EXPIRED_IN_MATCH = "EXPIRED_IN_MATCH"
    NOT_FOUND = "NOT_FOUND"

    @property
    def is_terminal(self) -> bool:
        return self in {
            self.FILLED,
            self.CANCELED,
            self.EXPIRED,
            self.REJECTED,
            self.EXPIRED_IN_MATCH,
        }


@dataclass(frozen=True)
class BinancePerpetualOrderSnapshot:
    client_order_id: str
    exchange_order_id: Optional[str]
    symbol: str
    trading_pair: str
    status: BinancePerpetualOrderStatus
    side: Optional[TradeType]
    position_side: Optional[str]
    order_type: Optional[OrderType]
    time_in_force: Optional[str]
    price: Optional[Decimal]
    average_price: Optional[Decimal]
    original_quantity: Optional[Decimal]
    executed_quantity: Optional[Decimal]
    cumulative_quote_quantity: Optional[Decimal]
    reduce_only: Optional[bool]
    close_position: Optional[bool]
    update_time_ms: Optional[int]
    data_time: float

    @property
    def is_not_found(self) -> bool:
        return self.status is BinancePerpetualOrderStatus.NOT_FOUND

    @property
    def is_terminal(self) -> bool:
        return self.status.is_terminal

    @classmethod
    def not_found(
            cls,
            client_order_id: str,
            symbol: str,
            trading_pair: str,
            data_time: float,
    ) -> "BinancePerpetualOrderSnapshot":
        return cls(
            client_order_id=validate_binance_client_order_id(client_order_id),
            exchange_order_id=None,
            symbol=symbol,
            trading_pair=trading_pair,
            status=BinancePerpetualOrderStatus.NOT_FOUND,
            side=None,
            position_side=None,
            order_type=None,
            time_in_force=None,
            price=None,
            average_price=None,
            original_quantity=None,
            executed_quantity=None,
            cumulative_quote_quantity=None,
            reduce_only=None,
            close_position=None,
            update_time_ms=None,
            data_time=_data_time(data_time),
        )

    @classmethod
    def from_payload(
            cls,
            payload: Mapping[str, Any],
            expected_symbol: str,
            trading_pair: str,
            data_time: float,
            expected_client_order_id: Optional[str] = None,
    ) -> "BinancePerpetualOrderSnapshot":
        context = "order reconciliation"
        payload = _mapping(payload, context)
        client_order_id = validate_binance_client_order_id(
            _string(payload, "clientOrderId", context)
        )
        if expected_client_order_id is not None and client_order_id != expected_client_order_id:
            raise BinancePerpetualOrderDataError("order reconciliation client order ID does not match request")
        symbol = _string(payload, "symbol", context)
        if symbol != expected_symbol:
            raise BinancePerpetualOrderDataError("order reconciliation symbol does not match request")

        raw_status = _string(payload, "status", context)
        try:
            status = BinancePerpetualOrderStatus(raw_status)
        except ValueError:
            raise BinancePerpetualOrderDataError("order reconciliation.status is unsupported") from None
        if status is BinancePerpetualOrderStatus.NOT_FOUND:
            raise BinancePerpetualOrderDataError("order reconciliation cannot encode NOT_FOUND as an order payload")

        raw_side = _string(payload, "side", context)
        if raw_side == "BUY":
            side = TradeType.BUY
        elif raw_side == "SELL":
            side = TradeType.SELL
        else:
            raise BinancePerpetualOrderDataError("order reconciliation.side is unsupported")

        raw_order_type = _string(payload, "type", context)
        if raw_order_type == "LIMIT":
            order_type = OrderType.LIMIT
        elif raw_order_type == "MARKET":
            order_type = OrderType.MARKET
        else:
            raise BinancePerpetualOrderDataError("order reconciliation.type is unsupported")

        price = _decimal(payload, "price", context)
        average_price = _decimal(payload, "avgPrice", context)
        original_quantity = _decimal(payload, "origQty", context)
        executed_quantity = _decimal(payload, "executedQty", context)
        cumulative_quote_quantity = _decimal(payload, "cumQuote", context)
        if any(value < 0 for value in (
            price,
            average_price,
            original_quantity,
            executed_quantity,
            cumulative_quote_quantity,
        )):
            raise BinancePerpetualOrderDataError("order reconciliation quantities and prices must be non-negative")
        if original_quantity == 0:
            raise BinancePerpetualOrderDataError("order reconciliation original quantity must be positive")
        if executed_quantity > original_quantity:
            raise BinancePerpetualOrderDataError("order reconciliation executed quantity exceeds original quantity")
        if status is BinancePerpetualOrderStatus.NEW and executed_quantity != 0:
            raise BinancePerpetualOrderDataError("NEW order reconciliation must have zero executed quantity")
        if status is BinancePerpetualOrderStatus.PARTIALLY_FILLED and not (
            0 < executed_quantity < original_quantity
        ):
            raise BinancePerpetualOrderDataError(
                "PARTIALLY_FILLED order reconciliation must have an incomplete positive fill"
            )
        if status is BinancePerpetualOrderStatus.FILLED and executed_quantity != original_quantity:
            raise BinancePerpetualOrderDataError("FILLED order reconciliation must have the full quantity executed")
        if status is BinancePerpetualOrderStatus.REJECTED and executed_quantity != 0:
            raise BinancePerpetualOrderDataError("REJECTED order reconciliation must have zero executed quantity")
        if executed_quantity == 0 and (average_price != 0 or cumulative_quote_quantity != 0):
            raise BinancePerpetualOrderDataError("unfilled order reconciliation contains fill values")
        if executed_quantity > 0 and (average_price <= 0 or cumulative_quote_quantity <= 0):
            raise BinancePerpetualOrderDataError("filled order reconciliation contains invalid fill values")

        return cls(
            client_order_id=client_order_id,
            exchange_order_id=_numeric_identifier(payload, "orderId", context),
            symbol=symbol,
            trading_pair=trading_pair,
            status=status,
            side=side,
            position_side=_position_side(payload, context),
            order_type=order_type,
            time_in_force=_string(payload, "timeInForce", context),
            price=price,
            average_price=average_price,
            original_quantity=original_quantity,
            executed_quantity=executed_quantity,
            cumulative_quote_quantity=cumulative_quote_quantity,
            reduce_only=_boolean(payload, "reduceOnly", context),
            close_position=_boolean(payload, "closePosition", context),
            update_time_ms=_timestamp_ms(payload, "updateTime", context),
            data_time=_data_time(data_time),
        )


@dataclass(frozen=True)
class BinancePerpetualTrade:
    trade_id: str
    exchange_order_id: str
    symbol: str
    trading_pair: str
    side: TradeType
    position_side: str
    price: Decimal
    quantity: Decimal
    quote_quantity: Decimal
    commission: Decimal
    commission_asset: str
    realized_pnl: Decimal
    timestamp_ms: int
    is_buyer: bool
    is_maker: bool
    data_time: float

    @classmethod
    def from_payload(
            cls,
            payload: Mapping[str, Any],
            expected_symbol: str,
            trading_pair: str,
            data_time: float,
            expected_exchange_order_id: Optional[str] = None,
    ) -> "BinancePerpetualTrade":
        context = "account trade"
        payload = _mapping(payload, context)
        symbol = _string(payload, "symbol", context)
        if symbol != expected_symbol:
            raise BinancePerpetualOrderDataError("account trade symbol does not match request")
        exchange_order_id = _numeric_identifier(payload, "orderId", context)
        if expected_exchange_order_id is not None and exchange_order_id != expected_exchange_order_id:
            raise BinancePerpetualOrderDataError("account trade orderId does not match request")

        raw_side = _string(payload, "side", context)
        if raw_side == "BUY":
            side = TradeType.BUY
        elif raw_side == "SELL":
            side = TradeType.SELL
        else:
            raise BinancePerpetualOrderDataError("account trade.side is unsupported")

        price = _decimal(payload, "price", context)
        quantity = _decimal(payload, "qty", context)
        quote_quantity = _decimal(payload, "quoteQty", context)
        commission = _decimal(payload, "commission", context)
        realized_pnl = _decimal(payload, "realizedPnl", context)
        if price <= 0:
            raise BinancePerpetualOrderDataError("account trade.price must be positive")
        if quantity <= 0:
            raise BinancePerpetualOrderDataError("account trade.qty must be positive")
        if quote_quantity <= 0:
            raise BinancePerpetualOrderDataError("account trade.quoteQty must be positive")

        is_buyer = _boolean(payload, "buyer", context)
        if is_buyer != (side is TradeType.BUY):
            raise BinancePerpetualOrderDataError("account trade buyer flag conflicts with side")

        return cls(
            trade_id=_numeric_identifier(payload, "id", context),
            exchange_order_id=exchange_order_id,
            symbol=symbol,
            trading_pair=trading_pair,
            side=side,
            position_side=_position_side(payload, context),
            price=price,
            quantity=quantity,
            quote_quantity=quote_quantity,
            commission=commission,
            commission_asset=_string(payload, "commissionAsset", context),
            realized_pnl=realized_pnl,
            timestamp_ms=_timestamp_ms(payload, "time", context),
            is_buyer=is_buyer,
            is_maker=_boolean(payload, "maker", context),
            data_time=_data_time(data_time),
        )
