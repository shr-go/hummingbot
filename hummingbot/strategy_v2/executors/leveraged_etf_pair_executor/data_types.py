from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, Literal, Mapping, Optional, Tuple

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
    StrictInt,
    WithJsonSchema,
    model_validator,
)

from hummingbot.strategy_v2.executors.data_types import ExecutorConfigBase

_CANONICAL_DECIMAL_PATTERN = re.compile(r"^(?:0|-?(?:[1-9][0-9]*(?:\.[0-9]*[1-9])?|0\.[0-9]*[1-9]))$")
_CANONICAL_UTC_PATTERN = re.compile(
    r"^[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])" r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9]\.[0-9]{6}Z$"
)
_SECRET_FIELD_NAMES = frozenset(
    {
        "apikey",
        "credential",
        "credentials",
        "passphrase",
        "password",
        "pem",
        "pkcs8",
        "privatekey",
        "secret",
        "secretkey",
    }
)


def _canonical_decimal_text(value: Decimal) -> str:
    if not value.is_finite():
        raise ValueError("decimal must be finite")
    if value == 0:
        return "0"
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _validate_canonical_decimal(value: Any) -> Decimal:
    if isinstance(value, str):
        if _CANONICAL_DECIMAL_PATTERN.fullmatch(value) is None:
            raise ValueError("decimal string is not canonical base-10")
        parsed = Decimal(value)
    elif isinstance(value, Decimal):
        parsed = value
    else:
        raise ValueError("decimal must be supplied as a canonical string or Decimal")
    if not parsed.is_finite():
        raise ValueError("decimal must be finite")
    return Decimal("0") if parsed == 0 else parsed


def _validate_non_negative(value: Decimal) -> Decimal:
    if value < 0:
        raise ValueError("decimal must be non-negative")
    return value


def _validate_positive(value: Decimal) -> Decimal:
    if value <= 0:
        raise ValueError("decimal must be positive")
    return value


CanonicalDecimal = Annotated[
    Decimal,
    BeforeValidator(_validate_canonical_decimal),
    PlainSerializer(_canonical_decimal_text, return_type=str),
]
CanonicalNonNegativeDecimal = Annotated[CanonicalDecimal, AfterValidator(_validate_non_negative)]
CanonicalPositiveDecimal = Annotated[CanonicalDecimal, AfterValidator(_validate_positive)]


def _validate_schema_version_v1(value: int) -> int:
    if value != 1:
        raise ValueError("schema_version must be exactly 1")
    return value


SchemaVersionV1 = Annotated[
    StrictInt,
    AfterValidator(_validate_schema_version_v1),
    WithJsonSchema({"type": "integer", "const": 1}),
]
StrictNonNegativeInt = Annotated[StrictInt, Field(ge=0)]
StrictPositiveInt = Annotated[StrictInt, Field(ge=1)]


def _validate_canonical_utc(value: Any) -> datetime:
    if isinstance(value, str):
        if _CANONICAL_UTC_PATTERN.fullmatch(value) is None:
            raise ValueError("UTC instant must use six fractional digits and a Z suffix")
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
        except ValueError as exc:
            raise ValueError("UTC instant is not a valid calendar timestamp") from exc
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("datetime must be timezone-aware UTC")
        return value.astimezone(timezone.utc)
    raise ValueError("UTC instant must be supplied as a canonical string or datetime")


def _canonical_utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


CanonicalUtcInstant = Annotated[
    datetime,
    BeforeValidator(_validate_canonical_utc),
    PlainSerializer(_canonical_utc_text, return_type=str),
]
Sha256Hex = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
StableIdentifier = Annotated[
    str,
    Field(min_length=1, max_length=255, pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$"),
]
TradingPair = Annotated[str, Field(pattern=r"^[A-Z0-9]+-[A-Z0-9]+$")]
CloseReason = Annotated[str, Field(min_length=1, max_length=255)]


def _normalized_field_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def _reject_secret_bearing_fields(value: Any) -> Any:
    pending = [value]
    visited_containers = set()
    while pending:
        candidate = pending.pop()
        if isinstance(candidate, Mapping):
            if id(candidate) in visited_containers:
                continue
            visited_containers.add(id(candidate))
            for key, nested_value in candidate.items():
                normalized_key = _normalized_field_name(key)
                if any(normalized_key.endswith(secret_name) for secret_name in _SECRET_FIELD_NAMES):
                    raise ValueError(f"secret-bearing field is not permitted in executor wire data: {key}")
                pending.append(nested_value)
        elif isinstance(candidate, (list, tuple)):
            if id(candidate) in visited_containers:
                continue
            visited_containers.add(id(candidate))
            pending.extend(candidate)

    return value


class CanonicalWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def reject_secret_bearing_fields(cls, value: Any) -> Any:
        return _reject_secret_bearing_fields(value)

    def canonical_json(self) -> str:
        return json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()


class LeveragedEtfPairOperation(str, Enum):
    OPEN = "OPEN"
    ADD = "ADD"
    REDUCE = "REDUCE"
    CLOSE = "CLOSE"
    EMERGENCY_FLATTEN = "EMERGENCY_FLATTEN"


class LeveragedEtfPairDirection(str, Enum):
    SHORT_ETF_LONG_STOCK = "SHORT_ETF_LONG_STOCK"
    LONG_ETF_SHORT_STOCK = "LONG_ETF_SHORT_STOCK"


class LeveragedEtfPairState(str, Enum):
    CREATED = "CREATED"
    PREFLIGHT = "PREFLIGHT"
    MAKER_SUBMITTING = "MAKER_SUBMITTING"
    MAKER_WORKING = "MAKER_WORKING"
    MAKER_CANCEL_PENDING = "MAKER_CANCEL_PENDING"
    STOCK_HEDGE_PENDING = "STOCK_HEDGE_PENDING"
    ETF_ROLLBACK_PENDING = "ETF_ROLLBACK_PENDING"
    RECONCILING = "RECONCILING"
    COMPLETED = "COMPLETED"
    ABORTED_NO_FILL = "ABORTED_NO_FILL"
    FAILED_SAFE = "FAILED_SAFE"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


class OrderReferenceV1(CanonicalWireModel):
    sequence: StrictNonNegativeInt
    client_order_id: StableIdentifier
    exchange_order_id: Optional[StableIdentifier]


class LeverageReservationV1(CanonicalWireModel):
    etf_quantity: CanonicalNonNegativeDecimal
    stock_quantity: CanonicalNonNegativeDecimal
    etf_leverage: StrictPositiveInt
    stock_leverage: StrictPositiveInt
    etf_notional_cap: CanonicalPositiveDecimal
    stock_notional_cap: CanonicalPositiveDecimal


def _validate_distinct_leg_roles(
    etf_trading_pair: str,
    stock_trading_pair: str,
) -> None:
    if etf_trading_pair == stock_trading_pair:
        raise ValueError("ETF and stock must have distinct connector/trading-pair roles")


def _validate_order_references(references: Tuple[OrderReferenceV1, ...], field_name: str) -> None:
    sequences = tuple(reference.sequence for reference in references)
    if any(current >= following for current, following in zip(sequences, sequences[1:])):
        raise ValueError(f"{field_name} sequence must be strictly increasing")
    client_order_ids = tuple(reference.client_order_id for reference in references)
    if len(set(client_order_ids)) != len(client_order_ids):
        raise ValueError(f"{field_name} must use a unique client_order_id per entry")


class LeveragedEtfPairExecutorConfig(ExecutorConfigBase):
    """Immutable native Executor config containing the static v1 snapshot fields."""

    type: Literal["leveraged_etf_pair_executor"] = "leveraged_etf_pair_executor"
    id: StableIdentifier
    timestamp: float = Field(allow_inf_nan=False)
    controller_id: StableIdentifier
    schema_version: SchemaVersionV1 = 1
    pair_id: StableIdentifier
    nav_cycle_id: StableIdentifier
    operation: LeveragedEtfPairOperation
    direction: LeveragedEtfPairDirection
    etf_connector_name: StableIdentifier
    etf_trading_pair: TradingPair
    stock_connector_name: StableIdentifier
    stock_trading_pair: TradingPair
    s0: CanonicalPositiveDecimal
    l0: CanonicalPositiveDecimal
    h: CanonicalPositiveDecimal
    created_raw_bp: CanonicalDecimal
    created_net_bp: CanonicalDecimal
    target_gross_notional: CanonicalNonNegativeDecimal
    etf_target_quantity: CanonicalNonNegativeDecimal
    stock_target_quantity: CanonicalNonNegativeDecimal
    leverage_reservation: LeverageReservationV1
    config_hash: Sha256Hex
    created_at_utc: CanonicalUtcInstant
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def reject_secret_bearing_fields(cls, value: Any) -> Any:
        return _reject_secret_bearing_fields(value)

    @model_validator(mode="after")
    def validate_leg_roles(self) -> LeveragedEtfPairExecutorConfig:
        _validate_distinct_leg_roles(
            self.etf_trading_pair,
            self.stock_trading_pair,
        )
        return self

    @property
    def connector_name(self) -> str:
        """Backward-compatible primary connector; the ETF maker leg is primary."""
        return self.etf_connector_name

    @property
    def trading_pair(self) -> str:
        """Backward-compatible primary pair; the ETF maker leg is primary."""
        return self.etf_trading_pair

    @property
    def connector_names(self) -> Tuple[str, str]:
        return self.etf_connector_name, self.stock_connector_name

    @property
    def trading_pairs(self) -> Tuple[str, str]:
        return self.etf_trading_pair, self.stock_trading_pair


class LeveragedEtfPairExecutorStateV1(CanonicalWireModel):
    """Mutable executor facts represented as an immutable versioned wire value."""

    schema_version: SchemaVersionV1 = 1
    executor_id: StableIdentifier
    state: LeveragedEtfPairState
    etf_submitted_quantity: CanonicalNonNegativeDecimal
    etf_filled_quantity: CanonicalNonNegativeDecimal
    etf_remaining_quantity: CanonicalNonNegativeDecimal
    stock_submitted_quantity: CanonicalNonNegativeDecimal
    stock_filled_quantity: CanonicalNonNegativeDecimal
    stock_remaining_quantity: CanonicalNonNegativeDecimal
    hedge_dust_quantity: CanonicalNonNegativeDecimal
    maker_order_ids: Tuple[OrderReferenceV1, ...]
    stock_order_ids: Tuple[OrderReferenceV1, ...]
    leverage_reservation: LeverageReservationV1
    updated_at_utc: CanonicalUtcInstant
    close_reason: Optional[CloseReason]
    last_journal_sequence: StrictNonNegativeInt

    @model_validator(mode="after")
    def validate_order_references(self) -> LeveragedEtfPairExecutorStateV1:
        _validate_order_references(self.maker_order_ids, "maker_order_ids")
        _validate_order_references(self.stock_order_ids, "stock_order_ids")
        return self


class LeveragedEtfPairExecutorSnapshotV1(CanonicalWireModel):
    """Exact F001 LeveragedEtfPairExecutorSnapshotV1 persistence envelope."""

    schema_version: SchemaVersionV1
    executor_id: StableIdentifier
    controller_id: StableIdentifier
    pair_id: StableIdentifier
    nav_cycle_id: StableIdentifier
    operation: LeveragedEtfPairOperation
    direction: LeveragedEtfPairDirection
    state: LeveragedEtfPairState
    etf_connector_name: StableIdentifier
    etf_trading_pair: TradingPair
    stock_connector_name: StableIdentifier
    stock_trading_pair: TradingPair
    s0: CanonicalPositiveDecimal
    l0: CanonicalPositiveDecimal
    h: CanonicalPositiveDecimal
    created_raw_bp: CanonicalDecimal
    created_net_bp: CanonicalDecimal
    target_gross_notional: CanonicalNonNegativeDecimal
    etf_target_quantity: CanonicalNonNegativeDecimal
    etf_submitted_quantity: CanonicalNonNegativeDecimal
    etf_filled_quantity: CanonicalNonNegativeDecimal
    etf_remaining_quantity: CanonicalNonNegativeDecimal
    stock_target_quantity: CanonicalNonNegativeDecimal
    stock_submitted_quantity: CanonicalNonNegativeDecimal
    stock_filled_quantity: CanonicalNonNegativeDecimal
    stock_remaining_quantity: CanonicalNonNegativeDecimal
    hedge_dust_quantity: CanonicalNonNegativeDecimal
    maker_order_ids: Tuple[OrderReferenceV1, ...]
    stock_order_ids: Tuple[OrderReferenceV1, ...]
    leverage_reservation: LeverageReservationV1
    config_hash: Sha256Hex
    created_at_utc: CanonicalUtcInstant
    updated_at_utc: CanonicalUtcInstant
    close_reason: Optional[CloseReason]
    last_journal_sequence: StrictNonNegativeInt

    @model_validator(mode="after")
    def validate_semantics(self) -> LeveragedEtfPairExecutorSnapshotV1:
        _validate_distinct_leg_roles(
            self.etf_trading_pair,
            self.stock_trading_pair,
        )
        _validate_order_references(self.maker_order_ids, "maker_order_ids")
        _validate_order_references(self.stock_order_ids, "stock_order_ids")
        return self

    @classmethod
    def from_config_and_state(
        cls,
        config: LeveragedEtfPairExecutorConfig,
        state: LeveragedEtfPairExecutorStateV1,
    ) -> LeveragedEtfPairExecutorSnapshotV1:
        if config.schema_version != state.schema_version:
            raise ValueError("config and state schema versions do not match")
        if config.id != state.executor_id:
            raise ValueError("config and state executor IDs do not match")
        return cls(
            schema_version=config.schema_version,
            executor_id=config.id,
            controller_id=config.controller_id,
            pair_id=config.pair_id,
            nav_cycle_id=config.nav_cycle_id,
            operation=config.operation,
            direction=config.direction,
            state=state.state,
            etf_connector_name=config.etf_connector_name,
            etf_trading_pair=config.etf_trading_pair,
            stock_connector_name=config.stock_connector_name,
            stock_trading_pair=config.stock_trading_pair,
            s0=config.s0,
            l0=config.l0,
            h=config.h,
            created_raw_bp=config.created_raw_bp,
            created_net_bp=config.created_net_bp,
            target_gross_notional=config.target_gross_notional,
            etf_target_quantity=config.etf_target_quantity,
            etf_submitted_quantity=state.etf_submitted_quantity,
            etf_filled_quantity=state.etf_filled_quantity,
            etf_remaining_quantity=state.etf_remaining_quantity,
            stock_target_quantity=config.stock_target_quantity,
            stock_submitted_quantity=state.stock_submitted_quantity,
            stock_filled_quantity=state.stock_filled_quantity,
            stock_remaining_quantity=state.stock_remaining_quantity,
            hedge_dust_quantity=state.hedge_dust_quantity,
            maker_order_ids=state.maker_order_ids,
            stock_order_ids=state.stock_order_ids,
            leverage_reservation=state.leverage_reservation,
            config_hash=config.config_hash,
            created_at_utc=config.created_at_utc,
            updated_at_utc=state.updated_at_utc,
            close_reason=state.close_reason,
            last_journal_sequence=state.last_journal_sequence,
        )


class LeveragedEtfPairExecutorReportV1(CanonicalWireModel):
    """Optional performance values; missing values remain valid for historical rows."""

    schema_version: SchemaVersionV1 = 1
    net_pnl_pct: Optional[CanonicalDecimal] = None
    net_pnl_quote: Optional[CanonicalDecimal] = None
    realized_pnl_quote: Optional[CanonicalDecimal] = None
    unrealized_pnl_quote: Optional[CanonicalDecimal] = None
    cum_fees_quote: Optional[CanonicalNonNegativeDecimal] = None
    filled_amount_quote: Optional[CanonicalNonNegativeDecimal] = None


class LeveragedEtfPairExecutorCustomInfoV1(CanonicalWireModel):
    """Typed content stored in Hummingbot ExecutorInfo.custom_info."""

    schema_version: SchemaVersionV1 = 1
    state: LeveragedEtfPairExecutorStateV1
    report: LeveragedEtfPairExecutorReportV1 = Field(default_factory=LeveragedEtfPairExecutorReportV1)
