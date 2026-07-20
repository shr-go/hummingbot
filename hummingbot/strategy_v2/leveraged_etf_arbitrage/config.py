import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from types import MappingProxyType
from typing import Annotated, Literal, Mapping, Self

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)


_DECIMAL_PATTERN = re.compile(r"-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?\Z")
_EXPECTED_YAHOO_BASE_URLS = (
    "https://query2.finance.yahoo.com",
    "https://query1.finance.yahoo.com",
)


def _parse_decimal(value: object) -> Decimal:
    if isinstance(value, Decimal):
        parsed = value
    elif isinstance(value, str) and _DECIMAL_PATTERN.fullmatch(value) is not None:
        parsed = Decimal(value)
    else:
        raise ValueError("trading decimal must be a fixed-point base-10 string or Decimal")
    if not parsed.is_finite():
        raise ValueError("trading decimal must be finite")
    if parsed.is_zero() and parsed.is_signed():
        raise ValueError("negative zero is not permitted")
    return parsed


def _immutable_decimal_mapping(value: Mapping[Decimal, Decimal]) -> Mapping[Decimal, Decimal]:
    return MappingProxyType(dict(sorted(value.items())))


def _secret_safe_validation_error(error: ValidationError) -> ValidationError:
    sanitized_errors = []
    for line_error in error.errors(include_url=False, include_input=False):
        sanitized_error = {
            "type": line_error["type"],
            "loc": line_error["loc"],
        }
        if "ctx" in line_error:
            sanitized_error["ctx"] = line_error["ctx"]
        sanitized_errors.append(sanitized_error)
    return ValidationError.from_exception_data(
        title=error.title,
        line_errors=sanitized_errors,
        hide_input=True,
    )


NonNegativeDecimal = Annotated[Decimal, BeforeValidator(_parse_decimal), Field(ge=Decimal("0"))]
PositiveDecimal = Annotated[Decimal, BeforeValidator(_parse_decimal), Field(gt=Decimal("0"))]
PositiveInt = Annotated[int, Field(gt=0)]


class StrictFrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True, strict=True)

    @model_validator(mode="wrap")
    @classmethod
    def sanitize_validation_errors(cls, value: object, handler):
        try:
            return handler(value)
        except ValidationError as error:
            raise _secret_safe_validation_error(error) from None


class SessionName(str, Enum):
    REGULAR = "regular"
    EXTENDED = "extended"
    WEEKEND_HOLIDAY = "weekend_holiday"


class BinanceConfig(StrictFrozenModel):
    connector_name: Literal["binance_perpetual"]
    api_key: SecretStr
    private_key: SecretStr
    account_mode: Literal["multi_assets"]
    margin_mode: Literal["cross"]
    position_mode: Literal["oneway"]

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, value: object) -> object:
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError("api_key must be a non-empty string")
        return raw_value

    @field_validator("private_key", mode="before")
    @classmethod
    def validate_private_key(cls, value: object) -> object:
        raw_value = value.get_secret_value() if isinstance(value, SecretStr) else value
        if not isinstance(raw_value, str):
            raise ValueError("private_key must contain a parseable PKCS#8 Ed25519 private key")
        stripped_value = raw_value.strip()
        if not (
            stripped_value.startswith("-----BEGIN PRIVATE KEY-----")
            and stripped_value.endswith("-----END PRIVATE KEY-----")
        ):
            raise ValueError("private_key must contain a parseable PKCS#8 Ed25519 private key")
        try:
            key = serialization.load_pem_private_key(stripped_value.encode("ascii"), password=None)
        except (TypeError, ValueError, UnicodeEncodeError) as exception:
            raise ValueError("private_key must contain a parseable PKCS#8 Ed25519 private key") from exception
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("private_key must contain a parseable PKCS#8 Ed25519 private key")
        return raw_value


class StrategyConfig(StrictFrozenModel):
    strategy_id: Literal["equity_leveraged_etf_arbitrage"]
    max_total_notional_ratio: PositiveDecimal
    controller_interval_ms: PositiveInt
    executor_safety_interval_ms: PositiveInt
    entry_confirmations: PositiveInt
    entry_confirmation_interval_ms: PositiveInt
    divergence_cancel_bp: NonNegativeDecimal
    divergence_confirmations: PositiveInt
    market_data_max_age_ms: PositiveInt
    account_data_max_age_ms: PositiveInt
    maker_fee_bp: NonNegativeDecimal
    taker_fee_bp: NonNegativeDecimal
    maker_slippage_bp_per_fill: NonNegativeDecimal
    include_funding_cost: Literal[False]
    hedge_submit_timeout_ms: PositiveInt
    hedge_reconcile_timeout_ms: PositiveInt
    unhedged_response_deadline_ms: PositiveInt
    hedge_phase_deadline_ms: PositiveInt
    hedge_max_attempts: PositiveInt
    hedge_retry_backoff_ms: tuple[PositiveInt, ...]
    rollback_submit_timeout_ms: PositiveInt
    rollback_reconcile_timeout_ms: PositiveInt
    rollback_phase_deadline_ms: PositiveInt
    rollback_max_attempts: PositiveInt
    rollback_retry_backoff_ms: tuple[PositiveInt, ...]
    order_eventual_consistency_grace_ms: PositiveInt

    @field_validator("hedge_retry_backoff_ms", "rollback_retry_backoff_ms", mode="before")
    @classmethod
    def freeze_backoff_values(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_response_deadlines(self) -> Self:
        if self.hedge_phase_deadline_ms + self.rollback_phase_deadline_ms > self.unhedged_response_deadline_ms:
            raise ValueError("hedge and rollback phases exceed the unhedged response deadline")
        if self.hedge_phase_deadline_ms < (
            self.hedge_submit_timeout_ms + self.order_eventual_consistency_grace_ms
        ):
            raise ValueError("hedge phase cannot contain one submit timeout and consistency grace")
        if self.rollback_phase_deadline_ms < (
            self.rollback_submit_timeout_ms + self.order_eventual_consistency_grace_ms
        ):
            raise ValueError("rollback phase cannot contain one submit timeout and consistency grace")
        if len(self.hedge_retry_backoff_ms) < self.hedge_max_attempts - 1:
            raise ValueError("hedge retry backoff does not cover all attempts")
        if len(self.rollback_retry_backoff_ms) < self.rollback_max_attempts - 1:
            raise ValueError("rollback retry backoff does not cover all attempts")
        return self


class NavConfig(StrictFrozenModel):
    timezone: Literal["America/New_York"]
    calendar: Literal["US_EQUITIES"]
    anchor_source: Literal["yahoo_finance_chart_http"]
    new_entry_cutoff_minutes: Annotated[int, Field(ge=30)]
    maker_close_lead_minutes: Literal[30]
    force_market_close_lead_seconds: Literal[60]
    anchor_wait_timeout_seconds: Annotated[int, Field(ge=1, le=600)]
    anchor_min_finalize_delay_seconds: PositiveInt
    anchor_poll_initial_interval_seconds: PositiveInt
    anchor_poll_max_interval_seconds: PositiveInt
    anchor_http_request_timeout_seconds: PositiveInt
    anchor_confirmation_count: Annotated[int, Field(ge=2)]
    anchor_confirmation_interval_seconds: PositiveInt
    anchor_pair_fetch_max_skew_seconds: PositiveInt
    yahoo_chart_range: Literal["5d"]
    yahoo_chart_interval: Literal["1d"]
    yahoo_include_pre_post: Literal[False]
    yahoo_base_urls: tuple[str, ...]
    yahoo_user_agent: Annotated[str, Field(min_length=1)]

    @field_validator("yahoo_base_urls", mode="before")
    @classmethod
    def freeze_base_urls(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_anchor_budget(self) -> Self:
        if self.yahoo_base_urls != _EXPECTED_YAHOO_BASE_URLS:
            raise ValueError("Yahoo base URLs must be the approved query2 then query1 sequence")
        if self.anchor_poll_initial_interval_seconds > self.anchor_poll_max_interval_seconds:
            raise ValueError("initial anchor polling interval exceeds its maximum")
        if self.anchor_http_request_timeout_seconds >= self.anchor_wait_timeout_seconds:
            raise ValueError("one Yahoo request timeout must be below the anchor deadline")
        if self.anchor_pair_fetch_max_skew_seconds > self.anchor_http_request_timeout_seconds:
            raise ValueError("paired response skew exceeds the HTTP request timeout")
        round_budget = (
            len(self.yahoo_base_urls) * self.anchor_http_request_timeout_seconds
            + self.anchor_pair_fetch_max_skew_seconds
        )
        conservative_budget = (
            self.anchor_min_finalize_delay_seconds
            + self.anchor_confirmation_count * round_budget
            + (self.anchor_confirmation_count - 1)
            * max(self.anchor_confirmation_interval_seconds, self.anchor_poll_max_interval_seconds)
        )
        if conservative_budget > self.anchor_wait_timeout_seconds:
            raise ValueError("conservative anchor success budget exceeds the configured deadline")
        return self


class RiskConfig(StrictFrozenModel):
    normal_initial_margin_budget_ratio: Annotated[
        Decimal,
        BeforeValidator(_parse_decimal),
        Field(gt=Decimal("0"), le=Decimal("1")),
    ]
    p99_initial_margin_budget_ratio: Annotated[
        Decimal,
        BeforeValidator(_parse_decimal),
        Field(gt=Decimal("0"), le=Decimal("1")),
    ]
    stress_extra_bp_above_p99: PositiveDecimal
    stress_equity_to_maintenance_margin_multiple: PositiveDecimal
    account_margin_reconciliation_tolerance: Annotated[
        Decimal,
        BeforeValidator(_parse_decimal),
        Field(ge=Decimal("0"), le=Decimal("0.01")),
    ]

    @model_validator(mode="after")
    def validate_margin_budgets(self) -> Self:
        if self.p99_initial_margin_budget_ratio < self.normal_initial_margin_budget_ratio:
            raise ValueError("P99 initial margin budget cannot be below the normal budget")
        return self


class PairExecutionConfig(StrictFrozenModel):
    max_stock_taker_impact_bp: NonNegativeDecimal
    maker_reprice_ticks: PositiveInt
    maker_min_reprice_interval_ms: PositiveInt
    max_maker_order_age_seconds: PositiveInt


class SessionConfig(StrictFrozenModel):
    p99_bp: NonNegativeDecimal
    historical_max_bp: NonNegativeDecimal
    position_tiers: Mapping[NonNegativeDecimal, PositiveDecimal]
    reduce_bp_by_current_target: Mapping[PositiveDecimal, NonNegativeDecimal]
    divergence_cancel_bp: NonNegativeDecimal | None = None
    divergence_confirmations: PositiveInt | None = None
    entry_confirmations: PositiveInt | None = None
    entry_confirmation_interval_ms: PositiveInt | None = None

    @field_validator("position_tiers", "reduce_bp_by_current_target", mode="before")
    @classmethod
    def reject_normalized_decimal_key_collisions(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        normalized_keys: set[Decimal] = set()
        for raw_key in value:
            normalized_key = _parse_decimal(raw_key)
            if normalized_key in normalized_keys:
                raise ValueError("duplicate normalized Decimal key")
            normalized_keys.add(normalized_key)
        return value

    @field_validator("position_tiers", "reduce_bp_by_current_target", mode="after")
    @classmethod
    def freeze_decimal_maps(cls, value: Mapping[Decimal, Decimal]) -> Mapping[Decimal, Decimal]:
        return _immutable_decimal_mapping(value)

    @model_validator(mode="after")
    def validate_tiers(self) -> Self:
        zero = Decimal("0")
        if zero not in self.position_tiers:
            raise ValueError("position tiers must contain the base entry key 0")
        ordered_tiers = tuple(self.position_tiers.items())
        targets = tuple(target for _, target in ordered_tiers)
        if any(current_target < previous_target for previous_target, current_target in zip(targets, targets[1:])):
            raise ValueError("position tier targets must be monotonic nondecreasing")
        target_set = set(targets)
        if set(self.reduce_bp_by_current_target) != target_set:
            raise ValueError("reduce thresholds must map every and only configured target")
        base_target = self.position_tiers[zero]
        if self.reduce_bp_by_current_target[base_target] != zero:
            raise ValueError("the base target reduce threshold must be 0")
        for entry_threshold, target in ordered_tiers:
            reduce_threshold = self.reduce_bp_by_current_target[target]
            if entry_threshold > zero and reduce_threshold >= entry_threshold:
                raise ValueError("higher target reduce thresholds must be below their entry thresholds")
        highest_entry = ordered_tiers[-1][0]
        if self.p99_bp < highest_entry:
            raise ValueError("P99 must not be below the highest entry tier")
        if self.historical_max_bp < self.p99_bp:
            raise ValueError("historical maximum must not be below P99")
        return self


@dataclass(frozen=True, slots=True)
class ResolvedSessionConfig:
    session_name: SessionName
    p99_bp: Decimal
    historical_max_bp: Decimal
    position_tiers: Mapping[Decimal, Decimal]
    reduce_bp_by_current_target: Mapping[Decimal, Decimal]
    divergence_cancel_bp: Decimal
    divergence_confirmations: int
    entry_confirmations: int
    entry_confirmation_interval_ms: int


class PairConfig(StrictFrozenModel):
    id: str
    enabled: bool
    stock_trading_pair: str
    etf_trading_pair: str
    stock_nav_symbol: str
    etf_nav_symbol: str
    etf_daily_multiplier: PositiveDecimal
    execution: PairExecutionConfig
    sessions: Mapping[SessionName, SessionConfig]
    divergence_cancel_bp: NonNegativeDecimal | None = None
    divergence_confirmations: PositiveInt | None = None

    @field_validator("id", "stock_trading_pair", "etf_trading_pair", "stock_nav_symbol", "etf_nav_symbol")
    @classmethod
    def validate_nonempty_identifiers(cls, value: str) -> str:
        if not value:
            raise ValueError("pair identifiers and symbols must not be empty")
        return value

    @field_validator("sessions", mode="before")
    @classmethod
    def parse_session_names(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        parsed: dict[SessionName, object] = {}
        for raw_name, session_config in value.items():
            try:
                name = raw_name if isinstance(raw_name, SessionName) else SessionName(raw_name)
            except (TypeError, ValueError) as exception:
                raise ValueError(f"unsupported session name: {raw_name}") from exception
            parsed[name] = session_config
        return parsed

    @field_validator("sessions", mode="after")
    @classmethod
    def freeze_sessions(cls, value: Mapping[SessionName, SessionConfig]) -> Mapping[SessionName, SessionConfig]:
        return MappingProxyType(dict(value))

    @model_validator(mode="after")
    def validate_regular_session(self) -> Self:
        if SessionName.REGULAR not in self.sessions:
            raise ValueError("the regular session is required")
        return self

    def session(self, session_name: SessionName) -> SessionConfig:
        return self.select_session(session_name)[1]

    def select_session(self, session_name: SessionName) -> tuple[SessionName, SessionConfig]:
        if not isinstance(session_name, SessionName):
            raise TypeError("session name must be a SessionName")
        if session_name in self.sessions:
            return session_name, self.sessions[session_name]
        return SessionName.REGULAR, self.sessions[SessionName.REGULAR]

    def resolve_session(
        self,
        session_name: SessionName,
        strategy: StrategyConfig,
    ) -> ResolvedSessionConfig:
        if not isinstance(strategy, StrategyConfig):
            raise TypeError("strategy must be a StrategyConfig")
        selected_name, selected_session = self.select_session(session_name)
        divergence_cancel_bp = selected_session.divergence_cancel_bp
        if divergence_cancel_bp is None:
            divergence_cancel_bp = self.divergence_cancel_bp
        if divergence_cancel_bp is None:
            divergence_cancel_bp = strategy.divergence_cancel_bp
        divergence_confirmations = selected_session.divergence_confirmations
        if divergence_confirmations is None:
            divergence_confirmations = self.divergence_confirmations
        if divergence_confirmations is None:
            divergence_confirmations = strategy.divergence_confirmations
        entry_confirmations = selected_session.entry_confirmations
        if entry_confirmations is None:
            entry_confirmations = strategy.entry_confirmations
        entry_confirmation_interval_ms = selected_session.entry_confirmation_interval_ms
        if entry_confirmation_interval_ms is None:
            entry_confirmation_interval_ms = strategy.entry_confirmation_interval_ms
        return ResolvedSessionConfig(
            session_name=selected_name,
            p99_bp=selected_session.p99_bp,
            historical_max_bp=selected_session.historical_max_bp,
            position_tiers=selected_session.position_tiers,
            reduce_bp_by_current_target=selected_session.reduce_bp_by_current_target,
            divergence_cancel_bp=divergence_cancel_bp,
            divergence_confirmations=divergence_confirmations,
            entry_confirmations=entry_confirmations,
            entry_confirmation_interval_ms=entry_confirmation_interval_ms,
        )


_PAIR_WHITELIST = {
    "sndk_snxx": ("SNDK-USDT", "SNXX-USDT", "SNDK", "SNXX", Decimal("2")),
    "intc_intw": ("INTC-USDT", "INTW-USDT", "INTC", "INTW", Decimal("2")),
    "mrvl_mvll": ("MRVL-USDT", "MVLL-USDT", "MRVL", "MVLL", Decimal("2")),
    "mu_muu": ("MU-USDT", "MUU-USDT", "MU", "MUU", Decimal("2")),
}


class EquityLeveragedEtfArbitrageConfig(StrictFrozenModel):
    binance: BinanceConfig
    strategy: StrategyConfig
    nav: NavConfig
    risk: RiskConfig
    pairs: Annotated[tuple[PairConfig, ...], Field(min_length=1, max_length=4)]

    @field_validator("pairs", mode="before")
    @classmethod
    def freeze_pairs(cls, value: object) -> object:
        return tuple(value) if isinstance(value, list) else value

    @model_validator(mode="after")
    def validate_pair_set(self) -> Self:
        pair_ids: set[str] = set()
        enabled_symbols: set[str] = set()
        enabled_count = 0
        for pair in self.pairs:
            expected = _PAIR_WHITELIST.get(pair.id)
            actual = (
                pair.stock_trading_pair,
                pair.etf_trading_pair,
                pair.stock_nav_symbol,
                pair.etf_nav_symbol,
                pair.etf_daily_multiplier,
            )
            if expected is None or actual != expected:
                raise ValueError(f"pair {pair.id!r} does not match the exact four-pair whitelist")
            if pair.id in pair_ids:
                raise ValueError(f"duplicate pair id: {pair.id}")
            pair_ids.add(pair.id)
            if pair.enabled:
                enabled_count += 1
                symbols = {
                    pair.stock_trading_pair,
                    pair.etf_trading_pair,
                    pair.stock_nav_symbol,
                    pair.etf_nav_symbol,
                }
                if enabled_symbols.intersection(symbols):
                    raise ValueError("an enabled trading or NAV symbol is used by more than one pair")
                enabled_symbols.update(symbols)
                for session in pair.sessions.values():
                    if session.p99_bp <= 0 or session.historical_max_bp <= 0:
                        raise ValueError("enabled pairs require positive P99 and historical maximum values")
        if enabled_count == 0:
            raise ValueError("at least one whitelisted pair must be enabled")
        return self

    @model_validator(mode="wrap")
    @classmethod
    def sanitize_root_validation_errors(cls, value: object, handler):
        try:
            return handler(value)
        except ValidationError as error:
            raise _secret_safe_validation_error(error) from None

    @property
    def enabled_pairs(self) -> tuple[PairConfig, ...]:
        return tuple(pair for pair in self.pairs if pair.enabled)

    def resolve_session(self, pair_id: str, session_name: SessionName) -> ResolvedSessionConfig:
        if not isinstance(pair_id, str):
            raise TypeError("pair id must be a str")
        try:
            pair = next(pair for pair in self.pairs if pair.id == pair_id)
        except StopIteration as exception:
            raise ValueError(f"unknown configured pair id: {pair_id}") from exception
        return pair.resolve_session(session_name, self.strategy)
