import copy
import json
import tomllib
from collections.abc import Mapping
from dataclasses import FrozenInstanceError
from decimal import Decimal
from enum import Enum
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import (
    EquityLeveragedEtfArbitrageConfig,
    PairConfig,
    SessionConfig,
    SessionName,
)


def _example_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config" / "equity_leveraged_etf_arbitrage.example.toml"
        if candidate.is_file():
            return candidate
    raise AssertionError("committed example TOML not found from test path")


@pytest.fixture
def example_data() -> dict:
    with _example_path().open("rb") as config_file:
        return tomllib.load(config_file)


def _set_path(data: dict, path: tuple[object, ...], value: object) -> None:
    current = data
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = value


def _raw_mapping_value(raw: Mapping, key: object) -> object:
    raw_key = key.value if isinstance(key, Enum) else key
    return raw[raw_key]


def _assert_model_fields_come_from_input(model: BaseModel, raw: Mapping) -> None:
    model_fields = type(model).model_fields
    assert set(raw) <= set(model_fields)
    for missing_field in set(model_fields) - set(raw):
        assert not model_fields[missing_field].is_required()
        assert model_fields[missing_field].default is None
    for field_name in raw:
        value = getattr(model, field_name)
        raw_value = raw[field_name]
        if isinstance(value, BaseModel):
            _assert_model_fields_come_from_input(value, raw_value)
        elif isinstance(value, tuple) and value and all(isinstance(item, BaseModel) for item in value):
            assert len(value) == len(raw_value)
            for nested_model, nested_raw in zip(value, raw_value):
                _assert_model_fields_come_from_input(nested_model, nested_raw)
        elif isinstance(value, Mapping) and value and all(isinstance(item, BaseModel) for item in value.values()):
            assert len(value) == len(raw_value)
            for key, nested_model in value.items():
                _assert_model_fields_come_from_input(nested_model, _raw_mapping_value(raw_value, key))


def _assert_validation_error_is_secret_safe(error: ValidationError, forbidden_fragments: tuple[str, ...]) -> None:
    structured_errors = error.errors(include_url=False)
    renderings = (
        str(error),
        repr(error),
        repr(structured_errors),
        json.dumps(structured_errors, default=repr),
        error.json(),
    )
    if any(fragment in rendering for fragment in forbidden_fragments for rendering in renderings):
        pytest.fail("validation error exposed a forbidden secret fragment")


def test_committed_example_parses_with_tomllib_and_full_pydantic_validation(example_data: dict):
    config = EquityLeveragedEtfArbitrageConfig.model_validate(example_data)

    assert len(config.pairs) == 4
    assert config.strategy.max_total_notional_ratio == Decimal("32")
    assert config.strategy.maker_fee_bp == Decimal("0")
    assert config.risk.normal_initial_margin_budget_ratio == Decimal("0.80")
    assert config.pairs[0].etf_daily_multiplier == Decimal("2")
    assert config.pairs[0].divergence_cancel_bp == Decimal("3")
    assert config.pairs[0].divergence_confirmations == 3
    regular = config.pairs[0].sessions[SessionName.REGULAR]
    assert regular.divergence_cancel_bp == Decimal("3")
    assert regular.divergence_confirmations == 3
    assert regular.entry_confirmations == 3
    assert regular.entry_confirmation_interval_ms == 1000
    assert config.pairs[0].sessions[SessionName.REGULAR].position_tiers[Decimal("140.99")] == Decimal("32")
    assert set(PairConfig.model_fields) == set().union(*(pair.keys() for pair in example_data["pairs"]))
    assert set(SessionConfig.model_fields) == set().union(
        *(session.keys() for pair in example_data["pairs"] for session in pair["sessions"].values())
    )
    _assert_model_fields_come_from_input(config, example_data)


def test_selected_session_overrides_pair_and_strategy_values(example_data: dict):
    strategy = example_data["strategy"]
    pair = example_data["pairs"][0]
    regular = pair["sessions"]["regular"]
    strategy["divergence_cancel_bp"] = "1"
    strategy["divergence_confirmations"] = 2
    strategy["entry_confirmations"] = 2
    strategy["entry_confirmation_interval_ms"] = 500
    pair["divergence_cancel_bp"] = "5"
    pair["divergence_confirmations"] = 4
    regular["divergence_cancel_bp"] = "7"
    regular["divergence_confirmations"] = 6
    regular["entry_confirmations"] = 8
    regular["entry_confirmation_interval_ms"] = 1500

    config = EquityLeveragedEtfArbitrageConfig.model_validate(example_data)
    resolved = config.resolve_session("sndk_snxx", SessionName.REGULAR)

    assert resolved.session_name is SessionName.REGULAR
    assert resolved.divergence_cancel_bp == Decimal("7")
    assert resolved.divergence_confirmations == 6
    assert resolved.entry_confirmations == 8
    assert resolved.entry_confirmation_interval_ms == 1500


def test_pair_override_wins_when_selected_session_has_no_divergence_override(example_data: dict):
    pair = example_data["pairs"][0]
    extended = copy.deepcopy(pair["sessions"]["regular"])
    for field_name in (
        "divergence_cancel_bp",
        "divergence_confirmations",
        "entry_confirmations",
        "entry_confirmation_interval_ms",
    ):
        extended.pop(field_name)
    pair["sessions"]["extended"] = extended
    pair["divergence_cancel_bp"] = "9"
    pair["divergence_confirmations"] = 7
    example_data["strategy"]["entry_confirmations"] = 5
    example_data["strategy"]["entry_confirmation_interval_ms"] = 750

    config = EquityLeveragedEtfArbitrageConfig.model_validate(example_data)
    resolved = config.resolve_session("sndk_snxx", SessionName.EXTENDED)

    assert resolved.session_name is SessionName.EXTENDED
    assert resolved.divergence_cancel_bp == Decimal("9")
    assert resolved.divergence_confirmations == 7
    assert resolved.entry_confirmations == 5
    assert resolved.entry_confirmation_interval_ms == 750


def test_strategy_globals_win_when_pair_and_selected_session_omit_overrides(example_data: dict):
    config = EquityLeveragedEtfArbitrageConfig.model_validate(example_data)
    resolved = config.resolve_session("intc_intw", SessionName.REGULAR)

    assert resolved.session_name is SessionName.REGULAR
    assert resolved.divergence_cancel_bp == config.strategy.divergence_cancel_bp
    assert resolved.divergence_confirmations == config.strategy.divergence_confirmations
    assert resolved.entry_confirmations == config.strategy.entry_confirmations
    assert resolved.entry_confirmation_interval_ms == config.strategy.entry_confirmation_interval_ms


def test_missing_optional_session_selects_complete_regular_before_precedence(example_data: dict):
    pair_data = example_data["pairs"][0]
    pair_data["divergence_cancel_bp"] = "11"
    pair_data["divergence_confirmations"] = 9
    config = EquityLeveragedEtfArbitrageConfig.model_validate(example_data)
    pair = config.pairs[0]
    regular = pair.sessions[SessionName.REGULAR]

    resolved = config.resolve_session(pair.id, SessionName.WEEKEND_HOLIDAY)

    assert resolved.session_name is SessionName.REGULAR
    assert resolved.p99_bp == regular.p99_bp
    assert resolved.historical_max_bp == regular.historical_max_bp
    assert resolved.position_tiers is regular.position_tiers
    assert resolved.reduce_bp_by_current_target is regular.reduce_bp_by_current_target
    assert resolved.divergence_cancel_bp == regular.divergence_cancel_bp
    assert resolved.divergence_confirmations == regular.divergence_confirmations
    assert resolved.entry_confirmations == regular.entry_confirmations
    assert resolved.entry_confirmation_interval_ms == regular.entry_confirmation_interval_ms
    with pytest.raises(FrozenInstanceError):
        resolved.divergence_confirmations = 99


def test_zero_divergence_bp_override_is_valid(example_data: dict):
    example_data["pairs"][0]["divergence_cancel_bp"] = "0"
    example_data["pairs"][0]["sessions"]["regular"]["divergence_cancel_bp"] = "0"

    config = EquityLeveragedEtfArbitrageConfig.model_validate(example_data)

    assert config.resolve_session("sndk_snxx", SessionName.REGULAR).divergence_cancel_bp == Decimal("0")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("pairs", 0, "divergence_cancel_bp"), "-0.01"),
        (("pairs", 0, "divergence_cancel_bp"), "NaN"),
        (("pairs", 0, "divergence_cancel_bp"), 1.0),
        (("pairs", 0, "divergence_confirmations"), 0),
        (("pairs", 0, "divergence_confirmations"), True),
        (("pairs", 0, "divergence_confirmations"), 3.0),
        (("pairs", 0, "sessions", "regular", "divergence_cancel_bp"), "-1"),
        (("pairs", 0, "sessions", "regular", "divergence_cancel_bp"), "Infinity"),
        (("pairs", 0, "sessions", "regular", "divergence_cancel_bp"), 1.0),
        (("pairs", 0, "sessions", "regular", "divergence_confirmations"), 0),
        (("pairs", 0, "sessions", "regular", "divergence_confirmations"), 3.0),
        (("pairs", 0, "sessions", "regular", "entry_confirmations"), 0),
        (("pairs", 0, "sessions", "regular", "entry_confirmations"), True),
        (("pairs", 0, "sessions", "regular", "entry_confirmations"), 3.0),
        (("pairs", 0, "sessions", "regular", "entry_confirmation_interval_ms"), 0),
        (("pairs", 0, "sessions", "regular", "entry_confirmation_interval_ms"), 1000.0),
    ],
)
def test_pair_and_session_override_values_are_strict(
    example_data: dict,
    path: tuple[object, ...],
    value: object,
):
    _set_path(example_data, path, value)

    with pytest.raises(ValidationError) as error_info:
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)
    matching_errors = [error for error in error_info.value.errors() if tuple(error["loc"]) == path]
    assert matching_errors
    assert all(error["type"] != "extra_forbidden" for error in matching_errors)


def test_optional_sessions_use_complete_regular_fallback(example_data: dict):
    pair_data = example_data["pairs"][0]
    pair_data["sessions"]["extended"] = copy.deepcopy(pair_data["sessions"]["regular"])
    pair_data["sessions"]["extended"]["p99_bp"] = "150"
    pair_data["sessions"]["extended"]["historical_max_bp"] = "175"

    pair = EquityLeveragedEtfArbitrageConfig.model_validate(example_data).pairs[0]

    assert pair.session(SessionName.EXTENDED).p99_bp == Decimal("150")
    assert pair.session(SessionName.WEEKEND_HOLIDAY) is pair.session(SessionName.REGULAR)


def test_models_and_nested_collections_are_immutable(example_data: dict):
    config = EquityLeveragedEtfArbitrageConfig.model_validate(example_data)

    with pytest.raises(ValidationError, match="frozen"):
        config.strategy.max_total_notional_ratio = Decimal("1")
    with pytest.raises(TypeError):
        config.pairs[0].sessions[SessionName.REGULAR].position_tiers[Decimal("0")] = Decimal("9")
    assert isinstance(config.pairs, tuple)
    assert isinstance(config.strategy.hedge_retry_backoff_ms, tuple)
    assert isinstance(config.nav.yahoo_base_urls, tuple)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("strategy", "drawdown_breaker_bp"), "10"),
        (("risk", "max_drawdown_ratio"), "0.10"),
        (("pairs", 0, "sessions", "regular", "drawdown_enabled"), True),
        (("strategy", "unexpected"), 1),
    ],
)
def test_unknown_and_drawdown_fields_are_forbidden(example_data: dict, path: tuple[object, ...], value: object):
    _set_path(example_data, path, value)

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("strategy", "max_total_notional_ratio"), 32.0),
        (("strategy", "maker_fee_bp"), 0),
        (("risk", "normal_initial_margin_budget_ratio"), 0.8),
        (("pairs", 0, "etf_daily_multiplier"), 2),
        (("pairs", 0, "execution", "max_stock_taker_impact_bp"), 10.0),
        (("pairs", 0, "sessions", "regular", "position_tiers", "0"), 1.0),
        (("strategy", "controller_interval_ms"), 1000.0),
        (("pairs", 0, "enabled"), 1),
    ],
)
def test_trading_decimals_and_scalar_types_reject_binary_float_coercion(
    example_data: dict,
    path: tuple[object, ...],
    value: object,
):
    _set_path(example_data, path, value)

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


@pytest.mark.parametrize("bad_decimal", ["NaN", "Infinity", "-Infinity", "1e-2", "+1", " 1", "1 ", "-0", ""])
def test_malformed_decimal_strings_are_rejected(example_data: dict, bad_decimal: str):
    example_data["strategy"]["max_total_notional_ratio"] = bad_decimal

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


def test_exact_pair_whitelist_allows_any_nonempty_subset(example_data: dict):
    only_sndk = copy.deepcopy(example_data)
    only_sndk["pairs"] = [only_sndk["pairs"][0]]
    assert len(EquityLeveragedEtfArbitrageConfig.model_validate(only_sndk).pairs) == 1

    example_data["pairs"][1]["enabled"] = True
    example_data["pairs"][1]["sessions"]["regular"]["p99_bp"] = "10"
    example_data["pairs"][1]["sessions"]["regular"]["historical_max_bp"] = "20"
    enabled_ids = {pair.id for pair in EquityLeveragedEtfArbitrageConfig.model_validate(example_data).enabled_pairs}
    assert enabled_ids == {"sndk_snxx", "intc_intw"}


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_stock",
        "reversed_legs",
        "wrong_nav_symbol",
        "wrong_multiplier",
        "duplicate_pair",
        "duplicate_symbol",
        "no_pairs",
        "too_many_pairs",
        "no_enabled_pairs",
        "enabled_placeholder",
    ],
)
def test_invalid_pair_sets_are_rejected(example_data: dict, mutation: str):
    pairs = example_data["pairs"]
    if mutation == "unknown_stock":
        pairs[0]["stock_trading_pair"] = "NVDA-USDT"
    elif mutation == "reversed_legs":
        pairs[0]["stock_trading_pair"], pairs[0]["etf_trading_pair"] = (
            pairs[0]["etf_trading_pair"],
            pairs[0]["stock_trading_pair"],
        )
    elif mutation == "wrong_nav_symbol":
        pairs[0]["stock_nav_symbol"] = "SNXX"
    elif mutation == "wrong_multiplier":
        pairs[0]["etf_daily_multiplier"] = "3"
    elif mutation == "duplicate_pair":
        pairs[1] = copy.deepcopy(pairs[0])
    elif mutation == "duplicate_symbol":
        pairs[1]["enabled"] = True
        pairs[1]["stock_trading_pair"] = pairs[0]["stock_trading_pair"]
    elif mutation == "no_pairs":
        example_data["pairs"] = []
    elif mutation == "too_many_pairs":
        pairs.append(copy.deepcopy(pairs[0]))
    elif mutation == "no_enabled_pairs":
        for pair in pairs:
            pair["enabled"] = False
    elif mutation == "enabled_placeholder":
        pairs[1]["enabled"] = True

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("strategy", "max_total_notional_ratio"), "0"),
        (("strategy", "maker_fee_bp"), "-0.01"),
        (("strategy", "taker_fee_bp"), "-1"),
        (("strategy", "maker_slippage_bp_per_fill"), "-1"),
        (("strategy", "divergence_cancel_bp"), "-1"),
        (("risk", "normal_initial_margin_budget_ratio"), "0"),
        (("risk", "normal_initial_margin_budget_ratio"), "1.01"),
        (("risk", "p99_initial_margin_budget_ratio"), "1.01"),
        (("risk", "stress_extra_bp_above_p99"), "0"),
        (("risk", "stress_equity_to_maintenance_margin_multiple"), "0"),
        (("risk", "account_margin_reconciliation_tolerance"), "-0.001"),
        (("risk", "account_margin_reconciliation_tolerance"), "0.0101"),
        (("pairs", 0, "execution", "max_stock_taker_impact_bp"), "-1"),
    ],
)
def test_decimal_bounds_are_enforced(example_data: dict, path: tuple[object, ...], value: str):
    _set_path(example_data, path, value)

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


def test_p99_margin_budget_must_not_be_below_normal_budget(example_data: dict):
    example_data["risk"]["normal_initial_margin_budget_ratio"] = "0.90"
    example_data["risk"]["p99_initial_margin_budget_ratio"] = "0.85"

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("strategy", "controller_interval_ms"), 0),
        (("strategy", "executor_safety_interval_ms"), 0),
        (("strategy", "entry_confirmations"), 0),
        (("strategy", "entry_confirmation_interval_ms"), 0),
        (("strategy", "divergence_confirmations"), 0),
        (("pairs", 0, "execution", "maker_reprice_ticks"), 0),
        (("pairs", 0, "execution", "maker_min_reprice_interval_ms"), 0),
        (("pairs", 0, "execution", "max_maker_order_age_seconds"), 0),
    ],
)
def test_positive_integer_controls_are_enforced(example_data: dict, path: tuple[object, ...], value: int):
    _set_path(example_data, path, value)

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


@pytest.mark.parametrize(
    "mutation",
    [
        "phase_sum",
        "hedge_phase_capacity",
        "rollback_phase_capacity",
        "short_hedge_backoff",
        "short_rollback_backoff",
        "zero_timeout",
        "zero_attempts",
        "zero_backoff",
    ],
)
def test_hedge_and_rollback_deadline_contract_is_validated(example_data: dict, mutation: str):
    strategy = example_data["strategy"]
    if mutation == "phase_sum":
        strategy["unhedged_response_deadline_ms"] = 19999
    elif mutation == "hedge_phase_capacity":
        strategy["hedge_phase_deadline_ms"] = 2999
    elif mutation == "rollback_phase_capacity":
        strategy["rollback_phase_deadline_ms"] = 2999
    elif mutation == "short_hedge_backoff":
        strategy["hedge_retry_backoff_ms"] = [100]
    elif mutation == "short_rollback_backoff":
        strategy["rollback_retry_backoff_ms"] = [100]
    elif mutation == "zero_timeout":
        strategy["hedge_reconcile_timeout_ms"] = 0
    elif mutation == "zero_attempts":
        strategy["rollback_max_attempts"] = 0
    elif mutation == "zero_backoff":
        strategy["rollback_retry_backoff_ms"] = [100, 0]

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


@pytest.mark.parametrize(
    "mutation",
    [
        "timeout_too_low",
        "timeout_too_high",
        "http_not_below_deadline",
        "confirmation_count",
        "poll_order",
        "skew",
        "conservative_budget",
        "base_url_order",
        "force_after_maker_stage",
        "zero_confirmation_interval",
    ],
)
def test_nav_polling_and_close_deadlines_are_validated(example_data: dict, mutation: str):
    nav = example_data["nav"]
    if mutation == "timeout_too_low":
        nav["anchor_wait_timeout_seconds"] = 0
    elif mutation == "timeout_too_high":
        nav["anchor_wait_timeout_seconds"] = 601
    elif mutation == "http_not_below_deadline":
        nav["anchor_wait_timeout_seconds"] = 10
    elif mutation == "confirmation_count":
        nav["anchor_confirmation_count"] = 1
    elif mutation == "poll_order":
        nav["anchor_poll_initial_interval_seconds"] = 16
    elif mutation == "skew":
        nav["anchor_pair_fetch_max_skew_seconds"] = 11
    elif mutation == "conservative_budget":
        nav["anchor_wait_timeout_seconds"] = 124
    elif mutation == "base_url_order":
        nav["yahoo_base_urls"] = list(reversed(nav["yahoo_base_urls"]))
    elif mutation == "force_after_maker_stage":
        nav["force_market_close_lead_seconds"] = 1801
    elif mutation == "zero_confirmation_interval":
        nav["anchor_confirmation_interval_seconds"] = 0

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("maker_close_lead_minutes", 1),
        ("maker_close_lead_minutes", 29),
        ("maker_close_lead_minutes", 31),
        ("force_market_close_lead_seconds", 59),
        ("force_market_close_lead_seconds", 61),
        ("new_entry_cutoff_minutes", 1),
        ("new_entry_cutoff_minutes", 29),
    ],
)
def test_nav_close_boundaries_cannot_be_weakened(
    example_data: dict,
    field_name: str,
    invalid_value: int,
):
    example_data["nav"][field_name] = invalid_value

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


def test_new_entry_cutoff_may_be_earlier_than_close_thirty(example_data: dict):
    example_data["nav"]["new_entry_cutoff_minutes"] = 45

    config = EquityLeveragedEtfArbitrageConfig.model_validate(example_data)

    assert config.nav.new_entry_cutoff_minutes == 45
    assert config.nav.maker_close_lead_minutes == 30
    assert config.nav.force_market_close_lead_seconds == 60


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("binance", "connector_name"), "binance"),
        (("binance", "account_mode"), "single_asset"),
        (("binance", "margin_mode"), "isolated"),
        (("binance", "position_mode"), "hedge"),
        (("strategy", "strategy_id"), "other"),
        (("strategy", "include_funding_cost"), True),
        (("nav", "timezone"), "UTC"),
        (("nav", "calendar"), "XNYS"),
        (("nav", "anchor_source"), "other"),
        (("nav", "yahoo_chart_range"), "1d"),
        (("nav", "yahoo_chart_interval"), "5m"),
        (("nav", "yahoo_include_pre_post"), True),
    ],
)
def test_fixed_runtime_choices_are_strict(example_data: dict, path: tuple[object, ...], value: object):
    _set_path(example_data, path, value)

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


def test_private_key_must_be_parseable_pkcs8_ed25519(example_data: dict):
    example_data["binance"]["private_key"] = """-----BEGIN PRIVATE KEY-----
not-a-key
-----END PRIVATE KEY-----"""

    with pytest.raises(ValidationError, match="Ed25519"):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


def test_malformed_private_key_validation_error_hides_all_secret_inputs(example_data: dict):
    api_key_sentinel = "api-key-sentinel-83f3d9"
    private_key_sentinel = "private-key-sentinel-7ac441"
    example_data["binance"]["api_key"] = api_key_sentinel
    example_data["binance"]["private_key"] = (
        "-----BEGIN PRIVATE KEY-----\n" + private_key_sentinel + "\n-----END PRIVATE KEY-----"
    )

    with pytest.raises(ValidationError) as error_info:
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)

    _assert_validation_error_is_secret_safe(
        error_info.value,
        (api_key_sentinel, private_key_sentinel, "-----BEGIN PRIVATE KEY-----"),
    )


def test_unrelated_root_validation_error_hides_api_key_and_valid_pem(example_data: dict):
    api_key_sentinel = "api-key-sentinel-e2c59b"
    public_test_key_fragment = "MC4CAQAwBQYDK2VwBCIEILcA8v5gDH7X"
    example_data["binance"]["api_key"] = api_key_sentinel
    for pair in example_data["pairs"]:
        pair["enabled"] = False

    with pytest.raises(ValidationError) as error_info:
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)

    _assert_validation_error_is_secret_safe(
        error_info.value,
        (api_key_sentinel, public_test_key_fragment, "-----BEGIN PRIVATE KEY-----"),
    )


@pytest.mark.parametrize(
    ("map_name", "colliding_key", "colliding_value"),
    [
        ("position_tiers", "22.290", "3"),
        ("reduce_bp_by_current_target", "3.0", "17"),
    ],
)
def test_raw_tier_map_keys_must_not_collide_after_decimal_normalization(
    example_data: dict,
    map_name: str,
    colliding_key: str,
    colliding_value: str,
):
    regular = example_data["pairs"][0]["sessions"]["regular"]
    regular[map_name][colliding_key] = colliding_value

    with pytest.raises(ValidationError, match="duplicate normalized Decimal key"):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_regular",
        "unknown_session",
        "missing_zero_tier",
        "negative_entry",
        "zero_target",
        "decreasing_target",
        "missing_reduce_target",
        "extra_reduce_target",
        "base_reduce_nonzero",
        "higher_reduce_equal_entry",
        "negative_reduce",
        "p99_below_tier",
        "history_below_p99",
    ],
)
def test_session_and_tier_maps_are_strict(example_data: dict, mutation: str):
    sessions = example_data["pairs"][0]["sessions"]
    regular = sessions["regular"]
    tiers = regular["position_tiers"]
    reductions = regular["reduce_bp_by_current_target"]
    if mutation == "missing_regular":
        sessions["extended"] = sessions.pop("regular")
    elif mutation == "unknown_session":
        sessions["overnight"] = copy.deepcopy(regular)
    elif mutation == "missing_zero_tier":
        tiers.pop("0")
    elif mutation == "negative_entry":
        tiers["-1"] = tiers.pop("0")
    elif mutation == "zero_target":
        tiers["0"] = "0"
        reductions.pop("1")
        reductions["0"] = "0"
    elif mutation == "decreasing_target":
        tiers["22.29"] = "0.5"
        reductions.pop("3")
        reductions["0.5"] = "17"
    elif mutation == "missing_reduce_target":
        reductions.pop("3")
    elif mutation == "extra_reduce_target":
        reductions["999"] = "1"
    elif mutation == "base_reduce_nonzero":
        reductions["1"] = "0.01"
    elif mutation == "higher_reduce_equal_entry":
        reductions["3"] = "22.29"
    elif mutation == "negative_reduce":
        reductions["3"] = "-1"
    elif mutation == "p99_below_tier":
        regular["p99_bp"] = "100"
    elif mutation == "history_below_p99":
        regular["historical_max_bp"] = "140"

    with pytest.raises(ValidationError):
        EquityLeveragedEtfArbitrageConfig.model_validate(example_data)
