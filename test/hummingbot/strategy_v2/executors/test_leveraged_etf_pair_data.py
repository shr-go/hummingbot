import copy
import hashlib
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path

import pytest
from pydantic import ValidationError

from hummingbot.strategy_v2.executors.leveraged_etf_pair_executor.data_types import (
    LeveragedEtfPairDirection,
    LeveragedEtfPairExecutorConfig,
    LeveragedEtfPairExecutorCustomInfoV1,
    LeveragedEtfPairExecutorReportV1,
    LeveragedEtfPairExecutorSnapshotV1,
    LeveragedEtfPairExecutorStateV1,
    LeveragedEtfPairOperation,
    LeveragedEtfPairState,
    LeverageReservationV1,
    OrderReferenceV1,
)


def _contract_document(file_name: str) -> dict:
    relative_path = Path("contracts/equity_leveraged_etf/v1") / file_name
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.is_file():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise AssertionError(f"F001 contract document not found: {relative_path}")


def _contract_vectors() -> dict:
    return _contract_document("round_trip_vectors.json")


@pytest.fixture(scope="module")
def vectors() -> dict:
    return _contract_vectors()


def _config_from_fixture(fixture: dict) -> LeveragedEtfPairExecutorConfig:
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


def _state_from_fixture(fixture: dict) -> LeveragedEtfPairExecutorStateV1:
    return LeveragedEtfPairExecutorStateV1(
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


def _apply_json_mutation(document: dict, mutation: dict) -> dict:
    mutated = copy.deepcopy(document)
    path = mutation["path"].strip("/").split("/")
    parent = mutated
    for part in path[:-1]:
        parent = parent[int(part)] if isinstance(parent, list) else parent[part]
    final_part = int(path[-1]) if isinstance(parent, list) else path[-1]
    if mutation["op"] == "remove":
        parent.pop(final_part)
    elif mutation["op"] in {"add", "replace"}:
        parent[final_part] = mutation["value"]
    elif mutation["op"] == "move":
        source_path = mutation["from"].strip("/").split("/")
        source_parent = mutated
        for part in source_path[:-1]:
            source_parent = source_parent[int(part)] if isinstance(source_parent, list) else source_parent[part]
        source_part = int(source_path[-1]) if isinstance(source_parent, list) else source_path[-1]
        parent.insert(final_part, source_parent.pop(source_part))
    else:
        raise AssertionError(f"unsupported F001 test mutation: {mutation['op']}")
    return mutated


@pytest.mark.parametrize("fixture_name", ["executor_active", "executor_terminal"])
def test_f001_snapshot_vectors_round_trip_canonically(vectors: dict, fixture_name: str):
    fixture = vectors["fixtures"][fixture_name]

    snapshot = LeveragedEtfPairExecutorSnapshotV1.model_validate(fixture)

    assert snapshot.model_dump(mode="json") == fixture
    assert json.loads(snapshot.canonical_json()) == fixture
    assert snapshot.canonical_sha256() == vectors["fixture_canonical_sha256"][fixture_name]
    assert snapshot == LeveragedEtfPairExecutorSnapshotV1.model_validate_json(snapshot.model_dump_json())


def test_all_f001_executor_schema_and_semantic_vectors(vectors: dict):
    executor_vectors = [
        vector
        for group in ("schema_vectors", "semantic_vectors")
        for vector in vectors[group]
        if vector.get("fixture", vector.get("base_fixture", "")).startswith("executor_")
    ]

    for vector in executor_vectors:
        fixture_name = vector.get("fixture", vector.get("base_fixture"))
        payload = copy.deepcopy(vectors["fixtures"][fixture_name])
        if "mutation" in vector:
            payload = _apply_json_mutation(payload, vector["mutation"])
        if vector["valid"]:
            assert LeveragedEtfPairExecutorSnapshotV1.model_validate(payload)
        else:
            with pytest.raises(ValidationError):
                LeveragedEtfPairExecutorSnapshotV1.model_validate(payload)


def test_enum_wire_values_exactly_match_f001_schema():
    definitions = _contract_document("wire.schema.json")["$defs"]

    assert {member.value for member in LeveragedEtfPairOperation} == set(definitions["Operation"]["enum"])
    assert {member.value for member in LeveragedEtfPairDirection} == set(definitions["Direction"]["enum"])
    assert {member.value for member in LeveragedEtfPairState} == set(definitions["ExecutorState"]["enum"])


@pytest.mark.parametrize("fixture_name", ["executor_active", "executor_terminal"])
def test_config_and_state_reconstruct_the_f001_snapshot(vectors: dict, fixture_name: str):
    fixture = vectors["fixtures"][fixture_name]
    config = _config_from_fixture(fixture)
    state = _state_from_fixture(fixture)

    snapshot = LeveragedEtfPairExecutorSnapshotV1.from_config_and_state(config, state)

    assert snapshot.model_dump(mode="json") == fixture
    assert config.type == "leveraged_etf_pair_executor"
    assert config.connector_name == fixture["etf_connector_name"]
    assert config.trading_pair == fixture["etf_trading_pair"]
    assert config.connector_names == (
        fixture["etf_connector_name"],
        fixture["stock_connector_name"],
    )
    assert config.trading_pairs == (
        fixture["etf_trading_pair"],
        fixture["stock_trading_pair"],
    )


def test_wire_models_are_immutable(vectors: dict):
    snapshot = LeveragedEtfPairExecutorSnapshotV1.model_validate(vectors["fixtures"]["executor_active"])

    with pytest.raises(ValidationError, match="frozen"):
        snapshot.state = LeveragedEtfPairState.COMPLETED


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("schema_version", 2),
        ("operation", "LIQUIDATE"),
        ("direction", "UNKNOWN"),
        ("state", "RUNNING"),
        ("created_raw_bp", "1.2300"),
        ("created_net_bp", 42.08),
        ("s0", "1e3"),
        ("target_gross_notional", "-0"),
        ("config_hash", "A" * 64),
        ("created_at_utc", "2026-07-17T14:01:02Z"),
    ],
)
def test_snapshot_rejects_noncanonical_or_unknown_wire_values(vectors: dict, field: str, invalid_value):
    fixture = copy.deepcopy(vectors["fixtures"]["executor_active"])
    fixture[field] = invalid_value

    with pytest.raises(ValidationError):
        LeveragedEtfPairExecutorSnapshotV1.model_validate(fixture)


@pytest.mark.parametrize(
    "secret_field",
    ["api_key", "binance_api_key", "private_key", "secret", "passphrase"],
)
def test_secret_bearing_snapshot_input_is_rejected(vectors: dict, secret_field: str):
    fixture = copy.deepcopy(vectors["fixtures"]["executor_active"])
    fixture[secret_field] = "must-not-persist"

    with pytest.raises(ValidationError, match="secret-bearing field"):
        LeveragedEtfPairExecutorSnapshotV1.model_validate(fixture)


def test_pair_config_requires_distinct_complete_leg_roles(vectors: dict):
    fixture = vectors["fixtures"]["executor_active"]
    config_data = _config_from_fixture(fixture).model_dump(mode="json")
    config_data.pop("stock_trading_pair")

    with pytest.raises(ValidationError):
        LeveragedEtfPairExecutorConfig.model_validate(config_data)

    config_data["stock_trading_pair"] = config_data["etf_trading_pair"]
    with pytest.raises(ValidationError, match="distinct connector/trading-pair roles"):
        LeveragedEtfPairExecutorConfig.model_validate(config_data)


@pytest.mark.parametrize("order_field", ["maker_order_ids", "stock_order_ids"])
def test_order_references_require_strict_sequence_and_unique_client_ids(vectors: dict, order_field: str):
    fixture = copy.deepcopy(vectors["fixtures"]["executor_terminal"])
    fixture[order_field].reverse()

    with pytest.raises(ValidationError, match="strictly increasing"):
        LeveragedEtfPairExecutorSnapshotV1.model_validate(fixture)

    fixture = copy.deepcopy(vectors["fixtures"]["executor_terminal"])
    fixture[order_field][1]["client_order_id"] = fixture[order_field][0]["client_order_id"]
    with pytest.raises(ValidationError, match="unique client_order_id"):
        LeveragedEtfPairExecutorSnapshotV1.model_validate(fixture)


def test_report_and_custom_info_preserve_optional_forward_compatibility(vectors: dict):
    fixture = vectors["fixtures"]["executor_active"]
    state = _state_from_fixture(fixture)
    empty_report = LeveragedEtfPairExecutorReportV1()
    report = LeveragedEtfPairExecutorReportV1(
        net_pnl_pct="0.125",
        net_pnl_quote="12.5",
        realized_pnl_quote="10",
        unrealized_pnl_quote="2.5",
        cum_fees_quote="0.25",
        filled_amount_quote="1000",
    )
    custom_info = LeveragedEtfPairExecutorCustomInfoV1(state=state, report=report)

    assert empty_report.model_dump(mode="json", exclude_none=True) == {"schema_version": 1}
    assert custom_info == LeveragedEtfPairExecutorCustomInfoV1.model_validate_json(custom_info.model_dump_json())
    assert custom_info.model_dump(mode="json")["report"]["net_pnl_quote"] == "12.5"


def test_decimal_datetime_and_hash_fields_use_native_types(vectors: dict):
    snapshot = LeveragedEtfPairExecutorSnapshotV1.model_validate(vectors["fixtures"]["executor_active"])

    assert snapshot.s0 == Decimal("250")
    assert snapshot.updated_at_utc == datetime(2026, 7, 17, 14, 1, 3, 123456, tzinfo=timezone.utc)
    assert snapshot.config_hash == "a" * 64
    assert hashlib.sha256(snapshot.canonical_json().encode("utf-8")).hexdigest() == snapshot.canonical_sha256()
    assert isinstance(snapshot.leverage_reservation, LeverageReservationV1)
    assert snapshot.operation is LeveragedEtfPairOperation.OPEN
    assert snapshot.direction is LeveragedEtfPairDirection.SHORT_ETF_LONG_STOCK


@pytest.mark.parametrize("coercive_value", [True, "1"])
def test_schema_version_rejects_boolean_and_numeric_string(vectors: dict, coercive_value):
    fixture = vectors["fixtures"]["executor_active"]
    state = _state_from_fixture(fixture)
    versioned_payloads = [
        (LeveragedEtfPairExecutorConfig, _config_from_fixture(fixture).model_dump(mode="json")),
        (LeveragedEtfPairExecutorStateV1, state.model_dump(mode="json")),
        (LeveragedEtfPairExecutorSnapshotV1, copy.deepcopy(fixture)),
        (LeveragedEtfPairExecutorReportV1, LeveragedEtfPairExecutorReportV1().model_dump(mode="json")),
        (
            LeveragedEtfPairExecutorCustomInfoV1,
            LeveragedEtfPairExecutorCustomInfoV1(state=state).model_dump(mode="json"),
        ),
    ]

    for model, payload in versioned_payloads:
        payload["schema_version"] = coercive_value
        with pytest.raises(ValidationError):
            model.model_validate(payload)


@pytest.mark.parametrize("coercive_value", [True, "1"])
def test_order_reference_sequence_rejects_boolean_and_numeric_string(coercive_value):
    with pytest.raises(ValidationError):
        OrderReferenceV1.model_validate(
            {
                "sequence": coercive_value,
                "client_order_id": "exec-maker-1",
                "exchange_order_id": "123",
            }
        )


@pytest.mark.parametrize("field", ["etf_leverage", "stock_leverage"])
@pytest.mark.parametrize("coercive_value", [True, "20"])
def test_leverage_reservation_integers_reject_boolean_and_numeric_string(
    vectors: dict,
    field: str,
    coercive_value,
):
    payload = copy.deepcopy(vectors["fixtures"]["executor_active"]["leverage_reservation"])
    payload[field] = coercive_value

    with pytest.raises(ValidationError):
        LeverageReservationV1.model_validate(payload)


@pytest.mark.parametrize("coercive_value", [True, "17"])
def test_last_journal_sequence_rejects_boolean_and_numeric_string_in_state_and_snapshot(
    vectors: dict,
    coercive_value,
):
    fixture = vectors["fixtures"]["executor_active"]
    state_payload = _state_from_fixture(fixture).model_dump(mode="json")
    snapshot_payload = copy.deepcopy(fixture)
    state_payload["last_journal_sequence"] = coercive_value
    snapshot_payload["last_journal_sequence"] = coercive_value

    with pytest.raises(ValidationError):
        LeveragedEtfPairExecutorStateV1.model_validate(state_payload)
    with pytest.raises(ValidationError):
        LeveragedEtfPairExecutorSnapshotV1.model_validate(snapshot_payload)


@pytest.mark.parametrize("coercive_value", [True, "20"])
def test_nested_config_state_and_snapshot_reject_coercive_reservation_leverage(
    vectors: dict,
    coercive_value,
):
    fixture = vectors["fixtures"]["executor_active"]
    config_payload = _config_from_fixture(fixture).model_dump(mode="json")
    state_payload = _state_from_fixture(fixture).model_dump(mode="json")
    snapshot_payload = copy.deepcopy(fixture)
    for payload in (config_payload, state_payload, snapshot_payload):
        payload["leverage_reservation"]["etf_leverage"] = coercive_value

    with pytest.raises(ValidationError):
        LeveragedEtfPairExecutorConfig.model_validate(config_payload)
    with pytest.raises(ValidationError):
        LeveragedEtfPairExecutorStateV1.model_validate(state_payload)
    with pytest.raises(ValidationError):
        LeveragedEtfPairExecutorSnapshotV1.model_validate(snapshot_payload)


@pytest.mark.parametrize("coercive_value", [True, "0"])
def test_nested_state_and_snapshot_reject_coercive_order_sequence(vectors: dict, coercive_value):
    fixture = vectors["fixtures"]["executor_active"]
    state_payload = _state_from_fixture(fixture).model_dump(mode="json")
    snapshot_payload = copy.deepcopy(fixture)
    state_payload["maker_order_ids"][0]["sequence"] = coercive_value
    snapshot_payload["maker_order_ids"][0]["sequence"] = coercive_value

    with pytest.raises(ValidationError):
        LeveragedEtfPairExecutorStateV1.model_validate(state_payload)
    with pytest.raises(ValidationError):
        LeveragedEtfPairExecutorSnapshotV1.model_validate(snapshot_payload)
