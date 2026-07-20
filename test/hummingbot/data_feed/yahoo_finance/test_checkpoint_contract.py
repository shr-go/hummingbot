import copy
import json
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

import pytest

from hummingbot.data_feed.yahoo_finance.acquisition import (
    AnchorPollingCheckpoint,
    CheckpointIntegrityError,
    anchor_evidence_hash,
)

from .conftest import find_repository_file


def contract_vectors() -> dict:
    path = find_repository_file("contracts/equity_leveraged_etf/v1/round_trip_vectors.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("fixture_name", ["checkpoint_empty", "checkpoint_confirmed"])
def test_checkpoint_domain_round_trips_f001_v1_contract_fields_exactly(fixture_name):
    wire_fields = contract_vectors()["fixtures"][fixture_name]

    checkpoint = AnchorPollingCheckpoint.from_contract_fields(wire_fields)

    assert checkpoint.to_contract_fields() == wire_fields
    assert checkpoint.schema_version == 1
    assert checkpoint.cycle_id == "xnys-2026-07-17"
    assert checkpoint.target_session_date == date(2026, 7, 17)
    assert checkpoint.official_close_utc == datetime(2026, 7, 17, 20, tzinfo=timezone.utc)
    if fixture_name == "checkpoint_confirmed":
        assert checkpoint.candidate_stock_close == Decimal("250")
        assert checkpoint.candidate_etf_close == Decimal("30")


def test_checkpoint_is_immutable_and_exposes_no_repository_side_effect_interface():
    wire_fields = contract_vectors()["fixtures"]["checkpoint_empty"]
    checkpoint = AnchorPollingCheckpoint.from_contract_fields(wire_fields)

    with pytest.raises(FrozenInstanceError):
        checkpoint.attempt = 99
    assert not hasattr(checkpoint, "save")
    assert not hasattr(checkpoint, "compare_and_set")
    assert not hasattr(checkpoint, "finalize_if_absent")


@pytest.mark.parametrize(
    ("mutation", "expected_pattern"),
    [
        (lambda value: value.update({"revision": 0}), "revision"),
        (lambda value: value.update({"cycle_id": "xnys-2026-07-18"}), "cycle"),
        (lambda value: value.update({"candidate_etf_close": None}), "paired candidate"),
        (lambda value: value.update({"candidate_stock_raw_response_hash": "A" * 64}), "hash"),
        (lambda value: value.update({"confirmation_count": 0}), "confirmation"),
        (lambda value: value.update({"deadline_utc": value["official_close_utc"]}), "deadline"),
    ],
)
def test_corrupted_checkpoint_fields_are_rejected_for_caller_recovery(mutation, expected_pattern):
    wire_fields = copy.deepcopy(contract_vectors()["fixtures"]["checkpoint_confirmed"])
    mutation(wire_fields)

    with pytest.raises(CheckpointIntegrityError, match=expected_pattern):
        AnchorPollingCheckpoint.from_contract_fields(wire_fields)


def test_empty_checkpoint_rejects_hidden_partial_candidate_state():
    wire_fields = copy.deepcopy(contract_vectors()["fixtures"]["checkpoint_empty"])
    wire_fields["candidate_stock_close"] = "250"

    with pytest.raises(CheckpointIntegrityError, match="paired candidate"):
        AnchorPollingCheckpoint.from_contract_fields(wire_fields)


@pytest.mark.parametrize(
    ("field_name", "value", "expected_pattern"),
    [
        ("stock_received_at_utc", "2026-07-17T19:59:59.999999Z", "official close|acquisition window"),
        ("etf_received_at_utc", "2026-07-17T20:10:00.000000Z", "deadline|acquisition window"),
        ("next_poll_utc", "2026-07-17T19:59:59.999999Z", "next poll|official close"),
        ("next_poll_utc", "2026-07-17T20:01:59.999999Z", "next poll|receive"),
    ],
)
def test_checkpoint_rejects_preclose_deadline_equal_or_incoherently_ordered_candidate_times(
    field_name,
    value,
    expected_pattern,
):
    wire_fields = copy.deepcopy(contract_vectors()["fixtures"]["checkpoint_confirmed"])
    wire_fields[field_name] = value

    with pytest.raises(CheckpointIntegrityError, match=expected_pattern):
        AnchorPollingCheckpoint.from_contract_fields(wire_fields)


def test_empty_checkpoint_next_poll_cannot_precede_official_close():
    wire_fields = copy.deepcopy(contract_vectors()["fixtures"]["checkpoint_empty"])
    official_close = datetime.fromisoformat(wire_fields["official_close_utc"].replace("Z", "+00:00"))
    wire_fields["next_poll_utc"] = (official_close - timedelta(microseconds=1)).strftime(
        "%Y-%m-%dT%H:%M:%S.%fZ"
    )

    with pytest.raises(CheckpointIntegrityError, match="next poll|official close"):
        AnchorPollingCheckpoint.from_contract_fields(wire_fields)


def test_f001_sorted_key_utf8_evidence_hash_vector_is_reused_exactly():
    vector = contract_vectors()["hash_vectors"][0]
    payload = vector["payload"]

    actual = anchor_evidence_hash(
        cycle_id=payload["cycle_id"],
        stock_raw_response_hash=payload["stock_raw_response_hash"],
        etf_raw_response_hash=payload["etf_raw_response_hash"],
    )

    assert actual == vector["expected_sha256"]


@pytest.mark.parametrize("bad_hash", ["f" * 63, "g" * 64, "F" * 64, ""])
def test_evidence_hash_helper_rejects_noncanonical_sha256_inputs(bad_hash):
    with pytest.raises(ValueError, match="hash"):
        anchor_evidence_hash(
            cycle_id="xnys-2026-07-17",
            stock_raw_response_hash=bad_hash,
            etf_raw_response_hash="2" * 64,
        )
