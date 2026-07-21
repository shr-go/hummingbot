import asyncio
import copy
from collections import defaultdict, deque
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from hummingbot.data_feed.yahoo_finance import acquisition as acquisition_contract
from hummingbot.data_feed.yahoo_finance.acquisition import (
    AnchorAcquisitionStatus,
    AnchorCandidate,
    AnchorPollingCheckpoint,
    CheckpointIntegrityError,
    FinalizedAnchorStatus,
    YahooAnchorAcquisition,
)
from hummingbot.data_feed.yahoo_finance.parser import YahooCloseObservation

from .conftest import FakeClock, OFFICIAL_CLOSE, TARGET_SESSION_DATE, load_nav_config


UTC = timezone.utc


def observation(
    symbol: str,
    close: str,
    received_at: datetime,
    hash_character: str,
    round_index: int,
) -> YahooCloseObservation:
    return YahooCloseObservation(
        symbol=symbol,
        target_session_date=TARGET_SESSION_DATE,
        close=Decimal(close),
        bar_timestamp_utc=datetime(2026, 7, 17, 13, 30, round_index, tzinfo=UTC),
        regular_market_time_utc=datetime(2026, 7, 17, 19, 45, round_index, tzinfo=UTC),
        received_at_utc=received_at,
        raw_response_hash=hash_character * 64,
        source_url=(
            f"https://query{1 + round_index % 2}.finance.yahoo.com/"
            f"v8/finance/chart/{symbol}?round={round_index}"
        ),
    )


def observation_pair(
    stock_symbol: str,
    etf_symbol: str,
    received_at: datetime,
    stock_hash: str,
    etf_hash: str,
    round_index: int,
):
    return (
        observation(stock_symbol, "250", received_at, stock_hash, round_index),
        observation(etf_symbol, "30", received_at + timedelta(milliseconds=150), etf_hash, round_index),
    )


class PairProvider:
    def __init__(self, *pairs):
        self.outcomes = defaultdict(deque)
        for stock, etf in pairs:
            self.outcomes[stock.symbol].append(stock)
            self.outcomes[etf.symbol].append(etf)

    async def fetch_close(self, symbol, target_session_date, deadline_utc, *, budget=None):
        assert budget is not None
        await asyncio.sleep(0)
        return self.outcomes[symbol].popleft()


def acquisition(clock: FakeClock, provider: PairProvider) -> YahooAnchorAcquisition:
    return YahooAnchorAcquisition(
        nav_config=load_nav_config(),
        provider=provider,
        utc_clock=clock.utcnow,
        monotonic_clock=clock.monotonic,
        sleep=clock.sleep,
        jitter=lambda upper_bound: 0.0,
        etf_daily_multiplier=Decimal("2"),
    )


def new_checkpoint(
    anchor_acquisition: YahooAnchorAcquisition,
    pair_id: str,
    stock_symbol: str,
    etf_symbol: str,
) -> AnchorPollingCheckpoint:
    return anchor_acquisition.create_checkpoint(
        cycle_id="xnys-2026-07-17",
        target_session_date=TARGET_SESSION_DATE,
        official_close_utc=OFFICIAL_CLOSE,
        pair_id=pair_id,
        stock_symbol=stock_symbol,
        etf_symbol=etf_symbol,
    )


async def finalized_pair(pair_id: str, stock_symbol: str, etf_symbol: str, hash_offset: int):
    first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
    second_at = first_at + timedelta(seconds=5.15)
    provider = PairProvider(
        observation_pair(
            stock_symbol,
            etf_symbol,
            first_at,
            str(hash_offset),
            str(hash_offset + 1),
            1,
        ),
        observation_pair(
            stock_symbol,
            etf_symbol,
            second_at,
            str(hash_offset + 2),
            str(hash_offset + 3),
            2,
        ),
    )
    clock = FakeClock(first_at)
    anchor_acquisition = acquisition(clock, provider)
    checkpoint = new_checkpoint(anchor_acquisition, pair_id, stock_symbol, etf_symbol)
    first = await anchor_acquisition.advance(checkpoint, stock_symbol, etf_symbol)
    clock.advance(5.15)
    final = await anchor_acquisition.advance(first.checkpoint, stock_symbol, etf_symbol)
    assert final.status is AnchorAcquisitionStatus.FINALIZABLE
    return anchor_acquisition, checkpoint, final


class PairScopedMemoryRepository:
    """A no-SQL conformance harness for the F003 repository port."""

    def __init__(self):
        self.states = {}
        self.revisions = defaultdict(list)

    def load(self, key):
        return self.states.get(key)

    def compare_and_set_checkpoint(self, key, checkpoint, expected_revision):
        key.validate_checkpoint(checkpoint)
        current = self.states.get(key)
        current_revision = 0 if current is None else current.revision
        if isinstance(current, AnchorCandidate) or current_revision != expected_revision:
            raise RuntimeError("anchor revision conflict")
        self.states[key] = checkpoint
        return checkpoint

    def finalize_if_absent(self, key, candidate, expected_revision):
        key.validate_candidate(candidate)
        current = self.states.get(key)
        if isinstance(current, AnchorCandidate):
            if current.evidence_hash != candidate.evidence_hash:
                raise RuntimeError("anchor finalization conflict")
            return current
        if current is None or current.revision != expected_revision:
            raise RuntimeError("anchor revision conflict")
        self.states[key] = candidate
        return candidate

    def append_revision_observation(self, key, revision_observation):
        key.validate_revision_observation(revision_observation)
        if not isinstance(self.states.get(key), AnchorCandidate):
            raise RuntimeError("anchor is not finalized")
        self.revisions[key].append(revision_observation)


@pytest.mark.asyncio
async def test_boundary_round_is_retained_across_restart_and_binds_latest_candidate_and_revision():
    pair_id = "sndk_snxx"
    first_at = OFFICIAL_CLOSE
    second_at = OFFICIAL_CLOSE + timedelta(seconds=5.15)
    boundary = OFFICIAL_CLOSE + timedelta(seconds=60)
    initial_provider = PairProvider(
        observation_pair("SNDK", "SNXX", first_at, "1", "2", 1),
        observation_pair("SNDK", "SNXX", second_at, "3", "4", 2),
    )
    initial_clock = FakeClock(first_at)
    initial = acquisition(initial_clock, initial_provider)
    first = await initial.advance(new_checkpoint(initial, pair_id, "SNDK", "SNXX"), "SNDK", "SNXX")
    initial_clock.advance(5.15)
    confirmed = await initial.advance(first.checkpoint, "SNDK", "SNXX")
    persisted = copy.deepcopy(confirmed.checkpoint.to_recovery_fields())

    restart_provider = PairProvider(
        observation_pair("SNDK", "SNXX", boundary, "5", "6", 3)
    )
    restart_clock = FakeClock(boundary)
    restarted = acquisition(restart_clock, restart_provider)
    restored_checkpoint = AnchorPollingCheckpoint.from_recovery_fields(persisted)
    final = await restarted.advance(restored_checkpoint, "SNDK", "SNXX")

    assert final.status is AnchorAcquisitionStatus.FINALIZABLE
    assert final.checkpoint.pair_id == pair_id
    assert final.checkpoint.confirmation_count == 3
    assert final.candidate.observed_confirmation_count == 3
    assert final.candidate.finalized_at_utc == boundary + timedelta(milliseconds=150)
    assert final.candidate.stock_raw_response_hash == "5" * 64
    assert final.candidate.stock_bar_timestamp_utc.second == 3
    assert [record.confirmation_index for record in final.candidate.confirmation_evidence] == [1, 2, 3]
    assert len({record.stock_raw_response_hash for record in final.candidate.confirmation_evidence}) == 3
    assert len({record.stock_received_at_utc for record in final.candidate.confirmation_evidence}) == 3
    assert len({record.stock_source_url for record in final.candidate.confirmation_evidence}) == 3
    assert len({record.stock_bar_timestamp_utc for record in final.candidate.confirmation_evidence}) == 3

    candidate_fields = final.candidate.to_evidence_fields()
    restored_candidate = AnchorCandidate.from_evidence_fields(candidate_fields)
    assert restored_candidate == final.candidate
    revised_stock, revised_etf = observation_pair(
        "SNDK",
        "SNXX",
        boundary + timedelta(seconds=1),
        "7",
        "8",
        4,
    )
    revised_etf = replace(revised_etf, close=Decimal("30.01"))
    assessment = restarted.assess_finalized(restored_candidate, revised_stock, revised_etf)
    assert assessment.status is FinalizedAnchorStatus.DATA_REVISION
    assert assessment.revision_observation.pair_id == pair_id


@pytest.mark.asyncio
async def test_repository_v2_key_isolates_two_pairs_through_cas_reset_finalize_restart_and_revision():
    key_type = getattr(acquisition_contract, "AnchorRepositoryKey")
    repository_protocol = getattr(acquisition_contract, "AnchorRepositoryV2")
    key_sndk = key_type.create("sndk_snxx", "xnys-2026-07-17")
    key_intc = key_type.create("intc_intw", "xnys-2026-07-17")
    sndk_acquisition, sndk_empty, sndk_final = await finalized_pair("sndk_snxx", "SNDK", "SNXX", 1)
    intc_acquisition, intc_empty, intc_final = await finalized_pair("intc_intw", "INTC", "INTW", 5)
    repository = PairScopedMemoryRepository()

    assert isinstance(repository, repository_protocol)
    repository.compare_and_set_checkpoint(key_sndk, sndk_empty, expected_revision=0)
    repository.compare_and_set_checkpoint(key_intc, intc_empty, expected_revision=0)
    assert repository.load(key_sndk).pair_id == "sndk_snxx"
    assert repository.load(key_intc).pair_id == "intc_intw"

    reset_sndk = replace(
        sndk_empty,
        attempt=1,
        next_poll_utc=sndk_empty.official_close_utc + timedelta(seconds=1),
        revision=2,
        integrity_hash=None,
    )
    repository.compare_and_set_checkpoint(key_sndk, reset_sndk, expected_revision=1)
    assert repository.load(key_sndk) == reset_sndk
    assert repository.load(key_intc) == intc_empty

    sndk_candidate = AnchorCandidate.from_evidence_fields(sndk_final.candidate.to_evidence_fields())
    intc_candidate = AnchorCandidate.from_evidence_fields(intc_final.candidate.to_evidence_fields())
    assert repository.finalize_if_absent(key_sndk, sndk_candidate, expected_revision=2) == sndk_candidate
    assert repository.finalize_if_absent(key_sndk, sndk_candidate, expected_revision=2) == sndk_candidate
    assert repository.finalize_if_absent(key_intc, intc_candidate, expected_revision=1) == intc_candidate

    sndk_restart = AnchorCandidate.from_evidence_fields(repository.load(key_sndk).to_evidence_fields())
    intc_restart = AnchorCandidate.from_evidence_fields(repository.load(key_intc).to_evidence_fields())
    assert sndk_restart.pair_id == "sndk_snxx"
    assert intc_restart.pair_id == "intc_intw"

    sndk_stock, sndk_etf = observation_pair(
        "SNDK", "SNXX", OFFICIAL_CLOSE + timedelta(seconds=90), "9", "a", 5
    )
    intc_stock, intc_etf = observation_pair(
        "INTC", "INTW", OFFICIAL_CLOSE + timedelta(seconds=90), "b", "c", 5
    )
    sndk_etf = replace(sndk_etf, close=Decimal("30.01"))
    intc_etf = replace(intc_etf, close=Decimal("30.02"))
    sndk_revision = sndk_acquisition.assess_finalized(sndk_restart, sndk_stock, sndk_etf).revision_observation
    intc_revision = intc_acquisition.assess_finalized(intc_restart, intc_stock, intc_etf).revision_observation
    repository.append_revision_observation(key_sndk, sndk_revision)
    repository.append_revision_observation(key_intc, intc_revision)
    assert repository.revisions[key_sndk] == [sndk_revision]
    assert repository.revisions[key_intc] == [intc_revision]

    before = dict(repository.states)
    with pytest.raises((CheckpointIntegrityError, ValueError), match="pair|key"):
        repository.compare_and_set_checkpoint(key_sndk, intc_empty, expected_revision=2)
    with pytest.raises((CheckpointIntegrityError, ValueError), match="pair|key"):
        key_sndk.validate_candidate(intc_candidate)
    with pytest.raises((CheckpointIntegrityError, ValueError), match="pair|key"):
        repository.append_revision_observation(key_sndk, intc_revision)
    assert repository.states == before
