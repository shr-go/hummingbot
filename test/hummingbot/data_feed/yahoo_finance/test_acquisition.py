import asyncio
import unittest
from collections import defaultdict, deque
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from hummingbot.data_feed.yahoo_finance.acquisition import (
    AnchorAcquisitionStatus,
    FinalizedAnchorStatus,
    YahooAnchorAcquisition,
)
from hummingbot.data_feed.yahoo_finance.parser import YahooCloseObservation
from hummingbot.data_feed.yahoo_finance.provider import YahooHTTPError
from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import NavConfig

from .conftest import (
    FakeClock,
    OFFICIAL_CLOSE,
    TARGET_SESSION_DATE,
    load_nav_config,
)


UTC = timezone.utc
STOCK_BAR_TIME = datetime(2026, 7, 17, 13, 30, tzinfo=UTC)
STOCK_MARKET_TIME = datetime(2026, 7, 17, 19, 59, 58, tzinfo=UTC)
ETF_MARKET_TIME = datetime(2026, 7, 17, 19, 47, 11, tzinfo=UTC)


def close_observation(
    symbol: str,
    close: str,
    received_at: datetime,
    hash_character: str,
) -> YahooCloseObservation:
    return YahooCloseObservation(
        symbol=symbol,
        target_session_date=TARGET_SESSION_DATE,
        close=Decimal(close),
        bar_timestamp_utc=STOCK_BAR_TIME,
        regular_market_time_utc=STOCK_MARKET_TIME if symbol == "SNDK" else ETF_MARKET_TIME,
        received_at_utc=received_at,
        raw_response_hash=hash_character * 64,
        source_url=f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}",
    )


def observation_pair(
    received_at: datetime,
    *,
    stock_close: str = "250",
    etf_close: str = "30",
    stock_hash: str = "1",
    etf_hash: str = "2",
    etf_delay_seconds: float = 0.15,
):
    return (
        close_observation("SNDK", stock_close, received_at, stock_hash),
        close_observation("SNXX", etf_close, received_at + timedelta(seconds=etf_delay_seconds), etf_hash),
    )


class ScriptedPairProvider:
    def __init__(self, *pairs):
        self.outcomes = defaultdict(deque)
        for stock_outcome, etf_outcome in pairs:
            self.outcomes["SNDK"].append(stock_outcome)
            self.outcomes["SNXX"].append(etf_outcome)
        self.calls = []
        self.active = 0
        self.max_active = 0

    async def fetch_close(self, symbol, target_session_date, deadline_utc):
        self.calls.append((symbol, target_session_date, deadline_utc))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(0)
            if not self.outcomes[symbol]:
                raise AssertionError(f"unexpected fetch for {symbol}")
            outcome = self.outcomes[symbol].popleft()
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome
        finally:
            self.active -= 1


class YahooAnchorAcquisitionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.nav_config = load_nav_config()

    def acquisition(self, clock, provider):
        return YahooAnchorAcquisition(
            nav_config=self.nav_config,
            provider=provider,
            utc_clock=clock.utcnow,
            monotonic_clock=clock.monotonic,
            sleep=clock.sleep,
            jitter=lambda upper_bound: 0.0,
        )

    def new_checkpoint(self, acquisition):
        return acquisition.create_checkpoint(
            cycle_id="xnys-2026-07-17",
            target_session_date=TARGET_SESSION_DATE,
            official_close_utc=OFFICIAL_CLOSE,
        )

    async def test_checkpoint_deadline_is_fixed_from_official_close_not_restart_time(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=137))
        acquisition = self.acquisition(clock, ScriptedPairProvider())

        checkpoint = self.new_checkpoint(acquisition)

        assert checkpoint.deadline_utc == OFFICIAL_CLOSE + timedelta(seconds=600)
        assert checkpoint.next_poll_utc == OFFICIAL_CLOSE
        assert checkpoint.attempt == 0
        assert checkpoint.confirmation_count == 0
        assert checkpoint.revision == 1

    async def test_advance_before_minimum_finalize_delay_waits_without_fetch_or_revision(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=59, microseconds=999999))
        provider = ScriptedPairProvider()
        acquisition = self.acquisition(clock, provider)
        checkpoint = self.new_checkpoint(acquisition)

        result = await acquisition.advance(checkpoint, stock_symbol="SNDK", etf_symbol="SNXX")

        assert result.status is AnchorAcquisitionStatus.WAITING
        assert result.checkpoint is checkpoint
        assert result.candidate is None
        assert provider.calls == []

    async def test_same_round_stock_and_etf_fetches_are_concurrent_and_persist_first_confirmation(self):
        received_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        provider = ScriptedPairProvider(observation_pair(received_at))
        clock = FakeClock(received_at)
        acquisition = self.acquisition(clock, provider)

        result = await acquisition.advance(
            self.new_checkpoint(acquisition),
            stock_symbol="SNDK",
            etf_symbol="SNXX",
        )

        assert provider.max_active == 2
        assert result.status is AnchorAcquisitionStatus.POLLING
        assert result.candidate is None
        assert result.checkpoint.attempt == 1
        assert result.checkpoint.confirmation_count == 1
        assert result.checkpoint.candidate_stock_close == Decimal("250")
        assert result.checkpoint.candidate_etf_close == Decimal("30")
        assert result.checkpoint.candidate_stock_raw_response_hash == "1" * 64
        assert result.checkpoint.candidate_etf_raw_response_hash == "2" * 64
        assert result.checkpoint.revision == 2
        assert result.checkpoint.next_poll_utc == received_at + timedelta(seconds=5.15)

    async def test_two_matching_pairs_separated_by_confirmation_interval_produce_immutable_candidate(self):
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)

        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.current = second_at
        clock.monotonic_value += 5.15
        second = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")

        assert second.status is AnchorAcquisitionStatus.FINALIZABLE
        assert second.checkpoint.confirmation_count == 2
        assert second.checkpoint.attempt == 2
        assert second.candidate is not None
        assert second.candidate.stock_close == Decimal("250")
        assert second.candidate.etf_close == Decimal("30")
        assert second.candidate.stock_regular_market_time_utc == STOCK_MARKET_TIME
        assert second.candidate.etf_regular_market_time_utc == ETF_MARKET_TIME
        assert second.candidate.verify_evidence_hash()
        with self.assertRaises(FrozenInstanceError):
            second.candidate.stock_close = Decimal("251")

    async def test_valid_sub_600_second_anchor_window_still_completes_stably(self):
        shorter_values = self.nav_config.model_dump()
        shorter_values["anchor_wait_timeout_seconds"] = 180
        shorter_config = NavConfig.model_validate(shorter_values)
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(first_at)
        acquisition = YahooAnchorAcquisition(
            nav_config=shorter_config,
            provider=provider,
            utc_clock=clock.utcnow,
            monotonic_clock=clock.monotonic,
            sleep=clock.sleep,
            jitter=lambda upper_bound: 0.0,
        )

        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.advance(5.15)
        second = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")

        assert second.status is AnchorAcquisitionStatus.FINALIZABLE
        assert second.checkpoint.deadline_utc == OFFICIAL_CLOSE + timedelta(seconds=180)

    async def test_runner_uses_injected_sleep_and_emits_checkpoint_after_every_completed_round(self):
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(OFFICIAL_CLOSE)
        acquisition = self.acquisition(clock, provider)
        emitted = []

        result = await acquisition.run(
            self.new_checkpoint(acquisition),
            stock_symbol="SNDK",
            etf_symbol="SNXX",
            on_checkpoint=emitted.append,
        )

        assert result.status is AnchorAcquisitionStatus.FINALIZABLE
        assert clock.sleeps == [60.0, 5.15]
        assert [checkpoint.revision for checkpoint in emitted] == [2, 3]
        assert emitted[-1] is result.checkpoint

    async def test_either_leg_change_resets_paired_confirmation_to_one_current_observation():
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(
            observation_pair(first_at),
            observation_pair(second_at, etf_close="30.01", etf_hash="4"),
        )
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)

        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.current = second_at
        clock.monotonic_value += 5.15
        changed = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")

        assert changed.status is AnchorAcquisitionStatus.POLLING
        assert changed.checkpoint.confirmation_count == 1
        assert changed.checkpoint.candidate_stock_close == Decimal("250")
        assert changed.checkpoint.candidate_etf_close == Decimal("30.01")
        assert changed.checkpoint.candidate_stock_raw_response_hash == "1" * 64
        assert changed.checkpoint.candidate_etf_raw_response_hash == "4" * 64

    async def test_identical_pair_received_too_soon_does_not_slide_confirmation_baseline():
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        too_soon_at = first_at + timedelta(seconds=4)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(too_soon_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)

        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        early_checkpoint = replace(first.checkpoint, next_poll_utc=too_soon_at)
        clock.current = too_soon_at
        clock.monotonic_value += 4
        too_soon = await acquisition.advance(early_checkpoint, "SNDK", "SNXX")

        assert too_soon.checkpoint.confirmation_count == 1
        assert too_soon.checkpoint.stock_received_at_utc == first.checkpoint.stock_received_at_utc
        assert too_soon.checkpoint.etf_received_at_utc == first.checkpoint.etf_received_at_utc
        assert too_soon.checkpoint.next_poll_utc == first.checkpoint.etf_received_at_utc + timedelta(seconds=5)

    async def test_asymmetric_leg_delay_over_skew_limit_fails_whole_round_and_clears_candidate():
        received_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        provider = ScriptedPairProvider(observation_pair(received_at, etf_delay_seconds=5.000001))
        clock = FakeClock(received_at)
        acquisition = self.acquisition(clock, provider)

        result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        assert result.status is AnchorAcquisitionStatus.POLLING
        assert result.checkpoint.confirmation_count == 0
        assert result.checkpoint.candidate_stock_close is None
        assert result.checkpoint.candidate_etf_close is None
        assert result.failure_reason == "paired Yahoo receive skew exceeds configured maximum"

    async def test_one_leg_failure_fails_whole_round_and_uses_bounded_exponential_next_poll():
        received_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        stock, etf = observation_pair(received_at)
        provider = ScriptedPairProvider((YahooHTTPError("synthetic 503"), etf))
        clock = FakeClock(received_at)
        acquisition = self.acquisition(clock, provider)

        result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        assert result.status is AnchorAcquisitionStatus.POLLING
        assert result.checkpoint.confirmation_count == 0
        assert result.checkpoint.attempt == 1
        assert result.checkpoint.next_poll_utc == received_at + timedelta(seconds=2)
        assert "synthetic 503" in result.failure_reason

    async def test_conservative_remaining_schedule_budget_fails_closed_before_impossible_poll():
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=570))
        provider = ScriptedPairProvider()
        acquisition = self.acquisition(clock, provider)

        result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        assert result.status is AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE
        assert result.candidate is None
        assert provider.calls == []
        assert result.failure_reason == "insufficient conservative budget for remaining confirmations"

    async def test_deadline_equality_and_post_deadline_restart_never_issue_late_request():
        for seconds_after_close in (600, 601):
            with self.subTest(seconds_after_close=seconds_after_close):
                clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=seconds_after_close))
                provider = ScriptedPairProvider()
                acquisition = self.acquisition(clock, provider)

                result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

                assert result.status is AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE
                assert provider.calls == []
                assert result.checkpoint.deadline_utc == OFFICIAL_CLOSE + timedelta(seconds=600)

    async def test_response_received_after_absolute_deadline_cannot_finalize_late():
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        late_at = OFFICIAL_CLOSE + timedelta(seconds=601)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(late_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)
        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        clock.current = OFFICIAL_CLOSE + timedelta(seconds=575)
        clock.monotonic_value += 515
        late = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")

        assert late.status is AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE
        assert late.candidate is None

    async def test_restart_uses_prior_checkpoint_confirmation_and_original_remaining_deadline():
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        first_clock = FakeClock(first_at)
        first_acquisition = self.acquisition(first_clock, ScriptedPairProvider(observation_pair(first_at)))
        first = await first_acquisition.advance(self.new_checkpoint(first_acquisition), "SNDK", "SNXX")

        restart_clock = FakeClock(second_at, monotonic_value=42)
        restart_provider = ScriptedPairProvider(observation_pair(second_at))
        restarted = self.acquisition(restart_clock, restart_provider)
        result = await restarted.advance(first.checkpoint, "SNDK", "SNXX")

        assert result.status is AnchorAcquisitionStatus.FINALIZABLE
        assert result.checkpoint.deadline_utc == OFFICIAL_CLOSE + timedelta(seconds=600)
        assert result.checkpoint.confirmation_count == 2
        assert all(call[2] == OFFICIAL_CLOSE + timedelta(seconds=600) for call in restart_provider.calls)

    async def test_hash_corruption_requires_recovery_before_revision_comparison(self):
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)
        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.current = second_at
        clock.monotonic_value += 5.15
        final = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")
        corrupted = replace(final.candidate, evidence_hash="f" * 64)
        current_stock, current_etf = observation_pair(second_at + timedelta(seconds=10))

        assessment = acquisition.assess_finalized(corrupted, current_stock, current_etf)

        assert assessment.status is FinalizedAnchorStatus.RECOVERY_REQUIRED
        assert assessment.candidate is corrupted
        assert assessment.revision_observation is None

    async def test_post_finalization_price_revision_is_append_only_observation_not_replacement():
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)
        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.current = second_at
        clock.monotonic_value += 5.15
        final = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")
        revised_stock, revised_etf = observation_pair(
            second_at + timedelta(seconds=60),
            etf_close="30.01",
            etf_hash="4",
        )

        assessment = acquisition.assess_finalized(final.candidate, revised_stock, revised_etf)

        assert assessment.status is FinalizedAnchorStatus.DATA_REVISION
        assert assessment.candidate is final.candidate
        assert assessment.candidate.etf_close == Decimal("30")
        assert assessment.revision_observation.etf_close == Decimal("30.01")
        assert assessment.revision_observation.observed_at_utc == revised_etf.received_at_utc
        assert assessment.revision_observation.evidence_hash != final.candidate.evidence_hash

    async def test_post_finalization_same_closes_do_not_alert_when_irrelevant_raw_fields_change():
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)
        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.current = second_at
        clock.monotonic_value += 5.15
        final = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")
        current_stock, current_etf = observation_pair(
            second_at + timedelta(seconds=60),
            stock_hash="a",
            etf_hash="b",
        )

        assessment = acquisition.assess_finalized(final.candidate, current_stock, current_etf)

        assert assessment.status is FinalizedAnchorStatus.UNCHANGED
        assert assessment.candidate is final.candidate
        assert assessment.revision_observation is None
