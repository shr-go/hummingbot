import asyncio
import copy
import unittest
from collections import defaultdict, deque
from dataclasses import FrozenInstanceError, replace
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal

from hummingbot.data_feed.yahoo_finance.acquisition import (
    AnchorAcquisitionStatus,
    AnchorPollingCheckpoint,
    CheckpointIntegrityError,
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

    async def fetch_close(self, symbol, target_session_date, deadline_utc, *, budget=None):
        self.calls.append((symbol, target_session_date, deadline_utc, budget))
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


class MonotonicSkewProvider:
    def __init__(self, clock, stock, etf, skew_seconds):
        self.clock = clock
        self.stock = stock
        self.etf = etf
        self.skew_seconds = skew_seconds
        self.stock_returned = asyncio.Event()

    async def fetch_close(self, symbol, target_session_date, deadline_utc, *, budget=None):
        assert budget is not None
        if symbol == "SNDK":
            self.stock_returned.set()
            return self.stock
        await self.stock_returned.wait()
        await asyncio.sleep(0)
        self.clock.monotonic_value += self.skew_seconds
        return self.etf


class DeadlineEqualityPairProvider:
    def __init__(self, clock, stock, etf, advance_seconds):
        self.clock = clock
        self.stock = stock
        self.etf = etf
        self.advance_seconds = advance_seconds
        self.calls = 0

    async def fetch_close(self, symbol, target_session_date, deadline_utc, *, budget=None):
        assert budget is not None
        self.calls += 1
        if symbol == "SNDK":
            self.clock.monotonic_value += self.advance_seconds
            return self.stock
        await asyncio.sleep(0)
        return self.etf


class CancellingPairProvider:
    def __init__(self, stock, etf):
        self.stock = stock
        self.etf = etf
        self.started = 0
        self.both_started = asyncio.Event()
        self.sibling_cancelled = False

    async def fetch_close(self, symbol, target_session_date, deadline_utc, *, budget=None):
        assert budget is not None
        self.started += 1
        if self.started == 2:
            self.both_started.set()
        await self.both_started.wait()
        if symbol == "SNDK":
            raise asyncio.CancelledError()
        try:
            for _ in range(20):
                await asyncio.sleep(0)
            return self.etf
        except asyncio.CancelledError:
            self.sibling_cancelled = True
            raise


class ExponentMustNotBeEvaluated(int):
    def __rpow__(self, other):
        raise AssertionError("saturated polling interval evaluated an extreme exponent")


class ExtremeAttempt(int):
    def __sub__(self, other):
        return ExponentMustNotBeEvaluated(int(self) - other)


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

    async def finalized_with_distinct_evidence(self):
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(
            observation_pair(first_at, stock_hash="1", etf_hash="2"),
            observation_pair(second_at, stock_hash="3", etf_hash="4"),
        )
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)
        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.advance(5.15)
        final = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")
        return acquisition, final

    async def test_checkpoint_deadline_is_fixed_from_official_close_not_restart_time(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=137))
        acquisition = self.acquisition(clock, ScriptedPairProvider())

        checkpoint = self.new_checkpoint(acquisition)

        assert checkpoint.deadline_utc == OFFICIAL_CLOSE + timedelta(seconds=600)
        assert checkpoint.next_poll_utc == OFFICIAL_CLOSE
        assert checkpoint.attempt == 0
        assert checkpoint.confirmation_count == 0
        assert checkpoint.revision == 1

    async def test_polling_starts_at_official_close_and_collects_confirmations_during_finalize_delay(self):
        first_at = OFFICIAL_CLOSE
        second_at = OFFICIAL_CLOSE + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)
        checkpoint = self.new_checkpoint(acquisition)

        first = await acquisition.advance(checkpoint, stock_symbol="SNDK", etf_symbol="SNXX")
        clock.advance(5.15)
        second = await acquisition.advance(first.checkpoint, stock_symbol="SNDK", etf_symbol="SNXX")

        assert first.status is AnchorAcquisitionStatus.POLLING
        assert first.checkpoint.confirmation_count == 1
        assert second.status is AnchorAcquisitionStatus.POLLING
        assert second.candidate is None
        assert second.checkpoint.confirmation_count == self.nav_config.anchor_confirmation_count
        assert second.checkpoint.next_poll_utc == OFFICIAL_CLOSE + timedelta(
            seconds=self.nav_config.anchor_min_finalize_delay_seconds
        )
        assert len(provider.calls) == 4

    async def test_confirmed_preboundary_checkpoint_waits_then_finalizes_at_boundary_without_reset(self):
        first_at = OFFICIAL_CLOSE
        second_at = OFFICIAL_CLOSE + timedelta(seconds=5.15)
        boundary = OFFICIAL_CLOSE + timedelta(seconds=self.nav_config.anchor_min_finalize_delay_seconds)
        provider = ScriptedPairProvider(
            observation_pair(first_at),
            observation_pair(second_at),
            observation_pair(boundary, etf_delay_seconds=0),
        )
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)

        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.advance(5.15)
        confirmed = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")
        clock.current = boundary - timedelta(microseconds=1)
        clock.monotonic_value += (clock.current - second_at).total_seconds()
        before = await acquisition.advance(confirmed.checkpoint, "SNDK", "SNXX")
        clock.advance(0.000001)
        final = await acquisition.advance(before.checkpoint, "SNDK", "SNXX")

        assert before.status is AnchorAcquisitionStatus.WAITING
        assert before.checkpoint.confirmation_count == self.nav_config.anchor_confirmation_count
        assert len(provider.calls) == 6
        assert final.status is AnchorAcquisitionStatus.FINALIZABLE
        assert final.checkpoint.confirmation_count == self.nav_config.anchor_confirmation_count
        assert final.candidate is not None

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
        first_at = OFFICIAL_CLOSE
        second_at = first_at + timedelta(seconds=5.15)
        final_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        provider = ScriptedPairProvider(
            observation_pair(first_at),
            observation_pair(second_at),
            observation_pair(final_at, etf_delay_seconds=0),
        )
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
        assert clock.sleeps == [5.15, 54.85]
        assert [checkpoint.revision for checkpoint in emitted] == [2, 3, 4]
        assert emitted[-1] is result.checkpoint

    async def test_either_leg_change_resets_paired_confirmation_to_one_current_observation(self):
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

    async def test_identical_pair_received_too_soon_does_not_slide_confirmation_baseline(self):
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

    async def test_asymmetric_leg_delay_over_skew_limit_fails_whole_round_and_clears_candidate(self):
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

    async def test_one_leg_failure_fails_whole_round_and_uses_bounded_exponential_next_poll(self):
        received_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        stock, etf = observation_pair(received_at)
        provider = ScriptedPairProvider((YahooHTTPError("synthetic 503"), etf))
        clock = FakeClock(received_at)
        acquisition = self.acquisition(clock, provider)

        result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        assert result.status is AnchorAcquisitionStatus.POLLING
        assert result.checkpoint.confirmation_count == 0
        assert result.checkpoint.attempt == 1
        assert result.checkpoint.next_poll_utc == etf.received_at_utc + timedelta(seconds=2)
        assert "synthetic 503" in result.failure_reason

    async def test_conservative_remaining_schedule_budget_fails_closed_before_impossible_poll(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=570))
        provider = ScriptedPairProvider()
        acquisition = self.acquisition(clock, provider)

        result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        assert result.status is AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE
        assert result.candidate is None
        assert provider.calls == []
        assert result.failure_reason == "insufficient conservative budget for remaining confirmations"

    async def test_deadline_equality_and_post_deadline_restart_never_issue_late_request(self):
        for seconds_after_close in (600, 601):
            with self.subTest(seconds_after_close=seconds_after_close):
                clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=seconds_after_close))
                provider = ScriptedPairProvider()
                acquisition = self.acquisition(clock, provider)

                result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

                assert result.status is AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE
                assert provider.calls == []
                assert result.checkpoint.deadline_utc == OFFICIAL_CLOSE + timedelta(seconds=600)

    async def test_response_received_after_absolute_deadline_cannot_finalize_late(self):
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

    async def test_restart_uses_prior_checkpoint_confirmation_and_original_remaining_deadline(self):
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

    async def test_one_cycle_reuses_one_wall_and_monotonic_budget_for_both_legs_and_rounds(self):
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)

        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.advance(5.15)
        final = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")

        budgets = [call[3] for call in provider.calls]
        assert final.status is AnchorAcquisitionStatus.FINALIZABLE
        assert budgets and all(budget is not None for budget in budgets)
        assert len({id(budget) for budget in budgets}) == 1

    async def test_backward_wall_adjustment_cannot_restore_an_exhausted_cycle_monotonic_budget(self):
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)

        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.current = second_at
        clock.monotonic_value += 541
        exhausted = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")

        assert exhausted.status is AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE
        assert exhausted.candidate is None
        assert len(provider.calls) == 2

    async def test_pair_gather_completion_at_monotonic_deadline_equality_fails_closed(self):
        received_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        stock, etf = observation_pair(received_at)
        clock = FakeClock(received_at)
        provider = DeadlineEqualityPairProvider(
            clock,
            stock,
            etf,
            advance_seconds=540,
        )
        acquisition = self.acquisition(clock, provider)

        result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        assert result.status is AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE
        assert result.candidate is None

    async def test_cancelled_pair_leg_propagates_and_cancels_the_sibling_leg(self):
        received_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        stock, etf = observation_pair(received_at)
        provider = CancellingPairProvider(stock, etf)
        clock = FakeClock(received_at)
        acquisition = self.acquisition(clock, provider)

        with self.assertRaises(asyncio.CancelledError):
            await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        assert provider.sibling_cancelled is True

    async def test_pair_skew_uses_monotonic_completion_while_utc_receipts_remain_audit_evidence(self):
        received_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        stock, etf = observation_pair(received_at, etf_delay_seconds=0)
        clock = FakeClock(received_at)
        provider = MonotonicSkewProvider(
            clock,
            stock,
            etf,
            self.nav_config.anchor_pair_fetch_max_skew_seconds + 0.000001,
        )
        acquisition = self.acquisition(clock, provider)

        result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        assert result.status is AnchorAcquisitionStatus.POLLING
        assert result.checkpoint.confirmation_count == 0
        assert "monotonic" in result.failure_reason

    async def test_checkpoint_callback_completion_at_deadline_equality_fails_closed(self):
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)
        emitted = []

        async def checkpoint_callback(checkpoint):
            emitted.append(checkpoint)
            if checkpoint.confirmation_count == self.nav_config.anchor_confirmation_count:
                clock.monotonic_value += 600

        result = await acquisition.run(
            self.new_checkpoint(acquisition),
            "SNDK",
            "SNXX",
            on_checkpoint=checkpoint_callback,
        )

        assert result.status is AnchorAcquisitionStatus.ANCHOR_UNAVAILABLE
        assert result.candidate is None
        assert [checkpoint.confirmation_count for checkpoint in emitted] == [1, 2]

    async def test_restart_from_already_confirmed_checkpoint_is_saturated_and_repeatable(self):
        first_at = OFFICIAL_CLOSE + timedelta(seconds=60)
        second_at = first_at + timedelta(seconds=5.15)
        initial_provider = ScriptedPairProvider(observation_pair(first_at), observation_pair(second_at))
        initial_clock = FakeClock(first_at)
        initial = self.acquisition(initial_clock, initial_provider)
        first = await initial.advance(self.new_checkpoint(initial), "SNDK", "SNXX")
        initial_clock.advance(5.15)
        confirmed = await initial.advance(first.checkpoint, "SNDK", "SNXX")
        assert confirmed.checkpoint.confirmation_count == self.nav_config.anchor_confirmation_count

        replay_at = confirmed.checkpoint.next_poll_utc
        replay_clock = FakeClock(replay_at, monotonic_value=42)
        replay_provider = ScriptedPairProvider(
            observation_pair(replay_at, etf_delay_seconds=0),
            observation_pair(replay_at + timedelta(seconds=15), etf_delay_seconds=0),
        )
        restarted = self.acquisition(replay_clock, replay_provider)
        replayed = await restarted.advance(confirmed.checkpoint, "SNDK", "SNXX")

        assert replayed.status is AnchorAcquisitionStatus.FINALIZABLE
        assert replayed.checkpoint.confirmation_count == self.nav_config.anchor_confirmation_count
        persisted = AnchorPollingCheckpoint.from_recovery_fields(replayed.checkpoint.to_recovery_fields())
        assert persisted.confirmation_evidence == confirmed.checkpoint.confirmation_evidence
        replay_clock.current = persisted.next_poll_utc
        replay_clock.monotonic_value += 15
        replayed_again = await restarted.advance(persisted, "SNDK", "SNXX")
        assert replayed_again.status is AnchorAcquisitionStatus.FINALIZABLE
        assert replayed_again.checkpoint.confirmation_count == self.nav_config.anchor_confirmation_count

    async def test_checkpoint_creation_is_bound_to_exact_xnys_regular_dst_and_early_closes(self):
        clock = FakeClock(OFFICIAL_CLOSE)
        acquisition = self.acquisition(clock, ScriptedPairProvider())
        valid_sessions = (
            (date(2026, 1, 15), datetime(2026, 1, 15, 21, tzinfo=UTC)),
            (date(2026, 7, 17), datetime(2026, 7, 17, 20, tzinfo=UTC)),
            (date(2026, 11, 27), datetime(2026, 11, 27, 18, tzinfo=UTC)),
        )
        for session_date, official_close in valid_sessions:
            with self.subTest(session_date=session_date):
                checkpoint = acquisition.create_checkpoint(
                    f"xnys-{session_date.isoformat()}",
                    session_date,
                    official_close,
                )
                assert checkpoint.official_close_utc == official_close

        invalid_sessions = (
            (date(2026, 1, 15), datetime(2026, 1, 15, 20, tzinfo=UTC)),
            (date(2026, 7, 17), datetime(2026, 7, 17, 21, tzinfo=UTC)),
            (date(2026, 11, 27), datetime(2026, 11, 27, 21, tzinfo=UTC)),
            (date(2026, 1, 19), datetime(2026, 1, 19, 21, tzinfo=UTC)),
        )
        for session_date, official_close in invalid_sessions:
            with self.subTest(session_date=session_date, official_close=official_close):
                with self.assertRaises(CheckpointIntegrityError):
                    acquisition.create_checkpoint(
                        f"xnys-{session_date.isoformat()}",
                        session_date,
                        official_close,
                    )

    async def test_loaded_checkpoint_with_non_xnys_close_is_rejected_before_any_fetch(self):
        wrong_close = OFFICIAL_CLOSE + timedelta(hours=1)
        checkpoint = AnchorPollingCheckpoint(
            schema_version=1,
            cycle_id="xnys-2026-07-17",
            target_session_date=TARGET_SESSION_DATE,
            official_close_utc=wrong_close,
            deadline_utc=wrong_close + timedelta(seconds=600),
            attempt=0,
            next_poll_utc=wrong_close,
            confirmation_count=0,
            candidate_stock_close=None,
            candidate_etf_close=None,
            candidate_stock_raw_response_hash=None,
            candidate_etf_raw_response_hash=None,
            stock_received_at_utc=None,
            etf_received_at_utc=None,
            revision=1,
        )
        provider = ScriptedPairProvider()
        acquisition = self.acquisition(FakeClock(wrong_close), provider)

        with self.assertRaises(CheckpointIntegrityError):
            await acquisition.advance(checkpoint, "SNDK", "SNXX")

        assert provider.calls == []

    async def test_preclose_observations_fail_closed_instead_of_entering_confirmation_state(self):
        preclose = OFFICIAL_CLOSE - timedelta(microseconds=1)
        provider = ScriptedPairProvider(observation_pair(preclose, etf_delay_seconds=0))
        clock = FakeClock(OFFICIAL_CLOSE)
        acquisition = self.acquisition(clock, provider)

        result = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")

        assert result.status is AnchorAcquisitionStatus.POLLING
        assert result.checkpoint.confirmation_count == 0
        assert result.checkpoint.candidate_stock_close is None
        assert "acquisition window" in result.failure_reason

    async def test_extreme_attempt_uses_saturated_poll_interval_without_exponentiation(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        acquisition = self.acquisition(clock, ScriptedPairProvider())

        interval = acquisition._poll_interval(ExtremeAttempt(1_000_000))

        assert interval == self.nav_config.anchor_poll_max_interval_seconds

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

    async def test_post_finalization_price_revision_is_append_only_observation_not_replacement(self):
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

    async def test_post_finalization_same_closes_do_not_alert_when_irrelevant_raw_fields_change(self):
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

    async def test_candidate_binds_full_ordered_confirmation_trail_and_round_trips_adapter_fields(self):
        _, final = await self.finalized_with_distinct_evidence()

        candidate = final.candidate
        assert candidate.evidence_version == 2
        assert candidate.pair_id == "sndk_snxx"
        assert candidate.anchor_source == self.nav_config.anchor_source
        assert candidate.official_close_utc == OFFICIAL_CLOSE
        assert candidate.acquisition_started_at_utc == OFFICIAL_CLOSE
        assert candidate.etf_daily_multiplier == Decimal("2")
        assert candidate.hedge_ratio == Decimal("0.24")
        assert candidate.anchor_confirmation_count == self.nav_config.anchor_confirmation_count
        assert candidate.observed_confirmation_count == self.nav_config.anchor_confirmation_count
        assert len(candidate.confirmation_evidence) == 2
        assert [record.confirmation_index for record in candidate.confirmation_evidence] == [1, 2]
        assert [record.stock_raw_response_hash for record in candidate.confirmation_evidence] == [
            "1" * 64,
            "3" * 64,
        ]
        assert [record.etf_raw_response_hash for record in candidate.confirmation_evidence] == [
            "2" * 64,
            "4" * 64,
        ]
        assert all(record.stock_source_url.endswith("/SNDK") for record in candidate.confirmation_evidence)
        assert all(record.etf_source_url.endswith("/SNXX") for record in candidate.confirmation_evidence)
        assert candidate.verify_evidence_hash()

        persisted = candidate.to_evidence_fields()
        restored = type(candidate).from_evidence_fields(persisted)

        assert restored == candidate
        assert restored.to_evidence_fields() == persisted

    async def test_candidate_hash_binds_every_decision_field_before_revision_classification(self):
        acquisition, final = await self.finalized_with_distinct_evidence()
        candidate = final.candidate
        current_stock, current_etf = observation_pair(candidate.finalized_at_utc + timedelta(seconds=1))
        corruptions = (
            ("pair identity", {"pair_id": "intc_intw"}),
            ("symbol", {"stock_symbol": "INTC"}),
            ("stock anchor", {"stock_close": Decimal("251")}),
            ("ETF anchor", {"etf_close": Decimal("31")}),
            (
                "session date",
                {
                    "cycle_id": "xnys-2026-07-16",
                    "target_session_date": date(2026, 7, 16),
                },
            ),
            ("multiplier", {"etf_daily_multiplier": Decimal("3")}),
            ("derived hedge", {"hedge_ratio": Decimal("0.25")}),
            ("official close", {"official_close_utc": candidate.official_close_utc + timedelta(seconds=1)}),
            ("acquisition time", {"acquisition_started_at_utc": candidate.acquisition_started_at_utc + timedelta(seconds=1)}),
            ("finalization time", {"finalized_at_utc": candidate.finalized_at_utc + timedelta(seconds=1)}),
            ("deadline", {"deadline_utc": candidate.deadline_utc - timedelta(seconds=1)}),
            ("required count", {"anchor_confirmation_count": candidate.anchor_confirmation_count + 1}),
            ("observed count", {"observed_confirmation_count": candidate.observed_confirmation_count - 1}),
            (
                "confirmation interval",
                {"anchor_confirmation_interval_seconds": candidate.anchor_confirmation_interval_seconds + 1},
            ),
            (
                "minimum finalize delay",
                {"anchor_min_finalize_delay_seconds": candidate.anchor_min_finalize_delay_seconds + 1},
            ),
            (
                "pair skew",
                {"anchor_pair_fetch_max_skew_seconds": candidate.anchor_pair_fetch_max_skew_seconds + 1},
            ),
            ("evidence order", {"confirmation_evidence": tuple(reversed(candidate.confirmation_evidence))}),
            ("evidence drop", {"confirmation_evidence": candidate.confirmation_evidence[1:]}),
            (
                "evidence duplicate",
                {"confirmation_evidence": candidate.confirmation_evidence + candidate.confirmation_evidence[-1:]},
            ),
            (
                "evidence mutation",
                {
                    "confirmation_evidence": (
                        replace(candidate.confirmation_evidence[0], stock_source_url="https://example.invalid/SNDK"),
                        candidate.confirmation_evidence[1],
                    )
                },
            ),
        )

        for label, changes in corruptions:
            with self.subTest(label=label):
                corrupted = replace(candidate, **changes)
                assert not corrupted.verify_evidence_hash()
                assessment = acquisition.assess_finalized(corrupted, current_stock, current_etf)
                assert assessment.status is FinalizedAnchorStatus.RECOVERY_REQUIRED
                assert assessment.revision_observation is None

    async def test_candidate_adapter_rejects_reorder_drop_duplicate_and_mutation_with_old_hash(self):
        _, final = await self.finalized_with_distinct_evidence()
        candidate = final.candidate
        fields = candidate.to_evidence_fields()
        mutations = []

        reordered = copy.deepcopy(fields)
        reordered["confirmation_evidence"].reverse()
        mutations.append(("reordered", reordered))
        dropped = copy.deepcopy(fields)
        dropped["confirmation_evidence"].pop(0)
        mutations.append(("dropped", dropped))
        duplicated = copy.deepcopy(fields)
        duplicated["confirmation_evidence"].append(copy.deepcopy(duplicated["confirmation_evidence"][-1]))
        mutations.append(("duplicated", duplicated))
        changed_hash = copy.deepcopy(fields)
        changed_hash["confirmation_evidence"][0]["stock_raw_response_hash"] = "a" * 64
        mutations.append(("raw hash", changed_hash))
        changed_provenance = copy.deepcopy(fields)
        changed_provenance["confirmation_evidence"][0]["stock_source_url"] = "https://example.invalid/SNDK"
        mutations.append(("provenance", changed_provenance))

        for label, corrupted_fields in mutations:
            with self.subTest(label=label):
                with self.assertRaises(CheckpointIntegrityError):
                    type(candidate).from_evidence_fields(corrupted_fields)

    async def test_checkpoint_recovery_adapter_retains_and_integrity_binds_crash_state(self):
        first_at = OFFICIAL_CLOSE
        second_at = first_at + timedelta(seconds=5.15)
        provider = ScriptedPairProvider(
            observation_pair(first_at, stock_hash="1", etf_hash="2"),
            observation_pair(second_at, stock_hash="3", etf_hash="4"),
        )
        clock = FakeClock(first_at)
        acquisition = self.acquisition(clock, provider)
        first = await acquisition.advance(self.new_checkpoint(acquisition), "SNDK", "SNXX")
        clock.advance(5.15)
        confirmed = await acquisition.advance(first.checkpoint, "SNDK", "SNXX")
        assert confirmed.status is AnchorAcquisitionStatus.POLLING
        assert confirmed.checkpoint.confirmation_count == 2

        recovery_fields = confirmed.checkpoint.to_recovery_fields()
        restored = AnchorPollingCheckpoint.from_recovery_fields(recovery_fields)

        assert restored == confirmed.checkpoint
        assert len(restored.confirmation_evidence) == 2
        corruptions = []
        reordered = copy.deepcopy(recovery_fields)
        reordered["confirmation_evidence"].reverse()
        corruptions.append(reordered)
        dropped = copy.deepcopy(recovery_fields)
        dropped["confirmation_evidence"].pop()
        corruptions.append(dropped)
        duplicated = copy.deepcopy(recovery_fields)
        duplicated["confirmation_evidence"].append(copy.deepcopy(duplicated["confirmation_evidence"][-1]))
        corruptions.append(duplicated)
        mutated = copy.deepcopy(recovery_fields)
        mutated["confirmation_evidence"][0]["etf_source_url"] = "https://example.invalid/SNXX"
        corruptions.append(mutated)
        for corrupted in corruptions:
            with self.assertRaises(CheckpointIntegrityError):
                AnchorPollingCheckpoint.from_recovery_fields(corrupted)

    async def test_crash_before_finalize_and_repeated_finalize_are_full_trail_idempotent(self):
        first_at = OFFICIAL_CLOSE
        second_at = first_at + timedelta(seconds=5.15)
        boundary = OFFICIAL_CLOSE + timedelta(seconds=self.nav_config.anchor_min_finalize_delay_seconds)
        initial_provider = ScriptedPairProvider(
            observation_pair(first_at, stock_hash="1", etf_hash="2"),
            observation_pair(second_at, stock_hash="3", etf_hash="4"),
        )
        initial_clock = FakeClock(first_at)
        initial = self.acquisition(initial_clock, initial_provider)
        first = await initial.advance(self.new_checkpoint(initial), "SNDK", "SNXX")
        initial_clock.advance(5.15)
        confirmed = await initial.advance(first.checkpoint, "SNDK", "SNXX")
        persisted = confirmed.checkpoint.to_recovery_fields()

        candidates = []
        for offset, hash_pair in ((0, ("5", "6")), (10, ("7", "8"))):
            restart_at = boundary + timedelta(seconds=offset)
            restart_clock = FakeClock(restart_at, monotonic_value=100 + offset)
            restart_provider = ScriptedPairProvider(
                observation_pair(
                    restart_at,
                    stock_hash=hash_pair[0],
                    etf_hash=hash_pair[1],
                    etf_delay_seconds=0,
                )
            )
            restarted = self.acquisition(restart_clock, restart_provider)
            restored = AnchorPollingCheckpoint.from_recovery_fields(copy.deepcopy(persisted))
            finalized = await restarted.advance(restored, "SNDK", "SNXX")
            assert finalized.status is AnchorAcquisitionStatus.FINALIZABLE
            assert len(finalized.candidate.confirmation_evidence) == 2
            candidates.append(finalized.candidate)

        assert candidates[0] == candidates[1]
        assert candidates[0].finalized_at_utc == boundary
        assert [record.stock_raw_response_hash for record in candidates[0].confirmation_evidence] == [
            "1" * 64,
            "3" * 64,
        ]
