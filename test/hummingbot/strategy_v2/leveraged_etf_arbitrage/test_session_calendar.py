import copy
import tomllib
from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path

import pytest

from hummingbot.strategy_v2.leveraged_etf_arbitrage.calendar import (
    EXCHANGE_CALENDARS_VERSION,
    AmbiguousLocalTimeError,
    CalendarRangeError,
    NonexistentLocalTimeError,
    SessionStage,
    XnysNavCalendar,
    exchange_calendar_code,
)
from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import (
    EquityLeveragedEtfArbitrageConfig,
    SessionName,
)


UTC = timezone.utc


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=UTC)


_NORMAL_CLOSE = _utc(2026, 1, 15, 21)
_NORMAL_MAKER_EXIT_START = _NORMAL_CLOSE - timedelta(minutes=30)
_NORMAL_FORCE_MARKET_START = _NORMAL_CLOSE - timedelta(minutes=1)
_EARLY_CLOSE = _utc(2026, 11, 27, 18)
_EARLY_MAKER_EXIT_START = _EARLY_CLOSE - timedelta(minutes=30)
_EARLY_FORCE_MARKET_START = _EARLY_CLOSE - timedelta(minutes=1)


def _example_path() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "config" / "equity_leveraged_etf_arbitrage.example.toml"
        if candidate.is_file():
            return candidate
    raise AssertionError("committed example TOML not found from test path")


@pytest.fixture
def config() -> EquityLeveragedEtfArbitrageConfig:
    with _example_path().open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    return EquityLeveragedEtfArbitrageConfig.model_validate(raw_config)


class StubClock:
    def __init__(self, current: datetime):
        self.current = current

    def __call__(self) -> datetime:
        return self.current


def test_dependency_is_exactly_pinned_for_reproducible_xnys_rules():
    assert EXCHANGE_CALENDARS_VERSION == "4.13.2"
    assert version("exchange-calendars") == EXCHANGE_CALENDARS_VERSION


def test_us_equities_maps_only_to_official_xnys_calendar():
    assert exchange_calendar_code("US_EQUITIES") == "XNYS"

    with pytest.raises(ValueError, match="unsupported equity calendar"):
        exchange_calendar_code("XNAS")
    with pytest.raises(TypeError, match="calendar name must be a str"):
        exchange_calendar_code(None)


@pytest.mark.parametrize(
    (
        "observed_at",
        "session_date",
        "official_open",
        "official_close",
        "cycle_session_date",
    ),
    [
        # Ordinary EST and EDT sessions.
        (_utc(2026, 1, 15, 16), date(2026, 1, 15), _utc(2026, 1, 15, 14, 30), _utc(2026, 1, 15, 21), date(2026, 1, 14)),
        (_utc(2026, 7, 15, 16), date(2026, 7, 15), _utc(2026, 7, 15, 13, 30), _utc(2026, 7, 15, 20), date(2026, 7, 14)),
        # The UTC schedule changes on the first session after each DST transition.
        (_utc(2026, 3, 6, 17), date(2026, 3, 6), _utc(2026, 3, 6, 14, 30), _utc(2026, 3, 6, 21), date(2026, 3, 5)),
        (_utc(2026, 3, 9, 17), date(2026, 3, 9), _utc(2026, 3, 9, 13, 30), _utc(2026, 3, 9, 20), date(2026, 3, 6)),
        (
            _utc(2026, 10, 30, 17),
            date(2026, 10, 30),
            _utc(2026, 10, 30, 13, 30),
            _utc(2026, 10, 30, 20),
            date(2026, 10, 29),
        ),
        (
            _utc(2026, 11, 2, 17),
            date(2026, 11, 2),
            _utc(2026, 11, 2, 14, 30),
            _utc(2026, 11, 2, 21),
            date(2026, 10, 30),
        ),
        # The day after Thanksgiving is an official 13:00 America/New_York close.
        (
            _utc(2026, 11, 27, 16),
            date(2026, 11, 27),
            _utc(2026, 11, 27, 14, 30),
            _utc(2026, 11, 27, 18),
            date(2026, 11, 25),
        ),
    ],
)
def test_official_clock_table_tracks_est_edt_transitions_and_early_close(
    config: EquityLeveragedEtfArbitrageConfig,
    observed_at: datetime,
    session_date: date,
    official_open: datetime,
    official_close: datetime,
    cycle_session_date: date,
):
    snapshot = XnysNavCalendar(config.nav, clock=lambda: observed_at).snapshot()

    assert snapshot.observed_at_utc == observed_at
    assert snapshot.calendar_code == "XNYS"
    assert snapshot.session_date == session_date
    assert snapshot.official_open_utc == official_open
    assert snapshot.official_close_utc == official_close
    assert snapshot.cycle_session_date == cycle_session_date
    assert snapshot.cycle_id == f"xnys-{cycle_session_date.isoformat()}"
    assert snapshot.cycle_started_at_utc < observed_at
    assert snapshot.logical_session is SessionName.REGULAR
    assert snapshot.stage is SessionStage.NORMAL
    assert snapshot.regular_session_open
    assert snapshot.calendar_entry_allowed
    assert not snapshot.maker_exit_required
    assert not snapshot.force_market_flatten_required


@pytest.mark.parametrize(
    ("observed_at", "next_session", "next_open", "next_close", "active_cycle"),
    [
        # Saturday resolves to Monday without inventing a weekend session.
        (_utc(2026, 1, 10, 12), date(2026, 1, 12), _utc(2026, 1, 12, 14, 30), _utc(2026, 1, 12, 21), date(2026, 1, 9)),
        # Martin Luther King Jr. Day is a full XNYS holiday.
        (_utc(2026, 1, 19, 17), date(2026, 1, 20), _utc(2026, 1, 20, 14, 30), _utc(2026, 1, 20, 21), date(2026, 1, 16)),
    ],
)
def test_weekend_and_full_holiday_keep_the_last_closed_cycle_and_next_official_session(
    config: EquityLeveragedEtfArbitrageConfig,
    observed_at: datetime,
    next_session: date,
    next_open: datetime,
    next_close: datetime,
    active_cycle: date,
):
    snapshot = XnysNavCalendar(config.nav, clock=lambda: observed_at).snapshot()

    assert snapshot.logical_session is SessionName.WEEKEND_HOLIDAY
    assert not snapshot.regular_session_open
    assert snapshot.session_date == next_session
    assert snapshot.official_open_utc == next_open
    assert snapshot.official_close_utc == next_close
    assert snapshot.cycle_session_date == active_cycle
    assert snapshot.cycle_id == f"xnys-{active_cycle.isoformat()}"
    assert snapshot.stage is SessionStage.NORMAL
    assert snapshot.calendar_entry_allowed


@pytest.mark.parametrize(
    (
        "observed_at",
        "stage",
        "entry_allowed",
        "maker_exit",
        "force_market",
        "session_date",
        "cycle_session_date",
    ),
    [
        (
            _NORMAL_MAKER_EXIT_START - timedelta(microseconds=1),
            SessionStage.NORMAL,
            True,
            False,
            False,
            date(2026, 1, 15),
            date(2026, 1, 14),
        ),
        (
            _NORMAL_MAKER_EXIT_START,
            SessionStage.MAKER_EXIT,
            False,
            True,
            False,
            date(2026, 1, 15),
            date(2026, 1, 14),
        ),
        (
            _NORMAL_FORCE_MARKET_START - timedelta(microseconds=1),
            SessionStage.MAKER_EXIT,
            False,
            True,
            False,
            date(2026, 1, 15),
            date(2026, 1, 14),
        ),
        (
            _NORMAL_FORCE_MARKET_START,
            SessionStage.FORCED_MARKET_FLATTEN,
            False,
            False,
            True,
            date(2026, 1, 15),
            date(2026, 1, 14),
        ),
        (
            _NORMAL_CLOSE - timedelta(microseconds=1),
            SessionStage.FORCED_MARKET_FLATTEN,
            False,
            False,
            True,
            date(2026, 1, 15),
            date(2026, 1, 14),
        ),
        # Close equality belongs to the new cycle and the next close becomes Friday's.
        (_NORMAL_CLOSE, SessionStage.NORMAL, True, False, False, date(2026, 1, 16), date(2026, 1, 15)),
        (
            _NORMAL_CLOSE + timedelta(microseconds=1),
            SessionStage.NORMAL,
            True,
            False,
            False,
            date(2026, 1, 16),
            date(2026, 1, 15),
        ),
    ],
)
def test_close_boundaries_are_microsecond_exact_and_close_equality_rolls_cycle(
    config: EquityLeveragedEtfArbitrageConfig,
    observed_at: datetime,
    stage: SessionStage,
    entry_allowed: bool,
    maker_exit: bool,
    force_market: bool,
    session_date: date,
    cycle_session_date: date,
):
    snapshot = XnysNavCalendar(config.nav, clock=lambda: observed_at).snapshot()

    assert snapshot.stage is stage
    assert snapshot.calendar_entry_allowed is entry_allowed
    assert snapshot.maker_exit_required is maker_exit
    assert snapshot.force_market_flatten_required is force_market
    assert snapshot.session_date == session_date
    assert snapshot.cycle_session_date == cycle_session_date
    assert snapshot.cycle_id == f"xnys-{cycle_session_date.isoformat()}"


def test_early_close_uses_the_same_exact_lead_time_boundaries(config: EquityLeveragedEtfArbitrageConfig):
    calendar = XnysNavCalendar(config.nav, clock=lambda: _EARLY_MAKER_EXIT_START)

    before_maker = calendar.snapshot_at(_EARLY_MAKER_EXIT_START - timedelta(microseconds=1))
    at_maker = calendar.snapshot()
    before_force = calendar.snapshot_at(_EARLY_FORCE_MARKET_START - timedelta(microseconds=1))
    at_force = calendar.snapshot_at(_EARLY_FORCE_MARKET_START)
    before_close = calendar.snapshot_at(_EARLY_CLOSE - timedelta(microseconds=1))
    at_close = calendar.snapshot_at(_EARLY_CLOSE)
    after_close = calendar.snapshot_at(_EARLY_CLOSE + timedelta(microseconds=1))

    assert before_maker.stage is SessionStage.NORMAL
    assert before_maker.calendar_entry_allowed
    assert at_maker.official_close_utc == _EARLY_CLOSE
    assert at_maker.stage is SessionStage.MAKER_EXIT
    assert not at_maker.calendar_entry_allowed
    assert at_maker.maker_exit_required
    assert before_force.stage is SessionStage.MAKER_EXIT
    assert at_force.stage is SessionStage.FORCED_MARKET_FLATTEN
    assert at_force.force_market_flatten_required
    assert before_close.stage is SessionStage.FORCED_MARKET_FLATTEN
    assert at_close.stage is SessionStage.NORMAL
    assert at_close.cycle_session_date == date(2026, 11, 27)
    assert at_close.cycle_started_at_utc == _EARLY_CLOSE
    assert at_close.cycle_id == "xnys-2026-11-27"
    assert after_close.cycle_id == at_close.cycle_id


def test_configured_entry_cutoff_can_precede_the_maker_exit_stage():
    with _example_path().open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    raw_config["nav"]["new_entry_cutoff_minutes"] = 45
    config = EquityLeveragedEtfArbitrageConfig.model_validate(raw_config)
    entry_cutoff = _NORMAL_CLOSE - timedelta(minutes=45)
    calendar = XnysNavCalendar(config.nav, clock=lambda: entry_cutoff)

    at_entry_cutoff = calendar.snapshot()
    before_maker = calendar.snapshot_at(_NORMAL_MAKER_EXIT_START - timedelta(microseconds=1))
    at_maker = calendar.snapshot_at(_NORMAL_MAKER_EXIT_START)
    at_force = calendar.snapshot_at(_NORMAL_FORCE_MARKET_START)

    assert at_entry_cutoff.stage is SessionStage.NEW_ENTRY_CUTOFF
    assert not at_entry_cutoff.calendar_entry_allowed
    assert not at_entry_cutoff.maker_exit_required
    assert not at_entry_cutoff.force_market_flatten_required
    assert before_maker.stage is SessionStage.NEW_ENTRY_CUTOFF
    assert at_maker.stage is SessionStage.MAKER_EXIT
    assert at_maker.maker_exit_required
    assert at_force.stage is SessionStage.FORCED_MARKET_FLATTEN
    assert at_force.force_market_flatten_required


def test_injected_clock_and_snapshots_are_deterministic_and_immutable(config: EquityLeveragedEtfArbitrageConfig):
    clock = StubClock(_utc(2026, 1, 15, 16))
    calendar = XnysNavCalendar(config.nav, clock=clock)

    first = calendar.snapshot()
    assert calendar.snapshot() == first
    clock.current = _utc(2026, 1, 15, 20, 30)
    second = calendar.snapshot()

    assert second.observed_at_utc == clock.current
    assert second.stage is SessionStage.MAKER_EXIT
    with pytest.raises(FrozenInstanceError):
        second.stage = SessionStage.NORMAL


def test_restart_cycle_ids_are_stable_across_weekend_and_roll_only_at_official_close(
    config: EquityLeveragedEtfArbitrageConfig,
):
    friday_close = _utc(2026, 1, 9, 21)
    monday_close = _utc(2026, 1, 12, 21)
    calendar = XnysNavCalendar(
        config.nav,
        clock=lambda: friday_close - timedelta(microseconds=1),
    )

    before_friday_close = calendar.snapshot()
    at_friday_close = calendar.snapshot_at(friday_close)
    sunday_restart = calendar.snapshot_at(_utc(2026, 1, 11, 12))
    before_monday_close = calendar.snapshot_at(monday_close - timedelta(microseconds=1))
    at_monday_close = calendar.snapshot_at(monday_close)

    assert before_friday_close.cycle_id == "xnys-2026-01-08"
    assert at_friday_close.cycle_id == "xnys-2026-01-09"
    assert sunday_restart.cycle_id == at_friday_close.cycle_id
    assert before_monday_close.cycle_id == at_friday_close.cycle_id
    assert at_monday_close.cycle_id == "xnys-2026-01-12"
    assert calendar.snapshot_at(monday_close) == at_monday_close


def test_exact_close_rolls_active_cycle_to_the_f001_vector(config: EquityLeveragedEtfArbitrageConfig):
    vector_close = _utc(2026, 7, 17, 20)
    snapshot = XnysNavCalendar(config.nav, clock=lambda: vector_close).snapshot()

    assert snapshot.observed_at_utc == vector_close
    assert snapshot.cycle_session_date == date(2026, 7, 17)
    assert snapshot.cycle_started_at_utc == vector_close
    assert snapshot.cycle_id == "xnys-2026-07-17"
    assert snapshot.session_date == date(2026, 7, 20)
    assert snapshot.official_open_utc == _utc(2026, 7, 20, 13, 30)
    assert snapshot.official_close_utc == _utc(2026, 7, 20, 20)
    assert snapshot.stage is SessionStage.NORMAL


@pytest.mark.parametrize(
    ("observed_at", "logical_session"),
    [
        (_utc(2026, 1, 15, 12), SessionName.EXTENDED),
        (_utc(2026, 1, 15, 16), SessionName.REGULAR),
        (_utc(2026, 1, 15, 21), SessionName.EXTENDED),
        (_utc(2026, 1, 17, 16), SessionName.WEEKEND_HOLIDAY),
    ],
)
def test_logical_session_and_missing_optional_config_use_complete_regular_fallback(
    config: EquityLeveragedEtfArbitrageConfig,
    observed_at: datetime,
    logical_session: SessionName,
):
    snapshot = XnysNavCalendar(config.nav, clock=lambda: observed_at).snapshot()
    regular = config.pairs[0].sessions[SessionName.REGULAR]
    resolved = snapshot.resolve_session(config, config.pairs[0].id)

    assert snapshot.logical_session is logical_session
    assert resolved.session_name is SessionName.REGULAR
    assert resolved.p99_bp == regular.p99_bp
    assert resolved.historical_max_bp == regular.historical_max_bp
    assert resolved.position_tiers is regular.position_tiers
    assert resolved.reduce_bp_by_current_target is regular.reduce_bp_by_current_target
    assert resolved.divergence_cancel_bp == regular.divergence_cancel_bp
    assert resolved.divergence_confirmations == regular.divergence_confirmations
    assert resolved.entry_confirmations == regular.entry_confirmations
    assert resolved.entry_confirmation_interval_ms == regular.entry_confirmation_interval_ms


def test_present_optional_session_is_selected_instead_of_regular_fallback():
    with _example_path().open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    raw_config["pairs"][0]["sessions"]["weekend_holiday"] = copy.deepcopy(
        raw_config["pairs"][0]["sessions"]["regular"]
    )
    raw_config["pairs"][0]["sessions"]["weekend_holiday"]["p99_bp"] = "151"
    raw_config["pairs"][0]["sessions"]["weekend_holiday"]["historical_max_bp"] = "176"
    config = EquityLeveragedEtfArbitrageConfig.model_validate(raw_config)
    snapshot = XnysNavCalendar(config.nav, clock=lambda: _utc(2026, 1, 17, 16)).snapshot()

    resolved = snapshot.resolve_session(config, config.pairs[0].id)

    assert resolved.session_name is SessionName.WEEKEND_HOLIDAY
    assert resolved.p99_bp == config.pairs[0].sessions[SessionName.WEEKEND_HOLIDAY].p99_bp


def test_snapshot_rejects_naive_datetimes(config: EquityLeveragedEtfArbitrageConfig):
    calendar = XnysNavCalendar(config.nav, clock=lambda: datetime(2026, 1, 15, 16))

    with pytest.raises(ValueError, match="timezone-aware UTC"):
        calendar.snapshot()
    with pytest.raises(ValueError, match="timezone-aware UTC"):
        calendar.snapshot_at(datetime(2026, 1, 15, 16))


def test_local_wall_clock_conversion_rejects_nonexistent_and_ambiguous_times(
    config: EquityLeveragedEtfArbitrageConfig,
):
    calendar = XnysNavCalendar(config.nav, clock=lambda: _utc(2026, 7, 15, 15))

    with pytest.raises(NonexistentLocalTimeError, match="does not exist"):
        calendar.snapshot_at_local(datetime(2026, 3, 8, 2, 30))
    with pytest.raises(AmbiguousLocalTimeError, match="ambiguous"):
        calendar.snapshot_at_local(datetime(2026, 11, 1, 1, 30))
    with pytest.raises(ValueError, match="naive America/New_York wall time"):
        calendar.snapshot_at_local(_utc(2026, 7, 15, 15))

    assert calendar.snapshot_at_local(datetime(2026, 7, 15, 11)) == calendar.snapshot_at(
        _utc(2026, 7, 15, 15)
    )


@pytest.mark.parametrize(
    (
        "observed_at",
        "next_session",
        "next_open",
        "next_close",
        "cycle_session",
        "cycle_started_at",
        "cycle_id",
    ),
    [
        (
            _utc(2000, 1, 1, 12),
            date(2000, 1, 3),
            _utc(2000, 1, 3, 14, 30),
            _utc(2000, 1, 3, 21),
            date(1999, 12, 31),
            _utc(1999, 12, 31, 18),
            "xnys-1999-12-31",
        ),
        (
            _utc(2100, 12, 31, 22),
            date(2101, 1, 3),
            _utc(2101, 1, 3, 14, 30),
            _utc(2101, 1, 3, 21),
            date(2100, 12, 31),
            _utc(2100, 12, 31, 21),
            "xnys-2100-12-31",
        ),
    ],
)
def test_supported_range_edges_are_inclusive_with_buffered_previous_and_next_sessions(
    config: EquityLeveragedEtfArbitrageConfig,
    observed_at: datetime,
    next_session: date,
    next_open: datetime,
    next_close: datetime,
    cycle_session: date,
    cycle_started_at: datetime,
    cycle_id: str,
):
    snapshot = XnysNavCalendar(config.nav, clock=lambda: observed_at).snapshot()

    assert snapshot.observed_at_utc == observed_at
    assert snapshot.session_date == next_session
    assert snapshot.official_open_utc == next_open
    assert snapshot.official_close_utc == next_close
    assert snapshot.cycle_session_date == cycle_session
    assert snapshot.cycle_started_at_utc == cycle_started_at
    assert snapshot.cycle_id == cycle_id


@pytest.mark.parametrize(
    "observed_at",
    [
        _utc(1999, 12, 31, 20),
        _utc(2101, 1, 1, 20),
    ],
)
def test_snapshot_rejects_observations_outside_the_explicit_supported_range(
    config: EquityLeveragedEtfArbitrageConfig,
    observed_at: datetime,
):
    calendar = XnysNavCalendar(config.nav, clock=lambda: observed_at)

    with pytest.raises(CalendarRangeError, match="outside supported XNYS range"):
        calendar.snapshot()


def test_datetime_with_nonzero_offset_is_normalized_to_utc(config: EquityLeveragedEtfArbitrageConfig):
    observed_at = _utc(2026, 7, 15, 15).astimezone(timezone(timedelta(hours=-4)))
    snapshot = XnysNavCalendar(config.nav, clock=lambda: observed_at).snapshot()

    assert snapshot.observed_at_utc == _utc(2026, 7, 15, 15)
