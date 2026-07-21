"""Official XNYS session boundaries and deterministic NAV-cycle snapshots.

``exchange-calendars`` is intentionally exact-pinned in ``setup.py``. Its XNYS
rules avoid a hand-maintained holiday table that would silently expire, while
the pin and explicit supported range keep persisted cycle identifiers stable.
"""

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from typing import Final
from zoneinfo import ZoneInfo

import exchange_calendars
import pandas as pd

from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import (
    EquityLeveragedEtfArbitrageConfig,
    NavConfig,
    ResolvedSessionConfig,
    SessionName,
)


EXCHANGE_CALENDARS_VERSION: Final = "4.13.2"
XNYS_CALENDAR_CODE: Final = "XNYS"
NEW_YORK_TIMEZONE_NAME: Final = "America/New_York"
SUPPORTED_START_DATE: Final = date(2000, 1, 1)
SUPPORTED_END_DATE: Final = date(2100, 12, 31)

_UTC = timezone.utc
_NEW_YORK = ZoneInfo(NEW_YORK_TIMEZONE_NAME)
_INTERNAL_START_DATE = date(1999, 12, 15)
_INTERNAL_END_DATE = date(2101, 1, 15)


class CalendarRangeError(ValueError):
    """Raised when an observation is outside the deliberately bounded schedule."""


class NonexistentLocalTimeError(ValueError):
    """Raised for a local wall time skipped by the spring DST transition."""


class AmbiguousLocalTimeError(ValueError):
    """Raised for a local wall time repeated by the autumn DST transition."""


class SessionStage(str, Enum):
    """Calendar-only operational stage before the next official XNYS close."""

    NORMAL = "NORMAL"
    NEW_ENTRY_CUTOFF = "NEW_ENTRY_CUTOFF"
    MAKER_EXIT = "MAKER_EXIT"
    FORCED_MARKET_FLATTEN = "FORCED_MARKET_FLATTEN"


def exchange_calendar_code(calendar_name: str) -> str:
    """Map the strategy's sole supported calendar name to its official MIC."""

    if not isinstance(calendar_name, str):
        raise TypeError("calendar name must be a str")
    if calendar_name != "US_EQUITIES":
        raise ValueError(f"unsupported equity calendar: {calendar_name}")
    return XNYS_CALENDAR_CODE


def _to_utc_datetime(value: pd.Timestamp) -> datetime:
    converted = value.to_pydatetime(warn=False)
    if converted.tzinfo is None:
        converted = converted.replace(tzinfo=_UTC)
    return converted.astimezone(_UTC)


def _normalize_utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("observation must be a datetime")
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("observation must be a timezone-aware UTC instant")
    return value.astimezone(_UTC)


def _new_york_wall_time_to_utc(value: object) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError("local observation must be a datetime")
    if value.tzinfo is not None:
        raise ValueError("local observation must be a naive America/New_York wall time")

    candidates: set[datetime] = set()
    for fold in (0, 1):
        localized = value.replace(tzinfo=_NEW_YORK, fold=fold)
        candidate = localized.astimezone(_UTC)
        round_trip = candidate.astimezone(_NEW_YORK)
        if round_trip.replace(tzinfo=None) == value and round_trip.fold == fold:
            candidates.add(candidate)

    if not candidates:
        raise NonexistentLocalTimeError(
            f"America/New_York local time {value.isoformat()} does not exist"
        )
    if len(candidates) != 1:
        raise AmbiguousLocalTimeError(
            f"America/New_York local time {value.isoformat()} is ambiguous"
        )
    return candidates.pop()


@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """Immutable view of the active NAV cycle and its next close boundary.

    ``session_date`` and the official open/close refer to the session whose
    close is strictly after ``observed_at_utc``. ``cycle_session_date`` refers
    to the most recent session whose close is at or before the observation.
    Consequently, exact close equality belongs to the newly started cycle.

    ``calendar_entry_allowed`` is only the close-window gate. A Controller must
    additionally require a finalized anchor for ``cycle_id`` before opening.
    """

    observed_at_utc: datetime
    calendar_code: str
    timezone_name: str
    logical_session: SessionName
    regular_session_open: bool
    session_date: date
    official_open_utc: datetime
    official_close_utc: datetime
    cycle_id: str
    cycle_session_date: date
    cycle_started_at_utc: datetime
    stage: SessionStage
    calendar_entry_allowed: bool
    maker_exit_required: bool
    force_market_flatten_required: bool

    def resolve_session(
        self,
        config: EquityLeveragedEtfArbitrageConfig,
        pair_id: str,
    ) -> ResolvedSessionConfig:
        """Resolve logical-session settings using T001's complete fallback."""

        if not isinstance(config, EquityLeveragedEtfArbitrageConfig):
            raise TypeError("config must be an EquityLeveragedEtfArbitrageConfig")
        return config.resolve_session(pair_id, self.logical_session)


class XnysNavCalendar:
    """Create deterministic snapshots from an injected clock and official XNYS data."""

    def __init__(self, nav_config: NavConfig, *, clock: Callable[[], datetime]):
        if not isinstance(nav_config, NavConfig):
            raise TypeError("nav_config must be a NavConfig")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if nav_config.timezone != NEW_YORK_TIMEZONE_NAME:
            raise ValueError(f"unsupported equity timezone: {nav_config.timezone}")

        calendar_code = exchange_calendar_code(nav_config.calendar)
        self._nav_config = nav_config
        self._clock = clock
        self._calendar = exchange_calendars.get_calendar(
            calendar_code,
            start=_INTERNAL_START_DATE,
            end=_INTERNAL_END_DATE,
        )

    def snapshot(self) -> SessionSnapshot:
        """Return a snapshot at the injected clock's current instant."""

        return self.snapshot_at(self._clock())

    def snapshot_at_local(self, observed_at_local: datetime) -> SessionSnapshot:
        """Convert an unambiguous New York wall time and return its snapshot."""

        return self.snapshot_at(_new_york_wall_time_to_utc(observed_at_local))

    def snapshot_at(self, observed_at: datetime) -> SessionSnapshot:
        """Return a snapshot for a timezone-aware instant."""

        observed_at_utc = _normalize_utc(observed_at)
        local_date = observed_at_utc.astimezone(_NEW_YORK).date()
        self._validate_supported_date(local_date)

        next_session = self._calendar.date_to_session(local_date, direction="next")
        next_close = self._session_close(next_session)
        if observed_at_utc >= next_close:
            next_session = self._calendar.next_session(next_session)
            next_close = self._session_close(next_session)

        active_cycle_session = self._calendar.date_to_session(local_date, direction="previous")
        active_cycle_close = self._session_close(active_cycle_session)
        if observed_at_utc < active_cycle_close:
            active_cycle_session = self._calendar.previous_session(active_cycle_session)
            active_cycle_close = self._session_close(active_cycle_session)

        current_is_session = self._calendar.is_session(local_date)
        regular_session_open = False
        if current_is_session:
            current_session = self._calendar.date_to_session(local_date)
            current_open = self._session_open(current_session)
            current_close = self._session_close(current_session)
            regular_session_open = current_open <= observed_at_utc < current_close
            logical_session = SessionName.REGULAR if regular_session_open else SessionName.EXTENDED
        else:
            logical_session = SessionName.WEEKEND_HOLIDAY

        next_open = self._session_open(next_session)
        new_entry_cutoff = next_close - timedelta(minutes=self._nav_config.new_entry_cutoff_minutes)
        maker_exit_start = next_close - timedelta(minutes=self._nav_config.maker_close_lead_minutes)
        force_market_start = next_close - timedelta(seconds=self._nav_config.force_market_close_lead_seconds)

        force_market_flatten_required = observed_at_utc >= force_market_start
        maker_exit_required = not force_market_flatten_required and observed_at_utc >= maker_exit_start
        calendar_entry_allowed = (
            observed_at_utc < new_entry_cutoff
            and not maker_exit_required
            and not force_market_flatten_required
        )
        if force_market_flatten_required:
            stage = SessionStage.FORCED_MARKET_FLATTEN
        elif maker_exit_required:
            stage = SessionStage.MAKER_EXIT
        elif not calendar_entry_allowed:
            stage = SessionStage.NEW_ENTRY_CUTOFF
        else:
            stage = SessionStage.NORMAL

        session_date = next_session.date()
        cycle_session_date = active_cycle_session.date()
        return SessionSnapshot(
            observed_at_utc=observed_at_utc,
            calendar_code=XNYS_CALENDAR_CODE,
            timezone_name=NEW_YORK_TIMEZONE_NAME,
            logical_session=logical_session,
            regular_session_open=regular_session_open,
            session_date=session_date,
            official_open_utc=next_open,
            official_close_utc=next_close,
            cycle_id=f"xnys-{cycle_session_date.isoformat()}",
            cycle_session_date=cycle_session_date,
            cycle_started_at_utc=active_cycle_close,
            stage=stage,
            calendar_entry_allowed=calendar_entry_allowed,
            maker_exit_required=maker_exit_required,
            force_market_flatten_required=force_market_flatten_required,
        )

    def official_close_for_session(self, session_date: date) -> datetime:
        """Return the exact official close for an XNYS session date.

        A holiday or weekend is not silently shifted to an adjacent session. This
        makes persisted NAV-cycle dates safe to validate independently of an
        observation clock.
        """

        if not isinstance(session_date, date) or isinstance(session_date, datetime):
            raise TypeError("session_date must be a date")
        self._validate_supported_date(session_date)
        if not self._calendar.is_session(session_date):
            raise ValueError(f"date {session_date.isoformat()} is not an XNYS session")
        session = self._calendar.date_to_session(session_date)
        return self._session_close(session)

    @staticmethod
    def _validate_supported_date(local_date: date) -> None:
        if not SUPPORTED_START_DATE <= local_date <= SUPPORTED_END_DATE:
            raise CalendarRangeError(
                f"date {local_date.isoformat()} is outside supported XNYS range "
                f"{SUPPORTED_START_DATE.isoformat()}..{SUPPORTED_END_DATE.isoformat()}"
            )

    def _session_open(self, session: pd.Timestamp) -> datetime:
        return _to_utc_datetime(self._calendar.session_open(session))

    def _session_close(self, session: pd.Timestamp) -> datetime:
        return _to_utc_datetime(self._calendar.session_close(session))
