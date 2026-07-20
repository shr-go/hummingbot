import asyncio
import math
import random
import time
from collections.abc import Awaitable, Callable
from datetime import date, datetime, timezone
from typing import Optional
from urllib.parse import quote

from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.data_feed.yahoo_finance.parser import YahooChartParser, YahooCloseObservation
from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import NavConfig


YAHOO_CHART_RATE_LIMIT_ID = "yahoo_finance_chart_http"
_YAHOO_RATE_LIMITS = [RateLimit(limit_id=YAHOO_CHART_RATE_LIMIT_ID, limit=8, time_interval=1.0)]


class YahooHTTPError(IOError):
    """Raised when the bounded Yahoo HTTP attempt cannot produce a response."""


class YahooDeadlineExceeded(YahooHTTPError):
    """Raised when the original absolute acquisition deadline has elapsed."""


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class YahooChartProvider:
    """Fetches raw Yahoo chart text through Hummingbot's shared REST lifecycle."""

    def __init__(
        self,
        nav_config: NavConfig,
        web_assistants_factory: Optional[WebAssistantsFactory] = None,
        utc_clock: Callable[[], datetime] = _utc_now,
        monotonic_clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        jitter: Callable[[float], float] = lambda upper_bound: random.uniform(0.0, upper_bound),
        parser: Optional[YahooChartParser] = None,
    ):
        if not isinstance(nav_config, NavConfig):
            raise TypeError("nav_config must be a NavConfig")
        self._nav_config = nav_config
        self._web_assistants_factory = web_assistants_factory or WebAssistantsFactory(
            throttler=AsyncThrottler(rate_limits=_YAHOO_RATE_LIMITS)
        )
        self._utc_clock = utc_clock
        self._monotonic_clock = monotonic_clock
        self._sleep = sleep
        self._jitter = jitter
        self._parser = parser or YahooChartParser()

    @property
    def web_assistants_factory(self) -> WebAssistantsFactory:
        return self._web_assistants_factory

    async def fetch_close(
        self,
        symbol: str,
        target_session_date: date,
        deadline_utc: datetime,
    ) -> YahooCloseObservation:
        self._validate_deadline(deadline_utc)
        monotonic_deadline = self._monotonic_clock() + max(
            0.0,
            (deadline_utc - self._utc_clock()).total_seconds(),
        )
        if self._remaining_seconds(deadline_utc, monotonic_deadline) <= 0:
            raise YahooDeadlineExceeded("Yahoo acquisition deadline has elapsed")

        rest_assistant = await self._web_assistants_factory.get_rest_assistant()
        encoded_symbol = quote(symbol, safe="")
        last_error: BaseException | None = None
        rate_limit_attempt = 0

        for base_index, base_url in enumerate(self._nav_config.yahoo_base_urls):
            remaining = self._remaining_seconds(deadline_utc, monotonic_deadline)
            if remaining <= 0:
                raise YahooDeadlineExceeded("Yahoo acquisition deadline has elapsed") from last_error
            request_url = f"{base_url}/v8/finance/chart/{encoded_symbol}"
            timeout = min(self._nav_config.anchor_http_request_timeout_seconds, remaining)
            try:
                response = await rest_assistant.execute_request_and_get_response(
                    url=request_url,
                    throttler_limit_id=YAHOO_CHART_RATE_LIMIT_ID,
                    params={
                        "range": self._nav_config.yahoo_chart_range,
                        "interval": self._nav_config.yahoo_chart_interval,
                        "events": "div,splits",
                        "includePrePost": str(self._nav_config.yahoo_include_pre_post).lower(),
                    },
                    method=RESTMethod.GET,
                    return_err=True,
                    timeout=timeout,
                    headers={"User-Agent": self._nav_config.yahoo_user_agent},
                )
                raw_text = await response.text()
            except asyncio.CancelledError:
                raise
            except Exception as exception:
                last_error = exception
                if base_index + 1 < len(self._nav_config.yahoo_base_urls):
                    continue
                raise YahooHTTPError(f"Yahoo request failed on all approved hosts: {exception}") from exception

            received_at = self._utc_clock()
            if self._remaining_seconds(deadline_utc, monotonic_deadline) <= 0:
                raise YahooDeadlineExceeded("Yahoo response completed at or after the acquisition deadline")

            status = response.status
            if status == 200:
                return self._parser.parse(
                    raw_text=raw_text,
                    expected_symbol=symbol,
                    target_session_date=target_session_date,
                    received_at_utc=received_at,
                    source_url=request_url,
                )
            if status == 429:
                last_error = YahooHTTPError(f"Yahoo returned HTTP 429 from {base_url}")
                if base_index + 1 >= len(self._nav_config.yahoo_base_urls):
                    raise last_error
                backoff = min(
                    self._nav_config.anchor_poll_initial_interval_seconds * (2**rate_limit_attempt),
                    self._nav_config.anchor_poll_max_interval_seconds,
                )
                rate_limit_attempt += 1
                jitter = self._jitter(float(backoff))
                if not isinstance(jitter, (int, float)) or isinstance(jitter, bool) or not math.isfinite(jitter):
                    raise ValueError("Yahoo backoff jitter must be a finite number")
                if jitter < 0 or jitter > backoff:
                    raise ValueError("Yahoo backoff jitter must be between zero and the backoff bound")
                remaining = self._remaining_seconds(deadline_utc, monotonic_deadline)
                if remaining <= 0:
                    raise YahooDeadlineExceeded("Yahoo acquisition deadline has elapsed") from last_error
                await self._sleep(min(float(backoff) + float(jitter), remaining))
                if self._remaining_seconds(deadline_utc, monotonic_deadline) <= 0:
                    raise YahooDeadlineExceeded("Yahoo 429 backoff reached the acquisition deadline") from last_error
                continue
            if 500 <= status <= 599:
                last_error = YahooHTTPError(f"Yahoo returned HTTP {status} from {base_url}")
                if base_index + 1 < len(self._nav_config.yahoo_base_urls):
                    continue
                raise last_error
            raise YahooHTTPError(f"Yahoo returned non-success HTTP status {status} from {base_url}")

        raise YahooHTTPError("Yahoo request exhausted approved hosts") from last_error

    def _remaining_seconds(self, deadline_utc: datetime, monotonic_deadline: float) -> float:
        wall_remaining = (deadline_utc - self._utc_clock()).total_seconds()
        monotonic_remaining = monotonic_deadline - self._monotonic_clock()
        return max(0.0, min(wall_remaining, monotonic_remaining))

    @staticmethod
    def _validate_deadline(deadline_utc: datetime) -> None:
        if (
            not isinstance(deadline_utc, datetime)
            or deadline_utc.tzinfo is None
            or deadline_utc.utcoffset() != timezone.utc.utcoffset(deadline_utc)
        ):
            raise TypeError("deadline_utc must be an aware UTC datetime")
