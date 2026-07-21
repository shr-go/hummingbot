import asyncio
import inspect
import math
import random
import time
from collections.abc import Awaitable, Callable, Mapping
from datetime import date, datetime, timedelta, timezone
from typing import Optional
from urllib.parse import quote, urlsplit

import aiohttp

from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.data_feed.yahoo_finance.parser import YahooChartParser, YahooCloseObservation
from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import NavConfig


YAHOO_CHART_RATE_LIMIT_ID = "yahoo_finance_chart_http"
YAHOO_MAX_CHART_RESPONSE_BYTES = 1024 * 1024
_YAHOO_RATE_LIMITS = [RateLimit(limit_id=YAHOO_CHART_RATE_LIMIT_ID, limit=8, time_interval=1.0)]
_TRANSPORT_ERRORS = (OSError, aiohttp.ClientError)


class YahooHTTPError(IOError):
    """Raised when the bounded Yahoo HTTP attempt cannot produce a response."""


class YahooDeadlineExceeded(YahooHTTPError):
    """Raised when the original absolute acquisition deadline has elapsed."""


class YahooAttemptTimeout(TimeoutError):
    """Raised when one approved Yahoo host exhausts its bounded attempt."""


class YahooResponseError(YahooHTTPError):
    """Raised for an untrusted HTTP response that must not trigger host failover."""


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _validate_deadline(deadline_utc: datetime) -> None:
    if (
        not isinstance(deadline_utc, datetime)
        or deadline_utc.tzinfo is None
        or deadline_utc.utcoffset() != timezone.utc.utcoffset(deadline_utc)
    ):
        raise TypeError("deadline_utc must be an aware UTC datetime")


class YahooCycleBudget:
    """One conservative wall-plus-monotonic budget for an entire NAV cycle."""

    def __init__(
        self,
        deadline_utc: datetime,
        monotonic_deadline: float,
        utc_clock: Callable[[], datetime],
        monotonic_clock: Callable[[], float],
    ):
        _validate_deadline(deadline_utc)
        self.deadline_utc = deadline_utc
        self.monotonic_deadline = monotonic_deadline
        self._utc_clock = utc_clock
        self._monotonic_clock = monotonic_clock

    @classmethod
    def start(
        cls,
        deadline_utc: datetime,
        utc_clock: Callable[[], datetime],
        monotonic_clock: Callable[[], float],
    ) -> "YahooCycleBudget":
        _validate_deadline(deadline_utc)
        wall_remaining = max(0.0, (deadline_utc - utc_clock()).total_seconds())
        return cls(
            deadline_utc=deadline_utc,
            monotonic_deadline=monotonic_clock() + wall_remaining,
            utc_clock=utc_clock,
            monotonic_clock=monotonic_clock,
        )

    def remaining_seconds(self) -> float:
        wall_remaining = (self.deadline_utc - self._utc_clock()).total_seconds()
        monotonic_remaining = self.monotonic_deadline - self._monotonic_clock()
        return max(0.0, min(wall_remaining, monotonic_remaining))

    def ensure_remaining(self, operation: str) -> float:
        remaining = self.remaining_seconds()
        if remaining <= 0:
            raise YahooDeadlineExceeded(f"{operation} completed at or after the Yahoo acquisition deadline")
        return remaining

    async def wait(self, awaitable, operation: str, *, check_after: bool = True):
        remaining = self.remaining_seconds()
        if remaining <= 0:
            self._discard_unstarted(awaitable)
            raise YahooDeadlineExceeded(f"Yahoo acquisition deadline elapsed before {operation}")
        try:
            result = await asyncio.wait_for(awaitable, timeout=remaining)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exception:
            if self.remaining_seconds() <= 0:
                raise YahooDeadlineExceeded(
                    f"Yahoo acquisition deadline elapsed during {operation}"
                ) from exception
            raise
        if check_after:
            self.ensure_remaining(operation)
        return result

    @staticmethod
    def _discard_unstarted(awaitable) -> None:
        if inspect.iscoroutine(awaitable):
            awaitable.close()
        elif isinstance(awaitable, asyncio.Future):
            awaitable.cancel()


class _YahooAttemptBudget:
    """A host-attempt deadline bounded by, and never extending, one cycle budget."""

    def __init__(
        self,
        cycle_budget: YahooCycleBudget,
        deadline_utc: datetime,
        monotonic_deadline: float,
    ):
        self._cycle_budget = cycle_budget
        self.deadline_utc = deadline_utc
        self.monotonic_deadline = monotonic_deadline

    @classmethod
    def start(
        cls,
        cycle_budget: YahooCycleBudget,
        maximum_seconds: float,
    ) -> "_YahooAttemptBudget":
        utc_now = cycle_budget._utc_clock()
        monotonic_now = cycle_budget._monotonic_clock()
        wall_remaining = (cycle_budget.deadline_utc - utc_now).total_seconds()
        monotonic_remaining = cycle_budget.monotonic_deadline - monotonic_now
        cycle_remaining = max(0.0, min(wall_remaining, monotonic_remaining))
        if cycle_remaining <= 0:
            raise YahooDeadlineExceeded("Yahoo acquisition deadline elapsed before host attempt")
        attempt_seconds = min(float(maximum_seconds), cycle_remaining)
        return cls(
            cycle_budget=cycle_budget,
            deadline_utc=utc_now + timedelta(seconds=attempt_seconds),
            monotonic_deadline=monotonic_now + attempt_seconds,
        )

    def remaining_seconds(self) -> float:
        wall_remaining = (self.deadline_utc - self._cycle_budget._utc_clock()).total_seconds()
        monotonic_remaining = self.monotonic_deadline - self._cycle_budget._monotonic_clock()
        return max(
            0.0,
            min(
                wall_remaining,
                monotonic_remaining,
                self._cycle_budget.remaining_seconds(),
            ),
        )

    def ensure_remaining(self, operation: str) -> float:
        remaining = self.remaining_seconds()
        if remaining <= 0:
            self._raise_timeout(operation)
        return remaining

    async def wait(self, awaitable, operation: str, *, check_after: bool = True):
        remaining = self.remaining_seconds()
        if remaining <= 0:
            YahooCycleBudget._discard_unstarted(awaitable)
            self._raise_timeout(operation)
        try:
            result = await asyncio.wait_for(awaitable, timeout=remaining)
        except asyncio.CancelledError:
            raise
        except TimeoutError as exception:
            self._raise_timeout(operation, exception)
        if check_after:
            self.ensure_remaining(operation)
        return result

    def _raise_timeout(self, operation: str, cause: BaseException | None = None) -> None:
        if self._cycle_budget.remaining_seconds() <= 0:
            error = YahooDeadlineExceeded(
                f"Yahoo acquisition deadline elapsed during {operation}"
            )
        else:
            error = YahooAttemptTimeout(f"Yahoo host attempt timed out during {operation}")
        if cause is None:
            raise error
        raise error from cause


class YahooChartProvider:
    """Fetch raw Yahoo chart bytes through Hummingbot's shared REST lifecycle."""

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
        self._approved_hosts = {
            urlsplit(base_url).hostname for base_url in self._nav_config.yahoo_base_urls
        }

    @property
    def web_assistants_factory(self) -> WebAssistantsFactory:
        return self._web_assistants_factory

    async def fetch_close(
        self,
        symbol: str,
        target_session_date: date,
        deadline_utc: datetime,
        *,
        budget: YahooCycleBudget | None = None,
    ) -> YahooCloseObservation:
        _validate_deadline(deadline_utc)
        if budget is None:
            budget = YahooCycleBudget.start(
                deadline_utc=deadline_utc,
                utc_clock=self._utc_clock,
                monotonic_clock=self._monotonic_clock,
            )
        elif budget.deadline_utc != deadline_utc:
            raise ValueError("Yahoo cycle budget deadline does not match the requested deadline")
        budget.ensure_remaining("Yahoo request")

        encoded_symbol = quote(symbol, safe="")
        last_error: BaseException | None = None
        rate_limit_attempt = 0

        for base_url in self._nav_config.yahoo_base_urls:
            request_url = f"{base_url}/v8/finance/chart/{encoded_symbol}"
            rate_limit_retry_used = False
            while True:
                try:
                    status, raw_bytes, source_url = await self._request_host(
                        request_url=request_url,
                        symbol=symbol,
                        budget=budget,
                    )
                except asyncio.CancelledError:
                    raise
                except YahooDeadlineExceeded:
                    raise
                except YahooHTTPError:
                    raise
                except _TRANSPORT_ERRORS as exception:
                    last_error = exception
                    break

                if status == 200:
                    assert raw_bytes is not None
                    try:
                        raw_text = raw_bytes.decode("utf-8", errors="strict")
                    except UnicodeDecodeError as exception:
                        raise YahooResponseError(
                            "Yahoo HTTP 200 chart body is not valid UTF-8"
                        ) from exception
                    received_at = self._utc_clock()
                    budget.ensure_remaining("Yahoo response parsing")
                    observation = self._parser.parse(
                        raw_text=raw_text,
                        expected_symbol=symbol,
                        target_session_date=target_session_date,
                        received_at_utc=received_at,
                        source_url=source_url,
                    )
                    budget.ensure_remaining("Yahoo response parsing")
                    return observation

                if status == 429:
                    last_error = YahooHTTPError(f"Yahoo returned HTTP 429 from {base_url}")
                    if rate_limit_retry_used:
                        raise last_error
                    backoff = min(
                        self._nav_config.anchor_poll_initial_interval_seconds * (2 ** rate_limit_attempt),
                        self._nav_config.anchor_poll_max_interval_seconds,
                    )
                    rate_limit_attempt += 1
                    jitter = self._jitter(float(backoff))
                    if not isinstance(jitter, (int, float)) or isinstance(jitter, bool) or not math.isfinite(jitter):
                        raise ValueError("Yahoo backoff jitter must be a finite number")
                    if jitter < 0 or jitter > backoff:
                        raise ValueError("Yahoo backoff jitter must be between zero and the backoff bound")
                    delay = min(float(backoff) + float(jitter), budget.ensure_remaining("Yahoo 429 backoff"))
                    await budget.wait(self._sleep(delay), "Yahoo 429 backoff")
                    rate_limit_retry_used = True
                    continue

                if 500 <= status <= 599:
                    last_error = YahooHTTPError(f"Yahoo returned HTTP {status} from {base_url}")
                    break
                raise YahooHTTPError(f"Yahoo returned non-success HTTP status {status} from {base_url}")

        raise YahooHTTPError("Yahoo request failed on all approved hosts") from last_error

    async def _request_host(
        self,
        request_url: str,
        symbol: str,
        budget: YahooCycleBudget,
    ) -> tuple[int, bytes | None, str]:
        attempt_budget = _YahooAttemptBudget.start(
            cycle_budget=budget,
            maximum_seconds=self._nav_config.anchor_http_request_timeout_seconds,
        )
        rest_assistant = await attempt_budget.wait(
            self._web_assistants_factory.get_rest_assistant(),
            "WebAssistantsFactory acquisition",
        )
        timeout = attempt_budget.ensure_remaining("Yahoo HTTP request")
        response = await attempt_budget.wait(
            rest_assistant.execute_request_and_get_response(
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
            ),
            "Yahoo throttler and HTTP request",
            check_after=False,
        )
        try:
            attempt_budget.ensure_remaining("Yahoo throttler and HTTP request")
            source_url = self._validate_final_url(response.url, symbol)
            try:
                status = int(response.status)
            except (TypeError, ValueError) as exception:
                raise YahooResponseError("Yahoo response status is invalid") from exception
            raw_bytes = None
            if status == 200:
                raw_bytes = await self._read_response_body(response, attempt_budget)
            return status, raw_bytes, source_url
        finally:
            self._release_response(response)

    async def _read_response_body(self, response, budget: _YahooAttemptBudget) -> bytes:
        headers = response.headers
        if headers is not None and isinstance(headers, Mapping):
            content_length = headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_length = int(content_length)
                except (TypeError, ValueError) as exception:
                    raise YahooResponseError("Yahoo Content-Length is invalid") from exception
                if declared_length < 0 or declared_length > YAHOO_MAX_CHART_RESPONSE_BYTES:
                    raise YahooResponseError("Yahoo chart response exceeds the body-size limit")

        limited_reader = getattr(response, "read_limited", None)
        try:
            if callable(limited_reader):
                body = await budget.wait(
                    limited_reader(YAHOO_MAX_CHART_RESPONSE_BYTES),
                    "Yahoo response body read",
                )
            else:
                text_body = await budget.wait(response.text(), "Yahoo response body read")
                body = text_body.encode("utf-8")
                if len(body) > YAHOO_MAX_CHART_RESPONSE_BYTES:
                    raise YahooResponseError("Yahoo chart response exceeds the body-size limit")
        except asyncio.CancelledError:
            raise
        except YahooDeadlineExceeded:
            raise
        except YahooResponseError:
            raise
        except ValueError as exception:
            raise YahooResponseError("Yahoo chart response exceeds the body-size limit") from exception
        if not isinstance(body, bytes):
            raise YahooResponseError("Yahoo chart response body is not bytes")
        return body

    def _validate_final_url(self, value: object, symbol: str) -> str:
        if not isinstance(value, str) or not value:
            raise YahooResponseError("Yahoo response does not expose its final URL")
        parsed = urlsplit(value)
        expected_path = f"/v8/finance/chart/{quote(symbol, safe='')}"
        try:
            final_port = parsed.port
        except ValueError as exception:
            raise YahooResponseError("Yahoo response final URL contains an invalid port") from exception
        if (
            parsed.scheme != "https"
            or parsed.hostname not in self._approved_hosts
            or final_port not in (None, 443)
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path != expected_path
            or parsed.fragment
        ):
            raise YahooResponseError("Yahoo response final URL is outside the approved chart endpoints")
        return value

    @staticmethod
    def _release_response(response) -> None:
        release = getattr(response, "release", None)
        if not callable(release):
            return
        try:
            result = release()
        except Exception:
            return
        if inspect.isawaitable(result):
            result.close()
