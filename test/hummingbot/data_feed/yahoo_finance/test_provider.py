import asyncio
import unittest
from collections import deque
from datetime import timedelta
from decimal import Decimal
from typing import Any

from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.api_throttler.data_types import RateLimit
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.core.web_assistant.connections.rest_connection import RESTConnection
from hummingbot.core.web_assistant.web_assistants_factory import WebAssistantsFactory
from hummingbot.data_feed.yahoo_finance.parser import YahooChartParseError
from hummingbot.data_feed.yahoo_finance.provider import (
    YAHOO_CHART_RATE_LIMIT_ID,
    YahooChartProvider,
    YahooDeadlineExceeded,
    YahooHTTPError,
)


class SyntheticContent:
    def __init__(self, response):
        self._response = response

    async def iter_chunked(self, chunk_size):
        self._response.run_read_hook()
        body = self._response.body
        for index in range(0, len(body), chunk_size):
            await asyncio.sleep(0)
            yield body[index:index + chunk_size]


class SyntheticAiohttpResponse:
    def __init__(self, status, body, url, *, headers=None, read_hook=None):
        self.status = status
        self.body = body if isinstance(body, bytes) else body.encode("utf-8")
        self.url = url
        self.method = "GET"
        self.headers = {} if headers is None else headers
        self.content_type = "application/json"
        self.content = SyntheticContent(self)
        self.read_hook = read_hook
        self.read_hook_called = False
        self.released = False
        self.release_calls = 0

    def run_read_hook(self):
        if self.read_hook is not None and not self.read_hook_called:
            self.read_hook_called = True
            self.read_hook()

    async def text(self):
        self.run_read_hook()
        return self.body.decode("utf-8")

    async def read(self):
        self.run_read_hook()
        return self.body

    def release(self):
        self.release_calls += 1
        self.released = True


class ScriptedClientSession:
    def __init__(self, *outcomes: Any):
        self.outcomes = deque(outcomes)
        self.calls = []

    async def request(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outcomes:
            raise AssertionError("unexpected synthetic HTTP request")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            outcome = outcome()
        return outcome


class StaticConnectionsFactory:
    def __init__(self, session, *, acquire_hook=None):
        self.session = session
        self.acquire_hook = acquire_hook
        self.calls = 0

    async def get_rest_connection(self):
        self.calls += 1
        if self.acquire_hook is not None:
            self.acquire_hook()
        return RESTConnection(aiohttp_client_session=self.session)


class DeadlineAdvancingLock:
    def __init__(self, hook):
        self._hook = hook
        self._entered = False

    async def __aenter__(self):
        if not self._entered:
            self._entered = True
            self._hook()
        await asyncio.sleep(0)
        return self

    async def __aexit__(self, exc_type, exc, traceback):
        return False

from .conftest import (
    FakeClock,
    FakeFactory,
    FakeResponse,
    OFFICIAL_CLOSE,
    ScriptedRestAssistant,
    TARGET_SESSION_DATE,
    fixture_text,
    load_nav_config,
)


class YahooChartProviderTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.nav_config = load_nav_config()

    def make_provider(self, nav_config, clock, assistant, *, jitter=None):
        self.factory = FakeFactory(assistant)
        self.jitter_calls = []

        def deterministic_jitter(upper_bound):
            self.jitter_calls.append(upper_bound)
            return 0.0 if jitter is None else jitter(upper_bound)

        return YahooChartProvider(
            nav_config=nav_config,
            web_assistants_factory=self.factory,
            utc_clock=clock.utcnow,
            monotonic_clock=clock.monotonic,
            sleep=clock.sleep,
            jitter=deterministic_jitter,
        )

    def make_real_assistant_provider(
        self,
        clock,
        *responses,
        nav_config=None,
        throttler=None,
        connection_acquire_hook=None,
    ):
        self.real_session = ScriptedClientSession(*responses)
        self.real_throttler = throttler or AsyncThrottler(
            rate_limits=[
                RateLimit(
                    limit_id=YAHOO_CHART_RATE_LIMIT_ID,
                    limit=100,
                    time_interval=1,
                )
            ]
        )
        self.real_connections_factory = StaticConnectionsFactory(
            self.real_session,
            acquire_hook=connection_acquire_hook,
        )
        factory = WebAssistantsFactory(
            throttler=self.real_throttler,
            connections_factory=self.real_connections_factory,
        )
        return YahooChartProvider(
            nav_config=self.nav_config if nav_config is None else nav_config,
            web_assistants_factory=factory,
            utc_clock=clock.utcnow,
            monotonic_clock=clock.monotonic,
            sleep=clock.sleep,
            jitter=lambda upper_bound: 0.0,
        )

    def nav_config_with_request_timeout(self, timeout_seconds):
        values = self.nav_config.model_dump(mode="python")
        values["anchor_http_request_timeout_seconds"] = timeout_seconds
        values["anchor_pair_fetch_max_skew_seconds"] = min(
            values["anchor_pair_fetch_max_skew_seconds"],
            timeout_seconds,
        )
        return type(self.nav_config).model_validate(values)

    async def test_query2_success_uses_raw_rest_response_and_exact_chart_request(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        assistant = ScriptedRestAssistant(FakeResponse(200, fixture_text("sndk_chart.json")))
        provider = self.make_provider(self.nav_config, clock, assistant)

        observation = await provider.fetch_close(
            symbol="SNDK",
            target_session_date=TARGET_SESSION_DATE,
            deadline_utc=OFFICIAL_CLOSE + timedelta(seconds=600),
        )

        assert observation.close == Decimal("250.125")
        assert observation.received_at_utc == clock.utcnow()
        assert self.factory.calls == 1
        assert len(assistant.calls) == 1
        request = assistant.calls[0]
        assert request["url"] == "https://query2.finance.yahoo.com/v8/finance/chart/SNDK"
        assert request["method"] is RESTMethod.GET
        assert request["params"] == {
            "range": "5d",
            "interval": "1d",
            "events": "div,splits",
            "includePrePost": "false",
        }
        assert request["headers"] == {"User-Agent": self.nav_config.yahoo_user_agent}
        assert "Cookie" not in request["headers"]
        assert "crumb" not in request["params"]
        assert request["return_err"] is True
        assert request["timeout"] == self.nav_config.anchor_http_request_timeout_seconds
        assert request["throttler_limit_id"]

    async def test_default_provider_builds_hummingbot_factory_with_async_throttler(self):
        provider = YahooChartProvider(nav_config=self.nav_config)

        assert isinstance(provider.web_assistants_factory.throttler, AsyncThrottler)

    async def test_network_timeout_and_5xx_fail_over_from_query2_to_query1(self):
        first_failures = (
            TimeoutError("synthetic timeout"),
            ConnectionError("synthetic disconnect"),
            FakeResponse(503, "synthetic unavailable"),
        )
        for first_failure in first_failures:
            with self.subTest(first_failure=type(first_failure).__name__):
                clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
                assistant = ScriptedRestAssistant(
                    first_failure,
                    FakeResponse(200, fixture_text("sndk_chart.json")),
                )
                provider = self.make_provider(self.nav_config, clock, assistant)

                observation = await provider.fetch_close(
                    "SNDK",
                    TARGET_SESSION_DATE,
                    OFFICIAL_CLOSE + timedelta(seconds=600),
                )

                assert observation.close == Decimal("250.125")
                assert [call["url"].split("/")[2] for call in assistant.calls] == [
                    "query2.finance.yahoo.com",
                    "query1.finance.yahoo.com",
                ]

    async def test_ordinary_4xx_fails_round_without_query1_failover(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        assistant = ScriptedRestAssistant(FakeResponse(404, "synthetic missing"))
        provider = self.make_provider(self.nav_config, clock, assistant)

        with self.assertRaisesRegex(YahooHTTPError, "404"):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert len(assistant.calls) == 1

    async def test_html_and_malformed_json_fail_round_without_cross_host_retry(self):
        for raw_body in ("<html>Will be right back</html>", "{malformed"):
            with self.subTest(raw_body=raw_body):
                clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
                assistant = ScriptedRestAssistant(FakeResponse(200, raw_body))
                provider = self.make_provider(self.nav_config, clock, assistant)

                with self.assertRaises(YahooChartParseError):
                    await provider.fetch_close(
                        "SNDK",
                        TARGET_SESSION_DATE,
                        OFFICIAL_CLOSE + timedelta(seconds=600),
                    )

                assert len(assistant.calls) == 1

    async def test_429_uses_injected_jittered_backoff_then_query1_within_same_deadline(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        assistant = ScriptedRestAssistant(
            FakeResponse(429, "Too Many Requests"),
            FakeResponse(200, fixture_text("sndk_chart.json")),
        )
        provider = self.make_provider(self.nav_config, clock, assistant, jitter=lambda upper: upper / 4)

        observation = await provider.fetch_close(
            "SNDK",
            TARGET_SESSION_DATE,
            OFFICIAL_CLOSE + timedelta(seconds=600),
        )

        assert observation.close == Decimal("250.125")
        assert self.jitter_calls == [float(self.nav_config.anchor_poll_initial_interval_seconds)]
        assert clock.sleeps == [2.5]
        assert len(assistant.calls) == 2

    async def test_429_backoff_is_clipped_at_absolute_deadline_and_never_recovers_late(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=599, milliseconds=500))
        assistant = ScriptedRestAssistant(
            FakeResponse(429, "Too Many Requests"),
            FakeResponse(200, fixture_text("sndk_chart.json")),
        )
        provider = self.make_provider(self.nav_config, clock, assistant)

        with self.assertRaises(YahooDeadlineExceeded):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert clock.sleeps == [0.5]
        assert len(assistant.calls) == 1

    async def test_each_failover_timeout_uses_smaller_wall_and_monotonic_remaining_budget(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))

        def delayed_503():
            # A backward/stalled wall clock cannot restore monotonic budget.
            clock.monotonic_value += 4
            return FakeResponse(503, "synthetic unavailable")

        assistant = ScriptedRestAssistant(
            delayed_503,
            FakeResponse(200, fixture_text("sndk_chart.json")),
        )
        provider = self.make_provider(self.nav_config, clock, assistant)

        await provider.fetch_close(
            "SNDK",
            TARGET_SESSION_DATE,
            OFFICIAL_CLOSE + timedelta(seconds=72),
        )

        assert assistant.calls[0]["timeout"] == 10
        assert assistant.calls[1]["timeout"] == 8

    async def test_cancelled_error_is_not_converted_to_a_failover_or_round_failure(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        assistant = ScriptedRestAssistant(asyncio.CancelledError())
        provider = self.make_provider(self.nav_config, clock, assistant)

        with self.assertRaises(asyncio.CancelledError):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert len(assistant.calls) == 1

    async def test_http_200_unicode_decode_failure_fails_round_without_query1_failover(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        assistant = ScriptedRestAssistant(
            FakeResponse(200, b"\xff\xfe"),
            FakeResponse(200, fixture_text("sndk_chart.json")),
        )
        provider = self.make_provider(self.nav_config, clock, assistant)

        with self.assertRaises((UnicodeDecodeError, YahooChartParseError, YahooHTTPError)):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert len(assistant.calls) == 1

    async def test_real_rest_assistant_and_async_throttler_seam_reads_status_and_body_without_network(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query2.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        provider = self.make_real_assistant_provider(clock, response)

        observation = await provider.fetch_close(
            "SNDK",
            TARGET_SESSION_DATE,
            OFFICIAL_CLOSE + timedelta(seconds=600),
        )

        assert observation.close == Decimal("250.125")
        assert observation.source_url == response.url
        assert len(self.real_session.calls) == 1
        assert len(self.real_throttler._task_logs) == 1
        assert response.released is True

    async def test_real_rest_assistant_transport_timeout_and_5xx_use_approved_failover(self):
        for first_outcome in (
            TimeoutError("synthetic timeout"),
            SyntheticAiohttpResponse(
                503,
                b"unavailable",
                "https://query2.finance.yahoo.com/v8/finance/chart/SNDK",
            ),
        ):
            with self.subTest(first_outcome=type(first_outcome).__name__):
                clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
                query1_response = SyntheticAiohttpResponse(
                    200,
                    fixture_text("sndk_chart.json"),
                    "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
                )
                provider = self.make_real_assistant_provider(clock, first_outcome, query1_response)

                observation = await provider.fetch_close(
                    "SNDK",
                    TARGET_SESSION_DATE,
                    OFFICIAL_CLOSE + timedelta(seconds=600),
                )

                assert observation.close == Decimal("250.125")
                assert len(self.real_session.calls) == 2
                assert query1_response.released is True
                if isinstance(first_outcome, SyntheticAiohttpResponse):
                    assert first_outcome.released is True

    async def test_real_query2_body_exceeding_attempt_timeout_fails_over_to_query1(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        nav_config = self.nav_config_with_request_timeout(1)
        query2_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query2.finance.yahoo.com/v8/finance/chart/SNDK",
            read_hook=lambda: setattr(clock, "monotonic_value", clock.monotonic_value + 1.2),
        )
        query1_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        provider = self.make_real_assistant_provider(
            clock,
            query2_response,
            query1_response,
            nav_config=nav_config,
        )

        observation = await provider.fetch_close(
            "SNDK",
            TARGET_SESSION_DATE,
            OFFICIAL_CLOSE + timedelta(seconds=600),
        )

        assert observation.source_url == query1_response.url
        assert [call["url"].split("/")[2] for call in self.real_session.calls] == [
            "query2.finance.yahoo.com",
            "query1.finance.yahoo.com",
        ]
        assert query2_response.release_calls == 1
        assert query1_response.release_calls == 1

    async def test_real_query1_body_timeout_ends_failover_and_releases_response_once(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        nav_config = self.nav_config_with_request_timeout(1)
        query1_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
            read_hook=lambda: setattr(clock, "monotonic_value", clock.monotonic_value + 1.2),
        )
        provider = self.make_real_assistant_provider(
            clock,
            TimeoutError("synthetic query2 timeout"),
            query1_response,
            nav_config=nav_config,
        )

        with self.assertRaisesRegex(YahooHTTPError, "all approved hosts"):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert len(self.real_session.calls) == 2
        assert query1_response.release_calls == 1

    async def test_real_body_cancellation_propagates_and_releases_response_once(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))

        def cancel_body_read():
            raise asyncio.CancelledError()

        query2_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query2.finance.yahoo.com/v8/finance/chart/SNDK",
            read_hook=cancel_body_read,
        )
        query1_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        provider = self.make_real_assistant_provider(clock, query2_response, query1_response)

        with self.assertRaises(asyncio.CancelledError):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert len(self.real_session.calls) == 1
        assert query2_response.release_calls == 1
        assert query1_response.release_calls == 0

    async def test_real_request_headers_are_bounded_by_same_attempt_deadline(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        nav_config = self.nav_config_with_request_timeout(1)
        query2_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query2.finance.yahoo.com/v8/finance/chart/SNDK",
        )

        def delayed_query2_headers():
            clock.monotonic_value += 1
            return query2_response

        query1_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        provider = self.make_real_assistant_provider(
            clock,
            delayed_query2_headers,
            query1_response,
            nav_config=nav_config,
        )

        observation = await provider.fetch_close(
            "SNDK",
            TARGET_SESSION_DATE,
            OFFICIAL_CLOSE + timedelta(seconds=600),
        )

        assert observation.source_url == query1_response.url
        assert query2_response.release_calls == 1
        assert query1_response.release_calls == 1

    async def test_real_rest_assistant_cancellation_propagates_without_failover(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        provider = self.make_real_assistant_provider(clock, asyncio.CancelledError())

        with self.assertRaises(asyncio.CancelledError):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert len(self.real_session.calls) == 1

    async def test_real_async_throttler_wait_is_bounded_by_attempt_and_releases_responses_once(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        nav_config = self.nav_config_with_request_timeout(1)
        query2_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query2.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        query1_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        throttler = AsyncThrottler(
            rate_limits=[RateLimit(YAHOO_CHART_RATE_LIMIT_ID, limit=100, time_interval=1)]
        )
        throttler._lock = DeadlineAdvancingLock(lambda: setattr(clock, "monotonic_value", clock.monotonic_value + 1))
        provider = self.make_real_assistant_provider(
            clock,
            query2_response,
            query1_response,
            nav_config=nav_config,
            throttler=throttler,
        )

        observation = await provider.fetch_close(
            "SNDK",
            TARGET_SESSION_DATE,
            OFFICIAL_CLOSE + timedelta(seconds=600),
        )

        assert observation.source_url == query1_response.url
        assert query2_response.release_calls == 1
        assert query1_response.release_calls == 1

    async def test_shorter_cycle_remaining_wins_over_attempt_timeout_at_equality(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=599))
        query2_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query2.finance.yahoo.com/v8/finance/chart/SNDK",
            read_hook=lambda: setattr(clock, "monotonic_value", clock.monotonic_value + 1),
        )
        query1_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        provider = self.make_real_assistant_provider(clock, query2_response, query1_response)

        with self.assertRaises(YahooDeadlineExceeded):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert len(self.real_session.calls) == 1
        assert query2_response.release_calls == 1
        assert query1_response.release_calls == 0

    async def test_real_factory_acquisition_is_bounded_by_attempt_before_request(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        nav_config = self.nav_config_with_request_timeout(1)
        first_acquisition = True

        def delay_first_acquisition():
            nonlocal first_acquisition
            if first_acquisition:
                first_acquisition = False
                clock.monotonic_value += 1

        query1_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        provider = self.make_real_assistant_provider(
            clock,
            query1_response,
            nav_config=nav_config,
            connection_acquire_hook=delay_first_acquisition,
        )

        observation = await provider.fetch_close(
            "SNDK",
            TARGET_SESSION_DATE,
            OFFICIAL_CLOSE + timedelta(seconds=600),
        )

        assert observation.source_url == query1_response.url
        assert self.real_connections_factory.calls == 2
        assert len(self.real_session.calls) == 1
        assert self.real_session.calls[0]["url"].split("/")[2] == "query1.finance.yahoo.com"
        assert query1_response.release_calls == 1

    async def test_final_response_url_must_remain_on_approved_https_chart_endpoint(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        redirected = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://example.invalid/v8/finance/chart/SNDK",
        )
        query1_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        provider = self.make_real_assistant_provider(clock, redirected, query1_response)

        with self.assertRaises(YahooHTTPError):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert len(self.real_session.calls) == 1
        assert redirected.released is True

    async def test_declared_oversized_chart_body_fails_closed_without_cross_host_retry(self):
        clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
        oversized = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query2.finance.yahoo.com/v8/finance/chart/SNDK",
            headers={"Content-Length": str(20 * 1024 * 1024)},
        )
        query1_response = SyntheticAiohttpResponse(
            200,
            fixture_text("sndk_chart.json"),
            "https://query1.finance.yahoo.com/v8/finance/chart/SNDK",
        )
        provider = self.make_real_assistant_provider(clock, oversized, query1_response)

        with self.assertRaises(YahooHTTPError):
            await provider.fetch_close(
                "SNDK",
                TARGET_SESSION_DATE,
                OFFICIAL_CLOSE + timedelta(seconds=600),
            )

        assert len(self.real_session.calls) == 1
        assert oversized.released is True
