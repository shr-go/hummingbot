import asyncio
import unittest
from datetime import timedelta
from decimal import Decimal

from hummingbot.core.api_throttler.async_throttler import AsyncThrottler
from hummingbot.core.web_assistant.connections.data_types import RESTMethod
from hummingbot.data_feed.yahoo_finance.parser import YahooChartParseError
from hummingbot.data_feed.yahoo_finance.provider import (
    YahooChartProvider,
    YahooDeadlineExceeded,
    YahooHTTPError,
)

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
