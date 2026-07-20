import json
import tomllib
from collections import deque
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import (
    EquityLeveragedEtfArbitrageConfig,
)


UTC = timezone.utc
TARGET_SESSION_DATE = date(2026, 7, 17)
OFFICIAL_CLOSE = datetime(2026, 7, 17, 20, tzinfo=UTC)


def fixture_text(name: str) -> str:
    return (Path(__file__).parent / "fixtures" / name).read_text(encoding="utf-8")


def fixture_payload(name: str) -> dict[str, Any]:
    return json.loads(fixture_text(name))


def dump_payload(payload: dict[str, Any]) -> str:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def find_repository_file(relative_path: str) -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.is_file():
            return candidate
    raise AssertionError(f"repository file not found: {relative_path}")


def load_nav_config():
    path = find_repository_file("config/equity_leveraged_etf_arbitrage.example.toml")
    with path.open("rb") as config_file:
        raw_config = tomllib.load(config_file)
    return EquityLeveragedEtfArbitrageConfig.model_validate(raw_config).nav


@pytest.fixture
def nav_config():
    return load_nav_config()


class FakeClock:
    def __init__(self, current: datetime, monotonic_value: float = 1_000.0):
        self.current = current
        self.monotonic_value = monotonic_value
        self.sleeps: list[float] = []

    def utcnow(self) -> datetime:
        return self.current

    def monotonic(self) -> float:
        return self.monotonic_value

    def advance(self, seconds: float) -> None:
        self.current += timedelta(seconds=seconds)
        self.monotonic_value += seconds

    async def sleep(self, seconds: float) -> None:
        assert seconds >= 0
        self.sleeps.append(seconds)
        self.advance(seconds)


class FakeResponse:
    def __init__(self, status: int, body: str):
        self.status = status
        self.body = body
        self.headers: dict[str, str] = {}

    async def text(self) -> str:
        return self.body


class ScriptedRestAssistant:
    def __init__(self, *outcomes: Any):
        self.outcomes = deque(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def execute_request_and_get_response(self, **kwargs):
        self.calls.append(kwargs)
        if not self.outcomes:
            raise AssertionError("unexpected Yahoo request")
        outcome = self.outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            outcome = outcome()
        return outcome


class FakeFactory:
    def __init__(self, assistant: ScriptedRestAssistant, throttler: Any = None):
        self.assistant = assistant
        self.throttler = throttler
        self.calls = 0

    async def get_rest_assistant(self):
        self.calls += 1
        return self.assistant
