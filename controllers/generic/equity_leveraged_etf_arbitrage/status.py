"""Secret-safe operational status for Controller, shadow, and recovery handoff."""

from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal

from controllers.generic.equity_leveraged_etf_arbitrage.nav import SessionStageDecision
from controllers.generic.equity_leveraged_etf_arbitrage.shadow import ShadowPlan


__all__ = ["ControllerOperationalStatus", "OperationalAlert", "redact_public_text"]


_PEM_PRIVATE_KEY = re.compile(
    (
        r"-----BEGIN(?: [A-Z0-9]+)* PRIVATE KEY-----.*?"
        r"-----END(?: [A-Z0-9]+)* PRIVATE KEY-----"
    ),
    flags=re.IGNORECASE | re.DOTALL,
)
_SECRET_VALUE = re.compile(
    r"""(?ix)
    (?P<prefix>
        (?P<quote>[\"'])?
        (?:
            api[_-]?key
            | private[_-]?key
            | (?:client[_-]?)?secret
            | (?:access[_-]?|refresh[_-]?)?token
            | password
            | passphrase
            | authorization
        )
        (?(quote)(?P=quote))
        \s*(?:=|:)\s*
    )
    (?:
        (?:Bearer\s+)?
        (?:
            \"(?:\\.|[^\"\\])*\"
            | '(?:\\.|[^'\\])*'
            | [^,;\s}\]]+
        )
    )
    """
)


def _decimal_text(value: Decimal) -> str:
    rendered = format(value, "f")
    return rendered.rstrip("0").rstrip(".") if "." in rendered else rendered


def redact_public_text(message: str) -> str:
    """Remove credential material from text crossing a public status boundary."""

    if not isinstance(message, str):
        raise TypeError("public status text must be a string")
    without_pem = _PEM_PRIVATE_KEY.sub("[REDACTED]", message)
    return _SECRET_VALUE.sub(lambda found: f"{found.group('prefix')}[REDACTED]", without_pem)


@dataclass(frozen=True, slots=True)
class OperationalAlert:
    code: str
    message: str

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("operational alert code must be non-empty")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("operational alert message must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "message": redact_public_text(self.message)}


@dataclass(frozen=True, slots=True)
class ControllerOperationalStatus:
    """Only public operational facts; configuration and credentials never enter it."""

    decisions: tuple[SessionStageDecision, ...]
    shadow_plan: ShadowPlan | None = None
    alerts: tuple[OperationalAlert, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.decisions, tuple) or any(
            not isinstance(decision, SessionStageDecision) for decision in self.decisions
        ):
            raise TypeError("operational decisions must be an immutable tuple")
        pair_ids = tuple(decision.pair_id for decision in self.decisions)
        if pair_ids != tuple(sorted(pair_ids)) or len(set(pair_ids)) != len(pair_ids):
            raise ValueError("operational decisions must be unique and sorted by pair")
        if self.shadow_plan is not None and not isinstance(self.shadow_plan, ShadowPlan):
            raise TypeError("shadow_plan must be a ShadowPlan when supplied")
        if not isinstance(self.alerts, tuple) or any(not isinstance(alert, OperationalAlert) for alert in self.alerts):
            raise TypeError("operational alerts must be an immutable tuple")

    def to_dict(self) -> dict[str, object]:
        pairs = [
            {
                "pair_id": decision.pair_id,
                "cycle_id": decision.cycle_id,
                "stage": decision.stage.value,
                "anchor_status": decision.operational_status.value,
                "entry_allowed": decision.entry_allowed,
                "pair_paused": decision.pair_paused,
                "emergency_market_requested": decision.emergency_market_requested,
                "intents": [
                    {
                        "kind": intent.kind.value,
                        "reason": redact_public_text(intent.reason),
                    }
                    for intent in decision.intents
                ],
            }
            for decision in self.decisions
        ]
        shadow: dict[str, object] | None = None
        if self.shadow_plan is not None:
            shadow = {
                "deterministic_hash": self.shadow_plan.deterministic_hash,
                "pair_ids": list(self.shadow_plan.pair_ids),
                "total_target_gross_notional": _decimal_text(
                    self.shadow_plan.total_target_gross_notional
                ),
                "metrics": {
                    "pair_count": self.shadow_plan.metrics.pair_count,
                    "entry_allowed_pair_count": self.shadow_plan.metrics.entry_allowed_pair_count,
                    "blocked_pair_count": self.shadow_plan.metrics.blocked_pair_count,
                    "emergency_market_request_count": self.shadow_plan.metrics.emergency_market_request_count,
                },
                "exchange_action_count": len(self.shadow_plan.exchange_actions),
                "has_exchange_side_effects": self.shadow_plan.has_exchange_side_effects,
            }
        return {
            "pairs": pairs,
            "shadow": shadow,
            "alerts": [alert.to_dict() for alert in self.alerts],
        }

    def to_lines(self) -> list[str]:
        lines = ["Equity Leveraged ETF operational status:"]
        for decision in self.decisions:
            lines.append(
                "  "
                f"{decision.pair_id}: {decision.stage.value}; "
                f"anchor={decision.operational_status.value}; "
                f"entry_allowed={decision.entry_allowed}; "
                f"emergency_market={decision.emergency_market_requested}"
            )
        if self.shadow_plan is not None:
            lines.append(
                "  "
                f"Shadow: hash={self.shadow_plan.deterministic_hash}; "
                f"exchange_actions={len(self.shadow_plan.exchange_actions)}"
            )
        lines.extend(
            f"  Alert {alert.code}: {redact_public_text(alert.message)}"
            for alert in self.alerts
        )
        return lines
