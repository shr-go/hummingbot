from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest

from controllers.generic.equity_leveraged_etf_arbitrage.anchor_repository import (
    F004AnchorRepositoryAdapter,
)
from controllers.generic.equity_leveraged_etf_arbitrage.controller import (
    EquityLeveragedEtfArbitrageController,
    PairEpochFacts,
)
from controllers.generic.equity_leveraged_etf_arbitrage.nav import (
    AnchorAlert,
    AnchorCycleCoordinator,
    AnchorRuntimeStatus,
    NavStage,
    SessionStageCoordinator,
    StageIntent,
    StageIntentKind,
)
from controllers.generic.equity_leveraged_etf_arbitrage.shadow import (
    ShadowPairInput,
    ShadowPlanner,
)
from controllers.generic.equity_leveraged_etf_arbitrage.status import (
    ControllerOperationalStatus,
    OperationalAlert,
    redact_public_text,
)
from hummingbot.client.config.client_config_map import ClientConfigMap
from hummingbot.client.config.config_helpers import ClientConfigAdapter
from hummingbot.data_feed.yahoo_finance.acquisition import (
    AnchorRepositoryKey,
    AnchorRepositoryV2,
    AnchorPollingCheckpoint,
    CheckpointIntegrityError,
    FinalizedAnchorStatus,
)
from hummingbot.model.leveraged_etf_repository import (
    AnchorIntegrityError,
    AnchorPollingCheckpointV1,
    AnchorRepositoryV1,
    AnchorRevisionConflict,
    PairScopedAnchorRepository,
)
from hummingbot.model.sql_connection_manager import SQLConnectionManager, SQLConnectionType
from hummingbot.strategy_v2.leveraged_etf_arbitrage.config import SessionName
from hummingbot.strategy_v2.models.executor_actions import CreateExecutorAction
from test.hummingbot.data_feed.yahoo_finance.conftest import (
    FakeClock,
    OFFICIAL_CLOSE,
    TARGET_SESSION_DATE,
    load_nav_config,
)
from test.hummingbot.data_feed.yahoo_finance.test_pair_scoped_anchor import (
    PairProvider,
    acquisition,
    finalized_pair,
    observation_pair,
)
from test.controllers.generic.equity_leveraged_etf_arbitrage.test_controller_actions import (
    _actions,
    _Connector,
    _configured_pair,
    _controller,
    _epoch,
    _frozen_pair,
)


D = Decimal


def _open_manager(db_path: Path) -> SQLConnectionManager:
    return SQLConnectionManager(
        ClientConfigAdapter(ClientConfigMap()),
        SQLConnectionType.TRADE_FILLS,
        db_path=str(db_path),
    )


@pytest.mark.asyncio
async def test_f003_pair_scoped_adapter_is_lossless_durable_and_never_falls_back_to_cycle_only(tmp_path: Path):
    """The production bridge is the only place F003 values become F004 opaque envelopes."""

    manager = _open_manager(tmp_path / "anchors.sqlite")
    try:
        adapter = F004AnchorRepositoryAdapter(PairScopedAnchorRepository(manager))
        assert isinstance(adapter, AnchorRepositoryV2)

        sndk_acquisition, sndk_empty, sndk_final = await finalized_pair(
            "sndk_snxx", "SNDK", "SNXX", 1
        )
        intc_acquisition, intc_empty, intc_final = await finalized_pair(
            "intc_intw", "INTC", "INTW", 5
        )
        sndk_key = AnchorRepositoryKey.create("sndk_snxx", sndk_empty.cycle_id)
        intc_key = AnchorRepositoryKey.create("intc_intw", intc_empty.cycle_id)

        assert adapter.compare_and_set_checkpoint(sndk_key, sndk_empty, expected_revision=0) == sndk_empty
        assert adapter.compare_and_set_checkpoint(intc_key, intc_empty, expected_revision=0) == intc_empty
        reset_sndk = replace(
            sndk_empty,
            attempt=1,
            next_poll_utc=sndk_empty.official_close_utc + timedelta(seconds=1),
            revision=2,
            integrity_hash=None,
        )
        assert adapter.compare_and_set_checkpoint(sndk_key, reset_sndk, expected_revision=1) == reset_sndk
        assert adapter.load(sndk_key).to_recovery_fields() == reset_sndk.to_recovery_fields()
        assert adapter.load(intc_key).to_recovery_fields() == intc_empty.to_recovery_fields()

        with pytest.raises(AnchorRevisionConflict):
            adapter.compare_and_set_checkpoint(sndk_key, reset_sndk, expected_revision=1)
        with pytest.raises(CheckpointIntegrityError, match="pair|key"):
            adapter.compare_and_set_checkpoint(sndk_key, intc_empty, expected_revision=2)
        with pytest.raises(CheckpointIntegrityError, match="pair|key"):
            adapter.finalize_if_absent(sndk_key, intc_final.candidate, expected_revision=2)

        assert adapter.finalize_if_absent(sndk_key, sndk_final.candidate, expected_revision=2) == sndk_final.candidate
        assert adapter.finalize_if_absent(sndk_key, sndk_final.candidate, expected_revision=2) == sndk_final.candidate
        assert adapter.finalize_if_absent(intc_key, intc_final.candidate, expected_revision=1) == intc_final.candidate
        assert adapter.load(sndk_key).to_evidence_fields() == sndk_final.candidate.to_evidence_fields()

        _, _, conflicting = await finalized_pair("sndk_snxx", "SNDK", "SNXX", 5)
        with pytest.raises(AnchorIntegrityError, match="evidence|immutable|conflict"):
            adapter.finalize_if_absent(sndk_key, conflicting.candidate, expected_revision=2)

        stock, etf = observation_pair(
            "SNDK", "SNXX", OFFICIAL_CLOSE + timedelta(seconds=90), "a", "b", 5
        )
        revised_etf = replace(etf, close=etf.close + D("0.01"))
        assessment = sndk_acquisition.assess_finalized(sndk_final.candidate, stock, revised_etf)
        assert assessment.status is FinalizedAnchorStatus.DATA_REVISION
        adapter.append_revision_observation(sndk_key, assessment.revision_observation)
        adapter.append_revision_observation(sndk_key, assessment.revision_observation)
        assert adapter.revision_observations(sndk_key) == (assessment.revision_observation,)

        intc_stock, intc_etf = observation_pair(
            "INTC", "INTW", OFFICIAL_CLOSE + timedelta(seconds=90), "c", "d", 5
        )
        intc_assessment = intc_acquisition.assess_finalized(
            intc_final.candidate,
            intc_stock,
            replace(intc_etf, close=intc_etf.close + D("0.01")),
        )
        with pytest.raises(CheckpointIntegrityError, match="pair|key"):
            adapter.append_revision_observation(sndk_key, intc_assessment.revision_observation)

        fields = sndk_empty.to_recovery_fields()
        with pytest.raises(CheckpointIntegrityError, match="fields"):
            AnchorPollingCheckpoint.from_recovery_fields({**fields, "unexpected": "field"})
        with pytest.raises(CheckpointIntegrityError, match="fields"):
            AnchorPollingCheckpoint.from_recovery_fields(
                {name: value for name, value in fields.items() if name != "pair_id"}
            )

        legacy_checkpoint = AnchorPollingCheckpoint.from_contract_fields(sndk_empty.to_contract_fields())
        with pytest.raises(CheckpointIntegrityError, match="version 3|pair"):
            adapter.compare_and_set_checkpoint(sndk_key, legacy_checkpoint, expected_revision=2)
        with pytest.raises((TypeError, CheckpointIntegrityError), match="key|AnchorRepositoryKey"):
            adapter.load(sndk_key.cycle_id)
        assert not hasattr(adapter, "load_cycle")
    finally:
        manager.engine.dispose()

    reopened = _open_manager(tmp_path / "anchors.sqlite")
    try:
        restored = F004AnchorRepositoryAdapter(PairScopedAnchorRepository(reopened))
        assert restored.load(sndk_key).to_evidence_fields() == sndk_final.candidate.to_evidence_fields()
        assert restored.load(intc_key).to_evidence_fields() == intc_final.candidate.to_evidence_fields()
    finally:
        reopened.engine.dispose()

    legacy_repository = AnchorRepositoryV1(_open_manager(tmp_path / "legacy.sqlite"))
    try:
        with pytest.raises(AnchorIntegrityError, match="pair|cycle-only|legacy"):
            legacy_repository.load_opaque(sndk_key.cycle_id)
    finally:
        legacy_repository.sql_manager.engine.dispose()

    historical = AnchorPollingCheckpointV1(
        cycle_id="xnys-2026-07-17",
        target_session_date="2026-07-17",
        official_close_utc="2026-07-17T20:00:00.000000Z",
        deadline_utc="2026-07-17T20:10:00.000000Z",
        attempt=0,
        next_poll_utc="2026-07-17T20:00:00.000000Z",
        confirmation_count=0,
        candidate_stock_close=None,
        candidate_etf_close=None,
        candidate_stock_raw_response_hash=None,
        candidate_etf_raw_response_hash=None,
        stock_received_at_utc=None,
        etf_received_at_utc=None,
        revision=1,
    )
    assert historical.schema_version == 1


@pytest.mark.asyncio
async def test_anchor_cycle_restarts_with_remaining_deadline_and_records_revisions(tmp_path: Path):
    """F003 acquisition stays persistence-free while F006 preserves its original deadline."""

    db_path = tmp_path / "restart.sqlite"
    first_clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=60))
    first_acquisition = acquisition(
        first_clock,
        PairProvider(
            observation_pair("SNDK", "SNXX", first_clock.utcnow(), "1", "2", 1),
        ),
    )
    first_manager = _open_manager(db_path)
    try:
        first_repository = F004AnchorRepositoryAdapter(PairScopedAnchorRepository(first_manager))
        first_coordinator = AnchorCycleCoordinator(
            acquisition=first_acquisition,
            repository=first_repository,
            utc_clock=first_clock.utcnow,
        )
        first = await first_coordinator.refresh(
            pair_id="sndk_snxx",
            stock_symbol="SNDK",
            etf_symbol="SNXX",
            cycle_id="xnys-2026-07-17",
            target_session_date=TARGET_SESSION_DATE,
            official_close_utc=OFFICIAL_CLOSE,
        )
        assert first.status is AnchorRuntimeStatus.ACQUIRING
        assert first.checkpoint is not None
        assert first.remaining_deadline_seconds == D("540")
    finally:
        first_manager.engine.dispose()

    second_clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=65.15))
    second_acquisition = acquisition(
        second_clock,
        PairProvider(
            observation_pair("SNDK", "SNXX", second_clock.utcnow(), "3", "4", 2),
        ),
    )
    second_manager = _open_manager(db_path)
    try:
        second_repository = F004AnchorRepositoryAdapter(PairScopedAnchorRepository(second_manager))
        second_coordinator = AnchorCycleCoordinator(
            acquisition=second_acquisition,
            repository=second_repository,
            utc_clock=second_clock.utcnow,
        )
        finalized = await second_coordinator.refresh(
            pair_id="sndk_snxx",
            stock_symbol="SNDK",
            etf_symbol="SNXX",
            cycle_id="xnys-2026-07-17",
            target_session_date=TARGET_SESSION_DATE,
            official_close_utc=OFFICIAL_CLOSE,
        )
        assert finalized.status is AnchorRuntimeStatus.AVAILABLE
        assert finalized.candidate is not None
        assert finalized.checkpoint is None
        anchor_facts = finalized.to_controller_anchor_facts(raw_bp=D("100"), net_bp=D("80"))
        assert anchor_facts is not None
        assert (anchor_facts.nav_cycle_id, anchor_facts.s0, anchor_facts.l0, anchor_facts.h) == (
            "xnys-2026-07-17",
            finalized.candidate.stock_close,
            finalized.candidate.etf_close,
            finalized.candidate.hedge_ratio,
        )

        stock, etf = observation_pair(
            "SNDK", "SNXX", OFFICIAL_CLOSE + timedelta(seconds=90), "5", "6", 3
        )
        revised = second_coordinator.assess_revision(
            finalized.key,
            stock,
            replace(etf, close=etf.close + D("0.01")),
        )
        assert revised.status is AnchorRuntimeStatus.AVAILABLE
        assert revised.alerts == (AnchorAlert.DATA_REVISION,)
        assert len(second_repository.revision_observations(finalized.key)) == 1

        late_clock = FakeClock(OFFICIAL_CLOSE + timedelta(seconds=601))
        late = AnchorCycleCoordinator(
            acquisition=acquisition(late_clock, PairProvider()),
            repository=second_repository,
            utc_clock=late_clock.utcnow,
        )
        unavailable = await late.refresh(
            pair_id="intc_intw",
            stock_symbol="INTC",
            etf_symbol="INTW",
            cycle_id="xnys-2026-07-17",
            target_session_date=TARGET_SESSION_DATE,
            official_close_utc=OFFICIAL_CLOSE,
        )
        assert unavailable.status is AnchorRuntimeStatus.ANCHOR_UNAVAILABLE
        assert second_repository.load(unavailable.key) is None
    finally:
        second_manager.engine.dispose()


def test_nav_stages_observe_exact_boundaries_pair_local_anomalies_and_normal_fallback():
    nav = load_nav_config()
    coordinator = SessionStageCoordinator(nav)
    normal_time = OFFICIAL_CLOSE - timedelta(minutes=30, microseconds=1)
    normal = coordinator.evaluate(
        pair_id="sndk_snxx",
        cycle_id="xnys-2026-07-17",
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=normal_time,
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("199.9999"),
        p99_bp=D("100"),
    )
    assert normal.stage is NavStage.NORMAL
    assert normal.entry_allowed
    assert not normal.pair_paused

    close_30 = coordinator.evaluate(
        pair_id="sndk_snxx",
        cycle_id="xnys-2026-07-17",
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=OFFICIAL_CLOSE - timedelta(minutes=30),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    assert close_30.stage is NavStage.CLOSE_30
    assert not close_30.entry_allowed
    assert {intent.kind for intent in close_30.intents} == {
        StageIntentKind.BLOCK_NEW_EXPOSURE,
        StageIntentKind.CANCEL_ENTRY_MAKERS,
        StageIntentKind.REQUEST_MAKER_EXIT,
    }
    epoch_facts = PairEpochFacts(
        pair_id="sndk_snxx",
        frozen_pair=None,
        anchor=None,
        market_data_fresh=True,
        bracket_data_fresh=True,
        nav_decision=close_30,
    )
    assert epoch_facts.nav_decision is close_30

    close_1 = coordinator.evaluate(
        pair_id="sndk_snxx",
        cycle_id="xnys-2026-07-17",
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=OFFICIAL_CLOSE - timedelta(seconds=60),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    assert close_1.stage is NavStage.CLOSE_1
    assert close_1.emergency_market_requested
    assert StageIntentKind.REQUEST_EMERGENCY_MARKET_FLATTEN in {
        intent.kind for intent in close_1.intents
    }

    anomalous = coordinator.evaluate(
        pair_id="sndk_snxx",
        cycle_id="xnys-2026-07-17",
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=normal_time,
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("200.0001"),
        p99_bp=D("100"),
    )
    assert anomalous.stage is NavStage.NORMAL
    assert anomalous.pair_paused and not anomalous.entry_allowed
    assert StageIntentKind.PAUSE_NEW_EXPOSURE in {intent.kind for intent in anomalous.intents}

    cross_cycle = coordinator.evaluate(
        pair_id="sndk_snxx",
        cycle_id="xnys-2026-07-17",
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=normal_time,
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        active_holding_cycle_id="xnys-2026-07-16",
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    assert cross_cycle.operational_status is AnchorRuntimeStatus.RECOVERY_REQUIRED
    assert not cross_cycle.entry_allowed
    assert StageIntentKind.RECOVERY_REQUIRED in {intent.kind for intent in cross_cycle.intents}

    stale = coordinator.evaluate(
        pair_id="sndk_snxx",
        cycle_id="xnys-2026-07-17",
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=normal_time,
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        anchor_cycle_id="xnys-2026-07-16",
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    assert stale.operational_status is AnchorRuntimeStatus.STALE
    assert not stale.entry_allowed

    early_close = OFFICIAL_CLOSE - timedelta(hours=3)
    assert coordinator.evaluate(
        pair_id="intc_intw",
        cycle_id="xnys-2026-07-17",
        official_close_utc=early_close,
        now_utc=early_close - timedelta(minutes=30),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("1"),
        p99_bp=D("100"),
    ).stage is NavStage.CLOSE_30
    assert coordinator.select_session(
        SessionName.EXTENDED, {SessionName.REGULAR: object()}
    ) is SessionName.REGULAR


@pytest.mark.parametrize(
    ("current_etf", "current_stock", "expected_operation"),
    (
        (D("0"), D("0"), "OPEN"),
        (D("-1"), D("1"), "ADD"),
    ),
)
def test_independent_new_entry_cutoff_blocks_controller_open_and_add(
    current_etf: Decimal,
    current_stock: Decimal,
    expected_operation: str,
):
    nav = load_nav_config().model_copy(update={"new_entry_cutoff_minutes": 45})
    coordinator = SessionStageCoordinator(nav)
    configured = _configured_pair(1)
    frozen = _frozen_pair(
        configured,
        current_etf=current_etf,
        current_stock=current_stock,
    )
    base_epoch = _epoch(frozen)
    normal = coordinator.evaluate(
        pair_id=configured.id,
        cycle_id=base_epoch.pairs[0].anchor.nav_cycle_id,
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=OFFICIAL_CLOSE - timedelta(minutes=45, microseconds=1),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    cutoff = coordinator.evaluate(
        pair_id=configured.id,
        cycle_id=base_epoch.pairs[0].anchor.nav_cycle_id,
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=OFFICIAL_CLOSE - timedelta(minutes=45),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    epoch = replace(
        base_epoch,
        pairs=(replace(base_epoch.pairs[0], nav_decision=cutoff),),
    )
    controller, _ = _controller((configured,), (epoch,))

    actions = _actions(controller)

    assert not any(
        isinstance(action, CreateExecutorAction)
        and action.executor_config.operation.value == expected_operation
        for action in actions
    )
    assert normal.entry_allowed
    assert cutoff.stage is NavStage.NEW_ENTRY_CUTOFF
    assert not cutoff.entry_allowed
    assert StageIntentKind.BLOCK_NEW_EXPOSURE in {intent.kind for intent in cutoff.intents}


@pytest.mark.parametrize(
    ("current_etf", "current_stock", "expected_operation"),
    (
        (D("0"), D("0"), "OPEN"),
        (D("-1"), D("1"), "ADD"),
    ),
)
def test_missing_nav_decision_fails_closed_for_open_and_add_in_close_window(
    current_etf: Decimal,
    current_stock: Decimal,
    expected_operation: str,
):
    configured = _configured_pair(1)
    frozen = _frozen_pair(
        configured,
        current_etf=current_etf,
        current_stock=current_stock,
    )
    base_epoch = _epoch(frozen)
    close_window = SessionStageCoordinator(load_nav_config()).evaluate(
        pair_id=configured.id,
        cycle_id=base_epoch.pairs[0].anchor.nav_cycle_id,
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=OFFICIAL_CLOSE - timedelta(minutes=30),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    assert not close_window.entry_allowed
    # An epoch source that omits the close-window decision cannot bypass the
    # Controller's exposure gate.
    epoch = replace(
        base_epoch,
        pairs=(replace(base_epoch.pairs[0], nav_decision=None),),
    )
    controller, _ = _controller((configured,), (epoch,))

    actions = _actions(controller)

    assert not any(
        isinstance(action, CreateExecutorAction)
        and action.executor_config.operation.value == expected_operation
        for action in actions
    )
    assert controller.last_operational_status is not None
    assert [alert.code for alert in controller.last_operational_status.alerts] == [
        "NAV_DECISION_UNAVAILABLE"
    ]


def test_anomalous_spread_p99_plus_100_is_inclusive_and_pair_local_in_status():
    coordinator = SessionStageCoordinator(load_nav_config())
    arguments = {
        "cycle_id": "xnys-2026-07-17",
        "official_close_utc": OFFICIAL_CLOSE,
        "now_utc": OFFICIAL_CLOSE - timedelta(minutes=30, microseconds=1),
        "anchor_status": AnchorRuntimeStatus.AVAILABLE,
        "p99_bp": D("100"),
    }
    below = coordinator.evaluate(pair_id="pair2", spread_bp=D("199.9999"), **arguments)
    exact = coordinator.evaluate(pair_id="pair1", spread_bp=D("200"), **arguments)
    above = coordinator.evaluate(pair_id="pair3", spread_bp=D("200.0001"), **arguments)

    assert below.entry_allowed and not below.pair_paused
    assert not exact.entry_allowed and exact.pair_paused
    assert not above.entry_allowed and above.pair_paused
    assert StageIntentKind.PAUSE_NEW_EXPOSURE in {intent.kind for intent in exact.intents}
    status_pairs = {
        item["pair_id"]: item
        for item in ControllerOperationalStatus(decisions=(exact, below, above)).to_dict()["pairs"]
    }
    assert status_pairs["pair1"]["pair_paused"]
    assert "PAUSE_NEW_EXPOSURE" in {
        intent["kind"] for intent in status_pairs["pair1"]["intents"]
    }
    assert not status_pairs["pair2"]["pair_paused"]


def test_public_status_and_processed_data_never_export_sensitive_failure_or_intent_text():
    secret_text = (
        'api_key="API_KEY_SENTINEL" '
        '{"token": "JSON_TOKEN_SENTINEL", "secret": "JSON_SECRET_SENTINEL"} '
        "authorization=Bearer AUTHORIZATION_SENTINEL "
        """private_key='-----BEGIN PRIVATE KEY-----
PEM_PRIVATE_KEY_SENTINEL
-----END PRIVATE KEY-----'"""
    )
    coordinator = SessionStageCoordinator(load_nav_config())
    decision = coordinator.evaluate(
        pair_id="pair1",
        cycle_id="xnys-2026-07-17",
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=OFFICIAL_CLOSE - timedelta(hours=2),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    secret_intent = StageIntent(
        pair_id=decision.pair_id,
        cycle_id=decision.cycle_id,
        kind=StageIntentKind.BLOCK_NEW_EXPOSURE,
        reason=secret_text,
    )
    status = ControllerOperationalStatus(
        decisions=(replace(decision, intents=(secret_intent,)),),
        alerts=(OperationalAlert(code="UPSTREAM_FAILURE", message=secret_text),),
    )
    configured = _configured_pair(1)
    controller, _ = _controller(
        (configured,),
        (_epoch(_frozen_pair(configured)),),
        connector=_Connector((RuntimeError(secret_text),)),
    )

    assert _actions(controller) == []
    public = {
        "status": status.to_dict(),
        "status_lines": status.to_lines(),
        "controller_processed_data": controller.processed_data,
        "controller_lines": controller.to_format_status(),
    }
    serialized = json.dumps(public, sort_keys=True)

    for sentinel in (
        "API_KEY_SENTINEL",
        "JSON_TOKEN_SENTINEL",
        "JSON_SECRET_SENTINEL",
        "AUTHORIZATION_SENTINEL",
        "PEM_PRIVATE_KEY_SENTINEL",
    ):
        assert sentinel not in serialized
    assert "[REDACTED]" in serialized


@pytest.mark.parametrize(
    ("secret_text", "sentinels"),
    (
        (
            r'api_key={"nested":"API_KEY_STRUCTURED_SENTINEL","array":["API_KEY_ARRAY_SENTINEL",{"escaped":"value \\\"API_KEY_ESCAPED_SENTINEL\\\""}]}',
            (
                "API_KEY_STRUCTURED_SENTINEL",
                "API_KEY_ARRAY_SENTINEL",
                "API_KEY_ESCAPED_SENTINEL",
            ),
        ),
        (
            """api_key = {
    "nested": "API_KEY_WHITESPACE_SENTINEL",
    "array": [ "API_KEY_SPACED_ARRAY_SENTINEL", { "escaped": "value \\\"API_KEY_SPACED_ESCAPED_SENTINEL\\\"" } ]
}""",
            (
                "API_KEY_WHITESPACE_SENTINEL",
                "API_KEY_SPACED_ARRAY_SENTINEL",
                "API_KEY_SPACED_ESCAPED_SENTINEL",
            ),
        ),
        (
            """private_key = [
    "-----BEGIN PRIVATE KEY-----
PEM_PRIVATE_KEY_ARRAY_SENTINEL
-----END PRIVATE KEY-----",
    { "nested": "PRIVATE_KEY_NESTED_ARRAY_SENTINEL" },
    "PRIVATE_KEY_SPACED_ARRAY_SENTINEL"
]""",
            (
                "PEM_PRIVATE_KEY_ARRAY_SENTINEL",
                "PRIVATE_KEY_NESTED_ARRAY_SENTINEL",
                "PRIVATE_KEY_SPACED_ARRAY_SENTINEL",
            ),
        ),
    ),
)
def test_public_boundaries_fail_closed_for_complete_structured_secret_assignments(
    secret_text: str,
    sentinels: tuple[str, ...],
):
    coordinator = SessionStageCoordinator(load_nav_config())
    decision = coordinator.evaluate(
        pair_id="pair1",
        cycle_id="xnys-2026-07-17",
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=OFFICIAL_CLOSE - timedelta(hours=2),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    alert = OperationalAlert(code="UPSTREAM_FAILURE", message=secret_text)
    intent = StageIntent(
        pair_id=decision.pair_id,
        cycle_id=decision.cycle_id,
        kind=StageIntentKind.BLOCK_NEW_EXPOSURE,
        reason=secret_text,
    )
    status = ControllerOperationalStatus(
        decisions=(replace(decision, intents=(intent,)),),
        alerts=(alert,),
    )
    configured = _configured_pair(1)
    controller, _ = _controller((configured,), (_epoch(_frozen_pair(configured)),))
    controller.last_failure = secret_text
    controller.last_operational_status = status
    controller._set_processed_data()

    public_values = (
        redact_public_text(secret_text),
        json.dumps(alert.to_dict(), sort_keys=True),
        json.dumps(status.to_dict(), sort_keys=True),
        "\n".join(status.to_lines()),
        json.dumps(controller.processed_data, sort_keys=True),
    )

    for serialized in public_values:
        for sentinel in sentinels:
            assert sentinel not in serialized
    assert all("[REDACTED]" in value for value in public_values)


def test_close_stage_decision_is_carried_by_controller_epoch_and_blocks_new_actions():
    configured = _configured_pair(1)
    frozen = _frozen_pair(configured)
    base_epoch = _epoch(frozen)
    decision = SessionStageCoordinator(load_nav_config()).evaluate(
        pair_id=configured.id,
        cycle_id=base_epoch.pairs[0].anchor.nav_cycle_id,
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=OFFICIAL_CLOSE - timedelta(minutes=30),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("1"),
        p99_bp=D("100"),
    )
    epoch = replace(
        base_epoch,
        pairs=(replace(base_epoch.pairs[0], nav_decision=decision),),
    )
    controller, _ = _controller((configured,), (epoch,))

    assert _actions(controller) == []
    assert controller.last_operational_status is not None
    assert controller.processed_data["operational_status"]["pairs"][0]["stage"] == "CLOSE_30"


def test_shadow_plan_is_deterministic_action_free_and_operational_status_redacts():
    nav = load_nav_config()
    coordinator = SessionStageCoordinator(nav)
    decision = coordinator.evaluate(
        pair_id="sndk_snxx",
        cycle_id="xnys-2026-07-17",
        official_close_utc=OFFICIAL_CLOSE,
        now_utc=OFFICIAL_CLOSE - timedelta(hours=2),
        anchor_status=AnchorRuntimeStatus.AVAILABLE,
        spread_bp=D("80"),
        p99_bp=D("100"),
    )
    planner = ShadowPlanner()
    inputs = (
        ShadowPairInput(
            pair_id="intc_intw",
            cycle_id="xnys-2026-07-17",
            target_gross_notional=D("2000"),
            etf_leverage=12,
            stock_leverage=8,
            raw_bp=D("90"),
            net_bp=D("70"),
            nav_decision=decision.for_pair("intc_intw"),
        ),
        ShadowPairInput(
            pair_id="sndk_snxx",
            cycle_id="xnys-2026-07-17",
            target_gross_notional=D("1000"),
            etf_leverage=10,
            stock_leverage=5,
            raw_bp=D("100"),
            net_bp=D("80"),
            nav_decision=decision,
        ),
    )
    first = planner.plan(inputs)
    second = planner.plan(tuple(reversed(inputs)))

    assert first == second
    assert first.pair_ids == ("intc_intw", "sndk_snxx")
    assert first.exchange_actions == ()
    assert not first.has_exchange_side_effects
    assert first.metrics.pair_count == 2
    assert first.metrics.entry_allowed_pair_count == 2
    assert first.metrics.blocked_pair_count == 0
    assert first.pairs[1].etf_leverage == 10
    assert first.pairs[1].stock_leverage == 5

    class _MutationTrapConnector:
        def __init__(self):
            self.calls: list[str] = []

        async def strict_account_preflight(self, *args, **kwargs):
            self.calls.append("preflight")
            raise AssertionError("shadow planning must not preflight or mutate an exchange")

        async def set_leverage_with_result(self, *args, **kwargs):
            self.calls.append("leverage")
            raise AssertionError("shadow planning must not set leverage")

    class _UnusedEpochSource:
        async def build_epoch(self, *args, **kwargs):
            raise AssertionError("shadow planning must not build an exchange epoch")

    class _UnusedReservations:
        def active(self):
            raise AssertionError("shadow planning must not read or mutate reservations")

        def reserve(self, *args, **kwargs):
            raise AssertionError("shadow planning must not reserve")

        def release(self, *args, **kwargs):
            raise AssertionError("shadow planning must not release")

    trap = _MutationTrapConnector()
    controller = EquityLeveragedEtfArbitrageController(
        config=SimpleNamespace(
            id="shadow-controller",
            pairs=(
                SimpleNamespace(
                    id="sndk_snxx",
                    stock_trading_pair="SNDK-USDT",
                    etf_trading_pair="SNXX-USDT",
                    enabled=True,
                ),
            ),
        ),
        market_data_provider=object(),
        actions_queue=object(),
        connector=trap,
        epoch_source=_UnusedEpochSource(),
        reservations=_UnusedReservations(),
    )
    controller_plan = controller.build_shadow_plan((inputs[1],))
    assert controller_plan.exchange_actions == ()
    assert controller.determine_executor_actions() == []
    assert trap.calls == []

    status = ControllerOperationalStatus(
        decisions=(decision,),
        shadow_plan=first,
        alerts=(
            OperationalAlert(
                code="YAHOO_FAILURE",
                message="api_key=do-not-leak private_key=also-do-not-leak",
            ),
        ),
    )
    serialized = json.dumps(status.to_dict(), sort_keys=True)
    assert "do-not-leak" not in serialized
    assert "also-do-not-leak" not in serialized
    assert "api_key=[REDACTED]" in serialized
    assert status.to_lines()[0] == "Equity Leveraged ETF operational status:"
