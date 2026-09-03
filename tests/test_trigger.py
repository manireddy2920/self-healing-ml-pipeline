"""
Module 7 tests — Retraining Trigger Controller (debounce + cooldown).

All tests use an in-memory SQLite DB and synthetic DataFrames.
No MLflow, no HTTP, no external services.
"""
from __future__ import annotations

import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.session import Base
from src.db import models as db_models
from src.drift.trigger import TriggerController, TriggerDecision
from src.ingestion.generator import generate_batch


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def db():
    engine = create_engine(
        "sqlite:///:memory:", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    yield session
    session.close()


@pytest.fixture(scope="module")
def reference():
    return generate_batch(n=1_500, seed=0, drift_alpha=0.0)


@pytest.fixture(scope="module")
def stable():
    return generate_batch(n=800, seed=10, drift_alpha=0.0)


@pytest.fixture(scope="module")
def drifted():
    return generate_batch(n=800, seed=20, drift_alpha=1.0)


def _make_controller(debounce=2, cooldown_mins=30, max_failures=3):
    """Build a TriggerController with overridden settings."""
    ctrl = TriggerController.__new__(TriggerController)
    from src.drift.engine import DriftEngine
    ctrl._debounce = debounce
    ctrl._cooldown_mins = cooldown_mins
    ctrl._max_failures = max_failures
    ctrl._engine = DriftEngine()
    return ctrl


# ══════════════════════════════════════════════════════════════════════════════
# Basic evaluation
# ══════════════════════════════════════════════════════════════════════════════

def test_evaluate_returns_decision(db, reference, stable):
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    decision = ctrl.evaluate(reference, stable, db, window_id="w001")
    assert isinstance(decision, TriggerDecision)


def test_evaluate_persists_drift_event(db, reference, stable):
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    ctrl.evaluate(reference, stable, db, window_id="w002")
    events = db.query(db_models.DriftEvent).all()
    assert len(events) >= 1
    assert events[-1].window_id == "w002"


def test_evaluate_drift_event_has_score(db, reference, drifted):
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    ctrl.evaluate(reference, drifted, db, window_id="w003")
    event = db.query(db_models.DriftEvent).filter_by(window_id="w003").first()
    assert event is not None
    assert 0.0 <= event.score <= 1.0


def test_stable_data_no_trigger(db, reference, stable):
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    decision = ctrl.evaluate(reference, stable, db)
    assert decision.should_trigger is False
    assert decision.reason == "no_drift"


def test_drifted_data_with_debounce_1_triggers(db, reference, drifted):
    """With debounce=1 (single window confirmation), drift triggers immediately."""
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    decision = ctrl.evaluate(reference, drifted, db, window_id="trigger_w")
    assert decision.should_trigger is True
    assert decision.reason == "drift_confirmed"


# ══════════════════════════════════════════════════════════════════════════════
# Debounce
# ══════════════════════════════════════════════════════════════════════════════

def test_debounce_blocks_first_window(db, reference, drifted):
    """With debounce=2, first drifted window should NOT trigger."""
    ctrl = _make_controller(debounce=2, cooldown_mins=0)
    decision = ctrl.evaluate(reference, drifted, db, window_id="d1")
    assert decision.should_trigger is False
    assert "debounce" in decision.reason
    assert decision.consecutive_drifts == 1


def test_debounce_triggers_on_second_window(db, reference, drifted):
    """With debounce=2, two consecutive drifted windows should trigger."""
    ctrl = _make_controller(debounce=2, cooldown_mins=0)
    ctrl.evaluate(reference, drifted, db, window_id="d2a")   # first: no trigger
    decision = ctrl.evaluate(reference, drifted, db, window_id="d2b")  # second: trigger
    assert decision.should_trigger is True
    assert decision.consecutive_drifts >= 2


def test_debounce_resets_after_stable_window(db, reference, stable, drifted):
    """
    Pattern: drift → stable → drift should NOT trigger on the last drift
    (debounce=2, stable window resets the streak).
    """
    ctrl = _make_controller(debounce=2, cooldown_mins=0)
    ctrl.evaluate(reference, drifted, db, window_id="dr1")
    ctrl.evaluate(reference, stable, db, window_id="st1")   # resets streak
    decision = ctrl.evaluate(reference, drifted, db, window_id="dr2")
    assert decision.should_trigger is False, (
        f"Expected no trigger after debounce reset, got: {decision.reason}"
    )


def test_debounce_3_needs_three_windows(db, reference, drifted):
    ctrl = _make_controller(debounce=3, cooldown_mins=0)
    d1 = ctrl.evaluate(reference, drifted, db, window_id="d3a")
    d2 = ctrl.evaluate(reference, drifted, db, window_id="d3b")
    d3 = ctrl.evaluate(reference, drifted, db, window_id="d3c")
    assert d1.should_trigger is False
    assert d2.should_trigger is False
    assert d3.should_trigger is True


# ══════════════════════════════════════════════════════════════════════════════
# Cooldown
# ══════════════════════════════════════════════════════════════════════════════

def test_cooldown_blocks_immediate_re_trigger(db, reference, drifted):
    """After a trigger fires, the next drift window should be blocked by cooldown."""
    ctrl = _make_controller(debounce=1, cooldown_mins=60)
    d1 = ctrl.evaluate(reference, drifted, db, window_id="c1")
    assert d1.should_trigger is True

    # Simulate a retraining job being created right now
    job = db_models.RetrainingJob(
        status="running",
        triggered_by="system",
        started_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db.add(job)
    db.commit()

    d2 = ctrl.evaluate(reference, drifted, db, window_id="c2")
    assert d2.should_trigger is False
    assert d2.reason == "cooldown_active"
    assert d2.cooldown_remaining_secs > 0


def test_cooldown_zero_allows_immediate_retrigger(db, reference, drifted):
    """cooldown_mins=0 means no cooldown — every consecutive window can trigger."""
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    # Add an ancient job (started 2h ago)
    ancient_job = db_models.RetrainingJob(
        status="success",
        triggered_by="system",
        started_at=datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(hours=2),
    )
    db.add(ancient_job)
    db.commit()

    decision = ctrl.evaluate(reference, drifted, db, window_id="c3")
    assert decision.should_trigger is True


def test_cooldown_expires_after_time(db, reference, drifted):
    """A job that finished cooldown_mins ago should NOT block a new trigger."""
    ctrl = _make_controller(debounce=1, cooldown_mins=30)
    # Job started 35 minutes ago → cooldown expired
    old_job = db_models.RetrainingJob(
        status="success",
        triggered_by="system",
        started_at=datetime.datetime.now(datetime.timezone.utc)
        - datetime.timedelta(minutes=35),
    )
    db.add(old_job)
    db.commit()

    decision = ctrl.evaluate(reference, drifted, db, window_id="c4")
    assert decision.should_trigger is True


def test_reset_cooldown_clears_block(db, reference, drifted):
    """reset_cooldown() should allow an immediate trigger."""
    ctrl = _make_controller(debounce=1, cooldown_mins=60)
    job = db_models.RetrainingJob(
        status="running",
        triggered_by="system",
        started_at=datetime.datetime.now(datetime.timezone.utc),
    )
    db.add(job)
    db.commit()

    # Confirm cooldown is active
    d1 = ctrl.evaluate(reference, drifted, db, window_id="rc1")
    assert d1.should_trigger is False

    # Reset cooldown
    ctrl.reset_cooldown(db)

    d2 = ctrl.evaluate(reference, drifted, db, window_id="rc2")
    assert d2.should_trigger is True


# ══════════════════════════════════════════════════════════════════════════════
# Human review flag
# ══════════════════════════════════════════════════════════════════════════════

def test_human_review_after_max_failures(db, reference, drifted):
    """After max_failures consecutive failed jobs, trigger should pause."""
    ctrl = _make_controller(debounce=1, cooldown_mins=0, max_failures=3)

    # Simulate 3 consecutive failed jobs
    for i in range(3):
        job = db_models.RetrainingJob(
            status="failed",
            triggered_by="system",
            started_at=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(minutes=i * 2),
        )
        db.add(job)
    db.commit()

    decision = ctrl.evaluate(reference, drifted, db, window_id="hr1")
    assert decision.should_trigger is False
    assert decision.needs_human_review is True
    assert decision.reason == "human_review_required"


def test_human_review_not_set_with_success_in_history(db, reference, drifted):
    """If at least one job succeeded recently, human review should NOT be set."""
    ctrl = _make_controller(debounce=1, cooldown_mins=0, max_failures=3)

    # 2 failed + 1 success (order: most recent first)
    times = [0, 1, 2]
    statuses = ["failed", "failed", "success"]
    for i, status in zip(times, statuses):
        job = db_models.RetrainingJob(
            status=status,
            triggered_by="system",
            started_at=datetime.datetime.now(datetime.timezone.utc)
            - datetime.timedelta(minutes=i * 5 + 200),
        )
        db.add(job)
    db.commit()

    decision = ctrl.evaluate(reference, drifted, db, window_id="hr2")
    assert decision.needs_human_review is False


# ══════════════════════════════════════════════════════════════════════════════
# Audit log integration
# ══════════════════════════════════════════════════════════════════════════════

def test_audit_log_written_on_evaluate(db, reference, stable):
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    ctrl.evaluate(reference, stable, db, window_id="audit_test")
    logs = db.query(db_models.AuditLog).all()
    actions = [l.action for l in logs]
    assert "DRIFT_HEALTHY" in actions or "DRIFT_DETECTED" in actions


def test_audit_log_retrain_triggered_written(db, reference, drifted):
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    decision = ctrl.evaluate(reference, drifted, db, window_id="audit_trigger")
    if decision.should_trigger:
        logs = db.query(db_models.AuditLog).all()
        actions = [l.action for l in logs]
        assert "RETRAIN_TRIGGERED" in actions


# ══════════════════════════════════════════════════════════════════════════════
# Decision fields
# ══════════════════════════════════════════════════════════════════════════════

def test_decision_has_drift_event_id(db, reference, stable):
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    decision = ctrl.evaluate(reference, stable, db, window_id="id_test")
    assert decision.drift_event_id is not None
    assert isinstance(decision.drift_event_id, int)


def test_decision_drift_result_attached(db, reference, drifted):
    ctrl = _make_controller(debounce=1, cooldown_mins=0)
    decision = ctrl.evaluate(reference, drifted, db, window_id="dr_test")
    from src.drift.engine import CompositeResult
    assert isinstance(decision.drift_result, CompositeResult)
