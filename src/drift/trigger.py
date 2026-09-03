"""
Retraining Trigger Controller.

Wraps the DriftEngine with two safety mechanisms before firing retraining:

1. DEBOUNCE  — drift must be confirmed in ≥ N consecutive windows before
               triggering (avoids spurious one-off spikes).
               Default N = settings.drift_debounce_windows (2).

2. COOLDOWN  — after a trigger fires, no new trigger is allowed for
               settings.drift_cooldown_minutes (30) minutes.
               Prevents thrashing when the retrained model hasn't yet
               been deployed.

Additionally, after MAX_CONSECUTIVE_FAILURES consecutive failed retraining
attempts the controller stops auto-triggering and sets needs_human_review=True
so a human can investigate.

State is stored entirely in the PostgreSQL DB (drift_events, retraining_jobs
tables) so the controller is stateless and safe to restart at any time.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from sqlalchemy.orm import Session

from src.config import get_settings
from src.db.audit import write as audit_write, Actions
from src.db.models import DriftEvent, RetrainingJob
from src.drift.engine import DriftEngine, CompositeResult


@dataclass
class TriggerDecision:
    window_id: str
    drift_result: CompositeResult
    should_trigger: bool
    reason: str                          # human-readable explanation
    consecutive_drifts: int = 0
    cooldown_remaining_secs: float = 0.0
    needs_human_review: bool = False
    drift_event_id: Optional[int] = None


class TriggerController:
    """
    Stateless trigger controller: all state is read from the DB on each call.

    Usage:
        controller = TriggerController()
        decision = controller.evaluate(reference_df, current_df, db, window_id)
        if decision.should_trigger:
            run_retraining_job(decision.drift_event_id)
    """

    def __init__(self):
        cfg = get_settings()
        self._debounce = cfg.drift_debounce_windows
        self._cooldown_mins = cfg.drift_cooldown_minutes
        self._max_failures = cfg.max_consecutive_failures
        self._engine = DriftEngine()

    # ── Public API ─────────────────────────────────────────────────────────────

    def evaluate(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        db: Session,
        window_id: str = "unknown",
    ) -> TriggerDecision:
        """
        Run drift detection and decide whether to trigger retraining.

        Persists a DriftEvent row regardless of the trigger decision.
        Returns a TriggerDecision; does NOT start the retraining job itself.
        """
        composite = self._engine.evaluate(reference, current, window_id)

        # 1. Persist the drift event
        event = DriftEvent(
            window_id=window_id,
            method="composite",
            score=composite.composite_score,
            threshold=composite.composite_threshold,
            is_drift=composite.is_drift,
            triggered_retrain=False,
            details=composite.to_dict(),
        )
        db.add(event)
        db.flush()   # get the id without committing yet

        decision = self._make_decision(composite, event.id, db)

        # 2. Update event with trigger outcome
        event.triggered_retrain = decision.should_trigger
        db.commit()
        db.refresh(event)

        decision.drift_event_id = event.id

        # 3. Audit
        action = Actions.DRIFT_DETECTED if composite.is_drift else Actions.DRIFT_HEALTHY
        audit_write(
            db,
            actor="system",
            action=action,
            entity_type="drift_event",
            entity_id=str(event.id),
            details={
                "window_id": window_id,
                "score": composite.composite_score,
                "is_drift": composite.is_drift,
                "should_trigger": decision.should_trigger,
                "reason": decision.reason,
            },
        )
        if decision.should_trigger:
            audit_write(
                db,
                actor="system",
                action=Actions.RETRAIN_TRIGGERED,
                entity_type="drift_event",
                entity_id=str(event.id),
            )

        return decision

    # ── Internal logic ─────────────────────────────────────────────────────────

    def _make_decision(
        self,
        composite: CompositeResult,
        current_event_id: int,
        db: Session,
    ) -> TriggerDecision:
        window_id = composite.window_id

        # Not drifted → no trigger
        if not composite.is_drift:
            return TriggerDecision(
                window_id=window_id,
                drift_result=composite,
                should_trigger=False,
                reason="no_drift",
            )

        # Check consecutive failures → human review required?
        if self._needs_human_review(db):
            return TriggerDecision(
                window_id=window_id,
                drift_result=composite,
                should_trigger=False,
                reason="human_review_required",
                needs_human_review=True,
            )

        # Check cooldown
        cooldown_remaining = self._cooldown_remaining_secs(db)
        if cooldown_remaining > 0:
            return TriggerDecision(
                window_id=window_id,
                drift_result=composite,
                should_trigger=False,
                reason="cooldown_active",
                cooldown_remaining_secs=cooldown_remaining,
            )

        # Check debounce — count consecutive drifted events BEFORE this one
        consecutive_before = self._consecutive_drift_windows(db, exclude_id=current_event_id)
        # +1 for the current window
        consecutive_including_current = consecutive_before + 1

        if consecutive_including_current < self._debounce:
            return TriggerDecision(
                window_id=window_id,
                drift_result=composite,
                should_trigger=False,
                reason=f"debounce_{consecutive_including_current}/{self._debounce}",
                consecutive_drifts=consecutive_including_current,
            )

        # All checks passed — trigger
        return TriggerDecision(
            window_id=window_id,
            drift_result=composite,
            should_trigger=True,
            reason="drift_confirmed",
            consecutive_drifts=consecutive_including_current,
        )

    def _consecutive_drift_windows(self, db: Session, exclude_id: int = -1) -> int:
        """
        Count how many of the most recent drift events (excluding the current one)
        were is_drift=True, stopping at the first non-drift window.
        """
        recent = (
            db.query(DriftEvent)
            .filter(DriftEvent.id != exclude_id)
            .order_by(DriftEvent.detected_at.desc())
            .limit(self._debounce + 5)
            .all()
        )
        count = 0
        for event in recent:
            if event.is_drift:
                count += 1
            else:
                break
        return count

    def _cooldown_remaining_secs(self, db: Session) -> float:
        """
        Return seconds remaining in the cooldown after the most recent trigger.
        Returns 0 if no active cooldown.
        """
        last_trigger = (
            db.query(RetrainingJob)
            .order_by(RetrainingJob.started_at.desc())
            .first()
        )
        if last_trigger is None or last_trigger.started_at is None:
            return 0.0

        now = datetime.datetime.now(datetime.timezone.utc)
        started = last_trigger.started_at
        if started.tzinfo is None:
            started = started.replace(tzinfo=datetime.timezone.utc)

        elapsed = (now - started).total_seconds()
        cooldown_secs = self._cooldown_mins * 60
        remaining = cooldown_secs - elapsed
        return max(0.0, remaining)

    def _needs_human_review(self, db: Session) -> bool:
        """
        True if the last N retraining jobs all failed.
        """
        recent_jobs = (
            db.query(RetrainingJob)
            .order_by(RetrainingJob.started_at.desc())
            .limit(self._max_failures)
            .all()
        )
        if len(recent_jobs) < self._max_failures:
            return False
        return all(j.status == "failed" for j in recent_jobs)

    # ── Manual override ────────────────────────────────────────────────────────

    def reset_cooldown(self, db: Session):
        """
        Admin override: remove the active cooldown by backdating the last job.
        Used by the /retrain/trigger endpoint when an admin forces retraining.
        """
        last_trigger = (
            db.query(RetrainingJob)
            .order_by(RetrainingJob.started_at.desc())
            .first()
        )
        if last_trigger and last_trigger.started_at:
            far_past = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
                minutes=self._cooldown_mins + 1
            )
            last_trigger.started_at = far_past
            db.commit()
