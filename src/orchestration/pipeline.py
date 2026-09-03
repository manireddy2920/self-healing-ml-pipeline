"""
Prefect 2 Pipeline — Module 11.

Defines the self-healing pipeline as a Prefect flow with tasks for:
  1. drift_check_task    — runs TriggerController on the latest data window
  2. retrain_task        — fires RetrainingJob if trigger says so
  3. validate_task       — ValidationGate (already called by runner, but
                           exposed here for observability in Prefect UI)

Schedule: daily at 03:00 UTC (configurable via PREFECT_SCHEDULE env var).
Also triggered on-demand by the /retrain/trigger API endpoint.

The flow is idempotent: re-running it when no drift is detected is a no-op.
"""
from __future__ import annotations

import datetime
import os

from prefect import flow, task, get_run_logger
from prefect.client.schemas.schedules import CronSchedule

from src.config import get_settings
from src.db.session import SessionLocal, get_engine, Base
from src.ingestion.loader import DataLoader
from src.drift.trigger import TriggerController, TriggerDecision
from src.retraining.runner import run_retraining_job
from src.db.models import RetrainingJob


# ── Tasks ──────────────────────────────────────────────────────────────────────

@task(name="check_drift", retries=1, retry_delay_seconds=30)
def drift_check_task(window_id: str) -> dict:
    """Load the recent data window and run the trigger controller."""
    logger = get_run_logger()
    db = SessionLocal()
    try:
        loader = DataLoader()
        current = loader.get_recent_window()

        if current.empty or len(current) < 100:
            logger.warning("Insufficient data for drift check — loading reference")
            current = loader.load_reference()

        reference = loader.load_reference()

        ctrl = TriggerController()
        decision: TriggerDecision = ctrl.evaluate(
            reference=reference,
            current=current,
            db=db,
            window_id=window_id,
        )

        logger.info(
            f"Drift check [{window_id}]: "
            f"score={decision.drift_result.composite_score:.4f} "
            f"is_drift={decision.drift_result.is_drift} "
            f"should_trigger={decision.should_trigger} "
            f"reason={decision.reason}"
        )
        return {
            "should_trigger": decision.should_trigger,
            "drift_event_id": decision.drift_event_id,
            "composite_score": decision.drift_result.composite_score,
            "is_drift": decision.drift_result.is_drift,
            "reason": decision.reason,
            "needs_human_review": decision.needs_human_review,
        }
    finally:
        db.close()


@task(name="run_retraining", retries=1, retry_delay_seconds=60)
def retrain_task(drift_event_id: int | None, triggered_by: str = "prefect") -> dict:
    """Create and execute a retraining job."""
    logger = get_run_logger()
    db = SessionLocal()
    try:
        job = RetrainingJob(
            drift_event_id=drift_event_id,
            status="pending",
            triggered_by=triggered_by,
        )
        db.add(job)
        db.commit()
        db.refresh(job)
        job_id = job.id
    finally:
        db.close()

    logger.info(f"Starting retraining job {job_id}")
    completed_job = run_retraining_job(job_id)
    logger.info(f"Retraining job {job_id} status: {completed_job.status}")

    return {"job_id": job_id, "status": completed_job.status}


# ── Flow ───────────────────────────────────────────────────────────────────────

@flow(
    name="self_healing_pipeline",
    description="Checks for drift, triggers retraining if confirmed, "
                "validates challenger and promotes or rolls back.",
    version="1.0.0",
)
def self_healing_pipeline(
    window_id: str | None = None,
    force_retrain: bool = False,
    triggered_by: str = "schedule",
) -> dict:
    """
    Main Prefect flow.

    Parameters
    ----------
    window_id     : identifier for this data window (auto-generated if None)
    force_retrain : bypass drift check and always retrain
    triggered_by  : "schedule" | "manual" | "api"
    """
    logger = get_run_logger()

    if window_id is None:
        window_id = datetime.datetime.now(datetime.timezone.utc).strftime(
            "window_%Y%m%d_%H%M%S"
        )

    logger.info(f"Pipeline run started: window_id={window_id}")

    # Step 1: Drift check
    if force_retrain:
        drift_outcome = {
            "should_trigger": True,
            "drift_event_id": None,
            "composite_score": 1.0,
            "is_drift": True,
            "reason": "force_retrain",
            "needs_human_review": False,
        }
        logger.info("Force retrain requested — skipping drift check")
    else:
        drift_outcome = drift_check_task(window_id=window_id)

    # Step 2: Retrain if triggered
    retrain_outcome = None
    if drift_outcome["should_trigger"]:
        if drift_outcome.get("needs_human_review"):
            logger.warning(
                "Human review required — max consecutive failures reached. "
                "Automatic retraining suspended."
            )
        else:
            retrain_outcome = retrain_task(
                drift_event_id=drift_outcome["drift_event_id"],
                triggered_by=triggered_by,
            )

    result = {
        "window_id": window_id,
        "drift": drift_outcome,
        "retrain": retrain_outcome,
        "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    logger.info(f"Pipeline run complete: {result}")
    return result


# ── Deployment helper ──────────────────────────────────────────────────────────

def deploy():
    """Register the flow with a daily schedule. Run once after `prefect server start`."""
    from prefect.deployments import Deployment

    deployment = Deployment.build_from_flow(
        flow=self_healing_pipeline,
        name="daily-drift-check",
        schedule=CronSchedule(cron="0 3 * * *", timezone="UTC"),
        work_queue_name="default",
        tags=["mlops", "drift"],
    )
    deployment.apply()
    print("Deployment 'daily-drift-check' registered.")


if __name__ == "__main__":
    # Local test run
    Base.metadata.create_all(bind=get_engine())
    result = self_healing_pipeline(window_id="local_test", force_retrain=False)
    print(result)
