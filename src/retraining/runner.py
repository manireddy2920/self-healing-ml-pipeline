"""
Retraining Job Runner — Module 8.

Executes a RetrainingJob end-to-end:
  1. Load the most recent labeled data window (falls back to reference)
  2. Train a challenger model via train_and_log
  3. Persist a ModelVersion row (stage=challenger)
  4. Update the job status and hand off to the ValidationGate (Module 9)

Called by:
  - The Prefect DAG (Module 11)
  - The /retrain/trigger API endpoint (manual override)
"""
from __future__ import annotations

import datetime

from src.config import get_settings
from src.db.audit import write as audit_write, Actions
from src.db.models import RetrainingJob, ModelVersion
from src.db.session import SessionLocal
from src.ingestion.loader import DataLoader
from src.retraining.trainer import train_and_log


def run_retraining_job(job_id: int) -> RetrainingJob:
    """
    Execute the retraining job identified by *job_id*.

    Updates the DB row in-place (running → success/failed) and returns it.
    Also runs the validation gate immediately after training.

    Safety guarantee: if training or validation raises, the job is marked
    'failed' and the champion model is left untouched.
    """
    cfg = get_settings()
    db = SessionLocal()
    try:
        job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()
        if job is None:
            raise ValueError(f"RetrainingJob {job_id} not found")

        job.status = "running"
        job.started_at = datetime.datetime.now(datetime.timezone.utc)
        db.commit()

        # ── 1. Load data ───────────────────────────────────────────────────────
        loader = DataLoader()
        df = loader.get_recent_window()
        if df.empty or len(df) < 200:
            df = loader.load_reference()

        window_desc = f"window_{cfg.training_window_days}d_{len(df)}rows"

        # ── 2. Train + log ─────────────────────────────────────────────────────
        result = train_and_log(
            df=df,
            run_name=f"challenger_job_{job_id}",
            trigger="drift_event" if job.drift_event_id else "manual",
            drift_event_id=job.drift_event_id,
            data_window_desc=window_desc,
        )

        # ── 3. Persist challenger ──────────────────────────────────────────────
        challenger = ModelVersion(
            version=f"challenger_{job_id}",
            stage="challenger",
            metric_name=cfg.validation_metric,
            metric_value=result.metrics.get(cfg.validation_metric, 0.0),
            mlflow_run_id=result.run_id,
            artifact_path=result.mlflow_model_uri,
        )
        db.add(challenger)
        db.flush()

        job.status = "success"
        job.finished_at = datetime.datetime.now(datetime.timezone.utc)
        job.candidate_model_id = challenger.id
        db.commit()
        db.refresh(job)

        audit_write(
            db, actor="system", action=Actions.RETRAIN_COMPLETED,
            entity_type="retraining_job", entity_id=str(job_id),
            details={"metrics": result.metrics, "run_id": result.run_id,
                     "challenger_id": challenger.id},
        )

        # ── 4. Run validation gate ─────────────────────────────────────────────
        from src.validation.gate import ValidationGate
        gate = ValidationGate()
        gate.evaluate(job_id=job_id, db=db)

        return job

    except Exception as exc:
        db.rollback()
        try:
            job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()
            if job:
                job.status = "failed"
                job.finished_at = datetime.datetime.now(datetime.timezone.utc)
                db.commit()
                audit_write(
                    db, actor="system", action=Actions.RETRAIN_FAILED,
                    entity_type="retraining_job", entity_id=str(job_id),
                    details={"error": str(exc)},
                )
        except Exception:
            pass
        raise
    finally:
        db.close()
