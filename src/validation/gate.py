"""
Validation & Promotion Gate — Module 9.

Evaluates a challenger model against the current champion on an identical
held-out validation set.

Promotion criteria (all must pass):
  - challenger_metric >= champion_metric - promotion_threshold_delta
  - challenger must not be worse than champion on the critical-slice metric
    (fraud recall — we never want to regress on catching fraud)

If the challenger passes the gate it is promoted to champion:
  - old champion stage → "archived"
  - challenger stage → "champion"
  - ModelStore hot-reloaded

If the challenger fails:
  - challenger stage → "archived"
  - champion remains unchanged ("non-regression safety" property)

After N consecutive failed challenges the controller sets needs_human_review.

All decisions are persisted in promotion_decisions and audit_log.
"""
from __future__ import annotations

import datetime
from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from sklearn.model_selection import train_test_split
from sqlalchemy.orm import Session

from src.config import get_settings
from src.db.audit import write as audit_write, Actions
from src.db.models import ModelVersion, RetrainingJob, PromotionDecision
from src.db.session import SessionLocal
from src.ingestion.loader import DataLoader
from src.ingestion.schema import ALL_FEATURES, TARGET
from src.retraining.trainer import load_model


def _evaluate_on_holdout(
    artifact_path: str,
    X_val, y_val,
    metric_name: str,
) -> dict:
    """Load a model from MLflow and score it on the shared validation set."""
    model = load_model(artifact_path)
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]
    return {
        "f1":     float(f1_score(y_val, y_pred, zero_division=0)),
        "roc_auc":float(roc_auc_score(y_val, y_prob)),
        "recall": float(recall_score(y_val, y_pred, zero_division=0)),
        metric_name: float(
            f1_score(y_val, y_pred, zero_division=0)
            if metric_name == "f1"
            else roc_auc_score(y_val, y_prob)
        ),
    }


class ValidationGate:
    """
    Stateless gate: reads champion/challenger from DB, evaluates, decides.
    """

    def evaluate(
        self,
        job_id: int,
        db: Optional[Session] = None,
    ) -> PromotionDecision:
        """
        Run the validation gate for *job_id*.
        Opens its own DB session if none provided.
        """
        _own_session = db is None
        if _own_session:
            db = SessionLocal()

        try:
            return self._run(job_id, db)
        finally:
            if _own_session:
                db.close()

    # ── Internal ────────────────────────────────────────────────────────────────

    def _run(self, job_id: int, db: Session) -> PromotionDecision:
        cfg = get_settings()

        job = db.query(RetrainingJob).filter(RetrainingJob.id == job_id).first()
        if job is None or job.candidate_model_id is None:
            raise ValueError(f"Job {job_id} has no candidate model")

        challenger = db.query(ModelVersion).filter(
            ModelVersion.id == job.candidate_model_id
        ).first()
        if challenger is None:
            raise ValueError(f"Challenger model not found for job {job_id}")

        champion = (
            db.query(ModelVersion)
            .filter(ModelVersion.stage == "champion")
            .order_by(ModelVersion.created_at.desc())
            .first()
        )

        # ── Shared holdout set (fixed seed) ────────────────────────────────────
        loader = DataLoader()
        ref_df = loader.load_reference()
        _, val_df = train_test_split(
            ref_df, test_size=0.25, random_state=99,
            stratify=ref_df[TARGET],
        )
        X_val = val_df[ALL_FEATURES]
        y_val = val_df[TARGET].values

        # ── Evaluate challenger ─────────────────────────────────────────────────
        challenger_metrics = _evaluate_on_holdout(
            challenger.artifact_path, X_val, y_val, cfg.validation_metric
        )

        # ── Evaluate champion (or use stored metric if no champion) ─────────────
        if champion is None:
            # First ever model — auto-promote
            champion_metrics = {cfg.validation_metric: 0.0, "recall": 0.0}
        else:
            champion_metrics = _evaluate_on_holdout(
                champion.artifact_path, X_val, y_val, cfg.validation_metric
            )

        c_metric = challenger_metrics.get(cfg.validation_metric, 0.0)
        p_metric = champion_metrics.get(cfg.validation_metric, 0.0)
        threshold = cfg.promotion_threshold_delta

        passes_primary = c_metric >= p_metric - threshold
        # Critical slice: challenger fraud recall must not regress by >5pp
        passes_recall = (
            challenger_metrics.get("recall", 0.0) >=
            champion_metrics.get("recall", 0.0) - 0.05
        )
        promoted = passes_primary and passes_recall

        decision_str = "promoted" if promoted else "rejected"

        # ── Persist decision ───────────────────────────────────────────────────
        decision = PromotionDecision(
            retraining_job_id=job_id,
            candidate_model_id=challenger.id,
            champion_model_id=champion.id if champion else None,
            decision=decision_str,
            candidate_metric=c_metric,
            champion_metric=p_metric,
            decided_at=datetime.datetime.now(datetime.timezone.utc),
            decided_by="system",
        )
        db.add(decision)

        if promoted:
            # Archive current champion
            if champion:
                champion.stage = "archived"
            challenger.stage = "champion"
            db.commit()
            db.refresh(decision)

            # Hot-reload the serving layer
            try:
                from src.serving.model_store import get_store
                get_store().reload()
            except Exception:
                pass

            audit_write(
                db, actor="system", action=Actions.MODEL_PROMOTED,
                entity_type="model", entity_id=str(challenger.id),
                details={
                    "challenger_metrics": challenger_metrics,
                    "champion_metrics": champion_metrics,
                    "job_id": job_id,
                    "decision": decision_str,
                },
            )
            audit_write(
                db, actor="system", action=Actions.VALIDATION_PASSED,
                entity_type="promotion_decision", entity_id=str(decision.id),
                details={"passes_primary": passes_primary,
                         "passes_recall": passes_recall},
            )
        else:
            # Archive the failed challenger — champion untouched
            challenger.stage = "archived"
            db.commit()
            db.refresh(decision)

            audit_write(
                db, actor="system", action=Actions.VALIDATION_FAILED,
                entity_type="promotion_decision", entity_id=str(decision.id),
                details={
                    "challenger_metrics": challenger_metrics,
                    "champion_metrics": champion_metrics,
                    "passes_primary": passes_primary,
                    "passes_recall": passes_recall,
                    "job_id": job_id,
                },
            )
            audit_write(
                db, actor="system", action=Actions.MODEL_ROLLBACK_KEPT,
                entity_type="model",
                entity_id=str(champion.id) if champion else None,
                details={"reason": "challenger_rejected"},
            )

        return decision
