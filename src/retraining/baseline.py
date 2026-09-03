"""
Baseline training script.

Run once to establish the champion model:
    python -m src.retraining.baseline

Steps:
1. Generate (or load) the reference dataset
2. Train and log to MLflow
3. Register the model in MLflow Model Registry
4. Persist a ModelVersion row to the database (stage=champion)
5. Seed default users if the DB is empty
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import datetime

from src.config import get_settings
from src.db.session import SessionLocal, get_engine, Base
from src.db import models as db_models
from src.db.audit import write as audit_write, Actions
from src.db.seed import seed as seed_users
from src.ingestion.generator import generate_reference
from src.ingestion.loader import DataLoader
from src.retraining.trainer import train_and_log


def run_baseline(n_reference: int = 20_000) -> db_models.ModelVersion:
    cfg = get_settings()

    # Ensure tables exist (idempotent)
    Base.metadata.create_all(bind=get_engine())
    seed_users()

    print(f"[baseline] MLflow URI : {cfg.mlflow_tracking_uri}")
    print(f"[baseline] Experiment : {cfg.mlflow_experiment_name}")
    print(f"[baseline] Model name : {cfg.mlflow_model_name}")

    # 1. Reference data
    import pathlib
    if not pathlib.Path(cfg.reference_path).exists():
        print(f"[baseline] Generating {n_reference:,} reference rows...")
        generate_reference(n=n_reference, seed=0, output_path=cfg.reference_path)
    else:
        print(f"[baseline] Reference already exists: {cfg.reference_path}")

    loader = DataLoader()
    ref_df = loader.load_reference()
    print(f"[baseline] Reference loaded: {len(ref_df):,} rows, "
          f"fraud rate={ref_df['isFraud'].mean():.3%}")

    # 2. Train + log
    print("[baseline] Training baseline model...")
    result = train_and_log(
        df=ref_df,
        run_name="baseline_champion",
        trigger="manual",
        data_window_desc=f"reference_{n_reference}",
        register_as=cfg.mlflow_model_name,
    )

    print(f"[baseline] run_id        : {result.run_id}")
    print(f"[baseline] f1            : {result.metrics['f1']:.4f}")
    print(f"[baseline] roc_auc       : {result.metrics['roc_auc']:.4f}")
    print(f"[baseline] MLflow version: {result.registered_version}")

    # 3. Persist to DB
    db = SessionLocal()
    try:
        # Archive any existing champion
        existing = (
            db.query(db_models.ModelVersion)
            .filter(db_models.ModelVersion.stage == "champion")
            .all()
        )
        for mv in existing:
            mv.stage = "archived"
        db.flush()

        # New champion
        champion = db_models.ModelVersion(
            version=str(result.registered_version or "1"),
            stage="champion",
            metric_name=cfg.validation_metric,
            metric_value=result.metrics.get(cfg.validation_metric, 0.0),
            mlflow_run_id=result.run_id,
            artifact_path=result.mlflow_model_uri,
        )
        db.add(champion)
        db.commit()
        db.refresh(champion)

        audit_write(
            db,
            actor="system",
            action=Actions.MODEL_PROMOTED,
            entity_type="model",
            entity_id=str(champion.id),
            details={
                "trigger": "baseline",
                "metrics": result.metrics,
                "run_id": result.run_id,
            },
        )

        print(f"[baseline] Champion persisted: DB id={champion.id}")
        return champion
    finally:
        db.close()


if __name__ == "__main__":
    run_baseline()
