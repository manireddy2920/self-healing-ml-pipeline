"""
Minimal Config D test - runs only the combined-window gate evaluation.
Shows exact gate scores to diagnose why promotion is failing.
"""
import os, sys, tempfile
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///./debug_d_mlruns.db")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "debug_d")
os.environ.setdefault("DATABASE_URL", "sqlite:///./debug_d.db")
sys.path.insert(0, ".")

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.db.session import Base
from src.db import models as db_models
from src.config import get_settings
from src.ingestion.generator import generate_reference, generate_batch
from src.ingestion.schema import ALL_FEATURES, TARGET
from src.retraining.trainer import train_and_log, load_model
from src.validation.gate import ValidationGate

cfg = get_settings()
cfg.__dict__["promotion_threshold_delta"] = 0.10
cfg.__dict__["recall_regression_tolerance"] = 0.20

# Setup DB
engine_db = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
Base.metadata.create_all(bind=engine_db)
Session = sessionmaker(bind=engine_db)

with tempfile.TemporaryDirectory() as tmp_dir:
    ref_path = f"{tmp_dir}/ref.csv"
    cfg.__dict__["reference_path"] = ref_path

    # Generate data
    ref_df = generate_reference(n=5_000, seed=0)
    ref_df.to_csv(ref_path, index=False)
    drifted = generate_batch(n=2_000, seed=205, drift_alpha=2.0, label_drift_delta=0.05)

    print(f"Reference: {len(ref_df):,} rows, fraud={ref_df[TARGET].mean():.3%}")
    print(f"Drifted:   {len(drifted):,} rows, fraud={drifted[TARGET].mean():.3%}")

    # Train champion on reference
    db = Session()
    champ_result = train_and_log(ref_df, run_name="champ",
                                  hyperparams={"n_estimators": 50, "max_depth": 4})
    champ_row = db_models.ModelVersion(
        version="1", stage="champion", metric_name="f1",
        metric_value=champ_result.metrics["f1"],
        mlflow_run_id=champ_result.run_id,
        artifact_path=champ_result.mlflow_model_uri,
    )
    db.add(champ_row)
    db.commit()
    db.refresh(champ_row)
    print(f"\nChampion trained: val_f1={champ_result.metrics['f1']:.4f}")

    # Train challenger on combined reference + drifted batch
    combined = pd.concat([ref_df, drifted], ignore_index=True).sample(
        frac=1, random_state=42
    )
    print(f"Combined training set: {len(combined):,} rows")
    chal_result = train_and_log(combined, run_name="challenger",
                                 hyperparams={"n_estimators": 50, "max_depth": 4})
    chal_row = db_models.ModelVersion(
        version="D_5", stage="challenger", metric_name="f1",
        metric_value=chal_result.metrics["f1"],
        mlflow_run_id=chal_result.run_id,
        artifact_path=chal_result.mlflow_model_uri,
    )
    db.add(chal_row)
    db.flush()
    print(f"Challenger trained: val_f1={chal_result.metrics['f1']:.4f}")

    job = db_models.RetrainingJob(
        status="success", triggered_by="debug",
        candidate_model_id=chal_row.id,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    import src.validation.gate as gate_mod
    original = gate_mod.SessionLocal
    gate_mod.SessionLocal = Session
    try:
        print("\nRunning gate evaluation...")
        decision = ValidationGate().evaluate(job_id=job.id, db=db)
        print(f"\nFINAL DECISION: {decision.decision}")
        print(f"  candidate_metric: {decision.candidate_metric:.4f}")
        print(f"  champion_metric:  {decision.champion_metric:.4f}")
    finally:
        gate_mod.SessionLocal = original

    db.close()
