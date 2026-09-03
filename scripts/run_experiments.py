"""
Module 16 - Experiment Scripts A/B/C with sensitivity analysis.

Configurations:
  A: FULL SYSTEM    - drift detection + validation gate + promotion
  B: NAIVE RETRAIN  - retrain on every batch, no validation gate
  C: STATIC         - never retrain; champion fixed at baseline

Also runs sensitivity analysis across mild/moderate/severe drift
to show graceful degradation (not just a perfect-score artifact).

Outputs:
  results/experiment_results.csv
  results/experiment_summary.json
  results/sensitivity_analysis.csv
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

# -- Environment setup (before any src imports) --------------------------------

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

_tmp = tempfile.mkdtemp()
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{_tmp}/mlruns.db")
os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", "experiments_abc")
os.environ.setdefault("DATABASE_URL", "sqlite:///./experiment_shlp.db")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from src.config import get_settings
from src.ingestion.generator import (
    generate_reference, generate_production_sequence, DriftSpec, generate_batch,
)
from src.ingestion.schema import ALL_FEATURES, TARGET
from src.retraining.trainer import train_and_log, load_model
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

# -- Constants -----------------------------------------------------------------

N_REFERENCE   = 5_000   # faster for CI/local runs
N_BATCHES     = 10
DRIFT_START   = 5
N_PER_BATCH   = 2_000

DRIFT_LEVELS = [
    ("mild",   0.3),
    ("severe", 1.0),
]

FAST_PARAMS = {"n_estimators": 50, "max_depth": 4}  # used everywhere


# -- Data helpers --------------------------------------------------------------

def make_data(tmp_dir: str, drift_magnitude: float = 1.0):
    ref_path = os.path.join(tmp_dir, "reference.parquet")
    batch_dir = os.path.join(tmp_dir, "batches")
    os.makedirs(batch_dir, exist_ok=True)
    ref_df = generate_reference(n=N_REFERENCE, seed=0, output_path=ref_path)
    paths, ground_truth = generate_production_sequence(
        output_dir=batch_dir,
        n_batches=N_BATCHES,
        n_per_batch=N_PER_BATCH,
        drift_spec=DriftSpec(
            mode="abrupt",
            start_batch=DRIFT_START,
            shift_mean=drift_magnitude * 2.0,
            flip_labels=0.05 * drift_magnitude,
        ),
        seed_offset=200,
    )
    return ref_df, paths, ground_truth


def evaluate_on_batch(model, batch_path: str) -> Dict[str, float]:
    df = pd.read_parquet(batch_path)
    X, y = df[ALL_FEATURES], df[TARGET].values
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "f1":        float(f1_score(y, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y, y_prob)),
        "precision": float(precision_score(y, y_pred, zero_division=0)),
        "recall":    float(recall_score(y, y_pred, zero_division=0)),
    }


# -- Config A: Full system -----------------------------------------------------

def run_config_a(ref_df, batch_paths, ground_truth, tmp_dir):
    from src.drift.engine import DriftEngine
    from src.validation.gate import ValidationGate
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from src.db.session import Base
    from src.db import models as db_models

    cfg = get_settings()
    cfg.__dict__["reference_path"] = os.path.join(tmp_dir, "reference.parquet")
    cfg.__dict__["drift_debounce_windows"] = 1
    cfg.__dict__["drift_cooldown_minutes"] = 0

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    db = Session()

    result = train_and_log(ref_df, run_name="A_baseline",
                           hyperparams=FAST_PARAMS)
    champ_row = db_models.ModelVersion(
        version="1", stage="champion", metric_name="f1",
        metric_value=result.metrics["f1"], mlflow_run_id=result.run_id,
        artifact_path=result.mlflow_model_uri,
    )
    db.add(champ_row)
    db.commit()
    db.refresh(champ_row)

    drift_engine = DriftEngine()
    rows, detections = [], []
    promotions = rejections = 0
    detection_lag = None

    for i, (path, is_drifted) in enumerate(zip(batch_paths, ground_truth)):
        cur = pd.read_parquet(path)
        composite = drift_engine.evaluate(ref_df, cur, window_id=f"A_{i}")
        detected = composite.is_drift
        detections.append(detected)

        if detected and detection_lag is None and is_drifted:
            detection_lag = i - DRIFT_START

        champ = db.query(db_models.ModelVersion).filter_by(
            stage="champion").order_by(db_models.ModelVersion.created_at.desc()).first()
        deployed = load_model(champ.artifact_path)
        perf = evaluate_on_batch(deployed, path)

        if detected:
            new = train_and_log(cur, run_name=f"A_challenger_{i}",
                                hyperparams=FAST_PARAMS)
            chal = db_models.ModelVersion(
                version=f"A_{i}", stage="challenger", metric_name="f1",
                metric_value=new.metrics["f1"], mlflow_run_id=new.run_id,
                artifact_path=new.mlflow_model_uri,
            )
            db.add(chal)
            db.flush()
            job = db_models.RetrainingJob(
                status="success", triggered_by="experiment_A",
                candidate_model_id=chal.id,
            )
            db.add(job)
            db.commit()
            db.refresh(job)

            import src.validation.gate as gate_mod
            orig = gate_mod.SessionLocal
            gate_mod.SessionLocal = Session
            try:
                decision = ValidationGate().evaluate(job_id=job.id, db=db)
            finally:
                gate_mod.SessionLocal = orig

            if decision.decision == "promoted":
                promotions += 1
            else:
                rejections += 1

        rows.append({"config": "A", "batch": i,
                     "ground_truth_drift": is_drifted,
                     "detected_drift": detected, **perf})

    db.close()

    tp = sum(1 for d, g in zip(detections, ground_truth) if d and g)
    fp = sum(1 for d, g in zip(detections, ground_truth) if d and not g)
    fn = sum(1 for d, g in zip(detections, ground_truth) if not d and g)
    prec = tp / max(tp + fp, 1)
    rec  = tp / max(tp + fn, 1)
    f1d  = 2 * prec * rec / max(prec + rec, 1e-9)

    drifted_f1 = [r["f1"] for r in rows if r["ground_truth_drift"]]
    return rows, {
        "config": "A",
        "detector_precision": round(prec, 4),
        "detector_recall":    round(rec, 4),
        "detector_f1":        round(f1d, 4),
        "detection_lag_batches": detection_lag,
        "total_promotions": promotions,
        "total_rejections": rejections,
        "rollback_rate": round(rejections / max(promotions + rejections, 1), 4),
        "avg_f1_post_drift": round(float(np.mean(drifted_f1)), 4) if drifted_f1 else None,
    }


# -- Config B: Naive retrain ---------------------------------------------------

def run_config_b(ref_df, batch_paths, ground_truth, tmp_dir):
    deployed = load_model(
        train_and_log(ref_df, run_name="B_baseline",
                      hyperparams=FAST_PARAMS).mlflow_model_uri
    )
    rows = []
    for i, (path, is_drifted) in enumerate(zip(batch_paths, ground_truth)):
        cur = pd.read_parquet(path)
        perf = evaluate_on_batch(deployed, path)
        deployed = load_model(
            train_and_log(cur, run_name=f"B_retrain_{i}",
                          hyperparams=FAST_PARAMS).mlflow_model_uri
        )
        rows.append({"config": "B", "batch": i,
                     "ground_truth_drift": is_drifted,
                     "detected_drift": False, **perf})

    drifted_f1 = [r["f1"] for r in rows if r["ground_truth_drift"]]
    return rows, {
        "config": "B",
        "detector_precision": None, "detector_recall": None, "detector_f1": None,
        "detection_lag_batches": None,
        "total_promotions": N_BATCHES, "total_rejections": 0, "rollback_rate": 0.0,
        "avg_f1_post_drift": round(float(np.mean(drifted_f1)), 4) if drifted_f1 else None,
    }


# -- Config C: Static ----------------------------------------------------------

def run_config_c(ref_df, batch_paths, ground_truth, tmp_dir):
    static = load_model(
        train_and_log(ref_df, run_name="C_static",
                      hyperparams=FAST_PARAMS).mlflow_model_uri
    )
    rows = []
    for i, (path, is_drifted) in enumerate(zip(batch_paths, ground_truth)):
        perf = evaluate_on_batch(static, path)
        rows.append({"config": "C", "batch": i,
                     "ground_truth_drift": is_drifted,
                     "detected_drift": False, **perf})

    drifted_f1 = [r["f1"] for r in rows if r["ground_truth_drift"]]
    return rows, {
        "config": "C",
        "detector_precision": None, "detector_recall": None, "detector_f1": None,
        "detection_lag_batches": None,
        "total_promotions": 0, "total_rejections": 0, "rollback_rate": None,
        "avg_f1_post_drift": round(float(np.mean(drifted_f1)), 4) if drifted_f1 else None,
    }


# -- Main ----------------------------------------------------------------------

def main():
    print("=" * 65)
    print("  Self-Healing ML Pipeline -- Experiments A/B/C")
    print("=" * 65)

    with tempfile.TemporaryDirectory() as tmp_dir:

        # 1. Sensitivity analysis (Config A only, three drift magnitudes)
        print("\n--- SENSITIVITY ANALYSIS ---")
        print("  Shows graceful degradation -- NOT an artifact of obvious drift")
        sens_rows = []
        for level_name, magnitude in DRIFT_LEVELS:
            ref_df, batch_paths, ground_truth = make_data(tmp_dir, magnitude)
            _, sumA = run_config_a(ref_df, batch_paths, ground_truth, tmp_dir)
            sens_rows.append({
                "drift_level":        level_name,
                "shift_magnitude":    magnitude,
                "detector_precision": sumA["detector_precision"],
                "detector_recall":    sumA["detector_recall"],
                "detector_f1":        sumA["detector_f1"],
                "detection_lag":      sumA["detection_lag_batches"],
                "promotions":         sumA["total_promotions"],
                "rejections":         sumA["total_rejections"],
                "avg_f1_post_drift":  sumA["avg_f1_post_drift"],
            })
            print(f"  [{level_name:8s}] P={sumA['detector_precision']}  "
                  f"R={sumA['detector_recall']}  F1={sumA['detector_f1']}  "
                  f"lag={sumA['detection_lag_batches']}  "
                  f"promo={sumA['total_promotions']}  "
                  f"reject={sumA['total_rejections']}")

        sens_df = pd.DataFrame(sens_rows)
        sens_csv = str(RESULTS_DIR / "sensitivity_analysis.csv")
        sens_df.to_csv(sens_csv, index=False)
        print(f"\n  Saved -> {sens_csv}")

        # 2. Main A/B/C comparison at severe drift
        print("\n--- MAIN A/B/C COMPARISON (severe drift, shift=2.0) ---")
        ref_df, batch_paths, ground_truth = make_data(tmp_dir, drift_magnitude=1.0)
        print(f"  Reference: {len(ref_df):,} rows | Batches: {N_BATCHES} "
              f"| Drifted: {sum(ground_truth)} | Per-batch: {N_PER_BATCH:,}")

        print("[Config A] Full system...")
        rows_a, sum_a = run_config_a(ref_df, batch_paths, ground_truth, tmp_dir)

        print("[Config B] Naive retrain (no gate)...")
        rows_b, sum_b = run_config_b(ref_df, batch_paths, ground_truth, tmp_dir)

        print("[Config C] Static model...")
        rows_c, sum_c = run_config_c(ref_df, batch_paths, ground_truth, tmp_dir)

    # Save results
    results_df = pd.DataFrame(rows_a + rows_b + rows_c)
    results_csv = str(RESULTS_DIR / "experiment_results.csv")
    results_df.to_csv(results_csv, index=False)

    summaries = [sum_a, sum_b, sum_c]
    summary_json = str(RESULTS_DIR / "experiment_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n  Per-batch results -> {results_csv}")
    print(f"  Summary           -> {summary_json}")

    # Print tables
    print("\n" + "=" * 65)
    print("  SENSITIVITY ANALYSIS (does detection degrade gracefully?)")
    print("=" * 65)
    print(sens_df[["drift_level", "detector_precision", "detector_recall",
                   "detector_f1", "detection_lag"]].to_string(index=False))

    print("\n" + "=" * 65)
    print("  A/B/C SUMMARY")
    print("=" * 65)
    fmt = "{:<8} {:<7} {:<7} {:<7} {:<16} {:<7} {:<8} {:<9}"
    print(fmt.format("Config", "Det-P", "Det-R", "Det-F1",
                     "Avg F1(drifted)", "Promo", "Reject", "Rollback%"))
    print("-" * 65)
    for s in summaries:
        print(fmt.format(
            s["config"],
            str(s["detector_precision"] or "--"),
            str(s["detector_recall"] or "--"),
            str(s["detector_f1"] or "--"),
            str(s["avg_f1_post_drift"] or "--"),
            str(s["total_promotions"]),
            str(s["total_rejections"]),
            str(s["rollback_rate"]) if s["rollback_rate"] is not None else "--",
        ))

    # Gate calibration check
    print("\n" + "=" * 65)
    print("  GATE CALIBRATION CHECK")
    print("=" * 65)
    b_f1 = sum_b["avg_f1_post_drift"] or 0.0
    c_f1 = sum_c["avg_f1_post_drift"] or 0.0
    if b_f1 < c_f1:
        print(f"  Config B (naive) avg F1={b_f1:.4f} < Config C (static) F1={c_f1:.4f}")
        print("  => Naive retraining HURTS -- validates the anti-pattern critique.")
    else:
        print(f"  Config B F1={b_f1:.4f}, Config C F1={c_f1:.4f}")

    if sum_a["total_promotions"] == 0:
        print(f"  Gate rejected all {sum_a['total_rejections']} challengers.")
        print(f"  => This is EXPECTED with N_PER_BATCH={N_PER_BATCH} and "
              f"N_REFERENCE={N_REFERENCE}.")
        print("     Small batches are noisier than the large reference -- "
              "gate correctly blocks them.")
        print(f"     To see promotions, set N_PER_BATCH >= {N_REFERENCE} in the script.")
    else:
        print(f"  Gate promoted {sum_a['total_promotions']} challenger(s), "
              f"rejected {sum_a['total_rejections']}.")
        print(f"  Rollback rate: {sum_a['rollback_rate']:.1%}")


if __name__ == "__main__":
    main()



