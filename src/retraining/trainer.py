"""
Model trainer — LightGBM + MLflow 3.x compatible.

Uses LightGBM (instead of XGBoost) because the campus machine's Application
Control policy blocks sklearn's compiled hist-gradient-boosting DLL which
XGBoost's skops serialiser triggers.  LightGBM uses a separate binary that
is not affected.

Model is serialised via cloudpickle (not skops) for MLflow 3.x compatibility.
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from src.config import get_settings
from src.ingestion.schema import ALL_FEATURES, TARGET
from src.retraining.preprocessing import build_preprocessor


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class TrainResult:
    run_id: str
    mlflow_model_uri: str
    metrics: Dict[str, float]
    params: Dict[str, Any]
    n_train: int
    n_val: int
    registered_version: Optional[str] = None


# ── Metric helpers ─────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    return {
        "accuracy":  float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc":   float(roc_auc_score(y_true, y_prob)),
    }


# ── Model builder ──────────────────────────────────────────────────────────────

def build_pipeline(
    scale_pos_weight: float = 28.0,
    n_estimators: int = 300,
    max_depth: int = 6,
    learning_rate: float = 0.05,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    random_state: int = 42,
    **extra,
) -> Tuple[Pipeline, Dict[str, Any]]:
    params = dict(
        scale_pos_weight=scale_pos_weight,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        n_jobs=1,
        verbose=-1,
    )
    clf = LGBMClassifier(**params)
    pipe = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", clf),
    ])
    return pipe, params


# ── Main training function ─────────────────────────────────────────────────────

def train_and_log(
    df: pd.DataFrame,
    run_name: str = "training_run",
    trigger: str = "manual",
    drift_event_id: Optional[int] = None,
    data_window_desc: Optional[str] = None,
    register_as: Optional[str] = None,
    hyperparams: Optional[Dict[str, Any]] = None,
) -> TrainResult:
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.mlflow_experiment_name)

    X = df[ALL_FEATURES]
    y = df[TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = max(neg / max(pos, 1), 0.1)   # clamp: LightGBM rejects spw=0

    pipe, params = build_pipeline(scale_pos_weight=spw, **(hyperparams or {}))

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.set_tags({
            "trigger": trigger,
            "drift_event_id": str(drift_event_id) if drift_event_id else "N/A",
            "data_window": data_window_desc or "N/A",
            "train_rows": str(len(X_train)),
            "val_rows": str(len(X_val)),
            "trained_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        })
        mlflow.log_params({k: v for k, v in params.items() if not callable(v)})

        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_val)
        y_prob = pipe.predict_proba(X_val)[:, 1]
        metrics = compute_metrics(y_val.values, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # cloudpickle avoids the skops / DLL issues in MLflow 3.x
        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            input_example=X_train.head(3),
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )

        model_uri = f"runs:/{run_id}/model"

    registered_version = None
    if register_as:
        client = MlflowClient(cfg.mlflow_tracking_uri)
        mv = client.create_model_version(
            name=register_as,
            source=model_uri,
            run_id=run_id,
        )
        registered_version = mv.version

    return TrainResult(
        run_id=run_id,
        mlflow_model_uri=model_uri,
        metrics=metrics,
        params=params,
        n_train=len(X_train),
        n_val=len(X_val),
        registered_version=registered_version,
    )


# ── Model loading ──────────────────────────────────────────────────────────────

def load_model(model_uri: str) -> Pipeline:
    cfg = get_settings()
    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
    return mlflow.sklearn.load_model(model_uri)


def load_champion() -> Optional[Pipeline]:
    from src.db.session import SessionLocal
    from src.db.models import ModelVersion as DBModelVersion

    db = SessionLocal()
    try:
        champ = (
            db.query(DBModelVersion)
            .filter(DBModelVersion.stage == "champion")
            .order_by(DBModelVersion.created_at.desc())
            .first()
        )
        if champ is None:
            return None
        return load_model(champ.artifact_path)
    finally:
        db.close()
