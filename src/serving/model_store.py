"""
In-process model store with thread-safe hot-reload.

Holds the current champion pipeline in memory. When a new champion is
promoted the API calls reload() — no process restart needed, no downtime.

Safety guarantee: the store always returns a valid pipeline or raises a
clear RuntimeError. The API layer converts that to a 503.
"""
from __future__ import annotations

import threading
from typing import Optional

import pandas as pd
from sklearn.pipeline import Pipeline

from src.ingestion.schema import ALL_FEATURES


class ModelStore:
    def __init__(self):
        self._lock = threading.RLock()
        self._pipeline: Optional[Pipeline] = None
        self._model_version: Optional[str] = None
        self._model_db_id: Optional[int] = None

    # ── Loading ────────────────────────────────────────────────────────────────

    def load_champion(self) -> bool:
        """
        Load the current DB-champion model.  Thread-safe.
        Returns True if a champion was found and loaded, False otherwise.
        """
        from src.retraining.trainer import load_champion
        from src.db.session import SessionLocal
        from src.db.models import ModelVersion

        db = SessionLocal()
        try:
            champ = (
                db.query(ModelVersion)
                .filter(ModelVersion.stage == "champion")
                .order_by(ModelVersion.created_at.desc())
                .first()
            )
            if champ is None:
                return False
            pipeline = load_champion()
            if pipeline is None:
                return False
            with self._lock:
                self._pipeline = pipeline
                self._model_version = champ.version
                self._model_db_id = champ.id
            return True
        finally:
            db.close()

    def reload(self) -> bool:
        """Hot-swap to the latest champion. Called after promotion."""
        return self.load_champion()

    def set_pipeline(
        self,
        pipeline: Pipeline,
        version: str,
        db_id: Optional[int] = None,
    ):
        """Directly inject a pipeline (used in tests and baseline seeding)."""
        with self._lock:
            self._pipeline = pipeline
            self._model_version = version
            self._model_db_id = db_id

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> dict:
        with self._lock:
            pipeline = self._pipeline
            version = self._model_version

        if pipeline is None:
            raise RuntimeError(
                "No champion model loaded. Run the baseline script first."
            )

        X_feat = X[ALL_FEATURES]
        predictions = pipeline.predict(X_feat).tolist()
        probabilities = pipeline.predict_proba(X_feat)[:, 1].tolist()

        return {
            "predictions": predictions,
            "probabilities": probabilities,
            "model_version": version,
        }

    # ── State ──────────────────────────────────────────────────────────────────

    @property
    def is_ready(self) -> bool:
        return self._pipeline is not None

    @property
    def model_version(self) -> Optional[str]:
        return self._model_version

    @property
    def model_db_id(self) -> Optional[int]:
        return self._model_db_id


# Singleton shared across the FastAPI process
_store: Optional[ModelStore] = None


def get_store() -> ModelStore:
    global _store
    if _store is None:
        _store = ModelStore()
    return _store
