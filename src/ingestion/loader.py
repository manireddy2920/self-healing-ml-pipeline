"""
Data loading and validation.

Handles loading reference data and incoming production batches from parquet,
validates schema consistency, and provides the sliding-window batch retrieval
used by the drift engine and retraining jobs.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

from src.ingestion.schema import ALL_FEATURES, NUMERICAL_FEATURES, TARGET
from src.config import get_settings


class DataLoader:
    """Caches reference data and provides batch access."""

    def __init__(self):
        self._cfg = get_settings()
        self._reference: Optional[pd.DataFrame] = None

    # ── Reference dataset ──────────────────────────────────────────────────────

    def load_reference(self, force: bool = False) -> pd.DataFrame:
        if self._reference is not None and not force:
            return self._reference
        path = self._cfg.reference_path
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Reference dataset not found at {path}. "
                "Run: python -m src.ingestion.generator"
            )
        self._reference = pd.read_parquet(path)
        return self._reference

    def invalidate_cache(self):
        self._reference = None

    # ── Production batches ─────────────────────────────────────────────────────

    def load_batch(self, path: str) -> pd.DataFrame:
        df = pd.read_parquet(path)
        self._validate(df)
        return df

    def load_from_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate(df)
        return df.copy()

    def save_batch(self, df: pd.DataFrame, name: str) -> str:
        out_dir = Path(self._cfg.data_dir) / "batches"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = str(out_dir / f"{name}.parquet")
        df.to_parquet(path, index=False)
        return path

    def get_recent_window(self, window_days: Optional[int] = None) -> pd.DataFrame:
        """
        Load and concatenate all saved production batches within the
        sliding time window.  Batches without an ingestion_ts are always included.
        """
        days = window_days or self._cfg.training_window_days
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=int(days))

        batch_dir = Path(self._cfg.data_dir) / "batches"
        if not batch_dir.exists():
            return pd.DataFrame(columns=ALL_FEATURES + [TARGET])

        frames = []
        for f in sorted(batch_dir.glob("*.parquet")):
            df = pd.read_parquet(f)
            if "ingestion_ts" in df.columns:
                ts = pd.to_datetime(df["ingestion_ts"], utc=True)
                df = df[ts >= cutoff]
            if not df.empty:
                frames.append(df)

        if not frames:
            return pd.DataFrame(columns=ALL_FEATURES + [TARGET])

        return pd.concat(frames, ignore_index=True)

    # ── Feature / label split ──────────────────────────────────────────────────

    @staticmethod
    def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        X = df[ALL_FEATURES].copy()
        y = df[TARGET].copy() if TARGET in df.columns else None
        return X, y

    # ── Validation ─────────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame):
        missing = [c for c in ALL_FEATURES if c not in df.columns]
        if missing:
            raise ValueError(f"Batch is missing columns: {missing}")
        for col in NUMERICAL_FEATURES:
            if not pd.api.types.is_numeric_dtype(df[col]):
                raise TypeError(f"Column '{col}' must be numeric, got {df[col].dtype}")
