"""
Composite Drift Engine.

Runs all three detectors (KS, PSI, Learned) and combines their signals
into a single composite score using configurable weights.

Composite score = w_ks * ks_norm + w_psi * psi_norm + w_learned * learned_norm

Where each detector's output is normalised to [0, 1]:
  KS score    : fraction of drifted features (already ∈ [0,1])
  PSI score   : clipped to [0, 1] (PSI > 1 is extreme, treated as 1)
  Learned AUC : already ∈ [0, 1]

is_drift = composite_score >= composite_threshold
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from src.config import get_settings
from src.drift.detectors import (
    KSDriftDetector, PSIDriftDetector, LearnedDriftDetector, DriftResult,
)


@dataclass
class CompositeResult:
    window_id: str
    timestamp: str
    composite_score: float
    composite_threshold: float
    is_drift: bool
    ks: DriftResult
    psi: DriftResult
    learned: DriftResult
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "window_id": self.window_id,
            "timestamp": self.timestamp,
            "composite_score": round(self.composite_score, 6),
            "composite_threshold": self.composite_threshold,
            "is_drift": self.is_drift,
            "ks": self.ks.to_dict(),
            "psi": self.psi.to_dict(),
            "learned": self.learned.to_dict(),
        }


class DriftEngine:
    """
    Orchestrates all drift detectors and returns a CompositeResult.

    Weights (default):
      KS       : 0.30
      PSI      : 0.35
      Learned  : 0.35
    """

    def __init__(
        self,
        w_ks: float = 0.30,
        w_psi: float = 0.35,
        w_learned: float = 0.35,
    ):
        cfg = get_settings()
        self._w_ks = w_ks
        self._w_psi = w_psi
        self._w_learned = w_learned
        self._composite_threshold = cfg.drift_composite_threshold

        self._ks = KSDriftDetector(
            alpha=cfg.drift_ks_threshold,
            threshold=0.1,   # flag if >10% of features drift on KS
        )
        self._psi = PSIDriftDetector(
            threshold=cfg.drift_psi_threshold,
        )
        self._learned = LearnedDriftDetector(
            auc_threshold=cfg.drift_learned_auc_threshold,
        )

    def evaluate(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        window_id: str = "unknown",
    ) -> CompositeResult:
        """
        Run all detectors and return a CompositeResult.

        Requires at least 30 rows in each DataFrame for reliable statistics.
        """
        if len(reference) < 30 or len(current) < 30:
            raise ValueError(
                f"Need ≥30 rows per DataFrame, got "
                f"reference={len(reference)}, current={len(current)}"
            )

        ks_result = self._ks.detect(reference, current)
        psi_result = self._psi.detect(reference, current)
        learned_result = self._learned.detect(reference, current)

        # Normalise each score to [0, 1]
        ks_norm = min(ks_result.score, 1.0)
        psi_norm = min(psi_result.score / max(self._psi.threshold * 3, 0.01), 1.0)
        # For learned: AUC 0.5 = random = no drift, 1.0 = perfect separation
        learned_norm = max(0.0, (learned_result.score - 0.5) / 0.5)

        composite = (
            self._w_ks * ks_norm
            + self._w_psi * psi_norm
            + self._w_learned * learned_norm
        )
        composite = float(min(composite, 1.0))

        return CompositeResult(
            window_id=window_id,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            composite_score=composite,
            composite_threshold=self._composite_threshold,
            is_drift=composite >= self._composite_threshold,
            ks=ks_result,
            psi=psi_result,
            learned=learned_result,
        )
