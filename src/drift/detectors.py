"""
Drift detectors.

Three detectors, each returning a DriftResult:

1. KSDriftDetector   — Kolmogorov–Smirnov test per numerical feature
2. PSIDriftDetector  — Population Stability Index per numerical feature
3. LearnedDriftDetector — Domain classifier (logistic regression)
   trains to distinguish reference vs. current samples;
   AUC > threshold ⇒ distributions are statistically distinguishable

All detectors accept reference and current DataFrames and return a
DriftResult with a scalar `score`, a `is_drift` boolean, and `details`
dict for per-feature breakdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from src.ingestion.schema import NUMERICAL_FEATURES, CATEGORICAL_FEATURES


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class DriftResult:
    method: str
    score: float           # composite scalar ∈ [0, 1]
    threshold: float
    is_drift: bool
    details: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        import json

        def _coerce(obj):
            """Recursively convert numpy scalars to Python natives."""
            if isinstance(obj, dict):
                return {k: _coerce(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_coerce(v) for v in obj]
            if hasattr(obj, "item"):        # numpy scalar
                return obj.item()
            if isinstance(obj, bool):
                return bool(obj)
            return obj

        return {
            "method": self.method,
            "score": round(float(self.score), 6),
            "threshold": float(self.threshold),
            "is_drift": bool(self.is_drift),
            "details": _coerce(self.details),
        }


# ── KS Detector ───────────────────────────────────────────────────────────────

class KSDriftDetector:
    """
    Two-sample Kolmogorov–Smirnov test on each numerical feature.

    Score = fraction of features whose p-value < alpha.
    is_drift = score >= threshold (default: any feature drifted → flag).
    """

    def __init__(self, alpha: float = 0.05, threshold: float = 0.1):
        self.alpha = alpha
        self.threshold = threshold

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> DriftResult:
        cols = features or NUMERICAL_FEATURES
        per_feature: Dict[str, dict] = {}
        n_drifted = 0

        for col in cols:
            if col not in reference.columns or col not in current.columns:
                continue
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values

            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            stat, pval = stats.ks_2samp(ref_vals, cur_vals)
            drifted = pval < self.alpha
            if drifted:
                n_drifted += 1
            per_feature[col] = {
                "ks_stat": round(float(stat), 6),
                "p_value": round(float(pval), 6),
                "drifted": drifted,
            }

        score = n_drifted / max(len(per_feature), 1)
        return DriftResult(
            method="ks",
            score=score,
            threshold=self.threshold,
            is_drift=score >= self.threshold,
            details={"per_feature": per_feature, "n_drifted": n_drifted,
                     "n_tested": len(per_feature)},
        )


# ── PSI Detector ───────────────────────────────────────────────────────────────

def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1  : no significant change
    PSI < 0.2  : moderate change (warning)
    PSI >= 0.2 : significant shift (action required — industry standard)
    """
    breakpoints = np.nanpercentile(reference, np.linspace(0, 100, bins + 1))
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Avoid zero-division / log(0)
    ref_pct = np.where(ref_pct == 0, 1e-6, ref_pct)
    cur_pct = np.where(cur_pct == 0, 1e-6, cur_pct)

    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


class PSIDriftDetector:
    """
    Population Stability Index per numerical feature.
    Score = mean PSI across features.
    """

    def __init__(self, threshold: float = 0.2, bins: int = 10):
        self.threshold = threshold
        self.bins = bins

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> DriftResult:
        cols = features or NUMERICAL_FEATURES
        per_feature: Dict[str, dict] = {}
        psi_values: List[float] = []

        for col in cols:
            if col not in reference.columns or col not in current.columns:
                continue
            ref_vals = reference[col].dropna().values
            cur_vals = current[col].dropna().values
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            psi_val = _psi(ref_vals, cur_vals, self.bins)
            psi_values.append(psi_val)
            per_feature[col] = {
                "psi": round(psi_val, 6),
                "drifted": psi_val >= self.threshold,
            }

        score = float(np.mean(psi_values)) if psi_values else 0.0
        return DriftResult(
            method="psi",
            score=score,
            threshold=self.threshold,
            is_drift=score >= self.threshold,
            details={"per_feature": per_feature,
                     "n_drifted": sum(1 for v in psi_values if v >= self.threshold),
                     "n_tested": len(psi_values)},
        )


# ── Learned Detector ───────────────────────────────────────────────────────────

class LearnedDriftDetector:
    """
    Domain Classifier Drift Detector.

    Trains a logistic regression to label reference samples as 0
    and current samples as 1.  If the classifier AUC > threshold,
    the two distributions are statistically distinguishable → drift.

    Reference: Rabanser et al. (2019) "Failing Loudly" (NeurIPS).

    Score = cross-validated AUC, normalised to [0, 1].
    is_drift = AUC >= threshold (default 0.65).
    """

    def __init__(self, auc_threshold: float = 0.65, max_samples: int = 5_000):
        self.auc_threshold = auc_threshold
        self.max_samples = max_samples

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
    ) -> DriftResult:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.preprocessing import StandardScaler, OrdinalEncoder
        from sklearn.pipeline import Pipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer

        # Sample to cap compute
        ref_s = reference.sample(
            n=min(self.max_samples, len(reference)), random_state=42
        )
        cur_s = current.sample(
            n=min(self.max_samples, len(current)), random_state=42
        )

        # Build feature matrix
        all_cols = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        ref_s = ref_s[[c for c in all_cols if c in ref_s.columns]]
        cur_s = cur_s[[c for c in all_cols if c in cur_s.columns]]

        X = pd.concat([ref_s, cur_s], ignore_index=True)
        y = np.concatenate([np.zeros(len(ref_s)), np.ones(len(cur_s))])

        num_cols = [c for c in NUMERICAL_FEATURES if c in X.columns]
        cat_cols = [c for c in CATEGORICAL_FEATURES if c in X.columns]

        pre = ColumnTransformer([
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("enc", OrdinalEncoder(handle_unknown="use_encoded_value",
                                       unknown_value=-1)),
            ]), cat_cols),
        ])

        clf = Pipeline([
            ("pre", pre),
            ("lr", LogisticRegression(
                max_iter=500, C=1.0, class_weight="balanced",
                solver="lbfgs", random_state=42, n_jobs=1,
            )),
        ])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
        mean_auc = float(np.mean(auc_scores))

        # Fit once more to get feature importances
        clf.fit(X, y)
        coefs = clf.named_steps["lr"].coef_[0]
        feat_names = num_cols + cat_cols
        top_features = sorted(
            zip(feat_names, np.abs(coefs).tolist()),
            key=lambda x: x[1], reverse=True,
        )[:10]

        return DriftResult(
            method="learned",
            score=mean_auc,
            threshold=self.auc_threshold,
            is_drift=mean_auc >= self.auc_threshold,
            details={
                "mean_auc": round(mean_auc, 6),
                "auc_per_fold": [round(v, 4) for v in auc_scores.tolist()],
                "top_drifted_features": [
                    {"feature": f, "importance": round(imp, 4)}
                    for f, imp in top_features
                ],
                "n_reference": len(ref_s),
                "n_current": len(cur_s),
            },
        )
