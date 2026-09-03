"""
Synthetic dataset generator with controllable drift injection.

Supports two drift modes:
  - gradual : distribution parameters shift linearly over N batches
  - abrupt  : distribution parameters flip at a specific batch index

Generating ground-truth labels for drift makes evaluation exact —
precision/recall vs. injected drift points is measurable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.ingestion.schema import (
    NUM_PARAMS, CAT_PROBS, ALL_FEATURES,
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    TARGET, BASE_FRAUD_RATE,
)


# ── Drift specification ────────────────────────────────────────────────────────

@dataclass
class DriftSpec:
    """
    Specifies a single drift episode.

    Attributes
    ----------
    mode         : "gradual" or "abrupt"
    start_batch  : batch index where drift begins
    end_batch    : batch index where drift is fully applied (gradual only)
    features     : which features to shift (None = all numerical)
    shift_mean   : additive shift applied to the mean/lam of each feature (as fraction of std)
    flip_labels  : additionally shift fraud rate by this delta (concept drift)
    """
    mode: str = "abrupt"
    start_batch: int = 5
    end_batch: int = 5
    features: Optional[List[str]] = None
    shift_mean: float = 2.0          # in units of original std / mean
    flip_labels: float = 0.0         # increase fraud rate by this amount


# ── Low-level samplers ─────────────────────────────────────────────────────────

def _sample_numerical(
    col: str,
    n: int,
    rng: np.random.Generator,
    drift_alpha: float = 0.0,   # 0 = no drift, 1 = full drift
    shift_mean: float = 2.0,
) -> np.ndarray:
    p = NUM_PARAMS[col]
    dist = p["dist"]

    if dist == "lognormal":
        shifted_mean = p["mean"] + drift_alpha * shift_mean * p["sigma"]
        vals = rng.lognormal(shifted_mean, p["sigma"], n)
        lo, hi = p.get("clip", (None, None))
        if lo is not None:
            vals = np.clip(vals, lo, hi)
    elif dist == "normal":
        shifted_mean = p["mean"] + drift_alpha * shift_mean * p["std"]
        vals = rng.normal(shifted_mean, p["std"], n)
    elif dist == "poisson":
        lam = max(0.01, p["lam"] * (1 + drift_alpha * shift_mean * 0.5))
        vals = rng.poisson(lam, n).astype(float)
    elif dist == "randint":
        # Shift low boundary upward
        lo = int(p["low"] + drift_alpha * shift_mean * 0.1 * (p["high"] - p["low"]))
        lo = min(lo, p["high"] - 1)
        vals = rng.integers(lo, p["high"], n).astype(float)
    elif dist == "choice":
        idx = rng.integers(0, len(p["vals"]), n)
        vals = np.array(p["vals"])[idx].astype(float)
    else:
        raise ValueError(f"Unknown dist: {dist}")
    return vals


def _sample_categorical(
    col: str,
    n: int,
    rng: np.random.Generator,
    drift_alpha: float = 0.0,
) -> np.ndarray:
    cats, probs_raw = zip(*CAT_PROBS[col])
    probs = np.array(probs_raw, dtype=float)

    if drift_alpha > 0.0:
        # Shift mass toward the last category
        shift = np.zeros_like(probs)
        shift[-1] = drift_alpha * 0.3
        shift[0] = -drift_alpha * 0.3
        probs = np.clip(probs + shift, 0.01, None)

    probs = probs / probs.sum()
    return rng.choice(cats, size=n, p=probs)


def _fraud_probability(df: pd.DataFrame, drift_label_delta: float = 0.0) -> np.ndarray:
    """
    Stronger logistic-style fraud scorer so LightGBM can learn a signal.
    High TransactionAmt + high C1/C2 + anonymous email domain = higher fraud risk.
    """
    log_amt = np.log1p(df["TransactionAmt"].clip(lower=0))
    score = (
        0.4 * (log_amt - log_amt.mean()) / (log_amt.std() + 1e-6)
        + 0.3 * (df["C1"] - df["C1"].mean()) / (df["C1"].std() + 1e-6)
        + 0.2 * (df["C2"] - df["C2"].mean()) / (df["C2"].std() + 1e-6)
        + 0.5 * (df["P_emaildomain"] == "anonymous.com").astype(float)
        + 0.3 * (df["card6"] == "credit").astype(float)
        - 0.2 * (df["D1"].fillna(0) > 100).astype(float)
    )
    prob = 1 / (1 + np.exp(-score * 1.5))
    # Rescale to keep average near BASE_FRAUD_RATE
    prob = prob * (BASE_FRAUD_RATE / (prob.mean() + 1e-9))
    return np.clip(prob + drift_label_delta, 0.001, 0.99)


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_batch(
    n: int = 5_000,
    seed: int = 42,
    drift_alpha: float = 0.0,
    drift_features: Optional[List[str]] = None,
    shift_mean: float = 2.0,
    label_drift_delta: float = 0.0,
    batch_id: Optional[str] = None,
    timestamp: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """
    Generate a single batch of synthetic fraud data.

    drift_alpha : float in [0, 1] controlling drift intensity.
    """
    rng = np.random.default_rng(seed)
    target_features = drift_features or NUMERICAL_FEATURES
    data: Dict[str, np.ndarray] = {}

    for col in NUMERICAL_FEATURES:
        alpha = drift_alpha if col in target_features else 0.0
        data[col] = _sample_numerical(col, n, rng, alpha, shift_mean)

    for col in CATEGORICAL_FEATURES:
        data[col] = _sample_categorical(col, n, rng, drift_alpha)

    df = pd.DataFrame(data)
    fraud_prob = _fraud_probability(df, label_drift_delta)
    df[TARGET] = rng.binomial(1, fraud_prob).astype(int)
    df["batch_id"] = batch_id or f"batch_{seed}"
    df["ingestion_ts"] = timestamp or pd.Timestamp.now(tz="UTC")
    return df


def generate_reference(
    n: int = 20_000,
    seed: int = 0,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Generate and optionally persist the reference (training) dataset."""
    df = generate_batch(
        n=n, seed=seed, drift_alpha=0.0,
        batch_id="reference",
        timestamp=pd.Timestamp("2024-01-01", tz="UTC"),
    )
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
    return df


def generate_production_sequence(
    output_dir: str,
    n_batches: int = 12,
    n_per_batch: int = 2_000,
    drift_spec: Optional[DriftSpec] = None,
    seed_offset: int = 100,
) -> Tuple[List[str], List[bool]]:
    """
    Generate a sequence of production batches with an optional drift episode.

    Returns
    -------
    paths      : list of saved parquet paths
    is_drifted : ground-truth bool list (True = drift injected in this batch)
    """
    spec = drift_spec or DriftSpec(mode="abrupt", start_batch=6)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    paths: List[str] = []
    is_drifted: List[bool] = []

    for i in range(n_batches):
        if i < spec.start_batch:
            alpha = 0.0
            label_delta = 0.0
            drifted = False
        elif spec.mode == "abrupt":
            alpha = 1.0
            label_delta = spec.flip_labels
            drifted = True
        else:  # gradual
            progress = (i - spec.start_batch) / max(1, spec.end_batch - spec.start_batch)
            alpha = min(progress, 1.0)
            label_delta = spec.flip_labels * alpha
            drifted = alpha > 0.0

        ts = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(days=i * 7)
        df = generate_batch(
            n=n_per_batch,
            seed=seed_offset + i,
            drift_alpha=alpha,
            drift_features=spec.features,
            shift_mean=spec.shift_mean,
            label_drift_delta=label_delta,
            batch_id=f"prod_{i:03d}",
            timestamp=ts,
        )
        path = os.path.join(output_dir, f"batch_{i:03d}.parquet")
        df.to_parquet(path, index=False)
        paths.append(path)
        is_drifted.append(drifted)

    return paths, is_drifted


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    print("Generating reference dataset...")
    generate_reference(n=20_000, seed=0, output_path=f"{data_dir}/reference.parquet")
    print("Generating production batches (abrupt drift at batch 6)...")
    paths, labels = generate_production_sequence(
        output_dir=f"{data_dir}/batches",
        n_batches=12,
        n_per_batch=2_000,
        drift_spec=DriftSpec(mode="abrupt", start_batch=6),
    )
    for p, d in zip(paths, labels):
        tag = "DRIFTED" if d else "stable "
        print(f"  [{tag}] {p}")
    print("Done.")
