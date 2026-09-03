"""Module 3 tests — data ingestion and drift injector."""
import os
import tempfile
import pytest
import numpy as np
import pandas as pd

os.environ.setdefault("DATABASE_URL", "sqlite:///./test_shlp.db")
os.environ.setdefault("JWT_SECRET_KEY", "test_secret")
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///./test_mlruns.db")

from src.ingestion.schema import ALL_FEATURES, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET
from src.ingestion.generator import (
    generate_batch, generate_reference, generate_production_sequence, DriftSpec
)
from src.ingestion.loader import DataLoader


# ── Schema tests ───────────────────────────────────────────────────────────────

def test_all_features_defined():
    assert len(NUMERICAL_FEATURES) > 0
    assert len(CATEGORICAL_FEATURES) > 0
    assert TARGET == "isFraud"
    assert set(ALL_FEATURES) == set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)


# ── Generator tests ────────────────────────────────────────────────────────────

def test_generate_batch_shape():
    df = generate_batch(n=500, seed=1)
    assert len(df) == 500
    assert set(ALL_FEATURES).issubset(df.columns)
    assert TARGET in df.columns


def test_generate_batch_target_is_binary():
    df = generate_batch(n=1_000, seed=2)
    assert set(df[TARGET].unique()).issubset({0, 1})


def test_generate_batch_no_nulls_numerical():
    df = generate_batch(n=500, seed=3)
    for col in NUMERICAL_FEATURES:
        assert df[col].isna().sum() == 0, f"NaNs in {col}"


def test_generate_batch_base_fraud_rate():
    df = generate_batch(n=5_000, seed=4, drift_alpha=0.0)
    rate = df[TARGET].mean()
    # Expect roughly BASE_FRAUD_RATE; allow ±3%
    assert 0.001 <= rate <= 0.15, f"Unexpected fraud rate: {rate:.4f}"


def test_generate_reference_shape():
    with tempfile.TemporaryDirectory() as d:
        path = f"{d}/reference.parquet"
        df = generate_reference(n=2_000, seed=0, output_path=path)
        assert len(df) == 2_000
        assert os.path.exists(path)
        reloaded = pd.read_parquet(path)
        assert len(reloaded) == 2_000


# ── Drift injection tests ──────────────────────────────────────────────────────

def test_abrupt_drift_shifts_distribution():
    """Mean of TransactionAmt should be significantly higher after abrupt drift."""
    no_drift = generate_batch(n=3_000, seed=10, drift_alpha=0.0)
    drifted = generate_batch(n=3_000, seed=11, drift_alpha=1.0)
    mean_base = no_drift["TransactionAmt"].mean()
    mean_drift = drifted["TransactionAmt"].mean()
    assert mean_drift > mean_base * 1.5, (
        f"Expected drift to significantly shift TransactionAmt mean: "
        f"base={mean_base:.2f} drifted={mean_drift:.2f}"
    )


def test_gradual_drift_is_monotone():
    """Mean TransactionAmt should increase monotonically with drift_alpha."""
    means = []
    for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
        df = generate_batch(n=2_000, seed=20, drift_alpha=alpha)
        means.append(df["TransactionAmt"].mean())
    for i in range(len(means) - 1):
        assert means[i] <= means[i + 1] * 1.5, (
            f"Expected monotone increase; got {means}"
        )


def test_label_drift_increases_fraud_rate():
    base = generate_batch(n=5_000, seed=30, label_drift_delta=0.0)
    shifted = generate_batch(n=5_000, seed=31, label_drift_delta=0.1)
    assert shifted[TARGET].mean() > base[TARGET].mean()


def test_concept_drift_only():
    """Covariate distributions should be similar; label rate different."""
    base = generate_batch(n=5_000, seed=40, drift_alpha=0.0, label_drift_delta=0.0)
    shifted = generate_batch(n=5_000, seed=41, drift_alpha=0.0, label_drift_delta=0.2)
    # Feature means should be close
    for col in NUMERICAL_FEATURES[:3]:
        diff = abs(base[col].mean() - shifted[col].mean())
        assert diff < base[col].std() * 0.5, f"Unexpected covariate shift in {col}"
    # Fraud rate should differ
    assert abs(shifted[TARGET].mean() - base[TARGET].mean()) > 0.001


def test_production_sequence_abrupt():
    with tempfile.TemporaryDirectory() as d:
        spec = DriftSpec(mode="abrupt", start_batch=4)
        paths, labels = generate_production_sequence(
            output_dir=d, n_batches=8, n_per_batch=500, drift_spec=spec
        )
        assert len(paths) == 8
        assert len(labels) == 8
        assert labels[:4] == [False, False, False, False]
        assert all(labels[4:])
        for p in paths:
            assert os.path.exists(p)


def test_production_sequence_gradual():
    with tempfile.TemporaryDirectory() as d:
        spec = DriftSpec(mode="gradual", start_batch=2, end_batch=6)
        paths, labels = generate_production_sequence(
            output_dir=d, n_batches=8, n_per_batch=300, drift_spec=spec
        )
        assert labels[0] is False
        assert labels[1] is False
        assert any(labels[2:])  # some gradual drift batches are drifted


def test_drift_ground_truth_precision_recall():
    """
    Simulate a downstream detector that naively flags everything from
    start_batch onward as drift, and verify ground truth is as expected.
    """
    n = 10
    start = 5
    with tempfile.TemporaryDirectory() as d:
        spec = DriftSpec(mode="abrupt", start_batch=start)
        _, labels = generate_production_sequence(
            output_dir=d, n_batches=n, n_per_batch=200, drift_spec=spec
        )
    # Ground truth: first `start` are False, rest True
    assert sum(labels) == n - start
    assert sum(not l for l in labels) == start


# ── DataLoader tests ───────────────────────────────────────────────────────────

def test_dataloader_validate_rejects_missing_column():
    df = generate_batch(n=100, seed=50)
    df = df.drop(columns=[NUMERICAL_FEATURES[0]])
    loader = DataLoader()
    with pytest.raises(ValueError, match="missing columns"):
        loader.load_from_dataframe(df)


def test_dataloader_split():
    df = generate_batch(n=200, seed=51)
    loader = DataLoader()
    df_valid = loader.load_from_dataframe(df)
    X, y = loader.split(df_valid)
    assert list(X.columns) == ALL_FEATURES
    assert y is not None
    assert len(X) == len(y) == 200


def test_dataloader_save_and_load():
    df = generate_batch(n=150, seed=52)
    loader = DataLoader()
    with tempfile.TemporaryDirectory() as d:
        # Patch data_dir temporarily
        loader._cfg = type("cfg", (), {
            "data_dir": d,
            "reference_path": f"{d}/ref.parquet",
            "training_window_days": 30,
        })()
        path = loader.save_batch(df, "test_batch")
        loaded = loader.load_batch(path)
        assert len(loaded) == 150


def test_dataloader_recent_window_empty_dir():
    loader = DataLoader()
    loader._cfg = type("cfg", (), {
        "data_dir": "/nonexistent_path_xyz",
        "training_window_days": 30,
        "reference_path": "/nonexistent_path_xyz/ref.parquet",
    })()
    result = loader.get_recent_window()
    assert result.empty
