"""
Load the real IEEE-CIS Fraud Detection dataset from Kaggle
and prepare it for the pipeline.

Steps:
  1. Download from Kaggle (requires kaggle API key) OR point to local CSV
  2. Align column names to our schema
  3. Save as reference.parquet + production batches split by time

Usage (after downloading train_transaction.csv from Kaggle):
    python scripts/load_kaggle_dataset.py --csv path/to/train_transaction.csv

Or with Kaggle CLI:
    kaggle competitions download -c ieee-fraud-detection
    python scripts/load_kaggle_dataset.py --csv train_transaction.csv

Dataset URL: https://www.kaggle.com/c/ieee-fraud-detection
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("OMP_NUM_THREADS", "1")

from src.config import get_settings
from src.ingestion.schema import (
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, TARGET, ALL_FEATURES
)

# Mapping from Kaggle column names to our schema names
# Our schema was designed to match IEEE-CIS — most are identical
KAGGLE_COLUMN_MAP = {
    "TransactionAmt": "TransactionAmt",
    "card1":          "card1",
    "card2":          "card2",
    "card3":          "card3",
    "card5":          "card5",
    "addr1":          "addr1",
    "addr2":          "addr2",
    "dist1":          "dist1",
    "C1":             "C1",
    "C2":             "C2",
    "C6":             "C6",
    "C11":            "C11",
    "D1":             "D1",
    "D10":            "D10",
    "V1":             "V1",
    "V2":             "V2",
    "V3":             "V3",
    "V4":             "V4",
    "ProductCD":      "ProductCD",
    "card4":          "card4",
    "card6":          "card6",
    "P_emaildomain":  "P_emaildomain",
    "R_emaildomain":  "R_emaildomain",
    "isFraud":        "isFraud",
    "TransactionDT":  "TransactionDT",  # used for time-based splitting
}


def load_and_align(csv_path: str) -> pd.DataFrame:
    """Load the Kaggle CSV and align to our feature schema."""
    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Raw shape: {df.shape}")

    # Keep only columns we need
    available = [c for c in KAGGLE_COLUMN_MAP if c in df.columns]
    missing   = [c for c in KAGGLE_COLUMN_MAP if c not in df.columns]
    if missing:
        print(f"  Warning: these columns not found in CSV: {missing}")
        print("  They will be filled with median/mode values.")

    df = df[available].rename(columns=KAGGLE_COLUMN_MAP)

    # Fill any missing schema columns with sensible defaults
    for col in NUMERICAL_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            df[col] = "unknown"
        else:
            df[col] = df[col].fillna("unknown").astype(str)

    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found. "
                         "Make sure you're using train_transaction.csv.")

    print(f"  Fraud rate: {df[TARGET].mean():.3%}")
    print(f"  Aligned shape: {df.shape}")
    return df


def split_by_time(
    df: pd.DataFrame,
    n_batches: int = 10,
    reference_frac: float = 0.4,
) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
    """
    Split by TransactionDT (time) into reference + production batches.

    reference_frac: fraction of earliest transactions used as reference.
    Remaining transactions split into n_batches equally sized windows.
    """
    if "TransactionDT" not in df.columns:
        print("  No TransactionDT column — using random split instead.")
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        df = df.sort_values("TransactionDT").reset_index(drop=True)

    n_ref = int(len(df) * reference_frac)
    reference = df.iloc[:n_ref].copy()
    production = df.iloc[n_ref:].copy()

    batch_size = len(production) // n_batches
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        end   = start + batch_size if i < n_batches - 1 else len(production)
        batch = production.iloc[start:end].copy()
        batch["ingestion_ts"] = pd.Timestamp.now(tz="UTC")
        batch["batch_id"]     = f"kaggle_{i:03d}"
        batches.append(batch)

    print(f"  Reference: {len(reference):,} rows  "
          f"({reference['isFraud'].mean():.3%} fraud)")
    for i, b in enumerate(batches):
        print(f"  Batch {i:03d}: {len(b):,} rows  "
              f"({b['isFraud'].mean():.3%} fraud)")

    return reference[ALL_FEATURES + [TARGET]], batches


def main():
    parser = argparse.ArgumentParser(
        description="Prepare IEEE-CIS Kaggle dataset for the pipeline"
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to train_transaction.csv from Kaggle"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Output data directory (default: settings.data_dir)"
    )
    parser.add_argument(
        "--n-batches", type=int, default=10,
        help="Number of production batches to create (default: 10)"
    )
    parser.add_argument(
        "--ref-frac", type=float, default=0.4,
        help="Fraction of data to use as reference (default: 0.4)"
    )
    args = parser.parse_args()

    cfg = get_settings()
    data_dir = args.data_dir or cfg.data_dir
    ref_path = os.path.join(data_dir, "reference.parquet")
    batch_dir = os.path.join(data_dir, "batches")
    os.makedirs(batch_dir, exist_ok=True)

    # Load and align
    df = load_and_align(args.csv)

    # Split
    reference, batches = split_by_time(df, args.n_batches, args.ref_frac)

    # Save reference
    reference.to_parquet(ref_path, index=False)
    print(f"\nReference saved -> {ref_path}")

    # Save batches
    for i, batch in enumerate(batches):
        path = os.path.join(batch_dir, f"batch_{i:03d}.parquet")
        batch[ALL_FEATURES + [TARGET, "ingestion_ts", "batch_id"]].to_parquet(
            path, index=False
        )

    print(f"Batches saved  -> {batch_dir}/batch_000.parquet ... batch_{len(batches)-1:03d}.parquet")
    print(f"\nDataset ready. Now run:")
    print(f"  python -m src.retraining.baseline")
    print(f"  python scripts/run_experiments.py")


if __name__ == "__main__":
    main()
