"""
Feature schema for the IEEE-CIS Fraud Detection dataset (Kaggle).

We use a realistic subset of features that exhibit clear distributional
properties, making drift easy to inject and detect.  All feature names
and types are declared here so every module has a single source of truth.
"""
from typing import List, Dict, Any

# ── Feature groups ─────────────────────────────────────────────────────────────

NUMERICAL_FEATURES: List[str] = [
    "TransactionAmt",       # transaction amount (USD)
    "card1",                # card type code
    "card2",                # card subtype code
    "card3",                # card category code
    "card5",                # card issuer code
    "addr1",                # billing region
    "addr2",                # billing country code
    "dist1",                # distance from home
    "C1",                   # counting feature – how many addresses correspond to card
    "C2",                   # counting feature – email matches
    "C6",                   # counting feature – transactions on same card
    "C11",                  # counting feature – transactions from same addr
    "D1",                   # timedelta: days between current and last transaction
    "D10",                  # timedelta: same card transaction gap
    "V1",                   # Vesta engineered feature (interaction)
    "V2",                   # Vesta engineered feature
    "V3",                   # Vesta engineered feature
    "V4",                   # Vesta engineered feature
]

CATEGORICAL_FEATURES: List[str] = [
    "ProductCD",            # product code: W H C R S
    "card4",                # card network: visa / mastercard / discover / amex
    "card6",                # card type: credit / debit
    "P_emaildomain",        # purchaser email domain
    "R_emaildomain",        # recipient email domain
]

TARGET: str = "isFraud"

ALL_FEATURES: List[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# ── Baseline distribution parameters (for synthetic generation) ────────────────

NUM_PARAMS: Dict[str, Dict[str, Any]] = {
    "TransactionAmt": {"dist": "lognormal", "mean": 4.5,  "sigma": 1.2,  "clip": (0.5, 10_000)},
    "card1":          {"dist": "randint",   "low": 1000,  "high": 18_000},
    "card2":          {"dist": "choice",    "vals": [111, 117, 121, 150, 204, 225, 226, 232, 320, 360]},
    "card3":          {"dist": "choice",    "vals": [150, 185]},
    "card5":          {"dist": "choice",    "vals": [102, 117, 138, 166, 184, 204, 226]},
    "addr1":          {"dist": "randint",   "low": 100,   "high": 540},
    "addr2":          {"dist": "choice",    "vals": [87, 60, 96, 65, 32]},
    "dist1":          {"dist": "lognormal", "mean": 3.0,  "sigma": 1.5,  "clip": (0, 5_000)},
    "C1":             {"dist": "poisson",   "lam": 1.5},
    "C2":             {"dist": "poisson",   "lam": 1.2},
    "C6":             {"dist": "poisson",   "lam": 1.1},
    "C11":            {"dist": "poisson",   "lam": 1.0},
    "D1":             {"dist": "lognormal", "mean": 2.5,  "sigma": 1.2,  "clip": (0, 600)},
    "D10":            {"dist": "lognormal", "mean": 2.0,  "sigma": 1.3,  "clip": (0, 600)},
    "V1":             {"dist": "normal",    "mean": 0.0,  "std": 1.0},
    "V2":             {"dist": "normal",    "mean": 0.0,  "std": 1.0},
    "V3":             {"dist": "normal",    "mean": 0.0,  "std": 1.0},
    "V4":             {"dist": "normal",    "mean": 0.0,  "std": 1.0},
}

CAT_PROBS: Dict[str, List] = {
    "ProductCD":     [("W", 0.60), ("H", 0.20), ("C", 0.10), ("R", 0.05), ("S", 0.05)],
    "card4":         [("visa", 0.55), ("mastercard", 0.30), ("discover", 0.10), ("amex", 0.05)],
    "card6":         [("debit", 0.65), ("credit", 0.35)],
    "P_emaildomain": [("gmail.com", 0.40), ("yahoo.com", 0.25), ("hotmail.com", 0.15),
                      ("outlook.com", 0.10), ("icloud.com", 0.10)],
    "R_emaildomain": [("gmail.com", 0.35), ("yahoo.com", 0.20), ("hotmail.com", 0.15),
                      ("outlook.com", 0.10), ("anonymous.com", 0.20)],
}

# Base fraud rate
BASE_FRAUD_RATE: float = 0.035
