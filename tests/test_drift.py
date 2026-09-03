"""
Module 6 tests — drift detectors (KS, PSI, Learned) and composite engine.

All tests are pure-function: no DB, no MLflow, no HTTP.
Uses the synthetic generator to create reference/drifted pairs with
known ground-truth, so we can assert precision/recall properties.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ingestion.generator import generate_batch
from src.ingestion.schema import NUMERICAL_FEATURES, ALL_FEATURES
from src.drift.detectors import (
    KSDriftDetector, PSIDriftDetector, LearnedDriftDetector,
    DriftResult, _psi,
)
from src.drift.engine import DriftEngine, CompositeResult


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def reference():
    return generate_batch(n=2_000, seed=0, drift_alpha=0.0)


@pytest.fixture(scope="module")
def stable():
    """Same distribution as reference — should NOT trigger drift."""
    return generate_batch(n=1_000, seed=99, drift_alpha=0.0)


@pytest.fixture(scope="module")
def drifted():
    """Strong drift — should trigger all detectors."""
    return generate_batch(n=1_000, seed=50, drift_alpha=1.0)


# ══════════════════════════════════════════════════════════════════════════════
# PSI helper
# ══════════════════════════════════════════════════════════════════════════════

def test_psi_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 2000)
    assert _psi(x, x) < 0.01


def test_psi_different_distributions():
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 2000)
    cur = rng.normal(3, 1, 2000)   # mean shifted by 3σ
    assert _psi(ref, cur) > 0.2


def test_psi_non_negative():
    rng = np.random.default_rng(2)
    ref = rng.exponential(1, 1000)
    cur = rng.exponential(2, 1000)
    assert _psi(ref, cur) >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# KS Detector
# ══════════════════════════════════════════════════════════════════════════════

class TestKSDetector:
    def test_returns_drift_result(self, reference, stable):
        det = KSDriftDetector()
        result = det.detect(reference, stable)
        assert isinstance(result, DriftResult)
        assert result.method == "ks"

    def test_score_range(self, reference, stable):
        det = KSDriftDetector()
        result = det.detect(reference, stable)
        assert 0.0 <= result.score <= 1.0

    def test_stable_has_low_score(self, reference, stable):
        det = KSDriftDetector(alpha=0.05, threshold=0.3)
        result = det.detect(reference, stable)
        # Same distribution — expect mostly non-drifted features
        # Allow up to 30% FP at alpha=0.05
        assert result.score <= 0.30, (
            f"KS flagged {result.score:.2%} of features on stable data "
            f"(expected ≤30%)"
        )

    def test_drifted_has_high_score(self, reference, drifted):
        det = KSDriftDetector(alpha=0.05, threshold=0.1)
        result = det.detect(reference, drifted)
        assert result.score > 0.5, (
            f"KS score={result.score:.3f} too low for strongly drifted data"
        )
        assert result.is_drift is True

    def test_details_contain_per_feature(self, reference, stable):
        det = KSDriftDetector()
        result = det.detect(reference, stable)
        assert "per_feature" in result.details
        assert len(result.details["per_feature"]) > 0
        first = next(iter(result.details["per_feature"].values()))
        assert "ks_stat" in first
        assert "p_value" in first
        assert "drifted" in first

    def test_feature_subset(self, reference, stable):
        det = KSDriftDetector()
        cols = NUMERICAL_FEATURES[:3]
        result = det.detect(reference, stable, features=cols)
        assert result.details["n_tested"] == 3


# ══════════════════════════════════════════════════════════════════════════════
# PSI Detector
# ══════════════════════════════════════════════════════════════════════════════

class TestPSIDetector:
    def test_returns_drift_result(self, reference, stable):
        det = PSIDriftDetector()
        result = det.detect(reference, stable)
        assert isinstance(result, DriftResult)
        assert result.method == "psi"

    def test_stable_score_near_zero(self, reference, stable):
        det = PSIDriftDetector(threshold=0.2)
        result = det.detect(reference, stable)
        # Same distribution → PSI should be low
        assert result.score < 0.3, (
            f"PSI={result.score:.4f} too high for stable data"
        )

    def test_drifted_score_above_threshold(self, reference, drifted):
        det = PSIDriftDetector(threshold=0.2)
        result = det.detect(reference, drifted)
        assert result.score > 0.2, (
            f"PSI={result.score:.4f} did not exceed threshold for drifted data"
        )
        assert result.is_drift is True

    def test_details_structure(self, reference, stable):
        det = PSIDriftDetector()
        result = det.detect(reference, stable)
        assert "per_feature" in result.details
        for v in result.details["per_feature"].values():
            assert "psi" in v
            assert "drifted" in v

    def test_increasing_drift_increases_psi(self):
        """PSI should increase monotonically as drift_alpha increases."""
        ref = generate_batch(n=1_500, seed=0, drift_alpha=0.0)
        psi_scores = []
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            cur = generate_batch(n=1_000, seed=10, drift_alpha=alpha)
            det = PSIDriftDetector()
            r = det.detect(ref, cur)
            psi_scores.append(r.score)
        # Not strictly monotone due to noise, but final > initial
        assert psi_scores[-1] > psi_scores[0], (
            f"PSI did not increase with drift: {psi_scores}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Learned Detector
# ══════════════════════════════════════════════════════════════════════════════

class TestLearnedDetector:
    def test_returns_drift_result(self, reference, stable):
        det = LearnedDriftDetector()
        result = det.detect(reference, stable)
        assert isinstance(result, DriftResult)
        assert result.method == "learned"

    def test_stable_auc_near_random(self, reference, stable):
        det = LearnedDriftDetector(auc_threshold=0.65)
        result = det.detect(reference, stable)
        # Same distribution → classifier should not beat random much
        assert result.score < 0.75, (
            f"Learned AUC={result.score:.4f} too high on stable data"
        )

    def test_drifted_auc_above_threshold(self, reference, drifted):
        det = LearnedDriftDetector(auc_threshold=0.65)
        result = det.detect(reference, drifted)
        assert result.score > 0.65, (
            f"Learned AUC={result.score:.4f} did not exceed 0.65 for "
            f"strongly drifted data"
        )
        assert result.is_drift is True

    def test_details_contain_top_features(self, reference, drifted):
        det = LearnedDriftDetector()
        result = det.detect(reference, drifted)
        assert "top_drifted_features" in result.details
        assert len(result.details["top_drifted_features"]) > 0

    def test_details_auc_per_fold(self, reference, stable):
        det = LearnedDriftDetector()
        result = det.detect(reference, stable)
        assert "auc_per_fold" in result.details
        assert len(result.details["auc_per_fold"]) == 5


# ══════════════════════════════════════════════════════════════════════════════
# Composite Engine
# ══════════════════════════════════════════════════════════════════════════════

class TestDriftEngine:
    def test_returns_composite_result(self, reference, stable):
        engine = DriftEngine()
        result = engine.evaluate(reference, stable, window_id="w001")
        assert isinstance(result, CompositeResult)

    def test_window_id_preserved(self, reference, stable):
        engine = DriftEngine()
        result = engine.evaluate(reference, stable, window_id="test_window")
        assert result.window_id == "test_window"

    def test_composite_score_in_range(self, reference, stable):
        engine = DriftEngine()
        result = engine.evaluate(reference, stable)
        assert 0.0 <= result.composite_score <= 1.0

    def test_stable_no_drift(self, reference, stable):
        """Same-distribution data should not trigger composite drift."""
        engine = DriftEngine()
        result = engine.evaluate(reference, stable)
        assert result.is_drift is False, (
            f"Composite flagged stable data as drift "
            f"(score={result.composite_score:.4f})"
        )

    def test_drifted_triggers_composite(self, reference, drifted):
        """Strongly drifted data must trigger composite drift."""
        engine = DriftEngine()
        result = engine.evaluate(reference, drifted)
        assert result.is_drift is True, (
            f"Composite missed drift (score={result.composite_score:.4f})"
        )

    def test_drifted_score_higher_than_stable(self, reference, stable, drifted):
        engine = DriftEngine()
        stable_score = engine.evaluate(reference, stable).composite_score
        drift_score = engine.evaluate(reference, drifted).composite_score
        assert drift_score > stable_score, (
            f"Drifted score {drift_score:.4f} ≤ stable score {stable_score:.4f}"
        )

    def test_to_dict_serialisable(self, reference, stable):
        import json
        engine = DriftEngine()
        result = engine.evaluate(reference, stable)
        d = result.to_dict()
        # Should not raise
        json.dumps(d)

    def test_insufficient_data_raises(self, reference):
        engine = DriftEngine()
        tiny = reference.head(5)
        with pytest.raises(ValueError, match="Need ≥30 rows"):
            engine.evaluate(reference, tiny)

    def test_all_sub_detectors_present(self, reference, stable):
        engine = DriftEngine()
        result = engine.evaluate(reference, stable)
        assert result.ks is not None
        assert result.psi is not None
        assert result.learned is not None

    def test_precision_recall_on_synthetic_drift(self):
        """
        End-to-end precision/recall test.

        Generate 10 windows: 4 stable, 6 drifted (abrupt).
        Run the engine on each.  Expect:
          - TP rate (recall) on drifted windows ≥ 0.50
          - FP rate on stable windows ≤ 0.50
        """
        ref = generate_batch(n=2_000, seed=0, drift_alpha=0.0)
        engine = DriftEngine()

        n_stable, n_drifted = 4, 6
        fp = 0
        tp = 0

        for i in range(n_stable):
            cur = generate_batch(n=800, seed=100 + i, drift_alpha=0.0)
            r = engine.evaluate(ref, cur, window_id=f"stable_{i}")
            if r.is_drift:
                fp += 1

        for i in range(n_drifted):
            cur = generate_batch(n=800, seed=200 + i, drift_alpha=1.0)
            r = engine.evaluate(ref, cur, window_id=f"drifted_{i}")
            if r.is_drift:
                tp += 1

        recall = tp / n_drifted
        fpr = fp / n_stable

        assert recall >= 0.50, f"Recall too low: {recall:.2f} ({tp}/{n_drifted})"
        assert fpr <= 0.50, f"FPR too high: {fpr:.2f} ({fp}/{n_stable})"
