"""
Module 5 tests — FastAPI serving layer.

Uses TestClient (synchronous HTTPX wrapper) with an in-memory SQLite DB
and a pre-loaded fake pipeline so no MLflow server is required.
"""
from __future__ import annotations

import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.db.session import Base
from src.db import models as db_models
from src.auth.security import hash_password, create_access_token
from src.serving.model_store import get_store, ModelStore


# ── Test DB + app setup ────────────────────────────────────────────────────────

TEST_DB_URL = "sqlite:///./test_api_shlp.db"


def _make_engine():
    return create_engine(TEST_DB_URL, connect_args={"check_same_thread": False})


@pytest.fixture(scope="module")
def app_client():
    """
    Spin up a TestClient with:
      - fresh in-memory DB (all tables created)
      - seeded admin + viewer users
      - a fake champion pipeline in the model store
    """
    from src.config import get_settings
    from src.db.session import get_db
    from src.serving.api import app

    engine = _make_engine()
    Base.metadata.create_all(bind=engine)
    TestSession = sessionmaker(bind=engine)

    # Seed users
    db = TestSession()
    db.add(db_models.User(
        username="admin", hashed_password=hash_password("admin123"), role="admin"
    ))
    db.add(db_models.User(
        username="engineer", hashed_password=hash_password("eng123"), role="ml_engineer"
    ))
    db.add(db_models.User(
        username="viewer", hashed_password=hash_password("view123"), role="viewer"
    ))
    # Seed a champion model row
    db.add(db_models.ModelVersion(
        version="1", stage="champion", metric_name="f1",
        metric_value=0.82, mlflow_run_id="abc123",
        artifact_path="runs:/abc123/model",
    ))
    db.commit()
    db.close()

    # Override DB dependency
    def override_get_db():
        s = TestSession()
        try:
            yield s
        finally:
            s.close()

    app.dependency_overrides[get_db] = override_get_db

    # Inject a dummy pipeline into the store (bypasses MLflow)
    from src.ingestion.schema import ALL_FEATURES
    from src.ingestion.generator import generate_batch
    from src.retraining.preprocessing import build_preprocessor
    from sklearn.pipeline import Pipeline

    dummy_pipe = Pipeline([
        ("pre", build_preprocessor()),
        ("clf", DummyClassifier(strategy="stratified", random_state=0)),
    ])
    sample = generate_batch(n=100, seed=0)
    dummy_pipe.fit(sample[ALL_FEATURES], sample["isFraud"])

    store = get_store()
    store.set_pipeline(dummy_pipe, version="1", db_id=1)

    with TestClient(app, raise_server_exceptions=False) as client:
        yield client

    # Cleanup
    app.dependency_overrides.clear()
    import gc, os, time
    gc.collect(); time.sleep(0.1)
    try:
        if os.path.exists("test_api_shlp.db"):
            os.remove("test_api_shlp.db")
    except PermissionError:
        pass


def _token(client, username="admin", password="admin123") -> str:
    resp = client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200, f"Login failed: {resp.text}"
    return resp.json()["access_token"]


def _auth(client, username="admin", password="admin123"):
    return {"Authorization": f"Bearer {_token(client, username, password)}"}


# ── Auth tests ─────────────────────────────────────────────────────────────────

def test_login_success(app_client):
    r = app_client.post("/auth/login", data={"username": "admin", "password": "admin123"})
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["role"] == "admin"
    assert body["username"] == "admin"


def test_login_wrong_password(app_client):
    r = app_client.post("/auth/login", data={"username": "admin", "password": "wrong"})
    assert r.status_code == 401


def test_login_unknown_user(app_client):
    r = app_client.post("/auth/login", data={"username": "nobody", "password": "x"})
    assert r.status_code == 401


def test_protected_route_no_token(app_client):
    r = app_client.get("/model/status")
    assert r.status_code == 401


def test_protected_route_bad_token(app_client):
    r = app_client.get("/model/status", headers={"Authorization": "Bearer bad_token"})
    assert r.status_code == 401


# ── Health tests ───────────────────────────────────────────────────────────────

def test_health(app_client):
    r = app_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


# ── Model status tests ─────────────────────────────────────────────────────────

def test_model_status_authenticated(app_client):
    r = app_client.get("/model/status", headers=_auth(app_client))
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["version"] == "1"
    assert body["metric_name"] == "f1"


def test_model_status_viewer_allowed(app_client):
    r = app_client.get(
        "/model/status",
        headers=_auth(app_client, "viewer", "view123"),
    )
    assert r.status_code == 200


# ── Predict tests ──────────────────────────────────────────────────────────────

def _sample_features() -> list[dict]:
    from src.ingestion.generator import generate_batch
    from src.ingestion.schema import ALL_FEATURES
    df = generate_batch(n=3, seed=42)
    return df[ALL_FEATURES].to_dict(orient="records")


def test_predict_success(app_client):
    r = app_client.post(
        "/predict",
        json={"features": _sample_features()},
        headers=_auth(app_client),
    )
    assert r.status_code == 200
    body = r.json()
    assert "predictions" in body
    assert "probabilities" in body
    assert body["n"] == 3
    assert len(body["predictions"]) == 3
    assert all(p in (0, 1) for p in body["predictions"])
    assert all(0.0 <= p <= 1.0 for p in body["probabilities"])


def test_predict_batch_size(app_client):
    from src.ingestion.generator import generate_batch
    from src.ingestion.schema import ALL_FEATURES
    df = generate_batch(n=20, seed=55)
    r = app_client.post(
        "/predict",
        json={"features": df[ALL_FEATURES].to_dict(orient="records")},
        headers=_auth(app_client),
    )
    assert r.status_code == 200
    assert r.json()["n"] == 20


def test_predict_missing_feature(app_client):
    from src.ingestion.schema import ALL_FEATURES
    feats = _sample_features()
    del feats[0][ALL_FEATURES[0]]   # remove first feature
    r = app_client.post(
        "/predict",
        json={"features": feats},
        headers=_auth(app_client),
    )
    assert r.status_code == 422


def test_predict_no_token(app_client):
    r = app_client.post("/predict", json={"features": _sample_features()})
    assert r.status_code == 401


def test_predict_model_version_in_response(app_client):
    r = app_client.post(
        "/predict",
        json={"features": _sample_features()},
        headers=_auth(app_client),
    )
    assert r.json()["model_version"] == "1"


# ── Drift history tests ────────────────────────────────────────────────────────

def test_drift_history_empty(app_client):
    r = app_client.get("/drift/history", headers=_auth(app_client))
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_drift_history_viewer_allowed(app_client):
    r = app_client.get(
        "/drift/history",
        headers=_auth(app_client, "viewer", "view123"),
    )
    assert r.status_code == 200


# ── Retraining history tests ───────────────────────────────────────────────────

def test_retraining_history_empty(app_client):
    r = app_client.get("/retraining/history", headers=_auth(app_client))
    assert r.status_code == 200
    assert isinstance(r.json(), list)


# ── Rollback tests ─────────────────────────────────────────────────────────────

def test_rollback_no_archived_model(app_client):
    """Should 409 when there's nothing to roll back to."""
    r = app_client.post("/model/rollback", headers=_auth(app_client))
    assert r.status_code == 409


def test_rollback_admin_only(app_client):
    r = app_client.post(
        "/model/rollback",
        headers=_auth(app_client, "viewer", "view123"),
    )
    assert r.status_code == 403


def test_rollback_engineer_forbidden(app_client):
    r = app_client.post(
        "/model/rollback",
        headers=_auth(app_client, "engineer", "eng123"),
    )
    assert r.status_code == 403


# ── Audit log tests ────────────────────────────────────────────────────────────

def test_audit_log_admin_only(app_client):
    r = app_client.get("/audit-log", headers=_auth(app_client))
    assert r.status_code == 200
    logs = r.json()
    # After login + predictions, there should be entries
    assert isinstance(logs, list)


def test_audit_log_viewer_forbidden(app_client):
    r = app_client.get(
        "/audit-log",
        headers=_auth(app_client, "viewer", "view123"),
    )
    assert r.status_code == 403


def test_audit_log_contains_login_entry(app_client):
    r = app_client.get("/audit-log?limit=200", headers=_auth(app_client))
    actions = [e["action"] for e in r.json()]
    assert "USER_LOGIN" in actions


# ── Retrain trigger tests ─────────────────────────────────────────────────────

def test_retrain_trigger_engineer_allowed(app_client):
    r = app_client.post(
        "/retrain/trigger",
        headers=_auth(app_client, "engineer", "eng123"),
    )
    # Either 200 (runner succeeded) or 200 (job created, runner silently failed)
    assert r.status_code == 200
    body = r.json()
    assert "job_id" in body


def test_retrain_trigger_viewer_forbidden(app_client):
    r = app_client.post(
        "/retrain/trigger",
        headers=_auth(app_client, "viewer", "view123"),
    )
    assert r.status_code == 403


# ── RBAC matrix ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("endpoint,method,role,password,expected", [
    ("/model/status",     "GET",  "viewer",  "view123", 200),
    ("/drift/history",    "GET",  "viewer",  "view123", 200),
    ("/retraining/history","GET", "viewer",  "view123", 200),
    ("/audit-log",        "GET",  "viewer",  "view123", 403),
    ("/audit-log",        "GET",  "engineer","eng123",  403),
    ("/audit-log",        "GET",  "admin",   "admin123",200),
    ("/model/rollback",   "POST", "engineer","eng123",  403),
    ("/model/rollback",   "POST", "admin",   "admin123",409),  # 409 = no archived model
    ("/retrain/trigger",  "POST", "viewer",  "view123", 403),
    ("/retrain/trigger",  "POST", "engineer","eng123",  200),
])
def test_rbac_matrix(app_client, endpoint, method, role, password, expected):
    headers = _auth(app_client, role, password)
    if method == "GET":
        r = app_client.get(endpoint, headers=headers)
    else:
        r = app_client.post(endpoint, headers=headers)
    assert r.status_code == expected, (
        f"{method} {endpoint} as {role} → expected {expected}, got {r.status_code}: {r.text}"
    )
