"""
FastAPI application.

Endpoints
---------
POST /auth/login            Public — returns JWT
POST /predict               Viewer+ — fraud prediction, logs every call
GET  /model/status          Viewer+ — current champion info
GET  /drift/history         Viewer+ — recent drift events
GET  /retraining/history    Viewer+ — recent retraining jobs
POST /retrain/trigger       ml_engineer/admin — force retraining
POST /model/rollback        admin — revert to previous champion
GET  /audit-log             admin — full audit trail
GET  /health                Public — liveness probe
"""
from __future__ import annotations

import contextlib
import datetime

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import List, Optional

from src.auth.security import (
    verify_password, create_access_token,
    require_viewer, require_engineer, require_admin,
    get_current_user,
)
from src.config import get_settings
from src.db.audit import write as audit_write, Actions
from src.db.models import (
    User, DriftEvent, RetrainingJob, PromotionDecision,
    AuditLog, ModelVersion,
)
from src.db.session import get_db, SessionLocal, get_engine, Base
from src.db.seed import seed as seed_users
from src.ingestion.schema import ALL_FEATURES
from src.serving.model_store import get_store


# ── Lifespan ───────────────────────────────────────────────────────────────────

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=get_engine())
    seed_users()
    store = get_store()
    if not store.load_champion():
        import logging
        logging.getLogger(__name__).warning(
            "No champion model found — /predict will return 503 until "
            "the baseline script is run."
        )
    yield


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Self-Healing ML Pipeline API",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    username: str


@app.post("/auth/login", response_model=TokenResponse, tags=["auth"])
def login(
    form: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.username == form.username).first()
    if not user or not verify_password(form.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_access_token({"sub": user.username, "role": user.role})
    audit_write(db, actor=user.username, action=Actions.USER_LOGIN)
    return TokenResponse(access_token=token, role=user.role, username=user.username)


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "viewer"


@app.post("/auth/register", tags=["auth"], status_code=201)
def register(
    req: RegisterRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """Admin-only: create a new user."""
    from src.auth.security import hash_password
    if req.role not in ("admin", "ml_engineer", "viewer"):
        raise HTTPException(status_code=400, detail="Invalid role")
    existing = db.query(User).filter(User.username == req.username).first()
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken")
    user = User(
        username=req.username,
        hashed_password=hash_password(req.password),
        role=req.role,
    )
    db.add(user)
    db.commit()
    audit_write(db, actor=current_user.username, action=Actions.USER_CREATED,
                entity_type="user", entity_id=req.username)
    return {"username": user.username, "role": user.role}


@app.get("/auth/me", tags=["auth"])
def me(current_user: User = Depends(require_viewer)):
    return {"username": current_user.username, "role": current_user.role}


# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    features: List[dict]
    """
    List of feature dicts.  Each dict must contain all keys in ALL_FEATURES.
    Example:
      [{"TransactionAmt": 150.0, "card1": 5000, ...}]
    """


class PredictResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    model_version: Optional[str]
    timestamp: str
    n: int


@app.post("/predict", response_model=PredictResponse, tags=["predict"])
def predict(
    req: PredictRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_viewer),
):
    store = get_store()
    if not store.is_ready:
        raise HTTPException(status_code=503, detail="No model loaded")

    # Validate required features
    missing = [c for c in ALL_FEATURES if c not in req.features[0]]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")

    try:
        df = pd.DataFrame(req.features)
        result = store.predict(df)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Log to audit (sampled — every call would be too noisy in production)
    audit_write(
        db,
        actor=current_user.username,
        action=Actions.PREDICTION_MADE,
        entity_type="model",
        entity_id=store.model_version,
        details={"n": len(req.features), "model_version": store.model_version},
    )

    return PredictResponse(
        predictions=result["predictions"],
        probabilities=result["probabilities"],
        model_version=result["model_version"],
        timestamp=ts,
        n=len(req.features),
    )


# ══════════════════════════════════════════════════════════════════════════════
# MODEL STATUS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/model/status", tags=["model"])
def model_status(
    db: Session = Depends(get_db),
    _: User = Depends(require_viewer),
):
    store = get_store()
    champion = (
        db.query(ModelVersion)
        .filter(ModelVersion.stage == "champion")
        .order_by(ModelVersion.created_at.desc())
        .first()
    )
    if champion is None:
        return {"status": "no_model", "model_loaded": store.is_ready}
    return {
        "status": "ok",
        "model_loaded": store.is_ready,
        "version": champion.version,
        "stage": champion.stage,
        "metric_name": champion.metric_name,
        "metric_value": champion.metric_value,
        "mlflow_run_id": champion.mlflow_run_id,
        "created_at": champion.created_at.isoformat() if champion.created_at else None,
    }


# ══════════════════════════════════════════════════════════════════════════════
# DRIFT HISTORY
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/drift/history", tags=["drift"])
def drift_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(require_viewer),
):
    events = (
        db.query(DriftEvent)
        .order_by(DriftEvent.detected_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": e.id,
            "detected_at": e.detected_at.isoformat(),
            "window_id": e.window_id,
            "method": e.method,
            "score": e.score,
            "threshold": e.threshold,
            "is_drift": e.is_drift,
            "triggered_retrain": e.triggered_retrain,
        }
        for e in events
    ]


# ══════════════════════════════════════════════════════════════════════════════
# RETRAINING HISTORY
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/retraining/history", tags=["retraining"])
def retraining_history(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, le=200),
    db: Session = Depends(get_db),
    _: User = Depends(require_viewer),
):
    jobs = (
        db.query(RetrainingJob)
        .order_by(RetrainingJob.started_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )
    return [
        {
            "id": j.id,
            "drift_event_id": j.drift_event_id,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "finished_at": j.finished_at.isoformat() if j.finished_at else None,
            "status": j.status,
            "triggered_by": j.triggered_by,
            "candidate_model_id": j.candidate_model_id,
        }
        for j in jobs
    ]


# ══════════════════════════════════════════════════════════════════════════════
# MANUAL RETRAIN TRIGGER
# ══════════════════════════════════════════════════════════════════════════════

class RetriggerResponse(BaseModel):
    job_id: int
    message: str


@app.post("/retrain/trigger", response_model=RetriggerResponse, tags=["retraining"])
def trigger_retrain(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_engineer),
):
    """
    Immediately enqueue a retraining job bypassing drift detection.
    Requires ml_engineer or admin role.
    """
    job = RetrainingJob(
        status="pending",
        triggered_by=current_user.username,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    audit_write(
        db,
        actor=current_user.username,
        action=Actions.MANUAL_RETRAIN,
        entity_type="retraining_job",
        entity_id=str(job.id),
        details={"triggered_by": current_user.username},
    )

    # In production the Prefect flow picks up pending jobs.
    # For immediate execution, import and call the runner directly.
    try:
        from src.retraining.runner import run_retraining_job
        run_retraining_job(job.id)
    except Exception as e:
        # Non-fatal: job is persisted, orchestrator will retry
        pass

    return RetriggerResponse(job_id=job.id, message="Retraining job enqueued")


# ══════════════════════════════════════════════════════════════════════════════
# MODEL ROLLBACK
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/model/rollback", tags=["model"])
def rollback(
    db: Session = Depends(get_db),
    current_user: User = Depends(require_admin),
):
    """
    Revert to the previous champion.
    The current champion is archived; the most recent archived model is promoted.
    """
    # Find current champion
    current = (
        db.query(ModelVersion)
        .filter(ModelVersion.stage == "champion")
        .order_by(ModelVersion.created_at.desc())
        .first()
    )
    if current is None:
        raise HTTPException(status_code=404, detail="No current champion found")

    # Find most recent archived model
    previous = (
        db.query(ModelVersion)
        .filter(ModelVersion.stage == "archived")
        .order_by(ModelVersion.created_at.desc())
        .first()
    )
    if previous is None:
        raise HTTPException(status_code=409, detail="No archived model to roll back to")

    # Swap
    current.stage = "archived"
    previous.stage = "champion"
    db.commit()

    # Reload serving model
    get_store().reload()

    audit_write(
        db,
        actor=current_user.username,
        action=Actions.MANUAL_ROLLBACK,
        entity_type="model",
        entity_id=str(previous.id),
        details={
            "demoted_version": current.version,
            "restored_version": previous.version,
        },
    )

    return {
        "message": "Rollback complete",
        "restored_version": previous.version,
        "demoted_version": current.version,
    }


# ══════════════════════════════════════════════════════════════════════════════
# AUDIT LOG
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/audit-log", tags=["audit"])
def audit_log(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, le=500),
    action: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    _: User = Depends(require_admin),
):
    q = db.query(AuditLog).order_by(AuditLog.timestamp.desc())
    if action:
        q = q.filter(AuditLog.action == action)
    entries = q.offset(skip).limit(limit).all()
    return [
        {
            "id": e.id,
            "timestamp": e.timestamp.isoformat(),
            "actor": e.actor,
            "action": e.action,
            "entity_type": e.entity_type,
            "entity_id": e.entity_id,
            "details": e.details,
        }
        for e in entries
    ]


# ══════════════════════════════════════════════════════════════════════════════
# HEALTH
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/health", tags=["health"])
def health():
    store = get_store()
    return {
        "status": "ok",
        "model_loaded": store.is_ready,
        "model_version": store.model_version,
    }
