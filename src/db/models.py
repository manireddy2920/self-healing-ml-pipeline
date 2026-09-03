"""ORM models for the Self-Healing ML Pipeline."""
import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, ForeignKey,
    Integer, String, Text, JSON,
)
from sqlalchemy.orm import relationship

from src.db.session import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    hashed_password = Column(String(256), nullable=False)
    role = Column(String(32), nullable=False, default="viewer")
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))


class ModelVersion(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True)
    version = Column(String(64), nullable=False)
    stage = Column(String(32), nullable=False)           # champion | challenger | archived
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    metric_name = Column(String(64))
    metric_value = Column(Float)
    mlflow_run_id = Column(String(128))
    artifact_path = Column(Text)

    retraining_jobs = relationship(
        "RetrainingJob", back_populates="candidate_model",
        foreign_keys="RetrainingJob.candidate_model_id",
    )
    promotions_as_candidate = relationship(
        "PromotionDecision", back_populates="candidate_model",
        foreign_keys="PromotionDecision.candidate_model_id",
    )
    promotions_as_champion = relationship(
        "PromotionDecision", back_populates="champion_model",
        foreign_keys="PromotionDecision.champion_model_id",
    )


class DriftEvent(Base):
    __tablename__ = "drift_events"

    id = Column(Integer, primary_key=True)
    detected_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False, index=True)
    window_id = Column(String(64))
    method = Column(String(64))                          # ks | psi | learned | composite
    score = Column(Float, nullable=False)
    threshold = Column(Float, nullable=False)
    is_drift = Column(Boolean, nullable=False)
    triggered_retrain = Column(Boolean, default=False)
    details = Column(JSON)

    retraining_jobs = relationship("RetrainingJob", back_populates="drift_event")


class RetrainingJob(Base):
    __tablename__ = "retraining_jobs"

    id = Column(Integer, primary_key=True)
    drift_event_id = Column(Integer, ForeignKey("drift_events.id"), nullable=True)
    started_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    finished_at = Column(DateTime, nullable=True)
    status = Column(String(32), default="running")       # running | success | failed
    candidate_model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    triggered_by = Column(String(64), default="system")

    drift_event = relationship("DriftEvent", back_populates="retraining_jobs")
    candidate_model = relationship(
        "ModelVersion", back_populates="retraining_jobs",
        foreign_keys=[candidate_model_id],
    )
    promotion_decisions = relationship("PromotionDecision", back_populates="retraining_job")


class PromotionDecision(Base):
    __tablename__ = "promotion_decisions"

    id = Column(Integer, primary_key=True)
    retraining_job_id = Column(Integer, ForeignKey("retraining_jobs.id"), nullable=False)
    candidate_model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    champion_model_id = Column(Integer, ForeignKey("models.id"), nullable=True)
    decision = Column(String(16), nullable=False)        # promoted | rejected
    candidate_metric = Column(Float)
    champion_metric = Column(Float)
    decided_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    decided_by = Column(String(64), default="system")

    retraining_job = relationship("RetrainingJob", back_populates="promotion_decisions")
    candidate_model = relationship(
        "ModelVersion", back_populates="promotions_as_candidate",
        foreign_keys=[candidate_model_id],
    )
    champion_model = relationship(
        "ModelVersion", back_populates="promotions_as_champion",
        foreign_keys=[champion_model_id],
    )


class AuditLog(Base):
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True)
    actor = Column(String(64), nullable=False)
    action = Column(String(64), nullable=False, index=True)
    entity_type = Column(String(64))
    entity_id = Column(String(128))
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc), nullable=False, index=True)
    details = Column(JSON)

