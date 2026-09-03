"""
Append-only audit log writer.

Every call inserts a new row — never updates or deletes — preserving
a tamper-evident trail of all system and user actions.
"""
import datetime
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from src.db.models import AuditLog


class Actions:
    # Auth
    USER_LOGIN = "USER_LOGIN"
    USER_CREATED = "USER_CREATED"

    # Drift
    DRIFT_CHECKED = "DRIFT_CHECKED"
    DRIFT_DETECTED = "DRIFT_DETECTED"
    DRIFT_HEALTHY = "DRIFT_HEALTHY"

    # Retraining
    RETRAIN_TRIGGERED = "RETRAIN_TRIGGERED"
    RETRAIN_COMPLETED = "RETRAIN_COMPLETED"
    RETRAIN_FAILED = "RETRAIN_FAILED"

    # Validation / Promotion
    VALIDATION_PASSED = "VALIDATION_PASSED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    VALIDATION_FAILED = "VALIDATION_FAILED"
    MODEL_PROMOTED = "MODEL_PROMOTED"
    MODEL_ROLLBACK_KEPT = "MODEL_ROLLBACK_KEPT"   # champion kept, challenger archived
    HUMAN_REVIEW_FLAGGED = "HUMAN_REVIEW_FLAGGED"

    # Manual overrides
    MANUAL_RETRAIN = "MANUAL_RETRAIN"
    MANUAL_ROLLBACK = "MANUAL_ROLLBACK"

    # Prediction
    PREDICTION_MADE = "PREDICTION_MADE"


def write(
    db: Session,
    actor: str,
    action: str,
    entity_type: Optional[str] = None,
    entity_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> AuditLog:
    """Insert an immutable audit log entry and return it."""
    entry = AuditLog(
        actor=actor,
        action=action,
        entity_type=entity_type,
        entity_id=str(entity_id) if entity_id is not None else None,
        timestamp=datetime.datetime.now(datetime.timezone.utc),
        details=details or {},
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry
