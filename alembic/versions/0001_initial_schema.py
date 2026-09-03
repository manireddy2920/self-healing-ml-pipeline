"""Initial schema

Revision ID: 0001
Revises:
Create Date: 2024-01-01
"""
from alembic import op
import sqlalchemy as sa

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("username", sa.String(64), nullable=False, unique=True),
        sa.Column("hashed_password", sa.String(256), nullable=False),
        sa.Column("role", sa.String(32), nullable=False, server_default="viewer"),
        sa.Column("is_active", sa.Boolean, server_default="1"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("ix_users_username", "users", ["username"])

    op.create_table(
        "models",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("version", sa.String(64), nullable=False),
        sa.Column("stage", sa.String(32), nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("metric_name", sa.String(64)),
        sa.Column("metric_value", sa.Float),
        sa.Column("mlflow_run_id", sa.String(128)),
        sa.Column("artifact_path", sa.Text),
    )

    op.create_table(
        "drift_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("detected_at", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("window_id", sa.String(64)),
        sa.Column("method", sa.String(64)),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("threshold", sa.Float, nullable=False),
        sa.Column("is_drift", sa.Boolean, nullable=False),
        sa.Column("triggered_retrain", sa.Boolean, server_default="0"),
        sa.Column("details", sa.JSON),
    )
    op.create_index("ix_drift_events_detected_at", "drift_events", ["detected_at"])

    op.create_table(
        "retraining_jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("drift_event_id", sa.Integer, sa.ForeignKey("drift_events.id"), nullable=True),
        sa.Column("started_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime, nullable=True),
        sa.Column("status", sa.String(32), server_default="running"),
        sa.Column("candidate_model_id", sa.Integer, sa.ForeignKey("models.id"), nullable=True),
        sa.Column("triggered_by", sa.String(64), server_default="system"),
    )

    op.create_table(
        "promotion_decisions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("retraining_job_id", sa.Integer, sa.ForeignKey("retraining_jobs.id"), nullable=False),
        sa.Column("candidate_model_id", sa.Integer, sa.ForeignKey("models.id"), nullable=False),
        sa.Column("champion_model_id", sa.Integer, sa.ForeignKey("models.id"), nullable=True),
        sa.Column("decision", sa.String(16), nullable=False),
        sa.Column("candidate_metric", sa.Float),
        sa.Column("champion_metric", sa.Float),
        sa.Column("decided_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("decided_by", sa.String(64), server_default="system"),
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("actor", sa.String(64), nullable=False),
        sa.Column("action", sa.String(64), nullable=False),
        sa.Column("entity_type", sa.String(64)),
        sa.Column("entity_id", sa.String(128)),
        sa.Column("timestamp", sa.DateTime, nullable=False, server_default=sa.func.now()),
        sa.Column("details", sa.JSON),
    )
    op.create_index("ix_audit_log_timestamp", "audit_log", ["timestamp"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])


def downgrade() -> None:
    op.drop_table("audit_log")
    op.drop_table("promotion_decisions")
    op.drop_table("retraining_jobs")
    op.drop_table("drift_events")
    op.drop_table("models")
    op.drop_table("users")
