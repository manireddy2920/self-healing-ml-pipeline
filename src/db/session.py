"""SQLAlchemy engine + session factory."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

from src.config import get_settings


class Base(DeclarativeBase):
    pass


def get_engine():
    cfg = get_settings()
    connect_args = {}
    if cfg.database_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
    return create_engine(
        cfg.database_url,
        connect_args=connect_args,
        pool_pre_ping=True,
    )


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
