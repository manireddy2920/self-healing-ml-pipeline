"""
Seed script — creates default users and ensures tables exist.
Run once: python -m src.db.seed
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.db.session import SessionLocal, get_engine, Base
from src.db import models  # noqa: registers models
from src.auth.security import hash_password


def seed():
    Base.metadata.create_all(bind=get_engine())

    db = SessionLocal()
    try:
        if db.query(models.User).count() == 0:
            defaults = [
                ("admin",       "admin123",    "admin"),
                ("ml_engineer", "engineer123", "ml_engineer"),
                ("viewer",      "viewer123",   "viewer"),
            ]
            for username, password, role in defaults:
                db.add(models.User(
                    username=username,
                    hashed_password=hash_password(password),
                    role=role,
                ))
            db.commit()
            print("Default users seeded: admin / ml_engineer / viewer")
        else:
            print("Users already exist — skipping seed.")
    finally:
        db.close()


if __name__ == "__main__":
    seed()
