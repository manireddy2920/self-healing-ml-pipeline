import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from alembic import context

# Make src importable
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.db.session import Base
from src.db import models  # noqa: F401 – registers all models

config = context.config

# Override URL from environment so docker-compose works
db_url = os.getenv("DATABASE_URL", "sqlite:///./shlp.db")
config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True,
                      dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
