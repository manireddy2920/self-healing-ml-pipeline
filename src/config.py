"""Central settings loaded from environment / .env file."""
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # Database
    database_url: str = "sqlite:///./shlp.db"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "shlp"
    mlflow_model_name: str = "credit_risk_champion"

    # Auth
    jwt_secret_key: str = "insecure_dev_secret_change_in_prod"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 480

    # Serving
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Drift
    drift_ks_threshold: float = 0.05
    drift_psi_threshold: float = 0.2
    drift_learned_auc_threshold: float = 0.65
    drift_composite_threshold: float = 0.5
    drift_debounce_windows: int = 2
    drift_cooldown_minutes: int = 30

    # Retraining / Validation
    validation_metric: str = "f1"
    promotion_threshold_delta: float = 0.0
    max_consecutive_failures: int = 3
    training_window_days: int = 30

    # Data
    data_dir: str = "./data"
    reference_path: str = "./data/reference.parquet"

    # Prefect
    prefect_api_url: str = "http://localhost:4200/api"

    # Dashboard
    streamlit_api_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
