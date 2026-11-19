"""Settings loaded from the project's .env file (one level above backend/)."""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=PROJECT_DIR / ".env", extra="ignore")

    database_url: str
    test_database_url: str = ""
    # 127.0.0.1, not "localhost": on Windows "localhost" tries IPv6 first and waits ~2 s per call.
    ollama_url: str = "http://127.0.0.1:11434"
    chat_model: str = "qwen3:8b"
    num_ctx: int = 8192

    secret_key: str
    access_token_minutes: int = 60

    artifacts_dir: Path = PROJECT_DIR / "model-artifacts"
    max_import_mb: int = 20
    max_row_errors_reported: int = 200      # a 20,000-row file with a broken column shouldn't return 20,000 errors


settings = Settings()
