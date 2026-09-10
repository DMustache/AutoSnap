from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_path: Path | None = Path("models/car_models_fastapi_bundle_v1.zip")
    hf_model_repo: str | None = "Dmustache/autosnap-car-model"
    hf_model_revision: str = "main"
    hf_cache_directory: Path = Path("models/.hf-cache")
    max_upload_bytes: int = 10 * 1024 * 1024
    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/png", "image/webp")


@lru_cache
def get_settings() -> Settings:
    return Settings()
