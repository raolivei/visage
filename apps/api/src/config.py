"""
Visage API Configuration

Environment-based configuration using pydantic-settings.
"""

from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    app_name: str = "Visage API"
    debug: bool = False
    api_prefix: str = "/api"

    # Database
    database_url: str = "postgresql+asyncpg://visage:visage@localhost:5436/visage"

    # Redis
    redis_url: str = "redis://localhost:6383"

    # MinIO / S3
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "visage"
    minio_secure: bool = False

    # CORS
    cors_origins: list[str] = ["http://localhost:3004", "https://visage.eldertree.local"]

    # File upload
    max_upload_size_mb: int = 20
    allowed_extensions: list[str] = [".jpg", ".jpeg", ".png", ".webp"]
    min_photos_per_pack: int = 8
    max_photos_per_pack: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
