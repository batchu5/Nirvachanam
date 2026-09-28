"""Application configuration — loaded from environment variables."""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Central config, loaded from .env or environment variables."""

    # --- GitHub App ---
    github_app_id: str = Field(default="", description="GitHub App ID")
    github_private_key_file: str = Field(
        default="private-key.pem", description="Path to PEM private key file for GitHub App"
    )
    github_webhook_secret: str = Field(
        default="dev-secret", description="HMAC secret for webhook verification"
    )

    @property
    def github_private_key(self) -> str:
        """Read the GitHub App private key from the PEM file."""
        key_path = Path(self.github_private_key_file)
        if key_path.exists():
            return key_path.read_text(encoding="utf-8")
        return ""

    # --- Redis (local) ---
    redis_host: str = Field(default="localhost", description="Redis host")
    redis_port: int = Field(default=6379, description="Redis port")
    redis_password: str | None = Field(default=None, description="Redis password (optional)")

    # --- Postgres ---
    database_url: str = Field(
        default="postgresql://pr_review:pr_review@localhost:5432/pr_review",
        description="Postgres connection string",
    )

    # --- App ---
    review_debounce_seconds: int = Field(
        default=60, description="Debounce window before triggering review (seconds)"
    )
    log_level: str = Field(default="INFO", description="Logging level")
    environment: str = Field(default="development", description="Runtime environment")

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


# Singleton — import this everywhere
settings = Settings()
