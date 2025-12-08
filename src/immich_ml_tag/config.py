"""Configuration management for immich-ml-tag."""

import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Settings:
    """Environment-based settings for the application."""

    immich_db: str = field(
        default_factory=lambda: os.environ.get("IMMICH_DB_DATABASE_NAME", "immich")
    )
    immich_db_user: str = field(
        default_factory=lambda: os.environ.get("IMMICH_DB_USERNAME", "postgres")
    )
    immich_db_password: str = field(
        default_factory=lambda: os.environ.get("IMMICH_DB_PASSWORD", "yourpassword")
    )
    immich_db_host: str = field(
        default_factory=lambda: os.environ.get("IMMICH_DB_HOST", "localhost")
    )
    immich_db_port: str = field(
        default_factory=lambda: os.environ.get("IMMICH_DB_PORT", "5433")
    )
    immich_api_key: str = field(
        default_factory=lambda: os.environ.get("IMMICH_API_KEY", "your_api_key")
    )
    immich_url: str = field(
        default_factory=lambda: os.environ.get(
            "IMMICH_URL", "https://immich.example.com"
        )
    )
    ml_resource_path: Path = field(
        default_factory=lambda: Path(
            os.environ.get("ML_RESOURCE_PATH", "./ml_resources")
        )
    )

    # Tag naming conventions
    ml_tag_suffix: str = "_predicted"
    contrast_tag: str = "ML Negative Examples"

    def __post_init__(self):
        self.ml_resource_path = Path(self.ml_resource_path)
        self.ml_resource_path.mkdir(parents=True, exist_ok=True)

    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.immich_db_user}:{self.immich_db_password}"
            f"@{self.immich_db_host}:{self.immich_db_port}/{self.immich_db}"
        )


# Global settings instance
settings = Settings()


@dataclass
class Model:
    """Represents a trained ML model for a specific tag."""

    tag: str
    model_path: Path
    last_trained: datetime | None = None


class ConfigStore:
    """SQLite-based configuration and model registry."""

    def __init__(self, folder: Path | None = None):
        self.folder = folder or settings.ml_resource_path
        self._init_storage()

    def _init_storage(self):
        self.db_path = self.folder / "config.db"
        self.conn = sqlite3.connect(self.db_path)
        self.cur = self.conn.cursor()
        self.cur.execute("""
            CREATE TABLE IF NOT EXISTS config (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                last_inference DATETIME,
                last_trained DATETIME
            )
        """)
        self.cur.execute(
            "CREATE TABLE IF NOT EXISTS models (tag TEXT PRIMARY KEY, model_path TEXT, training_data_hash TEXT)"
        )
        self.conn.commit()

    def close(self):
        """Close the database connection."""
        self.conn.close()

    def _delete_all_models(self):
        """Delete all model entries from the database."""
        self.cur.execute("DELETE FROM models")
        self.conn.commit()

    def get_models(self) -> list[Model]:
        """Get all registered models."""
        self.cur.execute("SELECT tag, model_path FROM models")
        data = self.cur.fetchall()
        models = []
        for tag, model_path in data:
            models.append(Model(tag=tag, model_path=Path(model_path)))
        return models

    def register_model(
        self, tag: str, model_path: Path, training_data_hash: str | None = None
    ):
        """Register or update a model in the database."""
        self.cur.execute(
            "INSERT OR REPLACE INTO models (tag, model_path, training_data_hash) VALUES (?, ?, ?)",
            (tag, str(model_path), training_data_hash),
        )
        self.conn.commit()

    def get_training_data_hash(self, tag: str) -> str | None:
        """Get the training data hash for a specific tag."""
        self.cur.execute(
            "SELECT training_data_hash FROM models WHERE tag = ?",
            (tag,),
        )
        data = self.cur.fetchone()
        if data is None or data[0] is None:
            return None
        return data[0]

    @property
    def last_trained(self) -> datetime | None:
        """Get the last training timestamp."""
        self.cur.execute("SELECT last_trained FROM config WHERE id = 1")
        data = self.cur.fetchone()
        if data is None or data[0] is None:
            return None
        return datetime.fromisoformat(data[0])

    @last_trained.setter
    def last_trained(self, value: datetime):
        """Set the last training timestamp."""
        self.cur.execute(
            "INSERT OR REPLACE INTO config (id, last_trained) VALUES (1, ?)",
            (value.isoformat(),),
        )
        self.conn.commit()

    @property
    def last_inference(self) -> datetime | None:
        """Get the last inference timestamp."""
        self.cur.execute("SELECT last_inference FROM config WHERE id = 1")
        data = self.cur.fetchone()
        if data is None or data[0] is None:
            return None
        return datetime.fromisoformat(data[0])

    @last_inference.setter
    def last_inference(self, value: datetime):
        """Set the last inference timestamp."""
        self.cur.execute(
            "INSERT OR REPLACE INTO config (id, last_inference) VALUES (1, ?)",
            (value.isoformat(),),
        )
        self.conn.commit()
