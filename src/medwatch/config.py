"""Configuração centralizada carregada de variáveis de ambiente / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = PROJECT_ROOT / ".env"


class AzureSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AZURE_", env_file=ENV_FILE, extra="ignore")

    speech_key: SecretStr | None = None
    speech_region: str = "brazilsouth"
    speech_language: str = "pt-BR"
    language_endpoint: str | None = None
    language_key: SecretStr | None = None

    @property
    def speech_configured(self) -> bool:
        return bool(self.speech_key and self.speech_key.get_secret_value() and self.speech_region)

    @property
    def language_configured(self) -> bool:
        return bool(
            self.language_key and self.language_key.get_secret_value() and self.language_endpoint
        )


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MEDWATCH_", env_file=ENV_FILE, extra="ignore")

    env: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = False
    azure_mock: bool = False

    data_dir: Path = PROJECT_ROOT / "data"
    models_dir: Path = PROJECT_ROOT / "models"
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    configs_dir: Path = PROJECT_ROOT / "configs"

    pose_backend: Literal["yolo", "openpose"] = "yolo"
    yolo_detection_weights: str = "yolov8n.pt"
    yolo_pose_weights: str = "yolov8n-pose.pt"

    azure: AzureSettings = Field(default_factory=AzureSettings)

    @property
    def logs_dir(self) -> Path:
        return self.outputs_dir / "logs"

    def ensure_dirs(self) -> None:
        for path in (
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.models_dir,
            self.outputs_dir / "reports",
            self.outputs_dir / "alerts",
            self.logs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
