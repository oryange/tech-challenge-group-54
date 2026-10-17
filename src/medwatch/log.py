"""Setup de logging. Nunca logar PII (nomes, transcrições completas)."""

import sys

from loguru import logger

from medwatch.config import Settings

_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{line}</cyan> | {message}"
)


def setup_logging(settings: Settings) -> None:
    logger.remove()
    logger.add(sys.stderr, level=settings.log_level, format=_FORMAT, serialize=settings.log_json)
    logger.add(
        settings.logs_dir / "medwatch.log",
        level="DEBUG",
        rotation="10 MB",
        retention=5,
        serialize=True,
        enqueue=True,
    )
