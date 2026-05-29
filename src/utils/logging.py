"""Centralized logging via loguru."""
from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logger(log_dir: str | Path = "logs", level: str = "INFO", rotation: str = "10 MB"):
    """Configure loguru: pretty stderr sink + rotating file sink."""
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | "
               "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>",
    )
    logger.add(
        log_path / "traffic_law_rag_{time:YYYY-MM-DD}.log",
        level=level,
        rotation=rotation,
        encoding="utf-8",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )
    return logger
