"""Application logging setup.

The helper creates consistent console and file logging for CLI, simulation,
and Streamlit entry points. Modules should call ``get_logger(__name__)`` and
avoid configuring handlers themselves.
"""

import logging
from pathlib import Path


def configure_logging(log_directory: Path, level: str = "INFO") -> None:
    """Configure process-wide logging with console and rotating-file-ready paths."""
    log_directory.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_directory / "application.log", encoding="utf-8"),
        ],
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a named logger for a module or bounded context."""
    return logging.getLogger(name)

