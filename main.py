"""Command-line bootstrap for the Smart Building AI Agent."""

from src.config.config import settings
from src.utils.logging import configure_logging, get_logger


logger = get_logger(__name__)


def main() -> None:
    """Initialize shared infrastructure for the future application runtime."""
    configure_logging(settings.log_directory, settings.log_level)
    logger.info("Smart Building AI Agent scaffold initialized")
    logger.info("Configure an application service before running optimization")


if __name__ == "__main__":
    main()

