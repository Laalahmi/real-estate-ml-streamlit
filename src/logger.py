import logging
from src.config import LOG_DIR, LOG_FILE


def setup_logger():
    """
    Configure and return a project-wide logger.
    """

    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(exist_ok=True)

    logging.basicConfig(
        filename=LOG_FILE,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    return logging.getLogger(__name__)