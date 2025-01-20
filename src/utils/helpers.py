"""Helper functions"""

import logging

logger = logging.getLogger(__name__)

def log_message(message: str, level: str = "info",):
    """Helper function to log messages"""
    if level == "info":
        logger.info(message)
    elif level == "error":
        logger.error(message)
    else:
        logger.warning(message)
