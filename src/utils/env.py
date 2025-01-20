import os
import logging

logger = logging.getLogger(__name__)

def get_env_var(var_name: str) -> str:
    """Helper function to get environment variables with error handling"""
    value = os.getenv(var_name)
    if not value:
        logger.error("Environment variable %s not set", var_name)
        raise ValueError(f"{var_name} environment variable not set")
    return value 