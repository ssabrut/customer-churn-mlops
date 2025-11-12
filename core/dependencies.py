"""
Defines FastAPI dependency injectors for the application.

This module provides reusable dependencies, such as the application settings,
which can be injected into route handlers. Using 'lru_cache' ensures that
the configuration is loaded only once.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends
from loguru import logger

from core.config import Settings
from core.config import load_config as core_load_config


@lru_cache
def load_config() -> Settings:
    """
    Loads application settings and caches the result.

    This function acts as a FastAPI dependency. It calls the core
    configuration loader and uses 'lru_cache' to ensure this expensive
    operation (reading .env files, validating variables) happens
    only once, on the first request or application startup.

    Returns:
        Settings: A Pydantic Settings object containing the application's
                  configuration.

    Raises:
        RuntimeError: If the configuration fails to load due to missing
                      environment variables or other 'EnvironmentError'.
    """
    try:
        return core_load_config()
    except EnvironmentError as e:
        logger.error(f"FATAL: Configuration failed to load: {e}")
        raise RuntimeError(f"Failed to load application configuration: {e}")


SettingsDependencies = Annotated[Settings, Depends(load_config)]
