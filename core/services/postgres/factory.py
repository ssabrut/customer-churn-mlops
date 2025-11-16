from functools import lru_cache

# Assuming 'Settings' is the type returned by load_config
from core.config import Settings, load_config
from core.services.postgres import PostgresClient


@lru_cache(maxsize=1)
def make_postgres_service() -> PostgresClient:
    """
    Initializes and returns a singleton PostgresClient instance.

    This function loads application configuration and uses it to
    instantiate the database client. The result is cached using
    lru_cache to ensure only one instance of the client is
    created and returned on subsequent calls.

    Returns:
        PostgresClient: The initialized, cached singleton PostgreSQL client instance.

    Raises:
        RuntimeError: If configuration loading fails (e.g., FileNotFoundError,
                      ValueError for parsing) or if the PostgresClient
                      instantiation fails due to invalid settings (e.g., ValueError).
    """
    try:
        settings: Settings = load_config()
        return PostgresClient(settings)
    except (FileNotFoundError, ValueError) as e:
        # Catching common errors associated with config loading (file not found,
        # parsing errors) or client instantiation (invalid settings).
        raise RuntimeError(f"Failed to initialize Postgres service: {e}") from e
    except Exception as e:
        # A general catch-all for any other unexpected initialization error.
        raise RuntimeError(
            f"An unexpected error occurred during Postgres service initialization: {e}"
        ) from e
