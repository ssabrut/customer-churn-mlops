from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from core.config import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


SettingDependencies = Annotated[Settings, Depends(get_settings)]
