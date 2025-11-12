from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from core.config import Settings, load_config


@lru_cache
def load_config() -> Settings:
    return load_config()


SettingsDependencies = Annotated[Settings, Depends(load_config)]
