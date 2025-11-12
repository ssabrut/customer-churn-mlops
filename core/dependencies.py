from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from core.config import Settings
from core.config import load_config as core_load_config


@lru_cache
def load_config() -> Settings:
    return core_load_config()


SettingsDependencies = Annotated[Settings, Depends(load_config)]
