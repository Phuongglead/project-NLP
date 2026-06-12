from __future__ import annotations

import json
import os
from typing import List

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    GOOGLE_GEMINI_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    SA_AQG_USE_STUBS: str = "false"
    CORS_ALLOW_ORIGINS: str = '["http://localhost:3000","http://127.0.0.1:3000"]'

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def cors_origins(self) -> List[str]:
        raw = self.CORS_ALLOW_ORIGINS
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        return [x.strip() for x in raw.split(",") if x.strip()]


settings = Settings()

# Mirror Gemini key for legacy batch_processor paths.
if settings.GOOGLE_GEMINI_API_KEY and not os.environ.get("GOOGLE_GEMINI_API_KEY"):
    os.environ["GOOGLE_GEMINI_API_KEY"] = settings.GOOGLE_GEMINI_API_KEY
if settings.GEMINI_API_KEY and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = settings.GEMINI_API_KEY
elif settings.GOOGLE_GEMINI_API_KEY and not os.environ.get("GEMINI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = settings.GOOGLE_GEMINI_API_KEY
if settings.SA_AQG_USE_STUBS:
    os.environ["SA_AQG_USE_STUBS"] = settings.SA_AQG_USE_STUBS
