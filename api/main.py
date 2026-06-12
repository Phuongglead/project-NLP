from __future__ import annotations

import os
import sys

# Ensure project root is on PYTHONPATH when running uvicorn from api/
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import settings
from api.routes import interview

app = FastAPI(
    title="SA-AQG Interview API",
    version="0.1.0",
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
)

cors_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(interview.router, prefix="/api/interview", tags=["interview"])


@app.get("/api/health", tags=["system"])
async def health() -> dict:
    return {"status": "ok", "stubs": os.environ.get("SA_AQG_USE_STUBS", "false")}
