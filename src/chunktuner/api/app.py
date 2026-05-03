"""FastAPI application factory (routes + optional middleware)."""

from __future__ import annotations

from fastapi import FastAPI

from chunktuner.api.auth import TokenAuthMiddleware
from chunktuner.api.routes import router


def create_app() -> FastAPI:
    """Build the HTTP API used by tests and local ``uvicorn`` runs."""
    app = FastAPI()
    app.add_middleware(TokenAuthMiddleware)
    app.include_router(router)
    return app
