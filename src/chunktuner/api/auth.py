"""Optional HTTP API authentication."""

from __future__ import annotations

import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Enforces Bearer token auth when ``CHUNK_TUNER_API_TOKEN`` is set."""

    def __init__(self, app, token: str | None = None):
        super().__init__(app)
        raw = token if token is not None else os.environ.get("CHUNK_TUNER_API_TOKEN")
        self._token = raw.strip() if raw else None

    async def dispatch(self, request: Request, call_next):
        if not self._token:
            return await call_next(request)
        if request.url.path == "/health":
            return await call_next(request)
        auth = request.headers.get("Authorization") or ""
        parts = auth.split(None, 1)
        if (
            len(parts) < 2
            or parts[0].lower() != "bearer"
            or parts[1] != self._token
        ):
            return JSONResponse({"detail": "Unauthorized"}, status_code=401)
        return await call_next(request)
