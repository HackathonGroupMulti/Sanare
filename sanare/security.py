from __future__ import annotations

import os
from collections.abc import Callable

from fastapi import Request
from starlette.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

PUBLIC_PATHS = {
    "/health",
    "/capabilities",
    "/docs",
    "/redoc",
    "/openapi.json",
}


class OptionalApiKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        expected_key = os.getenv("SANARE_API_KEY")
        if not expected_key or request.url.path in PUBLIC_PATHS:
            return await call_next(request)

        provided_key = request.headers.get("x-sanare-api-key")
        auth_header = request.headers.get("authorization", "")
        bearer_key = auth_header.removeprefix("Bearer ").strip() if auth_header.startswith("Bearer ") else None

        if provided_key == expected_key or bearer_key == expected_key:
            return await call_next(request)
        return JSONResponse({"detail": "invalid or missing Sanare API key"}, status_code=401)
