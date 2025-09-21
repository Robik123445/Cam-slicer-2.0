"""FastAPI application factory wiring routes and singletons."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from cam_slicer.api.routes_intent import router as intent_router
from cam_slicer.api.routes_probe import router as probe_router
from cam_slicer.api.routes_sender import SenderEventManager, router as sender_router
from cam_slicer.api.routes_vision import router as vision_router
from cam_slicer.core.orchestrator import Orchestrator
from cam_slicer.sender.service import SenderService


_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    _LOG_PATH = Path(__file__).resolve().parents[2] / "log.txt"
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        _FILE_HANDLER = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        _FILE_HANDLER.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        _LOGGER.addHandler(_FILE_HANDLER)
    except OSError:
        _LOGGER.addHandler(logging.NullHandler())
else:
    _LOGGER.addHandler(logging.NullHandler())


def _create_app() -> FastAPI:
    """Internal helper to construct the FastAPI application."""

    app = FastAPI(title="Cam Slicer API", version="2.0")

    sender_service = SenderService()
    sender_events = SenderEventManager()
    orchestrator = Orchestrator(sender_service)

    sender_service.set_event_sink(sender_events.publish)

    app.state.sender_service = sender_service
    app.state.sender_events = sender_events
    app.state.orchestrator = orchestrator

    allowed_origins: List[str] = [
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.on_event("startup")
    async def _startup() -> None:
        """Capture the running loop for event fan-out."""

        loop = asyncio.get_running_loop()
        sender_events.set_loop(loop)
        _LOGGER.info("Cam Slicer API ready")

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        """Cleanup resources on shutdown."""

        _LOGGER.info("Cam Slicer API shutting down")

    app.include_router(sender_router)
    app.include_router(vision_router)
    app.include_router(probe_router)
    app.include_router(intent_router)

    @app.get("/health")
    async def health() -> dict:
        """Simple health check endpoint."""

        return {"status": "ok"}

    return app


app = _create_app()

__all__ = ["app"]
