"""FastAPI router exposing GRBL sender operations."""

from __future__ import annotations

import asyncio
import logging
import tempfile
import threading
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, Field
from starlette.websockets import WebSocket, WebSocketDisconnect

from cam_slicer.sender.service import (
    SenderError,
    SenderService,
    SenderStateError,
)


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


class SenderEventManager:
    """Thread-safe fan-out for sender events towards websocket clients."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribers: set[asyncio.Queue] = set()
        self._lock = threading.RLock()

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the asyncio loop used for thread-safe callbacks."""

        self._loop = loop

    def subscribe(self) -> asyncio.Queue:
        """Register a new queue for websocket consumption."""

        queue: asyncio.Queue = asyncio.Queue()
        with self._lock:
            self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue) -> None:
        """Remove a queue from the broadcast list."""

        with self._lock:
            self._subscribers.discard(queue)

    def publish(self, event: dict) -> None:
        """Forward events into all subscriber queues."""

        loop = self._loop
        if loop is None:
            return
        with self._lock:
            subscribers = list(self._subscribers)
        for queue in subscribers:
            loop.call_soon_threadsafe(queue.put_nowait, event)


router = APIRouter(prefix="/sender", tags=["sender"])


class OpenPortRequest(BaseModel):
    """Request body for opening a serial port."""

    port: str
    baud: int = 115200


class LineRequest(BaseModel):
    """Single line execution request."""

    gcode: str = Field(..., min_length=1)


class JogRequest(BaseModel):
    """Jog move request."""

    mode: Literal["rel", "abs"]
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    feed: float = Field(default=1500.0, gt=0)


class JogResponse(BaseModel):
    """Response for jogging commands."""

    job_id: str


class LineResponse(BaseModel):
    """Response for queued G-code lines."""

    job_id: str


class StreamResponse(BaseModel):
    """Response when streaming a full G-code program."""

    job_id: str
    file_path: str


def get_sender(request: Request) -> SenderService:
    """Resolve the shared SenderService from the FastAPI app state."""

    return request.app.state.sender_service


def _handle_sender_exception(exc: Exception) -> None:
    """Convert sender errors into HTTP exceptions."""

    if isinstance(exc, SenderStateError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    if isinstance(exc, SenderError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    raise exc


@router.get("/ports")
async def list_ports(sender: SenderService = Depends(get_sender)) -> dict:
    """List available serial ports on the host."""

    ports = await asyncio.to_thread(sender.list_ports)
    return {"ports": ports}


@router.post("/open")
async def open_port(payload: OpenPortRequest, sender: SenderService = Depends(get_sender)) -> dict:
    """Open a serial connection to the GRBL controller."""

    success, message = await asyncio.to_thread(sender.open, payload.port, payload.baud)
    if not success:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=message)
    return {"status": "opened", "port": payload.port, "baud": payload.baud}


@router.get("/status")
async def sender_status(sender: SenderService = Depends(get_sender)) -> dict:
    """Return the cached machine status snapshot."""

    return await asyncio.to_thread(sender.status)


@router.post("/line", response_model=LineResponse)
async def enqueue_line(payload: LineRequest, sender: SenderService = Depends(get_sender)) -> LineResponse:
    """Queue a single G-code line."""

    try:
        job_id = await asyncio.to_thread(sender.enqueue_line, payload.gcode)
    except Exception as exc:  # pragma: no cover - defensive
        _handle_sender_exception(exc)
    return LineResponse(job_id=job_id)


@router.post("/jog", response_model=JogResponse)
async def enqueue_jog(payload: JogRequest, sender: SenderService = Depends(get_sender)) -> JogResponse:
    """Queue a jog command."""

    try:
        job_id = await asyncio.to_thread(
            sender.enqueue_jog,
            payload.mode,
            payload.dx,
            payload.dy,
            payload.dz,
            payload.feed,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _handle_sender_exception(exc)
    return JogResponse(job_id=job_id)


async def _invoke_simple(sender: SenderService, method: str) -> dict:
    """Invoke a simple sender method without arguments."""

    try:
        await asyncio.to_thread(getattr(sender, method))
    except Exception as exc:  # pragma: no cover - defensive
        _handle_sender_exception(exc)
    return {"status": method}


@router.post("/hold")
async def hold(sender: SenderService = Depends(get_sender)) -> dict:
    """Issue the GRBL hold command."""

    return await _invoke_simple(sender, "hold")


@router.post("/start")
async def start(sender: SenderService = Depends(get_sender)) -> dict:
    """Resume execution after a hold."""

    return await _invoke_simple(sender, "start")


@router.post("/reset")
async def reset(sender: SenderService = Depends(get_sender)) -> dict:
    """Reset the controller."""

    return await _invoke_simple(sender, "reset")


@router.post("/jog_cancel")
async def jog_cancel(sender: SenderService = Depends(get_sender)) -> dict:
    """Cancel an active jog."""

    return await _invoke_simple(sender, "jog_cancel")


@router.post("/stream", response_model=StreamResponse)
async def stream_gcode(
    file: UploadFile = File(...),
    sender: SenderService = Depends(get_sender),
) -> StreamResponse:
    """Stream a G-code file via the sender service."""

    uploads_dir = Path(tempfile.gettempdir()) / "cam_slicer_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "program.gcode").suffix or ".gcode"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=uploads_dir) as tmp:
        destination = Path(tmp.name)
        while True:
            chunk = await file.read(64 * 1024)
            if not chunk:
                break
            tmp.write(chunk)
    await file.close()
    _LOGGER.info("Stored upload %s to %s", file.filename, destination)

    try:
        job_id = await asyncio.to_thread(sender.enqueue_file, str(destination))
    except Exception as exc:  # pragma: no cover - defensive
        _handle_sender_exception(exc)
    return StreamResponse(job_id=job_id, file_path=str(destination))


@router.websocket("/ws/sender")
async def sender_events(websocket: WebSocket) -> None:
    """Stream sender events over a websocket connection."""

    await websocket.accept()
    manager: SenderEventManager = websocket.app.state.sender_events
    sender: SenderService = websocket.app.state.sender_service
    queue = manager.subscribe()
    try:
        await websocket.send_json({"type": "state", "data": await asyncio.to_thread(sender.status)})
        while True:
            event = await queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        _LOGGER.debug("Sender websocket disconnected")
    except Exception as exc:  # pragma: no cover - defensive
        _LOGGER.warning("Sender websocket error: %s", exc)
    finally:
        manager.unsubscribe(queue)


__all__ = ["router", "SenderEventManager"]
