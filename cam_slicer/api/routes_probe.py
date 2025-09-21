"""Probe-related routes supporting grid sampling."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from cam_slicer.core.state import app_state
from cam_slicer.sender.service import SenderError, SenderService, SenderStateError


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


router = APIRouter(prefix="/probe", tags=["probe"])


class ProbeGridRequest(BaseModel):
    """Grid probing configuration."""

    roi_mm: List[List[float]] = Field(..., min_items=2, max_items=2)
    step_mm: Union[float, List[float]]
    z_clear: float
    z_probe: float
    feed_probe: float = Field(..., gt=0)
    mode: Optional[str] = "raster"


def get_sender(request: Request) -> SenderService:
    """Retrieve the shared sender instance from FastAPI state."""

    return request.app.state.sender_service


def _normalize_step(step: Union[float, List[float]]) -> tuple[float, float]:
    """Convert the ``step_mm`` payload into an (x, y) tuple."""

    if isinstance(step, (int, float)):
        if step <= 0:
            raise ValueError("step_mm must be positive")
        return float(step), float(step)
    if len(step) != 2:
        raise ValueError("step_mm must be a scalar or a [x, y] pair")
    if step[0] <= 0 or step[1] <= 0:
        raise ValueError("step_mm values must be positive")
    return float(step[0]), float(step[1])


def _generate_grid(roi: List[List[float]], step_x: float, step_y: float) -> List[tuple[float, float]]:
    """Generate raster grid coordinates for probing."""

    (x0, y0), (x1, y1) = roi
    x_min, x_max = sorted((x0, x1))
    y_min, y_max = sorted((y0, y1))

    points: List[tuple[float, float]] = []
    y = y_min
    reverse = False
    while y <= y_max + 1e-9:
        row: List[tuple[float, float]] = []
        x = x_min
        while x <= x_max + 1e-9:
            row.append((round(x, 6), round(y, 6)))
            x += step_x
        if reverse:
            row.reverse()
        points.extend(row)
        y += step_y
        reverse = not reverse
    return points


async def _enqueue_probe(
    sender: SenderService,
    x: float,
    y: float,
    z_clear: float,
    z_probe: float,
    feed_probe: float,
) -> str:
    """Enqueue a single probe point."""

    try:
        return await asyncio.to_thread(
            sender.enqueue_probe_point, x, y, z_clear, z_probe, feed_probe
        )
    except (SenderError, SenderStateError) as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.post("/grid")
async def probe_grid(payload: ProbeGridRequest, sender: SenderService = Depends(get_sender)) -> dict:
    """Queue a grid probing routine covering ``roi_mm``."""

    try:
        step_x, step_y = _normalize_step(payload.step_mm)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    grid = _generate_grid(payload.roi_mm, step_x, step_y)
    if not grid:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty probing grid")

    job_ids = []
    for x, y in grid:
        job_ids.append(
            await _enqueue_probe(
                sender,
                x,
                y,
                payload.z_clear,
                payload.z_probe,
                payload.feed_probe,
            )
        )

    app_state.update(
        last_heightmap={
            "roi_mm": payload.roi_mm,
            "step_mm": [step_x, step_y],
            "points": len(grid),
            "mode": payload.mode or "raster",
        }
    )

    _LOGGER.info("Queued %d probe points", len(job_ids))
    return {
        "count": len(job_ids),
        "jobs": job_ids,
        "roi_mm": payload.roi_mm,
        "mode": payload.mode or "raster",
    }


__all__ = ["router"]
