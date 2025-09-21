"""Intent-level API bridging UI actions to orchestrator flows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from cam_slicer.core.orchestrator import Orchestrator, OrchestratorError


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


router = APIRouter(prefix="/intent", tags=["intent"])


class GuideToObjectRequest(BaseModel):
    """Intent payload for guiding the machine to an object."""

    object_class: str = Field(..., alias="class")
    strategy: str = "center"
    approach: Optional[float] = Field(default=None, alias="approach")
    clearance: Optional[float] = Field(default=None, alias="clearance")

    class Config:
        allow_population_by_field_name = True


class MeasureObjectRequest(BaseModel):
    """Intent payload requesting metrology."""

    object_class: str = Field(..., alias="class")
    metrics: Optional[List[str]] = None

    class Config:
        allow_population_by_field_name = True


class FindEdgesRequest(BaseModel):
    """Intent payload for edge detection."""

    mode: str = "rect"
    return_mode: str = Field(default="points", alias="return")

    class Config:
        allow_population_by_field_name = True


def get_orchestrator(request: Request) -> Orchestrator:
    """Retrieve orchestrator from FastAPI application state."""

    return request.app.state.orchestrator


@router.post("/guide_to_object")
async def guide_to_object(
    payload: GuideToObjectRequest, orchestrator: Orchestrator = Depends(get_orchestrator)
) -> dict:
    """Invoke the orchestrator guidance workflow."""

    safe_z = payload.approach if payload.approach is not None else 5.0
    clearance = payload.clearance if payload.clearance is not None else 10.0
    try:
        result = await orchestrator.guide_to_object(
            kind=payload.object_class,
            strategy=payload.strategy,
            safe_z=safe_z,
            xy_clearance=clearance,
        )
    except OrchestratorError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return result


@router.post("/measure_object")
async def measure_object(
    payload: MeasureObjectRequest, orchestrator: Orchestrator = Depends(get_orchestrator)
) -> dict:
    """Invoke object measurement routine."""

    try:
        result = await orchestrator.measure_object(kind=payload.object_class)
    except OrchestratorError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    if payload.metrics:
        result["requested_metrics"] = payload.metrics
    return result


@router.post("/find_edges")
async def find_edges(
    payload: FindEdgesRequest, orchestrator: Orchestrator = Depends(get_orchestrator)
) -> dict:
    """Invoke edge finding routine."""

    try:
        result = await orchestrator.find_edges(
            mode=payload.mode,
            return_mode=payload.return_mode,
        )
    except OrchestratorError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    return result


__all__ = ["router"]
