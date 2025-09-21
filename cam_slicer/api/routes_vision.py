"""Vision related FastAPI routes."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from cam_slicer.core.state import app_state


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


router = APIRouter(prefix="/vision", tags=["vision"])


class CalibrationRequest(BaseModel):
    """Payload containing the minimum required calibration triplet."""

    p1_px: List[float] = Field(..., min_items=2, max_items=2)
    p1_mm: List[float] = Field(..., min_items=2, max_items=2)
    p2_px: List[float] = Field(..., min_items=2, max_items=2)
    p2_mm: List[float] = Field(..., min_items=2, max_items=2)
    p3_px: List[float] = Field(..., min_items=2, max_items=2)
    p3_mm: List[float] = Field(..., min_items=2, max_items=2)
    ref_pts_px: Optional[List[List[float]]] = None
    mode: Optional[str] = "affine"


class CalibrationResponse(BaseModel):
    """Structured calibration response."""

    status: str
    A_px2cnc: List[List[float]]
    ref_pts_px: List[List[float]]


class RelockResponse(BaseModel):
    """Response after a relock request."""

    status: str
    prefer_id: Optional[str] = None
    frame_bytes: Optional[int] = None
    filename: Optional[str] = None


class Detection(BaseModel):
    """Detected object description."""

    kind: str
    confidence: float
    points: List[List[float]]
    metadata: dict = Field(default_factory=dict)


class DetectionRequest(BaseModel):
    """Vision detection request."""

    kind: str = "rectangle"


def _det3(matrix: List[List[float]]) -> float:
    """Compute determinant of a 3x3 matrix."""

    return (
        matrix[0][0] * matrix[1][1] * matrix[2][2]
        + matrix[0][1] * matrix[1][2] * matrix[2][0]
        + matrix[0][2] * matrix[1][0] * matrix[2][1]
        - matrix[0][2] * matrix[1][1] * matrix[2][0]
        - matrix[0][1] * matrix[1][0] * matrix[2][2]
        - matrix[0][0] * matrix[1][2] * matrix[2][1]
    )


def _replace_column(matrix: List[List[float]], column: List[float], index: int) -> List[List[float]]:
    """Return a copy of ``matrix`` with one column replaced."""

    clone = [row[:] for row in matrix]
    for i in range(3):
        clone[i][index] = column[i]
    return clone


def _solve_affine(px_pts: List[List[float]], mm_pts: List[List[float]]) -> List[List[float]]:
    """Solve the affine transformation matrix between pixels and machine space."""

    base = [[px[0], px[1], 1.0] for px in px_pts]
    det = _det3(base)
    if abs(det) < 1e-9:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Calibration points are collinear")

    tx = [mm[0] for mm in mm_pts]
    ty = [mm[1] for mm in mm_pts]

    row_x_det = _det3(_replace_column(base, tx, 0))
    row_y_det = _det3(_replace_column(base, ty, 0))
    col_x = [c / det for c in [row_x_det, _det3(_replace_column(base, tx, 1)), _det3(_replace_column(base, tx, 2))]]
    col_y = [c / det for c in [row_y_det, _det3(_replace_column(base, ty, 1)), _det3(_replace_column(base, ty, 2))]]

    matrix = [
        [col_x[0], col_x[1], col_x[2]],
        [col_y[0], col_y[1], col_y[2]],
        [0.0, 0.0, 1.0],
    ]
    return matrix


def _transform_points(points: Iterable[Iterable[float]], matrix: List[List[float]]) -> List[List[float]]:
    """Transform pixel coordinates using the calibration matrix."""

    transformed: List[List[float]] = []
    for point in points:
        if len(point) < 2:
            continue
        x, y = float(point[0]), float(point[1])
        vec = (x, y, 1.0)
        x_mm = matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2]
        y_mm = matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2]
        if len(matrix) >= 3:
            w = matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2]
            if abs(w) > 1e-9:
                x_mm /= w
                y_mm /= w
        transformed.append([x_mm, y_mm])
    return transformed


@router.post("/calibrate", response_model=CalibrationResponse)
async def calibrate(payload: CalibrationRequest) -> CalibrationResponse:
    """Compute the affine transform from three calibration pairs."""

    px_pts = [payload.p1_px, payload.p2_px, payload.p3_px]
    mm_pts = [payload.p1_mm, payload.p2_mm, payload.p3_mm]
    matrix = _solve_affine(px_pts, mm_pts)
    ref_pts = payload.ref_pts_px or px_pts
    updated = app_state.update(A_px2cnc=matrix, ref_pts_px=ref_pts)
    _LOGGER.info("Calibration updated via /vision/calibrate")
    return CalibrationResponse(status="ok", A_px2cnc=matrix, ref_pts_px=updated.ref_pts_px or [])


@router.post("/relock", response_model=RelockResponse)
async def relock(prefer_id: Optional[str] = None, frame: UploadFile | None = File(None)) -> RelockResponse:
    """Handle relock requests optionally carrying a new frame."""

    frame_bytes = None
    filename = None
    if frame is not None:
        data = await frame.read()
        frame_bytes = len(data)
        filename = frame.filename
        _LOGGER.debug("Received relock frame %s (%d bytes)", filename, frame_bytes)

    if prefer_id:
        _LOGGER.info("Relock requested with prefer_id=%s", prefer_id)

    return RelockResponse(status="queued", prefer_id=prefer_id, frame_bytes=frame_bytes, filename=filename)


@router.post("/detect", response_model=List[Detection])
async def detect(payload: DetectionRequest) -> List[Detection]:
    """Return detected objects based on stored calibration data."""

    state = app_state.read()
    if not state.ref_pts_px:
        _LOGGER.warning("Detection requested but no reference points are stored")
        return []
    if not state.A_px2cnc:
        return [
            Detection(kind=payload.kind, confidence=0.5, points=state.ref_pts_px, metadata={"space": "pixels"})
        ]

    points_mm = _transform_points(state.ref_pts_px, state.A_px2cnc)
    xs = [pt[0] for pt in points_mm]
    ys = [pt[1] for pt in points_mm]
    bbox = [[min(xs), min(ys)], [max(xs), max(ys)]]
    detection = Detection(
        kind=payload.kind,
        confidence=0.9,
        points=points_mm,
        metadata={"bbox_mm": bbox},
    )
    return [detection]


__all__ = ["router"]
