"""High-level vision helpers for calibration and relock workflows."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Literal, Sequence, Tuple

import cv2
import numpy as np
from pydantic import BaseModel, Field, validator


__all__ = [
    "PixelPt",
    "Detection",
    "PoseMap",
    "set_undistort_hook",
    "affine_from_3pts",
    "detect_aruco_corners",
    "detect_rectangle",
    "estimate_delta_affine",
    "update_px2cnc_with_delta",
    "px_to_mm",
    "roi_from_detection",
]


# ---------------------------------------------------------------------------
# Logging configuration (log.txt hook)
# ---------------------------------------------------------------------------
_LOGGER = logging.getLogger(__name__)
if not _LOGGER.handlers:
    _LOGGER.setLevel(logging.INFO)
    _LOG_PATH = Path(__file__).resolve().parents[2] / "log.txt"
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(_LOG_PATH, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        _LOGGER.addHandler(handler)
    except OSError:
        _LOGGER.addHandler(logging.NullHandler())
else:
    _LOGGER.addHandler(logging.NullHandler())


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class PixelPt(BaseModel):
    """2D pixel coordinate."""

    x: float
    y: float

    class Config:
        frozen = True

    def as_tuple(self) -> Tuple[float, float]:
        """Return the point as an ``(x, y)`` tuple."""

        return float(self.x), float(self.y)


class Detection(BaseModel):
    """Normalized detection description."""

    kind: Literal["rectangle", "aruco"]
    conf: float = Field(..., ge=0.0, le=1.0)
    bbox_px: List[float] = Field(..., min_items=4, max_items=4)
    angle_deg: float
    corners_px: List[PixelPt] = Field(..., min_items=4, max_items=4)

    @validator("bbox_px")
    def _validate_bbox(cls, value: Sequence[float]) -> List[float]:
        if len(value) != 4:
            raise ValueError("Bounding box must contain [cx, cy, w, h]")
        return [float(v) for v in value]


class PoseMap(BaseModel):
    """Affine transform between pixel and CNC coordinates."""

    A_px2cnc: List[List[float]] = Field(..., min_items=2, max_items=2)

    @validator("A_px2cnc")
    def _validate_matrix(cls, value: Sequence[Sequence[float]]) -> List[List[float]]:
        rows = []
        for row in value:
            if len(row) != 3:
                raise ValueError("Affine matrix must have shape 2x3")
            rows.append([float(col) for col in row])
        return rows

    def as_matrix(self) -> np.ndarray:
        """Return the affine map as a ``(2, 3)`` ndarray."""

        return np.asarray(self.A_px2cnc, dtype=float)


# ---------------------------------------------------------------------------
# Module level undistort hook
# ---------------------------------------------------------------------------
UndistortHook = Callable[[np.ndarray], np.ndarray]
_undistort_hook: UndistortHook | None = None


def set_undistort_hook(hook: UndistortHook | None) -> None:
    """Register an optional undistortion hook used before detection."""

    global _undistort_hook
    _undistort_hook = hook


def _apply_undistort(frame: np.ndarray) -> np.ndarray:
    """Apply the optional undistortion hook if present."""

    if _undistort_hook is None:
        return frame
    try:
        return _undistort_hook(frame)
    except Exception as exc:  # pragma: no cover - defensive guard
        _LOGGER.exception("Undistort hook failed: %s", exc)
        return frame


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------
def _as_point(value: Sequence[float]) -> Tuple[float, float]:
    if len(value) < 2:
        raise ValueError("Point must contain at least two coordinates")
    return float(value[0]), float(value[1])


def _sorted_corners(corners: np.ndarray) -> np.ndarray:
    """Return corners ordered as top-left, top-right, bottom-right, bottom-left."""

    if corners.shape != (4, 2):
        corners = corners.reshape(-1, 2)
    sums = corners.sum(axis=1)
    diffs = (corners[:, 0] - corners[:, 1])

    tl = corners[np.argmin(sums)]
    br = corners[np.argmax(sums)]
    tr = corners[np.argmin(diffs)]
    bl = corners[np.argmax(diffs)]
    ordered = np.vstack([tl, tr, br, bl]).astype(np.float32)
    return ordered


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------
def affine_from_3pts(
    p1_px: Sequence[float],
    p1_mm: Sequence[float],
    p2_px: Sequence[float],
    p2_mm: Sequence[float],
    p3_px: Sequence[float],
    p3_mm: Sequence[float],
) -> np.ndarray:
    """Compute the affine transform mapping pixel coordinates to millimetres."""

    px = np.array([
        [*_as_point(p1_px), 1.0],
        [*_as_point(p2_px), 1.0],
        [*_as_point(p3_px), 1.0],
    ])
    mm = np.array([
        _as_point(p1_mm),
        _as_point(p2_mm),
        _as_point(p3_mm),
    ])
    if abs(np.linalg.det(px)) < 1e-9:
        raise ValueError("Calibration points are collinear")

    solution_x = np.linalg.solve(px, mm[:, 0])
    solution_y = np.linalg.solve(px, mm[:, 1])
    A = np.vstack([solution_x, solution_y]).astype(np.float64)
    return A


_ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
_ARUCO_PARAMS = cv2.aruco.DetectorParameters()
_ARUCO_DETECTOR = cv2.aruco.ArucoDetector(_ARUCO_DICT, _ARUCO_PARAMS)


def detect_aruco_corners(
    image_bgr: np.ndarray, prefer_id: int | None = None
) -> Tuple[np.ndarray | None, int | None]:
    """Detect an ArUco marker and return its sorted corner array."""

    if image_bgr is None or image_bgr.size == 0:
        return None, None

    frame = _apply_undistort(image_bgr)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = _ARUCO_DETECTOR.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None, None

    ids = ids.flatten()
    chosen_index = 0
    if prefer_id is not None:
        matches = np.where(ids == prefer_id)[0]
        if matches.size:
            chosen_index = int(matches[0])
    if prefer_id is None or chosen_index == 0:
        areas = [
            cv2.contourArea(c.reshape(-1, 2).astype(np.float32))
            for c in corners
        ]
        chosen_index = int(np.argmax(areas))

    selected_corners = corners[chosen_index].reshape(4, 2)
    ordered = _sorted_corners(selected_corners)
    marker_id = int(ids[chosen_index])
    _LOGGER.debug("Detected ArUco id=%s", marker_id)
    return ordered, marker_id


def detect_rectangle(image_bgr: np.ndarray) -> Detection | None:
    """Find the largest rectangular contour in the frame."""

    if image_bgr is None or image_bgr.size == 0:
        return None

    frame = _apply_undistort(image_bgr)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    if w < 1 or h < 1:
        return None

    corners = cv2.boxPoints(rect)
    ordered = _sorted_corners(corners)

    image_area = float(frame.shape[0] * frame.shape[1])
    rect_area = float(w * h)
    confidence = max(0.0, min(1.0, rect_area / max(image_area, 1.0)))

    detection = Detection(
        kind="rectangle",
        conf=confidence,
        bbox_px=[float(cx), float(cy), float(w), float(h)],
        angle_deg=float(angle),
        corners_px=[PixelPt(x=float(pt[0]), y=float(pt[1])) for pt in ordered],
    )
    return detection


def estimate_delta_affine(
    ref_pts_px: np.ndarray, curr_pts_px: np.ndarray
) -> np.ndarray:
    """Estimate the affine delta mapping reference pixels to current pixels."""

    if ref_pts_px.shape != curr_pts_px.shape:
        raise ValueError("Reference and current points must share the same shape")
    if ref_pts_px.shape[0] < 2:
        raise ValueError("At least two point pairs are required")

    ref = np.asarray(ref_pts_px, dtype=np.float32)
    curr = np.asarray(curr_pts_px, dtype=np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(ref, curr, method=cv2.LMEDS)
    if matrix is None:
        _LOGGER.warning("estimateAffinePartial2D failed, using identity delta")
        matrix = np.eye(2, 3, dtype=np.float32)
    return matrix.astype(np.float64)


def update_px2cnc_with_delta(
    A_px2cnc: np.ndarray, delta_px_affine: np.ndarray
) -> np.ndarray:
    """Update the pixel-to-CNC matrix using an observed pixel delta."""

    A = np.asarray(A_px2cnc, dtype=np.float64)
    if A.shape != (2, 3):
        raise ValueError("A_px2cnc must be a (2, 3) matrix")

    delta = np.asarray(delta_px_affine, dtype=np.float64)
    if delta.shape != (2, 3):
        raise ValueError("delta_px_affine must be a (2, 3) matrix")

    A_aug = np.vstack([A, [0.0, 0.0, 1.0]])
    delta_aug = np.vstack([delta, [0.0, 0.0, 1.0]])
    try:
        delta_inv = np.linalg.inv(delta_aug)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Delta transform is not invertible") from exc

    updated = A_aug @ delta_inv
    return updated[:2, :]


def px_to_mm(A_px2cnc: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    """Convert a pixel coordinate to machine millimetres."""

    A = np.asarray(A_px2cnc, dtype=np.float64)
    if A.shape != (2, 3):
        raise ValueError("A_px2cnc must be a (2, 3) matrix")

    vec = np.array([float(u), float(v), 1.0])
    result = A @ vec
    return float(result[0]), float(result[1])


def roi_from_detection(
    det: Detection, margin_mm: float, A_px2cnc: np.ndarray
) -> Tuple[float, float, float, float]:
    """Return a millimetre ROI expanded by ``margin_mm`` around the detection."""

    if margin_mm < 0:
        raise ValueError("margin_mm must be non-negative")

    points_mm = np.array([px_to_mm(A_px2cnc, pt.x, pt.y) for pt in det.corners_px])
    xmin = float(points_mm[:, 0].min()) - margin_mm
    xmax = float(points_mm[:, 0].max()) + margin_mm
    ymin = float(points_mm[:, 1].min()) - margin_mm
    ymax = float(points_mm[:, 1].max()) + margin_mm
    return xmin, ymin, xmax, ymax
