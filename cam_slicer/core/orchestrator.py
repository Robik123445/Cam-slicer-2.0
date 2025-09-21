"""High-level motion orchestration helpers built on top of SenderService."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

from cam_slicer.core.state import AppState, app_state
from cam_slicer.sender.service import SenderService, SenderStateError


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


class OrchestratorError(RuntimeError):
    """Raised when orchestrated workflows cannot be executed."""


class Orchestrator:
    """Coordinate high-level workflows (vision → motion → probing)."""

    def __init__(self, sender: SenderService) -> None:
        """Store the sender dependency for later asynchronous usage."""

        self._sender = sender

    async def guide_to_object(
        self,
        kind: str,
        strategy: str = "center",
        safe_z: float = 5.0,
        xy_clearance: float = 10.0,
    ) -> dict:
        """Approach an object using the chosen strategy.

        The function reads calibration data from :mod:`cam_slicer.core.state`. If
        ``allow_execute_moves`` is ``True`` a simple two-step G-code approach is
        dispatched (raise Z, then move in XY). Otherwise only the plan is
        returned so callers can inspect it first.
        """

        state = self._require_calibration()
        target_x, target_y = self._resolve_target_point(state, strategy)
        plan = [f"G0 Z{safe_z:.3f}"]
        if xy_clearance > 0:
            approach_y = target_y + xy_clearance
            plan.append(f"G0 X{target_x:.3f} Y{approach_y:.3f}")
        plan.append(f"G0 X{target_x:.3f} Y{target_y:.3f}")

        executed = False
        job_ids: List[str] = []
        if state.allow_execute_moves:
            _LOGGER.info(
                "Executing guide_to_object for %s with plan %s", kind, plan
            )
            for line in plan:
                job_ids.append(await self._enqueue_line(line))
            executed = True
        else:
            _LOGGER.info(
                "Planning guide_to_object for %s without execution (safety lock)",
                kind,
            )

        return {
            "kind": kind,
            "strategy": strategy,
            "target": {"x": target_x, "y": target_y, "safe_z": safe_z},
            "plan": plan,
            "executed": executed,
            "jobs": job_ids,
        }

    async def measure_object(self, kind: str = "rectangle") -> dict:
        """Measure object dimensions in machine coordinates."""

        state = self._require_calibration()
        points_mm = self._transform_points(state.ref_pts_px or [], state)
        if len(points_mm) < 2:
            raise OrchestratorError("At least two reference points are required")

        xs = [pt[0] for pt in points_mm]
        ys = [pt[1] for pt in points_mm]
        width = max(xs) - min(xs)
        height = max(ys) - min(ys)
        diag = (width**2 + height**2) ** 0.5

        _LOGGER.debug(
            "Measured object kind=%s width=%.3f height=%.3f diag=%.3f",
            kind,
            width,
            height,
            diag,
        )

        return {
            "kind": kind,
            "points_mm": points_mm,
            "metrics": {
                "width_mm": width,
                "height_mm": height,
                "diagonal_mm": diag,
            },
        }

    async def find_edges(self, mode: str = "rect", return_mode: str = "points") -> dict:
        """Return edge definitions from the calibrated reference polygon."""

        state = self._require_calibration()
        points_mm = self._transform_points(state.ref_pts_px or [], state)
        if not points_mm:
            raise OrchestratorError("Reference points are missing")

        if return_mode == "points":
            result = points_mm
        elif return_mode == "segments":
            result = [
                {"start": points_mm[i], "end": points_mm[(i + 1) % len(points_mm)]}
                for i in range(len(points_mm))
            ]
        else:
            raise OrchestratorError(f"Unsupported return mode: {return_mode}")

        _LOGGER.debug(
            "find_edges computed %d elements in mode %s", len(points_mm), return_mode
        )

        return {
            "mode": mode,
            "return_mode": return_mode,
            "edges": result,
        }

    async def _enqueue_line(self, gcode: str) -> str:
        """Submit a G-code line using ``asyncio.to_thread`` for safety."""

        try:
            return await asyncio.to_thread(self._sender.enqueue_line, gcode)
        except SenderStateError as exc:  # pragma: no cover - defensive
            raise OrchestratorError(str(exc)) from exc

    def _require_calibration(self) -> AppState:
        """Ensure calibration data exists before running workflows."""

        state = app_state.read()
        if not state.A_px2cnc or not state.ref_pts_px:
            raise OrchestratorError("Calibration matrix or reference points missing")
        return state

    def _resolve_target_point(self, state: AppState, strategy: str) -> Tuple[float, float]:
        """Return target XY coordinates in machine space based on strategy."""

        points = self._transform_points(state.ref_pts_px or [], state)
        if not points:
            raise OrchestratorError("Reference points missing")

        if strategy == "center":
            x = sum(pt[0] for pt in points) / len(points)
            y = sum(pt[1] for pt in points) / len(points)
        elif strategy == "first":
            x, y = points[0]
        elif strategy == "last":
            x, y = points[-1]
        else:
            raise OrchestratorError(f"Unknown strategy: {strategy}")
        return x, y

    def _transform_points(self, points_px: Iterable[Iterable[float]], state: AppState) -> List[List[float]]:
        """Transform pixel coordinates into machine space."""

        matrix = state.A_px2cnc
        if not matrix:
            raise OrchestratorError("Calibration matrix missing")

        result: List[List[float]] = []
        for point in points_px:
            if len(point) < 2:
                raise OrchestratorError("Invalid pixel coordinate provided")
            x_px, y_px = float(point[0]), float(point[1])
            x_mm, y_mm = self._apply_transform(matrix, x_px, y_px)
            result.append([x_mm, y_mm])
        return result

    @staticmethod
    def _apply_transform(matrix: List[List[float]], x: float, y: float) -> Tuple[float, float]:
        """Apply a homogeneous 3x3 transform to a single point."""

        if len(matrix) < 2 or len(matrix[0]) < 3 or len(matrix[1]) < 3:
            raise OrchestratorError("Calibration matrix must be at least 3x3")

        vec = (x, y, 1.0)
        x_mm = matrix[0][0] * vec[0] + matrix[0][1] * vec[1] + matrix[0][2] * vec[2]
        y_mm = matrix[1][0] * vec[0] + matrix[1][1] * vec[1] + matrix[1][2] * vec[2]
        if len(matrix) >= 3 and len(matrix[2]) >= 3:
            w = matrix[2][0] * vec[0] + matrix[2][1] * vec[1] + matrix[2][2] * vec[2]
            if abs(w) > 1e-9:
                x_mm /= w
                y_mm /= w
        return x_mm, y_mm


__all__ = ["Orchestrator", "OrchestratorError"]
