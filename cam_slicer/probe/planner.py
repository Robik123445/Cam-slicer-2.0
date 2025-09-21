"""Probe planning helpers for building surface height maps."""

from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:  # pragma: no cover - optional import for Pydantic v2
    from pydantic import BaseModel, ConfigDict, field_validator, model_validator
except ImportError:  # pragma: no cover - fallback for Pydantic v1
    from pydantic import BaseModel, root_validator, validator

    ConfigDict = None  # type: ignore[assignment]

    def field_validator(*fields: str, **kwargs):  # type: ignore[misc]
        """Compatibility shim mapping ``field_validator`` to ``validator``."""

        kwargs.setdefault("allow_reuse", True)
        return validator(*fields, **kwargs)

    def model_validator(*, mode: str, **kwargs):  # type: ignore[misc]
        """Compatibility shim mapping ``model_validator`` to ``root_validator``."""

        kwargs.setdefault("allow_reuse", True)
        pre = mode == "before"
        return root_validator(pre=pre, **kwargs)


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
    except OSError:  # pragma: no cover - best effort logging setup
        _LOGGER.addHandler(logging.NullHandler())
else:  # pragma: no cover - logger configured by application
    _LOGGER.addHandler(logging.NullHandler())


class Roi(BaseModel):
    """Axis-aligned region of interest in machine coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    if ConfigDict is not None:  # pragma: no cover - executed under pydantic v2
        model_config = ConfigDict(validate_assignment=True)
    else:  # pragma: no cover - executed under pydantic v1
        class Config:
            """Pydantic configuration ensuring validation on assignment."""

            validate_assignment = True

    @model_validator(mode="after")
    def _check_bounds(cls, values: "Roi") -> "Roi":
        """Validate ROI bounds are well-formed."""

        if isinstance(values, dict):  # pragma: no cover - pydantic v1 compatibility
            xmin = float(values.get("xmin", 0.0))
            xmax = float(values.get("xmax", 0.0))
            ymin = float(values.get("ymin", 0.0))
            ymax = float(values.get("ymax", 0.0))
        else:
            xmin = float(values.xmin)
            xmax = float(values.xmax)
            ymin = float(values.ymin)
            ymax = float(values.ymax)
        if xmax < xmin or ymax < ymin:
            raise ValueError("ROI bounds must satisfy xmin <= xmax and ymin <= ymax")
        return values


class GridSpec(BaseModel):
    """Grid discretisation parameters."""

    step_mm: float = 10.0

    if ConfigDict is not None:  # pragma: no cover - executed under pydantic v2
        model_config = ConfigDict(validate_assignment=True)
    else:  # pragma: no cover - executed under pydantic v1
        class Config:
            """Pydantic configuration enforcing runtime validation."""

            validate_assignment = True

    @field_validator("step_mm")
    def _validate_step(cls, value: float) -> float:
        """Ensure that the grid step is positive."""

        step = float(value)
        if step <= 0:
            raise ValueError("step_mm must be positive")
        return step


class ProbeParams(BaseModel):
    """Parameters controlling the probing motion."""

    z_clear: float = 5.0
    z_probe: float = -8.0
    feed_probe: float = 120.0

    if ConfigDict is not None:  # pragma: no cover - executed under pydantic v2
        model_config = ConfigDict(validate_assignment=True)
    else:  # pragma: no cover - executed under pydantic v1
        class Config:
            """Pydantic configuration enforcing runtime validation."""

            validate_assignment = True

    @field_validator("feed_probe")
    def _validate_feed(cls, value: float) -> float:
        """Enforce a positive probing feed rate."""

        feed = float(value)
        if feed <= 0:
            raise ValueError("feed_probe must be positive")
        return feed


class HeightMap(BaseModel):
    """Structured representation of a probed surface."""

    x0: float
    y0: float
    nx: int
    ny: int
    dx: float
    dy: float
    z: List[float]

    if ConfigDict is not None:  # pragma: no cover - executed under pydantic v2
        model_config = ConfigDict(validate_assignment=True)
    else:  # pragma: no cover - executed under pydantic v1
        class Config:
            """Pydantic configuration enforcing runtime validation."""

            validate_assignment = True

    @field_validator("nx", "ny")
    def _validate_counts(cls, value: int) -> int:
        """Ensure grid counts are strictly positive."""

        count = int(value)
        if count <= 0:
            raise ValueError("Grid dimensions must be positive")
        return count

    @field_validator("dx", "dy")
    def _validate_steps(cls, value: float) -> float:
        """Ensure grid spacing is non-negative."""

        step = float(value)
        if step < 0:
            raise ValueError("Grid spacing must be non-negative")
        return step

    @model_validator(mode="after")
    def _validate_payload(cls, values: "HeightMap") -> "HeightMap":
        """Verify that the flattened Z buffer matches the grid size."""

        if isinstance(values, dict):  # pragma: no cover - pydantic v1 compatibility
            nx = int(values.get("nx", 0))
            ny = int(values.get("ny", 0))
            z = list(values.get("z", []))
        else:
            nx = int(values.nx)
            ny = int(values.ny)
            z = list(values.z)
        expected = nx * ny
        if len(z) != expected:
            raise ValueError(f"HeightMap.z must contain {expected} entries")
        return values


def parse_prb(line: str) -> Optional[float]:
    """Parse a GRBL ``PRB`` response and return the measured Z value."""

    if "PRB:" not in line:
        return None
    start = line.find("PRB:") + 4
    end = line.find("]", start)
    if end == -1:
        end = len(line)
    payload = line[start:end]
    coords = payload.split(":", 1)[0]
    parts = [segment for segment in coords.split(",") if segment]
    if len(parts) < 3:
        return None
    try:
        return float(parts[2])
    except ValueError:
        return None


def roi_to_grid(
    roi: Roi, spec: GridSpec
) -> tuple[list[tuple[float, float]], int, int, float, float, float, float]:
    """Generate a boustrophedon raster covering ``roi`` with ``spec`` spacing."""

    width = float(roi.xmax) - float(roi.xmin)
    height = float(roi.ymax) - float(roi.ymin)

    if width < 0 or height < 0:  # Defensive guard; Pydantic already checks this
        raise ValueError("ROI bounds are invalid")

    # Počet intervalov zvolíme tak, aby krok neprekročil požadovaný ``step_mm``.
    intervals_x = 1 if width <= 0 else max(1, math.ceil(width / spec.step_mm))
    intervals_y = 1 if height <= 0 else max(1, math.ceil(height / spec.step_mm))

    nx = 1 if width <= 0 else intervals_x + 1
    ny = 1 if height <= 0 else intervals_y + 1

    dx = 0.0 if nx <= 1 else width / (nx - 1)
    dy = 0.0 if ny <= 1 else height / (ny - 1)

    x_coords = [float(roi.xmin) + dx * i for i in range(nx)] or [float(roi.xmin)]
    y_coords = [float(roi.ymin) + dy * j for j in range(ny)] or [float(roi.ymin)]
    if x_coords:
        x_coords[-1] = float(roi.xmax)
    if y_coords:
        y_coords[-1] = float(roi.ymax)

    points: list[tuple[float, float]] = []
    for j, y in enumerate(y_coords):
        row: list[tuple[float, float]] = []
        for x in x_coords:
            row.append((x, y))
        # Boustrophedonický priebeh – každý druhý riadok obrátime.
        if j % 2 == 1:
            row.reverse()
        points.extend(row)

    return points, nx, ny, x_coords[0], y_coords[0], dx, dy


class _ProbeCollector:
    """Collect ``PRB`` readings and job states from :class:`SenderService`."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._active_job: Optional[str] = None
        self._latest_prb: Optional[float] = None
        self._results: Dict[str, Dict[str, object]] = {}

    def callback(self, payload: Dict[str, object]) -> None:
        """Handle sender events and store probe outcomes."""

        event_type = payload.get("type")
        if event_type == "rx":
            line = str(payload.get("data", ""))
            value = parse_prb(line)
            if value is None:
                return
            with self._condition:
                if self._active_job is not None:
                    self._latest_prb = value
        elif event_type == "job":
            data = payload.get("data", {})
            if not isinstance(data, dict):
                return
            job_id = str(data.get("id", ""))
            event = str(data.get("event", ""))
            with self._condition:
                if event == "started":
                    self._active_job = job_id
                    self._latest_prb = None
                elif event in {"finished", "error"}:
                    self._results[job_id] = {
                        "status": event,
                        "z": self._latest_prb,
                        "message": data.get("message"),
                    }
                    if self._active_job == job_id:
                        self._active_job = None
                        self._latest_prb = None
                    self._condition.notify_all()

    def wait_for_job(self, job_id: str, timeout: float) -> Dict[str, object]:
        """Block until ``job_id`` finishes or ``timeout`` expires."""

        deadline = time.monotonic() + timeout
        with self._condition:
            while job_id not in self._results:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Probe job {job_id} timed out")
                self._condition.wait(timeout=remaining)
            return self._results[job_id]


def probe_grid(sender: "SenderService", roi: Roi, spec: GridSpec, params: ProbeParams) -> HeightMap:
    """Probe ``roi`` using ``sender`` and return a dense :class:`HeightMap`."""

    points, nx, ny, x0, y0, dx, dy = roi_to_grid(roi, spec)
    collector = _ProbeCollector()
    previous_sink: Optional[Callable[[Dict[str, object]], None]] = getattr(
        sender, "_event_sink", None
    )

    def _composite_callback(payload: Dict[str, object]) -> None:
        collector.callback(payload)
        if previous_sink is not None:
            try:
                previous_sink(payload)
            except Exception:  # pragma: no cover - defensive logging
                _LOGGER.exception("Event sink raised during probe capture")

    sender.set_event_sink(_composite_callback)

    try:
        z_boustro: list[float] = []
        for index, (x, y) in enumerate(points):
            job_id = sender.enqueue_probe_point(x, y, params.z_clear, params.z_probe, params.feed_probe)
            try:
                result = collector.wait_for_job(job_id, timeout=30.0)
            except TimeoutError:
                _LOGGER.error("Probe job %s timed out at point #%d", job_id, index)
                z_boustro.append(float("nan"))
                continue

            status = result.get("status")
            measured = result.get("z")
            message = result.get("message")
            if status != "finished":
                _LOGGER.warning(
                    "Probe job %s ended with status %s (message=%s)",
                    job_id,
                    status,
                    message,
                )
            z_value = float(measured) if isinstance(measured, (int, float)) else float("nan")
            z_boustro.append(z_value)

        # Preložme hodnoty do riadkovo-majoritnej podoby očakávanej v ``HeightMap``.
        z_row_major = [float("nan")] * (nx * ny)
        cursor = 0
        for row in range(ny):
            columns = range(nx)
            if row % 2 == 1:
                columns = reversed(range(nx))
            for col in columns:
                if cursor >= len(z_boustro):
                    break
                z_row_major[row * nx + col] = z_boustro[cursor]
                cursor += 1

        return HeightMap(x0=x0, y0=y0, nx=nx, ny=ny, dx=dx, dy=dy, z=z_row_major)
    finally:
        sender.set_event_sink(previous_sink)  # type: ignore[arg-type]


def fit_plane(hm: HeightMap) -> tuple[float, float, float]:
    """Fit ``Z = a*x + b*y + c`` over the valid samples of ``hm``."""

    xs = np.linspace(hm.x0, hm.x0 + hm.dx * max(hm.nx - 1, 0), hm.nx)
    ys = np.linspace(hm.y0, hm.y0 + hm.dy * max(hm.ny - 1, 0), hm.ny)
    grid_z = np.asarray(hm.z, dtype=float).reshape(hm.ny, hm.nx)

    samples: list[tuple[float, float, float]] = []
    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            z_value = grid_z[j, i]
            if np.isnan(z_value):
                continue
            samples.append((x, y, z_value))

    if not samples:
        return 0.0, 0.0, 0.0

    A = np.array([[x, y, 1.0] for x, y, _ in samples], dtype=float)
    b = np.array([z for _, _, z in samples], dtype=float)
    coeffs, *_ = np.linalg.lstsq(A, b, rcond=None)
    a, b_coeff, c = coeffs.tolist()
    return float(a), float(b_coeff), float(c)


def bilinear_interp(hm: HeightMap) -> Callable[[float, float], float]:
    """Return a callable performing bilinear interpolation over ``hm``."""

    xs = np.linspace(hm.x0, hm.x0 + hm.dx * max(hm.nx - 1, 0), hm.nx)
    ys = np.linspace(hm.y0, hm.y0 + hm.dy * max(hm.ny - 1, 0), hm.ny)
    grid_z = np.asarray(hm.z, dtype=float).reshape(hm.ny, hm.nx)
    plane_a, plane_b, plane_c = fit_plane(hm)

    def _plane(x: float, y: float) -> float:
        return plane_a * x + plane_b * y + plane_c

    def _clamp_index(value: float, axis: np.ndarray) -> int:
        idx = int(np.searchsorted(axis, value, side="right") - 1)
        return max(0, min(idx, len(axis) - 2))

    def _interp(x: float, y: float) -> float:
        """Interpolated surface height in ``(x, y)``."""

        if hm.nx == 1 and hm.ny == 1:
            value = grid_z[0, 0]
            return float(value) if not np.isnan(value) else _plane(x, y)

        if hm.nx == 1:
            j = _clamp_index(y, ys) if hm.ny > 1 else 0
            y0 = ys[j]
            y1 = ys[j + 1] if hm.ny > 1 else ys[j]
            z0 = grid_z[j, 0]
            z1 = grid_z[j + 1, 0] if hm.ny > 1 else grid_z[j, 0]
            if np.isnan(z0) or np.isnan(z1) or y1 == y0:
                return _plane(x, y)
            t = 0.0 if y1 == y0 else (y - y0) / (y1 - y0)
            return float((1 - t) * z0 + t * z1)

        if hm.ny == 1:
            i = _clamp_index(x, xs) if hm.nx > 1 else 0
            x0 = xs[i]
            x1 = xs[i + 1] if hm.nx > 1 else xs[i]
            z0 = grid_z[0, i]
            z1 = grid_z[0, i + 1] if hm.nx > 1 else grid_z[0, i]
            if np.isnan(z0) or np.isnan(z1) or x1 == x0:
                return _plane(x, y)
            t = 0.0 if x1 == x0 else (x - x0) / (x1 - x0)
            return float((1 - t) * z0 + t * z1)

        i = _clamp_index(x, xs)
        j = _clamp_index(y, ys)
        x0, x1 = xs[i], xs[i + 1]
        y0, y1 = ys[j], ys[j + 1]
        q11 = grid_z[j, i]
        q21 = grid_z[j, i + 1]
        q12 = grid_z[j + 1, i]
        q22 = grid_z[j + 1, i + 1]
        if (
            np.isnan(q11)
            or np.isnan(q21)
            or np.isnan(q12)
            or np.isnan(q22)
            or x1 == x0
            or y1 == y0
        ):
            return _plane(x, y)
        tx = (x - x0) / (x1 - x0)
        ty = (y - y0) / (y1 - y0)
        return float(
            q11 * (1 - tx) * (1 - ty)
            + q21 * tx * (1 - ty)
            + q12 * (1 - tx) * ty
            + q22 * tx * ty
        )

    return _interp


__all__ = [
    "Roi",
    "GridSpec",
    "ProbeParams",
    "HeightMap",
    "parse_prb",
    "roi_to_grid",
    "probe_grid",
    "fit_plane",
    "bilinear_interp",
]
