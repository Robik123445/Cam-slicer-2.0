"""Tests for the probe planner utilities."""

from __future__ import annotations

import math
import threading
import time
import unittest
import uuid
from typing import Callable, Dict, Optional

from cam_slicer.probe.planner import (
    GridSpec,
    HeightMap,
    ProbeParams,
    Roi,
    bilinear_interp,
    fit_plane,
    probe_grid,
    roi_to_grid,
)


class FakeSender:
    """Minimal stand-in for :class:`SenderService` used in unit tests."""

    def __init__(self, surface: Callable[[float, float], float]) -> None:
        self._event_sink: Optional[Callable[[Dict[str, object]], None]] = None
        self._surface = surface

    def set_event_sink(self, callback: Optional[Callable[[Dict[str, object]], None]]) -> None:
        self._event_sink = callback

    def enqueue_probe_point(
        self, x: float, y: float, z_clear: float, z_probe: float, feed_probe: float
    ) -> str:
        job_id = uuid.uuid4().hex
        sink = self._event_sink
        if sink is None:
            raise RuntimeError("Event sink must be set before probing")

        def _run() -> None:
            sink({"type": "job", "data": {"id": job_id, "event": "started"}})
            time.sleep(0.001)
            z = self._surface(x, y)
            sink({"type": "rx", "data": f"[PRB:{x:.3f},{y:.3f},{z:.3f}:1]"})
            time.sleep(0.001)
            sink({"type": "job", "data": {"id": job_id, "event": "finished"}})

        threading.Thread(target=_run, daemon=True).start()
        return job_id


class ProbePlannerTests(unittest.TestCase):
    """Verify grid generation, probing, and interpolation helpers."""

    def test_roi_to_grid_generates_boustrophedon_order(self) -> None:
        roi = Roi(xmin=0.0, ymin=0.0, xmax=20.0, ymax=10.0)
        spec = GridSpec(step_mm=10.0)
        points, nx, ny, x0, y0, dx, dy = roi_to_grid(roi, spec)

        self.assertEqual(nx, 3)
        self.assertEqual(ny, 2)
        self.assertTrue(math.isclose(dx, 10.0))
        self.assertTrue(math.isclose(dy, 10.0))
        self.assertEqual(
            points,
            [
                (0.0, 0.0),
                (10.0, 0.0),
                (20.0, 0.0),
                (20.0, 10.0),
                (10.0, 10.0),
                (0.0, 10.0),
            ],
        )

    def test_probe_grid_collects_heightmap(self) -> None:
        roi = Roi(xmin=0.0, ymin=0.0, xmax=20.0, ymax=10.0)
        spec = GridSpec(step_mm=10.0)
        params = ProbeParams(z_clear=5.0, z_probe=-5.0, feed_probe=100.0)

        def _surface(x: float, y: float) -> float:
            return 0.5 * x + 0.25 * y

        sender = FakeSender(_surface)
        hm = probe_grid(sender, roi, spec, params)

        self.assertEqual((hm.nx, hm.ny), (3, 2))
        expected = [
            _surface(0.0, 0.0),
            _surface(10.0, 0.0),
            _surface(20.0, 0.0),
            _surface(0.0, 10.0),
            _surface(10.0, 10.0),
            _surface(20.0, 10.0),
        ]
        for measured, expected_value in zip(hm.z, expected):
            self.assertTrue(math.isclose(measured, expected_value, rel_tol=1e-6))

    def test_fit_plane_recovers_coefficients(self) -> None:
        a, b, c = 0.4, -0.2, 1.5
        xs = [0.0, 10.0, 20.0]
        ys = [0.0, 15.0]
        z = [a * x + b * y + c for y in ys for x in xs]
        hm = HeightMap(x0=0.0, y0=0.0, nx=3, ny=2, dx=10.0, dy=15.0, z=z)

        solved = fit_plane(hm)
        self.assertTrue(math.isclose(solved[0], a, abs_tol=1e-9))
        self.assertTrue(math.isclose(solved[1], b, abs_tol=1e-9))
        self.assertTrue(math.isclose(solved[2], c, abs_tol=1e-9))

    def test_bilinear_interp_continuity_and_nan_fallback(self) -> None:
        xs = [0.0, 10.0, 20.0]
        ys = [0.0, 10.0, 20.0]
        z = [x + y for y in ys for x in xs]
        z[0] = float("nan")
        hm = HeightMap(x0=0.0, y0=0.0, nx=3, ny=3, dx=10.0, dy=10.0, z=z)

        interp = bilinear_interp(hm)
        self.assertTrue(math.isclose(interp(20.0, 20.0), 40.0, rel_tol=1e-6))
        self.assertTrue(math.isfinite(interp(10.0, 5.0)))
        self.assertTrue(math.isfinite(interp(10.0, 15.0)))
        self.assertTrue(math.isclose(interp(5.0, 5.0), 10.0, rel_tol=1e-6))
        self.assertTrue(math.isclose(interp(2.0, 2.0), 4.0, rel_tol=1e-6))

    def test_parse_prb_variants(self) -> None:
        from cam_slicer.probe.planner import parse_prb

        fixtures = [
            ("[PRB:0.000,0.000,-5.123:1]", -5.123),
            ("ok", None),
            ("PRB:1.0,2.0,3.0", 3.0),
            ("junk", None),
        ]
        for line, expected in fixtures:
            result = parse_prb(line)
            if expected is None:
                self.assertIsNone(result)
            else:
                self.assertIsNotNone(result)
                self.assertTrue(math.isclose(result, expected, rel_tol=1e-9))


if __name__ == "__main__":  # pragma: no cover - convenience entry point
    unittest.main()
