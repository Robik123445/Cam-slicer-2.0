"""Tests for orchestrator workflows using a dummy sender."""

import asyncio
import unittest

from cam_slicer.core.orchestrator import Orchestrator, OrchestratorError
from cam_slicer.core.state import AppState, app_state


class DummySender:
    """Minimal sender stub capturing queued lines."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def enqueue_line(self, gcode: str) -> str:
        self.lines.append(gcode)
        return f"job-{len(self.lines)}"


class OrchestratorTests(unittest.TestCase):
    """Validate orchestrator behaviour without hardware."""

    def setUp(self) -> None:
        app_state.reset()
        matrix = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
        refs = [[0.0, 0.0], [10.0, 0.0], [10.0, 5.0]]
        app_state.replace(AppState(A_px2cnc=matrix, ref_pts_px=refs, allow_execute_moves=True))
        self.sender = DummySender()
        self.orchestrator = Orchestrator(self.sender)

    def tearDown(self) -> None:
        app_state.reset()

    def test_guide_to_object_executes_plan(self) -> None:
        """Guidance should queue a deterministic three-line approach."""

        result = asyncio.run(self.orchestrator.guide_to_object(kind="rectangle"))
        self.assertTrue(result["executed"])
        self.assertEqual(len(self.sender.lines), 3)
        self.assertEqual(self.sender.lines, result["plan"])

    def test_measure_object_returns_metrics(self) -> None:
        """Measurement should compute width/height from reference polygon."""

        result = asyncio.run(self.orchestrator.measure_object())
        self.assertAlmostEqual(result["metrics"]["width_mm"], 10.0)
        self.assertAlmostEqual(result["metrics"]["height_mm"], 5.0)

    def test_find_edges_segments(self) -> None:
        """Edge detection should support segment output."""

        result = asyncio.run(self.orchestrator.find_edges(return_mode="segments"))
        self.assertEqual(len(result["edges"]), 3)
        self.assertIn("start", result["edges"][0])

    def test_missing_calibration_raises(self) -> None:
        """Missing calibration data must raise an orchestrator error."""

        app_state.reset()
        with self.assertRaises(OrchestratorError):
            asyncio.run(self.orchestrator.measure_object())


if __name__ == "__main__":
    unittest.main()
