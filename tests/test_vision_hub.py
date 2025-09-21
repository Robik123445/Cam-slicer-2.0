"""Unit tests for cam_slicer.vision.hub."""

from __future__ import annotations

import math
import unittest

import cv2
import numpy as np

from cam_slicer.vision.hub import (
    Detection,
    PixelPt,
    affine_from_3pts,
    detect_aruco_corners,
    detect_rectangle,
    estimate_delta_affine,
    px_to_mm,
    roi_from_detection,
    update_px2cnc_with_delta,
)


class VisionHubTests(unittest.TestCase):
    """Validate core geometric helpers."""

    def test_affine_from_three_points(self) -> None:
        """Affine from three point pairs should match the known transform."""

        p1_px, p1_mm = (0.0, 0.0), (5.0, 2.0)
        p2_px, p2_mm = (1.0, 0.0), (7.0, 1.0)
        p3_px, p3_mm = (0.0, 1.0), (8.0, 6.0)
        matrix = affine_from_3pts(p1_px, p1_mm, p2_px, p2_mm, p3_px, p3_mm)

        test_vec = np.array([2.0, 3.0, 1.0])
        expected = np.array([2 * test_vec[0] + 3 * test_vec[1] + 5, -test_vec[0] + 4 * test_vec[1] + 2])
        result = matrix @ test_vec
        np.testing.assert_allclose(result, expected, atol=1e-6)

    def test_delta_and_update_pipeline(self) -> None:
        """Delta estimation and update should maintain physical coordinates."""

        ref_pts = np.array([[0.0, 0.0], [100.0, 0.0], [100.0, 50.0], [0.0, 50.0]], dtype=np.float32)
        angle = math.radians(12.0)
        rotation = np.array([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
        translation = np.array([15.0, -4.0])
        curr_pts = ref_pts @ rotation.T + translation

        delta = estimate_delta_affine(ref_pts, curr_pts)
        expected_delta = np.hstack([rotation, translation.reshape(2, 1)])
        np.testing.assert_allclose(delta, expected_delta, atol=1e-2)

        A_old = np.array([[0.02, 0.0, 1.0], [0.0, 0.02, 2.0]])
        A_new = update_px2cnc_with_delta(A_old, delta)
        for ref, curr in zip(ref_pts, curr_pts):
            mm_old = px_to_mm(A_old, float(ref[0]), float(ref[1]))
            mm_new = px_to_mm(A_new, float(curr[0]), float(curr[1]))
            np.testing.assert_allclose(mm_new, mm_old, atol=1e-3)

    def test_roi_from_detection(self) -> None:
        """ROI should expand millimetre bounds according to margin."""

        det = Detection(
            kind="rectangle",
            conf=1.0,
            bbox_px=[5.0, 2.5, 10.0, 5.0],
            angle_deg=0.0,
            corners_px=[
                PixelPt(x=0.0, y=0.0),
                PixelPt(x=10.0, y=0.0),
                PixelPt(x=10.0, y=5.0),
                PixelPt(x=0.0, y=5.0),
            ],
        )
        A = np.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0]])
        roi = roi_from_detection(det, margin_mm=1.0, A_px2cnc=A)
        self.assertEqual(roi, (-1.0, -1.0, 2.0, 1.5))

    def test_detect_rectangle(self) -> None:
        """Synthetic rotated rectangle should be detected."""

        image = np.zeros((200, 200, 3), dtype=np.uint8)
        rect = ((100, 100), (80, 40), 30)
        box = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillConvexPoly(image, box, (255, 255, 255))

        detection = detect_rectangle(image)
        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertGreater(detection.conf, 0.0)

        detected = np.array([[pt.x, pt.y] for pt in detection.corners_px])
        expected = cv2.boxPoints(rect)
        detected_sorted = self._sort_points(detected)
        expected_sorted = self._sort_points(expected)
        np.testing.assert_allclose(detected_sorted, expected_sorted, atol=3.0)

    def test_detect_aruco_marker(self) -> None:
        """Generated ArUco marker should be located with the right ID."""

        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker_id = 7
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, 60)
        image = np.full((120, 120, 3), 255, dtype=np.uint8)
        start = 30
        image[start : start + 60, start : start + 60] = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)

        corners, detected_id = detect_aruco_corners(image, prefer_id=marker_id)
        self.assertIsNotNone(corners)
        self.assertEqual(detected_id, marker_id)
        assert corners is not None
        expected = np.array(
            [
                [start, start],
                [start + 60, start],
                [start + 60, start + 60],
                [start, start + 60],
            ],
            dtype=np.float32,
        )
        sorted_detected = self._sort_points(corners)
        expected_sorted = self._sort_points(expected)
        np.testing.assert_allclose(sorted_detected, expected_sorted, atol=2.0)

    @staticmethod
    def _sort_points(points: np.ndarray) -> np.ndarray:
        sums = points.sum(axis=1)
        diffs = points[:, 0] - points[:, 1]
        tl = points[np.argmin(sums)]
        br = points[np.argmax(sums)]
        tr = points[np.argmin(diffs)]
        bl = points[np.argmax(diffs)]
        return np.vstack([tl, tr, br, bl])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
