import unittest
import sys
from pathlib import Path

ANNOTATION_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_TOOL_DIR))

from config import VIDEO_CLIP_DURATION_SECONDS
from heron_model_with_trajectory import HeronAnnotatorWithTrajectory


class SimpleVideoBaselineTest(unittest.TestCase):
    def setUp(self):
        self.annotator = HeronAnnotatorWithTrajectory(save_trajectory_frames=False)

    def test_compute_video_window_bounds_centered(self):
        start, end = self.annotator._compute_video_window_bounds(
            center_time=15.0,
            total_duration=30.0,
            window_duration=VIDEO_CLIP_DURATION_SECONDS,
        )
        self.assertAlmostEqual(start, 12.0)
        self.assertAlmostEqual(end, 18.0)

    def test_compute_video_window_bounds_near_start(self):
        start, end = self.annotator._compute_video_window_bounds(
            center_time=1.0,
            total_duration=30.0,
            window_duration=VIDEO_CLIP_DURATION_SECONDS,
        )
        self.assertAlmostEqual(start, 0.0)
        self.assertAlmostEqual(end, 6.0)

    def test_compute_video_window_bounds_short_video(self):
        start, end = self.annotator._compute_video_window_bounds(
            center_time=2.0,
            total_duration=4.0,
            window_duration=VIDEO_CLIP_DURATION_SECONDS,
        )
        self.assertAlmostEqual(start, 0.0)
        self.assertAlmostEqual(end, 4.0)


if __name__ == "__main__":
    unittest.main()
