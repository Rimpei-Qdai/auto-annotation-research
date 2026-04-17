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

    def test_format_clip_timestamp(self):
        self.assertEqual(self.annotator._format_clip_timestamp(3.0), "00:03")
        self.assertEqual(self.annotator._format_clip_timestamp(3.5), "00:03.5")

    def test_parse_structured_video_response(self):
        parsed = self.annotator._parse_structured_video_response(
            "DIRECTION=LEFT\nSPEED_STATE=DECEL\nMANEUVER=LEFT_TURN"
        )
        self.assertEqual(parsed["direction"], "LEFT")
        self.assertEqual(parsed["speed_state"], "DECEL")
        self.assertEqual(parsed["maneuver"], "LEFT_TURN")

    def test_map_structured_video_prediction_turn(self):
        label = self.annotator._map_structured_video_prediction_to_label(
            direction="LEFT",
            speed_state="DECEL",
            maneuver="LEFT_TURN",
        )
        self.assertEqual(label, 6)

    def test_map_structured_video_prediction_speed_event(self):
        label = self.annotator._map_structured_video_prediction_to_label(
            direction="STRAIGHT",
            speed_state="STARTING",
            maneuver="OTHER",
        )
        self.assertEqual(label, 5)

    def test_map_structured_video_prediction_direction_fallback(self):
        label = self.annotator._map_structured_video_prediction_to_label(
            direction="RIGHT",
            speed_state="UNKNOWN",
            maneuver="UNKNOWN",
        )
        self.assertEqual(label, 7)


if __name__ == "__main__":
    unittest.main()
