import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ANNOTATION_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_TOOL_DIR))

from heron_model_with_trajectory import HeronAnnotatorWithTrajectory  # noqa: E402


def make_frames():
    return [Image.new("RGB", (128, 72), color="white") for _ in range(4)]


class TrajectorySummaryTests(unittest.TestCase):
    def setUp(self):
        self.annotator = HeronAnnotatorWithTrajectory(save_trajectory_frames=False)

    def test_prepare_visual_inputs_returns_four_frames_plus_two_summaries(self):
        frames = make_frames()
        model_frames, geometry = self.annotator.prepare_visual_inputs(
            frames,
            {
                "speed": 24.0,
                "acc_x": 0.0,
                "acc_y": 0.0,
                "acc_z": 0.0,
                "gyro_z": 0.12,
            },
            sample_id=None,
        )

        self.assertEqual(len(model_frames), 6)
        self.assertEqual(model_frames[4].size, (720, 720))
        self.assertEqual(model_frames[5].size, (720, 720))
        self.assertIn("visible_count", geometry)

    def test_normalized_summary_changes_with_turn_sign(self):
        left_traj = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.5, 0.0],
                [6.0, 1.5, 0.0],
                [9.0, 3.0, 0.0],
            ],
            dtype=np.float32,
        )
        right_traj = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, -0.5, 0.0],
                [6.0, -1.5, 0.0],
                [9.0, -3.0, 0.0],
            ],
            dtype=np.float32,
        )

        left_img = np.array(self.annotator._render_normalized_summary(left_traj))
        right_img = np.array(self.annotator._render_normalized_summary(right_traj))

        self.assertFalse(np.array_equal(left_img, right_img))


if __name__ == "__main__":
    unittest.main()
