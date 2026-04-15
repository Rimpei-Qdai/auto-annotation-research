import sys
import unittest
from pathlib import Path


ANNOTATION_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_TOOL_DIR))

from driving_graph import MacroGraphVerifier  # noqa: E402


class MacroGraphVerifierTests(unittest.TestCase):
    def setUp(self):
        self.verifier = MacroGraphVerifier()

    def test_strong_right_turn_can_override_stage1_a(self):
        result = self.verifier.build(
            {
                "speed": 22.0,
                "acc_x": -0.1,
                "gyro_z": -0.22,
            },
            {
                "final_x_m": 12.0,
                "final_y_m": -5.5,
                "visible_count": 24,
                "trajectory_points": 31,
            },
            stage1_choice="A",
            stage2_choice=None,
        )

        self.assertIsNotNone(result["strong_candidate"])
        self.assertEqual(result["strong_candidate"]["macro_choice"], "C")

    def test_stop_like_geometry_prefers_other(self):
        result = self.verifier.build(
            {
                "speed": 1.5,
                "acc_x": -0.2,
                "gyro_z": 0.0,
            },
            {
                "final_x_m": 1.2,
                "final_y_m": 0.0,
                "visible_count": 0,
                "trajectory_points": 31,
            },
            stage1_choice="A",
            stage2_choice=None,
        )

        self.assertIsNotNone(result["strong_candidate"])
        self.assertEqual(result["strong_candidate"]["macro_choice"], "D")

    def test_right_geometry_can_counter_left_stage2(self):
        result = self.verifier.build(
            {
                "speed": 18.0,
                "acc_x": 0.0,
                "gyro_z": -0.18,
            },
            {
                "final_x_m": 10.0,
                "final_y_m": -4.5,
                "visible_count": 20,
                "trajectory_points": 31,
            },
            stage1_choice="N",
            stage2_choice="B",
        )

        self.assertIsNotNone(result["strong_candidate"])
        self.assertEqual(result["strong_candidate"]["macro_choice"], "C")


if __name__ == "__main__":
    unittest.main()
