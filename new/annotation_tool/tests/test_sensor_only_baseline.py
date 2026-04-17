import unittest
import sys
from pathlib import Path

ANNOTATION_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_TOOL_DIR))

from heron_model_with_trajectory import HeronAnnotatorWithTrajectory


class SensorOnlyBaselineTest(unittest.TestCase):
    def setUp(self):
        self.annotator = HeronAnnotatorWithTrajectory(save_trajectory_frames=False)

    def test_standstill_stop(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 0.4,
            "acc_x": -0.02,
            "gyro_z": 0.0,
        })
        self.assertEqual(label, 4)
        self.assertEqual(reason["rule"], "standstill_stop")

    def test_standstill_start(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 0.8,
            "acc_x": 0.28,
            "gyro_z": 0.0,
        })
        self.assertEqual(label, 5)
        self.assertEqual(reason["rule"], "standstill_start")

    def test_left_turn_from_strong_yaw(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 22.0,
            "acc_x": 0.01,
            "gyro_z": 0.22,
        })
        self.assertEqual(label, 6)
        self.assertEqual(reason["rule"], "strong_yaw_turn")

    def test_right_lane_change_from_moderate_yaw(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 30.0,
            "acc_x": 0.00,
            "gyro_z": -0.10,
        })
        self.assertEqual(label, 9)
        self.assertEqual(reason["rule"], "moderate_yaw_lane_change")

    def test_straight_acceleration(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 28.0,
            "acc_x": 0.31,
            "gyro_z": 0.01,
        })
        self.assertEqual(label, 2)
        self.assertEqual(reason["rule"], "straight_acceleration")

    def test_straight_constant_speed(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 24.0,
            "acc_x": 0.02,
            "gyro_z": 0.01,
        })
        self.assertEqual(label, 1)
        self.assertEqual(reason["rule"], "steady_constant_speed")


if __name__ == "__main__":
    unittest.main()
