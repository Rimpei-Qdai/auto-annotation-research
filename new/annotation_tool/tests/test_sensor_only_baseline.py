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

    def test_temporal_standstill_stop(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 0.4,
            "acc_x": -0.02,
            "gyro_z": 0.0,
            "brake": 1,
        }, temporal_context={
            "source": "preferred_window",
            "standstill_ratio": 1.0,
            "low_speed_ratio": 1.0,
            "speed_delta": -0.5,
        })
        self.assertEqual(label, 4)
        self.assertEqual(reason["rule"], "temporal_standstill_stop")

    def test_temporal_standstill_start(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 0.8,
            "acc_x": 0.10,
            "gyro_z": 0.0,
            "brake": 0,
        }, temporal_context={
            "source": "preferred_window",
            "standstill_ratio": 0.6,
            "low_speed_ratio": 1.0,
            "speed_delta": 8.0,
            "speed_slope_kmh_per_s": 1.8,
        })
        self.assertEqual(label, 5)
        self.assertEqual(reason["rule"], "temporal_standstill_start")

    def test_left_turn_from_temporal_yaw(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 22.0,
            "acc_x": 0.01,
            "gyro_z": 0.08,
        }, temporal_context={
            "source": "preferred_window",
            "gyro_z_integral": 0.62,
            "heading_delta_deg": 42.0,
            "max_abs_gyro_z": 0.20,
        })
        self.assertEqual(label, 6)
        self.assertEqual(reason["rule"], "temporal_turn")

    def test_right_lane_change_from_blinker_and_moderate_yaw(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 30.0,
            "acc_x": 0.00,
            "gyro_z": -0.06,
            "blinker_r": 1,
        }, temporal_context={
            "source": "preferred_window",
            "gyro_z_integral": -0.18,
            "heading_delta_deg": -9.0,
            "max_abs_gyro_z": 0.08,
        })
        self.assertEqual(label, 9)
        self.assertEqual(reason["rule"], "temporal_lane_change_right")

    def test_temporal_acceleration(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 28.0,
            "acc_x": 0.10,
            "gyro_z": 0.01,
        }, temporal_context={
            "source": "preferred_window",
            "speed_delta": 9.0,
            "speed_slope_kmh_per_s": 1.5,
            "acc_x_mean": 0.24,
            "gyro_z_integral": 0.0,
            "max_abs_gyro_z": 0.02,
        })
        self.assertEqual(label, 2)
        self.assertEqual(reason["rule"], "temporal_acceleration")

    def test_temporal_constant_speed(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 24.0,
            "acc_x": -0.02,
            "gyro_z": 0.01,
        }, temporal_context={
            "source": "preferred_window",
            "speed_delta": 1.0,
            "speed_slope_kmh_per_s": 0.1,
            "acc_x_mean": -0.01,
            "gyro_z_integral": 0.03,
            "max_abs_gyro_z": 0.03,
        })
        self.assertEqual(label, 1)
        self.assertEqual(reason["rule"], "temporal_constant_speed")

    def test_temporal_deceleration_overrides_noisy_snapshot(self):
        label, reason = self.annotator._classify_sensor_only_label({
            "speed": 35.0,
            "acc_x": 0.05,
            "gyro_z": 0.01,
            "brake": 1,
        }, temporal_context={
            "source": "preferred_window",
            "speed_delta": -12.0,
            "speed_slope_kmh_per_s": -2.0,
            "acc_x_mean": -0.28,
            "gyro_z_integral": 0.0,
            "max_abs_gyro_z": 0.02,
            "low_speed_ratio": 0.0,
        })
        self.assertEqual(label, 3)
        self.assertEqual(reason["rule"], "temporal_deceleration")


if __name__ == "__main__":
    unittest.main()
