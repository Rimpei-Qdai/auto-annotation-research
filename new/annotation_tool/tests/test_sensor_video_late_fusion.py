import unittest
import sys
from pathlib import Path

ANNOTATION_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_TOOL_DIR))

from heron_model_with_trajectory import HeronAnnotatorWithTrajectory


class SensorVideoLateFusionTest(unittest.TestCase):
    def setUp(self):
        self.annotator = HeronAnnotatorWithTrajectory(save_trajectory_frames=False)

    def test_turn_candidates_keep_same_direction_family(self):
        temporal_context = {
            "source": "preferred_window",
            "speed_delta": 0.0,
            "speed_slope_kmh_per_s": 0.0,
            "acc_x_mean": 0.0,
            "gyro_z_integral": -0.52,
            "max_abs_gyro_z": 0.22,
            "heading_delta_deg": -48.0,
            "low_speed_ratio": 0.0,
            "standstill_ratio": 0.0,
        }
        primary_label, _ = self.annotator._classify_sensor_only_label(
            {"speed": 24.0, "acc_x": 0.0, "gyro_z": -0.12, "blinker_r": 1},
            temporal_context=temporal_context,
        )
        scores, debug = self.annotator._build_sensor_candidate_scores(
            {"speed": 24.0, "acc_x": 0.0, "gyro_z": -0.12, "blinker_r": 1},
            temporal_context,
            primary_label,
        )
        self.assertEqual(primary_label, 7)
        self.assertIn(7, scores)
        self.assertIn(9, scores)
        self.assertTrue(any(label in scores for label in (7, 9, 10)))
        self.assertEqual(debug["selected_candidates"][0], 7)

    def test_video_bonus_can_flip_between_sensor_candidates(self):
        combined = self.annotator._combine_sensor_and_video_scores(
            {7: 0.55, 9: 0.42, 10: 0.15},
            video_choice=9,
        )
        best = self.annotator._choose_best_label_from_scores(combined)
        self.assertEqual(best, 9)

    def test_veto_only_graph_uses_macro_candidate_when_available(self):
        self.annotator.graph_verifier.build = lambda *args, **kwargs: {
            "strong_candidate": {"macro_choice": "C", "score": 0.9},
            "top_candidates": [
                {"macro_choice": "C", "score": 0.9},
                {"macro_choice": "A", "score": 0.2},
            ],
        }
        final_label, debug = self.annotator._apply_veto_only_graph(
            final_label=6,
            sensor_data={"speed": 25.0, "acc_x": 0.0, "gyro_z": -0.2},
            trajectory_features={
                "final_x_m": 12.0,
                "final_y_m": -4.0,
                "visible_count": 6,
                "trajectory_points": 30,
            },
            candidate_scores={6: 0.55, 7: 0.31, 9: 0.44},
        )
        self.assertEqual(final_label, 9)
        self.assertTrue(debug["veto_applied"])

    def test_extract_numeric_choice_ignores_out_of_candidate_numbers(self):
        choice = self.annotator._extract_numeric_choice(
            "最終回答: 7",
            [1, 2, 3],
        )
        self.assertIsNone(choice)
        choice = self.annotator._extract_numeric_choice(
            "最終回答: 2",
            [1, 2, 3],
        )
        self.assertEqual(choice, 2)


if __name__ == "__main__":
    unittest.main()
