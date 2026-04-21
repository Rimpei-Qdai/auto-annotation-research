import unittest
import sys
from pathlib import Path
from PIL import Image

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

    def test_video_plus_summary_content_contains_video_two_images_and_text(self):
        summary_images = [
            Image.new("RGB", (32, 32), color=(255, 255, 255)),
            Image.new("RGB", (32, 32), color=(240, 240, 240)),
        ]
        content = self.annotator._build_video_plus_summary_content(
            clip_path="/tmp/example.mp4",
            summary_images=summary_images,
            prompt_text="候補から選んでください。",
        )
        self.assertEqual([item["type"] for item in content], ["video", "image", "image", "text"])
        self.assertEqual(content[0]["video"], "/tmp/example.mp4")
        self.assertEqual(content[-1]["text"], "候補から選んでください。")

    def test_macro_prompt_uses_all_summary_images(self):
        summary_images = [
            Image.new("RGB", (32, 32), color=(255, 255, 255)),
            Image.new("RGB", (32, 32), color=(240, 240, 240)),
            Image.new("RGB", (32, 32), color=(220, 220, 220)),
        ]
        selected = self.annotator._select_summary_images_for_macro_prompt(summary_images)
        self.assertEqual(len(selected), 3)
        self.assertIs(selected[0], summary_images[0])
        self.assertIs(selected[1], summary_images[1])
        self.assertIs(selected[2], summary_images[2])

    def test_build_trajectory_summary_images_returns_two_images(self):
        summary_images, geometry = self.annotator._build_trajectory_summary_images(
            {
                "speed": 12.0,
                "acc_x": -0.2,
                "gyro_z": 0.18,
            },
            sample_id=None,
        )
        self.assertEqual(len(summary_images), 2)
        self.assertEqual(summary_images[0].size, (720, 720))
        self.assertEqual(summary_images[1].size, (720, 720))
        self.assertIn("visible_count", geometry)
        self.assertIn("trajectory_3d", geometry)

    def test_build_macro_summary_images_returns_three_images(self):
        summary_images, geometry = self.annotator._build_macro_summary_images(
            {
                "speed": 12.0,
                "acc_x": -0.2,
                "gyro_z": 0.18,
                "brake": 1,
                "blinker_l": 1,
            },
            sample_id=None,
        )
        self.assertEqual(len(summary_images), 3)
        self.assertTrue(all(image.size == (720, 720) for image in summary_images))
        self.assertIn("window_points", geometry)
        self.assertIn("trajectory_source", geometry)

    def test_build_sensor_macro_scores_aggregates_fine_candidates(self):
        macro_scores, debug = self.annotator._build_sensor_macro_scores(
            {1: 0.30, 3: 0.20, 6: 0.15, 8: 0.10, 4: 0.05},
            primary_label=3,
        )
        self.assertEqual(debug["primary_macro"], "A")
        self.assertGreater(macro_scores["A"], macro_scores["B"])
        self.assertGreater(macro_scores["B"], macro_scores["D"])

    def test_stop_and_start_are_grouped_into_macro_a(self):
        macro_scores, debug = self.annotator._build_sensor_direct_macro_scores(
            {"speed": 0.8, "acc_x": 0.05, "gyro_z": 0.0, "brake": 1},
            {
                "source": "preferred_window",
                "speed_delta": -1.0,
                "speed_slope_kmh_per_s": -0.2,
                "acc_x_mean": -0.05,
                "gyro_z_integral": 0.0,
                "max_abs_gyro_z": 0.01,
                "heading_delta_deg": 0.0,
                "low_speed_ratio": 0.9,
                "standstill_ratio": 0.9,
            },
        )
        self.assertEqual(debug["primary_macro"], "A")
        self.assertGreater(macro_scores["A"], macro_scores["D"])

    def test_direct_macro_scores_prefer_right_turn_family(self):
        macro_scores, debug = self.annotator._build_sensor_direct_macro_scores(
            {"speed": 24.0, "acc_x": 0.0, "gyro_z": -0.14, "blinker_r": 1},
            {
                "source": "preferred_window",
                "speed_delta": 0.0,
                "speed_slope_kmh_per_s": 0.0,
                "acc_x_mean": 0.0,
                "gyro_z_integral": -0.48,
                "max_abs_gyro_z": 0.20,
                "heading_delta_deg": -42.0,
                "low_speed_ratio": 0.0,
                "standstill_ratio": 0.0,
            },
        )
        self.assertEqual(debug["primary_macro"], "C")
        self.assertGreater(macro_scores["C"], macro_scores["A"])

    def test_combine_sensor_and_video_macro_scores_can_flip_macro(self):
        combined = self.annotator._combine_sensor_and_video_macro_scores(
            {"A": 0.55, "B": 0.42, "C": 0.10, "D": 0.05},
            video_choice="B",
        )
        best = self.annotator._choose_best_macro_from_scores(combined)
        self.assertEqual(best, "B")

    def test_apply_veto_only_graph_macro_changes_macro(self):
        self.annotator.graph_verifier.build = lambda *args, **kwargs: {
            "strong_candidate": {"macro_choice": "C", "score": 0.9},
            "top_candidates": [
                {"macro_choice": "C", "score": 0.9},
                {"macro_choice": "A", "score": 0.2},
            ],
        }
        final_macro, debug = self.annotator._apply_veto_only_graph_macro(
            final_macro="B",
            sensor_data={"speed": 25.0, "acc_x": 0.0, "gyro_z": -0.2},
            trajectory_features={
                "final_x_m": 12.0,
                "final_y_m": -4.0,
                "visible_count": 6,
                "trajectory_points": 30,
            },
            candidate_scores={"A": 0.12, "B": 0.55, "C": 0.44, "D": 0.01},
        )
        self.assertEqual(final_macro, "C")
        self.assertTrue(debug["veto_applied"])

    def test_precomputed_sensor_window_context_is_preferred_when_available(self):
        self.annotator._precomputed_sensor_windows = {
            42: {
                "sample_id": 42,
                "center_timestamp": 3000,
                "quality": "dense_window",
                "runtime_source": "preferred_window",
                "temporal_reliable": True,
                "center_row_offset_ms": 0,
                "max_gap_ms": 1000,
                "points": [
                    {"timestamp": 0, "speed": 10.0, "acc_x": 0.0, "gyro_z": 0.01, "heading": 0.0},
                    {"timestamp": 1000, "speed": 11.0, "acc_x": 0.1, "gyro_z": 0.02, "heading": 5.0},
                    {"timestamp": 2000, "speed": 12.0, "acc_x": 0.1, "gyro_z": 0.03, "heading": 10.0},
                    {"timestamp": 3000, "speed": 13.0, "acc_x": 0.1, "gyro_z": 0.04, "heading": 15.0},
                    {"timestamp": 4000, "speed": 14.0, "acc_x": 0.2, "gyro_z": 0.05, "heading": 20.0},
                    {"timestamp": 5000, "speed": 15.0, "acc_x": 0.2, "gyro_z": 0.06, "heading": 25.0},
                    {"timestamp": 6000, "speed": 16.0, "acc_x": 0.2, "gyro_z": 0.07, "heading": 30.0},
                ],
            }
        }
        self.annotator._sensor_temporal_lookup = {}

        context = self.annotator._build_sensor_temporal_context(
            42,
            {"speed": 13.0, "acc_x": 0.1, "gyro_z": 0.04},
        )

        self.assertEqual(context["source"], "preferred_window")
        self.assertEqual(context["window_quality"], "dense_window")
        self.assertTrue(context["temporal_reliable"])
        self.assertEqual(context["pre_count"], 3)
        self.assertEqual(context["post_count"], 3)
        self.assertGreater(context["speed_delta"], 0.0)
        self.assertEqual(context["temporal_center_offset_ms"], 0)

    def test_precomputed_sensor_window_geometry_is_used_for_macro_summaries(self):
        self.annotator._precomputed_sensor_windows = {
            42: {
                "sample_id": 42,
                "center_timestamp": 3000,
                "quality": "dense_window",
                "runtime_source": "preferred_window",
                "temporal_reliable": True,
                "center_row_offset_ms": 0,
                "max_gap_ms": 100,
                "points": [
                    {"timestamp": 0, "speed": 10.0, "acc_x": 0.0, "gyro_z": 0.00, "heading": 0.0, "brake": 0, "blinker_l": 0, "blinker_r": 0},
                    {"timestamp": 1000, "speed": 11.0, "acc_x": 0.1, "gyro_z": 0.02, "heading": 5.0, "brake": 0, "blinker_l": 0, "blinker_r": 0},
                    {"timestamp": 2000, "speed": 12.0, "acc_x": 0.1, "gyro_z": 0.04, "heading": 12.0, "brake": 0, "blinker_l": 0, "blinker_r": 0},
                    {"timestamp": 3000, "speed": 13.0, "acc_x": 0.1, "gyro_z": 0.05, "heading": 20.0, "brake": 0, "blinker_l": 0, "blinker_r": 1},
                    {"timestamp": 4000, "speed": 14.0, "acc_x": 0.2, "gyro_z": 0.06, "heading": 30.0, "brake": 0, "blinker_l": 0, "blinker_r": 1},
                    {"timestamp": 5000, "speed": 15.0, "acc_x": 0.2, "gyro_z": 0.07, "heading": 42.0, "brake": 0, "blinker_l": 0, "blinker_r": 1},
                    {"timestamp": 6000, "speed": 16.0, "acc_x": 0.2, "gyro_z": 0.08, "heading": 58.0, "brake": 0, "blinker_l": 0, "blinker_r": 1},
                ],
            }
        }

        summary_images, geometry = self.annotator._build_macro_summary_images(
            {
                "speed": 13.0,
                "acc_x": 0.1,
                "gyro_z": 0.05,
                "blinker_r": 1,
            },
            sample_id=42,
        )

        self.assertEqual(len(summary_images), 3)
        self.assertEqual(geometry["trajectory_source"], "precomputed_window")
        self.assertEqual(geometry["window_quality"], "dense_window")
        self.assertGreater(len(geometry["future_trajectory_3d"]), 2)


if __name__ == "__main__":
    unittest.main()
