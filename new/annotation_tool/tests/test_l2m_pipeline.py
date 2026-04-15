import json
import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ANNOTATION_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_TOOL_DIR))

from heron_l2m_pipeline import L2MCoTPipeline  # noqa: E402
from driving_graph import DrivingConceptGraphBuilder  # noqa: E402
from visual_prompting import TrajectoryVisualizer  # noqa: E402


class FakeFramePromptGenerator:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate_text(
        self,
        frames,
        prompt,
        *,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
        pad_token_id=None,
        eos_token_id=None,
    ):
        self.calls.append(
            {
                "frame_count": len(frames),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            }
        )
        output = self.outputs.pop(0)
        if isinstance(output, Exception):
            raise output
        return output


def make_frames():
    return [Image.new("RGB", (32, 32), color="white") for _ in range(4)]


class L2MCoTPipelineTests(unittest.TestCase):
    def test_pipeline_runs_with_fake_generator_only(self):
        generator = FakeFramePromptGenerator(
            [
                json.dumps(
                    {
                        "reasoning": "交差点が見え、右方向への変化がある",
                        "road_shape": "STRAIGHT",
                        "trajectory_relation": "CROSSING_RIGHT",
                        "slope": "FLAT",
                        "intersection_detected": "YES",
                        "direction_change": "TURNING",
                        "visual_shift": "SHIFT_RIGHT",
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "右旋回と減速が一致している",
                        "acceleration_cause": "DRIVER_BRAKE",
                        "speed_trend": "DECREASING",
                        "consistency_check": "CONSISTENT",
                    }
                ),
                json.dumps(
                    {
                        "final_reasoning": "交差点進入、右方向変化、右ウィンカーから右折と判断",
                        "category_id": 7,
                        "category_name": "右折",
                        "confidence": 0.91,
                    }
                ),
            ]
        )

        pipeline = L2MCoTPipeline(generator)
        result = pipeline.analyze_with_l2m(
            make_frames(),
            {
                "speed": 18.0,
                "speed_diff": -1.2,
                "acc_x": -0.4,
                "acc_y": 0.0,
                "gyro_z": -0.16,
                "brake": 1,
                "blinker_r": 1,
                "blinker_l": 0,
            },
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["final_category"], 7)
        self.assertEqual(result["level3"]["category_name"], "右折")
        self.assertEqual(len(generator.calls), 3)
        self.assertTrue(all(call["frame_count"] == 4 for call in generator.calls))

    def test_pipeline_failure_is_detected_without_gpu_runtime(self):
        generator = FakeFramePromptGenerator(
            [
                RuntimeError("Expected all tensors to be on the same device"),
                RuntimeError("Expected all tensors to be on the same device"),
                RuntimeError("Expected all tensors to be on the same device"),
            ]
        )

        pipeline = L2MCoTPipeline(generator)
        result = pipeline.analyze_with_l2m(
            make_frames(),
            {
                "speed": 12.0,
                "speed_diff": 0.0,
                "acc_x": 0.0,
                "acc_y": 0.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
            },
        )

        self.assertEqual(result["status"], "failed")
        self.assertIsNone(result["final_category"])
        self.assertIn("Expected all tensors", result["error"])

    def test_level2_result_is_normalized_by_sensor_motion(self):
        generator = FakeFramePromptGenerator(
            [
                json.dumps(
                    {
                        "reasoning": "直進に見える",
                        "road_shape": "STRAIGHT",
                        "trajectory_relation": "PARALLEL",
                        "slope": "FLAT",
                        "intersection_detected": "NO",
                        "direction_change": "STRAIGHT",
                        "visual_shift": "NO_SHIFT",
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "加速寄りに見える",
                        "acceleration_cause": "DRIVER_ACCEL",
                        "speed_trend": "STABLE",
                        "consistency_check": "CONSISTENT",
                    }
                ),
                json.dumps(
                    {
                        "final_reasoning": "ブレーキと減速が強いので減速",
                        "category_id": 3,
                        "category_name": "減速",
                        "confidence": 0.82,
                    }
                ),
            ]
        )

        pipeline = L2MCoTPipeline(generator)
        result = pipeline.analyze_with_l2m(
            make_frames(),
            {
                "speed": 18.0,
                "speed_diff": -7.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": -7.0,
                "acc_x": -0.4,
                "acc_y": 0.0,
                "gyro_z": 0.0,
                "brake": 1,
                "blinker_r": 0,
                "blinker_l": 0,
            },
        )

        self.assertEqual(result["level2"]["speed_trend"], "DECREASING")
        self.assertEqual(result["level2"]["acceleration_cause"], "DRIVER_BRAKE")
        self.assertEqual(result["graph"]["concepts"]["speed_state"], "DECELERATING")

    def test_post_process_corrects_stop_bias_to_constant_speed(self):
        generator = FakeFramePromptGenerator(
            [
                json.dumps(
                    {
                        "reasoning": "直進に見える",
                        "road_shape": "STRAIGHT",
                        "trajectory_relation": "PARALLEL",
                        "slope": "FLAT",
                        "intersection_detected": "NO",
                        "direction_change": "STRAIGHT",
                        "visual_shift": "NO_SHIFT",
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "速度変化は小さい",
                        "acceleration_cause": "MIXED",
                        "speed_trend": "STABLE",
                        "consistency_check": "CONSISTENT",
                    }
                ),
                json.dumps(
                    {
                        "final_reasoning": "停止",
                        "category_id": 4,
                        "category_name": "停止",
                        "confidence": 0.55,
                    }
                ),
            ]
        )

        pipeline = L2MCoTPipeline(generator)
        result = pipeline.analyze_with_l2m(
            make_frames(),
            {
                "speed": 12.0,
                "speed_diff": 0.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": 0.0,
                "acc_x": 0.0,
                "acc_y": 0.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
            },
        )

        self.assertEqual(result["final_category"], 1)

    def test_level1_contradicted_motion_cue_is_downgraded(self):
        pipeline = L2MCoTPipeline(FakeFramePromptGenerator([]))
        normalized = pipeline._normalize_level1_result(
            {
                "trajectory_motion_cue": "LEFT_TURN_CUE",
                "road_shape": "STRAIGHT",
                "trajectory_relation": "PARALLEL",
                "intersection_detected": "NO",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
            }
        )

        self.assertEqual(normalized["trajectory_motion_cue"], "AMBIGUOUS")
        self.assertEqual(normalized["raw_trajectory_motion_cue"], "LEFT_TURN_CUE")
        self.assertEqual(normalized["trajectory_motion_cue_consistency"], "CONTRADICTED")
        self.assertLess(normalized["trajectory_motion_cue_confidence"], 0.35)

    def test_long_lookback_disables_speed_delta_evidence(self):
        pipeline = L2MCoTPipeline(FakeFramePromptGenerator([]))
        motion = pipeline._build_motion_observation(
            {
                "speed": 20.0,
                "speed_diff": 15.0,
                "timestamp_diff_sec": 1800.0,
                "speed_change_rate": 0.008,
                "brake": 0,
                "motion_feature_reliable": False,
            }
        )

        self.assertFalse(motion["reliable"])
        self.assertEqual(motion["trend"], "STABLE")
        self.assertNotEqual(motion["trend"], "INCREASING")

    def test_deterministic_red_trajectory_overrides_vlm_geometry(self):
        pipeline = L2MCoTPipeline(FakeFramePromptGenerator([]))
        merged = pipeline._apply_deterministic_trajectory_features(
            {
                "trajectory_motion_cue": "LEFT_TURN_CUE",
                "trajectory_relation": "PARALLEL",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
                "intersection_detected": "NO",
            },
            {
                "trajectory_motion_cue_deterministic": "RIGHT_TURN_CUE",
                "trajectory_relation_deterministic": "CROSSING_RIGHT",
                "direction_change_deterministic": "TURNING",
                "visual_shift_deterministic": "SHIFT_RIGHT",
                "trajectory_visual_speed_state": "MEDIUM",
                "trajectory_point_like": False,
                "trajectory_heading_delta_rad": -0.42,
                "trajectory_lateral_offset_m": -1.2,
            },
        )
        normalized = pipeline._normalize_level1_result(merged)

        self.assertEqual(normalized["trajectory_motion_cue"], "RIGHT_TURN_CUE")
        self.assertEqual(normalized["raw_trajectory_motion_cue"], "RIGHT_TURN_CUE")
        self.assertEqual(normalized["trajectory_motion_cue_consistency"], "CONSISTENT")
        self.assertGreaterEqual(normalized["trajectory_motion_cue_confidence"], 0.6)

    def test_turn_cue_without_confirmed_intersection_becomes_uncertain_intersection(self):
        pipeline = L2MCoTPipeline(FakeFramePromptGenerator([]))
        normalized = pipeline._normalize_level1_result(
            {
                "trajectory_motion_cue": "LEFT_TURN_CUE",
                "road_shape": "STRAIGHT",
                "trajectory_relation": "CROSSING_LEFT",
                "intersection_detected": "NO",
                "direction_change": "TURNING",
                "visual_shift": "SHIFT_LEFT",
                "trajectory_heading_delta_rad": 0.42,
                "trajectory_lateral_offset_m": 1.2,
                "trajectory_point_like": False,
            }
        )

        self.assertEqual(normalized["intersection_detected"], "UNCERTAIN")

    def test_straight_stop_like_scene_downgrades_raw_yes_intersection(self):
        pipeline = L2MCoTPipeline(FakeFramePromptGenerator([]))
        normalized = pipeline._normalize_level1_result(
            {
                "trajectory_motion_cue": "STRAIGHT_CUE",
                "road_shape": "STRAIGHT",
                "trajectory_relation": "PARALLEL",
                "intersection_detected": "YES",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
                "trajectory_heading_delta_rad": 0.0,
                "trajectory_lateral_offset_m": 0.0,
                "trajectory_point_like": True,
            }
        )

        self.assertEqual(normalized["intersection_detected"], "NO")

    def test_level3_graph_constraint_prefers_lane_change_over_turn(self):
        generator = FakeFramePromptGenerator(
            [
                json.dumps(
                    {
                        "reasoning": "左へ曲がって見える",
                        "trajectory_motion_cue": "LEFT_TURN_CUE",
                        "road_shape": "STRAIGHT",
                        "trajectory_relation": "PARALLEL",
                        "slope": "FLAT",
                        "intersection_detected": "NO",
                        "direction_change": "STRAIGHT",
                        "visual_shift": "NO_SHIFT",
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "速度変化は小さい",
                        "acceleration_cause": "MIXED",
                        "speed_trend": "STABLE",
                        "consistency_check": "CONSISTENT",
                    }
                ),
                json.dumps(
                    {
                        "final_reasoning": "左方向の動きがあるため左折",
                        "category_id": 6,
                        "category_name": "左折",
                        "confidence": 0.71,
                    }
                ),
            ]
        )

        pipeline = L2MCoTPipeline(generator)
        result = pipeline.analyze_with_l2m(
            make_frames(),
            {
                "speed": 18.0,
                "speed_diff": 0.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": 0.0,
                "acc_x": 0.0,
                "acc_y": 0.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
                "motion_feature_reliable": True,
                "trajectory_motion_cue_deterministic": "LEFT_LANE_CHANGE_CUE",
                "trajectory_relation_deterministic": "CROSSING_LEFT",
                "direction_change_deterministic": "STRAIGHT",
                "visual_shift_deterministic": "SHIFT_LEFT",
                "trajectory_visual_speed_state": "FAST",
                "trajectory_point_like": False,
                "trajectory_length_state": "LONG",
                "trajectory_heading_delta_rad": 0.08,
                "trajectory_lateral_offset_m": 1.4,
                "trajectory_curvature_score": 0.08,
            },
        )

        self.assertEqual(result["level3"]["category_id"], 8)
        self.assertEqual(result["final_category"], 8)

    def test_turn_prediction_is_overridden_by_stronger_speed_event(self):
        pipeline = L2MCoTPipeline(FakeFramePromptGenerator([]))
        constrained = pipeline._apply_level3_graph_constraints(
            {
                "final_reasoning": "右折に見えるため右折",
                "category_id": 7,
                "category_name": "右折",
                "confidence": 0.62,
            },
            sensor_data={
                "speed": 25.0,
            },
            level1_result={
                "intersection_detected": "UNCERTAIN",
            },
            graph_result={
                "concepts": {
                    "intersection_state": "UNCERTAIN",
                    "stop_likelihood": "LOW",
                    "trajectory_point_like": False,
                    "motion_feature_reliable": False,
                },
                "label_support": {
                    3: 0.72,
                    7: 0.58,
                },
                "top_candidates": [
                    {"category_id": 3, "category_name": "減速", "score": 0.72, "reasons": ["速度差分またはトレンドが減速"]},
                    {"category_id": 7, "category_name": "右折", "score": 0.58, "reasons": ["右方向変化"]},
                ],
            },
        )

        self.assertEqual(constrained["category_id"], 3)

    def test_point_like_low_speed_without_brake_stays_constant(self):
        generator = FakeFramePromptGenerator(
            [
                json.dumps(
                    {
                        "reasoning": "直進に見える",
                        "trajectory_motion_cue": "STRAIGHT_CUE",
                        "road_shape": "STRAIGHT",
                        "trajectory_relation": "PARALLEL",
                        "slope": "FLAT",
                        "intersection_detected": "YES",
                        "direction_change": "STRAIGHT",
                        "visual_shift": "NO_SHIFT",
                    }
                ),
                json.dumps(
                    {
                        "reasoning": "速度変化は小さい",
                        "acceleration_cause": "MIXED",
                        "speed_trend": "STABLE",
                        "consistency_check": "CONSISTENT",
                    }
                ),
                json.dumps(
                    {
                        "final_reasoning": "直進に見えるため等速走行",
                        "category_id": 1,
                        "category_name": "等速走行",
                        "confidence": 0.62,
                    }
                ),
            ]
        )

        pipeline = L2MCoTPipeline(generator)
        result = pipeline.analyze_with_l2m(
            make_frames(),
            {
                "speed": 6.0,
                "speed_diff": 0.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": 0.0,
                "acc_x": 0.0,
                "acc_y": 0.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
                "motion_feature_reliable": True,
                "trajectory_motion_cue_deterministic": "STRAIGHT_CUE",
                "trajectory_relation_deterministic": "PARALLEL",
                "direction_change_deterministic": "STRAIGHT",
                "visual_shift_deterministic": "NO_SHIFT",
                "trajectory_visual_speed_state": "STOPPED",
                "trajectory_point_like": True,
                "trajectory_length_state": "POINT",
                "trajectory_heading_delta_rad": 0.0,
                "trajectory_lateral_offset_m": 0.0,
                "trajectory_curvature_score": 0.0,
            },
        )

        self.assertEqual(result["level3"]["category_id"], 1)
        self.assertEqual(result["final_category"], 1)


class DrivingConceptGraphBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = DrivingConceptGraphBuilder()

    def test_graph_prefers_direct_negative_motion_over_driver_accel_tag(self):
        result = self.builder.build(
            {
                "road_shape": "STRAIGHT",
                "trajectory_relation": "PARALLEL",
                "intersection_detected": "NO",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
            },
            {
                "acceleration_cause": "DRIVER_ACCEL",
                "speed_trend": "STABLE",
                "consistency_check": "CONSISTENT",
            },
            {
                "speed": 10.0,
                "speed_diff": -7.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": -7.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
            },
        )

        self.assertEqual(result["concepts"]["speed_state"], "DECELERATING")

    def test_low_speed_without_brake_does_not_raise_stop_likelihood(self):
        result = self.builder.build(
            {
                "road_shape": "STRAIGHT",
                "trajectory_relation": "PARALLEL",
                "trajectory_motion_cue": "STRAIGHT_CUE",
                "intersection_detected": "NO",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
            },
            {
                "acceleration_cause": "MIXED",
                "speed_trend": "STABLE",
                "consistency_check": "CONSISTENT",
            },
            {
                "speed": 3.0,
                "speed_diff": 0.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": 0.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
            },
        )

        self.assertEqual(result["concepts"]["stop_likelihood"], "LOW")

    def test_trajectory_motion_cue_directly_supports_lane_change(self):
        result = self.builder.build(
            {
                "road_shape": "STRAIGHT",
                "trajectory_relation": "PARALLEL",
                "trajectory_motion_cue": "LEFT_LANE_CHANGE_CUE",
                "intersection_detected": "NO",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
            },
            {
                "acceleration_cause": "MIXED",
                "speed_trend": "STABLE",
                "consistency_check": "CONSISTENT",
            },
            {
                "speed": 20.0,
                "speed_diff": 0.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": 0.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 1,
            },
        )

        self.assertEqual(result["concepts"]["trajectory_motion_cue"], "LEFT_LANE_CHANGE_CUE")
        self.assertEqual(result["concepts"]["lane_crossing_state"], "LEFT")
        self.assertEqual(result["top_candidates"][0]["category_id"], 8)

    def test_graph_ignores_contradicted_turn_cue(self):
        result = self.builder.build(
            {
                "road_shape": "STRAIGHT",
                "trajectory_relation": "PARALLEL",
                "trajectory_motion_cue": "AMBIGUOUS",
                "raw_trajectory_motion_cue": "LEFT_TURN_CUE",
                "trajectory_motion_cue_confidence": 0.0,
                "trajectory_motion_cue_consistency": "CONTRADICTED",
                "intersection_detected": "NO",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
            },
            {
                "acceleration_cause": "MIXED",
                "speed_trend": "STABLE",
                "consistency_check": "CONSISTENT",
            },
            {
                "speed": 18.0,
                "speed_diff": 0.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": 0.0,
                "motion_feature_reliable": True,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
            },
        )

        self.assertEqual(result["concepts"]["trajectory_direction"], "STRAIGHT")
        self.assertEqual(result["top_candidates"][0]["category_id"], 1)
        self.assertTrue(
            any(edge["type"] == "contradicts" for edge in result["edges"]),
            "contradicted trajectory cue should appear in graph edges",
        )

    def test_point_like_red_trajectory_raises_stop_likelihood(self):
        result = self.builder.build(
            {
                "road_shape": "STRAIGHT",
                "trajectory_relation": "PARALLEL",
                "trajectory_motion_cue": "STRAIGHT_CUE",
                "raw_trajectory_motion_cue": "STRAIGHT_CUE",
                "trajectory_motion_cue_confidence": 1.0,
                "trajectory_motion_cue_consistency": "CONSISTENT",
                "intersection_detected": "NO",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
            },
            {
                "acceleration_cause": "MIXED",
                "speed_trend": "STABLE",
                "consistency_check": "CONSISTENT",
            },
            {
                "speed": 1.5,
                "speed_diff": 0.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": 0.0,
                "motion_feature_reliable": True,
                "trajectory_point_like": True,
                "trajectory_visual_speed_state": "STOPPED",
                "trajectory_length_state": "POINT",
                "trajectory_lateral_offset_m": 0.0,
                "trajectory_heading_delta_rad": 0.0,
                "trajectory_curvature_score": 0.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
            },
        )

        self.assertIn(result["concepts"]["stop_likelihood"], {"MEDIUM", "HIGH"})
        self.assertEqual(result["concepts"]["trajectory_visual_speed_state"], "STOPPED")

    def test_stop_like_low_speed_without_brake_prefers_constant(self):
        result = self.builder.build(
            {
                "road_shape": "STRAIGHT",
                "trajectory_relation": "PARALLEL",
                "trajectory_motion_cue": "STRAIGHT_CUE",
                "raw_trajectory_motion_cue": "STRAIGHT_CUE",
                "trajectory_motion_cue_confidence": 1.0,
                "trajectory_motion_cue_consistency": "CONSISTENT",
                "intersection_detected": "NO",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
            },
            {
                "acceleration_cause": "MIXED",
                "speed_trend": "STABLE",
                "consistency_check": "CONSISTENT",
            },
            {
                "speed": 6.0,
                "speed_diff": 0.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": 0.0,
                "motion_feature_reliable": True,
                "trajectory_point_like": True,
                "trajectory_visual_speed_state": "STOPPED",
                "trajectory_length_state": "POINT",
                "trajectory_lateral_offset_m": 0.0,
                "trajectory_heading_delta_rad": 0.0,
                "trajectory_curvature_score": 0.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
            },
        )

        self.assertEqual(result["top_candidates"][0]["category_id"], 1)

    def test_starting_from_stop_like_state_scores_start(self):
        result = self.builder.build(
            {
                "road_shape": "STRAIGHT",
                "trajectory_relation": "PARALLEL",
                "trajectory_motion_cue": "STRAIGHT_CUE",
                "raw_trajectory_motion_cue": "STRAIGHT_CUE",
                "trajectory_motion_cue_confidence": 1.0,
                "trajectory_motion_cue_consistency": "CONSISTENT",
                "intersection_detected": "NO",
                "direction_change": "STRAIGHT",
                "visual_shift": "NO_SHIFT",
            },
            {
                "acceleration_cause": "DRIVER_ACCEL",
                "speed_trend": "INCREASING",
                "consistency_check": "CONSISTENT",
            },
            {
                "speed": 6.0,
                "speed_diff": 3.0,
                "timestamp_diff_sec": 1.0,
                "speed_change_rate": 3.0,
                "motion_feature_reliable": True,
                "trajectory_point_like": False,
                "trajectory_visual_speed_state": "SLOW",
                "trajectory_length_state": "SHORT",
                "trajectory_lateral_offset_m": 0.0,
                "trajectory_heading_delta_rad": 0.0,
                "trajectory_curvature_score": 0.0,
                "gyro_z": 0.0,
                "brake": 0,
                "blinker_r": 0,
                "blinker_l": 0,
            },
        )

        self.assertEqual(result["concepts"]["speed_state"], "STARTING")
        self.assertEqual(result["top_candidates"][0]["category_id"], 5)


class TrajectoryVisualizerTests(unittest.TestCase):
    def setUp(self):
        self.visualizer = TrajectoryVisualizer(image_size=(1280, 720))

    def test_extract_trajectory_features_detects_left_turn_from_curve(self):
        trajectory_3d = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.2, 0.0],
                [3.0, 0.8, 0.0],
                [4.0, 1.6, 0.0],
            ],
            dtype=np.float32,
        )
        image_points = np.array(
            [
                [640.0, 700.0],
                [640.0, 650.0],
                [625.0, 600.0],
                [590.0, 560.0],
                [545.0, 520.0],
            ],
            dtype=np.float32,
        )
        valid_mask = np.array([True, True, True, True, True])

        features = self.visualizer.extract_trajectory_features(
            trajectory_3d=trajectory_3d,
            image_points=image_points,
            valid_mask=valid_mask,
            speed_kmh=20.0,
        )

        self.assertEqual(features["trajectory_motion_cue_deterministic"], "LEFT_TURN_CUE")
        self.assertEqual(features["direction_change_deterministic"], "TURNING")
        self.assertEqual(features["visual_shift_deterministic"], "SHIFT_LEFT")
        self.assertFalse(features["trajectory_point_like"])

    def test_extract_trajectory_features_detects_point_like_stop(self):
        trajectory_3d = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0],
                [0.04, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        image_points = np.array(
            [
                [640.0, 700.0],
                [641.0, 700.0],
                [642.0, 700.0],
            ],
            dtype=np.float32,
        )
        valid_mask = np.array([True, True, True])

        features = self.visualizer.extract_trajectory_features(
            trajectory_3d=trajectory_3d,
            image_points=image_points,
            valid_mask=valid_mask,
            speed_kmh=0.5,
        )

        self.assertTrue(features["trajectory_point_like"])
        self.assertEqual(features["trajectory_visual_speed_state"], "STOPPED")
        self.assertEqual(features["trajectory_length_state"], "POINT")


if __name__ == "__main__":
    unittest.main()
