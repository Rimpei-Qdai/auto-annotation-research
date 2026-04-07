import json
import sys
import unittest
from pathlib import Path

from PIL import Image


ANNOTATION_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_TOOL_DIR))

from heron_l2m_pipeline import L2MCoTPipeline  # noqa: E402
from driving_graph import DrivingConceptGraphBuilder  # noqa: E402


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


if __name__ == "__main__":
    unittest.main()
