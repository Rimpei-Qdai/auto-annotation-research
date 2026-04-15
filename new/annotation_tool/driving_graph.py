# driving_graph.py
"""
Driving concept graph utilities for automatic annotation.

The graph is intentionally lightweight:
- extract a small set of interpretable concepts from sensor + L2M outputs
- connect them with simple support / contradiction edges
- produce label support scores that can be injected back into Level 3
"""

from __future__ import annotations

from typing import Any, Dict, List

from config import ACTION_LABELS, GYRO_THRESHOLD, SPEED_STOP_THRESHOLD


class DrivingConceptGraphBuilder:
    """Build a compact concept graph for downstream driving-action reasoning."""

    SPEED_DIFF_THRESHOLD = 0.5
    STRONG_SPEED_DIFF_THRESHOLD = 1.0
    SPEED_RATE_THRESHOLD = 0.6
    STABLE_SPEED_RATE_THRESHOLD = 0.25
    MAX_RELIABLE_TIME_DIFF_SEC = 4.0
    MAX_RELIABLE_MOTION_LOOKBACK_SEC = 10.0
    START_SPEED_THRESHOLD = 5.0
    STOPLIKE_SPEED_THRESHOLD = 12.0
    STRONG_SUPPORT_THRESHOLD = 0.75
    STRONG_MARGIN_THRESHOLD = 0.15

    def build(
        self,
        level1_result: Dict[str, Any],
        level2_result: Dict[str, Any],
        sensor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        concepts = self._extract_concepts(level1_result, level2_result, sensor_data)
        edges = self._build_edges(concepts)
        scored = self._score_labels(concepts, sensor_data)

        label_support = scored["label_support"]
        label_reasons = scored["label_reasons"]

        top_candidates = self._build_top_candidates(label_support, label_reasons)
        strong_candidate = self._select_strong_candidate(top_candidates)

        return {
            "concepts": concepts,
            "edges": edges,
            "label_support": label_support,
            "label_reasons": label_reasons,
            "top_candidates": top_candidates,
            "strong_candidate": strong_candidate,
            "summary_for_prompt": self._format_concept_summary(concepts, edges),
            "candidate_summary": self._format_candidate_summary(top_candidates),
        }

    def _extract_concepts(
        self,
        level1_result: Dict[str, Any],
        level2_result: Dict[str, Any],
        sensor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        speed = float(sensor_data.get("speed", 0.0))
        speed_diff = float(sensor_data.get("speed_diff", 0.0) or 0.0)
        timestamp_diff_sec = float(sensor_data.get("timestamp_diff_sec", 0.0) or 0.0)
        speed_change_rate = float(sensor_data.get("speed_change_rate", 0.0) or 0.0)
        brake = int(sensor_data.get("brake", 0) or 0)
        gyro_z = float(sensor_data.get("gyro_z", 0.0) or 0.0)
        blinker_r = int(sensor_data.get("blinker_r", 0) or 0)
        blinker_l = int(sensor_data.get("blinker_l", 0) or 0)
        motion_feature_reliable = self._is_motion_feature_reliable(sensor_data)
        trajectory_point_like = bool(sensor_data.get("trajectory_point_like", False))
        trajectory_visual_speed_state = sensor_data.get("trajectory_visual_speed_state", "UNKNOWN")
        trajectory_length_state = sensor_data.get("trajectory_length_state", "UNKNOWN")
        trajectory_lateral_offset_m = float(sensor_data.get("trajectory_lateral_offset_m", 0.0) or 0.0)
        trajectory_heading_delta_rad = float(sensor_data.get("trajectory_heading_delta_rad", 0.0) or 0.0)
        trajectory_curvature_score = float(sensor_data.get("trajectory_curvature_score", 0.0) or 0.0)

        speed_trend = level2_result.get("speed_trend", "STABLE")
        acceleration_cause = level2_result.get("acceleration_cause", "MIXED")
        consistency_check = level2_result.get("consistency_check", "CONSISTENT")

        road_shape = level1_result.get("road_shape", "STRAIGHT")
        trajectory_relation = level1_result.get("trajectory_relation", "PARALLEL")
        trajectory_motion_cue = level1_result.get("trajectory_motion_cue", "AMBIGUOUS")
        raw_trajectory_motion_cue = level1_result.get("raw_trajectory_motion_cue", trajectory_motion_cue)
        default_cue_confidence = 1.0 if trajectory_motion_cue not in {"AMBIGUOUS", None, ""} else 0.0
        trajectory_motion_cue_confidence = float(
            level1_result.get("trajectory_motion_cue_confidence", default_cue_confidence) or 0.0
        )
        trajectory_motion_cue_consistency = level1_result.get(
            "trajectory_motion_cue_consistency",
            "CONSISTENT" if default_cue_confidence > 0.0 else "UNKNOWN",
        )
        intersection_detected = level1_result.get("intersection_detected", "NO")
        direction_change = level1_result.get("direction_change", "STRAIGHT")
        visual_shift = level1_result.get("visual_shift", "NO_SHIFT")

        speed_state = self._infer_speed_state(
            speed=speed,
            speed_diff=speed_diff,
            timestamp_diff_sec=timestamp_diff_sec,
            speed_change_rate=speed_change_rate,
            brake=brake,
            motion_feature_reliable=motion_feature_reliable,
            trajectory_point_like=trajectory_point_like,
            trajectory_visual_speed_state=trajectory_visual_speed_state,
            speed_trend=speed_trend,
            acceleration_cause=acceleration_cause,
        )

        if blinker_l > 0:
            signal_state = "LEFT"
        elif blinker_r > 0:
            signal_state = "RIGHT"
        else:
            signal_state = "OFF"

        trusted_motion_cue = (
            trajectory_motion_cue_consistency == "CONSISTENT"
            and trajectory_motion_cue_confidence >= 0.6
        )

        if trusted_motion_cue and trajectory_motion_cue in {"LEFT_TURN_CUE", "LEFT_LANE_CHANGE_CUE"}:
            trajectory_direction = "LEFT"
        elif trusted_motion_cue and trajectory_motion_cue in {"RIGHT_TURN_CUE", "RIGHT_LANE_CHANGE_CUE"}:
            trajectory_direction = "RIGHT"
        elif gyro_z > GYRO_THRESHOLD or visual_shift == "SHIFT_LEFT":
            trajectory_direction = "LEFT"
        elif gyro_z < -GYRO_THRESHOLD or visual_shift == "SHIFT_RIGHT":
            trajectory_direction = "RIGHT"
        elif trajectory_lateral_offset_m > 0.45:
            trajectory_direction = "LEFT"
        elif trajectory_lateral_offset_m < -0.45:
            trajectory_direction = "RIGHT"
        elif direction_change == "TURNING":
            if road_shape == "CURVE_LEFT":
                trajectory_direction = "LEFT"
            elif road_shape == "CURVE_RIGHT":
                trajectory_direction = "RIGHT"
            else:
                trajectory_direction = "STRAIGHT"
        else:
            trajectory_direction = "STRAIGHT"

        if trusted_motion_cue and trajectory_motion_cue in {"LEFT_TURN_CUE", "RIGHT_TURN_CUE"}:
            turn_intensity = "HIGH"
        elif (
            abs(gyro_z) > GYRO_THRESHOLD * 1.8
            or direction_change == "TURNING"
            or trajectory_curvature_score >= 0.28
        ):
            turn_intensity = "HIGH"
        elif (
            abs(gyro_z) > GYRO_THRESHOLD * 0.8
            or visual_shift != "NO_SHIFT"
            or trajectory_curvature_score >= 0.14
        ):
            turn_intensity = "MEDIUM"
        else:
            turn_intensity = "LOW"

        if trusted_motion_cue and trajectory_motion_cue == "LEFT_LANE_CHANGE_CUE":
            lane_crossing_state = "LEFT"
        elif trusted_motion_cue and trajectory_motion_cue == "RIGHT_LANE_CHANGE_CUE":
            lane_crossing_state = "RIGHT"
        elif trajectory_relation == "CROSSING_LEFT":
            lane_crossing_state = "LEFT"
        elif trajectory_relation == "CROSSING_RIGHT":
            lane_crossing_state = "RIGHT"
        else:
            lane_crossing_state = "NONE"

        if intersection_detected == "YES":
            intersection_state = "YES"
        elif intersection_detected == "NO":
            intersection_state = "NO"
        elif intersection_detected == "UNCERTAIN":
            intersection_state = "UNCERTAIN"
        else:
            intersection_state = "UNKNOWN"

        stop_likelihood = self._infer_stop_likelihood(
            speed=speed,
            brake=brake,
            speed_state=speed_state,
            speed_diff=speed_diff,
            speed_change_rate=speed_change_rate,
            motion_feature_reliable=motion_feature_reliable,
            trajectory_point_like=trajectory_point_like,
            trajectory_visual_speed_state=trajectory_visual_speed_state,
        )

        return {
            "speed_state": speed_state,
            "signal_state": signal_state,
            "raw_trajectory_motion_cue": raw_trajectory_motion_cue,
            "trajectory_motion_cue": trajectory_motion_cue,
            "trajectory_motion_cue_confidence": trajectory_motion_cue_confidence,
            "trajectory_motion_cue_consistency": trajectory_motion_cue_consistency,
            "trajectory_direction": trajectory_direction,
            "turn_intensity": turn_intensity,
            "lane_crossing_state": lane_crossing_state,
            "intersection_state": intersection_state,
            "stop_likelihood": stop_likelihood,
            "consistency_check": consistency_check,
            "road_shape": road_shape,
            "motion_feature_reliable": motion_feature_reliable,
            "trajectory_point_like": trajectory_point_like,
            "trajectory_visual_speed_state": trajectory_visual_speed_state,
            "trajectory_length_state": trajectory_length_state,
            "trajectory_lateral_offset_m": trajectory_lateral_offset_m,
            "trajectory_heading_delta_rad": trajectory_heading_delta_rad,
            "trajectory_curvature_score": trajectory_curvature_score,
        }

    def _is_motion_feature_reliable(self, sensor_data: Dict[str, Any]) -> bool:
        explicit = sensor_data.get("motion_feature_reliable")
        if explicit is not None:
            return bool(explicit)

        timestamp_diff_sec = float(sensor_data.get("timestamp_diff_sec", 0.0) or 0.0)
        return 0.0 < timestamp_diff_sec <= self.MAX_RELIABLE_MOTION_LOOKBACK_SEC

    def _infer_speed_state(
        self,
        *,
        speed: float,
        speed_diff: float,
        timestamp_diff_sec: float,
        speed_change_rate: float,
        brake: int,
        motion_feature_reliable: bool,
        trajectory_point_like: bool,
        trajectory_visual_speed_state: str,
        speed_trend: str,
        acceleration_cause: str,
    ) -> str:
        has_reliable_rate = motion_feature_reliable and 0.0 < timestamp_diff_sec <= self.MAX_RELIABLE_TIME_DIFF_SEC
        direct_accel = (
            (motion_feature_reliable and speed_diff >= self.STRONG_SPEED_DIFF_THRESHOLD)
            or (has_reliable_rate and speed_change_rate >= self.SPEED_RATE_THRESHOLD)
        )
        direct_decel = (
            brake > 0
            or (motion_feature_reliable and speed_diff <= -self.STRONG_SPEED_DIFF_THRESHOLD)
            or (has_reliable_rate and speed_change_rate <= -self.SPEED_RATE_THRESHOLD)
        )
        near_constant = (
            ((motion_feature_reliable and abs(speed_diff) <= self.SPEED_DIFF_THRESHOLD) or not motion_feature_reliable)
            and (
                not has_reliable_rate
                or abs(speed_change_rate) <= self.STABLE_SPEED_RATE_THRESHOLD
            )
            and brake == 0
        )

        if (
            trajectory_point_like
            and trajectory_visual_speed_state == "STOPPED"
            and speed < self.STOPLIKE_SPEED_THRESHOLD
        ):
            if direct_accel and brake == 0:
                return "STARTING"
            if brake > 0 or (speed < self.START_SPEED_THRESHOLD and direct_decel):
                return "STOPPED"

        if speed < SPEED_STOP_THRESHOLD and brake > 0 and not direct_accel:
            return "STOPPED"

        if (
            speed < self.STOPLIKE_SPEED_THRESHOLD
            and direct_accel
            and brake == 0
            and trajectory_visual_speed_state in {"STOPPED", "SLOW"}
        ):
            return "STARTING"

        if direct_decel and not direct_accel:
            return "DECELERATING"
        if direct_accel and not direct_decel:
            return "ACCELERATING"
        if direct_decel and direct_accel:
            if brake > 0 or speed_diff < 0 or speed_change_rate < 0:
                return "DECELERATING"
            return "ACCELERATING"

        if speed_trend == "DECREASING" or acceleration_cause == "DRIVER_BRAKE":
            return "DECELERATING"
        if speed_trend == "INCREASING" or acceleration_cause == "DRIVER_ACCEL":
            return "ACCELERATING"
        if near_constant:
            return "CONSTANT"
        return "CONSTANT"

    def _infer_stop_likelihood(
        self,
        *,
        speed: float,
        brake: int,
        speed_state: str,
        speed_diff: float,
        speed_change_rate: float,
        motion_feature_reliable: bool,
        trajectory_point_like: bool,
        trajectory_visual_speed_state: str,
    ) -> str:
        if (
            trajectory_point_like
            and trajectory_visual_speed_state == "STOPPED"
            and speed < self.STOPLIKE_SPEED_THRESHOLD
        ):
            if brake > 0 or (speed_state in {"STOPPED", "DECELERATING"} and speed < SPEED_STOP_THRESHOLD):
                return "HIGH"
            return "MEDIUM"

        if speed < SPEED_STOP_THRESHOLD and brake > 0 and speed_state == "STOPPED":
            return "HIGH"

        if (
            speed < self.STOPLIKE_SPEED_THRESHOLD
            and trajectory_visual_speed_state in {"STOPPED", "SLOW"}
            and speed_state in {"STOPPED", "DECELERATING", "CONSTANT"}
        ):
            return "MEDIUM" if brake == 0 else "HIGH"

        if (
            speed < self.STOPLIKE_SPEED_THRESHOLD
            and speed_state == "DECELERATING"
            and (
                brake > 0
                or (motion_feature_reliable and speed_diff < -self.SPEED_DIFF_THRESHOLD)
                or (motion_feature_reliable and speed_change_rate < 0)
            )
        ):
            return "MEDIUM"

        return "LOW"

    def _build_edges(self, concepts: Dict[str, Any]) -> List[Dict[str, str]]:
        edges: List[Dict[str, str]] = []

        cue = concepts["trajectory_motion_cue"]
        raw_cue = concepts["raw_trajectory_motion_cue"]
        cue_confidence = float(concepts["trajectory_motion_cue_confidence"])
        cue_consistency = concepts["trajectory_motion_cue_consistency"]

        if cue_consistency == "CONTRADICTED" and raw_cue not in {"AMBIGUOUS", "STRAIGHT_CUE"}:
            target = (
                "lane_crossing_state"
                if raw_cue in {"LEFT_LANE_CHANGE_CUE", "RIGHT_LANE_CHANGE_CUE"}
                else "trajectory_direction"
            )
            edges.append(
                {
                    "source": "trajectory_motion_cue",
                    "target": target,
                    "type": "contradicts",
                    "reason": "赤い軌道キューが他の幾何特徴と矛盾している",
                }
            )

        if (
            cue in {"LEFT_TURN_CUE", "RIGHT_TURN_CUE"}
            and cue_consistency == "CONSISTENT"
            and cue_confidence >= 0.6
        ):
            edges.append(
                {
                    "source": "trajectory_motion_cue",
                    "target": "trajectory_direction",
                    "type": "supports_turn",
                    "reason": f"赤い軌道が将来の回頭を直接示している (confidence={cue_confidence:.2f})",
                }
            )
        elif (
            cue in {"LEFT_LANE_CHANGE_CUE", "RIGHT_LANE_CHANGE_CUE"}
            and cue_consistency == "CONSISTENT"
            and cue_confidence >= 0.6
        ):
            edges.append(
                {
                    "source": "trajectory_motion_cue",
                    "target": "lane_crossing_state",
                    "type": "supports_lane_change",
                    "reason": f"赤い軌道が横方向移動を直接示している (confidence={cue_confidence:.2f})",
                }
            )

        if concepts["signal_state"] == concepts["trajectory_direction"] and concepts["signal_state"] != "OFF":
            edges.append(
                {
                    "source": "signal_state",
                    "target": "trajectory_direction",
                    "type": "supports",
                    "reason": "ウィンカー方向と推定進行方向が一致",
                }
            )
        elif (
            concepts["signal_state"] != "OFF"
            and concepts["trajectory_direction"] != "STRAIGHT"
            and concepts["signal_state"] != concepts["trajectory_direction"]
        ):
            edges.append(
                {
                    "source": "signal_state",
                    "target": "trajectory_direction",
                    "type": "contradicts",
                    "reason": "ウィンカー方向と推定進行方向が不一致",
                }
            )

        if concepts["intersection_state"] == "YES" and concepts["trajectory_direction"] in {"LEFT", "RIGHT"}:
            edges.append(
                {
                    "source": "intersection_state",
                    "target": "trajectory_direction",
                    "type": "supports_turn",
                    "reason": "交差点あり + 左右方向変化",
                }
            )
        elif concepts["intersection_state"] == "UNCERTAIN" and concepts["trajectory_direction"] in {"LEFT", "RIGHT"}:
            edges.append(
                {
                    "source": "intersection_state",
                    "target": "trajectory_direction",
                    "type": "weak_turn_context",
                    "reason": "交差点文脈は不確実だが左右方向変化あり",
                }
            )

        if concepts["intersection_state"] == "NO" and concepts["lane_crossing_state"] in {"LEFT", "RIGHT"}:
            edges.append(
                {
                    "source": "lane_crossing_state",
                    "target": "intersection_state",
                    "type": "supports_lane_change",
                    "reason": "交差点なし + 車線横断あり",
                }
            )

        if concepts["speed_state"] == "DECELERATING" and concepts["stop_likelihood"] in {"HIGH", "MEDIUM"}:
            edges.append(
                {
                    "source": "speed_state",
                    "target": "stop_likelihood",
                    "type": "supports",
                    "reason": "減速状態は停止候補を支持",
                }
            )

        return edges

    def _score_labels(
        self,
        concepts: Dict[str, Any],
        sensor_data: Dict[str, Any],
    ) -> Dict[str, Dict[int, Any]]:
        label_support = {label_id: 0.0 for label_id in ACTION_LABELS.keys()}
        label_reasons = {label_id: [] for label_id in ACTION_LABELS.keys()}

        def add(label_id: int, amount: float, reason: str) -> None:
            label_support[label_id] = min(1.0, label_support[label_id] + amount)
            label_reasons[label_id].append(reason)

        speed = float(sensor_data.get("speed", 0.0))
        speed_diff = float(sensor_data.get("speed_diff", 0.0) or 0.0)
        brake = int(sensor_data.get("brake", 0) or 0)

        speed_state = concepts["speed_state"]
        direction = concepts["trajectory_direction"]
        lane_crossing = concepts["lane_crossing_state"]
        intersection = concepts["intersection_state"]
        signal = concepts["signal_state"]
        trajectory_motion_cue = concepts["trajectory_motion_cue"]
        raw_trajectory_motion_cue = concepts["raw_trajectory_motion_cue"]
        trajectory_motion_cue_confidence = float(concepts["trajectory_motion_cue_confidence"])
        trajectory_motion_cue_consistency = concepts["trajectory_motion_cue_consistency"]
        turn_intensity = concepts["turn_intensity"]
        stop_likelihood = concepts["stop_likelihood"]
        trajectory_point_like = bool(concepts.get("trajectory_point_like", False))
        trajectory_visual_speed_state = concepts.get("trajectory_visual_speed_state", "UNKNOWN")
        trusted_cue = (
            trajectory_motion_cue_consistency == "CONSISTENT"
            and trajectory_motion_cue_confidence >= 0.6
        )
        weak_cue = trajectory_motion_cue_consistency == "WEAK" and trajectory_motion_cue_confidence > 0.0
        cue_turn_bonus = round(0.20 * trajectory_motion_cue_confidence, 3)
        cue_lane_bonus = round(0.25 * trajectory_motion_cue_confidence, 3)

        if stop_likelihood == "HIGH":
            add(4, 0.80, "停止寄りの証拠が強い")
        elif stop_likelihood == "MEDIUM":
            add(4, 0.18, "低速で停止寄り")
        if trajectory_point_like and trajectory_visual_speed_state == "STOPPED":
            point_bonus = 0.12 if brake > 0 or speed < SPEED_STOP_THRESHOLD else 0.06
            add(4, point_bonus, "赤い軌道が点状で停止寄り")
            if speed < self.STOPLIKE_SPEED_THRESHOLD and brake > 0:
                add(4, 0.12, "低速かつブレーキONで停止候補を補強")

        if speed_state == "STARTING":
            add(5, 0.85, "停止寄り状態からの立ち上がりで発進候補")
        if speed_state == "ACCELERATING":
            add(2, 0.70, "速度差分またはトレンドが加速")
        if speed_state == "DECELERATING":
            add(3, 0.70, "速度差分またはトレンドが減速")
        if brake > 0 and speed > SPEED_STOP_THRESHOLD:
            add(3, 0.20, "ブレーキONで減速候補を補強")

        if speed_state == "CONSTANT" and direction == "STRAIGHT":
            constant_bonus = 0.35 if stop_likelihood in {"HIGH", "MEDIUM"} else 0.75
            add(1, constant_bonus, "直進かつ速度変化が小さい")
        if signal == "OFF" and lane_crossing == "NONE" and intersection != "YES" and stop_likelihood == "LOW":
            add(1, 0.15, "旋回・車線変更の証拠が弱い")

        if direction == "LEFT":
            add(6, 0.35, "進行方向が左")
            add(8, 0.20, "左方向変化は左車線変更候補にもなる")
        elif direction == "RIGHT":
            add(7, 0.35, "進行方向が右")
            add(9, 0.20, "右方向変化は右車線変更候補にもなる")

        if trusted_cue:
            if trajectory_motion_cue == "LEFT_TURN_CUE":
                add(6, cue_turn_bonus, f"赤い軌道が左折キューを示す (confidence={trajectory_motion_cue_confidence:.2f})")
            elif trajectory_motion_cue == "RIGHT_TURN_CUE":
                add(7, cue_turn_bonus, f"赤い軌道が右折キューを示す (confidence={trajectory_motion_cue_confidence:.2f})")
            elif trajectory_motion_cue == "LEFT_LANE_CHANGE_CUE":
                add(8, cue_lane_bonus, f"赤い軌道が左車線変更キューを示す (confidence={trajectory_motion_cue_confidence:.2f})")
            elif trajectory_motion_cue == "RIGHT_LANE_CHANGE_CUE":
                add(9, cue_lane_bonus, f"赤い軌道が右車線変更キューを示す (confidence={trajectory_motion_cue_confidence:.2f})")
        elif weak_cue:
            weak_turn_bonus = round(cue_turn_bonus * 0.5, 3)
            weak_lane_bonus = round(cue_lane_bonus * 0.5, 3)
            if trajectory_motion_cue == "LEFT_TURN_CUE":
                add(6, weak_turn_bonus, f"赤い軌道は左折キューだが根拠は弱い (confidence={trajectory_motion_cue_confidence:.2f})")
            elif trajectory_motion_cue == "RIGHT_TURN_CUE":
                add(7, weak_turn_bonus, f"赤い軌道は右折キューだが根拠は弱い (confidence={trajectory_motion_cue_confidence:.2f})")
            elif trajectory_motion_cue == "LEFT_LANE_CHANGE_CUE":
                add(8, weak_lane_bonus, f"赤い軌道は左車線変更キューだが根拠は弱い (confidence={trajectory_motion_cue_confidence:.2f})")
            elif trajectory_motion_cue == "RIGHT_LANE_CHANGE_CUE":
                add(9, weak_lane_bonus, f"赤い軌道は右車線変更キューだが根拠は弱い (confidence={trajectory_motion_cue_confidence:.2f})")
        elif trajectory_motion_cue_consistency == "CONTRADICTED":
            if raw_trajectory_motion_cue == "LEFT_TURN_CUE":
                add(6, -0.10, "左折キューは他の幾何特徴と矛盾")
            elif raw_trajectory_motion_cue == "RIGHT_TURN_CUE":
                add(7, -0.10, "右折キューは他の幾何特徴と矛盾")
            elif raw_trajectory_motion_cue == "LEFT_LANE_CHANGE_CUE":
                add(8, -0.10, "左車線変更キューは他の幾何特徴と矛盾")
            elif raw_trajectory_motion_cue == "RIGHT_LANE_CHANGE_CUE":
                add(9, -0.10, "右車線変更キューは他の幾何特徴と矛盾")

        if turn_intensity == "HIGH" and direction == "LEFT":
            add(6, 0.20, "旋回強度が高く左折寄り")
        if turn_intensity == "HIGH" and direction == "RIGHT":
            add(7, 0.20, "旋回強度が高く右折寄り")

        if lane_crossing == "LEFT":
            add(8, 0.35, "左方向の車線横断あり")
        elif lane_crossing == "RIGHT":
            add(9, 0.35, "右方向の車線横断あり")

        if intersection == "YES" and direction == "LEFT":
            add(6, 0.25, "交差点あり + 左方向変化で左折寄り")
            add(8, -0.15, "交差点ありのため左車線変更候補を減衰")
        elif intersection == "YES" and direction == "RIGHT":
            add(7, 0.25, "交差点あり + 右方向変化で右折寄り")
            add(9, -0.15, "交差点ありのため右車線変更候補を減衰")
        elif intersection == "UNCERTAIN" and direction == "LEFT":
            add(6, 0.10, "交差点文脈は不確実だが左折候補を弱く支持")
        elif intersection == "UNCERTAIN" and direction == "RIGHT":
            add(7, 0.10, "交差点文脈は不確実だが右折候補を弱く支持")

        if intersection == "NO" and lane_crossing == "LEFT":
            add(8, 0.20, "交差点なし + 左車線横断で左車線変更寄り")
            add(6, -0.10, "交差点なしのため左折候補を減衰")
        elif intersection == "NO" and lane_crossing == "RIGHT":
            add(9, 0.20, "交差点なし + 右車線横断で右車線変更寄り")
            add(7, -0.10, "交差点なしのため右折候補を減衰")

        if signal == "LEFT":
            add(6, 0.15, "左ウィンカーON")
            add(8, 0.10, "左ウィンカーは左車線変更候補も補強")
        elif signal == "RIGHT":
            add(7, 0.15, "右ウィンカーON")
            add(9, 0.10, "右ウィンカーは右車線変更候補も補強")

        if signal == "OFF" and speed_state == "CONSTANT" and intersection != "YES":
            if direction == "LEFT":
                add(6, -0.15, "交差点確証とウィンカーが弱く左折候補を減衰")
            elif direction == "RIGHT":
                add(7, -0.15, "交差点確証とウィンカーが弱く右折候補を減衰")

        # Keep scores in [0, 1]
        for label_id in label_support:
            label_support[label_id] = round(max(0.0, min(1.0, label_support[label_id])), 3)

        return {
            "label_support": label_support,
            "label_reasons": label_reasons,
        }

    def _build_top_candidates(
        self,
        label_support: Dict[int, float],
        label_reasons: Dict[int, List[str]],
    ) -> List[Dict[str, Any]]:
        ranked = sorted(
            label_support.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        top_candidates = []
        for label_id, score in ranked[:3]:
            top_candidates.append(
                {
                    "category_id": label_id,
                    "category_name": ACTION_LABELS.get(label_id, "その他"),
                    "score": score,
                    "reasons": label_reasons.get(label_id, [])[:3],
                }
            )
        return top_candidates

    def _select_strong_candidate(
        self,
        top_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any] | None:
        if not top_candidates:
            return None

        top = top_candidates[0]
        second_score = top_candidates[1]["score"] if len(top_candidates) > 1 else 0.0
        if (
            top["score"] >= self.STRONG_SUPPORT_THRESHOLD
            and (top["score"] - second_score) >= self.STRONG_MARGIN_THRESHOLD
        ):
            return top
        return None

    def _format_concept_summary(
        self,
        concepts: Dict[str, Any],
        edges: List[Dict[str, str]],
    ) -> str:
        lines = [
            f"- speed_state: {concepts['speed_state']}",
            f"- motion_feature_reliable: {concepts['motion_feature_reliable']}",
            f"- raw_trajectory_motion_cue: {concepts['raw_trajectory_motion_cue']}",
            f"- trajectory_motion_cue: {concepts['trajectory_motion_cue']}",
            f"- trajectory_motion_cue_confidence: {concepts['trajectory_motion_cue_confidence']:.2f}",
            f"- trajectory_motion_cue_consistency: {concepts['trajectory_motion_cue_consistency']}",
            f"- trajectory_visual_speed_state: {concepts.get('trajectory_visual_speed_state', 'UNKNOWN')}",
            f"- trajectory_point_like: {concepts.get('trajectory_point_like', False)}",
            f"- trajectory_length_state: {concepts.get('trajectory_length_state', 'UNKNOWN')}",
            f"- trajectory_lateral_offset_m: {concepts.get('trajectory_lateral_offset_m', 0.0):.2f}",
            f"- trajectory_heading_delta_rad: {concepts.get('trajectory_heading_delta_rad', 0.0):.2f}",
            f"- trajectory_direction: {concepts['trajectory_direction']}",
            f"- turn_intensity: {concepts['turn_intensity']}",
            f"- lane_crossing_state: {concepts['lane_crossing_state']}",
            f"- intersection_state: {concepts['intersection_state']}",
            f"- signal_state: {concepts['signal_state']}",
            f"- stop_likelihood: {concepts['stop_likelihood']}",
            f"- consistency_check: {concepts['consistency_check']}",
        ]
        if edges:
            lines.append("- key_edges:")
            lines.extend(
                f"  - {edge['type']}: {edge['reason']}"
                for edge in edges[:4]
            )
        return "\n".join(lines)

    def _format_candidate_summary(self, top_candidates: List[Dict[str, Any]]) -> str:
        if not top_candidates:
            return "- 候補なし"

        lines = []
        for candidate in top_candidates:
            reasons = " / ".join(candidate["reasons"]) if candidate["reasons"] else "理由なし"
            lines.append(
                f"- {candidate['category_id']}:{candidate['category_name']} "
                f"(score={candidate['score']:.2f}) -> {reasons}"
            )
        return "\n".join(lines)
