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

from config import ACTION_LABELS


DEFAULT_GYRO_THRESHOLD = 0.08
DEFAULT_SPEED_STOP_THRESHOLD = 3.0


class DrivingConceptGraphBuilder:
    """Build a compact concept graph for downstream driving-action reasoning."""

    SPEED_DIFF_THRESHOLD = 0.5
    START_SPEED_THRESHOLD = 5.0
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
        brake = int(sensor_data.get("brake", 0) or 0)
        gyro_z = float(sensor_data.get("gyro_z", 0.0) or 0.0)
        blinker_r = int(sensor_data.get("blinker_r", 0) or 0)
        blinker_l = int(sensor_data.get("blinker_l", 0) or 0)

        speed_trend = level2_result.get("speed_trend", "STABLE")
        acceleration_cause = level2_result.get("acceleration_cause", "MIXED")
        consistency_check = level2_result.get("consistency_check", "CONSISTENT")

        road_shape = level1_result.get("road_shape", "STRAIGHT")
        trajectory_relation = level1_result.get("trajectory_relation", "PARALLEL")
        intersection_detected = level1_result.get("intersection_detected", "NO")
        direction_change = level1_result.get("direction_change", "STRAIGHT")
        visual_shift = level1_result.get("visual_shift", "NO_SHIFT")

        if speed < DEFAULT_SPEED_STOP_THRESHOLD and brake > 0:
            speed_state = "STOPPED"
        elif speed < self.START_SPEED_THRESHOLD and (
            speed_diff > self.SPEED_DIFF_THRESHOLD or speed_trend == "INCREASING"
        ):
            speed_state = "STARTING"
        elif (
            speed_diff > self.SPEED_DIFF_THRESHOLD
            or speed_trend == "INCREASING"
            or acceleration_cause == "DRIVER_ACCEL"
        ):
            speed_state = "ACCELERATING"
        elif (
            speed_diff < -self.SPEED_DIFF_THRESHOLD
            or speed_trend == "DECREASING"
            or acceleration_cause == "DRIVER_BRAKE"
            or brake > 0
        ):
            speed_state = "DECELERATING"
        else:
            speed_state = "CONSTANT"

        if blinker_l > 0:
            signal_state = "LEFT"
        elif blinker_r > 0:
            signal_state = "RIGHT"
        else:
            signal_state = "OFF"

        if gyro_z > DEFAULT_GYRO_THRESHOLD or visual_shift == "SHIFT_LEFT":
            trajectory_direction = "LEFT"
        elif gyro_z < -DEFAULT_GYRO_THRESHOLD or visual_shift == "SHIFT_RIGHT":
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

        if abs(gyro_z) > DEFAULT_GYRO_THRESHOLD * 1.8 or direction_change == "TURNING":
            turn_intensity = "HIGH"
        elif abs(gyro_z) > DEFAULT_GYRO_THRESHOLD * 0.8 or visual_shift != "NO_SHIFT":
            turn_intensity = "MEDIUM"
        else:
            turn_intensity = "LOW"

        if trajectory_relation == "CROSSING_LEFT":
            lane_crossing_state = "LEFT"
        elif trajectory_relation == "CROSSING_RIGHT":
            lane_crossing_state = "RIGHT"
        else:
            lane_crossing_state = "NONE"

        if intersection_detected == "YES":
            intersection_state = "YES"
        elif intersection_detected == "NO":
            intersection_state = "NO"
        else:
            intersection_state = "UNKNOWN"

        if speed < DEFAULT_SPEED_STOP_THRESHOLD and brake > 0:
            stop_likelihood = "HIGH"
        elif speed < self.START_SPEED_THRESHOLD:
            stop_likelihood = "MEDIUM"
        else:
            stop_likelihood = "LOW"

        return {
            "speed_state": speed_state,
            "signal_state": signal_state,
            "trajectory_direction": trajectory_direction,
            "turn_intensity": turn_intensity,
            "lane_crossing_state": lane_crossing_state,
            "intersection_state": intersection_state,
            "stop_likelihood": stop_likelihood,
            "consistency_check": consistency_check,
            "road_shape": road_shape,
        }

    def _build_edges(self, concepts: Dict[str, Any]) -> List[Dict[str, str]]:
        edges: List[Dict[str, str]] = []

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
        turn_intensity = concepts["turn_intensity"]
        stop_likelihood = concepts["stop_likelihood"]

        if stop_likelihood == "HIGH":
            add(4, 0.75, "速度≈0かつブレーキONで停止候補")
        elif stop_likelihood == "MEDIUM" and brake > 0:
            add(4, 0.35, "低速かつブレーキONで停止寄り")

        if speed_state == "STARTING":
            add(5, 0.75, "低速からの加速で発進候補")
        if speed_state == "ACCELERATING":
            add(2, 0.70, "速度差分またはトレンドが加速")
        if speed_state == "DECELERATING":
            add(3, 0.70, "速度差分またはトレンドが減速")
        if brake > 0 and speed > DEFAULT_SPEED_STOP_THRESHOLD:
            add(3, 0.20, "ブレーキONで減速候補を補強")

        if speed_state == "CONSTANT" and direction == "STRAIGHT":
            add(1, 0.65, "直進かつ速度変化が小さい")
        if signal == "OFF" and lane_crossing == "NONE" and intersection != "YES":
            add(1, 0.15, "旋回・車線変更の証拠が弱い")

        if direction == "LEFT":
            add(6, 0.35, "進行方向が左")
            add(8, 0.20, "左方向変化は左車線変更候補にもなる")
        elif direction == "RIGHT":
            add(7, 0.35, "進行方向が右")
            add(9, 0.20, "右方向変化は右車線変更候補にもなる")

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


class MacroGraphVerifier:
    """Graph-style verifier / reranker for macro 4-way classification."""

    STRONG_SUPPORT_THRESHOLD = 0.62
    STRONG_MARGIN_THRESHOLD = 0.10

    def build(
        self,
        sensor_data: Dict[str, Any],
        trajectory_features: Dict[str, Any],
        *,
        stage1_choice: str | None = None,
        stage2_choice: str | None = None,
    ) -> Dict[str, Any]:
        observations = self._extract_observations(sensor_data, trajectory_features)
        edges = self._build_edges(observations, stage1_choice=stage1_choice, stage2_choice=stage2_choice)
        label_support, label_reasons = self._score_labels(
            observations,
            stage1_choice=stage1_choice,
            stage2_choice=stage2_choice,
        )
        top_candidates = self._build_top_candidates(label_support, label_reasons)
        strong_candidate = self._select_strong_candidate(top_candidates)
        return {
            "observations": observations,
            "edges": edges,
            "label_support": label_support,
            "label_reasons": label_reasons,
            "top_candidates": top_candidates,
            "strong_candidate": strong_candidate,
        }

    def _extract_observations(
        self,
        sensor_data: Dict[str, Any],
        trajectory_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        speed = float(sensor_data.get("speed", 0.0) or 0.0)
        acc_x = float(sensor_data.get("acc_x", 0.0) or 0.0)
        gyro_z = float(sensor_data.get("gyro_z", 0.0) or 0.0)

        forward_distance = float(trajectory_features.get("final_x_m", 0.0) or 0.0)
        lateral_offset = float(trajectory_features.get("final_y_m", 0.0) or 0.0)
        visible_count = int(trajectory_features.get("visible_count", 0) or 0)
        trajectory_points = int(trajectory_features.get("trajectory_points", 0) or 0)

        denom = max(abs(forward_distance), 1.0)
        lateral_ratio = abs(lateral_offset) / denom
        point_like = (forward_distance < 2.5 and visible_count <= 1) or forward_distance < 1.5
        moving_straight_like = (
            speed >= 8.0
            and forward_distance >= 6.0
            and visible_count >= 4
            and lateral_ratio < 0.08
            and abs(gyro_z) < 0.08
        )

        if point_like or (speed < 2.0 and forward_distance < 4.0):
            motion_state = "STOPLIKE"
        elif acc_x < -0.35:
            motion_state = "DECEL"
        elif acc_x > 0.35:
            motion_state = "ACCEL"
        else:
            motion_state = "CRUISE"

        if lateral_offset > 1.2 or gyro_z > DEFAULT_GYRO_THRESHOLD:
            turn_sign = "LEFT"
        elif lateral_offset < -1.2 or gyro_z < -DEFAULT_GYRO_THRESHOLD:
            turn_sign = "RIGHT"
        else:
            turn_sign = "STRAIGHT"

        if lateral_ratio >= 0.14 or abs(gyro_z) >= 0.12:
            turn_strength = "HIGH"
        elif lateral_ratio >= 0.07 or abs(gyro_z) >= 0.05:
            turn_strength = "MEDIUM"
        else:
            turn_strength = "LOW"

        return {
            "speed": speed,
            "acc_x": acc_x,
            "gyro_z": gyro_z,
            "forward_distance": round(forward_distance, 3),
            "lateral_offset": round(lateral_offset, 3),
            "lateral_ratio": round(lateral_ratio, 3),
            "visible_count": visible_count,
            "trajectory_points": trajectory_points,
            "point_like": point_like,
            "moving_straight_like": moving_straight_like,
            "motion_state": motion_state,
            "turn_sign": turn_sign,
            "turn_strength": turn_strength,
        }

    def _build_edges(
        self,
        observations: Dict[str, Any],
        *,
        stage1_choice: str | None,
        stage2_choice: str | None,
    ) -> List[Dict[str, str]]:
        edges: List[Dict[str, str]] = []

        if observations["turn_strength"] in {"MEDIUM", "HIGH"} and observations["turn_sign"] != "STRAIGHT":
            edges.append(
                {
                    "source": "trajectory_geometry",
                    "target": "turn_hypothesis",
                    "type": "supports",
                    "reason": "横偏位またはヨーレートが回転系を支持",
                }
            )

        if observations["point_like"] or observations["motion_state"] == "STOPLIKE":
            edges.append(
                {
                    "source": "motion_state",
                    "target": "other_hypothesis",
                    "type": "supports",
                    "reason": "低速かつ短い軌道がその他系を支持",
                }
            )

        if observations["moving_straight_like"]:
            edges.append(
                {
                    "source": "trajectory_geometry",
                    "target": "straight_hypothesis",
                    "type": "supports",
                    "reason": "十分な前進距離と低い横偏位が直線移動を支持",
                }
            )

        if stage1_choice == "A" and observations["turn_strength"] == "HIGH":
            edges.append(
                {
                    "source": "stage1_choice",
                    "target": "turn_hypothesis",
                    "type": "contradicts",
                    "reason": "VLMは直線系だが幾何は強い回転を示す",
                }
            )

        if stage2_choice in {"B", "C"} and observations["turn_sign"] == "STRAIGHT":
            edges.append(
                {
                    "source": "stage2_choice",
                    "target": "straight_hypothesis",
                    "type": "contradicts",
                    "reason": "VLMは回転系だが幾何の左右方向が弱い",
                }
            )

        return edges

    def _score_labels(
        self,
        observations: Dict[str, Any],
        *,
        stage1_choice: str | None,
        stage2_choice: str | None,
    ) -> tuple[Dict[str, float], Dict[str, List[str]]]:
        label_support = {label: 0.0 for label in ["A", "B", "C", "D"]}
        label_reasons = {label: [] for label in ["A", "B", "C", "D"]}

        def add(label: str, amount: float, reason: str) -> None:
            label_support[label] = min(1.0, max(0.0, label_support[label] + amount))
            label_reasons[label].append(reason)

        turn_sign = observations["turn_sign"]
        turn_strength = observations["turn_strength"]
        motion_state = observations["motion_state"]
        speed = observations["speed"]
        forward_distance = observations["forward_distance"]
        lateral_ratio = observations["lateral_ratio"]
        point_like = observations["point_like"]
        moving_straight_like = observations["moving_straight_like"]

        if stage1_choice == "A":
            add("A", 0.32, "Stage1 が直線系を選択")
        elif stage1_choice == "N":
            add("B", 0.05, "Stage1 が非直線系を選択")
            add("C", 0.05, "Stage1 が非直線系を選択")
            add("D", 0.05, "Stage1 が非直線系を選択")

        if stage2_choice in {"B", "C", "D"}:
            add(stage2_choice, 0.22, f"Stage2 が {stage2_choice} を選択")

        if stage2_choice == "D" and moving_straight_like:
            add("A", 0.26, "Stage2 はその他だが幾何は直線移動")
            add("D", -0.10, "十分な前進距離がその他寄り判定と矛盾")

        if turn_sign == "LEFT":
            add("B", 0.28 if turn_strength == "HIGH" else 0.18, "軌道終点またはヨーレートが左方向")
            add("A", -0.08, "左方向変化は直線系と矛盾")
        elif turn_sign == "RIGHT":
            add("C", 0.28 if turn_strength == "HIGH" else 0.18, "軌道終点またはヨーレートが右方向")
            add("A", -0.08, "右方向変化は直線系と矛盾")
        else:
            add("A", 0.18, "左右方向変化が弱い")

        if turn_strength == "HIGH":
            add("B" if turn_sign == "LEFT" else "C" if turn_sign == "RIGHT" else "A", 0.18, "回転強度が高い")
        elif turn_strength == "LOW":
            add("A", 0.12, "回転強度が低い")

        if moving_straight_like:
            add("A", 0.32, "速度・可視点数・前進距離が直線移動を支持")
            add("D", -0.14, "十分に動いているためその他へは倒しにくい")

        if not point_like and forward_distance >= 5.0 and observations["visible_count"] >= 4:
            add("A", 0.15, "軌道が十分に伸びており直線移動と整合")

        if stage1_choice == "A" and turn_strength == "HIGH" and turn_sign == "LEFT":
            add("B", 0.24, "Stage1 は直線系だが幾何は強い左回転")
        elif stage1_choice == "A" and turn_strength == "HIGH" and turn_sign == "RIGHT":
            add("C", 0.24, "Stage1 は直線系だが幾何は強い右回転")

        if stage2_choice == "B" and turn_sign == "RIGHT" and turn_strength in {"MEDIUM", "HIGH"}:
            add("C", 0.22, "Stage2 は左回転だが幾何は右方向")
        elif stage2_choice == "C" and turn_sign == "LEFT" and turn_strength in {"MEDIUM", "HIGH"}:
            add("B", 0.22, "Stage2 は右回転だが幾何は左方向")

        if stage1_choice == "A" and point_like:
            add("A", 0.06, "Stage1 は直進系で軌道は極短、停止/発進寄り")

        if point_like:
            add("A", 0.22, "軌道が短く停止/発進を含む直進系寄り")

        if motion_state == "STOPLIKE":
            add("A", 0.20, "速度と軌道長から停止/発進寄り")
        elif motion_state == "DECEL":
            add("A", 0.10, "減速は直線系にも含まれる")
        elif motion_state in {"ACCEL", "CRUISE"} and turn_sign == "STRAIGHT":
            add("A", 0.14, "直進かつ巡航/加速")

        if speed < DEFAULT_SPEED_STOP_THRESHOLD and forward_distance < 3.0:
            add("A", 0.10, "低速かつ前進距離が短く停止/発進寄り")

        if lateral_ratio > 0.18 and turn_sign == "LEFT":
            add("B", 0.14, "横偏位比が大きく左回転寄り")
        elif lateral_ratio > 0.18 and turn_sign == "RIGHT":
            add("C", 0.14, "横偏位比が大きく右回転寄り")

        for label in label_support:
            label_support[label] = round(max(0.0, min(1.0, label_support[label])), 3)

        return label_support, label_reasons

    def _build_top_candidates(
        self,
        label_support: Dict[str, float],
        label_reasons: Dict[str, List[str]],
    ) -> List[Dict[str, Any]]:
        ranked = sorted(label_support.items(), key=lambda item: item[1], reverse=True)
        top_candidates = []
        for label, score in ranked[:4]:
            top_candidates.append(
                {
                    "macro_choice": label,
                    "score": score,
                    "reasons": label_reasons.get(label, [])[:3],
                }
            )
        return top_candidates

    def _select_strong_candidate(self, top_candidates: List[Dict[str, Any]]) -> Dict[str, Any] | None:
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
