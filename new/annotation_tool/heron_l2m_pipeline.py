# heron_l2m_pipeline.py
"""
Least-to-Most Prompting + Chain-of-Thought Pipeline for Heron
複数回のVLM推論を組み合わせて段階的な分析を行う

このパイプラインは、複雑な運転シーン分析を以下の3段階に分解します：
Level 1: 幾何学的分析（道路形状、軌道の関係）
Level 2: 物理法則との整合性検証（センサーデータとの照合）
Level 3: 最終的な行動分類（11カテゴリへの分類）
"""

import json
import logging
from typing import List, Dict, Any, Optional
from PIL import Image
import re

from config import GYRO_THRESHOLD, ACC_X_THRESHOLD, SPEED_STOP_THRESHOLD, FEW_SHOT_EXAMPLES
from driving_graph import DrivingConceptGraphBuilder
from vlm_runtime import FramePromptGenerator

logger = logging.getLogger(__name__)


class L2MCoTPipeline:
    """
    Least-to-Most Prompting + Chain-of-Thought Pipeline for Heron
    複数回のVLM推論を組み合わせて段階的な分析を行う
    """

    SPEED_DIFF_THRESHOLD = 0.5
    STRONG_SPEED_DIFF_THRESHOLD = 1.0
    SPEED_RATE_THRESHOLD = 0.6
    STABLE_SPEED_RATE_THRESHOLD = 0.25
    MAX_RELIABLE_TIME_DIFF_SEC = 4.0
    MAX_RELIABLE_MOTION_LOOKBACK_SEC = 10.0
    STOPLIKE_SPEED_THRESHOLD = 12.0
    LEVEL3_TOP_MARGIN_THRESHOLD = 0.10
    LEVEL3_STRONG_SCORE_THRESHOLD = 0.78
    
    def __init__(self, frame_generator: FramePromptGenerator):
        """
        Args:
            frame_generator:
                フレーム+プロンプトからテキストを返す生成器。
                GPU/processor の詳細はこの層に隠蔽する。
        """
        self.frame_generator = frame_generator
        self.conversation_history = []
        self.graph_builder = DrivingConceptGraphBuilder()

    def _trajectory_overlay_guidelines(self) -> str:
        return """【軌道オーバーレイの読み方】
- 赤い点と線は、自車の今から3秒先までの予測軌道です。
- 赤い点は近い未来から遠い未来へ時間順に並んでいます。
- まず赤い軌道だけを見て、将来運動が TURN / LANE CHANGE / STRAIGHT のどれに近いかを判断してください。
- 赤い軌道の曲がり具合は左右旋回の主要 evidence です。
- 赤い軌道の長さは速度の主要 evidence です。ほぼ点に見える場合は停止寄りに解釈してください。
- 次に、緑の参照線との相対位置から、同一車線内の追従か横方向逸脱かを判断してください。
- 背景映像は交差点・道路形状・車線文脈の確認に使い、運動の向きそのものは赤い軌道を優先してください。
- 4フレーム全体で一貫する解釈を選んでください。"""

    def _is_motion_feature_reliable(self, sensor_data: Dict[str, Any]) -> bool:
        explicit = sensor_data.get('motion_feature_reliable')
        if explicit is not None:
            return bool(explicit)

        timestamp_diff_sec = float(sensor_data.get('timestamp_diff_sec', 0.0) or 0.0)
        return 0.0 < timestamp_diff_sec <= self.MAX_RELIABLE_MOTION_LOOKBACK_SEC
    
    def analyze_with_l2m(
        self,
        frames: List[Image.Image],
        sensor_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        L2M戦略で3段階の推論を実行
        
        Args:
            frames: 4フレームのPIL Imageリスト
            sensor_data: センサーデータ辞書
        
        Returns:
            各レベルの推論結果と最終分類を含む辞書
        """
        logger.info("=== Starting L2M+CoT Analysis ===")
        logger.info(f"Frames: {len(frames)}, Sensor data: {sensor_data}")

        # センサーヒントを事前計算（案2-1）
        sensor_hints = self._get_sensor_hints(sensor_data)
        if sensor_hints:
            logger.info(f"Sensor hints: {sensor_hints}")

        # Level 1: Geometric analysis
        logger.info("--- Level 1: Geometric Analysis ---")
        level1_result = self._level1_geometry(frames, sensor_data)
        logger.info(f"Level 1 完了: {level1_result}")

        # Level 2: Physics consistency check
        logger.info("--- Level 2: Physics Consistency Check ---")
        level2_result = self._level2_physics(frames, sensor_data, level1_result)
        logger.info(f"Level 2 完了: {level2_result}")

        # Graph: concept extraction + label support
        logger.info("--- Graph: Concept Structuring ---")
        graph_result = self.graph_builder.build(level1_result, level2_result, sensor_data)
        logger.info(f"Graph concepts: {graph_result['concepts']}")
        logger.info(f"Graph top candidates: {graph_result['top_candidates']}")

        # Level 3: Final classification
        logger.info("--- Level 3: Final Classification ---")
        level3_result = self._level3_classification(
            frames, sensor_data, level1_result, level2_result, graph_result, sensor_hints
        )
        logger.info(f"Level 3 完了: {level3_result}")

        pipeline_error = self._extract_pipeline_error(
            level1_result, level2_result, level3_result
        )
        if pipeline_error:
            logger.warning("[L2M+CoT] Generation error detected in pipeline: %s", pipeline_error)
            final_category = None
            confidence = 0.0
        else:
            # 後処理ルール適用（案4）
            raw_category = level3_result.get('category_id', 0)
            final_category = self._post_process_label(raw_category, sensor_data, graph_result)
            if final_category is None:
                # 旋回検出の矛盾 → 保守的に元の予測を維持（再推論は実装コスト高のため）
                logger.warning("[PostProcess] Contradiction detected (gyro>threshold but predicted constant speed). Keeping original.")
                final_category = raw_category
            if final_category != raw_category:
                logger.info(f"[PostProcess] Label corrected: {raw_category} → {final_category}")
            confidence = level3_result.get('confidence', 0.5)

        logger.info("=== L2M+CoT Analysis Complete ===")

        return {
            'level1': level1_result,
            'level2': level2_result,
            'graph': graph_result,
            'level3': level3_result,
            'final_category': final_category,
            'confidence': confidence,
            'status': 'failed' if pipeline_error else 'success',
            'error': pipeline_error,
        }
    
    def _level1_geometry(self, frames: List[Image.Image], sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Level 1: Analyze road geometry and trajectory relationship
        
        This step analyzes only visual geometric features.
        No sensor data is used, only objective facts are extracted.
        """
        prompt = f"""あなたは自動運転データの分析エキスパートです。以下の画像は車両の前方カメラから撮影された4枚の連続フレームです。

【画像の説明】
- 赤い線: 車両の予測軌道（これから進む予定の経路）
- 緑の線: 現在の車線区分線または車線中心

{self._trajectory_overlay_guidelines()}

【タスク】
以下の質問に段階的に答えてください。

[ステップ0: 軌道そのものの将来運動キュー]
赤い軌道だけを見て、将来の運動キューを分類してください。
- STRAIGHT_CUE: 赤い軌道が前方へほぼ直進し、横方向のずれが小さい
- LEFT_TURN_CUE: 赤い軌道が連続的に左へ曲がる
- RIGHT_TURN_CUE: 赤い軌道が連続的に右へ曲がる
- LEFT_LANE_CHANGE_CUE: 赤い軌道が左へ横移動するが、交差点での回頭ほどは曲がらない
- RIGHT_LANE_CHANGE_CUE: 赤い軌道が右へ横移動するが、交差点での回頭ほどは曲がらない
- AMBIGUOUS: どれとも言い切れない

重要:
- 背景の遠近感や道路の見え方だけで LEFT_TURN_CUE / RIGHT_TURN_CUE を選ばないでください。赤い軌道の形そのものだけを見てください。
- 4フレームで一貫していない、または赤い軌道がほぼ直進に見える場合は、無理に旋回・車線変更を選ばず AMBIGUOUS または STRAIGHT_CUE を選んでください。
- 左右の判断根拠が弱い場合は AMBIGUOUS を優先してください。

[ステップ1: 道路形状の特定]
道路は直線ですか、それとも左右にカーブしていますか？
- STRAIGHT（直線）: 道路がまっすぐ続いている
- CURVE_LEFT（左カーブ）: 道路が左方向に曲がっている
- CURVE_RIGHT（右カーブ）: 道路が右方向に曲がっている

[ステップ2: 軌道と車線の関係]
赤い軌道（予測進路）と緑の車線区分線の位置関係を分析してください。
- PARALLEL（平行）: 赤い軌道が緑の線に沿って平行に進んでいる
- CROSSING_LEFT（左へ交差）: 赤い軌道が緑の線を左側に越えている
- CROSSING_RIGHT（右へ交差）: 赤い軌道が緑の線を右側に越えている

重要: 道路がカーブしている場合でも、赤い軌道が緑の線と平行を保っていれば「PARALLEL」です。カーブに沿った走行と車線変更を混同しないでください。

[ステップ3: 道路の勾配]
道路は上り坂、下り坂、それとも平坦ですか？
- UPHILL（上り坂）
- DOWNHILL（下り坂）
- FLAT（平坦）

[ステップ4: 方向転換の兆候（重要）]
映像から、車両が方向転換している可能性を判断してください。

(a) 交差点または角が前方に見えるか、または通過中ですか？
- YES（交差点あり）
- NO（交差点なし）
- UNCERTAIN（映像だけでは断定できない）

(b) 車両の進行方向が直進から変化していますか？（景色の流れ方向、地平線の傾きで判断）
- TURNING（旋回中）
- STRAIGHT（直進）

(c) フロントカメラの映像で、景色が左右にシフトしていますか？
- SHIFT_LEFT（左へシフト）
- SHIFT_RIGHT（右へシフト）
- NO_SHIFT（シフトなし）

【出力形式】
以下のJSON形式でのみ回答してください。推論過程も「reasoning」フィールドに簡潔に記述してください。

```json
{{
  "reasoning": "道路形状と軌道の関係についての分析結果を簡潔に説明",
  "trajectory_motion_cue": "RIGHT_TURN_CUE",
  "road_shape": "STRAIGHT",
  "trajectory_relation": "PARALLEL",
  "slope": "FLAT",
  "intersection_detected": "UNCERTAIN",
  "direction_change": "STRAIGHT",
  "visual_shift": "NO_SHIFT"
}}
```"""
        
        try:
            raw_output = self._generate_with_frames(frames, prompt)
            parsed = self._parse_json_output(raw_output)
            
            if parsed is None:
                # Default values when parsing fails
                logger.warning(f"Level 1 JSON parse failed. Raw output: {raw_output[:300]}")
                parsed = {
                    'reasoning': raw_output[:500],
                    'trajectory_motion_cue': 'AMBIGUOUS',
                    'road_shape': 'STRAIGHT',
                    'trajectory_relation': 'PARALLEL',
                    'slope': 'FLAT',
                    'intersection_detected': 'NO',
                    'direction_change': 'STRAIGHT',
                    'visual_shift': 'NO_SHIFT'
                }
            parsed = self._apply_deterministic_trajectory_features(parsed, sensor_data)
            return self._normalize_level1_result(parsed)

        except Exception as e:
            logger.error(f"Level 1 推論エラー: {e}", exc_info=True)
            fallback = {
                'reasoning': f"Error: {str(e)}",
                'trajectory_motion_cue': 'AMBIGUOUS',
                'road_shape': 'STRAIGHT',
                'trajectory_relation': 'PARALLEL',
                'slope': 'FLAT',
                'intersection_detected': 'NO',
                'direction_change': 'STRAIGHT',
                'visual_shift': 'NO_SHIFT'
            }
            fallback = self._apply_deterministic_trajectory_features(fallback, sensor_data)
            return self._normalize_level1_result(fallback)
    
    def _level2_physics(
        self,
        frames: List[Image.Image],
        sensor_data: Dict[str, Any],
        level1_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Level 2: Verify physics consistency
        
        Cross-check Level 1 visual analysis with sensor data
        to detect physical contradictions.
        """
        # Get sensor data with default values
        speed = sensor_data.get('speed', 0)
        acc_x = sensor_data.get('acc_x', 0)
        acc_y = sensor_data.get('acc_y', 0)
        gyro_z = sensor_data.get('gyro_z', 0)
        brake = sensor_data.get('brake', 0)
        blinker_r = sensor_data.get('blinker_r', 0)
        blinker_l = sensor_data.get('blinker_l', 0)
        speed_diff = float(sensor_data.get('speed_diff', 0.0) or 0.0)
        timestamp_diff_sec = float(sensor_data.get('timestamp_diff_sec', 0.0) or 0.0)
        speed_change_rate = float(sensor_data.get('speed_change_rate', 0.0) or 0.0)
        motion_feature_reliable = self._is_motion_feature_reliable(sensor_data)

        brake_status = 'ON' if brake > 0 else 'OFF'
        blinker_status = '右ウィンカーON' if blinker_r > 0 else ('左ウィンカーON' if blinker_l > 0 else 'OFF')
        motion_summary = self._format_motion_summary(sensor_data)
        motion_reliability_summary = (
            "直前サンプルとの差分は信頼できる"
            if motion_feature_reliable else
            f"直前サンプルとの差分は {timestamp_diff_sec:.2f} 秒離れており、加減速判断では弱い evidence とみなす"
        )

        # ジャイロ閾値超過時の警告メッセージ（案1-2）
        gyro_warning = ""
        if abs(gyro_z) > GYRO_THRESHOLD:
            direction = "左" if gyro_z > 0 else "右"
            gyro_warning = f"\n⚠️ ジャイロセンサーが旋回を検出しています（gyro_z={gyro_z:.3f} rad/s）。{direction}への旋回を考慮してください。"

        # 横加速度による旋回の補足警告
        acc_x_warning = ""
        if abs(acc_x) > ACC_X_THRESHOLD:
            direction = "左" if acc_x > 0 else "右"
            acc_x_warning = f"\n⚠️ 横加速度あり（acc_x={acc_x:.3f} m/s²）。{direction}への旋回・車線変更の可能性があります。"

        prompt = f"""あなたは自動運転データの分析エキスパートです。Level 1の幾何学的分析結果と、実際の車両センサーデータを照合して、物理法則との整合性を検証してください。

【Level 1の分析結果】
- 軌道の将来運動キュー: {level1_result.get('trajectory_motion_cue', 'UNKNOWN')}
- 道路形状: {level1_result.get('road_shape', 'UNKNOWN')}
- 軌道と車線の関係: {level1_result.get('trajectory_relation', 'UNKNOWN')}
- 道路の勾配: {level1_result.get('slope', 'UNKNOWN')}
- 交差点検出: {level1_result.get('intersection_detected', 'UNKNOWN')}
- 方向変化: {level1_result.get('direction_change', 'UNKNOWN')}
- 映像シフト: {level1_result.get('visual_shift', 'UNKNOWN')}

【センサーデータ】
- 速度: {speed:.1f} km/h
- 前後加速度: {acc_x:.2f} m/s²
- 横加速度: {acc_y:.2f} m/s²
- ジャイロ（旋回速度）: {gyro_z:.3f} rad/s（正=左旋回、負=右旋回）
- ブレーキ: {brake_status}
- ウィンカー: {blinker_status}
- 直前サンプルとの差分: {speed_diff:+.1f} km/h
- サンプル間隔: {timestamp_diff_sec:.2f} s
- 時間正規化した速度変化率: {speed_change_rate:+.2f} km/h/s
- 差分特徴の信頼性: {motion_reliability_summary}
- センサー由来の運動要約: {motion_summary}{gyro_warning}{acc_x_warning}

{self._trajectory_overlay_guidelines()}

【タスク】
以下の分析を段階的に行ってください。

[ステップ1: 加速度の原因特定]
センサーが示す加速度は何が原因ですか？
- DRIVER_ACCEL（ドライバーの加速操作）: アクセルを踏んで加速している
- DRIVER_BRAKE（ドライバーの減速操作）: ブレーキを踏んで減速している
- GRAVITY（重力の影響）: 坂道による自然な加減速
- CURVE（遠心力）: カーブによる横方向の加速度
- MIXED（複合要因）: 複数の要因が組み合わさっている

重要: 下り坂で正の加速度が観測される場合、それはアクセル操作ではなく重力加速度である可能性があります。Level 1の勾配情報と照らし合わせて判断してください。

[ステップ2: 速度トレンドの予測]
今後3秒間で、車両の速度はどのように変化すると予想されますか？
- INCREASING（増加）: 速度が上がっていく
- DECREASING（減少）: 速度が下がっていく
- STABLE（安定）: 速度がほぼ一定に保たれる

[ステップ3: 整合性チェック]
視覚情報（映像）とセンサーデータの間に矛盾はありませんか？
- CONSISTENT（整合している）: 視覚とセンサーの情報が論理的に一致
- INCONSISTENT（矛盾している）: 視覚とセンサーの情報に不整合がある

【出力形式】
以下のJSON形式でのみ回答してください。

```json
{{
  "reasoning": "センサーデータと視覚情報の関係についての分析",
  "acceleration_cause": "DRIVER_ACCEL",
  "speed_trend": "STABLE",
  "consistency_check": "CONSISTENT"
}}
```"""
        
        try:
            raw_output = self._generate_with_frames(frames, prompt)
            parsed = self._parse_json_output(raw_output)
            
            if parsed is None:
                logger.warning(f"Level 2 JSON parse failed. Raw output: {raw_output[:300]}")
                return self._normalize_level2_result(sensor_data, {
                    'reasoning': raw_output[:500],
                    'acceleration_cause': 'MIXED',
                    'speed_trend': 'STABLE',
                    'consistency_check': 'CONSISTENT'
                })
            
            return self._normalize_level2_result(sensor_data, parsed)
            
        except Exception as e:
            logger.error(f"Level 2 推論エラー: {e}", exc_info=True)
            return {
                'reasoning': f"Error: {str(e)}",
                'acceleration_cause': 'MIXED',
                'speed_trend': 'STABLE',
                'consistency_check': 'CONSISTENT'
            }
    
    def _level3_classification(
        self,
        frames: List[Image.Image],
        sensor_data: Dict[str, Any],
        level1_result: Dict[str, Any],
        level2_result: Dict[str, Any],
        graph_result: Dict[str, Any],
        sensor_hints: str = ""
    ) -> Dict[str, Any]:
        """
        Level 3: Final action classification
        
        Integrate Level 1 and Level 2 analysis results
        to classify into one of 11 driving action categories.
        """
        # CRITICAL: Check for stopped vehicle first
        speed = sensor_data.get('speed', 0)
        brake = sensor_data.get('brake', 0)
        speed_diff = float(sensor_data.get('speed_diff', 0.0) or 0.0)
        speed_change_rate = float(sensor_data.get('speed_change_rate', 0.0) or 0.0)
        motion_observation = self._build_motion_observation(sensor_data)
        
        # 完全停止に見えても、直前まで強い減速が続いている場合は減速との境界事例として扱う
        if (
            speed < SPEED_STOP_THRESHOLD
            and brake > 0
            and motion_observation.get('trend') != 'DECREASING'
        ):
            logger.info("[Level 3] STOP detected: Speed=0, Brake=ON → Forcing category 4 (停止)")
            return {
                'final_reasoning': 'Vehicle is completely stopped (Speed=0km/h) with brake engaged.',
                'category_id': 4,
                'category_name': '停止',
                'confidence': 0.99
            }
        
        gyro_z = sensor_data.get('gyro_z', 0)
        blinker_r = sensor_data.get('blinker_r', 0)
        blinker_l = sensor_data.get('blinker_l', 0)
        blinker_status = '右ウィンカーON' if blinker_r > 0 else ('左ウィンカーON' if blinker_l > 0 else 'OFF')

        speed_diff_text = ""
        if speed_diff is not None:
            if speed_diff > 0.5:
                speed_diff_text = f"\n- 速度変化量: +{speed_diff:.1f} km/h（加速傾向）"
            elif speed_diff < -0.5:
                speed_diff_text = f"\n- 速度変化量: {speed_diff:.1f} km/h（減速傾向）"

        sensor_hints_section = f"\n\n◆ センサーヒント（事前フィルタリング）\n{sensor_hints}" if sensor_hints else ""
        motion_section = (
            "\n\n◆ センサー由来の運動要約\n"
            f"- direct_motion_trend: {motion_observation.get('trend', 'UNKNOWN')}\n"
            f"- direct_motion_cause: {motion_observation.get('cause', 'UNKNOWN')}\n"
            f"- direct_motion_reason: {motion_observation.get('reason', '情報なし')}\n"
            f"- motion_feature_reliable: {motion_observation.get('reliable', False)}"
        )
        graph_section = (
            f"\n\n◆ Graph 概念要約\n{graph_result.get('summary_for_prompt', '')}"
            f"\n\n◆ Graph 上位候補\n{graph_result.get('candidate_summary', '')}"
        )
        strong_candidate = graph_result.get("strong_candidate")
        strong_candidate_rule = ""
        if strong_candidate:
            strong_candidate_rule = (
                f"\n- Graph の強候補: "
                f"{strong_candidate['category_id']}（{strong_candidate['category_name']}）"
                f" score={strong_candidate['score']:.2f}"
            )

        prompt = f"""あなたは自動運転データの分析エキスパートです。これまでの分析結果を統合して、車両の運転行動を最終的に分類してください。

{FEW_SHOT_EXAMPLES}
【これまでの分析結果】

◆ Level 1: 幾何学的分析
- 軌道の将来運動キュー: {level1_result.get('trajectory_motion_cue', 'N/A')}
- 軌道キュー整合性: {level1_result.get('trajectory_motion_cue_consistency', 'N/A')}
- 軌道キュー信頼度: {level1_result.get('trajectory_motion_cue_confidence', 0.0):.2f}
- 軌道キュー生値: {level1_result.get('raw_trajectory_motion_cue', 'N/A')}
- 軌道特徴ソース: {level1_result.get('trajectory_feature_source', 'N/A')}
- 軌道の見え方による速度状態: {level1_result.get('trajectory_visual_speed_state', 'N/A')}
- 軌道が点状か: {level1_result.get('trajectory_point_like', 'N/A')}
- 軌道の長さ状態: {level1_result.get('trajectory_length_state', 'N/A')}
- 軌道の横偏位[m]: {level1_result.get('trajectory_lateral_offset_m', 0.0):.2f}
- 軌道の曲がり量[rad]: {level1_result.get('trajectory_heading_delta_rad', 0.0):.2f}
- 道路形状: {level1_result.get('road_shape')}
- 軌道と車線の関係: {level1_result.get('trajectory_relation')}
- 道路の勾配: {level1_result.get('slope')}
- 交差点検出: {level1_result.get('intersection_detected', 'N/A')}
- 方向変化: {level1_result.get('direction_change', 'N/A')}
- 映像シフト: {level1_result.get('visual_shift', 'N/A')}

◆ Level 2: 物理法則との整合性
- 加速度の原因: {level2_result.get('acceleration_cause')}
- 速度トレンド: {level2_result.get('speed_trend')}
{motion_section}{graph_section}

◆ センサーデータ
- 現在速度: {speed:.1f} km/h{speed_diff_text}
- 時間正規化した速度変化率: {speed_change_rate:+.2f} km/h/s
- ジャイロ（旋回速度）: {gyro_z:.3f} rad/s（正=左旋回、負=右旋回）
- ブレーキ: {'ON' if brake > 0 else 'OFF'}
- ウィンカー: {blinker_status}{sensor_hints_section}

{self._trajectory_overlay_guidelines()}

【必ず以下の11クラスから1つ選択してください】
0: その他
1: 等速走行
2: 加速
3: 減速
4: 停止
5: 発進
6: 左折
7: 右折
8: 車線変更（左）
9: 車線変更（右）
10: 転回（Uターン）

【分類ルール - 必ず守ること】

1. **停止判定（最優先）**:
   - 速度が0 km/hでブレーキがONの場合 → 必ずカテゴリ4（停止）

2. **発進判定**:
   - 速度が5 km/h未満で速度トレンドがINCREASINGの場合 → カテゴリ5（発進）
   - または、低速かつ stop-like な軌道から加速へ移る場合も発進候補として扱う

3. **旋回・車線変更の判定（等速走行より優先）**:
   - |gyro_z| > 0.1 rad/s の場合 → カテゴリ6/7/8/9/10 を積極的に検討
   - ウィンカーONの場合 → 旋回または車線変更の強い証拠
   - 交差点検出=YES かつ 映像シフトあり → カテゴリ6または7
   - 交差点検出=UNCERTAIN の場合、左右折と車線変更の両方を候補に残しつつ Graph の順位を優先する
   - 交差点なし かつ 車線シフト → カテゴリ8または9

4. **加速・減速の判定**:
   - 加速度の原因がDRIVER_BRAKEの場合 → カテゴリ3（減速）
   - 速度変化量が大きい場合 → カテゴリ2（加速）またはカテゴリ3（減速）
   - `direct_motion_trend` が INCREASING / DECREASING のときは、その符号を優先して整合するカテゴリを選ぶ

5. **Graph 概念を必ず確認**:
   - `Graph 上位候補` は中間概念から構成された候補である
   - 特に、等速走行(1)や停止(4)を選ぶ前に Graph 候補との矛盾がないか確認する
   - 右左折・車線変更・減速の候補が Graph に強く出ている場合は優先的に検討する{strong_candidate_rule}
   - Graph 1位と2位の score 差が明確な場合は、1位を不用意に覆さない
   - Graph 1位が車線変更で intersection_detected が NO または UNCERTAIN のときは、左右折へ戻さない

6. **等速走行（1）は最後の手段**:
   - 旋回・車線変更（6,7,8,9,10）の可能性を除外した後に選択
   - 速度変化（2,3）の可能性を除外した後に選択
   - 停止・発進（4,5）の可能性を除外した後に選択

7. **停止（4）の抑制ルール**:
   - 速度が 1 km/h を超えていて、かつブレーキが OFF の場合は停止（4）を選ばない
   - Graph や direct_motion_trend が減速・旋回を示している場合、停止よりそちらを優先する

8. **軌道オーバーレイ優先ルール**:
   - まず `trajectory_motion_cue` を見て、将来運動が TURN / LANE CHANGE / STRAIGHT のどれかを決める
   - `trajectory_feature_source=PROJECTED_RED_TRAJECTORY` のときは、赤い軌道の曲がり具合を左右旋回の主要 evidence とみなす
   - `trajectory_point_like=True` のときは、赤い軌道は停止寄りの visual evidence とみなす
   - 赤い軌道が連続的に曲がるなら TURN を優先する
   - 赤い軌道が横移動主体で交差点が弱いなら LANE CHANGE を優先する
   - 背景だけで曲がって見えても、赤い軌道が直進なら TURN にしない
   - `trajectory_motion_cue_consistency=CONTRADICTED` または `trajectory_motion_cue_confidence<0.5` の場合、その cue を単独で信用しない

【タスク】
上記の分析結果とルールを総合的に考慮して、最も適切なカテゴリを1つ選択してください。
推論過程を「final_reasoning」に記述し、カテゴリID、カテゴリ名、信頼度を出力してください。

以下のJSON形式でのみ回答してください：
```json
{{
  "final_reasoning": "推論内容",
  "category_id": -1,
  "category_name": "カテゴリ名",
  "confidence": 0.0
}}
```"""
        
        try:
            raw_output = self._generate_with_frames(frames, prompt)
            parsed = self._parse_json_output(raw_output)
            
            if parsed is None or 'category_id' not in parsed:
                logger.warning(f"Level 3 JSON parse failed. Raw output: {raw_output[:300]}")
                category_id = self._extract_category_from_text(raw_output)
                return {
                    'final_reasoning': raw_output[:500],
                    'category_id': category_id,
                    'category_name': self._get_category_name(category_id),
                    'confidence': 0.5
                }

            # category_id=-1 はテンプレートのコピーなのでテキストから再抽出
            if parsed.get('category_id') == -1:
                logger.warning(f"Level 3 returned template placeholder (category_id=-1). Falling back to text extraction. Raw: {raw_output[:300]}")
                category_id = self._extract_category_from_text(raw_output)
                return {
                    'final_reasoning': raw_output[:500],
                    'category_id': category_id,
                    'category_name': self._get_category_name(category_id),
                    'confidence': 0.4
                }

            # Supplement category name if not included
            if 'category_name' not in parsed:
                parsed['category_name'] = self._get_category_name(parsed['category_id'])

            # Default value if confidence not included
            if 'confidence' not in parsed:
                parsed['confidence'] = 0.7

            return self._apply_level3_graph_constraints(
                parsed,
                sensor_data=sensor_data,
                level1_result=level1_result,
                graph_result=graph_result,
            )
            
        except Exception as e:
            logger.error(f"Level 3 推論エラー: {e}", exc_info=True)
            return {
                'final_reasoning': f"Error: {str(e)}",
                'category_id': 0,
                'category_name': 'その他',
                'confidence': 0.0
            }

    def _apply_level3_graph_constraints(
        self,
        parsed: Dict[str, Any],
        *,
        sensor_data: Dict[str, Any],
        level1_result: Dict[str, Any],
        graph_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        constrained = dict(parsed)
        top_candidates = graph_result.get('top_candidates', []) if graph_result else []
        if not top_candidates:
            return constrained

        concepts = graph_result.get('concepts', {}) if graph_result else {}
        label_support = graph_result.get('label_support', {}) if graph_result else {}
        predicted_label = int(constrained.get('category_id', 0))
        top = top_candidates[0]
        top_label = int(top['category_id'])
        top_score = float(top['score'])
        second_score = float(top_candidates[1]['score']) if len(top_candidates) > 1 else 0.0
        predicted_score = float(label_support.get(predicted_label, 0.0))
        margin = top_score - second_score
        intersection_state = concepts.get('intersection_state', level1_result.get('intersection_detected', 'UNKNOWN'))
        stop_likelihood = concepts.get('stop_likelihood', 'LOW')
        trajectory_point_like = bool(concepts.get('trajectory_point_like', False))
        motion_feature_reliable = bool(concepts.get('motion_feature_reliable', False))
        speed = float(sensor_data.get('speed', 0.0) or 0.0)

        if (
            top_label != predicted_label
            and top_score >= self.LEVEL3_STRONG_SCORE_THRESHOLD
            and margin >= 0.18
        ):
            return self._override_level3_result(
                constrained,
                top_label,
                reason=(
                    f"Graph strong candidate {top_label}:{top['category_name']} "
                    f"(score={top_score:.2f}, margin={margin:.2f})"
                ),
                confidence=max(float(constrained.get('confidence', 0.0) or 0.0), min(0.99, top_score)),
            )

        if (
            predicted_label in {6, 7}
            and top_label in {8, 9}
            and intersection_state in {'NO', 'UNCERTAIN'}
            and top_score >= 0.65
            and (top_score - predicted_score) >= self.LEVEL3_TOP_MARGIN_THRESHOLD
        ):
            return self._override_level3_result(
                constrained,
                top_label,
                reason=(
                    f"lane-change constraint: intersection={intersection_state}, "
                    f"Graph prefers {top_label}:{top['category_name']} over turn"
                ),
                confidence=max(float(constrained.get('confidence', 0.0) or 0.0), min(0.95, top_score)),
            )

        if (
            predicted_label == 1
            and top_label in {3, 4}
            and speed <= self.STOPLIKE_SPEED_THRESHOLD
            and stop_likelihood in {'HIGH', 'MEDIUM'}
            and trajectory_point_like
            and top_score >= 0.55
        ):
            target_label = 4 if stop_likelihood == 'HIGH' else top_label
            return self._override_level3_result(
                constrained,
                target_label,
                reason=(
                    f"low-speed stop-like trajectory: stop_likelihood={stop_likelihood}, "
                    f"Graph top={top_label}:{top['category_name']}"
                ),
                confidence=max(float(constrained.get('confidence', 0.0) or 0.0), min(0.95, top_score)),
            )

        if (
            predicted_label in {1, 2}
            and top_label == 5
            and top_score >= 0.60
        ):
            return self._override_level3_result(
                constrained,
                5,
                reason=f"Graph prefers start event (score={top_score:.2f})",
                confidence=max(float(constrained.get('confidence', 0.0) or 0.0), min(0.95, top_score)),
            )

        if (
            predicted_label in {6, 7}
            and top_label in {1, 2, 3, 4, 5}
            and top_score >= 0.65
            and (top_score - predicted_score) >= self.LEVEL3_TOP_MARGIN_THRESHOLD
        ):
            if top_label in {2, 3} or not motion_feature_reliable or intersection_state != 'YES':
                return self._override_level3_result(
                    constrained,
                    top_label,
                    reason=(
                        f"Graph prefers non-turn event {top_label}:{top['category_name']} "
                        f"over turn (intersection={intersection_state}, motion_reliable={motion_feature_reliable})"
                    ),
                    confidence=max(float(constrained.get('confidence', 0.0) or 0.0), min(0.95, top_score)),
                )

        return constrained

    def _override_level3_result(
        self,
        parsed: Dict[str, Any],
        category_id: int,
        *,
        reason: str,
        confidence: float,
    ) -> Dict[str, Any]:
        updated = dict(parsed)
        previous_label = updated.get('category_id')
        previous_name = updated.get('category_name', self._get_category_name(previous_label))
        updated['category_id'] = category_id
        updated['category_name'] = self._get_category_name(category_id)
        updated['confidence'] = round(confidence, 2)
        base_reasoning = updated.get('final_reasoning', '')
        adjustment = (
            f"post_constraint: {previous_label}:{previous_name} -> "
            f"{category_id}:{updated['category_name']} ({reason})"
        )
        updated['final_reasoning'] = (
            f"{base_reasoning} | {adjustment}" if base_reasoning else adjustment
        )
        return updated
    
    def _generate_with_frames(self, frames: List[Image.Image], prompt: str) -> str:
        """
        Execute Heron inference with frames and prompt
        
        Leverages existing HeronAnnotator internal methods
        for multi-frame input inference.
        """
        try:
            output_text = self.frame_generator.generate_text(
                frames,
                prompt,
                max_new_tokens=512,  # 長いプロンプト対応のため300→512に増加
                do_sample=False,  # Greedy decoding for reliability
            )
            logger.debug(f"Generated output (first 200 chars): {output_text[:200]}")
            
            return output_text
            
        except Exception as e:
            logger.error(f"Generation error: {e}", exc_info=True)
            raise
    
    def _parse_json_output(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Extract and parse JSON from VLM output
        
        VLM output may contain explanatory text,
        so we search for JSON blocks and parse them.
        """
        # 1. ```json ... ```の形式を探す
        json_code_block_pattern = r'```json\s*(.*?)\s*```'
        matches = re.findall(json_code_block_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode failed for code block: {e}")
                continue
        
        # 2. 中括弧で囲まれた部分を探す（貪欲でないマッチング）
        json_object_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_object_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError as e:
                logger.debug(f"JSON decode failed for object: {e}")
                continue
        
        # 3. テキスト全体をJSONとしてパース
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from output. Raw text: {text[:300]}...")
            return None
    
    def _extract_category_from_text(self, text: str) -> int:
        """
        Heuristically extract category ID from text when JSON parsing fails
        """
        # Search for category ID pattern
        category_pattern = r'category[_\s]*id["\s:]*(\d+)'
        match = re.search(category_pattern, text, re.IGNORECASE)
        if match:
            try:
                cat_id = int(match.group(1))
                if 0 <= cat_id <= 10:
                    return cat_id
            except ValueError:
                pass
        
        # Keyword-based estimation
        text_lower = text.lower()
        if '車線変更' in text or 'lane change' in text_lower:
            if '左' in text or 'left' in text_lower:
                return 8
            elif '右' in text or 'right' in text_lower:
                return 9
        elif '転回' in text or 'u-turn' in text_lower or 'uturn' in text_lower:
            return 10
        elif '左折' in text or 'left turn' in text_lower:
            return 6
        elif '右折' in text or 'right turn' in text_lower:
            return 7
        elif '停止' in text or 'stop' in text_lower:
            return 4
        elif '発進' in text or 'start' in text_lower:
            return 5
        elif '加速' in text or 'accel' in text_lower:
            return 2
        elif '減速' in text or 'decel' in text_lower or 'brake' in text_lower:
            return 3
        elif '等速' in text or 'constant speed' in text_lower:
            return 1
        
        # Default is other
        return 0

    def _extract_pipeline_error(
        self,
        level1_result: Dict[str, Any],
        level2_result: Dict[str, Any],
        level3_result: Dict[str, Any]
    ) -> Optional[str]:
        """
        Detect hard generation failures that were converted into default values.
        """
        checks = [
            ("level1.reasoning", level1_result.get("reasoning")),
            ("level2.reasoning", level2_result.get("reasoning")),
            ("level3.final_reasoning", level3_result.get("final_reasoning")),
        ]

        for field_name, value in checks:
            if isinstance(value, str) and value.startswith("Error:"):
                return f"{field_name}: {value}"

        return None

    def _apply_deterministic_trajectory_features(
        self,
        level1_result: Dict[str, Any],
        sensor_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Override low-level geometric fields with deterministic red-line features."""
        deterministic_mapping = {
            'trajectory_motion_cue_deterministic': 'trajectory_motion_cue',
            'trajectory_relation_deterministic': 'trajectory_relation',
            'direction_change_deterministic': 'direction_change',
            'visual_shift_deterministic': 'visual_shift',
        }

        merged = dict(level1_result)
        applied = {}
        for sensor_key, level1_key in deterministic_mapping.items():
            value = sensor_data.get(sensor_key)
            if value in {None, ""}:
                continue
            merged[f'vlm_{level1_key}'] = merged.get(level1_key)
            merged[level1_key] = value
            applied[level1_key] = value

        for sensor_key in [
            'trajectory_feature_source',
            'trajectory_path_length_m',
            'trajectory_visible_length_px',
            'trajectory_visible_point_count',
            'trajectory_point_like',
            'trajectory_lateral_offset_m',
            'trajectory_forward_extent_m',
            'trajectory_heading_delta_rad',
            'trajectory_curvature_score',
            'trajectory_length_state',
            'trajectory_visual_speed_state',
        ]:
            if sensor_key in sensor_data:
                merged[sensor_key] = sensor_data.get(sensor_key)

        if applied:
            reasoning = merged.get('reasoning', '')
            override_text = ", ".join(f"{k}={v}" for k, v in applied.items())
            merged['reasoning'] = (
                f"{reasoning} | deterministic_red_trajectory: {override_text}"
                if reasoning else f"deterministic_red_trajectory: {override_text}"
            )

        return merged

    def _normalize_level1_result(self, level1_result: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(level1_result)
        raw_cue = normalized.get('trajectory_motion_cue')
        trajectory_relation = normalized.get('trajectory_relation', 'PARALLEL')
        visual_shift = normalized.get('visual_shift', 'NO_SHIFT')
        direction_change = normalized.get('direction_change', 'STRAIGHT')
        normalized['vlm_intersection_detected'] = normalized.get('intersection_detected', 'NO')
        intersection_detected = self._stabilize_intersection_detected(normalized)
        normalized['intersection_detected'] = intersection_detected

        fallback_cue = self._infer_fallback_trajectory_motion_cue(
            trajectory_relation=trajectory_relation,
            visual_shift=visual_shift,
            direction_change=direction_change,
        )
        raw_cue = raw_cue or fallback_cue
        validated_cue, cue_confidence, cue_consistency = self._validate_trajectory_motion_cue(
            cue=raw_cue,
            trajectory_relation=trajectory_relation,
            visual_shift=visual_shift,
            direction_change=direction_change,
            intersection_detected=intersection_detected,
        )

        normalized['raw_trajectory_motion_cue'] = raw_cue
        normalized['trajectory_motion_cue'] = validated_cue
        normalized['trajectory_motion_cue_confidence'] = cue_confidence
        normalized['trajectory_motion_cue_consistency'] = cue_consistency
        return normalized

    def _stabilize_intersection_detected(
        self,
        level1_result: Dict[str, Any],
    ) -> str:
        raw_intersection = level1_result.get('intersection_detected', 'NO')
        if raw_intersection == 'UNCERTAIN':
            return 'UNCERTAIN'

        trajectory_motion_cue = level1_result.get('trajectory_motion_cue', 'AMBIGUOUS')
        trajectory_relation = level1_result.get('trajectory_relation', 'PARALLEL')
        direction_change = level1_result.get('direction_change', 'STRAIGHT')
        road_shape = level1_result.get('road_shape', 'STRAIGHT')
        visual_shift = level1_result.get('visual_shift', 'NO_SHIFT')
        trajectory_point_like = bool(level1_result.get('trajectory_point_like', False))
        heading_delta = abs(float(level1_result.get('trajectory_heading_delta_rad', 0.0) or 0.0))
        lateral_offset = abs(float(level1_result.get('trajectory_lateral_offset_m', 0.0) or 0.0))

        if trajectory_point_like:
            return 'NO'

        if trajectory_motion_cue in {'LEFT_LANE_CHANGE_CUE', 'RIGHT_LANE_CHANGE_CUE'}:
            return 'NO'

        strong_turn_context = (
            road_shape == 'STRAIGHT'
            and direction_change == 'TURNING'
            and visual_shift != 'NO_SHIFT'
            and heading_delta >= 0.22
            and lateral_offset >= 0.75
        )

        if raw_intersection == 'YES':
            if (
                trajectory_motion_cue == 'STRAIGHT_CUE'
                and trajectory_relation == 'PARALLEL'
                and direction_change == 'STRAIGHT'
                and visual_shift == 'NO_SHIFT'
            ):
                return 'NO'
            if strong_turn_context and trajectory_motion_cue in {'LEFT_TURN_CUE', 'RIGHT_TURN_CUE'}:
                return 'YES'
            return 'UNCERTAIN'

        if trajectory_motion_cue in {'LEFT_TURN_CUE', 'RIGHT_TURN_CUE'}:
            expected_curve = 'CURVE_LEFT' if trajectory_motion_cue == 'LEFT_TURN_CUE' else 'CURVE_RIGHT'
            if road_shape == expected_curve and trajectory_relation == 'PARALLEL':
                return 'NO'
            if strong_turn_context:
                return 'UNCERTAIN'

        return 'NO'

    def _infer_fallback_trajectory_motion_cue(
        self,
        *,
        trajectory_relation: str,
        visual_shift: str,
        direction_change: str,
    ) -> str:
        if trajectory_relation == 'CROSSING_LEFT':
            return 'LEFT_LANE_CHANGE_CUE'
        if trajectory_relation == 'CROSSING_RIGHT':
            return 'RIGHT_LANE_CHANGE_CUE'
        if direction_change == 'TURNING' and visual_shift == 'SHIFT_LEFT':
            return 'LEFT_TURN_CUE'
        if direction_change == 'TURNING' and visual_shift == 'SHIFT_RIGHT':
            return 'RIGHT_TURN_CUE'
        if trajectory_relation == 'PARALLEL':
            return 'STRAIGHT_CUE'
        return 'AMBIGUOUS'

    def _validate_trajectory_motion_cue(
        self,
        *,
        cue: str,
        trajectory_relation: str,
        visual_shift: str,
        direction_change: str,
        intersection_detected: str,
    ) -> tuple[str, float, str]:
        if not cue or cue == 'AMBIGUOUS':
            return 'AMBIGUOUS', 0.0, 'UNKNOWN'

        supports = 0
        contradictions = 0

        if cue == 'STRAIGHT_CUE':
            if trajectory_relation == 'PARALLEL':
                supports += 1
            if direction_change == 'STRAIGHT':
                supports += 1
            if visual_shift == 'NO_SHIFT':
                supports += 1
            if intersection_detected == 'NO':
                supports += 1
        elif cue in {'LEFT_TURN_CUE', 'RIGHT_TURN_CUE'}:
            expected_shift = 'SHIFT_LEFT' if cue == 'LEFT_TURN_CUE' else 'SHIFT_RIGHT'
            if direction_change == 'TURNING':
                supports += 1
            else:
                contradictions += 1
            if visual_shift == expected_shift:
                supports += 1
            elif visual_shift == 'NO_SHIFT':
                contradictions += 1
            if intersection_detected == 'YES':
                supports += 1
            elif intersection_detected == 'NO':
                contradictions += 1
            if trajectory_relation != 'PARALLEL':
                supports += 1
            else:
                contradictions += 1
        elif cue in {'LEFT_LANE_CHANGE_CUE', 'RIGHT_LANE_CHANGE_CUE'}:
            expected_relation = 'CROSSING_LEFT' if cue == 'LEFT_LANE_CHANGE_CUE' else 'CROSSING_RIGHT'
            if trajectory_relation == expected_relation:
                supports += 2
            elif trajectory_relation == 'PARALLEL':
                contradictions += 1
            if intersection_detected == 'NO':
                supports += 1
            elif intersection_detected == 'YES':
                contradictions += 1
            if direction_change == 'STRAIGHT':
                supports += 1
            elif direction_change == 'TURNING':
                contradictions += 1

        total = supports + contradictions
        confidence = 0.0 if total == 0 else round(supports / total, 2)

        if contradictions >= 3 and supports == 0:
            return 'AMBIGUOUS', 0.0, 'CONTRADICTED'
        if confidence < 0.35:
            return 'AMBIGUOUS', confidence, 'CONTRADICTED'
        if confidence < 0.6:
            return cue, confidence, 'WEAK'
        return cue, confidence, 'CONSISTENT'

    def _build_motion_observation(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        speed_diff = float(sensor_data.get('speed_diff', 0.0) or 0.0)
        timestamp_diff_sec = float(sensor_data.get('timestamp_diff_sec', 0.0) or 0.0)
        speed_change_rate = float(sensor_data.get('speed_change_rate', 0.0) or 0.0)
        brake = int(sensor_data.get('brake', 0) or 0)
        motion_feature_reliable = self._is_motion_feature_reliable(sensor_data)

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

        if direct_decel and not direct_accel:
            cause = 'DRIVER_BRAKE' if brake > 0 else 'MIXED'
            reason = 'brake=ON' if brake > 0 else 'negative speed delta / rate'
            return {'trend': 'DECREASING', 'cause': cause, 'reason': reason, 'reliable': motion_feature_reliable}

        if direct_accel and not direct_decel:
            return {
                'trend': 'INCREASING',
                'cause': 'DRIVER_ACCEL',
                'reason': 'positive speed delta / rate',
                'reliable': motion_feature_reliable,
            }

        if (
            ((motion_feature_reliable and abs(speed_diff) <= self.SPEED_DIFF_THRESHOLD) or not motion_feature_reliable)
            and (not has_reliable_rate or abs(speed_change_rate) <= self.STABLE_SPEED_RATE_THRESHOLD)
            and brake == 0
        ):
            reason = 'small speed delta / rate' if motion_feature_reliable else 'speed delta ignored due to long lookback'
            return {'trend': 'STABLE', 'cause': 'MIXED', 'reason': reason, 'reliable': motion_feature_reliable}

        reason = 'sensor evidence weak'
        if not motion_feature_reliable and brake == 0:
            reason = f'ignored speed delta because lookback={timestamp_diff_sec:.2f}s'
        return {'trend': None, 'cause': None, 'reason': reason, 'reliable': motion_feature_reliable}

    def _format_motion_summary(self, sensor_data: Dict[str, Any]) -> str:
        motion = self._build_motion_observation(sensor_data)
        trend = motion.get('trend') or 'UNKNOWN'
        cause = motion.get('cause') or 'UNKNOWN'
        reason = motion.get('reason') or 'unknown'
        return f"{trend} / {cause} ({reason})"

    def _normalize_level2_result(
        self,
        sensor_data: Dict[str, Any],
        level2_result: Dict[str, Any],
    ) -> Dict[str, Any]:
        normalized = dict(level2_result)
        motion = self._build_motion_observation(sensor_data)
        overrides = []
        motion_feature_reliable = self._is_motion_feature_reliable(sensor_data)

        observed_trend = motion.get('trend')
        observed_cause = motion.get('cause')

        if observed_trend in {'INCREASING', 'DECREASING', 'STABLE'}:
            original_trend = normalized.get('speed_trend', 'STABLE')
            if original_trend != observed_trend:
                overrides.append(f"speed_trend {original_trend}->{observed_trend}")
            normalized['speed_trend'] = observed_trend

        if observed_cause in {'DRIVER_ACCEL', 'DRIVER_BRAKE'}:
            original_cause = normalized.get('acceleration_cause', 'MIXED')
            if original_cause != observed_cause:
                overrides.append(f"acceleration_cause {original_cause}->{observed_cause}")
            normalized['acceleration_cause'] = observed_cause
        elif not motion_feature_reliable:
            original_cause = normalized.get('acceleration_cause', 'MIXED')
            if original_cause in {'DRIVER_ACCEL', 'DRIVER_BRAKE'}:
                overrides.append(f"acceleration_cause {original_cause}->MIXED")
                normalized['acceleration_cause'] = 'MIXED'

        if overrides:
            reasoning = normalized.get('reasoning', '')
            override_text = '; '.join(overrides)
            normalized['reasoning'] = (
                f"{reasoning} | sensor_normalized: {override_text}"
                if reasoning else f"sensor_normalized: {override_text}"
            )

        return normalized
    
    def _get_sensor_hints(self, sensor_data: Dict[str, Any]) -> str:
        """
        センサー値から事前フィルタリングヒントを生成（案2-1）

        VLMへのヒントとして、センサー値から絞り込んだクラス候補を返す。
        """
        hints = []
        speed = sensor_data.get('speed', 0)
        gyro_z = abs(sensor_data.get('gyro_z', 0))
        acc_x = abs(sensor_data.get('acc_x', 0))
        brake = sensor_data.get('brake', 0)
        blinker_r = sensor_data.get('blinker_r', 0)
        blinker_l = sensor_data.get('blinker_l', 0)

        if speed < SPEED_STOP_THRESHOLD and brake > 0:
            hints.append("センサー: 速度≈0かつブレーキON → 停止(4)の可能性が高い")
        if gyro_z > GYRO_THRESHOLD:
            raw_gyro = sensor_data.get('gyro_z', 0)
            direction = "左" if raw_gyro > 0 else "右"
            hints.append(
                f"センサー: 旋回検出 (gyro_z={raw_gyro:.3f} rad/s) → "
                f"{direction}折・車線変更(6/7/8/9)の可能性"
            )
        if acc_x > ACC_X_THRESHOLD:
            raw_acc_x = sensor_data.get('acc_x', 0)
            direction = "左" if raw_acc_x > 0 else "右"
            hints.append(
                f"センサー: 横加速度あり (acc_x={raw_acc_x:.3f} m/s²) → "
                f"{direction}への旋回・車線変更の可能性"
            )
        if speed > 5 and brake == 0 and gyro_z < GYRO_THRESHOLD:
            hints.append("センサー: 走行中・旋回なし → 停止(4)・発進(5)は除外")
        if blinker_r > 0:
            hints.append("センサー: 右ウィンカーON → 右折(7)または車線変更右(9)の可能性")
        if blinker_l > 0:
            hints.append("センサー: 左ウィンカーON → 左折(6)または車線変更左(8)の可能性")

        return "\n".join(hints)

    def _post_process_label(
        self,
        predicted_label: int,
        sensor_data: Dict[str, Any],
        graph_result: Optional[Dict[str, Any]] = None
    ) -> Optional[int]:
        """
        VLM予測とセンサーデータが著しく矛盾する場合にルールベースで補正（案4）

        Returns:
            補正後のラベル。矛盾検出時はNone（呼び出し側で元の予測を保持）。
        """
        speed = sensor_data.get('speed', 0)
        gyro_z = abs(sensor_data.get('gyro_z', 0))
        brake = sensor_data.get('brake', 0)
        motion_observation = self._build_motion_observation(sensor_data)
        strong_candidate = graph_result.get("strong_candidate") if graph_result else None
        top_candidates = graph_result.get("top_candidates", []) if graph_result else []
        top_candidate = top_candidates[0] if top_candidates else None
        concepts = graph_result.get("concepts", {}) if graph_result else {}
        intersection_state = concepts.get("intersection_state", "UNKNOWN")
        stop_likelihood = concepts.get("stop_likelihood", "LOW")
        trajectory_point_like = bool(concepts.get("trajectory_point_like", False))

        # ルール1: 速度0かつブレーキON → 停止(4)に強制補正
        if (
            speed < SPEED_STOP_THRESHOLD
            and brake > 0
            and motion_observation.get('trend') != 'DECREASING'
            and predicted_label != 4
        ):
            logger.info(
                f"[PostProcess] Rule1: speed={speed:.1f}, brake={brake} → force category 4 (停止)"
            )
            return 4

        # ルール2: 明確な旋回なのに等速走行(1)と判定 → Noneで矛盾を通知
        if gyro_z > GYRO_THRESHOLD * 1.5 and predicted_label == 1:
            logger.warning(
                f"[PostProcess] Rule2: gyro_z={gyro_z:.3f} > threshold but predicted 1 (等速走行). "
                "Contradiction detected."
            )
            return None

        # ルール3: Graph が強い非 1/4 候補を示し、かつ 1/4 への偏りが疑われる場合は補正
        if graph_result:
            label_support = graph_result.get("label_support", {})
            if strong_candidate:
                strong_id = strong_candidate["category_id"]
                strong_score = float(strong_candidate["score"])
                predicted_score = float(label_support.get(predicted_label, 0.0))
                margin = strong_score - predicted_score

                if (
                    predicted_label == 1
                    and strong_id not in {1, 4}
                    and strong_score >= 0.80
                    and margin >= 0.25
                ):
                    logger.info(
                        "[PostProcess] Rule3(Graph): correcting biased label %s -> %s "
                        "(strong_score=%.2f, margin=%.2f)",
                        predicted_label,
                        strong_id,
                        strong_score,
                        margin,
                    )
                    return strong_id

            if (
                top_candidate
                and predicted_label in {6, 7}
                and top_candidate["category_id"] in {8, 9}
                and intersection_state in {"NO", "UNCERTAIN"}
                and float(top_candidate["score"]) >= 0.65
            ):
                predicted_score = float(label_support.get(predicted_label, 0.0))
                if (float(top_candidate["score"]) - predicted_score) >= self.LEVEL3_TOP_MARGIN_THRESHOLD:
                    logger.info(
                        "[PostProcess] Rule3b(Graph): correcting turn -> lane change %s -> %s",
                        predicted_label,
                        top_candidate["category_id"],
                    )
                    return top_candidate["category_id"]

        # ルール4: 停止予測だが停止条件を満たさない場合は stop bias を補正
        if predicted_label == 4 and (speed > SPEED_STOP_THRESHOLD or brake == 0):
            if (
                strong_candidate
                and strong_candidate["category_id"] == 4
                and stop_likelihood in {"HIGH", "MEDIUM"}
                and trajectory_point_like
                and speed <= self.STOPLIKE_SPEED_THRESHOLD
            ):
                logger.info("[PostProcess] Rule4: keep stop due to strong stop-like graph evidence")
                return 4
            if strong_candidate and strong_candidate["category_id"] != 4 and strong_candidate["score"] >= 0.55:
                logger.info(
                    "[PostProcess] Rule4(Graph): correcting stop bias %s -> %s",
                    predicted_label,
                    strong_candidate["category_id"],
                )
                return strong_candidate["category_id"]
            if motion_observation.get('trend') == 'DECREASING':
                return 3
            if motion_observation.get('trend') == 'STABLE' and gyro_z <= GYRO_THRESHOLD:
                return 1

        # ルール4b: 低速で stop-like な軌道なら、等速走行より停止/減速を優先
        if (
            predicted_label == 1
            and speed <= self.STOPLIKE_SPEED_THRESHOLD
            and trajectory_point_like
            and stop_likelihood == "HIGH"
        ):
            if stop_likelihood == "HIGH":
                logger.info("[PostProcess] Rule4b: stop-like trajectory -> 停止")
                return 4
            if top_candidate and top_candidate["category_id"] in {3, 4} and float(top_candidate["score"]) >= 0.55:
                logger.info(
                    "[PostProcess] Rule4b(Graph): constant speed -> %s due to stop-like trajectory",
                    top_candidate["category_id"],
                )
                return top_candidate["category_id"]

        # ルール5: 加速/減速が direct_motion_trend と逆なら補正
        if predicted_label == 2 and motion_observation.get('trend') == 'DECREASING':
            if strong_candidate and strong_candidate["category_id"] != 2 and strong_candidate["score"] >= 0.55:
                return strong_candidate["category_id"]
            return 3
        if predicted_label == 3 and motion_observation.get('trend') == 'INCREASING':
            if strong_candidate and strong_candidate["category_id"] != 3 and strong_candidate["score"] >= 0.55:
                return strong_candidate["category_id"]
            return 2

        # ルール5b: turn 予測より speed-event の Graph 候補が明確に強い場合は補正
        if (
            top_candidate
            and predicted_label in {6, 7}
            and top_candidate["category_id"] in {1, 2, 3, 4, 5}
            and float(top_candidate["score"]) >= 0.65
        ):
            predicted_score = float(graph_result.get("label_support", {}).get(predicted_label, 0.0))
            if (float(top_candidate["score"]) - predicted_score) >= self.LEVEL3_TOP_MARGIN_THRESHOLD:
                logger.info(
                    "[PostProcess] Rule5b(Graph): correcting turn -> speed event %s -> %s",
                    predicted_label,
                    top_candidate["category_id"],
                )
                return top_candidate["category_id"]

        # ルール6: Graph の発進候補が強い場合は発進を優先
        if (
            predicted_label in {1, 2}
            and top_candidate
            and top_candidate["category_id"] == 5
            and float(top_candidate["score"]) >= 0.60
        ):
            logger.info("[PostProcess] Rule6(Graph): correcting %s -> 発進", predicted_label)
            return 5

        return predicted_label

    def _get_category_name(self, category_id: int) -> str:
        """
        Get Japanese category name from category ID
        """
        category_names = {
            0: 'その他',
            1: '等速走行',
            2: '加速',
            3: '減速',
            4: '停止',
            5: '発進',
            6: '左折',
            7: '右折',
            8: '車線変更（左）',
            9: '車線変更（右）',
            10: '転回（Uターン）'
        }
        return category_names.get(category_id, 'その他')
