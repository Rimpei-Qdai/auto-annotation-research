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
        level1_result = self._level1_geometry(frames)
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
    
    def _level1_geometry(self, frames: List[Image.Image]) -> Dict[str, Any]:
        """
        Level 1: Analyze road geometry and trajectory relationship
        
        This step analyzes only visual geometric features.
        No sensor data is used, only objective facts are extracted.
        """
        prompt = """あなたは自動運転データの分析エキスパートです。以下の画像は車両の前方カメラから撮影された4枚の連続フレームです。

【画像の説明】
- 赤い線: 車両の予測軌道（これから進む予定の経路）
- 緑の線: 現在の車線区分線または車線中心

【タスク】
以下の3つの質問に段階的に答えてください。

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
{
  "reasoning": "道路形状と軌道の関係についての分析結果を簡潔に説明",
  "road_shape": "STRAIGHT",
  "trajectory_relation": "PARALLEL",
  "slope": "FLAT",
  "intersection_detected": "NO",
  "direction_change": "STRAIGHT",
  "visual_shift": "NO_SHIFT"
}
```"""
        
        try:
            raw_output = self._generate_with_frames(frames, prompt)
            parsed = self._parse_json_output(raw_output)
            
            if parsed is None:
                # Default values when parsing fails
                logger.warning(f"Level 1 JSON parse failed. Raw output: {raw_output[:300]}")
                return {
                    'reasoning': raw_output[:500],
                    'road_shape': 'STRAIGHT',
                    'trajectory_relation': 'PARALLEL',
                    'slope': 'FLAT',
                    'intersection_detected': 'NO',
                    'direction_change': 'STRAIGHT',
                    'visual_shift': 'NO_SHIFT'
                }

            return parsed

        except Exception as e:
            logger.error(f"Level 1 推論エラー: {e}", exc_info=True)
            return {
                'reasoning': f"Error: {str(e)}",
                'road_shape': 'STRAIGHT',
                'trajectory_relation': 'PARALLEL',
                'slope': 'FLAT',
                'intersection_detected': 'NO',
                'direction_change': 'STRAIGHT',
                'visual_shift': 'NO_SHIFT'
            }
    
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

        brake_status = 'ON' if brake > 0 else 'OFF'
        blinker_status = '右ウィンカーON' if blinker_r > 0 else ('左ウィンカーON' if blinker_l > 0 else 'OFF')
        motion_summary = self._format_motion_summary(sensor_data)

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
- センサー由来の運動要約: {motion_summary}{gyro_warning}{acc_x_warning}

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
            f"- direct_motion_reason: {motion_observation.get('reason', '情報なし')}"
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

3. **旋回・車線変更の判定（等速走行より優先）**:
   - |gyro_z| > 0.1 rad/s の場合 → カテゴリ6/7/8/9/10 を積極的に検討
   - ウィンカーONの場合 → 旋回または車線変更の強い証拠
   - 交差点検出=YES かつ 映像シフトあり → カテゴリ6または7
   - 交差点なし かつ 車線シフト → カテゴリ8または9

4. **加速・減速の判定**:
   - 加速度の原因がDRIVER_BRAKEの場合 → カテゴリ3（減速）
   - 速度変化量が大きい場合 → カテゴリ2（加速）またはカテゴリ3（減速）
   - `direct_motion_trend` が INCREASING / DECREASING のときは、その符号を優先して整合するカテゴリを選ぶ

5. **Graph 概念を必ず確認**:
   - `Graph 上位候補` は中間概念から構成された候補である
   - 特に、等速走行(1)や停止(4)を選ぶ前に Graph 候補との矛盾がないか確認する
   - 右左折・車線変更・減速の候補が Graph に強く出ている場合は優先的に検討する{strong_candidate_rule}

6. **等速走行（1）は最後の手段**:
   - 旋回・車線変更（6,7,8,9,10）の可能性を除外した後に選択
   - 速度変化（2,3）の可能性を除外した後に選択
   - 停止・発進（4,5）の可能性を除外した後に選択

7. **停止（4）の抑制ルール**:
   - 速度が 1 km/h を超えていて、かつブレーキが OFF の場合は停止（4）を選ばない
   - Graph や direct_motion_trend が減速・旋回を示している場合、停止よりそちらを優先する

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

            return parsed
            
        except Exception as e:
            logger.error(f"Level 3 推論エラー: {e}", exc_info=True)
            return {
                'final_reasoning': f"Error: {str(e)}",
                'category_id': 0,
                'category_name': 'その他',
                'confidence': 0.0
            }
    
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

    def _build_motion_observation(self, sensor_data: Dict[str, Any]) -> Dict[str, Any]:
        speed_diff = float(sensor_data.get('speed_diff', 0.0) or 0.0)
        timestamp_diff_sec = float(sensor_data.get('timestamp_diff_sec', 0.0) or 0.0)
        speed_change_rate = float(sensor_data.get('speed_change_rate', 0.0) or 0.0)
        brake = int(sensor_data.get('brake', 0) or 0)

        has_reliable_rate = 0.0 < timestamp_diff_sec <= self.MAX_RELIABLE_TIME_DIFF_SEC
        direct_accel = (
            speed_diff >= self.STRONG_SPEED_DIFF_THRESHOLD
            or (has_reliable_rate and speed_change_rate >= self.SPEED_RATE_THRESHOLD)
        )
        direct_decel = (
            brake > 0
            or speed_diff <= -self.STRONG_SPEED_DIFF_THRESHOLD
            or (has_reliable_rate and speed_change_rate <= -self.SPEED_RATE_THRESHOLD)
        )

        if direct_decel and not direct_accel:
            cause = 'DRIVER_BRAKE' if brake > 0 else 'MIXED'
            reason = 'brake=ON' if brake > 0 else 'negative speed delta / rate'
            return {'trend': 'DECREASING', 'cause': cause, 'reason': reason}

        if direct_accel and not direct_decel:
            return {'trend': 'INCREASING', 'cause': 'DRIVER_ACCEL', 'reason': 'positive speed delta / rate'}

        if (
            abs(speed_diff) <= self.SPEED_DIFF_THRESHOLD
            and (not has_reliable_rate or abs(speed_change_rate) <= self.STABLE_SPEED_RATE_THRESHOLD)
            and brake == 0
        ):
            return {'trend': 'STABLE', 'cause': 'MIXED', 'reason': 'small speed delta / rate'}

        return {'trend': None, 'cause': None, 'reason': 'sensor evidence weak'}

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

        # ルール4: 停止予測だが停止条件を満たさない場合は stop bias を補正
        if predicted_label == 4 and (speed > SPEED_STOP_THRESHOLD or brake == 0):
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

        # ルール5: 加速/減速が direct_motion_trend と逆なら補正
        if predicted_label == 2 and motion_observation.get('trend') == 'DECREASING':
            if strong_candidate and strong_candidate["category_id"] != 2 and strong_candidate["score"] >= 0.55:
                return strong_candidate["category_id"]
            return 3
        if predicted_label == 3 and motion_observation.get('trend') == 'INCREASING':
            if strong_candidate and strong_candidate["category_id"] != 3 and strong_candidate["score"] >= 0.55:
                return strong_candidate["category_id"]
            return 2

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
