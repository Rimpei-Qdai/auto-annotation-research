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
import torch
import re

logger = logging.getLogger(__name__)


class L2MCoTPipeline:
    """
    Least-to-Most Prompting + Chain-of-Thought Pipeline for Heron
    複数回のVLM推論を組み合わせて段階的な分析を行う
    """
    
    def __init__(self, heron_annotator):
        """
        Args:
            heron_annotator: HeronAnnotatorのインスタンス（既にモデルがロード済み）
        """
        self.annotator = heron_annotator
        self.conversation_history = []
    
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
        
        # Level 1: Geometric analysis
        logger.info("--- Level 1: Geometric Analysis ---")
        level1_result = self._level1_geometry(frames)
        logger.info(f"Level 1 完了: {level1_result}")
        
        # Level 2: Physics consistency check
        logger.info("--- Level 2: Physics Consistency Check ---")
        level2_result = self._level2_physics(frames, sensor_data, level1_result)
        logger.info(f"Level 2 完了: {level2_result}")
        
        # Level 3: Final classification
        logger.info("--- Level 3: Final Classification ---")
        level3_result = self._level3_classification(frames, sensor_data, level1_result, level2_result)
        logger.info(f"Level 3 完了: {level3_result}")
        
        logger.info("=== L2M+CoT Analysis Complete ===")
        
        return {
            'level1': level1_result,
            'level2': level2_result,
            'level3': level3_result,
            'final_category': level3_result.get('category_id', 0),
            'confidence': level3_result.get('confidence', 0.5)
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

【出力形式】
以下のJSON形式でのみ回答してください。推論過程も「reasoning」フィールドに簡潔に記述してください。

```json
{
  "reasoning": "道路形状と軌道の関係についての分析結果を簡潔に説明",
  "road_shape": "STRAIGHT",
  "trajectory_relation": "PARALLEL",
  "slope": "FLAT"
}
```"""
        
        try:
            raw_output = self._generate_with_frames(frames, prompt)
            parsed = self._parse_json_output(raw_output)
            
            if parsed is None:
                # Default values when parsing fails
                logger.warning(f"Level 1 JSON parse failed. Raw output: {raw_output[:300]}")
                return {
                    'reasoning': raw_output[:500],  # Save partial raw output
                    'road_shape': 'STRAIGHT',
                    'trajectory_relation': 'PARALLEL',
                    'slope': 'FLAT'
                }
            
            return parsed
            
        except Exception as e:
            logger.error(f"Level 1 推論エラー: {e}", exc_info=True)
            return {
                'reasoning': f"Error: {str(e)}",
                'road_shape': 'STRAIGHT',
                'trajectory_relation': 'PARALLEL',
                'slope': 'FLAT'
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
        brake = sensor_data.get('brake', 0)
        
        brake_status = 'ON' if brake > 0 else 'OFF'
        prompt = f"""あなたは自動運転データの分析エキスパートです。Level 1の幾何学的分析結果と、実際の車両センサーデータを照合して、物理法則との整合性を検証してください。

【Level 1の分析結果】
- 道路形状: {level1_result.get('road_shape', 'UNKNOWN')}
- 軌道と車線の関係: {level1_result.get('trajectory_relation', 'UNKNOWN')}
- 道路の勾配: {level1_result.get('slope', 'UNKNOWN')}

【センサーデータ】
- 速度: {speed:.1f} km/h
- 前後加速度: {acc_x:.2f} m/s²
- 横加速度: {acc_y:.2f} m/s²
- ブレーキ: {brake_status}

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
                return {
                    'reasoning': raw_output[:500],
                    'acceleration_cause': 'MIXED',
                    'speed_trend': 'STABLE',
                    'consistency_check': 'CONSISTENT'
                }
            
            return parsed
            
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
        level2_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Level 3: Final action classification
        
        Integrate Level 1 and Level 2 analysis results
        to classify into one of 11 driving action categories.
        """
        # CRITICAL: Check for stopped vehicle first
        speed = sensor_data.get('speed', 0)
        brake = sensor_data.get('brake', 0)
        
        # If Speed=0 and Brake=ON, it's a STOP (category 4: 停止)
        if speed == 0 and brake > 0:
            logger.info("[Level 3] STOP detected: Speed=0, Brake=ON → Forcing category 4 (停止)")
            return {
                'final_reasoning': 'Vehicle is completely stopped (Speed=0km/h) with brake engaged.',
                'category_id': 4,
                'category_name': '停止',
                'confidence': 0.99
            }
        
        prompt = f"""あなたは自動運転データの分析エキスパートです。これまでの分析結果を統合して、車両の運転行動を最終的に分類してください。

【これまでの分析結果】

◆ Level 1: 幾何学的分析
- 道路形状: {level1_result.get('road_shape')}
- 軌道と車線の関係: {level1_result.get('trajectory_relation')}
- 道路の勾配: {level1_result.get('slope')}

◆ Level 2: 物理法則との整合性
- 加速度の原因: {level2_result.get('acceleration_cause')}
- 速度トレンド: {level2_result.get('speed_trend')}

◆ センサーデータ
- 現在速度: {speed:.1f} km/h
- ブレーキ: {'ON' if brake > 0 else 'OFF'}

【運転行動カテゴリ一覧】
0: その他（Other）
1: 等速走行（Constant speed）
2: 加速（Acceleration）
3: 減速（Deceleration）
4: 停止（Stop）
5: 発進（Start）
6: 左折（Left turn）
7: 右折（Right turn）
8: 車線変更（左）（Lane change left）
9: 車線変更（右）（Lane change right）
10: 転回・Uターン（U-turn）

【分類ルール - 必ず守ること】

1. **停止判定（最優先）**:
   - 速度が0 km/hでブレーキがONの場合 → 必ずカテゴリ4（停止）

2. **発進判定**:
   - 速度が5 km/h未満で速度トレンドがINCREASINGの場合 → カテゴリ5（発進）

3. **カーブと車線変更の区別（重要）**:
   - 軌道が「PARALLEL」の場合:
     * 道路に沿って走行しているため、カテゴリは 1/2/3 のみ
     * 軌道が曲がっていても、道路がカーブしていれば車線変更ではない
   - 軌道が「CROSSING_LEFT」または「CROSSING_RIGHT」の場合:
     * 車線を越える動きがあるため、カテゴリ 6/7/8/9 が可能

4. **加速・減速の判定**:
   - 加速度の原因がDRIVER_BRAKEの場合 → カテゴリ3（減速）
   - 速度が5 km/h未満で減速中 → カテゴリ3または4

5. **右左折の判定**:
   - 道路自体がカーブしており、軌道がPARALLELの場合は右左折ではない
   - 交差点で軌道がCROSSINGしている場合は右左折の可能性

【タスク】
上記の分析結果とルールを総合的に考慮して、最も適切なカテゴリを1つ選択してください。
推論過程を「final_reasoning」に記述し、カテゴリID、カテゴリ名、信頼度を出力してください。

Output JSON only:
```json
{{
  "final_reasoning": "Brief integration of L1+L2",
  "category_id": 0,
  "category_name": "Constant speed",
  "confidence": 0.95
}}
```"""
        
        try:
            raw_output = self._generate_with_frames(frames, prompt)
            parsed = self._parse_json_output(raw_output)
            
            if parsed is None or 'category_id' not in parsed:
                logger.warning(f"Level 3 JSON parse failed. Raw output: {raw_output[:300]}")
                # If category ID cannot be obtained, estimate heuristically from output
                category_id = self._extract_category_from_text(raw_output)
                return {
                    'final_reasoning': raw_output[:500],
                    'category_id': category_id,
                    'category_name': self._get_category_name(category_id),
                    'confidence': 0.5
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
        # モデルタイプの判定
        is_qwen = "qwen" in self.annotator.model.__class__.__name__.lower()
        
        try:
            if is_qwen:
                # Qwen2-VL format
                messages = [
                    {
                        "role": "user",
                        "content": [
                            *[{"type": "image", "image": frame} for frame in frames],
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                text_prompt = self.annotator.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.annotator.processor(
                    text=[text_prompt],
                    images=frames,
                    padding=True,
                    return_tensors="pt"
                )
            else:
                # Heron format - Processor handles text and image separately
                inputs = self.annotator.processor(
                    images=frames,
                    text=[prompt],
                    return_tensors="pt",
                    padding=True
                )
            
            # Move to device
            inputs = {k: v.to(self.annotator.device) for k, v in inputs.items()}
            
            # Generate
            logger.info(f"Generating with {len(frames)} frames, prompt length: {len(prompt)} chars")
            with torch.no_grad():
                output_ids = self.annotator.model.generate(
                    **inputs,
                    max_new_tokens=300,  # Sufficient length for JSON output
                    do_sample=False  # Greedy decoding for reliability
                )
            
            # Decode
            output_text = self.annotator.processor.batch_decode(
                output_ids,
                skip_special_tokens=True
            )[0]
            
            # Remove prompt part (some models include the prompt)
            if prompt in output_text:
                output_text = output_text.replace(prompt, "").strip()
            
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
