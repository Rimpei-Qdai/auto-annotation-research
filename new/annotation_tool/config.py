# config.py
"""
Moondream2 VLM Configuration for Automatic Annotation
"""

# VLM Model Settings
# ======== GPU サーバー設定 (kiwi: RTX 4090, 24GB VRAM) ========
# Option 1: Qwen2-VL-2B (4GB VRAM, ローカル RTX 4060 向け)
# Option 2: Qwen2-VL-7B (14GB VRAM, kiwi RTX 4090 向け, 高精度)
HERON_MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
USE_MULTI_FRAME = True  # Enable multi-frame temporal understanding
USE_GPU = True  # GPU有効（kiwi: RTX 4090使用）
# bfloat16: RTX 4090 (Ada Lovelace) はネイティブサポートで float16 より安定
TORCH_DTYPE = "bfloat16"
DEVICE_PREFERENCE = "cuda" if USE_GPU else "cpu"  # Device preference for model loading

# Video Processing Settings
# RTX 4090 (24GB VRAM) では 8 フレームを使用可能
NUM_FRAMES_TO_EXTRACT = 8  # Number of frames to extract from video
NUM_FRAMES_TO_USE = 8  # Number of frames to actually use
FRAME_EXTRACTION_METHOD = "uniform"  # "uniform" or "temporal"
MAX_IMAGE_SIZE = 1080  # Maximum dimension (width or height) for image processing

# L2M+CoT Settings
USE_L2M_COT = True  # Enable Least-to-Most + Chain-of-Thought reasoning
# L2M+CoTを有効にすると、1つの動画に対して3回の推論を実行します（Level 1, 2, 3）
# RTX 4090 の高速推論で処理時間の問題を解消

# Generation Settings (RTX 4090 では十分な VRAM があるため上限を緩和)
MAX_NEW_TOKENS_STANDARD = 100  # 標準推論の最大トークン数
MAX_NEW_TOKENS_L2M = 300  # L2M+CoT 推論の最大トークン数（JSON 出力に十分な長さ）

USE_KV_CACHE = True  # KVキャッシュの使用 (高速化)
NUM_FRAMES_OPTIMIZED = 8  # フレーム数（RTX 4090 では削減不要）

# Action Label Mapping
ACTION_LABELS = {
    0: "その他",
    1: "等速走行",
    2: "加速",
    3: "減速",
    4: "停止",
    5: "発進",
    6: "左折",
    7: "右折",
    8: "車線変更(左)",
    9: "車線変更(右)",
    10: "転回(Uターン)"
}

# Prompt Template for LLaVA
PROMPT_TEMPLATE = """You are an AI assistant analyzing taxi driving behavior.
Classify the driver's action from the provided video frames and sensor data.

**CRITICAL**: Analyze the RED trajectory line primarily to determine the vehicle's action.

Visual Indicators in the frames:
- RED LINE/DOTS: Predicted trajectory of the vehicle for the next 3 seconds
  * Each red dot represents a future position
  * If RED trajectory curves significantly LEFT → Consider "6" (Left turn)
  * If RED trajectory curves significantly RIGHT → Consider "7" (Right turn)
  * If RED trajectory follows GREEN line (straight) → Consider "1", "2", "3", "5" based on speed
- GREEN LINE: Straight-ahead reference line

Reference Sensor Data (supplementary):
- Speed: {speed} km/h
- Acceleration X: {acc_x} m/s²
- Acceleration Y: {acc_y} m/s²
- Acceleration Z: {acc_z} m/s²
- Brake: {brake}

**Decision Rules**:
1. If Speed = 0 km/h → Must be "4" (Stop)
2. If RED trajectory curves LEFT significantly → "6" (Left turn)
3. If RED trajectory curves RIGHT significantly → "7" (Right turn)
4. If RED trajectory is straight AND Speed > 0:
   - If Speed is increasing (Acc X > 0.5) → "2" (Acceleration) or "5" (Start if was stopped)
   - If Speed is decreasing (Acc X < -0.5) OR Brake = 1 → "3" (Deceleration)
   - If Speed is constant (Acc X ≈ 0) → "1" (Constant speed)
5. Lane changes ("8"/"9"): Small lateral shift without intersection
6. For U-turn ("10"): Very sharp turn with trajectory reversing direction

Action Categories:
0: Other
1: Constant speed driving - moving straight at constant speed
2: Acceleration - speed increasing
3: Deceleration - speed decreasing (including braking)
4: Stop - completely stopped (speed 0 km/h)
5: Start - beginning to move from stopped state
6: Left turn - turning left (at intersections)
7: Right turn - turning right (at intersections)
8: Lane change (left) - moving to left lane
9: Lane change (right) - moving to right lane
10: U-turn - 180-degree direction change

**Respond with ONLY the number (0-10). Nothing else.**"""

# Sensor Thresholds for Turning Detection (案2-1, 案1-2)
GYRO_THRESHOLD = 0.1        # rad/s - 旋回検出のジャイロ閾値
ACC_X_THRESHOLD = 0.3       # m/s² - 横加速度による旋回検出閾値
SPEED_STOP_THRESHOLD = 1.0  # km/h - 停止とみなす速度閾値

# Few-shot examples for rare classes (案5-2)
FEW_SHOT_EXAMPLES = """
【参考例（センサー → 正解クラス）】
例1: gyro_z=+0.25 rad/s、速度=15km/h、交差点で右に曲がっている → 7（右折）
例2: gyro_z=+0.18 rad/s、速度=20km/h、車線が左にシフト → 8（車線変更（左））
例3: gyro_z=0.00 rad/s、速度=0km/h、ブレーキON → 4（停止）
例4: gyro_z=-0.20 rad/s、速度=12km/h、交差点で左に曲がっている → 6（左折）
例5: acc_x=-1.5 m/s²、ブレーキON、速度低下 → 3（減速）
例6: 速度=0km/h → 速度トレンドINCREASING → 5（発進）
"""

# Model Loading Settings
MODEL_LOAD_TIMEOUT = 300  # seconds
ENABLE_FLASH_ATTENTION = True  # RTX 4090 では flash-attn が利用可能
