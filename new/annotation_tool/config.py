# config.py
"""
Moondream2 VLM Configuration for Automatic Annotation
"""

# VLM Model Settings
# Multi-frame video understanding models
# Option 1: Qwen2-VL-2B (Multi-frame, 4GB VRAM, faster, good for 8GB GPU)
# Option 2: Qwen2-VL-7B (Multi-frame, 8GB+ VRAM, better accuracy, requires more memory)
# Note: Moondream2 is NOT supported (single-frame only)
# Note: Heron has Windows tokenizer compatibility issues
HERON_MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"
USE_MULTI_FRAME = True  # Enable multi-frame temporal understanding
USE_GPU = True  # GPU有効（RTX 4060使用）
TORCH_DTYPE = "float16"  # GPUではfloat16で高速化
DEVICE_PREFERENCE = "cuda" if USE_GPU else "cpu"  # Device preference for model loading

# Video Processing Settings
NUM_FRAMES_TO_EXTRACT = 4  # Number of frames to extract from video (reduced for 8GB VRAM)
NUM_FRAMES_TO_USE = 4  # Number of frames to actually use (reduced to fit in memory)
FRAME_EXTRACTION_METHOD = "uniform"  # "uniform" or "temporal"
MAX_IMAGE_SIZE = 1080  # Maximum dimension (width or height) for image processing

# L2M+CoT Settings
USE_L2M_COT = True  # Enable Least-to-Most + Chain-of-Thought reasoning (experimental)
# L2M+CoTを有効にすると、1つの動画に対して3回の推論を実行します（Level 1, 2, 3）
# より論理的で説明可能な推論が可能になりますが、推論時間が約3倍になります

# Generation Optimization Settings (速度最適化設定)
# これらの設定を調整することで、機能を維持しながら推論速度を向上できます
MAX_NEW_TOKENS_STANDARD = 50  # 標準推論の最大トークン数 (デフォルト: 100 → 50に削減)
MAX_NEW_TOKENS_L2M = 150  # L2M+CoT推論の最大トークン数 (デフォルト: 300 → 150に削減)
# 理由: アクション分類は0-10の数字1つだけなので、長い出力は不要
# 50トークンでも十分な推論結果が得られます

USE_KV_CACHE = True  # KVキャッシュの使用 (高速化)
NUM_FRAMES_OPTIMIZED = 3  # フレーム数の最適化 (4 → 3に削減可能、精度とのトレードオフ)

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

# Model Loading Settings
MODEL_LOAD_TIMEOUT = 300  # seconds
ENABLE_FLASH_ATTENTION = False  # Set to True if flash-attn is installed
