# config.py
"""
Simple video baseline configuration.
"""

# VLM Model Settings
# Backend selection:
# - "hf": local Hugging Face VLM (Qwen3-VL etc.)
# - "gemini": Google Gemini API
MODEL_PROVIDER = "hf"
HERON_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
GEMINI_MODEL_ID = "gemini-2.5-flash"
GEMINI_API_KEY_ENV = "GEMINI_API_KEY"
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
USE_MULTI_FRAME = False
USE_GPU = True
TORCH_DTYPE = "float16"
DEVICE_PREFERENCE = "cuda" if USE_GPU else "cpu"

# Video Processing Settings
NUM_FRAMES_TO_EXTRACT = 4
NUM_FRAMES_TO_USE = 4
FRAME_EXTRACTION_METHOD = "uniform"
MAX_IMAGE_SIZE = 720
VIDEO_CLIP_DURATION_SECONDS = 6.0
PROMPT_VERSION = "simple_video_v2_qwen3_timestamp_description"

# Baseline Behavior Flags
USE_L2M_COT = False
USE_VLM_DIRECT = True

# Generation Optimization Settings
MAX_NEW_TOKENS_STANDARD = 32
MAX_NEW_TOKENS_DESCRIPTION = 96
MAX_NEW_TOKENS_STRUCTURED = 32
MAX_NEW_TOKENS_L2M = 150
USE_KV_CACHE = True
NUM_FRAMES_OPTIMIZED = 4

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
    10: "転回(Uターン)",
}

TIMESTAMP_DESCRIPTION_PROMPT_TEMPLATE = """あなたは運転行動を観察するAIです。
この動画は判定対象時刻を中心とした前後3秒、合計6秒のクリップです。
特にクリップ内の {target_timestamp_text} 地点の自車に注目してください。

質問:
{target_timestamp_text} 地点で、自車はどのような行動をしていますか？
直前直後の様子も踏まえて、2〜4文で短く説明してください。

注意:
- 自車の行動だけを説明してください
- ラベル番号は書かないでください
- 箇条書きではなく自然文で書いてください
"""

STRUCTURED_VIDEO_PROMPT_TEMPLATE = """あなたは運転行動を構造化するAIです。
この動画は判定対象時刻を中心とした前後3秒、合計6秒のクリップです。
特にクリップ内の {target_timestamp_text} 地点の自車に注目してください。

参考説明:
{description_text}

上の説明と動画の内容を踏まえて、{target_timestamp_text} 地点の自車について次の3項目を固定語彙で答えてください。

DIRECTION=STRAIGHT|LEFT|RIGHT|UNKNOWN
SPEED_STATE=CONSTANT|ACCEL|DECEL|STOPPED|STARTING|UNKNOWN
MANEUVER=LEFT_TURN|RIGHT_TURN|LEFT_LANE_CHANGE|RIGHT_LANE_CHANGE|U_TURN|OTHER|UNKNOWN

出力は上の3行のみ。説明文は不要です。
"""

SIMPLE_VIDEO_PROMPT_TEMPLATE = TIMESTAMP_DESCRIPTION_PROMPT_TEMPLATE

# Legacy prompt names are kept for import compatibility on this branch.
PROMPT_STAGE1_TEMPLATE = SIMPLE_VIDEO_PROMPT_TEMPLATE
PROMPT_STAGE2_ROTATION_TEMPLATE = SIMPLE_VIDEO_PROMPT_TEMPLATE
PROMPT_STAGE2_NONROTATION_TEMPLATE = SIMPLE_VIDEO_PROMPT_TEMPLATE

# Model Loading Settings
MODEL_LOAD_TIMEOUT = 300
ENABLE_FLASH_ATTENTION = False
