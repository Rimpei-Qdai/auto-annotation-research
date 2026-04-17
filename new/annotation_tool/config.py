# config.py
"""
Simple Qwen3-VL video baseline configuration.
"""

# VLM Model Settings
HERON_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
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
PROMPT_VERSION = "simple_video_v1_qwen3vl"

# Baseline Behavior Flags
USE_L2M_COT = False
USE_VLM_DIRECT = True

# Generation Optimization Settings
MAX_NEW_TOKENS_STANDARD = 16
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

SIMPLE_VIDEO_PROMPT_TEMPLATE = """あなたは運転行動を分類するAIです。
この動画は判定対象時刻を中心とした前後3秒、合計6秒のクリップです。
動画全体を見て、中央付近の車両の主な運転行動を次の11分類から1つだけ選んでください。

0: その他
1: 等速走行
2: 加速
3: 減速
4: 停止
5: 発進
6: 左折
7: 右折
8: 車線変更(左)
9: 車線変更(右)
10: 転回(Uターン)

最終回答は 0 から 10 の数字1つのみで出力してください。説明は不要です。"""

# Legacy prompt names are kept for import compatibility on this branch.
PROMPT_STAGE1_TEMPLATE = SIMPLE_VIDEO_PROMPT_TEMPLATE
PROMPT_STAGE2_ROTATION_TEMPLATE = SIMPLE_VIDEO_PROMPT_TEMPLATE
PROMPT_STAGE2_NONROTATION_TEMPLATE = SIMPLE_VIDEO_PROMPT_TEMPLATE

# Model Loading Settings
MODEL_LOAD_TIMEOUT = 300
ENABLE_FLASH_ATTENTION = False
