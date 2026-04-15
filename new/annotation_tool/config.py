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
USE_GPU = True  # GPU有効
TORCH_DTYPE = "float16"  # GPUではfloat16で高速化
DEVICE_PREFERENCE = "cuda" if USE_GPU else "cpu"  # Device preference for model loading

# Video Processing Settings
NUM_FRAMES_TO_EXTRACT = 4  # Number of frames to extract from video (reduced for 8GB VRAM)
NUM_FRAMES_TO_USE = 4  # Number of frames to actually use (reduced to fit in memory)
FRAME_EXTRACTION_METHOD = "uniform"  # "uniform" or "temporal"
MAX_IMAGE_SIZE = 720  # Ver.4 レポートに合わせて 720p

# L2M+CoT Settings
USE_L2M_COT = False  # Ver.4 baseline は direct classification

# Generation Optimization Settings (速度最適化設定)
# これらの設定を調整することで、機能を維持しながら推論速度を向上できます
MAX_NEW_TOKENS_STANDARD = 50  # 標準推論の最大トークン数 (デフォルト: 100 → 50に削減)
MAX_NEW_TOKENS_L2M = 150  # L2M+CoT推論の最大トークン数 (デフォルト: 300 → 150に削減)
# 理由: アクション分類は0-10の数字1つだけなので、長い出力は不要
# 50トークンでも十分な推論結果が得られます

USE_KV_CACHE = True  # KVキャッシュの使用 (高速化)
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
    10: "転回(Uターン)"
}

# Prompt Template for LLaVA
PROMPT_TEMPLATE = """あなたは運転行動を分析するAIです。
以下のセンサーデータと画像から運転行動を分類してください。

【画像の見方】
- 赤い線: 車両の予測進路（今後3秒間の軌道）
- 緑の線: 車線中心または基準方向
- 赤い線が緑の線と平行 → 直進
- 赤い線が緑の線を横切る → 車線変更または旋回

【センサーデータ】
速度: {speed} km/h
加速度(X/Y/Z): {acc_x}/{acc_y}/{acc_z} m/s²
ブレーキ: {brake}

【注意事項】
- 4枚の画像は時系列順に並んでいます
- 赤い線と道路の曲率を比較してください
- 道路がカーブしていて、赤い線も同様にカーブしている場合は「等速走行」です
- 赤い線が道路と異なる方向に向かう場合のみ「車線変更」または「左右折」です

以下から1つ選んでください:
0: その他, 1: 等速走行, 2: 加速, 3: 減速, 4: 停止,
5: 発進, 6: 左折, 7: 右折, 8: 車線変更(左),
9: 車線変更(右), 10: 転回

数字のみで回答してください。"""

# Model Loading Settings
MODEL_LOAD_TIMEOUT = 300  # seconds
ENABLE_FLASH_ATTENTION = False  # Set to True if flash-attn is installed
