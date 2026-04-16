# config.py
"""
Moondream2 VLM Configuration for Automatic Annotation
"""

# VLM Model Settings
# Multi-frame video understanding models
# Option 1: Qwen2-VL-2B (Multi-frame, 4GB VRAM, faster, good for 8GB GPU)
# Option 2: Qwen3-VL-2B (Multi-frame, newer VLM generation, similar scale)
# Option 3: Qwen2-VL-7B (Multi-frame, 8GB+ VRAM, better accuracy, requires more memory)
# Note: Moondream2 is NOT supported (single-frame only)
# Note: Heron has Windows tokenizer compatibility issues
HERON_MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
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

# Macro 4-class mapping used for Qwen3-VL comparison runs.
# We keep representative 11-class IDs for CSV compatibility:
# A -> 1 (直線系), B -> 6 (左回転系), C -> 7 (右回転系), D -> 4 (その他)
MACRO_OUTPUT_TO_LABEL = {
    "A": 1,
    "B": 6,
    "C": 7,
    "D": 4,
}

MACRO_OUTPUT_NAMES = {
    "A": "直線系",
    "B": "左回転系",
    "C": "右回転系",
    "D": "その他",
}

PROMPT_VERSION = "summary_v5_summary_only"

# Legacy one-shot macro prompt kept for fallback / comparison.
PROMPT_TEMPLATE = """あなたは運転行動を分析するAIです。
以下のセンサーデータと画像から運転行動を4分類してください。

【画像の見方】
- 画像1-4: 時系列順の元フレームです
- 画像5: fixed-scale の top-down trajectory summary です
- 画像6: 左右の曲がりを強調した normalized geometry summary です
- 画像5,6 の緑線は直進基準、赤線は今後3秒間の予測軌道です
- 画像5,6 を主に見て、画像1-4は道路文脈の確認に使ってください

【センサーデータ】
速度: {speed} km/h
加速度(X/Y/Z): {acc_x}/{acc_y}/{acc_z} m/s²
Yaw rate: {gyro_z} rad/s

以下から1つ選んでください:
A: 直線系（等速走行・加速・減速）
B: 左回転系（左折・左車線変更）
C: 右回転系（右折・右車線変更・転回）
D: その他（停止・発進・その他）

A, B, C, D のいずれか1文字のみで回答してください。"""

STAGE1_PROMPT_TEMPLATE = """あなたは運転行動を分析するAIです。
画像1は fixed-scale の top-down trajectory summary、画像2は normalized geometry summary です。

【見る対象】
- この2枚だけで判断してください
- 緑線は直進基準、赤線は今後3秒間の予測軌道です
- LEFT / RIGHT アンカーは車両基準の左 / 右を示します

【判断】
- A = 前へ伸びる直線移動
- N = 回転系またはその他
- A にするのは、赤線が十分に前へ伸び、終点が中央付近にあり、LEFT / RIGHT のどちらにも明確には寄っていないときです
- N にするのは、終点が LEFT / RIGHT のどちらかへ明確に寄る、軌道全体が一方向へ強く曲がる、または軌道が短すぎるときです
- 軽い曲がりや軽い横ずれだけなら A を選んでください
- 少しでも明確な横方向の偏りや stop-like な短さがあれば N を選んでください

【センサーデータ】
速度: {speed} km/h
Yaw rate: {gyro_z} rad/s

最終回答は A または N の1文字のみで回答してください。"""

STAGE2_ROUTE_PROMPT_TEMPLATE = """あなたは運転行動を分析するAIです。
画像1は fixed-scale の top-down trajectory summary、画像2は normalized geometry summary です。

【見る対象】
- この2枚だけで判断してください
- 緑線は直進基準、赤線は今後3秒間の予測軌道です
- LEFT / RIGHT アンカーは車両基準の左 / 右を示します

【判断】
- R = 回転系
- D = その他
- R にするのは、終点が LEFT / RIGHT のどちらかへ明確に寄り、軌道全体も同じ向きへ曲がるときです
- D にするのは、軌道が短い、前進が弱い、または左右の回転方向がまだ明確でないときです
- 曖昧なら R ではなく D を選んでください

【センサーデータ】
速度: {speed} km/h
Yaw rate: {gyro_z} rad/s

最終回答は R または D の1文字のみで回答してください。"""

STAGE3_TURN_PROMPT_TEMPLATE = """あなたは運転行動を分析するAIです。
画像1は fixed-scale の top-down trajectory summary、画像2は normalized geometry summary です。

【見る対象】
- この2枚だけで判断してください
- LEFT / RIGHT アンカーは車両基準の左 / 右を示します
- 緑線は直進基準、赤線は今後3秒間の予測軌道です

【判断】
- B = 左回転系
- C = 右回転系
- 画像2で終点が LEFT 側にあり、軌道全体も左へ曲がるなら B です
- 画像2で終点が RIGHT 側にあり、軌道全体も右へ曲がるなら C です
- 画像1でも同じ向きを確認し、2枚で一致する側を選んでください
- B と C は対称です。C をデフォルトにしないでください
- 曖昧なときは、終点が中央線からどちら側へ遠いかで選んでください

【センサーデータ】
速度: {speed} km/h
Yaw rate: {gyro_z} rad/s

最終回答は B または C の1文字のみで回答してください。"""

# Model Loading Settings
MODEL_LOAD_TIMEOUT = 300  # seconds
ENABLE_FLASH_ATTENTION = False  # Set to True if flash-attn is installed
