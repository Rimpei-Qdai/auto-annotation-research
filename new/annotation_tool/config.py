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

PROMPT_VERSION = "summary_v3_prompt_split"

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
画像1-4は時系列順の元フレームです。
画像5は fixed-scale の top-down trajectory summary、画像6は normalized geometry summary です。

【最重要ルール】
- まず画像5と画像6だけで判断してください。画像1-4は確認用です
- 緑線は直進基準、赤線は今後3秒間の予測軌道です
- A を選んでよいのは、次の3条件をすべて満たす場合だけです
  1. 画像5で赤線が十分に前へ伸びている
  2. 画像5でも画像6でも、赤線が緑線の近くにあり、左右どちらにも大きく離れていない
  3. 画像6で赤線がほぼまっすぐ上へ伸びており、明確な左曲がり・右曲がり・極短軌道がない
- 上の3条件のうち1つでも満たさないなら N を選んでください
- 少しでも左右の曲がり、横ずれ、極短軌道が見えるなら A ではなく N です
- 不確実でも A に逃げず、少しでも非直線の証拠があれば N を選んでください
- 画像1-4は、summary の判断を大きく覆すためではなく、道路文脈の確認にだけ使ってください

【センサーデータ】
速度: {speed} km/h
加速度(X/Y/Z): {acc_x}/{acc_y}/{acc_z} m/s²
Yaw rate: {gyro_z} rad/s

質問:
- A: 直線系（等速走行・加速・減速）
- N: 直線系以外（左回転系・右回転系・その他）

最終回答は A または N の1文字のみで回答してください。"""

STAGE2_ROUTE_PROMPT_TEMPLATE = """あなたは運転行動を分析するAIです。
画像1-4は時系列順の元フレームです。
画像5は fixed-scale の top-down trajectory summary、画像6は normalized geometry summary です。

【最重要ルール】
- まず画像5と画像6だけで判断してください。画像1-4は確認用です
- 緑線は直進基準、赤線は今後3秒間の予測軌道です
- 画像5と画像6の両方で赤線が極端に短い、またはほとんど前進していないなら D です
- 左右どちらかへ明確に曲がる、または終点が緑線から明確に左右へずれるなら R です
- 少しでも回転らしさがあるなら D ではなく R を選んでください
- 画像1-4は、交差点や道路文脈の確認にだけ使ってください

【センサーデータ】
速度: {speed} km/h
加速度(X/Y/Z): {acc_x}/{acc_y}/{acc_z} m/s²
Yaw rate: {gyro_z} rad/s

以下から1つ選んでください:
D: その他（停止・発進・その他）
R: 回転系（左回転系または右回転系）

最終回答は D または R の1文字のみで回答してください。"""

STAGE3_TURN_PROMPT_TEMPLATE = """あなたは運転行動を分析するAIです。
画像1-4は時系列順の元フレームです。
画像5は fixed-scale の top-down trajectory summary、画像6は normalized geometry summary です。

【最重要ルール】
- これは回転系だと分かっている前提です。B か C のどちらかを選んでください
- まず画像6を見てください
- 画像6で赤線の終点が緑線より左にあり、軌道全体も左へ曲がるなら B です
- 画像6で赤線の終点が緑線より右にあり、軌道全体も右へ曲がるなら C です
- 画像5でも終点の左右位置を確認し、画像6と同じ向きならその方向を選んでください
- 少しでも右向きの証拠が優勢なら C を選んでください。左へ固定しないでください
- 画像1-4は、道路文脈や交差点文脈の確認にだけ使ってください

【センサーデータ】
速度: {speed} km/h
加速度(X/Y/Z): {acc_x}/{acc_y}/{acc_z} m/s²
Yaw rate: {gyro_z} rad/s

以下から1つ選んでください:
B: 左回転系（左折・左車線変更）
C: 右回転系（右折・右車線変更・転回）

最終回答は B または C の1文字のみで回答してください。"""

# Model Loading Settings
MODEL_LOAD_TIMEOUT = 300  # seconds
ENABLE_FLASH_ATTENTION = False  # Set to True if flash-attn is installed
