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
USE_VLM_DIRECT = False  # direct path は VLM を使わず deterministic classifier に切り替える

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

# Prompt Templates for staged macro classification
PROMPT_STAGE1_TEMPLATE = """あなたは運転行動を分析するAIです。
5枚の画像を見て、まず大きな運動の種類を1つだけ選んでください。

【画像構成】
- 画像1〜4: 元の時系列フレームです
- 画像5: trajectory summary です。白背景の上に、今後3秒の進路だけを簡潔に描いています

【trajectory summary の意味】
- 緑の破線: 直進基準
- 赤〜黄の太い軌道: 今後3秒間の予測進路
- 黄色い矢印: 3秒後の進行方向
- 軌道がほとんど伸びていないときは、ほぼ停止に近い可能性があります

【センサ要約】
速度: {speed} km/h
加速度(X/Y/Z): {acc_x}/{acc_y}/{acc_z} m/s²
Yaw rate: {gyro_z} rad/s
Latitude: {latitude}
Longitude: {longitude}
visible trajectory points: {visible_trajectory_points}

【候補】
{candidate_lines}

【判断ルール】
- 元画像1〜4は道路状況を見るため、trajectory summary は進路形状を見るために使ってください
- S: 主に前後方向の変化。等速・加速・減速のような直線系
- R: 主に左右方向の変化。左折・右折・車線変更・転回のような回転系
- O: 停止・発進・その他
- 軌道がほとんど伸びていない、または speed が極端に低いときは O を強く疑ってください
- trajectory summary の終点矢印が左右を向くときは S より R を優先してください
- 候補以外は選ばないでください

最終回答は S / R / O の1文字のみで出力してください。"""

PROMPT_STAGE2_ROTATION_TEMPLATE = """あなたは運転行動を分析するAIです。
このサンプルは回転系だと分かっています。左回転か右回転かを1つだけ選んでください。

【画像構成】
- 画像1〜4: 元の時系列フレーム
- 画像5: trajectory summary

【trajectory summary の意味】
- 緑の破線: 直進基準
- 赤〜黄の太い軌道: 今後3秒間の予測進路
- 黄色い矢印: 3秒後の進行方向

【センサ要約】
速度: {speed} km/h
Yaw rate: {gyro_z} rad/s
Latitude: {latitude}
Longitude: {longitude}

【候補】
{candidate_lines}

【判断ルール】
- L: 左折・左車線変更のような左方向変化
- R: 右折・右車線変更・転回のような右方向変化
- trajectory summary の終点矢印と軌道の曲がる向きを最優先してください
- yaw の符号は補助的に使ってください
- 候補以外は選ばないでください

最終回答は L / R の1文字のみで出力してください。"""

PROMPT_STAGE2_NONROTATION_TEMPLATE = """あなたは運転行動を分析するAIです。
このサンプルは非回転系です。直線系かその他かを1つだけ選んでください。

【画像構成】
- 画像1〜4: 元の時系列フレーム
- 画像5: trajectory summary

【trajectory summary の意味】
- 緑の破線: 直進基準
- 赤〜黄の太い軌道: 今後3秒間の予測進路
- 黄色い矢印: 3秒後の進行方向
- 軌道がほとんど伸びていないときは、ほぼ停止に近い可能性があります

【センサ要約】
速度: {speed} km/h
加速度(X/Y/Z): {acc_x}/{acc_y}/{acc_z} m/s²
Yaw rate: {gyro_z} rad/s
Latitude: {latitude}
Longitude: {longitude}
visible trajectory points: {visible_trajectory_points}

【候補】
{candidate_lines}

【判断ルール】
- A: 直線系（等速・加速・減速）
- D: その他（停止・発進・その他）
- 速度が極端に低い、進みがほとんど見えない場合は D を優先してください
- trajectory summary で進路が左右へ大きく曲がっていないことを確認してください
- それ以外で主に前後方向の変化なら A を選んでください
- 候補以外は選ばないでください

最終回答は A / D の1文字のみで出力してください。"""

# Model Loading Settings
MODEL_LOAD_TIMEOUT = 300  # seconds
ENABLE_FLASH_ATTENTION = False  # Set to True if flash-attn is installed
