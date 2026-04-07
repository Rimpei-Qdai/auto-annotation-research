# NVIDIA PyTorch base image (CUDA 12.3, PyTorch 2.2, Python 3.10)
# RTX 4090 (Ada Lovelace, sm_89) に対応
FROM nvcr.io/nvidia/pytorch:24.10-py3

# OpenCV に必要なシステムライブラリ + ffmpeg（mp4動画読み込みに必要）
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 依存関係のみを先にインストール（キャッシュ活用）
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Qwen2-VL の動画処理に必要
RUN pip install --no-cache-dir qwen-vl-utils

# ベースイメージの cv2 を完全削除してから opencv-python-headless を再インストール
# 4.10.0 は DictValue が完全削除済みで ffmpeg バンドル済み
RUN pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true && \
    rm -rf /usr/local/lib/python3.10/dist-packages/cv2 && \
    pip install --no-cache-dir "opencv-python-headless==4.10.0.84" && \
    pip install --no-cache-dir "numpy<2"

# flash-attention: RTX 4090 で高速化（コンパイルに時間がかかるため最後に実施）
RUN pip install flash-attn --no-build-isolation 2>/dev/null \
    || echo "WARNING: flash-attn のインストールに失敗しました。通常の attention を使用します。"

# プロジェクトはボリュームマウントで提供されるためここではコピーしない
# setup_kiwi.sh の docker run コマンドで -v オプションによりマウントされる

WORKDIR /project/new/annotation_tool

EXPOSE 8000

CMD ["python3", "app.py"]
