#!/bin/bash
# kiwi GPUサーバー (172.16.0.76, RTX 4090 24GB) 用セットアップスクリプト
#
# 前提:
#   - kiwi に SSH 接続済みであること
#   - このスクリプトを new/annotation_tool/ から実行すること
#   - プロジェクトが ~/workspace/auto-annotation-research/ に配置されていること
#
# 使い方:
#   ssh -i ~/.ssh/id_rsa hata.rimpei@172.16.0.76
#   cd ~/workspace/auto-annotation-research/new/annotation_tool
#   bash setup_kiwi.sh

set -e

CONTAINER_NAME="hata_annotation"
IMAGE_NAME="auto-annotation:latest"
PROJECT_DIR="$HOME/workspace/auto-annotation-research"
# ホストポート: 1<ユーザーID> (例: UID=1234 → ポート11234)
HOST_PORT="1$(id -u)"

echo "=== kiwi GPU セットアップ ==="
echo "コンテナ名: $CONTAINER_NAME"
echo "プロジェクトディレクトリ: $PROJECT_DIR"
echo "ホストポート: $HOST_PORT"
echo ""

# プロジェクトディレクトリの存在確認
if [ ! -d "$PROJECT_DIR" ]; then
    echo "ERROR: $PROJECT_DIR が見つかりません。"
    echo "プロジェクトを以下のコマンドで配置してください:"
    echo "  mkdir -p ~/workspace"
    echo "  # ローカルから転送する場合:"
    echo "  # (ローカル) rsync -av --exclude='filterd_video' /path/to/auto-annotation-research/ hata.rimpei@172.16.0.76:~/workspace/auto-annotation-research/"
    exit 1
fi

# 既存コンテナの確認と停止
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "既存のコンテナ '$CONTAINER_NAME' を停止・削除します..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME"
fi

# Docker イメージをビルド
echo ""
echo "Docker イメージをビルドします..."
echo "(初回は flash-attn のコンパイルで 10〜20 分かかります)"
docker build -t "$IMAGE_NAME" .

# GPU の確認
echo ""
echo "GPU の確認:"
docker run --rm --gpus all "$IMAGE_NAME" python3 -c \
    "import torch; print(f'CUDA 利用可能: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"

# コンテナを起動
echo ""
echo "コンテナを起動します..."
docker run \
    --gpus all \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -v "${PROJECT_DIR}:/project" \
    -p "${HOST_PORT}:8000" \
    -dt \
    --name "$CONTAINER_NAME" \
    "$IMAGE_NAME"

echo ""
echo "=== 起動完了 ==="
echo "コンテナ ID:"
docker ps --filter "name=$CONTAINER_NAME" --format "  {{.ID}} ({{.Status}})"
echo ""
echo "ログを確認:"
echo "  docker logs -f $CONTAINER_NAME"
echo ""
echo "ローカルからアクセスするには SSH ポートフォワードを使用:"
echo "  (ローカル) ssh -L 8000:localhost:${HOST_PORT} -i ~/.ssh/id_rsa hata.rimpei@172.16.0.76"
echo "  ブラウザで http://localhost:8000 を開く"
echo ""
echo "コンテナ操作:"
echo "  docker stop $CONTAINER_NAME   # 停止"
echo "  docker start $CONTAINER_NAME  # 再起動"
echo "  docker logs -f $CONTAINER_NAME  # ログ確認"
