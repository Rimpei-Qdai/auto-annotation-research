# localhost で GPU 自動アノテーションを使う手順

ローカル PC で推論を回さず、`kiwi-rmt` 上のアノテーションツールを `localhost:8000` 経由で使うための手順です。

## 前提

- 研究室外から使う場合は `kiwi-rmt` を使う
- `~/.ssh/config` に `kiwi-rmt` と `uranus` の設定がある
- プロジェクトは `~/workspace/auto-annotation-research` に配置されている
- 動画ディレクトリ `new/filterd_video` は kiwi 側にも存在している

## 1. ローカルで動いている app を止める

ローカルで `python app.py` などを起動していると、GPU サーバーではなく手元の CPU で推論されます。

```bash
lsof -i :8000
kill <PID>
```

## 2. 必要なら最新コードを kiwi に同期する

`filterd_video` は容量が大きいため通常は除外します。

```bash
rsync -av --exclude='.git' --exclude='new/filterd_video' \
  --exclude='new/annotation_tool/annotation.log' \
  --exclude='new/annotation_tool/inference_process.jsonl' \
  /Users/rimpeihata/Desktop/auto-annotation-research/ \
  kiwi-rmt:~/workspace/auto-annotation-research/
```

## 3. kiwi 側でアノテーションツールを起動する

初回または Docker イメージを作り直したい場合:

```bash
ssh kiwi-rmt
cd ~/workspace/auto-annotation-research/new/annotation_tool
bash setup_kiwi.sh
```

すでにコンテナが作成済みなら、再ビルドせず再起動だけでもよい:

```bash
ssh kiwi-rmt
docker start hata_annotation
docker ps --filter name=hata_annotation
```

`docker ps` で `0.0.0.0:11577->8000/tcp` のような表示が出れば OK です。

## 4. ローカルから localhost:8000 に SSH トンネルする

別ターミナルで実行し、このコマンドは開いたままにします。

```bash
ssh -N -L 8000:localhost:11577 kiwi-rmt
```

## 4.5. アノテーション完了後にログを自動同期する

別ターミナルで次を実行しておくと、`kiwi` 上の自動アノテーションが完了したタイミングでログと結果 CSV がローカルへ自動コピーされます。

```bash
bash /Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/sync_kiwi_logs.sh --wait-complete
```

同期先:

- `/Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/kiwi_logs/annotation.log`
- `/Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/kiwi_logs/inference_process.jsonl`
- `/Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/kiwi_logs/annotated_samples_auto.csv`

## 5. ブラウザで開く

```text
http://localhost:8000
```

この UI 上で「自動アノテーション」を押すと、推論は `kiwi-rmt` 上の GPU で実行されます。

## 6. GPU 側で動いていることを確認する

別ターミナルでログを見る:

```bash
ssh kiwi-rmt 'docker logs -f hata_annotation'
```

CUDA 利用可否を確認する:

```bash
ssh kiwi-rmt 'docker exec hata_annotation python3 -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"'
```

`True` と `NVIDIA GeForce RTX 4090` が出れば GPU 側で実行できます。

## 停止方法

アノテーションツールを止める:

```bash
ssh kiwi-rmt 'docker stop hata_annotation'
```

SSH トンネルを止める:

- `ssh -N -L 8000:localhost:11577 kiwi-rmt` を実行しているターミナルで `Ctrl+C`

## よくあるハマりどころ

### ローカルで CPU 推論になってしまう

以下のようなログが出たら、ローカルで動いています。

- `Video path for prediction: /Users/...`
- `Using CPU`

この場合はローカルの `app.py` を止めて、上の手順で `kiwi-rmt` の UI に接続し直してください。

### `uranus.ait.kyushu-u.ac.jp` を解決できない

`ssh -N -L 8000:localhost:11577 kiwi-rmt` 実行時に `Could not resolve hostname uranus...` が出る場合は、ネットワーク接続か DNS 解決に問題があります。

確認項目:

- インターネット接続が有効か
- `~/.ssh/config` に `Host uranus` と `Host kiwi kiwi-rmt` があるか
- 学外ネットワークから学内ホストへ接続できる状態か

## 一番大事な注意

`python app.py` をローカルで直接起動しないこと。  
`localhost:8000` は必ず `ssh -N -L 8000:localhost:11577 kiwi-rmt` のトンネル経由で使うこと。
