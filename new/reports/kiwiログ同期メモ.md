# kiwi ログ同期メモ

`kiwi` 上で回した自動アノテーションのログは、ローカルの `annotation.log` とは別物です。  
GPU サーバー側で実行した本体ログは `kiwi` 上の `~/workspace/auto-annotation-research/new/annotation_tool/` に保存されます。

## どこに保存されるか

- `annotation.log`
- `inference_process.jsonl`
- `annotated_samples_auto.csv`

## コンテナを kill したら消えるか

基本的には消えません。  
`setup_kiwi.sh` では `-v "${PROJECT_DIR}:/project"` でホスト側のプロジェクトディレクトリをコンテナにマウントしているため、ログはコンテナの中ではなく kiwi ホスト側のファイルとして残ります。

消える可能性があるのは次のような場合です。

- kiwi ホスト上のファイル自体を手動で削除した
- 別の `rsync` やコピーで上書きした
- ログファイルを `: > annotation.log` のように明示的に truncate した

`docker stop`、`docker kill`、`docker rm` だけであれば、通常はログファイルは残ります。

## ローカルへ持ってくる方法

同期用スクリプト:

- `/Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/sync_kiwi_logs.sh`

1回だけ同期する:

```bash
bash /Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/sync_kiwi_logs.sh --once
```

アノテーション完了を待って、自動で同期する:

```bash
bash /Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/sync_kiwi_logs.sh --wait-complete
```

定期同期する:

```bash
bash /Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/sync_kiwi_logs.sh --watch
```

同期先:

- `/Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/kiwi_logs/annotation.log`
- `/Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/kiwi_logs/inference_process.jsonl`
- `/Users/rimpeihata/Desktop/auto-annotation-research/new/annotation_tool/kiwi_logs/annotated_samples_auto.csv`

`--watch` は `5` 秒おきに `rsync` で追従します。停止したいときは `Ctrl+C` です。

`--wait-complete` は `annotation.log` の `Starting auto-annotation for ...` と `Auto-annotation completed: ...` を監視し、新しい完了ログを検知したら 1 回だけ同期して終了します。  
ブラウザで全自動実行を押す前に、このコマンドを別ターミナルで起動しておくと運用しやすいです。
