# agent.md

## 目的

このファイルは、別の AI エージェントがこのプロジェクトを操作するときの共通ルールを定義する。
作業前に必ず読み、ディレクトリの役割、生成物の扱い、GPU サーバー運用、コミット方針をそろえること。

---

## プロジェクト概要

- 目的: タクシー動画に対して運転行動ラベルを付与する研究用アノテーション基盤
- 主機能:
  - 手動アノテーション
  - VLM による自動アノテーション
- 現在の自動アノテーション:
  - FastAPI UI から実行
  - `Qwen/Qwen2-VL-7B-Instruct`
  - 軌道描画付きマルチフレーム入力
  - `L2M + CoT`
  - 軽量な Graph 推論を補助的に導入済み

---

## ルート構成

- `new/annotation_tool/`
  - アノテーションツール本体
  - 実装を触るときの主作業場所
- `sample/`
  - 入力 CSV の正本
  - 現在の `app.py` はここを読む
- `new/filterd_video/`
  - 動画データ
  - 容量が大きいので通常は同期対象から除外する
- `new/reports/`
  - 研究メモ、結果整理、設計書、提出物関連文書
  - 研究上の整理・報告は基本ここに置く
- `new/implementation_drafts/`
  - 未実装案や実装草案
  - 実験前のたたき台はここに置く
- `gpu_manual/`
  - GPU サーバー運用手順
  - `kiwi` 利用時は必ず参照する
- `presentation/`
  - 発表資料やその関連ファイル

---

## `new/annotation_tool/` の役割

### 主要コード

- `app.py`
  - FastAPI エントリポイント
  - CSV 読み込み、UI、推論 API、ログ保存を担当
- `config.py`
  - モデル設定、フレーム数、閾値、few-shot 例など
- `heron_model_with_trajectory.py`
  - モデルロード、フレーム抽出、軌道描画、通常推論 / L2M 呼び出し
- `vlm_runtime.py`
  - GPU / processor / device placement / `generate()` を閉じ込める専用ランタイム
  - 推論過程のコードから GPU 入出力詳細を分離するための境界
- `heron_l2m_pipeline.py`
  - Level 1 / Level 2 / Graph / Level 3 の段階推論
- `driving_graph.py`
  - 軽量な Graph 推論
  - 中間概念の生成、ラベル支持スコア算出
- `visual_prompting.py`
  - 軌道可視化
- `setup_kiwi.sh`
  - `kiwi` 上の Docker ビルド / コンテナ起動
- `sync_kiwi_logs.sh`
  - `kiwi` 側のログと結果 CSV をローカルへ同期

### 入力

- `sample/annotation_samples.csv`
  - 現在の入力 CSV の正本
- `new/filterd_video/`
  - 対応する動画群

### 出力

- `new/annotation_tool/annotated_samples_manual.csv`
  - 手動アノテーション結果
- `new/annotation_tool/annotated_samples_auto.csv`
  - 自動アノテーション結果
- `new/annotation_tool/annotation.log`
  - ローカルでアプリを動かしたときの通常ログ
- `new/annotation_tool/inference_process.jsonl`
  - 構造化された推論ログ
- `new/annotation_tool/kiwi_logs/`
  - `kiwi` から同期したログ置き場
  - 生成物なのでコミットしない
- `new/annotation_tool/auto_annotation_trajectory_frames/`
  - 軌道描画済みフレームの生成先
  - 生成物なのでコミットしない

---

## 研究ドキュメントの置き方

- 研究結果の整理や実験レビュー:
  - `new/reports/`
- 実装前の案、比較案、構想:
  - `new/implementation_drafts/`
- GPU 運用手順:
  - `gpu_manual/`

新しい文書を追加する場合は、以下を守ること。

- 結果や原因分析は `new/reports/`
- 未実装の提案書は `new/implementation_drafts/`
- 端末操作やサーバー利用手順は `gpu_manual/`
- 一時メモや生成ログをレポートとして置かない

---

## 頻出依頼の制約

この研究では、次の依頼が何度も出る。新しいエージェントは、これらを個別に解釈せず、
毎回ほぼ同じ品質で処理すること。

### 1. 「自動アノテーションが終わりました。結果をローカルに反映し、人手のアノテーションとの一致率を見て、結果をレポートしてください。原因と改善案まで」

この依頼を受けたら、最低限次を行うこと。

- 基本の接続先は `kiwi`
- `kiwi-rmt` はユーザーが明示したときだけ使う
- まず `new/annotation_tool/sync_kiwi_logs.sh --once` 相当で結果を同期する
- 同期対象は少なくとも:
  - `annotated_samples_auto.csv`
  - `annotation.log`
  - `inference_process.jsonl`
- ローカルの `new/annotation_tool/annotated_samples_auto.csv` を更新する
- 人手ラベルとの比較では、可能なら:
  - 11分類
  - 4分類
  の両方を出す
- レポートには必ず含める:
  - 有効予測件数
  - 未出力 sample ID
  - クラス分布
  - 一致率
  - 主な誤りパターン
  - 原因
  - 次の改善案

重要:

- CSV と `inference_process.jsonl` が食い違う場合は、最新 run の `jsonl` を優先し、その旨をレポートに明記する
- レポートは必ず `new/reports/` に日付付きで保存する
- 生成物の CSV やログを安易にコミットしない

### 2. 「試しに描画したものを出力してみて」

この依頼は、人間向けの説明画像ではなく、**現在のモデルが本当に見ている入力**を可視化する依頼として扱うこと。

- 既存の実装経路を使い、別の描画コードを新設しない
- 代表 sample は、指定がなければ典型例を5件程度選ぶ
- 保存先は `new/reports/` 配下の日付付きディレクトリにする
- 画像は基本:
  - `frame_1.png` から `frame_4.png`
  - summary 系入力
  のみを保存する
- `contact_sheet` や説明用コラージュは、ユーザーが明示したときだけ作る

重要:

- 元画像に軌道を重ねた旧方式と、summary 分離後の方式を混同しない
- 「今の実装が何枚入力か」を必ず確認してから出力する

### 3. 「改善案を実装してください」

この依頼を受けたら、まず最新の比較レポートとログから、**今いちばん支配的な失敗要因を1つだけ選ぶ**こと。

- 一度に多くを直そうとしない
- 1つの改善意図ごとにコミットを分ける
- prompt を主に変更する場合は、先に文献根拠を確認する
- 文献的に支えられていない場合は、その変更が repo 内観測ベースの仮説であることを明示する
- 実装後は少なくとも:
  - `py_compile`
  - 関連 unit test
  - 小さな smoke test
  のいずれかを走らせる

重要:

- 改善実装の前に、対象 branch の状態と `git status` を確認する
- 自分の変更と無関係な生成物やレポートをコミットに混ぜない
- 改善の前提となるレポートが古い場合は、先に同期・比較をやり直す

### 4. 「現状を把握して、文献を調査し抜本的な改善案を提案してください」

この依頼は、単なる感想ではなく、**repo の現状把握 + 文献調査 + 実装順つき提案**として処理すること。

- まず現在のコードと最新 run のログを読む
- 次に文献を調べる
- 文献は原則として一次情報を優先する
- レポートでは必ず区別する:
  - repo の観測事実
  - コードやログからの推論
  - 文献に支えられた主張
- 提案は必ず実装順に落とす

重要:

- 「とりあえず prompt を変える」「モデルを変える」だけで終わらせない
- 何を変えるか、なぜ効くか、どう評価するかまで書く
- 調査メモや提案書も `new/reports/` に日付付きで保存する

### 5. 頻出依頼全体の共通制約

- 既定のリモートは `kiwi`
- レポートや比較結果は `new/reports/`
- 実装前の草案は `new/implementation_drafts/`
- prompt 変更は、できるだけ文献根拠を確認してから行う
- 描画確認は「人間向けの見やすさ」ではなく「モデル入力の忠実な再現」を優先する
- 精度レポートでは、単なる accuracy だけでなく、分布崩壊や bias の有無を必ず確認する

---

## Graph 実装の現状

- `driving_graph.py` は軽量な概念グラフ実装である
- 現在の粒度は「理想系の本格 graph reasoning」ではなく、最小構成の補助的 Graph 推論
- 主なノード:
  - `speed_state`
  - `signal_state`
  - `trajectory_direction`
  - `turn_intensity`
  - `lane_crossing_state`
  - `intersection_state`
  - `stop_likelihood`
  - `consistency_check`
  - `road_shape`
- 主な役割:
  - 中間概念を整理する
  - ラベル支持スコアを作る
  - Level 3 プロンプトへ Graph 要約を渡す
  - 後処理で `1` への偏りを補正する

Graph を強化する場合は、まず `new/reports/研究設計案_Graph推論.md` を読むこと。

---

## 推論過程の修正方針

- この研究の本筋は、`L2M / CoT / Graph / prompt design / label selection` などの推論過程を改善することである
- したがって、推論改善の作業では、原則として `heron_l2m_pipeline.py` や `driving_graph.py` のみを修正対象にする
- GPU 入出力、processor 呼び出し、device placement、`generate()` 実行経路は `vlm_runtime.py` に隔離されている
- 推論過程の改善のために、`processor` や `model.generate()` を直接触る実装を `heron_l2m_pipeline.py` 側へ再び書き戻さないこと
- `heron_model_with_trajectory.py` は、動画からフレームを作ることと、推論ランタイム / 推論パイプラインを接続することに責務を限定する

判断基準:

- 推論ロジックを変えたい:
  - `heron_l2m_pipeline.py`
  - `driving_graph.py`
- GPU や Qwen の入出力経路を直したい:
  - `vlm_runtime.py`
  - 必要最小限で `heron_model_with_trajectory.py`

重要:

- 推論過程の修正が GPU の不安定化に波及しないよう、研究用のロジック変更と実行基盤変更は分離する
- 新しい推論案を試すときは、まず `heron_l2m_pipeline.py` のみで完結できないかを先に検討する
- GPU 周りに触る必要がある場合は、「推論改善のため」ではなく「実行基盤修正のため」と明示して扱う

---

## GPU サーバー `kiwi` の運用ルール

### 基本方針

- 研究室外からは `kiwi-rmt` を使う
- ローカルで `python app.py` を直接起動しない
- GPU 推論は `kiwi` 上のコンテナで実行し、ローカルは `localhost:8000` を SSH トンネルで開く

### 起動の基準

- Python コードだけ変えた場合:
  - `rsync` でソース同期
  - `docker restart hata_annotation`
- `Dockerfile` や `requirements.txt` を変えた場合:
  - `setup_kiwi.sh` を使って再ビルド

### よく使う手順

- 同期手順:
  - `gpu_manual/localhostでGPU自動アノテーションを使う手順.md`
- ログ同期:
  - `new/annotation_tool/sync_kiwi_logs.sh --wait-complete`

### `rsync` で除外すべきもの

通常のコード同期では少なくとも以下を除外すること。

- `.git`
- `new/filterd_video`
- `new/annotation_tool/__pycache__/`
- `new/annotation_tool/auto_annotation_trajectory_frames/`
- `new/annotation_tool/kiwi_logs/`
- `new/annotation_tool/annotation.log`
- `new/annotation_tool/inference_process.jsonl`
- `new/annotation_tool/annotated_samples_auto.csv`

理由:

- 動画は重い
- ログや結果 CSV は環境依存の生成物
- `__pycache__` や軌道フレームは `kiwi` 側で root 所有になることがあり、`rsync` 失敗の原因になる

---

## ログの見方

- ローカルで見ている `new/annotation_tool/annotation.log` は、ローカル実行分しか入らない
- `kiwi` 上で実行した本体ログは、`kiwi` 側の `~/workspace/auto-annotation-research/new/annotation_tool/` にある
- `sync_kiwi_logs.sh` を使うと、`new/annotation_tool/kiwi_logs/` にローカルコピーできる

ログの役割:

- `annotation.log`
  - 通常ログ + 例外スタックトレース
- `inference_process.jsonl`
  - `level1 / level2 / graph / level3 / final_category / confidence` などの構造化ログ

---

## 実装時の注意

- 作業前に `git status` を確認する
- 作業ツリーは汚れている前提で扱う
- 自分のタスクに関係ない変更はコミットしない
- 生成物やログは原則コミットしない
- CSV 結果ファイルは研究結果そのものなので、意図せず上書きしない
- `sample/annotation_samples.csv` は入力の正本なので慎重に扱う
- `annotated_samples_auto.csv` は生成物なので、コード変更と同じコミットに安易に含めない

---

## コミット方針

- 1つの目的ごとにコミットを分ける
- ドキュメント追記とコード修正は、意味が違うなら分ける
- 既存のユーザー変更を巻き込まない
- 何を直したかが分かる短い英語コミットメッセージを使う

---

## まず読むべきファイル

新しいエージェントは、少なくとも以下を順に確認すること。

1. `agent.md`
2. `new/annotation_tool/README.md`
3. `new/annotation_tool/config.py`
4. `new/annotation_tool/app.py`
5. `new/annotation_tool/heron_model_with_trajectory.py`
6. `new/annotation_tool/heron_l2m_pipeline.py`
7. 必要に応じて `new/reports/` の関連メモ
8. `kiwi` を使うなら `gpu_manual/localhostでGPU自動アノテーションを使う手順.md`

注意:

- `new/annotation_tool/README.md` は実装より古い記述を含むことがある
- モデル名、フレーム数、推論構成の正本は `config.py` と実装コード側とみなす

---

## 現時点で重要な研究メモ

- `new/reports/Qwen_device不一致修正メモ.md`
  - GPU 推論の `cpu/cuda` 不一致と、その修正履歴
- `new/reports/研究設計案_Graph推論.md`
  - Graph 推論の設計と根拠論文
- `new/reports/研究設計案_RAGフィードバック推論.md`
  - RAG 拡張の構想
- `new/reports/自動アノテーション精度比較レポート.md`
  - バージョン比較
- `new/reports/20260407_アノテーション結果レビュー.md`
  - 最新レビューがある場合は参照する

---

## 禁止事項

- ローカルで GPU サーバー用のアノテーションを回したと誤認しない
- `localhost:8000` を使うとき、SSH トンネルなしでローカル app を開かない
- `kiwi` のログや生成物をコード同期で上書きしない
- 研究結果 CSV を意図なく削除・初期化しない
- ユーザーの未コミット変更を勝手に巻き戻さない

---

## このファイルの更新ルール

以下が変わったら `agent.md` も更新すること。

- ディレクトリ構成
- 入力 / 出力ファイルの正本
- GPU サーバー運用手順
- 主要モデル
- 推論フローの構成
- ログ保存場所
