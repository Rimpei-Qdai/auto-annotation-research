# Qwen device 不一致修正メモ

## 概要

2026-04-01 の GPU サーバー実行では、自動アノテーション結果が `1` と `4` に偏るというより、後半は `0` に崩れる状態になっていた。
ログを確認したところ、Qwen2-VL の L2M 推論中に `cpu` と `cuda:0` のデバイス不一致が発生し、Level 1 / Level 2 / Level 3 の生成が失敗していた。

## 原因

`Qwen2-VL-7B-Instruct` は `device_map="auto"` でロードしていたが、推論時に processor の出力テンソルを一律 `self.device` (`cuda`) に移動していた。

このため、

- モデル側は `Accelerate` により一部の重みが CPU 側を含む配置になる
- 入力だけが強制的に GPU に移動する
- embedding 層などで `cpu` と `cuda:0` の不一致が発生する

という状態になっていた。

実際のエラーは以下。

```text
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cpu and cuda:0!
```

## 影響

L2M パイプラインでは生成失敗時に保守的な既定値へフォールバックする実装になっているため、視覚情報が落ちると以下のような偏りが発生する。

- Level 1 は `STRAIGHT / PARALLEL / NO_SHIFT / NO`
- Level 2 は `MIXED / STABLE / CONSISTENT`
- Level 3 も失敗すると `category_id = 0`
- ただし停止条件だけは後処理ルールで `4` に補正される

このため、結果として `1` や `4` に寄る、あるいは `0` が大量に出る状態になっていた。

## 実装した修正

### 0. 単一 GPU 前提のロード戦略へ変更

`kiwi` は `RTX 4090 (24GB)` の単一 GPU 環境であり、現在使っている `Qwen2-VL-7B-Instruct` はこの環境に収まる。

そのため、`device_map="auto"` による曖昧な自動配置をやめ、モデル全体を明示的に単一デバイスへ載せる方針に変更した。

これにより、

- モデルの一部だけが CPU に残る
- `model.device` と embedding 層の実デバイスが食い違う
- 実行時に `cpu/cuda` の境界でエラーになる

といった不安定要因を減らす。

### 1. モデル配置戦略の判定を追加

`new/annotation_tool/heron_model_with_trajectory.py` に `uses_device_map` プロパティを追加した。

目的:

- モデルが `hf_device_map` を持つかを判定する
- `Accelerate` 管理モデルか単一デバイスモデルかを分岐できるようにする

### 2. 入力テンソルの移動処理を共通化し、基準点を embedding 層に変更

`new/annotation_tool/heron_model_with_trajectory.py` に `prepare_inputs_for_generation()` を追加した。

さらに `generation_device` プロパティを追加し、入力を単に `self._device` へ送るのではなく、`model.get_input_embeddings()` の重みが存在する実デバイスへ合わせるようにした。

動作:

- まず embedding 層の実デバイスを調べる
- 取得できない場合は `hf_device_map` や先頭パラメータのデバイスへフォールバックする
- processor 出力はそのデバイスへ統一してから `generate()` に渡す

これにより、`model.device` 表示や `device_map` の有無に振られず、`input_ids` と embedding 層の不一致を避ける。

### 3. 通常推論経路を修正

`new/annotation_tool/heron_model_with_trajectory.py` の通常推論経路で、processor 出力に直接 `.to(self.device)` をかける処理を削除し、`prepare_inputs_for_generation()` を通すように変更した。

### 4. L2M 推論経路を修正

`new/annotation_tool/heron_l2m_pipeline.py` でも同様に、以下の処理

```python
inputs = {k: v.to(self.annotator.device) for k, v in inputs.items()}
```

をやめて、

```python
inputs = self.annotator.prepare_inputs_for_generation(inputs)
```

へ変更した。

これにより、通常推論と L2M 推論の両方で同じデバイス処理ルールを使うようにした。

### 5. ログ出力を追加

モデルロード時に、最終的に `generate()` 入力をどのデバイスへ送るかをログへ出すようにした。

これにより、再度不一致が出た場合でも、

- モデルをどの戦略でロードしたか
- embedding 層がどこにあるか
- 入力をどこへ送る設計だったか

を追跡しやすくなる。

## この修正にした理由

今回の問題は Graph 推論や後処理ルールの設計より前に、Qwen2-VL の生成処理そのものが失敗していたことが本質だった。

したがって優先順位としては、

1. まず Level 1 / Level 2 / Level 3 の生成を正常完走させる
2. その上で Graph 補助の効果を見る

の順に戻す必要があった。

`device_map="auto"` をやめて単純に `.to("cuda")` に寄せる案もあるが、Qwen2-VL 7B の将来的な VRAM 余裕や運用安定性を考えると、`device_map` を前提に入力処理だけ正す方が安全で影響範囲も小さいと判断した。

## 確認したこと

ローカルで以下を確認した。

```bash
python3 -m py_compile new/annotation_tool/heron_model_with_trajectory.py new/annotation_tool/heron_l2m_pipeline.py
```

加えて、`prepare_inputs_for_generation()` について簡単なスモークテストを行い、

- `hf_device_map` があるときは入力をそのまま返す
- 単一デバイスモデルのときは `.to(device)` を適用する

ことを確認した。

## 関連 PR

- PR #3
- ブランチ: `codex/graph-reasoning`
- 修正コミット: `ef67011`

## 次に見るべき点

この修正後は、まず kiwi 上で再度全件自動アノテーションを回し、以下を確認する必要がある。

- `annotation.log` に `Expected all tensors to be on the same device` が再発していないか
- `inference_process.jsonl` に `level1 / level2 / graph / level3` が正常に保存されているか
- そのうえで、まだ `1` と `4` に偏るのか、それとも Graph 候補が効き始めるのか

つまり、この修正は「分類性能改善」そのものではなく、「Graph 改良を正しく評価できる状態へ戻すための前提修正」である。
