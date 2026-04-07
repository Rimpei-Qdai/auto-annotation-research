# RAG推論案 実装草案

## 1. 位置づけ

この案は、グラフ推論案が十分に機能しなかった場合のフォールバックである。  
狙いは、今のプロジェクトで弱い `境界事例` と `少数クラス` に対して、過去の正解事例や誤認識パターンを検索して再推論し、最終ラベルを補正することにある。

---

## 2. 基本アイデア

VLM が最初に出したラベルをそのまま採用せず、以下を検索して再評価する。

- 手動ラベル付きの過去事例
- 過去の誤認識パターン
- 行動判定ルール

つまり、RAG を一般知識検索として使うのではなく、`この研究データ専用のケースメモリ` として使う。

---

## 3. 今のプロジェクトでやる意味

今のシステムでは、以下のような失敗が多いと考えられる。

- 直進に見えるため `減速` が `等速走行` に吸われる
- 左へ寄る動きが `左折` と `車線変更` で混同される
- `右折` が `等速走行` や `左折` に誤認識される

これらは、一般的なルールだけではなく、`似たケースを見て判断する` 方が改善しやすい。

---

## 4. 既存コードとの対応

### 活用する既存ファイル

- `new/annotation_tool/app.py`
  - `speed_diff` の事前計算
  - 全件アノテーション
  - `append_inference_log`
- `new/annotation_tool/heron_l2m_pipeline.py`
  - Level 3 の最終判定直前で RAG を差し込める
- `new/annotation_tool/heron_model_with_trajectory.py`
  - 推論の入口はそのまま利用可能
- `new/annotation_tool/annotated_samples_auto.csv`
  - 既存自動アノテーションの結果
- 手動アノテーション CSV
  - 正解付きケースメモリの元データ

### 追加する新規ファイル案

- `new/annotation_tool/case_memory.py`
- `new/annotation_tool/retrieval_index.py`
- `new/annotation_tool/rag_refiner.py`

---

## 5. 実装のイメージ

### Step 1: Case Memory を作る

各サンプルについて以下を保存する。

- `sample_id`
- `manual_label`
- `speed`
- `speed_diff`
- `gyro_z`
- `brake`
- `blinker_l`
- `blinker_r`
- `trajectory_summary`
- `short_reason`

例:

```json
{
  "sample_id": 42,
  "label": 3,
  "features": {
    "speed": 18.2,
    "speed_diff": -3.1,
    "gyro_z": 0.01,
    "brake": 1
  },
  "summary": "直進に近いがブレーキONかつ速度低下のため減速"
}
```

### Step 2: Failure Memory を作る

誤認識パターンをテンプレート化して保存する。

例:

- `deceleration_misread_as_constant`
- `lane_change_misread_as_left_turn`
- `right_turn_misread_as_constant`

これは完全自動でなく、最初は手作業で 10〜20 件作れば十分。

### Step 3: Retrieval Index を作る

最初は重い埋め込みモデルは使わず、数値特徴ベースでよい。

検索に使う特徴候補:

- `speed`
- `speed_diff`
- `gyro_z`
- `brake`
- `blinker`
- `predicted_label`

### Step 4: Level 3 直前で検索する

VLM が出した候補ラベルに対して、以下のように使う。

- 候補ラベルが `等速走行`
- しかし `speed_diff < 0`, `brake=ON`
- 類似ケース top-k を見ると `減速` が多い

なら再推論または再ランキングする。

---

## 6. 推論フロー

```text
動画 + センサ
  ↓
L2M/CoT 初回推論
  ↓
候補ラベル + 特徴量抽出
  ↓
Case Memory / Failure Memory 検索
  ↓
検索結果を添えて再推論
  ↓
最終ラベル
```

---

## 7. 再推論プロンプト案

```text
Initial prediction: 1 (等速走行)

Retrieved similar cases:
- Case A: 減速, speed_diff=-3.0, brake=ON
- Case B: 減速, speed_diff=-2.4, brake=ON
- Case C: 停止, speed=0.3, brake=ON

Failure pattern:
- 直進に見えるため減速が等速走行に誤認識されやすい

Re-evaluate the final label using the retrieved evidence.
```

---

## 8. 実装順序

### Phase 1

- `case_memory.py` 追加
- 手動ラベル付き CSV から case memory を構築
- 数値特徴だけで top-k 検索

### Phase 2

- `Failure Memory` を手動で作成
- Level 3 の再推論に検索結果を渡す
- `append_inference_log` に検索結果を保存

### Phase 3

- 埋め込みベース検索を追加
- グラフ特徴が使えるなら、Graph の中間概念も検索キーにする

---

## 9. 評価方法

### 主評価

- Accuracy
- Macro F1
- 少数クラス Recall

### 重点観察

- `減速`
- `右折`
- `車線変更`

### 追加で見る指標

- 再推論発火率
- 再推論での改善率
- 検索が原因の誤修正率

---

## 10. 期待効果

- グラフより短期で精度改善しやすい
- 少数クラスや境界事例に効きやすい
- `等速走行` への過剰集中を補正しやすい

---

## 11. リスク

- ケース数が少ないと検索が不安定になる
- 類似事例が悪いと誤誘導される
- 新規性としては Graph より弱い

---

## 12. この案をフォールバックにする理由

精度改善の即効性は高いが、研究の主張としては `類似事例検索による補正` に見えやすく、単独では新規性が弱い。  
そのため、まずは Graph を本命として進め、Graph が不安定な場合の現実的な改善策として RAG 案を使うのがよい。

