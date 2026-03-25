# 研究設計案3: Retrieval-Augmented Feedback Inference

**作成日**: 2026年3月12日  
**対象プロジェクト**: タクシー動画の運転行動自動アノテーション

---

## 1. 目的

本設計案は、VLM の初回推論で迷いやすい境界事例に対し、
**過去の正解事例・失敗事例・判定ルールを検索し、その結果を使って再推論する**
フィードバック機構を導入することを目的とする。

本研究のデータでは、以下のような境界事例が多い。

- 直進しながら減速
- 直進しながら加速
- 左折と見えるが実際は車線変更
- 曲率は小さいが右折に近い挙動
- 停止直前の減速と停止の境界

これらは、一般知識だけではなく、
**このデータセットに固有の近傍事例**
を参照した方が改善しやすい。

---

## 2. 中核アイデア

RAG を単なる知識補完ではなく、
**ケースベース誤り補正器**
として使う。

検索対象は一般文書ではなく、以下の 3 層からなる。

- `Case Memory`: 正解付き過去サンプル
- `Failure Memory`: 典型的な誤認識パターン
- `Rule Memory`: 運転行動判別ルール

---

## 3. 知識ベース設計

### 3.1 Case Memory

各サンプルから以下を保存する。

- sample_id
- 手動ラベル
- 画像要約
- センサ要約
- 軌道特徴
- 交差点/車線関係
- 説明文

#### 例

```json
{
  "sample_id": 42,
  "label": "減速",
  "features": {
    "speed": 18.2,
    "acc_x": -2.1,
    "brake": 1,
    "trajectory_curvature": 0.01,
    "lateral_displacement": 0.2
  },
  "summary": "直進に近い軌道だが、ブレーキONかつ負の加速度で減速"
}
```

### 3.2 Failure Memory

誤認識パターンをテンプレートとして保存する。

#### 例

```json
{
  "pattern": "deceleration_misread_as_constant_speed",
  "conditions": [
    "trajectory is almost straight",
    "acc_x is negative",
    "brake is on"
  ],
  "warning": "Do not classify as constant speed only because the trajectory is straight."
}
```

### 3.3 Rule Memory

手工的な簡潔ルールを文章として保持する。

- `speed≈0 and trajectory short -> stop`
- `right blinker + right curvature + intersection -> right turn`
- `small curvature + lane crossing + no intersection -> lane change`

---

## 4. クエリ設計

初回推論後、以下をクエリとして RAG に渡す。

- 初回候補ラベル
- センサ特徴
- 軌道特徴
- 信号特徴
- 交差点/車線関係
- 推論理由

### クエリ例

```json
{
  "candidate_label": "等速走行",
  "speed": 12.0,
  "acc_x": -1.9,
  "brake": 1,
  "trajectory_curvature": 0.01,
  "lateral_displacement": 0.2,
  "blinker_r": 0,
  "blinker_l": 0,
  "reasoning": "trajectory is straight and stable"
}
```

このクエリに対して、Top-k の類似事例と関連ルールを返す。

---

## 5. フィードバック推論フロー

```text
入力動画 + センサ
    ↓
VLM 初回推論
    ↓
クエリ生成
    ↓
RAG 検索
    ↓
類似事例 / 失敗パターン / ルール取得
    ↓
フィードバック付き再推論
    ↓
最終ラベル
```

---

## 6. 再推論プロンプト例

```text
Initial prediction: 1 (constant speed)

Retrieved similar cases:
1. Case #42 -> deceleration
   Reason: straight trajectory, brake ON, negative acceleration
2. Case #18 -> stop
   Reason: near-zero speed and very short trajectory

Retrieved failure pattern:
- deceleration is often mistaken for constant speed when trajectory is straight

Driving rule:
- If brake is ON and acceleration is negative, reconsider deceleration.

Re-evaluate the final label based on the retrieved evidence.
Output only one label.
```

---

## 7. このプロジェクトに合わせた利点

### 7.1 少数クラスの救済

右折や車線変更はデータ数が少なく、VLM 単独では検出しづらい。  
RAG によって、少数クラスの具体例を推論時に参照させられる。

### 7.2 境界事例の補正

加速・減速・停止直前は、静的な一般ルールだけでは分かれにくい。  
近い事例を参照する方が有効である。

### 7.3 誤認識知識の蓄積

単に正解事例を溜めるだけでなく、
`何をどう誤認識しやすいか`
を Failure Memory に残せる点が重要である。

---

## 8. 実装対象

### 8.1 追加ファイル候補

- `new/annotation_tool/retrieval_index.py`
- `new/annotation_tool/case_memory.py`
- `new/annotation_tool/feedback_prompting.py`

### 8.2 既存ファイルの修正候補

- `new/annotation_tool/heron_l2m_pipeline.py`
- `new/annotation_tool/app.py`

### 8.3 実装内容

#### `case_memory.py`

- CSV と既存推論ログから case memory を構築
- 正解/誤認識テンプレートを整形

#### `retrieval_index.py`

- 数値特徴ベースの近傍検索
- 文章埋め込みベースの検索
- ハイブリッド検索

#### `feedback_prompting.py`

- 検索結果を再推論用プロンプトに整形

---

## 9. 推奨する検索戦略

最初から大規模 embedding RAG を入れる必要はない。  
このプロジェクトでは、以下の順で十分である。

### Stage 1

数値特徴ベース検索

- speed
- acc_x
- brake
- gyro_z
- blinker
- curvature
- lateral displacement

### Stage 2

数値特徴 + 文章説明のハイブリッド検索

### Stage 3

誤認識パターンに重みを置いた検索

---

## 10. 実験計画

### 比較条件

- Baseline: Ver.4
- Proposed A: RAG only
- Proposed B: RAG + 再推論
- Proposed C: Verifier + RAG
- Proposed D: Graph + Verifier + RAG

### 主評価

- Accuracy
- Macro F1
- 少数クラス Recall
- 境界事例の改善率

### 追加分析

- Top-k retrieval の妥当性
- 検索結果が正答修正に寄与した割合
- 誤った事例検索による悪化率

---

## 11. 研究上の主張

RAG 単体では新規性が弱いが、本案では次を主張できる。

- 運転行動分類において、一般知識ではなく `domain-specific case memory` を用いる
- 誤認識パターン自体を検索知識として扱う
- 検索結果を最終説明ではなく再推論フィードバックへ使う

つまり、
**retrieval-augmented feedback correction**
として位置づけるのが重要である。

---

## 12. リスク

- データ数が少ないため検索が不安定になりうる
- 類似事例の質が悪いと誤誘導を招く
- ラベルリークと見なされないよう、評価分割を厳密に管理する必要がある

---

## 13. 最小実装の提案

まずは以下で十分である。

- 手動アノテーション済み CSV から case memory を作る
- 数値特徴のみで top-3 類似事例を取得する
- Failure Memory を 10 パターン程度手動で定義する
- 再推論で最終ラベルのみ更新する

これなら実装コストを抑えつつ、効果検証まで持っていける。

