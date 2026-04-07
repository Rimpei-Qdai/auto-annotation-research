# 研究設計案1: Graph-of-Driving-Behavior Reasoning

**作成日**: 2026年3月12日  
**対象プロジェクト**: タクシー動画の運転行動自動アノテーション

---

## 1. 目的

本設計案は、運転行動分類を単純な 11 クラス分類として扱うのではなく、
**運転行動を構成する要素とその関係をグラフとして表現し、そのグラフ上で推論する**
枠組みに拡張することを目的とする。

現行システムでは、VLM が 4 フレームと軌道可視化から直接ラベルを出力するため、以下の失敗が起きやすい。

- 右折と左折の混同
- 車線変更と旋回の混同
- 減速と停止の混同
- 加速/減速の見逃し

これらは、ラベルそのものよりも、
**「何を見てそう判断したか」が構造化されていない**
ことに起因している。

---

## 2. 中核アイデア

各サンプルを以下のような `Driving Behavior Graph` として表現する。

### ノード候補

- `ego_state`: 自車状態
- `speed_state`: 停止 / 低速 / 一定 / 増加 / 減少
- `trajectory_shape`: 直進 / 左カーブ / 右カーブ / S字
- `trajectory_relation_to_lane`: 平行 / 左逸脱 / 右逸脱
- `intersection_state`: 交差点進入あり / なし
- `lane_change_state`: 車線跨ぎあり / なし
- `signal_state`: 左ウィンカー / 右ウィンカー / なし
- `brake_state`: ON / OFF
- `yaw_state`: 左回頭 / 右回頭 / なし

### エッジ候補

- `supports`
- `contradicts`
- `causes`
- `consistent_with`
- `incompatible_with`

### 例

右折の典型例:

```text
trajectory_shape=RIGHT_CURVE
signal_state=RIGHT
intersection_state=ENTERING
yaw_state=RIGHT
trajectory_relation_to_lane=CROSSING_RIGHT
```

これらの関係が強く結ばれると、`label=右折` に収束する。

---

## 3. このプロジェクトに合わせた設計

### 3.1 既存資産の活用

現行コードには既に以下の資産がある。

- `app.py`: サンプル管理と動画対応
- `heron_model_with_trajectory.py`: フレーム抽出と軌道描画
- `visual_prompting.py`: 3D 軌道の投影
- `heron_l2m_pipeline.py`: 多段推論の枠組み
- CSV 上のセンサ情報: `speed`, `acc_x`, `acc_y`, `gyro_z`, `brake`, `blinker_l`, `blinker_r`

このため、Graph Reasoning は全面作り直しではなく、
**L2M パイプラインの中間表現をグラフ化する形**
で導入できる。

### 3.2 追加すべき中間特徴

以下を新たに算出してグラフノードへ変換する。

- `speed_trend_score`
- `stop_likelihood`
- `trajectory_curvature`
- `lateral_displacement`
- `intersection_likelihood`
- `lane_crossing_likelihood`
- `turn_direction_score`
- `brake_consistency_score`

### 3.3 可視化の拡張

現行の赤線・緑線に加えて、以下をオーバーレイ候補とする。

- 軌道点の時間順番号
- 軌道点間隔の変化
- 左右方向の矢印
- 車線境界の推定線
- 交差点領域の候補表示

---

## 4. 推論フロー

### Step 1: Feature Extraction

動画・センサから定量特徴を抽出する。

```json
{
  "speed": 12.5,
  "acc_x": -1.8,
  "brake": 1,
  "gyro_z": 0.12,
  "trajectory_curvature": 0.03,
  "lateral_displacement": 0.8,
  "turn_direction_score": "RIGHT",
  "intersection_likelihood": 0.72
}
```

### Step 2: Graph Construction

特徴量を離散化し、ノードとエッジへ変換する。

```json
{
  "nodes": [
    {"id": "speed_state", "value": "DECREASING"},
    {"id": "brake_state", "value": "ON"},
    {"id": "trajectory_shape", "value": "RIGHT_CURVE"},
    {"id": "signal_state", "value": "RIGHT"},
    {"id": "intersection_state", "value": "ENTERING"}
  ],
  "edges": [
    {"source": "brake_state", "target": "speed_state", "type": "supports"},
    {"source": "trajectory_shape", "target": "signal_state", "type": "consistent_with"},
    {"source": "intersection_state", "target": "trajectory_shape", "type": "supports"}
  ]
}
```

### Step 3: Graph-aware Prompting

VLM に対し、画像だけではなくグラフ要約も与える。

```text
Graph summary:
- speed_state: DECREASING
- brake_state: ON
- trajectory_shape: RIGHT_CURVE
- signal_state: RIGHT
- intersection_state: ENTERING

Infer the final action label by checking consistency among these graph nodes.
```

### Step 4: Label Selection

VLM が最終ラベルと、使用したノード関係を出力する。

---

## 5. 期待効果

### 5.1 右折と左折の分離

現状では「曲がる」ことは認識しても、左右が不安定である。  
グラフ化により、`trajectory direction` と `blinker` と `yaw` を別々に保持できるため、右左の誤判定を減らせる。

### 5.2 車線変更と左折の分離

車線変更は `lane crossing` が主であり、左折は `intersection entry` が主である。  
この違いをノードで表現することで、両者の構造差を VLM に明示できる。

### 5.3 加速・減速の救済

速度変化をグラフノードに昇格させることで、単なる視覚的直進と、速度が変化している直進を分けて扱える。

---

## 6. 既存コードへの落とし込み

### 6.1 変更対象

- `new/annotation_tool/heron_l2m_pipeline.py`
- `new/annotation_tool/heron_model_with_trajectory.py`
- `new/annotation_tool/visual_prompting.py`
- 追加候補: `new/annotation_tool/driving_graph.py`

### 6.2 実装案

#### `driving_graph.py`

責務:

- センサと軌道から中間特徴を算出
- ノード・エッジを生成
- JSON 形式で返す

#### `heron_l2m_pipeline.py`

変更点:

- Level 1 を「幾何学的分析」から「グラフ特徴抽出」に変更
- Level 2 でノード間の整合性を評価
- Level 3 でラベル選択

#### `visual_prompting.py`

変更点:

- 軌道の向き、点間隔、車線関係を視覚的に表現

---

## 7. 実験計画

### 比較条件

- Baseline A: 現行 Ver.4
- Baseline B: L2M/CoT のみ
- Proposed 1: Graph Reasoning のみ
- Proposed 2: Graph Reasoning + 強化 Visual Prompting

### 指標

- Accuracy
- Macro F1
- クラス別 Recall
- 右折/左折識別精度
- 車線変更 Recall
- 加速/減速 Recall

### 見たいアブレーション

- グラフノードなし vs あり
- `signal_state` なし vs あり
- `intersection_state` なし vs あり
- `speed_state` なし vs あり

---

## 8. 研究上の新規性

既存研究では CoT や階層推論はあるが、本設計案では以下を主張できる。

- 運転行動分類をグラフ構造化推論問題として扱う
- 軌道・車線・速度・信号の関係を明示的にモデルへ渡す
- ラベルではなく「関係の整合性」を経由して分類する

---

## 9. 根拠となる既存研究

本設計案の根拠として、以下の論文を主に参照する。

### 9.1 中核となる論文

- `DriveLM: Driving with Graph Visual Question Answering`  
  本設計案で最も重要な参照元である。運転シーンを単純な 1 回の分類ではなく、`perception → prediction → planning` の関係構造として扱う発想を示している。  
  今回の `Driving Behavior Graph` は、この考え方を運転行動ラベル付与へ応用し、`speed_state`、`trajectory_shape`、`intersection_state`、`signal_state` などの中間概念ノードを介して最終ラベルを決めるよう再構成したものである。  
  論文: https://arxiv.org/abs/2312.14150

### 9.2 設計を補強する論文

- `DriveVLM: The Convergence of Autonomous Driving and Large Vision-Language Models`  
  VLM 単独では自動運転タスクの空間理解や構造的判断に限界があり、外部の中間表現や従来モジュールとの組み合わせが重要であることを示している。  
  本プロジェクトで、画像から直接 11 クラス分類を行わず、中間概念を明示して推論する方針の背景として使える。  
  論文: https://arxiv.org/abs/2402.12289

- `Concept Bottleneck Models Without Predefined Concepts`  
  最終ラベルの手前に意味的な中間概念を置く設計の一般論として参考にしている。  
  今回の `speed_state`、`turn_direction`、`lane_change_state`、`intersection_state` などを「推論のための部品」として扱う考え方の理論的な支えになる。  
  論文: https://arxiv.org/abs/2407.03921

- `SpatialVLM: Endowing Vision-Language Models with Spatial Reasoning Capabilities`  
  一般 VLM の空間推論能力が十分ではないことを示し、位置関係や方向関係を明示的に扱う必要性を補強する。  
  本プロジェクトで `右折/左折` や `車線変更/旋回` の混同が起きている状況とも整合しており、グラフによる関係表現の必要性を説明しやすい。  
  論文: https://arxiv.org/abs/2401.12168

- `DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model`  
  多段推論や説明可能性を持つ運転 VLM の流れを把握するための参考文献である。  
  本設計案は単なる CoT の延長ではなく、説明をグラフ構造に昇格させ、関係の整合性を使ってラベル決定する点で差別化できる。  
  論文: https://arxiv.org/abs/2310.01412

### 9.3 将来的な拡張の参考

- `G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering`  
  将来的に `Graph + RAG` を統合する場合の参考文献である。  
  現時点では Graph 単独実装を優先するが、今後「概念グラフを検索キーにした類似事例検索」へ発展させる際の基礎として位置づけられる。  
  論文: https://arxiv.org/abs/2402.07630

### 9.4 このプロジェクトとの対応関係

本プロジェクトにおける Graph 推論案は、上記研究をそのまま流用するのではなく、以下のように接続する。

- `DriveLM` のグラフ発想を、運転行動アノテーションのラベル分類へ再定式化する
- `DriveVLM` と `SpatialVLM` の問題意識を踏まえ、VLM 単独の曖昧な判断を中間概念で構造化する
- `Concept Bottleneck` の考え方を利用し、右左・停止・速度変化などを明示的な概念ノードとして扱う
- 将来的には `G-Retriever` 的な検索機構を追加し、グラフを検索キーにした事例参照へ拡張できる

---

## 10. リスク

- 小規模データではノード設計が過学習的になる可能性
- 交差点・車線境界の推定が不安定だと逆効果
- グラフが複雑すぎると VLM への入力負荷が増える

---

## 11. 最小実装の提案

最初は大きく作らず、以下に絞る。

- `speed_state`
- `trajectory_shape`
- `trajectory_relation_to_lane`
- `signal_state`
- `intersection_state`

この 5 ノードだけでも、現行の主要失敗にかなり直結する。

