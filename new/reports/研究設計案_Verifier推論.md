# 研究設計案2: Verifier-Guided Driving Action Inference

**作成日**: 2026年3月12日  
**対象プロジェクト**: タクシー動画の運転行動自動アノテーション

---

## 1. 目的

本設計案は、VLM の初回推論結果をそのまま採用せず、
**物理法則・交通行動制約・センサ整合性によって検証し、必要なら再判定する**
推論パイプラインを構築することを目的とする。

現行システムの失敗には、明らかに検証可能な矛盾が含まれている。

- 速度 0 付近なのに等速走行と予測
- ブレーキ ON かつ負の加速度なのに等速走行と予測
- 右ウィンカーと右方向軌道があるのに左折と予測
- 小さい横移動なのに旋回と予測

これらは、VLM の認識能力不足だけでなく、
**推論結果を検証する層が存在しない**
ことが原因である。

---

## 2. 中核アイデア

推論を次の 3 層に分離する。

1. `Proposer`
VLM が候補ラベルと理由を出す

2. `Verifier`
候補ラベルがセンサ・軌道・交通ルールと矛盾しないか検証する

3. `Refiner`
矛盾がある場合のみ再推論し、ラベルを更新する

---

## 3. このプロジェクトに合わせた設計

### 3.1 現行構成との対応

現状の `heron_l2m_pipeline.py` は、

- Level 1: 幾何学
- Level 2: 物理整合
- Level 3: 最終分類

という構造を持っている。

したがって Verifier 設計は、既存 L2M パイプラインの延長として実装しやすい。

ただし現状の Level 2 は「説明生成」に近く、
**失敗時の強制補正ロジック**
が入っていない。

本設計では、Level 2 を厳密な検証器へ昇格させる。

---

## 4. Verifier の構造

### 4.1 入力

- 初回予測ラベル
- 初回推論理由
- センサ値
- 軌道特徴
- 信号特徴
- 交差点・車線関係

### 4.2 出力

```json
{
  "candidate_label": 1,
  "is_valid": false,
  "violations": [
    "speed_near_zero_conflicts_with_constant_speed",
    "brake_on_conflicts_with_constant_speed"
  ],
  "alternative_labels": [4, 3],
  "confidence_adjustment": -0.35
}
```

### 4.3 基本ルール例

#### 停止

- `speed <= threshold_stop` なら停止候補を優先
- `speed <= threshold_stop` かつ `trajectory_length` 極小なら停止を強く支持

#### 減速

- `acc_x < negative_threshold` または `brake = 1` なら減速候補を上げる
- ただし `speed≈0` なら停止と比較する

#### 加速

- `acc_x > positive_threshold` かつ `speed > 0` なら加速候補を上げる

#### 左右折

- `blinker_l = 1` かつ `turn_direction = LEFT` かつ `intersection_likelihood` 高
  → 左折を支持
- `blinker_r = 1` かつ `turn_direction = RIGHT` かつ `intersection_likelihood` 高
  → 右折を支持

#### 車線変更

- `lateral_displacement` 中程度
- `curvature` は小さい
- `intersection_likelihood` 低い
  → 車線変更を支持

---

## 5. 推論フロー

```text
動画/センサ入力
    ↓
VLM 初回推論 (candidate_label)
    ↓
Verifier rules
    ↓
矛盾なし -> 採用
矛盾あり -> 候補ラベル再ランキング
    ↓
再推論プロンプト投入
    ↓
最終ラベル
```

---

## 6. 再推論プロンプトの設計

初回ラベルに対して Verifier が矛盾を検出した場合、VLM に以下を与える。

```text
Initial prediction: 1 (constant speed)

Detected inconsistencies:
- speed is close to zero
- brake is ON
- trajectory is very short

Alternative plausible labels:
- 4 (stop)
- 3 (deceleration)

Re-evaluate the action label and output only one final label.
```

ここで重要なのは、VLM に最初から自由推論させるのではなく、
**矛盾を局所化した上で再判定させる**
点である。

---

## 7. 実装対象

### 7.1 追加ファイル候補

- `new/annotation_tool/verifier.py`
- `new/annotation_tool/feature_rules.py`

### 7.2 既存ファイルの修正候補

- `new/annotation_tool/heron_l2m_pipeline.py`
- `new/annotation_tool/config.py`

### 7.3 実装責務

#### `verifier.py`

- ルール定義
- 閾値管理
- 違反検出
- 代替ラベル候補生成

#### `heron_l2m_pipeline.py`

- 初回推論結果を Verifier に渡す
- 違反があれば再推論を呼ぶ
- ログ保存

---

## 8. この案が今回の課題にどう効くか

### 8.1 停止 vs 等速走行

現状の主要な誤りの一つは `停止 → 等速走行` である。  
これは verifier の最初の対象にすべきで、比較的簡単に改善が見込める。

### 8.2 減速 vs 等速走行

減速は Visual Prompting だけでは難しいが、
`acc_x` と `brake` が使えるため verifier 向きである。

### 8.3 右折 vs 左折

左右誤判定は視覚情報だけに頼ると不安定だが、
ウィンカー・回頭方向・交差点進入の整合性で補正できる。

### 8.4 車線変更

車線変更は厳密な検出が難しいが、
「旋回と見えるが交差点進入はない」という否定的制約で候補を絞れる。

---

## 9. 実験計画

### 比較条件

- Baseline: Ver.4
- Proposed A: Verifier only
- Proposed B: Verifier + 再推論
- Proposed C: Verifier + 強化特徴量

### 主評価

- Accuracy
- Macro F1
- 停止 Recall
- 減速 Recall
- 右折 Recall
- Confusion matrix の改善量

### 副評価

- Verifier 発火率
- 修正成功率
- 誤修正率

---

## 10. 研究上の主張

本案の主張点は、
**VLM の出力をそのまま信頼するのではなく、物理的・交通的に検証可能な中間制約を使って修正する**
ことにある。

これは一般的な CoT ではなく、
**process verification**
に近い立場を取る。

---

## 11. リスク

- ルールが強すぎると誤修正が増える
- 閾値設計がデータ依存になる
- クラス間の境界条件で verifier が過剰介入する可能性がある

---

## 12. 最小実装の提案

最初は以下の 4 ルールだけで始めるのがよい。

- 停止ルール
- 減速ルール
- 右折ルール
- 車線変更否定ルール

この 4 つは現行の主要誤りに最も直結する。

