# 2026-04-13 赤軌道 deterministic 幾何修正メモ

**作成日**: 2026-04-13  
**対象プロジェクト**: タクシー動画の運転行動自動アノテーション  
**目的**: 赤い予測軌道の「曲がり具合」と「長さ」を、VLM 任せではなく deterministic な特徴として推論に使う

---

## 1. 背景

これまでのログ分析では、Level 1 の幾何特徴が以下のようにほぼ全件で潰れていた。

- `trajectory_relation = PARALLEL`
- `intersection_detected = NO`
- `direction_change = STRAIGHT`
- `visual_shift = NO_SHIFT`

その結果、

- 赤い軌道の曲がりを十分に使えない
- 左右折や車線変更が VLM の曖昧な解釈に依存する
- 停止も赤線の「点らしさ」を直接使えていない

という問題が残っていた。

今回の修正では、赤い軌道そのものから

- 右左折
- 車線変更
- 直進
- 停止寄りかどうか

を直接計算し、Level 1 と Graph に流す構成へ変更した。

---

## 2. 今回の設計方針

今回の方針は次の通り。

1. 赤い軌道の幾何は VLM に丸投げしない  
2. 赤い軌道の曲率・左右偏位・見える長さを deterministic に算出する  
3. Level 1 の低レベル幾何項目は、その deterministic feature で上書きする  
4. VLM は道路文脈や交差点などの意味付け寄りに使う  
5. Graph でも赤い軌道の特徴を stop / turn / lane change の evidence として使う

---

## 3. 実装内容

## 3.1 `visual_prompting.py`

追加したこと:

- `extract_trajectory_features()` を追加
- 赤線から以下を算出
  - `trajectory_path_length_m`
  - `trajectory_visible_length_px`
  - `trajectory_visible_point_count`
  - `trajectory_point_like`
  - `trajectory_lateral_offset_m`
  - `trajectory_heading_delta_rad`
  - `trajectory_curvature_score`
  - `trajectory_length_state`
  - `trajectory_visual_speed_state`
  - `trajectory_motion_cue_deterministic`
  - `trajectory_relation_deterministic`
  - `direction_change_deterministic`
  - `visual_shift_deterministic`

今回の解釈ルール:

- 赤線の曲がりが大きく、左右偏位もある
  - `LEFT_TURN_CUE` / `RIGHT_TURN_CUE`
- 横偏位は大きいが曲率が小さい
  - `LEFT_LANE_CHANGE_CUE` / `RIGHT_LANE_CHANGE_CUE`
- ほぼ点に見える、または極端に短い
  - `trajectory_visual_speed_state = STOPPED`
- 曲がりも横偏位も小さい
  - `STRAIGHT_CUE`

意図:

- 「赤い線の曲がり具合で右左折」
- 「赤い線の長さでスピード」
- 「点になっていれば停車」

という解釈を、そのままコードに落とした。

---

## 3.2 `heron_model_with_trajectory.py`

追加したこと:

- `_build_trajectory_bundle()`
  - 描画に使うのと同じ 3 秒軌道から deterministic feature を作る
- `_augment_sensor_data_with_trajectory_features()`
  - 上で作った feature を `sensor_data` に混ぜる
- `draw_trajectory_on_frames()`
  - 事前計算済みの trajectory bundle を再利用できるように変更
- `_predict_with_l2m()`
  - L2M に渡す前に trajectory feature を `sensor_data` に注入
- `_predict_multi_frame_with_trajectory()`
  - 標準推論側でも同じ feature を使えるように変更

意図:

- 描画用の赤線と、推論用の赤線解釈がずれないようにする
- 「描画は 3 秒軌道、推論は別ロジック」という分離をなくす

---

## 3.3 `heron_l2m_pipeline.py`

変更したこと:

- 軌道オーバーレイの読み方ガイドを強化
  - 曲がり具合は左右旋回の主要 evidence
  - 長さは速度 evidence
  - 点状なら停止寄り
- `_level1_geometry()` に `sensor_data` を渡すよう変更
- `_apply_deterministic_trajectory_features()` を追加
  - Level 1 の
    - `trajectory_motion_cue`
    - `trajectory_relation`
    - `direction_change`
    - `visual_shift`
    を deterministic feature で上書き
  - 元の VLM 値は `vlm_...` として保持
- Level 3 prompt にも
  - `trajectory_visual_speed_state`
  - `trajectory_point_like`
  - `trajectory_length_state`
  - `trajectory_lateral_offset_m`
  - `trajectory_heading_delta_rad`
  を追加

意図:

- 低レベル幾何は deterministic にする
- VLM は交差点や道路文脈の意味付けに集中させる

---

## 3.4 `driving_graph.py`

変更したこと:

- Graph 概念抽出で以下を受け取るようにした
  - `trajectory_point_like`
  - `trajectory_visual_speed_state`
  - `trajectory_length_state`
  - `trajectory_lateral_offset_m`
  - `trajectory_heading_delta_rad`
  - `trajectory_curvature_score`
- `trajectory_direction` 決定時に
  - `trajectory_lateral_offset_m`
  を fallback evidence として使用
- `turn_intensity` 決定時に
  - `trajectory_curvature_score`
  を使用
- `speed_state` と `stop_likelihood` で
  - `trajectory_point_like`
  - `trajectory_visual_speed_state = STOPPED`
  を stop evidence として使用
- label scoring でも
  - 点状の赤線なら停止候補を加点

意図:

- 赤線の形が Graph の中で実際に効くようにする
- 単なる説明用 feature で終わらせない

---

## 3.5 テスト

`new/annotation_tool/tests/test_l2m_pipeline.py` に追加したもの:

- deterministic な赤線特徴が VLM 幾何を上書きするテスト
- point-like な赤線で stop likelihood が上がるテスト
- 左カーブ軌道から `LEFT_TURN_CUE` を抽出するテスト
- 点状軌道から `STOPPED` を抽出するテスト

確認コマンド:

```bash
python3 -m py_compile new/annotation_tool/visual_prompting.py \
  new/annotation_tool/heron_model_with_trajectory.py \
  new/annotation_tool/heron_l2m_pipeline.py \
  new/annotation_tool/driving_graph.py \
  new/annotation_tool/tests/test_l2m_pipeline.py

python3 -m unittest discover -s new/annotation_tool/tests
```

---

## 4. 期待される改善

今回の修正で直接改善を狙っているのは次の点。

- `trajectory_relation = PARALLEL` 全件固定の緩和
- 左右折の方向判定を VLM 任せにしない
- 車線変更を「横移動」として扱えるようにする
- 停止を「点状の赤線」として扱えるようにする
- Graph が赤線の形そのものを evidence として使えるようにする

---

## 5. まだ残る課題

今回の修正は、赤線の deterministic 化に絞ったものなので、以下はまだ残る。

- `intersection_detected` はまだ VLM 依存が大きい
- `road_shape` も deterministic ではない
- 速度変化は `speed_diff` の信頼性問題がまだ残る
- 停止 / 発進の定義は人手ラベル寄りにさらに調整が必要

---

## 6. 次に確認すべきこと

GPU サーバーで再アノテーションしたあと、まず次を確認する。

1. `trajectory_relation = PARALLEL` が全件固定でなくなっているか  
2. `trajectory_motion_cue = AMBIGUOUS` に過度に潰れすぎていないか  
3. `停止 -> 等速走行` の混同が減っているか  
4. `左折` の false positive がさらに減っているか  
5. `車線変更(左)` が少しでも拾えるようになっているか

---

## 7. 関連ファイル

- `new/annotation_tool/visual_prompting.py`
- `new/annotation_tool/heron_model_with_trajectory.py`
- `new/annotation_tool/heron_l2m_pipeline.py`
- `new/annotation_tool/driving_graph.py`
- `new/annotation_tool/tests/test_l2m_pipeline.py`

---

## 8. 一言でまとめると

今回の修正は、

**「赤い軌道を見せるだけ」から、「赤い軌道の曲がり具合と長さを直接数値化して推論に使う」**

への移行である。
