# Annotation Tool

タクシー動画の運転行動を扱うアノテーションツールです。このブランチでは、Ver.4 baseline の再現を優先しています。

- 手動アノテーション
- 軌道描画付き 4 フレームの VLM 自動アノテーション

## 主要ファイル

- `app.py`: FastAPI アプリ本体
- `config.py`: モデル・推論設定
- `heron_model_with_trajectory.py`: 軌道描画付き自動アノテータ
- `heron_l2m_pipeline.py`: L2M/CoT 推論パイプライン
- `visual_prompting.py`: 軌道可視化
- `requirements.txt`: 実行に必要な依存関係
- `templates/`: UI テンプレート

## 入力データ

- `../sample/annotation_samples.csv`
- `../filterd_video/`

## 出力データ

- `annotated_samples_manual.csv`
- `annotated_samples_auto.csv`

## 起動

```bash
cd new/annotation_tool
python3 app.py
```

ブラウザで `http://localhost:8000` を開きます。

## 自動アノテーション

`config.py` では現在 `Qwen/Qwen2-VL-2B-Instruct` を使い、軌道描画付き 4 フレーム入力で直接分類する Ver.4 baseline を有効にしています。L2M/CoT はこのブランチでは既定で無効です。
