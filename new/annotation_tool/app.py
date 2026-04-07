# app.py

import os
import json
import bisect
import time
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone

JST = timezone(timedelta(hours=9))
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Configure logging（コンソール + ファイル両方に出力）
_LOG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'annotation.log')
_log_handler_file = logging.FileHandler(_LOG_FILE_PATH, encoding='utf-8')
_log_handler_file.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), _log_handler_file]
)
logger = logging.getLogger(__name__)

# Import production annotator
try:
    from heron_model_with_trajectory import get_annotator
    HERON_ENABLED = True
    logger.info("Using trajectory-enabled annotator with L2M/CoT support")
except ImportError as e:
    HERON_ENABLED = False
    logger.warning(f"Heron auto-annotation disabled: {e}")

# ============ 初期設定 ============
app = FastAPI()

# パス設定
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# プロジェクトのルートディレクトリ (new/ の親)
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
SAMPLE_DIR = os.path.join(PROJECT_ROOT, 'sample')
VIDEO_DIR = os.path.join(os.path.dirname(BASE_DIR), 'filterd_video')
INFERENCE_LOG_PATH = os.path.join(BASE_DIR, 'inference_process.jsonl')

CSV_INPUT = os.path.join(SAMPLE_DIR, 'annotation_samples.csv')
CSV_OUTPUT_MANUAL = os.path.join(BASE_DIR, 'annotated_samples_manual.csv')
CSV_OUTPUT_AUTO = os.path.join(BASE_DIR, 'annotated_samples_auto.csv')

# 静的ファイルとテンプレート
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
os.makedirs(VIDEO_DIR, exist_ok=True)
app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# ============ データ読み込み ============

# ベースとなるCSVデータの読み込み（手動アノテーション用）
df = pd.read_csv(CSV_INPUT)

# データ型の確認・修正
df['sample_id'] = df['sample_id'].astype(int)

# # 既存の手動アノテーションファイルがあれば、action_labelをマージ
# if os.path.exists(CSV_OUTPUT_MANUAL):
#     df_manual_saved = pd.read_csv(CSV_OUTPUT_MANUAL)
#     df_manual_saved['sample_id'] = df_manual_saved['sample_id'].astype(int)
#     # action_labelがあればマージ
#     if 'action_label' in df_manual_saved.columns:
#         # 既存のaction_labelをクリアしてからマージ
#         if 'action_label' in df.columns:
#             df = df.drop(columns=['action_label'])
#         df = df.merge(df_manual_saved[['sample_id', 'action_label']], 
#                      on='sample_id', how='left')

# action_label列が存在しない場合は作成
if 'action_label' not in df.columns:
    df['action_label'] = None

# 自動アノテーション用のDataFrameを読み込み
df_auto = pd.read_csv(CSV_INPUT)

# データ型の確認・修正
df_auto['sample_id'] = df_auto['sample_id'].astype(int)

# 既存の自動アノテーションファイルがあれば、action_labelをマージ
if os.path.exists(CSV_OUTPUT_AUTO):
    df_auto_saved = pd.read_csv(CSV_OUTPUT_AUTO)
    df_auto_saved['sample_id'] = df_auto_saved['sample_id'].astype(int)
    # action_labelがあればマージ
    if 'action_label' in df_auto_saved.columns:
        # 既存のaction_labelをクリアしてからマージ
        if 'action_label' in df_auto.columns:
            df_auto = df_auto.drop(columns=['action_label'])
        df_auto = df_auto.merge(df_auto_saved[['sample_id', 'action_label']], 
                               on='sample_id', how='left')

# action_label列が存在しない場合は作成
if 'action_label' not in df_auto.columns:
    df_auto['action_label'] = None

def _augment_motion_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    taxi_id ごとの時系列から速度差分と時間正規化速度変化率を計算する。

    行順は保持しつつ、各タクシー内では timestamp 順の前サンプルとの差分を使う。
    """
    required_columns = {'sample_id', 'taxi_id', 'timestamp', 'speed'}
    if not required_columns.issubset(dataframe.columns):
        return dataframe

    ordered = dataframe[['sample_id', 'taxi_id', 'timestamp', 'speed']].copy()
    ordered = ordered.sort_values(['taxi_id', 'timestamp', 'sample_id'])

    grouped = ordered.groupby('taxi_id', sort=False)
    ordered['prev_timestamp'] = grouped['timestamp'].shift()
    ordered['prev_speed'] = grouped['speed'].shift()
    ordered['speed_diff'] = (ordered['speed'] - ordered['prev_speed']).fillna(0.0)
    ordered['timestamp_diff_sec'] = (
        (ordered['timestamp'] - ordered['prev_timestamp']) / 1000.0
    ).fillna(0.0)
    ordered['timestamp_diff_sec'] = ordered['timestamp_diff_sec'].clip(lower=0.0)

    ordered['speed_change_rate'] = 0.0
    valid_delta = ordered['timestamp_diff_sec'] > 0
    ordered.loc[valid_delta, 'speed_change_rate'] = (
        ordered.loc[valid_delta, 'speed_diff'] / ordered.loc[valid_delta, 'timestamp_diff_sec']
    )
    ordered['motion_feature_reliable'] = (
        (ordered['timestamp_diff_sec'] > 0.0) & (ordered['timestamp_diff_sec'] <= 10.0)
    )

    feature_columns = ordered[
        ['sample_id', 'speed_diff', 'timestamp_diff_sec', 'speed_change_rate', 'motion_feature_reliable']
    ]
    merged = dataframe.merge(feature_columns, on='sample_id', how='left')
    merged['speed_diff'] = merged['speed_diff'].fillna(0.0)
    merged['timestamp_diff_sec'] = merged['timestamp_diff_sec'].fillna(0.0)
    merged['speed_change_rate'] = merged['speed_change_rate'].fillna(0.0)
    merged['motion_feature_reliable'] = merged['motion_feature_reliable'].fillna(False).astype(bool)
    return merged


df_auto = _augment_motion_features(df_auto)
if {'speed_diff', 'timestamp_diff_sec', 'speed_change_rate', 'motion_feature_reliable'}.issubset(df_auto.columns):
    logger.info("運動特徴を計算しました (speed_diff / timestamp_diff_sec / speed_change_rate / motion_feature_reliable)")

# ============ ヘルパー関数 ============

def unix_ms_to_datetime(unix_ms):
    """UNIXミリ秒をJST datetimeに変換"""
    return datetime.fromtimestamp(unix_ms / 1000.0, tz=JST).replace(tzinfo=None)

def extract_video_timestamp(filename):
    """動画ファイル名からタイムスタンプを抽出"""
    parts = filename.split('_')
    if len(parts) >= 3:
        timestamp_str = parts[2]
        return datetime.strptime(timestamp_str, '%Y%m%d%H%M%S')
    return None


# taxi_idごとの動画インデックスをキャッシュして、ページ表示の待ち時間を削減
_video_index_cache = {}


def _build_video_index_for_taxi(taxi_id):
    """指定タクシーの動画インデックスを構築して返す。"""
    taxi_id = str(taxi_id)
    taxi_folder = os.path.join(VIDEO_DIR, taxi_id)
    if not os.path.isdir(taxi_folder):
        return None

    indexed = []
    try:
        for entry in os.scandir(taxi_folder):
            if not entry.is_file():
                continue
            name = entry.name
            if not name.startswith('EVT_') or not name.endswith('.mp4'):
                continue

            if name.endswith('_FRONT.mp4'):
                base_name = name[:-10]
            elif name.endswith('_INNER.mp4'):
                base_name = name[:-10]
            else:
                continue

            video_timestamp = extract_video_timestamp(name)
            if video_timestamp is None:
                continue
            indexed.append((video_timestamp, base_name))
    except OSError as e:
        logger.warning(f"Failed to scan taxi folder {taxi_folder}: {e}")
        return None

    if not indexed:
        return None

    # 同一base_nameが重複するため、timestamp+base_nameでユニーク化
    indexed = sorted(set(indexed), key=lambda x: x[0])
    timestamps = [item[0] for item in indexed]
    base_names = [item[1] for item in indexed]

    result = {
        'taxi_folder': taxi_folder,
        'timestamps': timestamps,
        'base_names': base_names,
    }
    _video_index_cache[taxi_id] = result
    logger.info(f"Built video index for taxi_id={taxi_id}: {len(base_names)} entries")
    return result


def _get_video_index_for_taxi(taxi_id):
    taxi_id = str(taxi_id)
    if taxi_id in _video_index_cache:
        return _video_index_cache[taxi_id]
    return _build_video_index_for_taxi(taxi_id)

def find_matching_videos(taxi_id, data_timestamp_unix_ms):
    """データのタイムスタンプに対応する動画を検索"""
    taxi_id = str(taxi_id)
    data_time = unix_ms_to_datetime(data_timestamp_unix_ms)

    video_index = _get_video_index_for_taxi(taxi_id)
    if video_index is None:
        return None

    taxi_folder = video_index['taxi_folder']
    timestamps = video_index['timestamps']
    base_names = video_index['base_names']
    if not timestamps:
        return None

    # 条件: data_time が [video_timestamp-15s, video_timestamp+15s] に入る
    lower = data_time - timedelta(seconds=15)
    upper = data_time + timedelta(seconds=15)
    left = bisect.bisect_left(timestamps, lower)
    right = bisect.bisect_right(timestamps, upper)

    for i in range(left, right):
        video_timestamp = timestamps[i]
        base_name = base_names[i]

        # 動画開始時刻 = タイムスタンプ - 15秒
        video_start = video_timestamp - timedelta(seconds=15)
        video_end = video_start + timedelta(seconds=30)  # Extended from 20 to 30 seconds

        if video_start <= data_time <= video_end:
            offset = (data_time - video_start).total_seconds()

            front_file = f"{base_name}_FRONT.mp4"
            inner_file = f"{base_name}_INNER.mp4"

            front_path = os.path.join(taxi_folder, front_file)
            inner_path = os.path.join(taxi_folder, inner_file)

            has_front = os.path.exists(front_path)
            has_inner = os.path.exists(inner_path)
            result = {
                'offset_seconds': offset,
                'video_start_time': video_start,
                'matched_timestamp': video_timestamp,
                'front': f"/videos/{taxi_id}/{front_file}" if has_front else None,
                'inner': f"/videos/{taxi_id}/{inner_file}" if has_inner else None,
                'has_front': has_front,
                'has_inner': has_inner
            }

            if result['front'] or result['inner']:
                return result

    return None

def get_next_unannotated_sample():
    """次の未アノテーションサンプルを取得（手動アノテーション用）"""
    unannotated = df[df['action_label'].isna()]
    if not unannotated.empty:
        row = unannotated.iloc[0]
        return row['sample_id'], row
    return None, None

def save_dataframe_manual():
    """手動アノテーション用DataFrameをCSVに保存（sample_id, timestamp, action_labelのみ）"""
    output_df = df[['sample_id', 'timestamp', 'action_label']].copy()
    output_df.to_csv(CSV_OUTPUT_MANUAL, index=False)

def save_dataframe_auto():
    """自動アノテーション用DataFrameをCSVに保存（sample_id, timestamp, action_labelのみ）"""
    output_df = df_auto[['sample_id', 'timestamp', 'action_label']].copy()
    output_df.to_csv(CSV_OUTPUT_AUTO, index=False)
    logger.info(f"Auto-annotation results saved to {CSV_OUTPUT_AUTO}")


def append_inference_log(event):
    """推論過程ログをJSON Lines形式で保存する。"""
    payload = {
        'logged_at': datetime.now(JST).isoformat(),
        **event,
    }
    with open(INFERENCE_LOG_PATH, 'a', encoding='utf-8') as f:
        f.write(json.dumps(payload, ensure_ascii=False) + '\n')


def get_prediction_failure_reason(predicted_label, l2m_details):
    """推論結果が保存可能かを判定し、失敗理由を返す。"""
    if predicted_label is None:
        return "Prediction returned None"

    if isinstance(l2m_details, dict) and l2m_details.get('status') == 'failed':
        return l2m_details.get('error') or 'L2M inference failed'

    return None

# ============ ルーティング ============

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    """メインアノテーション画面"""
    req_start = time.perf_counter()
    sample_id, row = get_next_unannotated_sample()
    
    if sample_id is None:
        return templates.TemplateResponse("completed.html", {"request": request})
    
    # 動画マッチング
    match_start = time.perf_counter()
    match_result = find_matching_videos(row['taxi_id'], row['timestamp'])
    logger.info(f"main_page: video_match_elapsed={time.perf_counter() - match_start:.3f}s sample_id={sample_id}")
    
    if match_result is None:
        match_result = {
            'has_front': False,
            'has_inner': False,
            'front': None,
            'inner': None,
            'offset_seconds': 0
        }
    
    offset = match_result['offset_seconds']
    
    # 進捗計算（手動アノテーションのみ）
    total = len(df)
    manual_done = len(df[df['action_label'].notna()])
    
    progress = {
        'total': total,
        'annotated': manual_done,
        'remaining': total - manual_done,
        'manual_done': manual_done,
        'auto_done': 0  # 別ファイルなので表示しない
    }
    
    # データ時刻をフォーマット
    data_datetime = unix_ms_to_datetime(row['timestamp'])
    
    # イベント情報
    has_event = pd.notna(row.get('eventType'))
    event_info = None
    if has_event:
        event_info = {
            'type': int(row['eventType']) if pd.notna(row['eventType']) else None,
            'id': row.get('eventId'),
            'datetime': row.get('eventDateTime')
        }
    
    # センサー情報
    sensor_info = {
        'acc_x': float(row['acc_x']),
        'acc_y': float(row['acc_y']),
        'acc_z': float(row['acc_z']),
        'speed': float(row['speed']),
        'brake': int(row['brake']),
        'blinker_r': int(row['blinker_r']),
        'blinker_l': int(row['blinker_l']),
        'speed_diff': float(row['speed_diff']) if 'speed_diff' in row.index else 0.0,
        'speed_change_rate': float(row['speed_change_rate']) if 'speed_change_rate' in row.index else 0.0,
        'motion_feature_reliable': bool(row['motion_feature_reliable']) if 'motion_feature_reliable' in row.index else False,
    }
    
    response = templates.TemplateResponse("index.html", {
        "request": request,
        "sample_id": sample_id,
        "row": row.to_dict(),
        "data_datetime": data_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        "videos": match_result,
        "offset_seconds": offset,
        "progress": progress,
        "has_event": has_event,
        "event_info": event_info,
        "sensor_info": sensor_info
    })
    logger.info(f"main_page: total_elapsed={time.perf_counter() - req_start:.3f}s sample_id={sample_id}")
    return response

@app.post("/annotate")
async def annotate(
    sample_id: int = Form(...),
    action_label: int = Form(...)
):
    """手動アノテーション送信"""
    idx = df[df['sample_id'] == sample_id].index[0]
    df.at[idx, 'action_label'] = action_label
    save_dataframe_manual()
    return RedirectResponse("/", status_code=302)

@app.post("/undo")
async def undo():
    """直前のアノテーションを取り消し"""
    annotated = df[df['action_label'].notna()]
    if not annotated.empty:
        last_idx = annotated.index[-1]
        df.at[last_idx, 'action_label'] = None
        save_dataframe_manual()
    return RedirectResponse("/", status_code=302)

@app.post("/auto_annotate")
async def auto_annotate(sample_id: int = Form(...)):
    """単一サンプルの自動アノテーション"""
    if not HERON_ENABLED:
        return JSONResponse(
            {"error": "Heron auto-annotation is not available"},
            status_code=503
        )
    
    try:
        # Get sample data from auto dataframe
        row = df_auto[df_auto['sample_id'] == sample_id].iloc[0]
        request_id = f"single-{sample_id}-{int(datetime.now(JST).timestamp() * 1000)}"
        append_inference_log({
            'request_id': request_id,
            'endpoint': '/auto_annotate',
            'phase': 'start',
            'sample_id': int(sample_id),
            'taxi_id': str(row['taxi_id']),
            'timestamp': int(row['timestamp'])
        })
        
        # Find matching video
        match_result = find_matching_videos(row['taxi_id'], row['timestamp'])
        
        if match_result is None or not (match_result['has_front'] or match_result['has_inner']):
            append_inference_log({
                'request_id': request_id,
                'endpoint': '/auto_annotate',
                'phase': 'video_not_found',
                'sample_id': int(sample_id),
                'taxi_id': str(row['taxi_id'])
            })
            return JSONResponse(
                {"error": "No video found for this sample"},
                status_code=404
            )
        
        # Use front camera if available, otherwise inner
        video_url = match_result['front'] if match_result['has_front'] else match_result['inner']
        # video_url is like "/videos/taxi.../file.mp4", need to replace /videos with actual VIDEO_DIR
        video_relative_path = video_url.replace('/videos/', '')
        video_path = os.path.join(VIDEO_DIR, video_relative_path)
        video_path = os.path.normpath(video_path)
        logger.info(f"Video URL: {video_url}")
        logger.info(f"Video path for prediction: {video_path}")
        logger.info(f"Video file exists: {os.path.exists(video_path)}")
        
        # Prepare sensor data
        sensor_data = {
            'speed': float(row['speed']),
            'acc_x': float(row['acc_x']),
            'acc_y': float(row['acc_y']),
            'acc_z': float(row['acc_z']),
            'gyro_z': float(row['gyro_z']),
            'brake': int(row['brake']),
            'blinker_r': int(row['blinker_r']),
            'blinker_l': int(row['blinker_l']),
            'speed_diff': float(row['speed_diff']) if 'speed_diff' in row.index else 0.0,
            'timestamp_diff_sec': float(row['timestamp_diff_sec']) if 'timestamp_diff_sec' in row.index else 0.0,
            'speed_change_rate': float(row['speed_change_rate']) if 'speed_change_rate' in row.index else 0.0,
            'motion_feature_reliable': bool(row['motion_feature_reliable']) if 'motion_feature_reliable' in row.index else False,
        }
        append_inference_log({
            'request_id': request_id,
            'endpoint': '/auto_annotate',
            'phase': 'predicting',
            'sample_id': int(sample_id),
            'video_path': video_path,
            'offset_seconds': float(match_result['offset_seconds']),
            'sensor_data': sensor_data
        })

        # Get annotator and predict
        annotator = get_annotator()
        predicted_label, l2m_details = annotator.predict_action_with_details(
            video_path=video_path,
            sensor_data=sensor_data,
            start_time=match_result['offset_seconds'],
            sample_id=sample_id
        )

        failure_reason = get_prediction_failure_reason(predicted_label, l2m_details)

        if failure_reason is None:
            # Save prediction to auto dataframe
            idx = df_auto[df_auto['sample_id'] == sample_id].index[0]
            df_auto.at[idx, 'action_label'] = predicted_label
            save_dataframe_auto()
            append_inference_log({
                'request_id': request_id,
                'endpoint': '/auto_annotate',
                'phase': 'completed',
                'sample_id': int(sample_id),
                'predicted_label': int(predicted_label),
                'status': 'success',
                'l2m_details': l2m_details
            })
            
            return JSONResponse({
                "success": True,
                "sample_id": sample_id,
                "predicted_label": int(predicted_label)
            })
        else:
            append_inference_log({
                'request_id': request_id,
                'endpoint': '/auto_annotate',
                'phase': 'completed',
                'sample_id': int(sample_id),
                'status': 'failed',
                'error': failure_reason
            })
            return JSONResponse(
                {"error": failure_reason},
                status_code=500
            )
            
    except Exception as e:
        append_inference_log({
            'endpoint': '/auto_annotate',
            'phase': 'exception',
            'sample_id': int(sample_id),
            'status': 'failed',
            'error': str(e)
        })
        logger.error(f"Auto-annotation error: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.post("/auto_annotate_batch")
async def auto_annotate_batch(num_samples: int = Form(10)):
    """複数サンプルの自動アノテーション（バッチ処理）"""
    if not HERON_ENABLED:
        return JSONResponse(
            {"error": "Heron auto-annotation is not available"},
            status_code=503
        )
    
    try:
        # Get unannotated samples from auto dataframe
        unannotated = df_auto[df_auto['action_label'].isna()].head(num_samples)
        
        if unannotated.empty:
            return JSONResponse({
                "success": True,
                "message": "No samples to annotate",
                "results": []
            })
        
        results = []
        annotator = get_annotator()
        
        for idx, row in unannotated.iterrows():
            sample_id = row['sample_id']
            request_id = f"batch-{sample_id}-{int(datetime.now(JST).timestamp() * 1000)}"
            append_inference_log({
                'request_id': request_id,
                'endpoint': '/auto_annotate_batch',
                'phase': 'start',
                'sample_id': int(sample_id),
                'taxi_id': str(row['taxi_id']),
                'timestamp': int(row['timestamp'])
            })
            
            try:
                # Find matching video
                match_result = find_matching_videos(row['taxi_id'], row['timestamp'])
                
                if match_result is None or not (match_result['has_front'] or match_result['has_inner']):
                    logger.warning(f"Sample {sample_id}: No video found for taxi_id={row['taxi_id']}, timestamp={row['timestamp']}")
                    append_inference_log({
                        'request_id': request_id,
                        'endpoint': '/auto_annotate_batch',
                        'phase': 'video_not_found',
                        'sample_id': int(sample_id),
                        'taxi_id': str(row['taxi_id'])
                    })
                    results.append({
                        "sample_id": sample_id,
                        "success": False,
                        "error": "No video found"
                    })
                    continue
                
                # Use front camera if available
                video_url = match_result['front'] if match_result['has_front'] else match_result['inner']
                # video_url is like "/videos/taxi.../file.mp4", need to replace /videos with actual VIDEO_DIR
                video_relative_path = video_url.replace('/videos/', '')
                video_path = os.path.join(VIDEO_DIR, video_relative_path)
                video_path = os.path.normpath(video_path)
                
                # Prepare sensor data
                sensor_data = {
                    'speed': float(row['speed']),
                    'acc_x': float(row['acc_x']),
                    'acc_y': float(row['acc_y']),
                    'acc_z': float(row['acc_z']),
                    'gyro_z': float(row['gyro_z']),
                    'brake': int(row['brake']),
                    'blinker_r': int(row['blinker_r']),
                    'blinker_l': int(row['blinker_l']),
                    'speed_diff': float(row['speed_diff']) if 'speed_diff' in row.index else 0.0,
                    'timestamp_diff_sec': float(row['timestamp_diff_sec']) if 'timestamp_diff_sec' in row.index else 0.0,
                    'speed_change_rate': float(row['speed_change_rate']) if 'speed_change_rate' in row.index else 0.0,
                    'motion_feature_reliable': bool(row['motion_feature_reliable']) if 'motion_feature_reliable' in row.index else False,
                }
                append_inference_log({
                    'request_id': request_id,
                    'endpoint': '/auto_annotate_batch',
                    'phase': 'predicting',
                    'sample_id': int(sample_id),
                    'video_path': video_path,
                    'offset_seconds': float(match_result['offset_seconds']),
                    'sensor_data': sensor_data
                })

                # Predict
                predicted_label, l2m_details = annotator.predict_action_with_details(
                    video_path=video_path,
                    sensor_data=sensor_data,
                    start_time=match_result['offset_seconds'],
                    sample_id=sample_id
                )

                failure_reason = get_prediction_failure_reason(predicted_label, l2m_details)

                if failure_reason is None:
                    # Save prediction to auto dataframe
                    df_auto.at[idx, 'action_label'] = predicted_label
                    append_inference_log({
                        'request_id': request_id,
                        'endpoint': '/auto_annotate_batch',
                        'phase': 'completed',
                        'sample_id': int(sample_id),
                        'predicted_label': int(predicted_label),
                        'status': 'success',
                        'l2m_details': l2m_details
                    })
                    results.append({
                        "sample_id": sample_id,
                        "success": True,
                        "predicted_label": int(predicted_label)
                    })
                else:
                    append_inference_log({
                        'request_id': request_id,
                        'endpoint': '/auto_annotate_batch',
                        'phase': 'completed',
                        'sample_id': int(sample_id),
                        'status': 'failed',
                        'error': failure_reason
                    })
                    results.append({
                        "sample_id": sample_id,
                        "success": False,
                        "error": failure_reason
                    })
                    
            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}")
                append_inference_log({
                    'request_id': request_id,
                    'endpoint': '/auto_annotate_batch',
                    'phase': 'exception',
                    'sample_id': int(sample_id),
                    'status': 'failed',
                    'error': str(e)
                })
                results.append({
                    "sample_id": sample_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Save all predictions
        save_dataframe_auto()
        
        successful = sum(1 for r in results if r.get('success', False))
        
        return JSONResponse({
            "success": True,
            "total_processed": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Batch auto-annotation error: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.get("/auto_annotate_status")
async def auto_annotate_status():
    """自動アノテーション機能の状態を取得"""
    return JSONResponse({
        "enabled": HERON_ENABLED,
        "message": "Heron auto-annotation is available" if HERON_ENABLED 
                   else "Heron auto-annotation is not available"
    })

@app.post("/auto_annotate_all")
async def auto_annotate_all():
    """全ての未アノテーションサンプルを自動でアノテーション"""
    if not HERON_ENABLED:
        return JSONResponse(
            {"error": "Heron auto-annotation is not enabled"},
            status_code=503
        )
    
    try:
        global df
        
        # 全サンプルを取得（既存のアノテーションを上書き）
        # Clear all existing action_labels for re-annotation
        df_auto['action_label'] = None
        unannotated = df_auto.copy()
        
        if len(unannotated) == 0:
            return JSONResponse({
                "success": True,
                "message": "No samples to annotate",
                "total_processed": 0
            })
        
        logger.info(f"Starting auto-annotation for {len(unannotated)} samples (re-annotating all)")
        
        annotator = get_annotator()
        results = []
        
        for idx, (_, row) in enumerate(unannotated.iterrows()):
            sample_id = int(row['sample_id'])
            request_id = f"all-{sample_id}-{int(datetime.now(JST).timestamp() * 1000)}"
            append_inference_log({
                'request_id': request_id,
                'endpoint': '/auto_annotate_all',
                'phase': 'start',
                'sample_id': int(sample_id),
                'taxi_id': str(row['taxi_id']),
                'timestamp': int(row['timestamp'])
            })
            try:
                # Find matching video
                match_result = find_matching_videos(
                    row['taxi_id'],
                    row['timestamp']
                )
                
                if match_result is None or not (match_result['has_front'] or match_result['has_inner']):
                    logger.warning(f"Sample {sample_id}: No video found for taxi_id={row['taxi_id']}, timestamp={row['timestamp']}")
                    append_inference_log({
                        'request_id': request_id,
                        'endpoint': '/auto_annotate_all',
                        'phase': 'video_not_found',
                        'sample_id': int(sample_id),
                        'taxi_id': str(row['taxi_id'])
                    })
                    results.append({
                        "sample_id": sample_id,
                        "success": False,
                        "error": "No video found"
                    })
                    continue
                
                # Get video path
                video_url = match_result['front'] if match_result['has_front'] else match_result['inner']
                video_relative_path = video_url.replace('/videos/', '')
                video_path = os.path.join(VIDEO_DIR, video_relative_path)
                video_path = os.path.normpath(video_path)
                
                # Prepare sensor data
                sensor_data = {
                    'speed': float(row['speed']),
                    'acc_x': float(row['acc_x']),
                    'acc_y': float(row['acc_y']),
                    'acc_z': float(row['acc_z']),
                    'gyro_z': float(row['gyro_z']),
                    'brake': int(row['brake']),
                    'blinker_r': int(row['blinker_r']),
                    'blinker_l': int(row['blinker_l']),
                    'speed_diff': float(row['speed_diff']) if 'speed_diff' in row.index else 0.0,
                    'timestamp_diff_sec': float(row['timestamp_diff_sec']) if 'timestamp_diff_sec' in row.index else 0.0,
                    'speed_change_rate': float(row['speed_change_rate']) if 'speed_change_rate' in row.index else 0.0,
                    'motion_feature_reliable': bool(row['motion_feature_reliable']) if 'motion_feature_reliable' in row.index else False,
                }
                append_inference_log({
                    'request_id': request_id,
                    'endpoint': '/auto_annotate_all',
                    'phase': 'predicting',
                    'sample_id': int(sample_id),
                    'video_path': video_path,
                    'offset_seconds': float(match_result['offset_seconds']),
                    'sensor_data': sensor_data
                })

                # Predict action
                predicted_label, l2m_details = annotator.predict_action_with_details(
                    video_path=video_path,
                    sensor_data=sensor_data,
                    start_time=match_result['offset_seconds'],
                    sample_id=sample_id
                )

                failure_reason = get_prediction_failure_reason(predicted_label, l2m_details)

                if failure_reason is None:
                    # Save to auto dataframe
                    df_auto.loc[df_auto['sample_id'] == sample_id, 'action_label'] = predicted_label
                    append_inference_log({
                        'request_id': request_id,
                        'endpoint': '/auto_annotate_all',
                        'phase': 'completed',
                        'sample_id': int(sample_id),
                        'predicted_label': int(predicted_label),
                        'status': 'success',
                        'l2m_details': l2m_details
                    })
                    results.append({
                        "sample_id": sample_id,
                        "success": True,
                        "predicted_label": int(predicted_label)
                    })
                    logger.info(f"Sample {sample_id}: predicted label {predicted_label}")
                else:
                    append_inference_log({
                        'request_id': request_id,
                        'endpoint': '/auto_annotate_all',
                        'phase': 'completed',
                        'sample_id': int(sample_id),
                        'status': 'failed',
                        'error': failure_reason
                    })
                    results.append({
                        "sample_id": sample_id,
                        "success": False,
                        "error": failure_reason
                    })
                
                # Progress logging
                if (idx + 1) % 10 == 0:
                    logger.info(f"Progress: {idx + 1}/{len(unannotated)} samples processed")
                    
            except Exception as e:
                logger.error(f"Error processing sample {sample_id}: {e}")
                append_inference_log({
                    'request_id': request_id,
                    'endpoint': '/auto_annotate_all',
                    'phase': 'exception',
                    'sample_id': int(sample_id),
                    'status': 'failed',
                    'error': str(e)
                })
                results.append({
                    "sample_id": sample_id,
                    "success": False,
                    "error": str(e)
                })
        
        # Save all predictions to auto file
        save_dataframe_auto()
        
        successful = sum(1 for r in results if r.get('success', False))
        
        logger.info(f"Auto-annotation completed: {successful}/{len(results)} successful")
        
        return JSONResponse({
            "success": True,
            "total_processed": len(results),
            "successful": successful,
            "failed": len(results) - successful,
            "message": f"Processed {len(results)} samples, {successful} successful"
        })
        
    except Exception as e:
        logger.error(f"Auto-annotate all error: {e}", exc_info=True)
        return JSONResponse(
            {"error": str(e)},
            status_code=500
        )

@app.get("/statistics", response_class=HTMLResponse)
async def statistics(request: Request):
    """統計ダッシュボード"""
    total = len(df)
    annotated = len(df[df['action_label'].notna()])
    
    label_stats = {}
    action_labels = {
        0: "その他", 1: "等速走行", 2: "加速", 3: "減速", 4: "停止",
        5: "発進", 6: "左折", 7: "右折", 8: "車線変更（左）",
        9: "車線変更（右）", 10: "転回（Uターン）"
    }
    
    for label, name in action_labels.items():
        count = len(df[df['action_label'] == label])
        if count > 0:
            label_stats[label] = {
                "name": name,
                "count": count,
                "percentage": count / total * 100
            }
    
    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "total": total,
        "annotated": annotated,
        "completion_rate": (annotated / total * 100) if total > 0 else 0,
        "label_stats": label_stats
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
