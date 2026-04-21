"""
Utilities for building sample-aligned dense sensor windows.

The current runtime falls back to `sample/annotation_samples.csv`, which is
useful as a bootstrap but is not a real dense telemetry source. This module
adds a reusable builder that can consume external telemetry CSVs and emit
sample-wise window records for later training / inference.
"""

from __future__ import annotations

import bisect
import collections
import csv
import datetime as dt
import json
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


DEFAULT_WINDOW_SECONDS = 3.0
DEFAULT_MIN_WINDOW_ROWS = 3
DEFAULT_MAX_DENSE_GAP_SECONDS = 1.5
DEFAULT_MAX_CENTER_OFFSET_SECONDS = 1.5
DEFAULT_MAX_VIDEO_DELTA_SECONDS = 60.0
DEFAULT_DATASET_RELATIVE_PATH = os.path.join("annotation_tool", "data", "sensor_windows.jsonl")
VIDEO_FILENAME_RE = re.compile(r"^EVT_\d+_(\d{14})_(FRONT|INNER)\.mp4$")
JST = dt.timezone(dt.timedelta(hours=9))


def default_sensor_window_dataset_path(repo_root: str) -> str:
    return os.path.join(repo_root, DEFAULT_DATASET_RELATIVE_PATH)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _resolve_taxi_id(row: Dict[str, Any], fallback_taxi_id: Optional[str] = None) -> str:
    return str(row.get("taxi_id") or row.get("imei") or fallback_taxi_id or "").strip()


def _normalize_sensor_row(
    row: Dict[str, Any],
    *,
    fallback_taxi_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    taxi_id = _resolve_taxi_id(row, fallback_taxi_id=fallback_taxi_id)
    timestamp = _safe_int(row.get("timestamp"), 0)
    if not taxi_id or timestamp <= 0:
        return None

    sample_id_raw = row.get("sample_id")
    sample_id = _safe_int(sample_id_raw, -1) if sample_id_raw not in (None, "") else None
    if sample_id is not None and sample_id < 0:
        sample_id = None

    return {
        "sample_id": sample_id,
        "taxi_id": taxi_id,
        "timestamp": timestamp,
        "speed": _safe_float(row.get("speed")),
        "acc_x": _safe_float(row.get("acc_x")),
        "gyro_z": _safe_float(row.get("gyro_z")),
        "heading": _safe_float(row.get("heading")),
        "brake": _safe_int(row.get("brake")),
        "blinker_l": _safe_int(row.get("blinker_l")),
        "blinker_r": _safe_int(row.get("blinker_r")),
        "rapidAccelerator": _safe_int(row.get("rapidAccelerator")),
        "rapidDecelerator": _safe_int(row.get("rapidDecelerator")),
        "eventType": str(row.get("eventType") or "").strip(),
        "eventId": str(row.get("eventId") or "").strip(),
        "eventDateTime": str(row.get("eventDateTime") or "").strip(),
        "eventName": str(row.get("name") or "").strip(),
        "eventRecordType": str(row.get("type") or "").strip(),
        "action_label": (
            _safe_int(row.get("action_label"), -1)
            if row.get("action_label") not in (None, "")
            else None
        ),
    }


def _fallback_taxi_id_from_path(csv_path: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(csv_path))[0]
    if stem.startswith("formatted_taxi"):
        return stem.replace("formatted_", "", 1)
    if stem.startswith("taxi"):
        return stem
    return None


def load_sensor_rows_from_csv(
    csv_path: str,
    *,
    fallback_taxi_id: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    rows: List[Dict[str, Any]] = []
    per_taxi: Dict[str, List[Dict[str, Any]]] = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for raw_row in csv.DictReader(f):
            row = _normalize_sensor_row(raw_row, fallback_taxi_id=fallback_taxi_id)
            if row is None:
                continue
            rows.append(row)
            per_taxi.setdefault(row["taxi_id"], []).append(row)

    for taxi_rows in per_taxi.values():
        taxi_rows.sort(key=lambda item: item["timestamp"])
    rows.sort(key=lambda item: ((item["sample_id"] is None), item["sample_id"] or 0, item["timestamp"]))
    return rows, per_taxi


def load_sensor_rows_from_path(source_path: str) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    if os.path.isdir(source_path):
        rows: List[Dict[str, Any]] = []
        per_taxi: Dict[str, List[Dict[str, Any]]] = {}
        for entry in sorted(os.listdir(source_path)):
            if not entry.lower().endswith(".csv"):
                continue
            csv_path = os.path.join(source_path, entry)
            file_rows, file_per_taxi = load_sensor_rows_from_csv(
                csv_path,
                fallback_taxi_id=_fallback_taxi_id_from_path(csv_path),
            )
            rows.extend(file_rows)
            for taxi_id, taxi_rows in file_per_taxi.items():
                per_taxi.setdefault(taxi_id, []).extend(taxi_rows)

        for taxi_rows in per_taxi.values():
            taxi_rows.sort(key=lambda item: item["timestamp"])
        rows.sort(key=lambda item: ((item["sample_id"] is None), item["sample_id"] or 0, item["timestamp"]))
        return rows, per_taxi

    return load_sensor_rows_from_csv(
        source_path,
        fallback_taxi_id=_fallback_taxi_id_from_path(source_path),
    )


def _rows_in_window(
    taxi_rows: List[Dict[str, Any]],
    center_timestamp: int,
    window_seconds: float,
) -> List[Dict[str, Any]]:
    timestamps = [row["timestamp"] for row in taxi_rows]
    window_ms = int(window_seconds * 1000.0)
    lower = center_timestamp - window_ms
    upper = center_timestamp + window_ms
    left = bisect.bisect_left(timestamps, lower)
    right = bisect.bisect_right(timestamps, upper)
    return taxi_rows[left:right]


def _window_quality(
    rows: List[Dict[str, Any]],
    center_timestamp: int,
    *,
    min_window_rows: int,
    max_dense_gap_ms: int,
    max_center_offset_ms: int,
) -> Dict[str, Any]:
    ordered = sorted(rows, key=lambda item: item["timestamp"])
    if not ordered:
        return {
            "quality": "snapshot_only",
            "runtime_source": "snapshot_only",
            "temporal_reliable": False,
            "pre_count": 0,
            "post_count": 0,
            "span_ms": 0,
            "max_gap_ms": 0,
            "gap_break_count": 0,
            "center_row_offset_ms": None,
        }

    pre = [row for row in ordered if row["timestamp"] < center_timestamp]
    post = [row for row in ordered if row["timestamp"] > center_timestamp]
    span_ms = ordered[-1]["timestamp"] - ordered[0]["timestamp"] if len(ordered) >= 2 else 0
    gaps = [ordered[idx + 1]["timestamp"] - ordered[idx]["timestamp"] for idx in range(len(ordered) - 1)]
    max_gap_ms = max(gaps, default=0)
    gap_break_count = sum(1 for gap in gaps if gap > max_dense_gap_ms)
    center_row_offset_ms = min(abs(row["timestamp"] - center_timestamp) for row in ordered)

    dense_window = (
        len(ordered) >= min_window_rows
        and bool(pre)
        and bool(post)
        and center_row_offset_ms <= max_center_offset_ms
        and max_gap_ms <= max_dense_gap_ms
    )
    sparse_window = len(ordered) >= 2

    if dense_window:
        quality = "dense_window"
        runtime_source = "preferred_window"
        temporal_reliable = True
    elif sparse_window:
        quality = "sparse_window"
        runtime_source = "expanded_window"
        temporal_reliable = False
    else:
        quality = "snapshot_only"
        runtime_source = "snapshot_only"
        temporal_reliable = False

    return {
        "quality": quality,
        "runtime_source": runtime_source,
        "temporal_reliable": temporal_reliable,
        "pre_count": len(pre),
        "post_count": len(post),
        "span_ms": span_ms,
        "max_gap_ms": max_gap_ms,
        "gap_break_count": gap_break_count,
        "center_row_offset_ms": center_row_offset_ms,
    }


def build_video_event_index(video_dir: str) -> Dict[str, List[Dict[str, Any]]]:
    index: Dict[str, List[Dict[str, Any]]] = {}
    if not video_dir or not os.path.isdir(video_dir):
        return index

    for taxi_id in sorted(os.listdir(video_dir)):
        taxi_path = os.path.join(video_dir, taxi_id)
        if not os.path.isdir(taxi_path):
            continue

        grouped: Dict[str, Dict[str, Any]] = {}
        for filename in sorted(os.listdir(taxi_path)):
            match = VIDEO_FILENAME_RE.match(filename)
            if not match:
                continue

            event_timestamp = match.group(1)
            camera = match.group(2)
            base_name = f"EVT_{taxi_id.replace('taxi', '')}_{event_timestamp}"
            event = grouped.setdefault(
                base_name,
                {
                    "base_name": base_name,
                    "event_timestamp_text": event_timestamp,
                    "event_timestamp_unix_ms": int(
                        dt.datetime.strptime(event_timestamp, "%Y%m%d%H%M%S").replace(tzinfo=JST).timestamp() * 1000.0
                    ),
                    "front_video": None,
                    "inner_video": None,
                },
            )
            if camera == "FRONT":
                event["front_video"] = filename
            elif camera == "INNER":
                event["inner_video"] = filename

        values = list(grouped.values())
        values.sort(key=lambda item: item["event_timestamp_unix_ms"])
        index[taxi_id] = values

    return index


def _match_video_event(
    video_events: List[Dict[str, Any]],
    center_timestamp: int,
    *,
    max_video_delta_seconds: float,
) -> Dict[str, Any]:
    if not video_events:
        return {
            "matched": False,
            "nearest_video_delta_seconds": None,
            "matched_video_delta_seconds": None,
            "matched_video_signed_delta_seconds": None,
            "matched_video_base": None,
            "matched_video_front": None,
            "matched_video_inner": None,
            "matched_video_has_front": False,
            "matched_video_has_inner": False,
        }

    nearest = min(video_events, key=lambda item: abs(item["event_timestamp_unix_ms"] - center_timestamp))
    signed_delta_seconds = (nearest["event_timestamp_unix_ms"] - center_timestamp) / 1000.0
    nearest_delta_seconds = abs(signed_delta_seconds)
    matched = nearest_delta_seconds <= max_video_delta_seconds
    return {
        "matched": matched,
        "nearest_video_delta_seconds": round(nearest_delta_seconds, 3),
        "matched_video_delta_seconds": round(nearest_delta_seconds, 3) if matched else None,
        "matched_video_signed_delta_seconds": round(signed_delta_seconds, 3) if matched else None,
        "matched_video_base": nearest["base_name"] if matched else None,
        "matched_video_front": nearest.get("front_video") if matched else None,
        "matched_video_inner": nearest.get("inner_video") if matched else None,
        "matched_video_has_front": bool(nearest.get("front_video")) if matched else False,
        "matched_video_has_inner": bool(nearest.get("inner_video")) if matched else False,
    }


def _summarize_event_metadata(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    event_points = [point for point in points if point.get("eventType")]
    event_type_counts = collections.Counter(point["eventType"] for point in event_points)
    unique_event_ids = sorted({point["eventId"] for point in event_points if point.get("eventId")})
    unique_event_datetimes = sorted({point["eventDateTime"] for point in event_points if point.get("eventDateTime")})
    return {
        "event_row_count": len(event_points),
        "event_type_counts": dict(sorted(event_type_counts.items())),
        "unique_event_ids": unique_event_ids,
        "unique_event_datetimes": unique_event_datetimes,
    }


def build_sensor_window_records(
    sample_csv_path: str,
    *,
    source_path: Optional[str] = None,
    source_csv_path: Optional[str] = None,
    video_dir: Optional[str] = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    min_window_rows: int = DEFAULT_MIN_WINDOW_ROWS,
    max_dense_gap_seconds: float = DEFAULT_MAX_DENSE_GAP_SECONDS,
    max_center_offset_seconds: float = DEFAULT_MAX_CENTER_OFFSET_SECONDS,
    max_video_delta_seconds: float = DEFAULT_MAX_VIDEO_DELTA_SECONDS,
) -> List[Dict[str, Any]]:
    effective_source_path = source_path or source_csv_path or sample_csv_path
    sample_rows, _ = load_sensor_rows_from_csv(sample_csv_path)
    _, source_per_taxi = load_sensor_rows_from_path(effective_source_path)
    video_index = build_video_event_index(video_dir) if video_dir else {}

    max_dense_gap_ms = int(max_dense_gap_seconds * 1000.0)
    max_center_offset_ms = int(max_center_offset_seconds * 1000.0)

    records: List[Dict[str, Any]] = []
    for sample in sample_rows:
        if sample["sample_id"] is None:
            continue

        taxi_rows = source_per_taxi.get(sample["taxi_id"], [])
        window_rows = _rows_in_window(taxi_rows, sample["timestamp"], window_seconds) if taxi_rows else []
        quality = _window_quality(
            window_rows,
            sample["timestamp"],
            min_window_rows=min_window_rows,
            max_dense_gap_ms=max_dense_gap_ms,
            max_center_offset_ms=max_center_offset_ms,
        )

        points = []
        for row in sorted(window_rows, key=lambda item: item["timestamp"]):
            point = dict(row)
            point["offset_ms"] = row["timestamp"] - sample["timestamp"]
            points.append(point)

        event_summary = _summarize_event_metadata(points)
        video_match = _match_video_event(
            video_index.get(sample["taxi_id"], []),
            sample["timestamp"],
            max_video_delta_seconds=max_video_delta_seconds,
        )

        records.append(
            {
                "sample_id": sample["sample_id"],
                "taxi_id": sample["taxi_id"],
                "center_timestamp": sample["timestamp"],
                "action_label": sample.get("action_label"),
                "window_seconds": window_seconds,
                "source_path": effective_source_path,
                "source_row_count": len(points),
                "quality": quality["quality"],
                "runtime_source": quality["runtime_source"],
                "temporal_reliable": quality["temporal_reliable"],
                "pre_count": quality["pre_count"],
                "post_count": quality["post_count"],
                "span_ms": quality["span_ms"],
                "max_gap_ms": quality["max_gap_ms"],
                "gap_break_count": quality["gap_break_count"],
                "center_row_offset_ms": quality["center_row_offset_ms"],
                **event_summary,
                **video_match,
                "points": points,
            }
        )

    records.sort(key=lambda item: item["sample_id"])
    return records


def save_sensor_window_records_jsonl(records: Iterable[Dict[str, Any]], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")


def load_sensor_window_records(jsonl_path: str) -> Dict[int, Dict[str, Any]]:
    records: Dict[int, Dict[str, Any]] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            sample_id = _safe_int(record.get("sample_id"), -1)
            if sample_id < 0:
                continue
            records[sample_id] = record
    return records
