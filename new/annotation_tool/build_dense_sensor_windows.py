#!/usr/bin/env python3
"""
Build sample-aligned sensor windows from an external telemetry source.

By default, when `original_data/formatted_data` exists, this script uses it as
the dense sensor source and also attaches nearest FRONT/INNER event video
metadata from `original_data/filtered_video`.
"""

from __future__ import annotations

import argparse
import collections
import os

from sensor_window_dataset import (
    DEFAULT_MAX_DENSE_GAP_SECONDS,
    DEFAULT_MAX_VIDEO_DELTA_SECONDS,
    build_sensor_window_records,
    default_sensor_window_dataset_path,
    save_sensor_window_records_jsonl,
)


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project_root() -> str:
    return os.path.dirname(_repo_root())


def main() -> int:
    repo_root = _repo_root()
    project_root = _project_root()
    sample_csv_default = os.path.join(repo_root, "sample", "annotation_samples.csv")
    formatted_data_default = os.path.join(project_root, "original_data", "formatted_data")
    filtered_video_default = os.path.join(project_root, "original_data", "filtered_video")
    source_default = formatted_data_default if os.path.isdir(formatted_data_default) else sample_csv_default
    video_default = filtered_video_default if os.path.isdir(filtered_video_default) else None
    max_dense_gap_default = 0.1 if os.path.isdir(formatted_data_default) else DEFAULT_MAX_DENSE_GAP_SECONDS

    parser = argparse.ArgumentParser(description="Build dense sensor window dataset")
    parser.add_argument(
        "--sample-csv",
        default=sample_csv_default,
        help="Evaluation sample CSV with sample_id/taxi_id/timestamp",
    )
    parser.add_argument(
        "--source-path",
        "--source-csv",
        dest="source_path",
        default=source_default,
        help="Dense telemetry file or directory. Defaults to original_data/formatted_data when available.",
    )
    parser.add_argument(
        "--video-dir",
        default=video_default,
        help="Optional filtered_video directory for nearest FRONT/INNER clip metadata.",
    )
    parser.add_argument(
        "--output",
        default=default_sensor_window_dataset_path(repo_root),
        help="Output JSONL path",
    )
    parser.add_argument("--window-seconds", type=float, default=3.0)
    parser.add_argument("--min-window-rows", type=int, default=3)
    parser.add_argument("--max-dense-gap-seconds", type=float, default=max_dense_gap_default)
    parser.add_argument("--max-center-offset-seconds", type=float, default=1.5)
    parser.add_argument("--max-video-delta-seconds", type=float, default=DEFAULT_MAX_VIDEO_DELTA_SECONDS)
    args = parser.parse_args()

    records = build_sensor_window_records(
        args.sample_csv,
        source_path=args.source_path,
        video_dir=args.video_dir,
        window_seconds=args.window_seconds,
        min_window_rows=args.min_window_rows,
        max_dense_gap_seconds=args.max_dense_gap_seconds,
        max_center_offset_seconds=args.max_center_offset_seconds,
        max_video_delta_seconds=args.max_video_delta_seconds,
    )
    save_sensor_window_records_jsonl(records, args.output)

    quality_counts = collections.Counter(record["quality"] for record in records)
    matched_video_count = sum(1 for record in records if record.get("matched"))
    eventful_count = sum(1 for record in records if record.get("event_row_count", 0) > 0)
    print(f"built_records={len(records)}")
    for quality, count in sorted(quality_counts.items()):
        print(f"{quality}={count}")
    print(f"matched_video={matched_video_count}")
    print(f"eventful_windows={eventful_count}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
