import csv
import sys
import tempfile
import unittest
from pathlib import Path


ANNOTATION_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_TOOL_DIR))

from sensor_window_dataset import (  # noqa: E402
    build_sensor_window_records,
    load_sensor_window_records,
    save_sensor_window_records_jsonl,
)


FIELDNAMES = [
    "sample_id",
    "timestamp",
    "taxi_id",
    "imei",
    "speed",
    "acc_x",
    "gyro_z",
    "heading",
    "brake",
    "blinker_l",
    "blinker_r",
    "rapidAccelerator",
    "rapidDecelerator",
    "eventType",
    "eventId",
    "eventDateTime",
    "name",
    "type",
    "action_label",
]


class SensorWindowDatasetTests(unittest.TestCase):
    def _write_csv(self, path: Path, rows):
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def test_builds_dense_window_when_source_is_contiguous(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sample_csv = tmp_path / "samples.csv"
            source_csv = tmp_path / "source.csv"
            self._write_csv(
                sample_csv,
                [
                    {
                        "sample_id": 1,
                        "timestamp": 4000,
                        "taxi_id": "taxi-a",
                        "imei": "",
                        "speed": 12.0,
                        "acc_x": 0.0,
                        "gyro_z": 0.05,
                        "heading": 10.0,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": 1,
                    }
                ],
            )
            source_rows = []
            for idx, ts in enumerate(range(1000, 8000, 1000)):
                source_rows.append(
                    {
                        "sample_id": "",
                        "timestamp": ts,
                        "taxi_id": "taxi-a",
                        "imei": "",
                        "speed": 10.0 + idx,
                        "acc_x": 0.1,
                        "gyro_z": 0.02 * idx,
                        "heading": 5.0 * idx,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": "",
                    }
                )
            self._write_csv(source_csv, source_rows)

            records = build_sensor_window_records(
                str(sample_csv),
                source_csv_path=str(source_csv),
                window_seconds=3.0,
                min_window_rows=5,
                max_dense_gap_seconds=1.5,
                max_center_offset_seconds=1.5,
            )
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["quality"], "dense_window")
            self.assertEqual(record["runtime_source"], "preferred_window")
            self.assertTrue(record["temporal_reliable"])
            self.assertEqual(record["pre_count"], 3)
            self.assertEqual(record["post_count"], 3)
            self.assertEqual(record["max_gap_ms"], 1000)
            self.assertEqual(record["event_row_count"], 0)
            self.assertFalse(record["matched"])
            self.assertEqual(len(record["points"]), 7)

    def test_marks_sparse_window_when_large_gap_breaks_density(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sample_csv = tmp_path / "samples.csv"
            source_csv = tmp_path / "source.csv"
            self._write_csv(
                sample_csv,
                [
                    {
                        "sample_id": 3,
                        "timestamp": 3000,
                        "taxi_id": "taxi-b",
                        "imei": "",
                        "speed": 8.0,
                        "acc_x": -0.1,
                        "gyro_z": 0.0,
                        "heading": 20.0,
                        "brake": 1,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 1,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": 4,
                    }
                ],
            )
            self._write_csv(
                source_csv,
                [
                    {
                        "sample_id": "",
                        "timestamp": 0,
                        "taxi_id": "taxi-b",
                        "imei": "",
                        "speed": 6.0,
                        "acc_x": -0.1,
                        "gyro_z": 0.0,
                        "heading": 20.0,
                        "brake": 1,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": "",
                    },
                    {
                        "sample_id": "",
                        "timestamp": 3000,
                        "taxi_id": "taxi-b",
                        "imei": "",
                        "speed": 5.0,
                        "acc_x": -0.1,
                        "gyro_z": 0.0,
                        "heading": 20.0,
                        "brake": 1,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": "",
                    },
                    {
                        "sample_id": "",
                        "timestamp": 6000,
                        "taxi_id": "taxi-b",
                        "imei": "",
                        "speed": 4.0,
                        "acc_x": -0.1,
                        "gyro_z": 0.0,
                        "heading": 20.0,
                        "brake": 1,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": "",
                    },
                ],
            )

            record = build_sensor_window_records(
                str(sample_csv),
                source_csv_path=str(source_csv),
                window_seconds=3.0,
                min_window_rows=3,
                max_dense_gap_seconds=1.5,
                max_center_offset_seconds=1.5,
            )[0]
            self.assertEqual(record["quality"], "sparse_window")
            self.assertEqual(record["runtime_source"], "expanded_window")
            self.assertFalse(record["temporal_reliable"])
            self.assertEqual(record["max_gap_ms"], 3000)
            self.assertGreaterEqual(record["gap_break_count"], 1)

    def test_loads_directory_source_and_fallback_taxi_id_from_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sample_csv = tmp_path / "samples.csv"
            source_dir = tmp_path / "formatted_data"
            source_dir.mkdir()
            source_csv = source_dir / "formatted_taxi-c.csv"
            self._write_csv(
                sample_csv,
                [
                    {
                        "sample_id": 5,
                        "timestamp": 2000,
                        "taxi_id": "taxi-c",
                        "imei": "",
                        "speed": 10.0,
                        "acc_x": 0.0,
                        "gyro_z": 0.0,
                        "heading": 0.0,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": 1,
                    }
                ],
            )
            self._write_csv(
                source_csv,
                [
                    {
                        "sample_id": "",
                        "timestamp": 1000,
                        "taxi_id": "",
                        "imei": "",
                        "speed": 8.0,
                        "acc_x": 0.0,
                        "gyro_z": 0.0,
                        "heading": 0.0,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": "",
                    },
                    {
                        "sample_id": "",
                        "timestamp": 2000,
                        "taxi_id": "",
                        "imei": "",
                        "speed": 9.0,
                        "acc_x": 0.0,
                        "gyro_z": 0.0,
                        "heading": 0.0,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": "",
                    },
                    {
                        "sample_id": "",
                        "timestamp": 3000,
                        "taxi_id": "",
                        "imei": "",
                        "speed": 10.0,
                        "acc_x": 0.0,
                        "gyro_z": 0.0,
                        "heading": 0.0,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": "",
                    },
                ],
            )

            record = build_sensor_window_records(
                str(sample_csv),
                source_path=str(source_dir),
                window_seconds=1.0,
                min_window_rows=3,
                max_dense_gap_seconds=1.5,
                max_center_offset_seconds=1.5,
            )[0]
            self.assertEqual(record["taxi_id"], "taxi-c")
            self.assertEqual(record["source_row_count"], 3)

    def test_adds_event_and_video_metadata_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            sample_csv = tmp_path / "samples.csv"
            source_dir = tmp_path / "formatted_data"
            video_dir = tmp_path / "filtered_video" / "taxi-d"
            source_dir.mkdir()
            video_dir.mkdir(parents=True)
            source_csv = source_dir / "formatted_taxi-d.csv"
            self._write_csv(
                sample_csv,
                [
                    {
                        "sample_id": 8,
                        "timestamp": 1726408303246,
                        "taxi_id": "taxi-d",
                        "imei": "",
                        "speed": 4.0,
                        "acc_x": 0.0,
                        "gyro_z": 0.0,
                        "heading": 0.0,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": 1,
                    }
                ],
            )
            self._write_csv(
                source_csv,
                [
                    {
                        "sample_id": "",
                        "timestamp": 1726408302246,
                        "taxi_id": "",
                        "imei": "",
                        "speed": 3.0,
                        "acc_x": 0.0,
                        "gyro_z": 0.0,
                        "heading": 0.0,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "11",
                        "eventId": "evt-1",
                        "eventDateTime": "20240915225144.0",
                        "name": "EVT_352176111000000_20240915225144.gz",
                        "type": "EVT",
                        "action_label": "",
                    },
                    {
                        "sample_id": "",
                        "timestamp": 1726408303246,
                        "taxi_id": "",
                        "imei": "",
                        "speed": 4.0,
                        "acc_x": 0.0,
                        "gyro_z": 0.0,
                        "heading": 0.0,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "11",
                        "eventId": "evt-1",
                        "eventDateTime": "20240915225144.0",
                        "name": "EVT_352176111000000_20240915225144.gz",
                        "type": "EVT",
                        "action_label": "",
                    },
                    {
                        "sample_id": "",
                        "timestamp": 1726408304246,
                        "taxi_id": "",
                        "imei": "",
                        "speed": 5.0,
                        "acc_x": 0.0,
                        "gyro_z": 0.0,
                        "heading": 0.0,
                        "brake": 0,
                        "blinker_l": 0,
                        "blinker_r": 0,
                        "rapidAccelerator": 0,
                        "rapidDecelerator": 0,
                        "eventType": "",
                        "eventId": "",
                        "eventDateTime": "",
                        "name": "",
                        "type": "",
                        "action_label": "",
                    },
                ],
            )
            (video_dir / "EVT_352176111000000_20240915225144_FRONT.mp4").touch()
            (video_dir / "EVT_352176111000000_20240915225144_INNER.mp4").touch()

            record = build_sensor_window_records(
                str(sample_csv),
                source_path=str(source_dir),
                video_dir=str(tmp_path / "filtered_video"),
                window_seconds=1.0,
                min_window_rows=3,
                max_dense_gap_seconds=2.0,
                max_center_offset_seconds=1.5,
                max_video_delta_seconds=60.0,
            )[0]
            self.assertEqual(record["event_row_count"], 2)
            self.assertEqual(record["event_type_counts"]["11"], 2)
            self.assertTrue(record["matched"])
            self.assertTrue(record["matched_video_has_front"])
            self.assertTrue(record["matched_video_has_inner"])
            self.assertIsNotNone(record["matched_video_base"])

    def test_jsonl_round_trip_preserves_records(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            output_path = tmp_path / "sensor_windows.jsonl"
            records = [
                {
                    "sample_id": 1,
                    "taxi_id": "taxi-a",
                    "center_timestamp": 3000,
                    "quality": "dense_window",
                    "runtime_source": "preferred_window",
                    "temporal_reliable": True,
                    "pre_count": 3,
                    "post_count": 3,
                    "span_ms": 6000,
                    "max_gap_ms": 1000,
                    "gap_break_count": 0,
                    "center_row_offset_ms": 0,
                    "points": [{"timestamp": 3000, "speed": 10.0, "acc_x": 0.0, "gyro_z": 0.0, "heading": 0.0}],
                }
            ]
            save_sensor_window_records_jsonl(records, str(output_path))
            loaded = load_sensor_window_records(str(output_path))
            self.assertIn(1, loaded)
            self.assertEqual(loaded[1]["quality"], "dense_window")
            self.assertEqual(loaded[1]["points"][0]["timestamp"], 3000)


if __name__ == "__main__":
    unittest.main()
