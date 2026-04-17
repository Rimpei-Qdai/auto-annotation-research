"""
Case memory for local RAG feedback.

This module builds a compact memory of manually annotated cases from local CSVs.
It does not require a vector database or external service.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from config import ACTION_LABELS, MACRO_OUTPUT_NAMES
from retrieval_index import compute_retrieval_features


MANUAL_FILENAME = "annotated_samples_manual.0121.csv"
SAMPLE_FILENAME = "annotation_samples.csv"


def _label_to_macro(action_label: int) -> str:
    if action_label in {1, 2, 3}:
        return "A"
    if action_label in {6, 8}:
        return "B"
    if action_label in {7, 9, 10}:
        return "C"
    return "D"


def _manual_csv_candidates(repo_root: str) -> List[str]:
    root = Path(repo_root).resolve()
    return [
        str(root / "new" / "results" / MANUAL_FILENAME),
        str(root / "results" / MANUAL_FILENAME),
        "/Users/rimpeihata/Desktop/auto-annotation-research/new/results/annotated_samples_manual.0121.csv",
    ]


def _sample_csv_candidates(repo_root: str) -> List[str]:
    root = Path(repo_root).resolve()
    return [
        str(root / "new" / "sample" / SAMPLE_FILENAME),
        str(root / "sample" / SAMPLE_FILENAME),
        "/Users/rimpeihata/Desktop/auto-annotation-research/new/sample/annotation_samples.csv",
    ]


def _resolve_existing_path(candidates: List[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No existing path found in candidates: {candidates}")


class CaseMemory:
    """Load and expose manually labeled cases for retrieval."""

    def __init__(self, repo_root: str):
        self.repo_root = str(Path(repo_root).resolve())
        self.manual_csv_path = _resolve_existing_path(_manual_csv_candidates(repo_root))
        self.sample_csv_path = _resolve_existing_path(_sample_csv_candidates(repo_root))
        self._cases: List[Dict[str, Any]] = []
        self._loaded = False

    @property
    def cases(self) -> List[Dict[str, Any]]:
        if not self._loaded:
            self._cases = self._load_cases()
            self._loaded = True
        return self._cases

    def _load_cases(self) -> List[Dict[str, Any]]:
        df_manual = pd.read_csv(self.manual_csv_path)
        df_manual["sample_id"] = df_manual["sample_id"].astype(int)
        df_manual["action_label"] = df_manual["action_label"].astype("Int64")

        df_samples = pd.read_csv(self.sample_csv_path)
        df_samples["sample_id"] = df_samples["sample_id"].astype(int)

        merge_cols = [
            "sample_id",
            "timestamp",
            "speed",
            "acc_x",
            "gyro_z",
            "latitude",
            "longitude",
        ]
        df = df_manual.merge(df_samples[merge_cols], on=["sample_id", "timestamp"], how="left")
        df = df[df["action_label"].notna()].copy()

        cases: List[Dict[str, Any]] = []
        for row in df.to_dict(orient="records"):
            action_label = int(row["action_label"])
            macro_choice = _label_to_macro(action_label)
            retrieval_features = compute_retrieval_features(
                speed_kmh=float(row.get("speed", 0.0) or 0.0),
                acc_x=float(row.get("acc_x", 0.0) or 0.0),
                gyro_z=float(row.get("gyro_z", 0.0) or 0.0),
            )
            cases.append(
                {
                    "sample_id": int(row["sample_id"]),
                    "timestamp": int(row["timestamp"]),
                    "action_label_11": action_label,
                    "action_label_name": ACTION_LABELS.get(action_label, str(action_label)),
                    "macro_choice": macro_choice,
                    "macro_name": MACRO_OUTPUT_NAMES.get(macro_choice, macro_choice),
                    "speed": float(row.get("speed", 0.0) or 0.0),
                    "acc_x": float(row.get("acc_x", 0.0) or 0.0),
                    "gyro_z": float(row.get("gyro_z", 0.0) or 0.0),
                    "latitude": float(row.get("latitude", 0.0) or 0.0),
                    "longitude": float(row.get("longitude", 0.0) or 0.0),
                    "retrieval_features": retrieval_features,
                    "summary": self._summarize_case(
                        macro_choice=macro_choice,
                        action_label=action_label,
                        retrieval_features=retrieval_features,
                    ),
                }
            )

        return cases

    def _summarize_case(
        self,
        *,
        macro_choice: str,
        action_label: int,
        retrieval_features: Dict[str, float],
    ) -> str:
        return (
            f"{MACRO_OUTPUT_NAMES.get(macro_choice, macro_choice)} / "
            f"{ACTION_LABELS.get(action_label, action_label)}; "
            f"forward={retrieval_features['forward_distance_m']:.1f}m, "
            f"lateral={retrieval_features['lateral_offset_m']:.1f}m, "
            f"heading={retrieval_features['heading_delta_rad']:.2f}rad"
        )

    def find_case_by_sample_id(self, sample_id: int) -> Optional[Dict[str, Any]]:
        for case in self.cases:
            if case["sample_id"] == sample_id:
                return case
        return None
