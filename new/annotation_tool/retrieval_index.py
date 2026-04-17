"""
Numeric retrieval utilities for case-based RAG feedback.

The first version intentionally avoids embeddings and external services.
We retrieve similar cases using a small set of interpretable sensor / geometry
features so that the behavior is easy to inspect and debug.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import math


def compute_retrieval_features(
    speed_kmh: float,
    acc_x: float,
    gyro_z: float,
    *,
    horizon_s: float = 3.0,
    dt: float = 0.1,
) -> Dict[str, float]:
    """
    Approximate future trajectory geometry from current sensor values.

    This is intentionally lightweight and deterministic:
    - speed / acc_x control forward progression
    - gyro_z controls signed heading change and lateral offset
    """
    steps = max(1, int(horizon_s / dt))
    speed_ms = max(0.0, float(speed_kmh) / 3.6)
    acc_x = float(acc_x)
    gyro_z = float(gyro_z)

    heading = 0.0
    x = 0.0
    y = 0.0
    speeds: List[float] = []

    for step in range(steps):
        t = step * dt
        current_speed = max(0.0, speed_ms + acc_x * t)
        speeds.append(current_speed)
        heading += gyro_z * dt
        x += current_speed * math.cos(heading) * dt
        y += current_speed * math.sin(heading) * dt

    curvature = y / max(abs(x), 1e-3)
    avg_speed_ms = sum(speeds) / len(speeds)
    end_speed_ms = speeds[-1]

    return {
        "speed_kmh": float(speed_kmh),
        "acc_x": acc_x,
        "gyro_z": gyro_z,
        "forward_distance_m": x,
        "lateral_offset_m": y,
        "heading_delta_rad": heading,
        "curvature": curvature,
        "avg_speed_ms": avg_speed_ms,
        "end_speed_ms": end_speed_ms,
    }


@dataclass(frozen=True)
class RetrievedCase:
    sample_id: int
    distance: float
    case: Dict[str, Any]


class NumericCaseRetriever:
    """Retrieve similar cases using weighted numeric distance."""

    FEATURE_WEIGHTS = {
        "speed_kmh": 0.07,
        "acc_x": 0.9,
        "gyro_z": 3.2,
        "forward_distance_m": 0.15,
        "lateral_offset_m": 0.8,
        "heading_delta_rad": 1.6,
        "curvature": 1.2,
    }

    def __init__(self, cases: Iterable[Dict[str, Any]]):
        self.cases = list(cases)

    def _distance(
        self,
        query_features: Dict[str, float],
        case_features: Dict[str, float],
    ) -> float:
        distance = 0.0
        for feature_name, weight in self.FEATURE_WEIGHTS.items():
            q = float(query_features.get(feature_name, 0.0))
            c = float(case_features.get(feature_name, 0.0))
            distance += weight * abs(q - c)
        return distance

    def query(
        self,
        query_features: Dict[str, float],
        *,
        top_k: int = 3,
        exclude_sample_id: Optional[int] = None,
        min_cases: int = 1,
        max_distance: Optional[float] = None,
        require_diverse_macros: bool = False,
    ) -> List[RetrievedCase]:
        ranked: List[RetrievedCase] = []
        for case in self.cases:
            if exclude_sample_id is not None and case.get("sample_id") == exclude_sample_id:
                continue

            case_features = case.get("retrieval_features", {})
            distance = self._distance(query_features, case_features)
            ranked.append(
                RetrievedCase(
                    sample_id=int(case["sample_id"]),
                    distance=distance,
                    case=case,
                )
            )

        ranked.sort(key=lambda item: (item.distance, item.sample_id))

        if max_distance is not None:
            ranked = [item for item in ranked if item.distance <= max_distance]

        if require_diverse_macros and ranked:
            selected: List[RetrievedCase] = []
            selected_ids = set()
            seen_macros = set()

            for item in ranked:
                macro_choice = item.case.get("macro_choice")
                if macro_choice in seen_macros:
                    continue
                selected.append(item)
                selected_ids.add(item.sample_id)
                seen_macros.add(macro_choice)
                if len(selected) >= top_k:
                    break

            if len(selected) < top_k:
                for item in ranked:
                    if item.sample_id in selected_ids:
                        continue
                    selected.append(item)
                    selected_ids.add(item.sample_id)
                    if len(selected) >= top_k:
                        break

            ranked = selected

        if len(ranked) < min_cases:
            return []

        return ranked[:top_k]
