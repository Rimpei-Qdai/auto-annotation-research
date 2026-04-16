"""
Prompt builders for RAG-style feedback inference.
"""

from __future__ import annotations

from typing import Iterable

from config import MACRO_OUTPUT_NAMES, RAG_FEEDBACK_PROMPT_TEMPLATE
from retrieval_index import RetrievedCase


def format_retrieved_cases_block(retrieved_cases: Iterable[RetrievedCase]) -> str:
    lines = []
    for idx, retrieved in enumerate(retrieved_cases, start=1):
        case = retrieved.case
        features = case.get("retrieval_features", {})
        lines.append(
            (
                f"{idx}. sample {case['sample_id']}: {case['macro_choice']} ({case['macro_name']}) / "
                f"{case['action_label_name']} | distance={retrieved.distance:.3f}\n"
                f"   speed={case['speed']:.1f} km/h, acc_x={case['acc_x']:.3f}, gyro_z={case['gyro_z']:.3f}, "
                f"forward={features.get('forward_distance_m', 0.0):.2f}m, "
                f"lateral={features.get('lateral_offset_m', 0.0):.2f}m, "
                f"heading={features.get('heading_delta_rad', 0.0):.3f}rad\n"
                f"   summary: {case['summary']}"
            )
        )
    return "\n".join(lines) if lines else "類似事例はありません。初回判定と画像をもとに再評価してください。"


def build_rag_feedback_prompt(
    *,
    initial_macro_choice: str,
    query_features: dict,
    retrieved_cases: Iterable[RetrievedCase],
) -> str:
    return RAG_FEEDBACK_PROMPT_TEMPLATE.format(
        initial_macro_choice=initial_macro_choice,
        initial_macro_name=MACRO_OUTPUT_NAMES.get(initial_macro_choice, initial_macro_choice),
        speed=float(query_features.get("speed_kmh", 0.0)),
        acc_x=float(query_features.get("acc_x", 0.0)),
        gyro_z=float(query_features.get("gyro_z", 0.0)),
        forward_distance_m=float(query_features.get("forward_distance_m", 0.0)),
        lateral_offset_m=float(query_features.get("lateral_offset_m", 0.0)),
        heading_delta_rad=float(query_features.get("heading_delta_rad", 0.0)),
        retrieved_cases_block=format_retrieved_cases_block(retrieved_cases),
    )
