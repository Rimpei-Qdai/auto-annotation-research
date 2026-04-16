import sys
import unittest
from pathlib import Path


ANNOTATION_TOOL_DIR = Path(__file__).resolve().parents[1]
if str(ANNOTATION_TOOL_DIR) not in sys.path:
    sys.path.insert(0, str(ANNOTATION_TOOL_DIR))

from feedback_prompting import build_rag_feedback_prompt  # noqa: E402
from retrieval_index import NumericCaseRetriever, RetrievedCase, compute_retrieval_features  # noqa: E402


class RetrievalIndexTests(unittest.TestCase):
    def test_compute_retrieval_features_preserves_turn_sign(self):
        left = compute_retrieval_features(speed_kmh=20.0, acc_x=0.0, gyro_z=0.20)
        right = compute_retrieval_features(speed_kmh=20.0, acc_x=0.0, gyro_z=-0.20)

        self.assertGreater(left["lateral_offset_m"], 0.0)
        self.assertLess(right["lateral_offset_m"], 0.0)

    def test_retriever_prefers_closer_case(self):
        cases = [
            {
                "sample_id": 10,
                "macro_choice": "B",
                "action_label_11": 6,
                "action_label_name": "左折",
                "macro_name": "左回転系",
                "speed": 12.0,
                "acc_x": 0.0,
                "gyro_z": 0.18,
                "summary": "left-like",
                "retrieval_features": compute_retrieval_features(12.0, 0.0, 0.18),
            },
            {
                "sample_id": 20,
                "macro_choice": "C",
                "action_label_11": 7,
                "action_label_name": "右折",
                "macro_name": "右回転系",
                "speed": 12.0,
                "acc_x": 0.0,
                "gyro_z": -0.18,
                "summary": "right-like",
                "retrieval_features": compute_retrieval_features(12.0, 0.0, -0.18),
            },
        ]
        retriever = NumericCaseRetriever(cases)
        query = compute_retrieval_features(12.5, 0.0, -0.16)

        results = retriever.query(query, top_k=1)

        self.assertEqual(results[0].sample_id, 20)


class FeedbackPromptingTests(unittest.TestCase):
    def test_feedback_prompt_contains_retrieved_case_evidence(self):
        query = compute_retrieval_features(8.0, -0.2, 0.0)
        retrieved = [
            RetrievedCase(
                sample_id=42,
                distance=0.123,
                case={
                    "sample_id": 42,
                    "macro_choice": "D",
                    "macro_name": "その他",
                    "action_label_11": 4,
                    "action_label_name": "停止",
                    "speed": 2.0,
                    "acc_x": -0.4,
                    "gyro_z": 0.0,
                    "summary": "stop-like case",
                    "retrieval_features": compute_retrieval_features(2.0, -0.4, 0.0),
                },
            )
        ]

        prompt = build_rag_feedback_prompt(
            initial_macro_choice="A",
            query_features=query,
            retrieved_cases=retrieved,
        )

        self.assertIn("sample 42", prompt)
        self.assertIn("停止", prompt)
        self.assertIn("A/B/C/D", prompt)


if __name__ == "__main__":
    unittest.main()
