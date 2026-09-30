import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.common.inference_adapters import resolve_adapter_spec
from benchmarks.perception.evaluation.structured_qa_evaluator import evaluate_item


class PublicInterfaceTests(unittest.TestCase):
    def test_local_adapter_spec_uses_only_config(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "config.json").write_text(
                json.dumps({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}),
                encoding="utf-8",
            )
            spec = resolve_adapter_spec(path)
            self.assertEqual(spec.adapter_name, "transformers_vlm")
            self.assertEqual(spec.model_type, "qwen3_vl")

    def test_triplet_f1_uses_explicit_reference_facts(self):
        gold = {
            "answer_type": "structured_label_plus_natural_language",
            "answer": {"clinical_facts": [["LV", "function", "reduced"]]},
        }
        prediction = {
            "prediction": {"clinical_facts": [["LV", "function", "reduced"]]},
        }
        result = evaluate_item(prediction, gold, "tee")
        self.assertEqual(result["components"]["fact_f1"], 1.0)

    def test_open_question_without_reference_facts_is_not_scorable(self):
        gold = {"answer_type": "structured_label_plus_natural_language", "answer": {"natural_language": "LV function reduced"}}
        result = evaluate_item({"prediction": {"natural_language": "LV function reduced"}}, gold, "tee")
        self.assertFalse(result["valid"])


if __name__ == "__main__":
    unittest.main()
