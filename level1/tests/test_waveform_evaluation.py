import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from level1.evaluation.waveform_metrics import evaluate_records
from level1.evaluation.waveform_schema import BenchmarkSample, PredictionRecord


PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = PROJECT_ROOT / "level1/evaluation/evaluate_waveform_choices.py"


class WaveformChoiceEvaluationTests(unittest.TestCase):
    def test_exact_choice_metrics_and_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            benchmark = root / "benchmark.jsonl"
            predictions = root / "predictions.jsonl"
            metrics = root / "metrics.json"
            details = root / "details.jsonl"
            benchmark.write_text(
                json.dumps(
                    {"question_id": "q1", "evaluation": {"answer_key": "B"}}
                )
                + "\n",
                encoding="utf-8",
            )
            predictions.write_text(
                json.dumps({"question_id": "q1", "answer": "b"}) + "\n",
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(EVALUATOR),
                    "--benchmark",
                    str(benchmark),
                    "--predictions",
                    str(predictions),
                    "--metrics-output",
                    str(metrics),
                    "--details-output",
                    str(details),
                ],
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(metrics.read_text(encoding="utf-8"))
            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(result["coverage"], 1.0)
            self.assertTrue(result["complete"])

    def test_structured_waveform_micro_f1(self):
        sample = BenchmarkSample.model_validate(
            {
                "sample_id": "wave-1",
                "patient_id": "patient-1",
                "task": "waveform_understanding",
                "modality": "ecg",
                "media": [{"path": "wave.png", "role": "short_waveform"}],
                "prompt": "Describe the waveform.",
                "answer": {
                    "waveforms": [
                        {
                            "channel": "ECG",
                            "morphology": ["narrow QRS"],
                            "rhythm": ["sinus rhythm"],
                        }
                    ]
                },
                "provenance": {
                    "dataset": "fixture",
                    "source_record": "wave-1",
                    "label_source": "unit-test",
                    "label_tier": "gold",
                },
                "split": "test",
            }
        )
        prediction = PredictionRecord.model_validate(
            {
                "sample_id": "wave-1",
                "prediction": {
                    "waveforms": [
                        {
                            "channel": "ECG",
                            "morphology": ["narrow QRS"],
                            "rhythm": ["sinus rhythm"],
                        }
                    ]
                },
            }
        )

        metrics = evaluate_records([sample], [prediction])
        self.assertEqual(metrics["waveform_label_micro_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
