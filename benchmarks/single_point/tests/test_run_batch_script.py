import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNNER = PROJECT_ROOT / "benchmarks/single_point/inference/run_batch.sh"
BASH = shutil.which("bash")


@unittest.skipUnless(BASH, "bash is required for shell-runner tests")
class LevelTwoRunnerScriptTests(unittest.TestCase):
    def _fixture(self, root):
        model = root / "model-a"
        model.mkdir()
        (model / "config.json").write_text("{}\n", encoding="utf-8")
        log = root / "calls.tsv"
        data_root = root / "data"
        data_root.mkdir()
        for language in ("zh", "en"):
            (data_root / f"Level_two_B5_v2_{language}.jsonl").touch()
        fake_python = root / "fake-python"
        fake_python.write_text(
            "#!/usr/bin/env bash\n"
            "exec >>\"$ANESTRACE_TEST_LOG\"\n"
            "for argument in \"$@\"; do printf '%s\\t' \"$argument\"; done\n"
            "printf '\\n'\n",
            encoding="utf-8",
        )
        fake_python.chmod(0o755)
        environment = os.environ.copy()
        environment["ANESTRACE_TEST_LOG"] = str(log)
        environment["ANESTRACE_DATA_DIR"] = str(data_root)
        return model, log, fake_python, environment

    def test_english_mode_uses_english_data_and_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, log, fake_python, environment = self._fixture(Path(tmp))
            completed = subprocess.run(
                [
                    BASH,
                    str(RUNNER),
                    "--python",
                    str(fake_python),
                    "--model-path",
                    str(model),
                    "--language",
                    "en",
                    "--limit",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            call = log.read_text(encoding="utf-8").rstrip("\n").split("\t")[:-1]
            self.assertEqual(
                Path(call[1]).name, "Level_two_B5_v2_en.jsonl"
            )
            self.assertEqual(call[call.index("--language") + 1], "en")
            self.assertEqual(
                Path(call[call.index("--output") + 1]).name,
                "b5-text-only-en.jsonl",
            )

    def test_both_runs_english_then_chinese(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, log, fake_python, environment = self._fixture(Path(tmp))
            completed = subprocess.run(
                [
                    BASH,
                    str(RUNNER),
                    "--python",
                    str(fake_python),
                    "--model-path",
                    str(model),
                    "--language",
                    "both",
                    "--limit",
                    "1",
                ],
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            calls = [
                line.split("\t")[:-1]
                for line in log.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(calls), 2)
            self.assertEqual(
                [call[call.index("--language") + 1] for call in calls],
                ["en", "zh"],
            )
            self.assertEqual(
                [Path(call[1]).name for call in calls],
                ["Level_two_B5_v2_en.jsonl", "Level_two_B5_v2_zh.jsonl"],
            )
            self.assertEqual(
                [Path(call[call.index("--output") + 1]).name for call in calls],
                [
                    "b5-text-only-en.jsonl",
                    "b5-text-only.jsonl",
                ],
            )

    def test_failed_middle_model_is_reported_and_later_model_still_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = tuple(root / name for name in ("model-a", "model-b", "model-c"))
            for model in models:
                model.mkdir()
                (model / "config.json").write_text("{}\n", encoding="utf-8")
            log = root / "calls.tsv"
            data_root = root / "data"
            data_root.mkdir()
            (data_root / "Level_two_B5_v2_en.jsonl").touch()
            fake_python = root / "fake-python"
            fake_python.write_text(
                "#!/usr/bin/env bash\n"
                "status=0\n"
                "exec >>\"$ANESTRACE_TEST_LOG\"\n"
                "for argument in \"$@\"; do\n"
                "  printf '%s\\t' \"$argument\"\n"
                "  if [[ \"$argument\" == */model-b ]]; then status=7; fi\n"
                "done\n"
                "printf '\\n'\n"
                "exit \"$status\"\n",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            environment = os.environ.copy()
            environment["ANESTRACE_TEST_LOG"] = str(log)
            environment["ANESTRACE_DATA_DIR"] = str(data_root)
            command = [BASH, str(RUNNER), "--python", str(fake_python), "--language", "en"]
            for model in models:
                command.extend(("--model-path", str(model)))

            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 7, completed.stderr)
            calls = log.read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(calls), 3)
            self.assertIn(str(models[2]), calls[2])
            self.assertIn(
                "continuing with the next run", completed.stderr
            )
            self.assertIn("model-b [en] exit=7", completed.stderr)
            self.assertIn(
                f"Starting AnesTRACE model: {models[2]}", completed.stdout
            )

    def test_invalid_language_and_both_overrides_are_rejected(self):
        completed = subprocess.run(
            [BASH, str(RUNNER), "--language", "fr"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("--language must be zh, en, or both", completed.stderr)

        for option in ("--data", "--output"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as tmp:
                value = Path(tmp) / "value.jsonl"
                value.touch()
                completed = subprocess.run(
                    [BASH, str(RUNNER), "--language", "both", option, str(value)],
                    cwd=PROJECT_ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 2)
                self.assertIn(
                    f"{option} cannot be used with --language both",
                    completed.stderr,
                )

    def test_model_path_is_required(self):
        completed = subprocess.run(
            [BASH, str(RUNNER)],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertIn("At least one --model-path is required", completed.stderr)


if __name__ == "__main__":
    unittest.main()
