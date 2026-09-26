import json
import tempfile
import unittest
from pathlib import Path

import run_inference as runner


def valid_record(**overrides):
    record = {
        "qa_id": "case-1-B5_complete_plan",
        "sample_id": "case-1",
        "task_group": "perioperative_decision_making",
        "task_name": "B5_complete_plan",
        "answer_type": "open_ended",
        "split": "train",
        "patient_information": "【患者档案】\n- 年龄：50岁。",
    }
    record.update(overrides)
    return record


def valid_response():
    bodies = {
        "风险预测": "可能事件：低血压。\n发生可能性：中。\n严重程度：中度。\n预计演变方向：不确定。",
        "预测依据": "- MAP呈下降趋势。",
        "诊断结论": "当前状态与严重程度：中度低血压。\n最可能诊断：低血压，病因待查。",
        "诊断依据": "- 当前MAP低于基线。",
        "决策结论": "处置等级：及时干预。\n当前处理目标：恢复灌注压。",
        "具体动作": "- 首先核验血压。",
        "复评计划": "复评时间：3分钟。\n重点复评：MAP。\n治疗目标：MAP回升。\n干预失败或升级标准：MAP继续下降。",
        "备用与升级方案": "首选方案无效时请求上级支持。",
    }
    return "\n\n".join(
        f"【{heading}】\n{bodies[heading]}" for heading in runner.EXPECTED_HEADINGS
    )


def valid_english_response():
    bodies = {
        "Risk Prediction": (
            "Predicted Event: Hypotension.\nLikelihood: Moderate.\n"
            "Severity: Moderate.\nExpected Trajectory: Uncertain."
        ),
        "Prediction Evidence": "- MAP is declining.",
        "Acute Diagnosis": (
            "Current State and Severity: Moderate hypotension.\n"
            "Most Likely Diagnosis: Hypotension; etiology remains uncertain."
        ),
        "Diagnostic Evidence": "- Current MAP is below baseline.",
        "Management Decision": (
            "Intervention Level: Prompt Intervention.\n"
            "Immediate Management Goal: Restore perfusion pressure."
        ),
        "Specific Actions": "- First verify the blood pressure measurement.",
        "Reassessment Plan": (
            "Reassessment Time: 3 minutes.\nKey Parameters: MAP.\n"
            "Treatment Target: MAP improves.\n"
            "Failure / Escalation Criteria: MAP continues to decline."
        ),
        "Backup & Escalation Plan": (
            "Request senior assistance if the initial plan is ineffective."
        ),
    }
    return "\n\n".join(
        f"【{heading}】\n{bodies[heading]}"
        for heading in runner.EXPECTED_HEADINGS_EN
    )


class LevelTwoRunnerTests(unittest.TestCase):
    def test_default_paths_and_generation_defaults(self):
        args = runner.parse_args([])
        self.assertEqual(args.input, runner.DEFAULT_INPUT)
        self.assertEqual(args.max_new_tokens, 3072)
        self.assertFalse(args.enable_thinking)
        self.assertTrue(args.retry_errors)
        self.assertEqual(
            runner.resolve_generation_options(args),
            {"max_new_tokens": 3072, "do_sample": False},
        )
        self.assertEqual(
            args.output,
            runner.PROJECT_ROOT
            / "outputs/Qwen3-8B/b5-text-only.jsonl",
        )

    def test_english_defaults_select_data_prompt_and_separate_output(self):
        args = runner.parse_args(["--language", "en"])
        self.assertEqual(args.language, "en")
        self.assertEqual(args.input, runner.DEFAULT_INPUTS_BY_LANGUAGE["en"])
        self.assertEqual(
            args.prompt_template,
            runner.DEFAULT_PROMPT_TEMPLATES_BY_LANGUAGE["en"],
        )
        self.assertEqual(
            args.output,
            runner.PROJECT_ROOT
            / "outputs/Qwen3-8B/b5-text-only-en.jsonl",
        )

    def test_explicit_input_and_prompt_override_language_defaults(self):
        args = runner.parse_args(
            [
                "custom.jsonl",
                "--language",
                "en",
                "--prompt-template",
                "custom.md",
            ]
        )
        self.assertEqual(args.input, Path("custom.jsonl"))
        self.assertEqual(args.prompt_template, Path("custom.md"))

    def test_thinking_resolves_recommended_sampling_defaults(self):
        args = runner.parse_args(["--enable-thinking"])
        self.assertEqual(
            runner.resolve_generation_options(args),
            {
                "max_new_tokens": 3072,
                "do_sample": True,
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": 20,
            },
        )

    def test_prompt_template_has_one_placeholder_and_all_headings(self):
        template, digest = runner.load_prompt_template(
            runner.DEFAULT_PROMPT_TEMPLATE
        )
        self.assertEqual(template.count("{{patient_information}}"), 1)
        self.assertEqual(len(digest), 64)
        for heading in runner.EXPECTED_HEADINGS:
            self.assertIn(f"【{heading}】", template)
        self.assertIn("合成示例一", template)
        self.assertIn("合成示例二", template)

    def test_english_prompt_template_has_one_placeholder_and_all_headings(self):
        template, digest = runner.load_prompt_template(
            runner.DEFAULT_PROMPT_TEMPLATES_BY_LANGUAGE["en"]
        )
        self.assertEqual(template.count("{{patient_information}}"), 1)
        self.assertEqual(len(digest), 64)
        for heading in runner.EXPECTED_HEADINGS_EN:
            self.assertIn(f"【{heading}】", template)
        self.assertIn("Use concise, professional clinical English.", template)

    def test_messages_only_use_patient_information_from_record(self):
        template = "病例如下：\n{{patient_information}}"
        record = valid_record(
            answer={"自然语言答案": "SECRET_ANSWER"},
            ground_truth={"note": "SECRET_GROUND_TRUTH"},
            review_status="SECRET_REVIEW",
            media={"images": [{"path": "SECRET_MEDIA_PATH"}]},
        )
        messages = runner.build_messages(record, template)
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertIn(record["patient_information"], messages[1]["content"])
        for secret in (
            "SECRET_ANSWER",
            "SECRET_GROUND_TRUTH",
            "SECRET_REVIEW",
            "SECRET_MEDIA_PATH",
            record["qa_id"],
            record["sample_id"],
        ):
            self.assertNotIn(secret, serialized)
        self.assertTrue(all(isinstance(message["content"], str) for message in messages))
        self.assertNotIn('"type": "image"', serialized)
        self.assertNotIn('"type": "video"', serialized)

    def test_english_messages_use_english_system_prompt_and_no_metadata(self):
        record = valid_record(
            patient_information="Patient profile: age 50 years.",
            answer={"value": "SECRET_ANSWER"},
            media={"images": [{"path": "SECRET_MEDIA_PATH"}]},
        )
        messages = runner.build_messages(
            record, "Case:\n{{patient_information}}", "en"
        )
        serialized = json.dumps(messages, ensure_ascii=False)
        self.assertEqual(messages[0]["content"], runner.SYSTEM_PROMPTS["en"])
        self.assertIn(record["patient_information"], messages[1]["content"])
        self.assertNotIn("SECRET_ANSWER", serialized)
        self.assertNotIn("SECRET_MEDIA_PATH", serialized)
        self.assertNotIn(record["qa_id"], serialized)

    def test_record_contract_rejects_non_b5_and_missing_patient_text(self):
        with self.assertRaisesRegex(ValueError, "expected task_name"):
            runner.validate_record(valid_record(task_name="B1_risk_prediction"))
        with self.assertRaisesRegex(ValueError, "patient_information"):
            runner.validate_record(valid_record(patient_information=""))

    def test_loading_rejects_duplicate_qa_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.jsonl"
            line = json.dumps(valid_record(), ensure_ascii=False)
            path.write_text(line + "\n" + line + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate qa_id"):
                runner.load_items(path)

    def test_split_then_offset_then_limit(self):
        items = [
            runner.WorkItem(Path("data.jsonl"), index + 1, record)
            for index, record in enumerate(
                [
                    valid_record(qa_id="q1", sample_id="s1", split="train"),
                    valid_record(qa_id="q2", sample_id="s2", split="test"),
                    valid_record(qa_id="q3", sample_id="s3", split="test"),
                    valid_record(qa_id="q4", sample_id="s4", split="validation"),
                ]
            )
        ]
        args = runner.parse_args(["--split", "test", "--offset", "1", "--limit", "1"])
        selected = runner.select_items(items, args)
        self.assertEqual([item.record["qa_id"] for item in selected], ["q3"])

    def test_frontend_routing_for_requested_models(self):
        self.assertEqual(
            runner.frontend_kind_for_model_type("qwen2"), "qwen2_causal_lm"
        )
        self.assertEqual(
            runner.frontend_kind_for_model_type("qwen3"), "qwen3_causal_lm"
        )
        self.assertEqual(
            runner.frontend_kind_for_model_type("qwen3_5"),
            "qwen3_5_image_text_to_text_text_only",
        )
        self.assertEqual(
            runner.frontend_kind_for_model_type("gpt_oss"),
            "gpt_oss_causal_lm",
        )
        self.assertEqual(
            runner.frontend_kind_for_model_type("gemma3_text"),
            "gemma3_text_causal_lm",
        )
        with self.assertRaisesRegex(ValueError, "unsupported model_type"):
            runner.frontend_kind_for_model_type("llama")

    def test_gpt_oss_uses_supported_eager_attention(self):
        spec = runner.ModelSpec("gpt_oss", ("GptOssForCausalLM",), "gpt_oss_causal_lm")
        self.assertEqual(runner.resolve_attn_implementation("sdpa", spec), "eager")
        qwen = runner.ModelSpec("qwen3", ("Qwen3ForCausalLM",), "qwen3_causal_lm")
        self.assertEqual(runner.resolve_attn_implementation("sdpa", qwen), "sdpa")

    def test_morpheus_prefills_first_required_heading(self):
        qwen2 = runner.ModelSpec("qwen2", ("Qwen2ForCausalLM",), "qwen2_causal_lm")
        self.assertEqual(
            runner.resolve_assistant_prefill(Path("/models/Morpheus-32B"), qwen2),
            "【风险预测】\n",
        )
        self.assertEqual(
            runner.resolve_assistant_prefill(Path("/models/DeepSeek-7B"), qwen2),
            "",
        )
        self.assertEqual(
            runner.resolve_assistant_prefill(
                Path("/models/Morpheus-32B"), qwen2, "en"
            ),
            "【Risk Prediction】\n",
        )

    def test_resolve_model_spec_reads_config_without_loading_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text(
                json.dumps(
                    {
                        "model_type": "qwen3_5",
                        "architectures": ["Qwen3_5ForConditionalGeneration"],
                    }
                ),
                encoding="utf-8",
            )
            spec = runner.resolve_model_spec(root)
            self.assertEqual(spec.model_type, "qwen3_5")
            self.assertIn("text_only", spec.frontend_kind)

    def test_medgemma_and_fleming_model_specs_use_text_causal_frontends(self):
        cases = (
            (
                "gemma3_text",
                "Gemma3ForCausalLM",
                "gemma3_text_causal_lm",
            ),
            ("qwen2", "Qwen2ForCausalLM", "qwen2_causal_lm"),
        )
        for model_type, architecture, expected_frontend in cases:
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": model_type,
                            "architectures": [architecture],
                        }
                    ),
                    encoding="utf-8",
                )
                spec = runner.resolve_model_spec(root)
                self.assertEqual(spec.architectures, (architecture,))
                self.assertEqual(spec.frontend_kind, expected_frontend)

    def test_context_window_and_capacity_are_checked_without_truncation(self):
        config = {"text_config": {"max_position_embeddings": 4096}}
        self.assertEqual(runner.resolve_context_window(config), 4096)
        runner.ensure_context_capacity(1024, 3072, 4096)
        with self.assertRaisesRegex(ValueError, "input was not truncated"):
            runner.ensure_context_capacity(1025, 3072, 4096)
        with self.assertRaisesRegex(ValueError, "could not be determined"):
            runner.ensure_context_capacity(10, 10, None)

    def test_strict_eight_heading_response_is_extracted(self):
        parsed = runner.parse_model_response(valid_response())
        self.assertTrue(parsed.strict_output_format_valid)
        self.assertEqual(parsed.extraction_method, "strict_eight_heading_v1")
        self.assertEqual(list(parsed.prediction_sections), list(runner.EXPECTED_HEADINGS))
        self.assertIn("【风险预测】", parsed.prediction["b1"])
        self.assertIn("【预测依据】", parsed.prediction["b1"])
        self.assertIn("【诊断结论】", parsed.prediction["b2"])
        self.assertIn("【具体动作】", parsed.prediction["b3"])
        self.assertIn("【备用与升级方案】", parsed.prediction["b4"])

    def test_code_fence_is_recovered_but_not_strict(self):
        raw = "```text\n" + valid_response() + "\n```"
        parsed = runner.parse_model_response(raw)
        self.assertIsNotNone(parsed.prediction)
        self.assertFalse(parsed.strict_output_format_valid)
        self.assertIn("code_fence", parsed.extraction_method)
        self.assertEqual(parsed.final_answer_text, raw)

    def test_strict_english_response_is_extracted_and_paired(self):
        parsed = runner.parse_model_response(valid_english_response(), "en")
        self.assertTrue(parsed.strict_output_format_valid)
        self.assertEqual(
            list(parsed.prediction_sections), list(runner.EXPECTED_HEADINGS_EN)
        )
        self.assertIn("【Risk Prediction】", parsed.prediction["b1"])
        self.assertIn("【Prediction Evidence】", parsed.prediction["b1"])
        self.assertIn("【Acute Diagnosis】", parsed.prediction["b2"])
        self.assertIn("【Specific Actions】", parsed.prediction["b3"])
        self.assertIn("【Backup & Escalation Plan】", parsed.prediction["b4"])

    def test_english_markdown_headings_and_tail_are_recovered(self):
        raw = valid_english_response()
        for heading in runner.EXPECTED_HEADINGS_EN:
            raw = raw.replace(f"【{heading}】", f"### {heading}")
        raw += "\n\n### Final Clinical Conclusion\nMust not enter b4."
        parsed = runner.parse_model_response(raw, "en")
        self.assertIsNotNone(parsed.prediction)
        self.assertFalse(parsed.strict_output_format_valid)
        self.assertIn("markdown_headings", parsed.extraction_method)
        self.assertNotIn("Must not enter", parsed.prediction["b4"])

    def test_markdown_headings_and_forbidden_tail_are_recovered(self):
        raw = valid_response()
        for heading in runner.EXPECTED_HEADINGS:
            raw = raw.replace(f"【{heading}】", f"### {heading}")
        raw += "\n\n### 最终临床结论\n不应并入备用方案\n```json\n{}\n```"
        parsed = runner.parse_model_response(raw)
        self.assertIsNotNone(parsed.prediction)
        self.assertFalse(parsed.strict_output_format_valid)
        self.assertIn("markdown_headings", parsed.extraction_method)
        self.assertNotIn("最终临床结论", parsed.prediction["b4"])

    def test_forbidden_tail_after_exact_sections_is_not_added_to_b4(self):
        raw = valid_response() + "\n\n### 临床决策分析\n重复内容"
        parsed = runner.parse_model_response(raw)
        self.assertIsNotNone(parsed.prediction)
        self.assertFalse(parsed.strict_output_format_valid)
        self.assertIn("discarded_tail", parsed.extraction_method)
        self.assertNotIn("重复内容", parsed.prediction["b4"])

    def test_gpt_oss_final_channel_is_extracted(self):
        raw = (
            "<|channel|>analysis<|message|>internal reasoning<|end|>"
            "<|start|>assistant<|channel|>final<|message|>"
            + valid_response()
            + "<|return|>"
        )
        parsed = runner.parse_model_response(raw)
        self.assertIsNotNone(parsed.prediction)
        self.assertNotIn("internal reasoning", parsed.final_answer_text)
        self.assertFalse(parsed.strict_output_format_valid)
        self.assertIn("gpt_oss_final_channel", parsed.extraction_method)

    def test_complete_think_block_is_removed_but_not_strict(self):
        raw = "<think>内部分析</think>\n" + valid_response()
        parsed = runner.parse_model_response(raw)
        self.assertIsNotNone(parsed.prediction)
        self.assertNotIn("内部分析", parsed.final_answer_text)
        self.assertFalse(parsed.strict_output_format_valid)
        self.assertIn("think", parsed.extraction_method)

    def test_prompt_prefilled_think_block_is_recovered(self):
        raw = "内部分析</think>\n" + valid_response()
        parsed = runner.parse_model_response(raw)
        self.assertIsNotNone(parsed.prediction)
        self.assertNotIn("内部分析", parsed.final_answer_text)
        self.assertFalse(parsed.strict_output_format_valid)
        self.assertIn("think", parsed.extraction_method)

    def test_unclosed_think_block_is_rejected(self):
        parsed = runner.parse_model_response("<think>未闭合\n" + valid_response())
        self.assertIsNone(parsed.prediction)
        self.assertIn("unclosed", parsed.prediction_error)

    def test_missing_duplicate_out_of_order_empty_and_unexpected_are_rejected(self):
        response = valid_response()
        cases = {
            "missing": response.replace("【预测依据】", "预测依据", 1),
            "duplicate": response.replace(
                "【预测依据】", "【风险预测】\n重复\n\n【预测依据】", 1
            ),
            "out of order": response.replace("【风险预测】", "【TEMP】", 1)
            .replace("【预测依据】", "【风险预测】", 1)
            .replace("【TEMP】", "【预测依据】", 1),
            "empty": response.replace("【预测依据】\n- MAP呈下降趋势。", "【预测依据】"),
            "unexpected": response.replace(
                "【诊断结论】", "【其他标题】\n内容\n\n【诊断结论】", 1
            ),
        }
        for name, text in cases.items():
            with self.subTest(name=name):
                parsed = runner.parse_model_response(text)
                self.assertIsNone(parsed.prediction)
                self.assertFalse(parsed.strict_output_format_valid)
                self.assertTrue(parsed.format_errors)

    def test_preamble_is_rejected(self):
        parsed = runner.parse_model_response("以下是回答：\n" + valid_response())
        self.assertIsNone(parsed.prediction)
        self.assertIn("before the first heading", parsed.prediction_error)

    def test_run_one_sends_no_media_and_maps_all_sections(self):
        class FakeFrontend:
            model_type = "qwen3"
            frontend_kind = "qwen3_causal_lm"

            def __init__(self):
                self.messages = None

            def generate(self, messages, args):
                self.messages = messages
                return runner.GenerationResult(valid_response(), 100, 200, False)

        record = valid_record(
            answer={"value": "SECRET_ANSWER"},
            media={"images": [{"path": "SECRET_IMAGE"}]},
        )
        item = runner.WorkItem(Path("source.jsonl"), 1, record)
        args = runner.parse_args([])
        frontend = FakeFrontend()
        result = runner.run_one(
            item,
            args,
            frontend,
            "病例：{{patient_information}}",
            "a" * 64,
        )
        serialized = json.dumps(frontend.messages, ensure_ascii=False)
        self.assertNotIn("SECRET_ANSWER", serialized)
        self.assertNotIn("SECRET_IMAGE", serialized)
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["response_format_valid"])
        self.assertFalse(result["media_included"])
        self.assertEqual(result["language"], "zh")
        self.assertEqual(set(result["prediction"]), {"b1", "b2", "b3", "b4"})

    def test_resume_retries_only_latest_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            rows = [
                {"qa_id": "q1", "status": "error"},
                {"qa_id": "q1", "status": "invalid_response"},
                {"qa_id": "q2", "status": "error"},
                {"qa_id": "q3", "status": "ok", "response_format_valid": False},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            statuses = runner.load_previous_statuses(path)
            self.assertEqual(
                statuses,
                {"q1": "invalid_response", "q2": "error", "q3": "ok"},
            )
            self.assertEqual(
                runner.completed_qa_ids(statuses, retry_errors=True),
                {"q1", "q3"},
            )
            self.assertEqual(
                runner.completed_qa_ids(statuses, retry_errors=False),
                {"q1", "q2", "q3"},
            )

    def test_atomic_latest_result_rewrite_deduplicates_in_dataset_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.jsonl"
            rows = [
                {"qa_id": "q2", "status": "error", "attempt": 1},
                {"qa_id": "q1", "status": "invalid_response"},
                {"qa_id": "q2", "status": "ok", "attempt": 2},
                {"qa_id": "foreign", "status": "ok"},
            ]
            path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            latest = runner.load_latest_results(path, {"q1", "q2", "q3"})
            runner.write_latest_results(path, latest, ["q1", "q2", "q3"])
            rewritten = [record for _, record in runner.read_jsonl(path)]
            self.assertEqual([record["qa_id"] for record in rewritten], ["q1", "q2"])
            self.assertEqual(rewritten[1]["attempt"], 2)
            self.assertEqual(len(rewritten), len({row["qa_id"] for row in rewritten}))

    def test_run_one_uses_english_parser_and_records_language(self):
        class EnglishFrontend:
            model_type = "qwen3"
            frontend_kind = "qwen3_causal_lm"

            def generate(self, messages, args):
                return runner.GenerationResult(
                    valid_english_response(), 100, 200, False
                )

        args = runner.parse_args(["--language", "en"])
        result = runner.run_one(
            runner.WorkItem(
                Path("source.jsonl"),
                1,
                valid_record(patient_information="Patient profile: age 50 years."),
            ),
            args,
            EnglishFrontend(),
            "Case: {{patient_information}}",
            "a" * 64,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["language"], "en")
        self.assertEqual(
            result["prompt_version"], runner.PROMPT_VERSIONS["en"]
        )
        self.assertIn("Risk Prediction", result["prediction"]["b1"])

    def test_truncated_generation_is_not_strictly_valid(self):
        class TruncatedFrontend:
            model_type = "qwen3"
            frontend_kind = "qwen3_causal_lm"

            def generate(self, messages, args):
                return runner.GenerationResult(valid_response(), 100, 3072, True)

        item = runner.WorkItem(Path("source.jsonl"), 1, valid_record())
        result = runner.run_one(
            item,
            runner.parse_args([]),
            TruncatedFrontend(),
            "病例：{{patient_information}}",
            "a" * 64,
        )
        self.assertEqual(result["status"], "invalid_response")
        self.assertFalse(result["strict_output_format_valid"])
        self.assertFalse(result["response_format_valid"])
        self.assertIn("max_new_tokens", result["format_errors"][-1])

    def test_complete_recovered_response_is_usable_but_not_strict(self):
        class ChannelFrontend:
            model_type = "gpt_oss"
            frontend_kind = "gpt_oss_causal_lm"

            def generate(self, messages, args):
                text = (
                    "<|channel|>analysis<|message|>reasoning<|end|>"
                    "<|start|>assistant<|channel|>final<|message|>"
                    + valid_response()
                    + "<|return|>"
                )
                return runner.GenerationResult(text, 100, 500, False)

        result = runner.run_one(
            runner.WorkItem(Path("source.jsonl"), 1, valid_record()),
            runner.parse_args([]),
            ChannelFrontend(),
            "病例：{{patient_information}}",
            "a" * 64,
        )
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["response_format_valid"])
        self.assertFalse(result["strict_output_format_valid"])


    def test_run_many_uses_one_text_only_batch_and_records_batch_size(self):
        class BatchFrontend:
            model_type = "qwen3"
            frontend_kind = "qwen3_causal_lm"

            def __init__(self):
                self.messages_batch = None

            def generate_batch(self, messages_batch, args):
                self.messages_batch = messages_batch
                return [
                    runner.GenerationResult(valid_response(), 100 + index, 200, False)
                    for index, _ in enumerate(messages_batch)
                ]

        args = runner.parse_args(["--batch-size", "2"])
        frontend = BatchFrontend()
        items = [
            runner.WorkItem(Path("source.jsonl"), index + 1, valid_record())
            for index in range(2)
        ]
        results = runner.run_many(
            items,
            args,
            frontend,
            "病例：{{patient_information}}",
            "a" * 64,
        )
        self.assertEqual(len(frontend.messages_batch), 2)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(result["status"] == "ok" for result in results))
        self.assertEqual([result["batch_size"] for result in results], [2, 2])
        self.assertTrue(all(result["media_included"] is False for result in results))

    def test_gpt_oss_rejects_incompatible_mxfp4_kernels(self):
        spec = runner.ModelSpec(
            model_type="gpt_oss",
            architectures=("GptOssForCausalLM",),
            frontend_kind="gpt_oss_causal_lm",
        )
        with self.assertRaisesRegex(
            RuntimeError, r"0\.15\.2 <= kernels < 0\.16\.0"
        ):
            runner.ensure_gpt_oss_mxfp4_runtime(
                spec,
                availability_check=lambda: False,
                installed_version="0.16.1",
                minimum_version="0.15.2",
                maximum_version="0.16.0",
            )
        runner.ensure_gpt_oss_mxfp4_runtime(
            spec,
            availability_check=lambda: True,
        )

if __name__ == "__main__":
    unittest.main()
