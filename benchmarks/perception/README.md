# Level 1: visual perception

Level 1 contains TEE and waveform inference plus deterministic evaluation.

## Inference

```bash
bash benchmarks/perception/inference/run_tee.sh --help
bash benchmarks/perception/inference/run_waveform.sh --help
```

The Python implementation shared by both tasks is `inference/run.py`.

## Evaluation

Three evaluation interfaces are separate because they consume different
schemas:

- `evaluation/evaluate.py`: public `qa_id` reference-based scorer. Official
  clinician triplets and adjudicated references are not included.
- `evaluation/evaluate_waveform_choices.py`: merged WaveQA A1 exact-choice
  scorer for records keyed by `question_id`.
- `evaluation/evaluate_waveform_structured.py`: merged structured waveform
  scorer for WaveQA `BenchmarkSample`/`PredictionRecord` JSONL.

Exact-choice example:

```bash
python benchmarks/perception/evaluation/evaluate_waveform_choices.py \
  --benchmark benchmark.jsonl \
  --predictions predictions.jsonl \
  --metrics-output metrics.json \
  --details-output details.jsonl
```

Structured waveform example:

```bash
python benchmarks/perception/evaluation/evaluate_waveform_structured.py \
  --manifest manifest.jsonl \
  --predictions predictions.jsonl \
  --output metrics.json
```

Synthetic interface examples are in `../../examples/synthetic/`. They are not
official benchmark items or official scores.
