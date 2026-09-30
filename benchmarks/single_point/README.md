# Level 2: clinical reasoning

Level 2 owns text-only and paired-image inference, prompts, evaluation, and
tests.

```bash
python benchmarks/single_point/inference/run_text.py --help
bash benchmarks/single_point/inference/run_extended.sh --help
bash benchmarks/single_point/inference/run_multimodal.sh --help
python benchmarks/single_point/evaluation/evaluate.py --help
```

`inference/run_text.py` is the tested text-only runner.
`inference/run_extended.py` adds the multi-provider workflow used by the shell
batch launcher. Prompt templates live in `prompts/`.

Run the unit tests with:

```bash
python -m unittest discover -s benchmarks/single_point/tests -v
```
