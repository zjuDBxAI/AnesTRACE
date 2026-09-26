# Level 2: clinical reasoning

Level 2 owns text-only and paired-image inference, prompts, evaluation, and
tests.

```bash
python level2/inference/run_text.py --help
bash level2/inference/run_extended.sh --help
bash level2/inference/run_multimodal.sh --help
python level2/evaluation/evaluate.py --help
```

`inference/run_text.py` is the original tested text-only runner.
`inference/run_extended.py` adds the multi-provider workflow used by the shell
batch launcher. Prompt templates live in `prompts/`.

Run the unit tests with:

```bash
python -m unittest discover -s level2/tests -v
```
