# AnesTRACE

The code is organized by benchmark level. Each level owns its inference,
evaluation, documentation, and level-specific tests or runtime package.

```text
AnesTRACE-main/
├── level1/
│   ├── inference/          # TEE and waveform inference
│   └── evaluation/         # L1 metrics, including merged WaveQA scorers
├── level2/
│   ├── inference/          # text-only and paired-image inference
│   ├── evaluation/         # AnesTRACE-Eval judge
│   ├── prompts/
│   └── tests/
├── level3/
│   ├── inference/          # history ablations and agent launcher
│   ├── evaluation/         # turn/trajectory evaluation
│   └── agent/              # installable LangGraph agent package
├── data/                   # protected datasets (not committed)
└── scripts/                # backward-compatible wrappers only
```

Start with the README for the required level:

- [Level 1](level1/README.md)
- [Level 2](level2/README.md)
- [Level 3](level3/README.md)

Old commands under `scripts/inference/`, `scripts/eval/`, `scripts/run_batch.sh`,
and root `run_inference.py` remain available as compatibility wrappers. New
code should import or invoke the entrypoints inside `level1`, `level2`, or
`level3` directly.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e level3/agent
```

## Tests

```bash
python -m unittest discover -s level1/tests -v
python -m unittest discover -s level2/tests -v
cd level3/agent && PYTHONPATH=src python -m pytest
```

Protected clinical data, generated outputs, credentials, and model weights
must not be committed. This software is for research evaluation only.

## Inherited optional modules

The supplied snapshot already referenced private/internal modules named
`inference_adapters` and `eval.*`, but those modules were not present in the
target folder or the WaveQA source tree. The merged waveform evaluators are
self-contained; the inherited extended multimodal/structured entrypoints still
require those modules to be restored before they can run.
