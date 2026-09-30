# AnesTRACE Level Three Agent

This subproject is the installable inference runner for AnesTRACE multi-step
decision episodes. It is separate from the single-point runner.

Level Three uses one LangGraph agent per decision point. Each Episode is
processed sequentially, while independent Episodes may run concurrently. The
agent starts from a partial observation and may autonomously request case
context or external knowledge before returning a structured intervention.

This is a research benchmark runner, not a clinical decision system.

## Level Two versus Level Three

| | Level Two | Level Three Agent |
| --- | --- | --- |
| Entry point | `benchmarks/single_point/inference/run_text.py` | `anestrace-l3` |
| Task | Single B5 case | Multi-turn Atomic Episode |
| Model frontend | Transformers | LangGraph + OpenAI-compatible chat API |
| Tools | None | Six context tools + four knowledge tools |
| Memory | None | Episode-scoped diagnosis/action summaries |
| Languages | Chinese and English | English only in this release |
| Output | One row per B5 item | Turn traces plus Episode summaries |

The L3 package lives entirely under `benchmarks/multi_step/agent/`; it does not import
the root L2 runner.

## Agent graph

```text
START
  -> prepare_turn
  -> decision_agent
       -> tool_executor -> decision_agent
       -> validate_answer -> repair_answer (at most once)
  -> update_episode_memory
  -> persist_turn_result
  -> END
```

The initial message contains only the current procedure, decision-point
ordinal, elapsed time, and current vital-sign trends. Hidden case information
becomes visible only if the model calls one of these no-argument, state-bound
tools:

```text
get_patient_profile
get_procedure_anesthesia_context
get_anesthesia_medication_state
get_previous_intervention
get_visible_test_results
get_episode_memory
```

The tools cannot accept an Episode ID, decision point, path, query, or time
range. They therefore cannot request another patient or a future turn.

The model may also generate arguments for:

```text
get_anesthesia_drug_information(drug_name)
search_guidelines(query)
search_miller_anesthesia(query)
search_pubmed(query)
```

See [knowledge_sources.md](docs/knowledge_sources.md) before enabling knowledge
retrieval. Public API tool code is included; guideline and textbook content is
not.

## Installation

```bash
cd benchmarks/multi_step/agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[test]'
```

Install vLLM separately in the model-serving environment. The inference client
and server may use different virtual environments.

## Data

Place an authorized agent-format JSONL at:

```text
data/Level_three_v3_en_agent.jsonl
```

The data are not included. See [data/README.md](data/README.md) and the
[JSON Schema](schemas/level_three_v3_en_agent.schema.json).

Validate without calling a model:

```bash
anestrace-l3 validate --input data/Level_three_v3_en_agent.jsonl
```

## Configure knowledge sources

The public default starts with network drug/PubMed tools enabled and local
guideline/textbook paths unset:

```json
{
  "tools": {
    "top_k": 3,
    "network_enabled": true,
    "cache_dir": "data/tool_cache",
    "guideline_corpus_path": null,
    "miller_corpus_path": null
  }
}
```

With null local paths, guideline or Miller calls return an explicit
`unavailable` result. To use those tools, copy `configs/offline.example.json`
and set paths to your legally obtained, provenance-preserving JSONL corpora.

## Start Qwen3.5

The provided profile matches the Level Two non-thinking generation settings:

```text
temperature=0, top_p=1, top_k=0, seed=42,
max_tokens=3072, enable_thinking=false
```

Start an OpenAI-compatible vLLM endpoint:

```bash
export ANESTRACE_MODEL_PATH=/path/to/Qwen3.5-27B
export CUDA_VISIBLE_DEVICES=0,1
./scripts/start_vllm_qwen35.sh
```

The default served model name is `Qwen3.5-27B` at
`http://127.0.0.1:8002/v1`.

## Run

First run one complete Episode:

```bash
ANESTRACE_L3_OUTPUT=outputs/smoke \
  ./scripts/run_agent.sh --limit 1 --workers 1
```

Then use a new output directory for the full dataset:

```bash
ANESTRACE_L3_OUTPUT=outputs/qwen35-27b-full \
  ./scripts/run_agent.sh --workers 4
```

Resume an interrupted run without repeating completed turns:

```bash
ANESTRACE_L3_OUTPUT=outputs/qwen35-27b-full \
  ./scripts/run_agent.sh --workers 4 --resume
```

For a remote compatible service, pass `--api-base`, `--model`, and
`--api-key-env`. Only the environment-variable name, never its value, is
saved.

## Tool budgets

Defaults per turn are six tool rounds, six context calls, four knowledge calls,
and eight total calls. Each context section may be returned once. Tool failures
remain visible to the model and do not terminate the Episode.

## Output

The validated model prediction uses `Specific_intervention` as the canonical key for the recommended intervention. The former `expert_recommendation` key is not accepted by the current schema and is not emitted by new runs.

Each run directory contains:

- `predictions.jsonl`: direct input, accessed sections, prediction, and memory;
- `tool_calls.jsonl`: tool class, normalized arguments, status, time, and hash;
- `tool_results/`: complete tool returns;
- `episodes.jsonl`: ordered Episode-level summaries;
- `run_manifest.json`: non-secret configuration, counts, hashes, and usage;
- `checkpoints.sqlite`: thread-scoped LangGraph persistence.

Dataset answers, ground truth, media, and future turns are not sent to the
model. Review tool-result files before publication because public API metadata
or licensed local corpus excerpts may be present.

## Tests

```bash
python -m pytest
bash -n scripts/start_vllm_qwen35.sh scripts/run_agent.sh
```

No network access, model weights, or clinical dataset is required for the unit
tests.
