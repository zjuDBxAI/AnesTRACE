# Level 3: longitudinal decisions and agent behavior

The multi-step benchmark contains history-ablation inference and the independent
LangGraph agent package.

```bash
bash benchmarks/multi_step/inference/run_ablation.sh --help
bash benchmarks/multi_step/inference/run_agent.sh --help
python benchmarks/multi_step/evaluation/evaluate.py --help
```

Install and test the agent package separately:

```bash
pip install -e benchmarks/multi_step/agent
cd benchmarks/multi_step/agent
PYTHONPATH=src python -m pytest
```

See [agent/README.md](agent/README.md) for the data contract, configuration,
knowledge tools, and runtime details.
