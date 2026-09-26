# Level 3: longitudinal decisions and agent behavior

Level 3 contains both history-ablation inference and the independent
LangGraph agent package.

```bash
bash level3/inference/run_ablation.sh --help
bash level3/inference/run_agent.sh --help
python level3/evaluation/evaluate.py --help
```

Install and test the agent package separately:

```bash
pip install -e level3/agent
cd level3/agent
PYTHONPATH=src python -m pytest
```

See [agent/README.md](agent/README.md) for the data contract, configuration,
knowledge tools, and runtime details.
