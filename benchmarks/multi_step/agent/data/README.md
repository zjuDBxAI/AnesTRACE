# Level Three data

No patient data are distributed in this repository.

Place an authorized copy of the agent-formatted English dataset at:

\`\`\`text
benchmarks/multi_step/agent/data/Level_three_v3_en_agent.jsonl
\`\`\`

or pass an explicit path to \`anestrace-l3 run --input\`. The required contract
is defined by:

\`\`\`text
benchmarks/multi_step/agent/schemas/level_three_v3_en_agent.schema.json
\`\`\`

The adapter reads only the whitelisted observation fields. Dataset answers,
ground truth, media metadata, and future turns are never inserted into the
model's current-turn prompt.

Do not commit protected clinical data. The repository \`.gitignore\` excludes
the default Level Three JSONL and all runtime tool caches.
