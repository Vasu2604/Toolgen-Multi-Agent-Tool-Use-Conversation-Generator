# Toolgen — Multi-Agent Tool-Use Conversation Generator
**An offline “conversation factory” that generates realistic tool-using chats for training and testing AI assistants.**

## Start Here
The full documentation lives inside the `toolgen/` folder:

- **Main README:** `toolgen/README.md`
- **Architecture + design doc:** `toolgen/DESIGN.md`

Why this repo has a nested folder: the real Python package is inside `toolgen/`, and that folder contains the CLI, tests, scripts, and data used for the assignment.

## Quick jump (GitHub UI)
1. Click the `toolgen/` folder  
2. Open `README.md`

## One-liner to verify everything works
Run this from the `toolgen/` directory:

```bash
cd toolgen && \
python3 -m unittest discover -s tests -v && \
python3 scripts/generate_dataset.py && \
python3 -m toolgen.cli validate output/dataset.jsonl && \
python3 -m toolgen.cli metrics output/dataset.jsonl
```

Expected highlights:
- `Ran 61 tests ... OK`
- `✅ Passed: 50/50`
- `pct_with_3plus_tool_calls: 100.0`
- `pct_with_clarification: 54.0`
- `avg_memory_grounding_rate: 1.0`

