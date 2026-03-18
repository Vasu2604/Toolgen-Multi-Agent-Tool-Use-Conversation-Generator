# toolgen — Multi-Agent Tool-Use Conversation Generator

A system that generates synthetic multi-turn conversations where an AI assistant uses multiple tools (APIs) to complete user goals. The output is a dataset suitable for training/evaluating tool-use AI agents.

---

## Quick Start

### 1. Prerequisites

Install [Ollama](https://ollama.com) (local LLM server):
```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the required models:
```bash
ollama pull llama3.2
ollama pull nomic-embed-text   # for mem0 embeddings
```

Start Ollama:
```bash
ollama serve
```

### 2. Install Python dependencies

```bash
cd toolgen
pip install -r requirements.txt
```

### 3. Run end-to-end

```bash
# Step 1: Build the tool registry and graph
python -m toolgen.cli build

# Step 2: Generate 50 conversations
python -m toolgen.cli generate --count 50 --output output/dataset.jsonl

# Step 3: Validate the dataset
python -m toolgen.cli validate output/dataset.jsonl

# Step 4: Compute metrics
python -m toolgen.cli metrics output/dataset.jsonl
```

---

## Full CLI Reference

### `build` — Build registry and graph
```bash
python -m toolgen.cli build \
  --data-dir data/sample_tools \   # directory with ToolBench JSON files
  --output-dir artifacts \          # where to save built artifacts
  --model llama3.2                  # Ollama model to use
```

### `generate` — Generate conversations
```bash
python -m toolgen.cli generate \
  --count 50 \                      # number of conversations
  --seed 42 \                       # random seed for reproducibility
  --output output/dataset.jsonl \   # output file
  --model llama3.2 \                # Ollama model

# Disable corpus memory (Run A for diversity experiment)
python -m toolgen.cli generate --count 20 --no-corpus-memory --output output/run_a.jsonl --seed 42
# Enable corpus memory (Run B)
python -m toolgen.cli generate --count 20 --output output/run_b.jsonl --seed 42
```

### `validate` — Validate a dataset
```bash
python -m toolgen.cli validate output/dataset.jsonl
```

### `metrics` — Compute metrics
```bash
# Single dataset
python -m toolgen.cli metrics output/dataset.jsonl

# Compare Run A vs Run B (diversity experiment)
python -m toolgen.cli metrics output/run_a.jsonl --compare output/run_b.jsonl
```

---

## Using Your Own ToolBench Data

Download ToolBench from [https://github.com/OpenBMB/ToolBench](https://github.com/OpenBMB/ToolBench) and put the JSON files in `data/toolbench/`:

```bash
python -m toolgen.cli build --data-dir data/toolbench
```

The loader handles ToolBench's inconsistent field names automatically.

---

## Running Tests

```bash
# Run all tests (no Ollama needed — tests use a mock LLM)
pytest tests/ -v

# Run specific test files
pytest tests/test_registry.py -v
pytest tests/test_memory.py -v
pytest tests/test_e2e.py -v        # includes the 50-sample E2E test
```

---

## Output Format

Each line in the JSONL output is one conversation:

```json
{
  "id": "abc12345",
  "messages": [
    {"role": "user", "content": "I want to plan a trip to Paris"},
    {"role": "assistant", "content": "[Calling weather_api::get_current_weather]",
     "tool_call": {"endpoint": "weather_api::get_current_weather", "arguments": {...}}},
    {"role": "tool", "content": "{\"temperature\": 22, ...}", "tool_output": {...}},
    {"role": "assistant", "content": "The weather in Paris looks great! ..."}
  ],
  "tool_calls": [
    {
      "endpoint": "weather_api::get_current_weather",
      "tool_name": "weather_api",
      "arguments": {"city": "Paris", "country_code": "FR"},
      "output": {"temperature": 22, "condition": "sunny"},
      "step": 0
    }
  ],
  "metadata": {
    "seed": 42,
    "tool_ids_used": ["weather_api", "flight_search"],
    "num_turns": 6,
    "num_clarification_questions": 1,
    "memory_grounding_rate": 0.75,
    "corpus_memory_enabled": true,
    "pattern_type": "sequential"
  }
}
```

---

## Project Structure

```
toolgen/
├── toolgen/
│   ├── llm.py          # Ollama LLM wrapper
│   ├── registry.py     # Tool Registry (loads ToolBench JSON)
│   ├── graph.py        # Tool Graph + Chain Sampler (NetworkX)
│   ├── execution.py    # Offline tool executor (mock responses)
│   ├── memory.py       # MemoryStore (mem0 backed, session + corpus)
│   ├── agents.py       # 5 agents: Sampler, Planner, UserProxy, Assistant, Validator
│   ├── generator.py    # Conversation orchestrator
│   ├── metrics.py      # Evaluation metrics + diversity comparison
│   └── cli.py          # Click CLI (build/generate/validate/metrics)
├── tests/
│   ├── test_registry.py   # Registry unit tests
│   ├── test_memory.py     # MemoryStore unit tests (scope isolation)
│   └── test_e2e.py        # End-to-end test (50 samples)
├── data/
│   └── sample_tools/      # 8 sample tools (weather, flights, hotels, etc.)
├── output/                # Generated datasets go here
├── requirements.txt
├── README.md
└── DESIGN.md
```
