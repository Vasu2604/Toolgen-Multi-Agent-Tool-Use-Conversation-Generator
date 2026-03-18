# 🤖 toolgen — Multi-Agent Tool-Use Conversation Generator

> *Automatically generate thousands of realistic AI training conversations where an assistant uses multiple APIs to help users — no real API keys, no manual labelling, fully offline.*

---

## 🎯 What This Project Does

Imagine you want to train an AI assistant to book flights, check the weather, find restaurants, and convert currencies — all in the same conversation. To train it, you need thousands of example conversations showing how a good assistant behaves. Writing those by hand would take months.

**toolgen generates them automatically.**

It reads API documentation, builds a map of how tools connect, then runs five AI agents that role-play a realistic conversation — one agent plays the user, one plays the assistant, one picks the tools, one plans the scenario, and one checks the quality. The result is a dataset of rich, multi-step conversations ready for training or evaluation.

---

## 📊 What It Produces

Every generated conversation looks like this:

```
User:      "Hi, I need help planning my trip to Paris."
Assistant: "Which airport are you flying from?"          ← clarification question
User:      "JFK, New York."
Assistant: [Calls flight_search → returns Air France AF007, $650, non-stop]
Assistant: "I found 2 flights. Air France AF007 departs at 10:30am non-stop for $650."
User:      "Great, can you find a hotel too?"
Assistant: [Calls hotel_booking → returns Grand Hotel Paris, $280/night, rated 4.7]
Assistant: "Grand Hotel Paris is available at $280/night with a 4.7 rating."
Assistant: [Calls restaurant_finder → Le Jules Verne available at 19:30]
Assistant: "Here's your full Paris plan: Air France AF007 at 10:30am for $650,
            Grand Hotel Paris at $280/night, and a table at Le Jules Verne at 7:30pm."
```

Each conversation is saved as a JSONL record with full metadata:

```json
{
  "id": "abc12345",
  "messages": [
    {"role": "user",      "content": "Hi, I need help planning my trip to Paris."},
    {"role": "assistant", "content": "Which airport are you flying from?"},
    {"role": "user",      "content": "JFK, New York."},
    {"role": "assistant", "content": "[Calling flight_search::search_flights]",
     "tool_call": {"endpoint": "flight_search::search_flights", "arguments": {"origin": "JFK"}}},
    {"role": "tool",      "content": "{\"flight_id\": \"F001\", \"airline\": \"Air France\"...}"},
    {"role": "assistant", "content": "I found 2 flights. Air France AF007 departs at 10:30am..."}
  ],
  "tool_calls": [
    {
      "endpoint":  "flight_search::search_flights",
      "tool_name": "flight_search",
      "arguments": {"origin": "JFK", "destination": "CDG", "departure_date": "2025-06-01"},
      "output":    {"flight_id": "F001", "airline": "Air France", "price": 650.0},
      "step": 0
    }
  ],
  "metadata": {
    "seed":                        42,
    "tool_ids_used":               ["flight_search", "hotel_booking", "restaurant_finder"],
    "num_turns":                   9,
    "num_clarification_questions": 1,
    "memory_grounding_rate":       1.0,
    "corpus_memory_enabled":       true,
    "pattern_type":                "sequential"
  }
}
```

---

## ⚡ Quick Start (5 Minutes)

### Step 1 — Install Ollama (local AI model runner)

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

Pull the model and start the server:

```bash
ollama pull llama3.2          # the LLM that powers all agents
ollama serve                  # keep this running in a separate terminal
```

### Step 2 — Install Python dependencies

```bash
cd toolgen
pip install -r requirements.txt
```

### Step 3 — Run the full pipeline

```bash
# Build the tool registry and knowledge graph
python -m toolgen.cli build

# Generate 50 conversations
python -m toolgen.cli generate --count 50 --output output/dataset.jsonl

# Validate the output
python -m toolgen.cli validate output/dataset.jsonl

# See quality metrics
python -m toolgen.cli metrics output/dataset.jsonl
```

That's it. You'll have `output/dataset.jsonl` with 50 realistic conversations.

---

## 🛠️ All CLI Commands

### `build` — Load tools and build the knowledge graph

```bash
python -m toolgen.cli build \
  --data-dir   data/sample_tools \  # folder containing ToolBench JSON files
  --output-dir artifacts            # where to save the built registry and graph
```

Produces `artifacts/registry.json` (all tools indexed) and `artifacts/graph.json` (the tool network).

---

### `generate` — Generate conversations

```bash
python -m toolgen.cli generate \
  --count  50 \                       # how many conversations to generate
  --seed   42 \                       # random seed (same seed = same output every time)
  --output output/dataset.jsonl \     # where to save the dataset
  --model  llama3.2                   # which Ollama model to use
```

**Diversity experiment flags** (required by the assessment):

```bash
# Run A — corpus memory OFF (baseline)
python -m toolgen.cli generate \
  --count 20 --seed 42 \
  --no-corpus-memory \
  --output output/run_a.jsonl

# Run B — corpus memory ON (should be more diverse)
python -m toolgen.cli generate \
  --count 20 --seed 42 \
  --output output/run_b.jsonl
```

**LangGraph mode** (alternative production-grade pipeline):

```bash
python -m toolgen.cli generate --count 10 --use-langgraph
```

---

### `validate` — Check conversation quality

```bash
python -m toolgen.cli validate output/dataset.jsonl
```

Checks every record for:
- At least 3 tool calls per conversation
- At least 2 distinct tools used
- All required metadata fields present
- All tool calls have endpoint, arguments, and output

---

### `metrics` — Compute evaluation metrics

```bash
# Single dataset
python -m toolgen.cli metrics output/dataset.jsonl

# Compare Run A vs Run B (diversity experiment)
python -m toolgen.cli metrics output/run_a.jsonl --compare output/run_b.jsonl
```

Output example:
```json
{
  "coverage": {
    "pct_with_3plus_tool_calls":    100.0,
    "pct_with_2plus_distinct_tools": 100.0,
    "pct_with_clarification":        54.0,
    "avg_tool_calls":                3.96,
    "avg_turns":                     9.08
  },
  "memory": {
    "avg_memory_grounding_rate": 1.0
  },
  "diversity": {
    "tool_chain_jaccard_dissimilarity": 0.598,
    "distinct_bigrams":                 0.061,
    "pattern_entropy":                  1.419
  }
}
```

---

## 🧪 Running Tests

All tests use a MockLLM — **no Ollama needed**, runs in under 1 second.

```bash
# Run all 61 tests
python -m unittest discover -s tests -v

# Run specific test files
python -m unittest tests.test_registry -v   # tool loading + parsing (22 tests)
python -m unittest tests.test_memory   -v   # scope isolation + add/search (12 tests)
python -m unittest tests.test_e2e      -v   # full pipeline, 50 conversations (27 tests)
```

Expected output:
```
Ran 61 tests in 0.1s
OK
```

---

## 📁 Project Structure

```
toolgen/
│
├── toolgen/                      ← Python package (importable source code)
│   ├── llm.py                    ← Talks to Ollama with streaming (no silent hangs)
│   ├── registry.py               ← Loads ToolBench JSON → clean Python objects
│   ├── graph.py                  ← Builds NetworkX knowledge graph + chain sampler
│   ├── execution.py              ← Validates args, generates realistic mock outputs
│   ├── memory.py                 ← MemoryStore: session + corpus scopes via mem0
│   ├── agents.py                 ← 5 agents: Sampler, Planner, UserProxy, Assistant, Validator
│   ├── generator.py              ← Orchestrates all agents into one conversation
│   ├── langgraph_generator.py    ← Same pipeline rebuilt as a LangGraph state machine
│   ├── metrics.py                ← Coverage + diversity metrics (Jaccard, entropy, bigrams)
│   └── cli.py                    ← build / generate / validate / metrics CLI
│
├── tests/
│   ├── test_registry.py          ← 22 unit tests: loading, parsing, malformed data
│   ├── test_memory.py            ← 12 unit tests: add/search, scope isolation
│   └── test_e2e.py               ← 27 tests including 50-conversation E2E test
│
├── data/
│   └── sample_tools/
│       └── tools.json            ← 8 hand-crafted tools (weather, flights, hotels…)
│
├── scripts/
│   └── generate_dataset.py       ← Standalone script: generates output/dataset.jsonl
│
├── output/                       ← Generated datasets (created at runtime)
│   ├── dataset.jsonl             ← Main 50-conversation dataset
│   ├── run_a.jsonl               ← Diversity experiment: no corpus memory
│   └── run_b.jsonl               ← Diversity experiment: corpus memory enabled
│
├── requirements.txt
├── README.md                     ← This file
└── DESIGN.md                     ← Architecture decisions and analysis
```

---

## 🔌 Using Real ToolBench Data

The sample tools folder has 8 tools. The real ToolBench has 16,000+. To use real data:

```bash
# Clone ToolBench (about 2GB)
git clone --depth 1 https://github.com/OpenBMB/ToolBench.git

# Pick a diverse 5–10% subset (about 150 tools across 49 categories)
python scripts/pick_subset.py \
  --source ToolBench/data/toolenv/tools \
  --output data/toolbench_subset \
  --tools-per-category 3

# Build with the real data
python -m toolgen.cli build --data-dir data/toolbench_subset
```

The registry handles all ToolBench format variants automatically.

---

## 📋 Requirements

| Requirement | Version |
|---|---|
| Python | 3.10+ |
| Ollama | Latest |
| llama3.2 model | via `ollama pull llama3.2` |
| See `requirements.txt` | for all Python packages |

---

## 🏃 TL;DR — One-liner to verify everything works

```bash
python -m unittest discover -s tests -v && \
python scripts/generate_dataset.py && \
python -m toolgen.cli validate output/dataset.jsonl && \
python -m toolgen.cli metrics output/dataset.jsonl
```

Expected: 61 tests pass, 50 valid conversations, `pct_with_clarification: 54%`, `memory_grounding_rate: 1.0`.
