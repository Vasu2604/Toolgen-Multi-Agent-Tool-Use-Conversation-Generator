# toolgen
**An offline “conversation factory” that generates realistic tool-using chats for training and testing AI assistants.**

## What This Project Does
**Problem (plain English):** To train an AI assistant to use tools (like “search flights” or “check weather”), you need lots of example conversations. Writing thousands of good examples by hand can take months.

**Solution (plain English):** `toolgen` generates those conversations automatically. It reads tool schemas (from ToolBench-style JSON), builds a “map” of how tools connect, then runs five small agents that role‑play a conversation and produce a JSONL dataset.

Think of it like:
- **Tool schemas** = instruction manuals for each tool
- **Tool graph** = a map connecting those manuals
- **Offline executor** = a flight simulator (feels real, but no real APIs)
- **Memory** = post‑it notes (session) + a shared notebook (corpus)

## A Real Example (first scroll)
This is what one generated conversation looks like:

```text
[USER]
Hi, I need some help planning my upcoming trip.

[ASSISTANT]
Which city are you traveling to, and what dates?

[USER]
Paris, June 1st to June 8th.

[ASSISTANT]
[Calling flight_search::search_flights]

[TOOL]
{"flights":[{"flight_id":"F_001","airline":"Air France","departure_time":"2025-06-01T10:30:00","price":650.0}, ...]}

[ASSISTANT]
I found 2 flights. Air France AF007 leaves at 10:30am non‑stop for $650.

[ASSISTANT]
[Calling hotel_booking::search_hotels]

[TOOL]
{"hotels":[{"hotel_id":"H_001","name":"Grand Hotel Paris","price_per_night":280.0,"rating":4.7}, ...]}

[ASSISTANT]
Grand Hotel Paris is available for $280/night with a 4.7 rating.

[ASSISTANT]
[Calling restaurant_finder::check_reservation]

[TOOL]
{"restaurant_id":"R_321","available":true,"available_times":["19:00","19:30","20:00"]}

[ASSISTANT]
Here’s your plan: Air France at 10:30am, Grand Hotel Paris at $280/night, and dinner available at 7:30pm.
```

## What It Produces (JSONL format)
**Plain English first:** JSONL means “JSON Lines”. It’s like a notebook where **each line is one conversation**.

Each line contains:
- **messages**: the chat turns (user/assistant/tool)
- **tool_calls**: the tool-use trace (endpoint + arguments + output)
- **metadata**: numbers that help you measure quality and reproducibility

### Full record schema (including all required metadata fields)

```json
{
  "id": "abc12345",
  "messages": [
    {"role": "user", "content": "Hi, I need help planning my trip."},
    {"role": "assistant", "content": "Which city and dates?"},
    {"role": "user", "content": "Paris, June 1 to June 8."},
    {
      "role": "assistant",
      "content": "[Calling flight_search::search_flights]",
      "tool_call": {
        "endpoint": "flight_search::search_flights",
        "arguments": {"origin": "JFK", "destination": "CDG", "departure_date": "2025-06-01"}
      }
    },
    {"role": "tool", "content": "{\"flights\": [...]}",
     "tool_output": {"flights": [{"flight_id": "F_001", "price": 650.0}]}
    },
    {"role": "assistant", "content": "I found 2 flights. Air France AF007 is $650 non-stop."}
  ],
  "tool_calls": [
    {
      "endpoint": "flight_search::search_flights",
      "tool_name": "flight_search",
      "arguments": {"origin": "JFK", "destination": "CDG", "departure_date": "2025-06-01"},
      "output": {"flights": [{"flight_id": "F_001", "price": 650.0}]},
      "step": 0
    }
  ],
  "metadata": {
    "seed": 42,
    "tool_ids_used": ["flight_search", "hotel_booking", "restaurant_finder"],
    "num_turns": 9,
    "num_clarification_questions": 1,
    "memory_grounding_rate": 1.0,
    "corpus_memory_enabled": true,
    "pattern_type": "sequential"
  }
}
```

## Quick Start (exactly 3 steps)
### Step 1 — Install Ollama (local model runner)
**Plain English:** Ollama runs the AI model on your laptop, like a small “AI engine” you control.

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh

ollama pull llama3.2
ollama pull nomic-embed-text

ollama serve
```

### Step 2 — Install Python packages
**Plain English:** This installs the libraries the project needs.

```bash
pip install -r requirements.txt
```

### Step 3 — Run the 4 commands
**Plain English:** Build the tool map → generate conversations → validate → measure metrics.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/generate_dataset.py
python3 -m toolgen.cli validate output/dataset.jsonl
python3 -m toolgen.cli metrics output/dataset.jsonl
```

Expected highlights:
- `Ran 61 tests ... OK`
- `✅ Passed: 50/50`
- `pct_with_3plus_tool_calls: 100.0`
- `pct_with_clarification: 54.0`
- `avg_memory_grounding_rate: 1.0`
- `tool_chain_jaccard_dissimilarity: 0.598`

## CLI Reference (all 4 commands)
### `build` — load tools and build the Tool Graph
**Plain English:** This turns messy ToolBench-style JSON into a clean “tool registry” and a graph “map”.

```bash
python3 -m toolgen.cli build \
  --data-dir data/sample_tools \
  --output-dir artifacts \
  --model llama3.2
```

### `generate` — generate conversations
**Plain English:** This runs the 5-agent system to create multi-step tool traces and saves JSONL.

```bash
python3 -m toolgen.cli generate \
  --count 50 \
  --seed 42 \
  --output output/dataset.jsonl \
  --data-dir data/sample_tools \
  --model llama3.2 \
  --no-corpus-memory=false \
  --use-langgraph=false
```

#### Diversity experiment (Run A vs Run B)
**Plain English:** Run A is the baseline (no “shared notebook”). Run B turns the notebook on.

```bash
# Run A — no corpus memory
python3 -m toolgen.cli generate \
  --count 20 --seed 42 \
  --no-corpus-memory \
  --output output/run_a.jsonl

# Run B — corpus memory on
python3 -m toolgen.cli generate \
  --count 20 --seed 42 \
  --output output/run_b.jsonl

# Compare
python3 -m toolgen.cli metrics output/run_a.jsonl --compare output/run_b.jsonl
```

#### LangGraph mode
**Plain English:** LangGraph makes the agent flow a real graph (like a flowchart) instead of nested if‑statements.

```bash
python3 -m toolgen.cli generate --count 10 --use-langgraph
```

### `validate` — check dataset quality
**Plain English:** This is a checklist: “does each conversation have enough tool calls and the right fields?”

```bash
python3 -m toolgen.cli validate output/dataset.jsonl
```

### `metrics` — measure coverage, memory grounding, and diversity
**Plain English:** This prints the scorecard: tool-call coverage + clarification rate + diversity metrics.

```bash
# Single dataset
python3 -m toolgen.cli metrics output/dataset.jsonl

# Compare Run A vs Run B
python3 -m toolgen.cli metrics output/run_a.jsonl --compare output/run_b.jsonl
```

## Testing (fast, offline, no Ollama needed)
**Plain English:** The tests do not call real models. They use a `MockLLM`, so they’re fast and reliable.

- **Runs in ~0.1 seconds**
- **No Ollama required**

```bash
python3 -m unittest discover -s tests -v
```

## Project Structure (annotated)
```text
toolgen/
├── toolgen/                          # main Python package
│   ├── cli.py                        # build / generate / validate / metrics commands
│   ├── registry.py                   # “filing cabinet” for ToolBench JSON → Tool/Endpoint/Parameter objects
│   ├── graph.py                      # “map” (Tool Graph) + chain sampler (sequential/parallel/mixed)
│   ├── execution.py                  # “flight simulator”: validate args + generate mock tool outputs + SessionState
│   ├── memory.py                     # MemoryStore (mem0-backed) + safe fallback store
│   ├── agents.py                     # 5 agents (Sampler/Planner/UserProxy/Assistant/Validator)
│   ├── generator.py                  # runs the full conversation loop (plain Python)
│   ├── langgraph_generator.py        # same pipeline as a LangGraph state machine
│   ├── llm.py                        # talks to Ollama with streaming + timeouts
│   ├── metrics.py                    # coverage + diversity metrics (Jaccard, bigrams, entropy)
│   └── __main__.py                   # `python -m toolgen` entry
├── scripts/
│   ├── generate_dataset.py           # makes `output/dataset.jsonl` using MockLLM (deterministic)
│   └── pick_subset.py                # pick a small, diverse subset from real ToolBench
├── data/
│   └── sample_tools/                 # 8 sample tools: weather, flights, hotels, currency, maps, restaurants, news, calendar
├── tests/                            # 61 tests, all offline
│   ├── test_registry.py
│   ├── test_memory.py
│   └── test_e2e.py
├── output/                           # generated JSONL datasets (created at runtime)
├── requirements.txt
├── README.md
└── DESIGN.md
```

## Using Real ToolBench Data (not just the 8 sample tools)
**Plain English:** ToolBench is huge. You usually want a smaller slice first, so runs are fast.

1) Put real ToolBench tool JSON files somewhere on disk.  
2) Use `pick_subset.py` to select a diverse subset.  
3) Build using that subset.

```bash
python3 scripts/pick_subset.py \
  --source /path/to/ToolBench/tools \
  --output data/toolbench_subset \
  --tools-per-category 3

python3 -m toolgen.cli build --data-dir data/toolbench_subset
```

## TL;DR (one line)
```bash
python3 -m unittest discover -s tests -v && \
python3 scripts/generate_dataset.py && \
python3 -m toolgen.cli validate output/dataset.jsonl && \
python3 -m toolgen.cli metrics output/dataset.jsonl
```

Expected: **61 tests OK, ✅ Passed 50/50, 100% with ≥3 tool calls, 54% clarifications, memory grounding 1.0.**
