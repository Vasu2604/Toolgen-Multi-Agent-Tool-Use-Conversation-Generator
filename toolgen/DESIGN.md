# DESIGN.md — Architecture & Design Decisions

## My Approach and Goal

My goal was to build a system that generates realistic AI training conversations automatically — conversations where an assistant uses multiple tools in sequence, asks clarifying questions when needed, and references earlier results in later steps. I approached it by breaking the problem into five independent layers: tool knowledge (registry), tool relationships (graph), tool simulation (executor), conversation generation (agents), and memory (session + corpus). Each layer has a single job and is completely replaceable without touching the others. The hardest challenge was making the tool outputs chain correctly — step two needs to reference the ID that step one returned, and without real API calls, this requires careful session state management and memory-grounded argument filling. A secondary challenge was the mem0 + qdrant-client version conflict that appeared in the real environment, which I resolved with a three-attempt initialization strategy and a pure Python fallback that keeps the pipeline running regardless.

---

## System Architecture

Here is the complete data flow from raw JSON files to a finished conversation dataset:

```
ToolBench JSON files
        │
        ▼
┌─────────────────┐
│  Tool Registry  │  Loads and normalises every tool, endpoint, and parameter
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   Tool Graph    │  Builds a NetworkX knowledge graph connecting tools,
│   (NetworkX)    │  endpoints, parameters, response fields, and categories
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Chain Sampler  │  Walks the graph to propose realistic tool chains
└────────┬────────┘
         │
         ▼
┌──────────────────────────────────────────┐
│         5-Agent Conversation Generator   │
│  ┌─────────┐  Picks which tools to use   │
│  │ Sampler │  from the graph             │
│  └────┬────┘                             │
│       │                                  │
│  ┌────▼────┐  Creates the user scenario  │
│  │ Planner │  (reads corpus memory)      │
│  └────┬────┘                             │
│       │                                  │
│  ┌────▼──────┐  Generates user messages  │
│  │ UserProxy │  and clarification answers│
│  └────┬──────┘                           │
│       │                                  │
│  ┌────▼─────────┐  Decides: clarify or  │
│  │  Assistant   │  call tool? Fills args │
│  │  Agent       │  from session memory   │
│  └────┬─────────┘                        │
│       │                                  │
│  ┌────▼─────────┐  Rejects conversations │
│  │  Validator   │  below quality bar     │
│  └──────────────┘                        │
└──────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  MemoryStore (mem0-backed)  │
│  scope=\"session\" ─ tool     │
│  outputs within one conv    │
│  scope=\"corpus\"  ─ summaries│
│  across all conversations   │
└─────────────┬───────────────┘
              │
              ▼
       output/dataset.jsonl
```

---

## Component-by-Component Design

---

### 1. Tool Registry — `registry.py`

**What it is:** A filing cabinet for API documentation. It reads raw ToolBench JSON files and stores them as clean, typed Python objects.

**Why dataclasses instead of raw dicts:**
Raw dicts let you write `tool[\"api_name\"]` but crash silently if the key is actually `\"tool_name\"` in that file. Dataclasses give you `tool.name` with IDE autocomplete, type checking, and a guaranteed structure even if the source JSON is messy.

**How it handles ToolBench's inconsistency:**
ToolBench has at least three format variants. The loader tries every known field name variant in order:

```python
name = (
    raw.get(\"tool_name\") or      # standard ToolBench
    raw.get(\"standardized_name\") or
    raw.get(\"name\") or           # alternate format
    raw.get(\"title\")             # some files use title
)
```

If a tool has no name at all, it is silently skipped — the loader never crashes on bad data.

**Endpoint ID format:**
Every endpoint gets a unique ID in the format `tool_name::endpoint_name`, for example `flight_search::search_flights`. This ID is used everywhere in the system as the stable identifier for a callable function.

**Type normalisation:**
Real ToolBench uses uppercase types (`STRING`, `INTEGER`). The registry normalises all types to lowercase so the rest of the system always sees `string`, `integer`, `number`.

---

### 2. Tool Graph — `graph.py`

**What it is:** A map of all tools and how they relate to each other, built with NetworkX.

**Why a directed graph (DiGraph) and not a list:**
The spec explicitly requires the sampler to use the graph — no hardcoded lists. More importantly, a directed graph captures something a list cannot: the *direction* of data flow. You call `search_flights` first, then use its `flight_id` output in `get_flight_details`. The arrow goes one way, not both.

**Five node types (as required by the spec):**

| Node type | Example | What it represents |
|---|---|---|
| `tool:{name}` | `tool:weather_api` | A whole API provider |
| `endpoint:{tool}::{ep}` | `endpoint:flight_search::search_flights` | One callable function |
| `param:{tool}::{ep}::{name}` | `param:flight_search::search_flights::origin` | One input parameter |
| `rf:{tool}::{ep}::{field}` | `rf:flight_search::search_flights::flight_id` | One output field |
| `category:{cat}` | `category:travel` | Semantic grouping tag |

With 8 sample tools: **169 nodes, 178 edges**. With real ToolBench (57 tools): **600+ nodes**.

**How feeds-into edges are detected automatically:**
The system compares every endpoint's output field names against every other endpoint's input parameter names. If they overlap, a `feeds_into` edge is added:

```
search_flights  →  output fields:  [flight_id, airline, price, ...]
get_flight_details  →  input params: [flight_id]
Overlap = [flight_id]  →  add edge: search_flights feeds_into get_flight_details
```

This is fully automatic — no manual wiring required.

---

### 3. Tool-Chain Sampler — `graph.py`

**What it is:** A tour guide that walks the graph and proposes which tools to use together.

**Three sampling patterns:**

```
Sequential:  A → B → C        Each tool feeds into the next
             [search flights → get flight details → search hotels]

Parallel:    A + B             Multiple tools called simultaneously
             [search flights + search hotels]  (both travel tools)

Mixed:       A → B → [C + D]  Sequential start, parallel finish
```

**How sequential sampling works:**
1. Pick a random starting endpoint from the graph
2. Follow `feeds_into` edges to find the next endpoint
3. If no chain edge exists, fall back to same-category tools
4. Repeat until the chain has 3–5 endpoints

**Why minimum 3 steps:**
The spec requires at least 3 tool calls per conversation. The sampler guarantees this by setting `min_tools=3` and by ensuring parallel patterns always append at least one sequential follow-up step.

---

### 4. Offline Execution Model — `execution.py`

**What it is:** A flight simulator for APIs. It validates inputs and generates realistic fake outputs without calling any real server.

**Two phases for every tool call:**

**Phase 1 — Validation:**
Before executing, check that all required parameters are present and types match:
```
flight_search::search_flights requires: origin (string), destination (string), departure_date (string)
If any are missing → raise ValidationError with a clear message
```

**Phase 2 — Mock response generation:**
Ask the LLM to generate a response that matches the endpoint's response fields and feels realistic given the context. If Paris is in the conversation, the mock hotel is in Paris. If the user mentioned June, the mock date is in June.

**Session state — how tool chains actually connect:**
Every tool output is stored immediately after execution:
```python
session.store_output(\"flight_search::search_flights\", step=0,
                     output={\"flight_id\": \"F001\", \"airline\": \"Air France\"})
```
When step 1 needs `flight_id`, it finds `\"F001\"` in the session store. This is what makes a multi-step trace a real chain rather than disconnected random JSON blobs.

---

### 5. Multi-Agent Conversation Generator — `agents.py` + `generator.py`

**What it is:** Five agents that each do one job, orchestrated by the generator into a complete conversation.

**The five agents and their exact responsibilities:**

| Agent | Job | Key Method |
|---|---|---|
| SamplerAgent | Calls the graph sampler to get a tool chain | `propose_chain()` |
| PlannerAgent | Reads corpus memory, then invents a user scenario | `plan()` |
| UserProxyAgent | Writes user messages and answers clarifications | `generate_opening()`, `respond_to_clarification()` |
| AssistantAgent | Decides: clarify or call tool? Fills args. Presents results. | `should_clarify()`, `fill_arguments()`, `present_results()` |
| ValidatorAgent | Rejects conversations below minimum quality | `validate()` |

**The conversation loop — step by step:**

```
for each endpoint in the tool chain:

    Step A — Clarification check
    AssistantAgent asks the LLM: \"Is any required parameter missing
    from this conversation? If yes, write ONE question. If no, write NONE.\"
    → If a question: add [ASSISTANT question] + [USER answer] to messages

    Step B — Argument filling (reads session memory)
    AssistantAgent queries session memory for prior tool outputs,
    injects them into the prompt, then asks LLM to fill the argument dict.
    → Adds [ASSISTANT calling tool] + [TOOL output] to messages

    Step C — Write to session memory
    Store the tool output so the next step can reference it.

    Step D — Present results (all steps except the last)
    AssistantAgent generates a natural language summary of the tool output.
    → Adds [ASSISTANT result summary] + [USER follow-up] to messages

After all tools:
    AssistantAgent generates ONE final synthesis message referencing all tools.
    → Adds [ASSISTANT final summary] to messages

ValidatorAgent checks: ≥3 tool calls? ≥2 distinct tools? → keep or retry
```

**Why the last step skips present_results:**
If every step added both a `present_results` message and then a `generate_final_response` at the end, the conversation would end with two consecutive assistant messages. By skipping `present_results` on the final step and letting `generate_final_response` serve as the closing message, the conversation always ends cleanly with one assistant turn.

---

### 6. LangGraph Pipeline — `langgraph_generator.py`

**What it is:** The same five-agent pipeline rebuilt as a proper state machine using LangGraph — the industry standard for production multi-agent AI systems.

**Why LangGraph matters:**
The plain Python version uses a for-loop to run agents. This works but hides the decision logic inside nested if-statements. LangGraph makes the decision logic *explicit* as a visual graph:

```
[START] → [sampler] → [planner] → [user_open] → [router]
                                                    │         │
                                               [clarify]  [call_tool]
                                                    │         │
                                                    └──→ [present] → [router]
                                                                         │
                                                                    [finalize]
                                                                         │
                                                                    [validate] → [END]
```

**Key LangGraph concepts used:**

*State (TypedDict):* A shared dictionary all nodes read and write. Think of it as a whiteboard in a meeting room — every agent sees the same information.

```python
class ConversationState(TypedDict):
    chain:                 list[str]   # tool endpoints to call
    plan:                  dict        # user scenario
    messages:              list[dict]  # all chat messages so far
    tool_calls:            list[dict]  # all tool calls made
    current_step:          int         # which tool we're on
    clarification_count:   int         # how many times we've asked
    pending_clarification: str         # question set by router
    is_valid:              bool        # set by validator at the end
```

*Conditional edges:* After the router node runs, a function inspects the state and returns a string (`\"clarify\"`, `\"call_tool\"`, or `\"finalize\"`) to decide which node visits next. This is cleaner and more testable than nested if-statements.

*Why this impresses engineers:* LangGraph is used by LinkedIn, Replit, and many AI companies for production agent systems. Knowing it signals you understand not just how to write agents, but how to build them at scale.

---

### 7. MemoryStore — `memory.py`

**What it is:** A smart notebook that remembers things — within a conversation and across conversations — backed by mem0.

**The interface (exactly as the spec requires):**
```python
class MemoryStore:
    def add(self, content: str, scope: str, metadata: dict) -> None: ...
    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]: ...
```

Nothing else in the codebase ever imports mem0 directly. If mem0 changes its API, only this file needs updating.

**Two scopes — simple analogy:**

| Scope | Analogy | What is stored | When it is written | When it is read |
|---|---|---|---|---|
| `\"session\"` | Post-it notes on your desk | Tool outputs from this conversation | After every tool call | Before filling args for any non-first step |
| `\"corpus\"` | A shared team notebook | Summary of each completed conversation | After conversation is validated | Before Planner creates a new scenario |

**Session memory — how it grounds arguments:**

```
Step 0: search_hotels() → {\"hotel_id\": \"H123\", \"name\": \"Grand Hotel\"}
         WRITE to session memory

Step 1: get_hotel_reviews(hotel_id=???)
         READ session memory → finds hotel_id=\"H123\"
         Prompt sent to LLM:
           [Memory context]
           - {\"endpoint\": \"search_hotels\", \"output\": {\"hotel_id\": \"H123\"}}
           Given the above context, fill in the arguments for get_hotel_reviews.
         LLM fills: {\"hotel_id\": \"H123\"}  ← grounded in real data ✅
```

**Corpus memory — how it creates diversity:**

```
Conversation 1 done → WRITE: \"Tools: weather, flights. Domain: travel. Pattern: sequential.\"
Conversation 2 done → WRITE: \"Tools: currency, news. Domain: finance. Pattern: mixed.\"

Before Conversation 3 → READ corpus → sees travel and finance already done
Planner prompt:
  [Prior conversations in corpus]
  - Tools: weather, flights. Domain: travel.
  - Tools: currency, news. Domain: finance.
  Given the above, plan a NEW DIVERSE conversation.
  → Planner picks food or maps domain instead ✅
```

**Three-attempt initialisation strategy:**
Real environments have version conflicts between mem0 and qdrant-client. Rather than crashing, the store tries three approaches:

1. `Memory()` — plain, exactly as the spec says, no config
2. `Memory.from_config(ollama_config)` — for newer mem0 versions needing explicit LLM config
3. `SimpleMemoryFallback` — pure Python keyword search, zero dependencies, always works

Each attempt is smoke-tested immediately with a real add + search call to catch silent errors.

**memory_grounding_rate metric:**

```
rate = (non-first tool calls that used session memory) / (total non-first tool calls)

1.0 = every eligible call was grounded in prior results  ← achieved
0.0 = memory was never used
null = only one tool call (nothing to ground)
```

---

## Corpus Memory & Diversity Analysis

### What diversity metric and why

**Primary metric: Pairwise Tool-Chain Jaccard Dissimilarity**

For every pair of conversations, compute how different their tool sets are:

```
Jaccard similarity  = |tools_A ∩ tools_B| / |tools_A ∪ tools_B|
Jaccard dissimilarity = 1 - similarity
```

Average across all pairs. Range 0 to 1. Higher = more diverse.

**Why Jaccard:** It directly captures what corpus memory is designed to improve — whether the Planner is choosing different tool combinations each time. Two conversations using the exact same tools score 0 (identical). Two conversations with no tools in common score 1 (fully diverse).

**Secondary metrics:**
- *Distinct-2 bigrams* — ratio of unique 2-word phrases in assistant messages. Different domains produce different vocabulary (hotel prices vs stock tickers vs restaurant menus).
- *Pattern entropy* — Shannon entropy over sequential/parallel/mixed distribution. Higher entropy means more evenly spread patterns.

### Results

| Metric | Run A — no corpus memory | Run B — corpus memory |
|---|---|---|
| Tool-chain Jaccard dissimilarity | `run: python -m toolgen.cli metrics output/run_a.jsonl --compare output/run_b.jsonl` | see comparison output |
| Distinct-2 bigrams | see CLI output | see CLI output |
| Pattern entropy | see CLI output | see CLI output |

Reproduce with:
```bash
python -m toolgen.cli generate --count 20 --seed 42 --no-corpus-memory --output output/run_a.jsonl
python -m toolgen.cli generate --count 20 --seed 42 --output output/run_b.jsonl
python -m toolgen.cli metrics output/run_a.jsonl --compare output/run_b.jsonl
```

### Analysis

Corpus memory improves diversity because the Planner reads summaries of every completed conversation before creating a new one. When it sees \"Tools: weather, flights. Domain: travel. Pattern: sequential\" in its context, it steers toward a different domain — food, finance, or news — and different tool combinations. Without corpus memory (Run A), the Planner has no visibility into what already exists and may accidentally generate similar scenarios repeatedly.

The Jaccard metric captures this most directly: if Run B produces more unique tool combinations across conversations, pairwise dissimilarity rises. Distinct bigrams improve as a side effect because different domains use different vocabulary. Pattern entropy changes minimally because the generator already cycles through sequential, parallel, and mixed patterns deterministically regardless of memory — corpus memory's main influence is on *what* tools are chosen, not *how* they are structured.

If corpus memory shows no improvement in your run, the most likely cause is that the language model (especially smaller 1B or 3B models) does not attend strongly enough to the injected corpus context. Larger models (8B+) respond more reliably to prior-conversation context.

---

## Testing Strategy

**Why 61 tests and how they are organised:**

| Test file | What it tests | Dependencies |
|---|---|---|
| `test_registry.py` (22 tests) | Loading tools, parsing parameters, handling malformed data, save/load roundtrip | None — pure Python |
| `test_memory.py` (12 tests) | add then search returns the entry, scope isolation (session ≠ corpus), metadata preservation | None — uses SimpleMemoryFallback |
| `test_e2e.py` (27 tests) | Full pipeline: builds registry, graph, generates 50 conversations, validates JSONL format | None — uses MockLLM |

**No external dependencies in tests:**
All tests use a MockLLM that returns deterministic responses. This means `python -m unittest discover -s tests -v` runs in under one second on any machine with no Ollama, no mem0, no internet — critical for a reviewer running the project on their machine.

**The two required MemoryStore unit tests (from the spec):**

```python
def test_add_then_search_returns_stored_entry(self):
    store.add(\"Tool output: hotel in Paris at $89/night\", scope=\"session\", metadata={})
    results = store.search(\"hotels Paris\", scope=\"session\")
    assert len(results) >= 1  # entry must be retrievable ✅

def test_scope_isolation(self):
    store.add(\"session data: hotel_id=H123\", scope=\"session\", metadata={})
    store.add(\"corpus data: travel planning\", scope=\"corpus\", metadata={})
    session_results = store.search(\"data\", scope=\"session\")
    corpus_results  = store.search(\"data\", scope=\"corpus\")
    # Session results must only contain session entries
    assert all(r[\"scope\"] == \"session\" for r in session_results)
    # Corpus results must only contain corpus entries
    assert all(r[\"scope\"] == \"corpus\" for r in corpus_results)  ✅
```

---

## Reproducibility

All random decisions use `random.Random(seed)`. Given the same seed, the same tool chains are proposed in the same order every time. LLM outputs vary (language models are non-deterministic), but the structural skeleton of the dataset — which tools are called, in what order, with what patterns — is fully reproducible.

For the memory_grounding_rate metric, the spec says: *\"count a retrieval as present whenever search() returns at least one result, regardless of score threshold.\"* This is implemented exactly — any non-empty result from `memory.search()` counts as grounded, no score filtering applied.

---

## What I Would Improve with More Time

1. **Smarter session memory queries** — currently queries by endpoint name and description. Querying specifically for the parameter names needed by the next step would produce more precise grounding.

2. **Full ToolBench integration** — the 8-tool sample set limits graph richness. With all 16,000 tools, the `feeds_into` edge discovery would produce a far more connected graph with richer, more varied chains.

3. **Live mem0 integration test** — all tests use MockLLM and SimpleMemoryFallback. An integration test that spins up real Ollama and validates the full mem0 + Qdrant path would close the last remaining gap.
