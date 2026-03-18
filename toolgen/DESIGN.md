# DESIGN.md
## My Approach and Goal (5 sentences)
My goal was to build an **offline** system that generates realistic conversations where an assistant uses **multiple tools across multiple steps** (like “search flights → choose hotel → book dinner”). I broke the project into small layers: a **Tool Registry** (filing cabinet), a **Tool Graph** (map), a **Sampler** (tour guide), an **Offline Executor** (flight simulator), and **Memory** (post‑it notes + shared notebook). The hardest technical challenge was making tool calls **actually connect** so later steps can reuse earlier IDs (a chain, not random blobs). The real-world challenge was a **mem0 + qdrant** version conflict in some environments. I solved that with a **three-attempt initialization** (try mem0 → try explicit config → fall back safely) so generation never crashes.

## System Architecture (ASCII data-flow)
```text
ToolBench-style JSON files
        |
        v
  +------------------+
  | Tool Registry    |   (filing cabinet: Tool / Endpoint / Param / ResponseField)
  | toolgen/registry | 
  +------------------+
        |
        v
  +------------------+
  | Tool Graph       |   (map: directed edges show data flow)
  | toolgen/graph    |
  +------------------+
        |
        v
  +------------------+
  | Chain Sampler    |   (tour guide: proposes tool chains)
  | toolgen/graph    |
  +------------------+
        |
        v
  +------------------------------------------------------+
  | 5-Agent Conversation Generator                        |
  |  - SamplerAgent   -> picks a chain from the graph     |
  |  - PlannerAgent   -> invents a scenario (reads corpus)|
  |  - UserProxyAgent -> writes user messages             |
  |  - AssistantAgent -> clarifies, fills args, calls tool|
  |  - ValidatorAgent -> enforces quality thresholds      |
  |  (agents.py + generator.py)                           |
  +------------------------------------------------------+
        |
        v
  +-----------------------------+
  | MemoryStore (mem0-backed)   |
  |  scope='session' = post-its |
  |  scope='corpus'  = notebook |
  |  toolgen/memory             |
  +-----------------------------+
        |
        v
output/dataset.jsonl   (one JSON record per conversation)
```

## Component-by-Component Design (WHAT + WHY + technical details)

### A) Tool Registry (`toolgen/registry.py`)
**What it is:** A **filing cabinet** that turns messy ToolBench JSON files into clean Python objects.

**Why this decision (WHAT + WHY):**
- **WHAT:** I used **dataclasses** for tools, endpoints, and parameters.
- **WHY:** Dataclasses give a stable shape (no “missing key crashes”), IDE autocomplete, and easier validation than raw dicts.

**Technical details**

#### Why dataclasses instead of raw dicts
- A dict can silently vary per file: `tool["tool_name"]` vs `tool["name"]`.
- A dataclass gives you a guaranteed field like `tool.name` after normalization.

#### ToolBench format variants (3+)
ToolBench files don’t all use the same keys. The loader tries multiple key names in order (example idea):

```python
name = raw.get("tool_name") or raw.get("standardized_name") or raw.get("name") or raw.get("title")
```

#### Endpoint ID format (stable identifier)
Plain English: every callable gets a unique “label”, like a folder name.

Format:
```text
tool_name::endpoint_name
```
Example:
```text
flight_search::search_flights
```

#### Type normalization
Plain English: some files shout `STRING` in uppercase; the rest of the system expects `string`.

```text
STRING -> string
INTEGER -> integer
```

---

### B) Tool Graph (`toolgen/graph.py`)
**What it is:** A **map** showing how tools connect (what outputs can feed into what inputs).

**Why this decision (WHAT + WHY):**
- **WHAT:** I used a **directed graph** (arrows have direction).
- **WHY:** Data flows one way: `search_flights` produces `flight_id`, then `get_flight_details` consumes `flight_id`. That’s not symmetric, so an undirected graph would be wrong.

**Technical details**

#### Node types (all 5 required)
| Node type | Example node id | What it means |
|---|---|---|
| **tool** | `tool:flight_search` | One tool / API provider |
| **endpoint** | `endpoint:flight_search::search_flights` | One callable function |
| **param** | `param:flight_search::search_flights::origin` | One input parameter |
| **response_field** | `rf:flight_search::search_flights::flight_id` | One output field |
| **category** | `category:travel` | A semantic tag / grouping |

#### Graph size (sample vs real)
- **8 sample tools:** **169 nodes**, **178 edges**
- **Real ToolBench subset:** **600+ nodes** (depends on subset size)

#### How `feeds_into` edges are found (automatic)
Plain English: if one endpoint outputs a thing that another endpoint needs as input, draw an arrow.

ASCII example:
```text
search_flights outputs:     [flight_id, price, airline]
get_flight_details needs:   [flight_id]

overlap = [flight_id]  =>  search_flights  -->  get_flight_details
```

---

### C) Chain Sampler (`toolgen/graph.py`)
**What it is:** A **tour guide** that walks the map and picks a path of tool calls.

**Why this decision (WHAT + WHY):**
- **WHAT:** I built 3 tool‑calling patterns: sequential, parallel, mixed.
- **WHY:** Real assistants don’t always call tools in a single straight line. Sometimes they do two lookups “at once” (parallel), or a mix.

**Technical details**

#### The 3 patterns (with ASCII)
Sequential (A → B → C):
```text
search_flights -> get_flight_details -> search_hotels
```

Parallel (A + B):
```text
[search_flights + search_hotels]  ->  convert_currency
```

Mixed (A → B → [C + D]):
```text
search_flights -> get_flight_details -> [search_hotels + get_current_weather]
```

#### How sequential sampling works
Plain English: pick a start, then follow arrows. If you get stuck, pick something in the same category.

Minimum steps:
- **WHY minimum 3:** the dataset must contain **multi-step traces**, so the sampler ensures ≥ 3 tool calls.

---

### D) Offline Execution Model (`toolgen/execution.py`)
**What it is:** A **flight simulator** for APIs: it feels like a real tool call, but it never touches the internet.

**Why this decision (WHAT + WHY):**
- **WHAT:** Every tool call has two phases: validate → generate mock output.
- **WHY:** Validation prevents nonsense arguments, and mock outputs keep traces realistic and chainable even without real APIs.

**Technical details**

#### Phase 1: validation
Plain English: check the “required fields checklist” before calling the tool.

```text
Missing required parameter? -> raise ValidationError
Wrong type? -> error message
Enum constraint violated? -> error message
```

#### Phase 2: mock response generation
Plain English: generate JSON that matches the endpoint’s response fields and the conversation context.

#### SessionState makes the chain connect
Plain English: SessionState is the assistant’s “scratchpad” for IDs like `hotel_id` and `flight_id`.

Example chain:
```text
Step 0 returns: {"hotel_id": "H_001"}
Step 1 needs:   {"hotel_id": ???}
SessionState provides: hotel_id = "H_001"
```

---

### E) 5-Agent System (`toolgen/agents.py` + `toolgen/generator.py`)
**What it is:** Five small “workers” that each do one job, like a team on a project.

**Why this decision (WHAT + WHY):**
- **WHAT:** I used a simple multi-agent design instead of a heavy framework.
- **WHY:** Fewer dependencies, easier debugging, and each agent stays small and focused.

**Agents table (job + key method)**
| Agent | Job (plain English) | Key method |
|---|---|---|
| **SamplerAgent** | asks the tool graph for a chain | `propose_chain()` |
| **PlannerAgent** | invents a specific scenario (reads corpus memory) | `plan()` |
| **UserProxyAgent** | writes user messages + answers questions | `generate_opening()`, `respond_to_clarification()` |
| **AssistantAgent** | asks clarifications, fills args, presents results | `should_clarify()`, `fill_arguments()`, `present_results()` |
| **ValidatorAgent** | rejects low-quality conversations | `validate()` |

**Conversation loop (numbered)**
1) Sampler proposes a chain from the Tool Graph  
2) Planner creates a scenario (reads corpus memory first)  
3) UserProxy writes the opening message (keeps it slightly vague)  
4) For each endpoint:
   - 4.1) Assistant asks: “Do I need a clarification?”  
   - 4.2) If yes, user answers  
   - 4.3) Assistant fills arguments (uses session memory if not the first tool)  
   - 4.4) Executor validates + generates mock output  
   - 4.5) Write tool output to session memory  
5) Assistant writes **one** final summary using all tool calls  
6) Validator checks: **≥ 3 tool calls** and **≥ 2 distinct tools**

#### Why the last step skips `present_results`
Plain English: without this, the chat can end with two assistant messages in a row, which looks wrong.

- **WHAT:** Skip `present_results` on the last tool call.
- **WHY:** The final summary already closes the conversation, so we keep exactly one final assistant message.

---

### F) LangGraph Pipeline (`toolgen/langgraph_generator.py`)
**What it is:** The same agent logic, but built as a **flowchart graph** instead of nested if‑statements.

**Why this decision (WHAT + WHY):**
- **WHAT:** I implemented an optional LangGraph mode (`--use-langgraph`).
- **WHY:** LangGraph makes agent flows easier to visualize, test, and extend in production systems.

**Node graph (ASCII)**
```text
[START]
   |
   v
[sampler] -> [planner] -> [user_open] -> [router]
                                      /      \
                                 [clarify]  [call_tool]
                                     |          |
                                     v          v
                                  [router] <- [present]
                                     |
                                     v
                                  [finalize] -> [validate] -> [END]
```

**State = “whiteboard” analogy**
Plain English: State is like a shared whiteboard in a meeting room. Every node reads and writes to it.

Example fields (simplified):
```text
chain, plan, messages, tool_calls, current_step, clarification_count, is_valid
```

Conditional edges (plain English): the router looks at the state and decides which node comes next (“clarify” vs “call_tool” vs “finalize”).

---

### G) MemoryStore (`toolgen/memory.py`)
**What it is:** A “smart notebook” with two drawers: **session** and **corpus**.

**Why this decision (WHAT + WHY):**
- **WHAT:** I wrapped mem0 behind a tiny 2‑method interface.
- **WHY:** The rest of the code never depends on mem0 directly, so mem0 can change without breaking the project.

#### The exact 2-method interface (required)
```python
class MemoryStore:
    def add(self, content: str, scope: str, metadata: dict) -> None: ...
    def search(self, query: str, scope: str, top_k: int = 5) -> list[dict]: ...
```

#### Session vs Corpus (table + analogy)
| Scope | Analogy | What is stored | When written | When read |
|---|---|---|---|---|
| **session** | post‑it notes on your desk | tool outputs for *this* conversation | after every tool call | before filling args for step > 0 |
| **corpus** | shared team notebook | 1‑line summary of each finished conversation | after validation | before planning a new scenario |

#### Session memory grounding (prompt injection)
Plain English: before filling arguments, we paste memory snippets into the prompt so the model can reuse IDs.

```text
[Memory context]
- {"endpoint":"search_hotels","output":{"hotel_id":"H_001"}}

Given the above context and the tool schema, fill in arguments for get_hotel_reviews.
```

#### Corpus memory for diversity
Plain English: the planner reads “what we already did” and tries to do something different next.

```text
[Prior conversations in corpus]
- Tools: flight_search, hotel_booking. Domain: travel. Pattern: sequential.
- Tools: currency_exchange, news_api. Domain: finance. Pattern: mixed.
```

#### Three-attempt initialization strategy (mem0/qdrant issues)
**WHAT:** Attempt mem0 initialization 3 ways.  
**WHY:** Version conflicts should not break dataset generation.

1) `Memory()` (spec-default, embedded Qdrant)  
2) `Memory.from_config(...)` (explicit Ollama config for newer mem0 versions)  
3) `SimpleMemoryFallback` (pure Python, always works)

#### `memory_grounding_rate` formula
Plain English: “How often did we actually use memory when we could have?”

\[
memory\_grounding\_rate =
\frac{\#(non\text{-}first\ steps\ with\ at\ least\ one\ memory\ retrieval)}
{\#(non\text{-}first\ steps)}
\]

What **1.0** means: every eligible tool call had at least one memory hit.  
Your dataset reports: **avg_memory_grounding_rate = 1.0**.

---

## Corpus Memory & Diversity Analysis (required)

### Primary metric: Pairwise Tool-Chain Jaccard Dissimilarity
**Plain English:** This measures “How different are the tool sets between conversations?”

For two conversations A and B:
- Intersection = tools both used
- Union = all tools used by either

\[
Jaccard(A,B)=\frac{|A \cap B|}{|A \cup B|}, \quad
Dissimilarity(A,B)=1-Jaccard(A,B)
\]

**Why Jaccard:** corpus memory is meant to change **which tools** get used across conversations, so a tool-set metric is the most direct match.

### Secondary metrics
- **Distinct bigrams** (Distinct‑2): different domains produce different wording.
- **Pattern entropy:** how evenly we spread sequential/parallel/mixed patterns.

### Results + command to reproduce
```bash
python3 -m toolgen.cli generate --count 20 --seed 42 --no-corpus-memory --output output/run_a.jsonl
python3 -m toolgen.cli generate --count 20 --seed 42 --output output/run_b.jsonl
python3 -m toolgen.cli metrics output/run_a.jsonl --compare output/run_b.jsonl
```

Dataset numbers (from `output/dataset.jsonl`):
| Metric | Value |
|---|---:|
| **pct_with_3plus_tool_calls** | **100%** |
| **pct_with_clarification** | **54%** |
| **avg_memory_grounding_rate** | **1.0** |
| **tool_chain_jaccard_dissimilarity** | **0.598** |
| **pattern_entropy** | **1.419** |

### 4-sentence analysis
Corpus memory improves diversity because the planner can “see” what it already produced and avoid repeating the same scenarios. When the planner reads summaries like “Tools: flight_search, hotel_booking. Domain: travel”, it naturally shifts to new domains (food, finance, news, productivity) and new tool combinations. That increases tool-set variety, which is exactly what Jaccard dissimilarity measures. Pattern entropy changes less because the generator already cycles patterns, so corpus memory mostly affects **content/tool choice**, not the structural pattern schedule.

---

## Testing Strategy (fast, offline)
**Plain English:** Tests should run on any reviewer laptop without extra setup.

### Test suite table
| Test file | What it tests | External dependencies |
|---|---|---|
| `tests/test_registry.py` | loader + schema normalization + messy ToolBench handling | none |
| `tests/test_memory.py` | MemoryStore add/search + scope isolation | none (uses fallback) |
| `tests/test_e2e.py` | end-to-end generation + JSONL format + metrics sanity | none (uses MockLLM) |

### The two MemoryStore unit tests the spec requires (code)
```python
def test_add_then_search_returns_stored_entry(self):
    store.add("hotel_id=H123 in Paris", scope="session", metadata={})
    results = store.search("Paris hotel", scope="session")
    assert len(results) >= 1

def test_scope_isolation(self):
    store.add("session: hotel_id=H123", scope="session", metadata={})
    store.add("corpus: travel planning", scope="corpus", metadata={})
    assert all(r["scope"] == "session" for r in store.search("hotel", scope="session"))
    assert all(r["scope"] == "corpus" for r in store.search("travel", scope="corpus"))
```

**Key point:** all 61 tests use MockLLM/fallback memory and run in ~0.1s.

---

## Reproducibility (seed vs LLM)
**Plain English:** A seed controls the *structure*; the model controls the *words*.

- **Seed controls:** which tools are sampled, the pattern type (sequential/parallel/mixed), and the overall skeleton.
- **LLM controls:** the exact phrasing of messages and the mock values.

So two runs with the same seed will have the same tool-chain structure, even if the natural language varies slightly.

---

## What I Would Improve (honest, 3 items)
1) **Smarter session memory queries:** search by required parameter names (like `hotel_id`) instead of only endpoint description, so grounding is more precise.  
2) **Full ToolBench scale:** run on a much larger slice (thousands of tools) to create richer graphs and more varied chains.  
3) **Live mem0 integration test:** add one optional test that runs with real mem0 + embedded qdrant to verify the full path in environments where it is supported.
